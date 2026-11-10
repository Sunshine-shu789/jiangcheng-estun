#!/usr/bin/env python3
"""Minimal eye-in-hand visual search and approach prototype.

Useful modes:

* ``--mock``       no camera, no robot: verifies search/servo math end-to-end
* ``--camera-only`` opens RealSense and runs YOLO, but never imports ROS
* ``--robot-status`` reads /RobotInfo only; no camera and no motion
* default          reads /RobotInfo and checks only the current camera view
* ``--scan-only``  with ``--execute``: search for the target, then stop
* ``--execute``    allows small joint-space search and Cartesian approach moves

The RGB-D frame is rotated 180 degrees by default because the current D405 is
mounted upside down. The effective (rotated) camera axes align with the tool
axes in the default ``camera_to_tool`` assumption.

Safety defaults:
* motion is always dry-run unless ``--execute`` is explicit
* each Cartesian step is capped by ``servo.max_step_m``
* the target must remain inside ``workspace``
* the controller must report ``AutoReady`` before every real move
* total travel from the initial pose is capped
* no gripper command is sent
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Controller kinematics
# ---------------------------------------------------------------------------

# The live controller uses this modified-DH model.  It was verified against
# /RobotInfo: FK(q) reproduces end_positions to well below one millimetre.
# Do not replace this with the URDF model for motion planning; the URDF on this
# machine does not match the controller exactly.
JOINT_NAMES = [f"Joint{i}" for i in range(1, 7)]
CONTROLLER_DH = [
    [0.0, 0.0, 176.0, 0.0],
    [0.0, -90.0, 158.964, -90.0],
    [249.388, 0.0, -158.0, 0.0],
    [220.792, 180.0, -106.0, -90.0],
    [0.0, 90.0, -105.546, 0.0],
    [0.0, 90.0, 123.5, 0.0],
]
JOINT_LOW = np.radians([-200.0, -200.0, -163.0, -200.0, -200.0, -200.0])
JOINT_HIGH = -JOINT_LOW


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionConfig:
    width: int
    height: int
    fps: int
    serial: str
    warmup_frames: int
    weights: str
    confidence: float
    imgsz: int
    device: str
    target_class: str
    depth_window: int
    anchor: str
    rotate_180: bool
    output_dir: Path
    show: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionConfig":
        return cls(
            width=int(data.get("width", 640)),
            height=int(data.get("height", 480)),
            fps=int(data.get("fps", 15)),
            serial=str(data.get("serial", "")),
            warmup_frames=int(data.get("warmup_frames", 20)),
            weights=str(data.get("weights", "")),
            confidence=float(data.get("confidence", 0.35)),
            imgsz=int(data.get("imgsz", 640)),
            device=str(data.get("device", "cpu")),
            target_class=str(data.get("target_class", "")),
            depth_window=max(1, int(data.get("depth_window", 5))),
            anchor=str(data.get("anchor", "center")),
            rotate_180=bool(data.get("rotate_180", False)),
            output_dir=Path(data.get("output_dir", "out")).expanduser(),
            show=bool(data.get("show", False)),
        )


@dataclass(frozen=True)
class ServoConfig:
    desired_standoff_m: float
    min_depth_m: float
    center_tolerance_px: float
    depth_tolerance_m: float
    max_steps: int
    max_step_m: float
    max_total_travel_m: float
    speed_mps: float
    angular_speed_rad_s: float
    lateral_gain: float
    forward_gain: float
    move_timeout_s: float
    move_reject_timeout_s: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    poll_period_s: float
    settle_s_after_move: float
    convergence_frames: int
    camera_to_tool: np.ndarray

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServoConfig":
        matrix = np.asarray(
            data.get("camera_to_tool", np.eye(3).tolist()), dtype=float
        )
        if matrix.shape != (3, 3):
            raise ValueError("servo.camera_to_tool must be a 3x3 matrix")
        cfg = cls(
            desired_standoff_m=float(data.get("desired_standoff_m", 0.18)),
            min_depth_m=float(data.get("min_depth_m", 0.12)),
            center_tolerance_px=float(data.get("center_tolerance_px", 35.0)),
            depth_tolerance_m=float(data.get("depth_tolerance_m", 0.015)),
            max_steps=int(data.get("max_steps", 15)),
            max_step_m=float(data.get("max_step_m", 0.02)),
            max_total_travel_m=float(data.get("max_total_travel_m", 0.35)),
            speed_mps=float(data.get("speed_mps", 0.02)),
            angular_speed_rad_s=float(data.get("angular_speed_rad_s", 0.10)),
            lateral_gain=float(data.get("lateral_gain", 0.8)),
            forward_gain=float(data.get("forward_gain", 0.8)),
            move_timeout_s=float(data.get("move_timeout_s", 60.0)),
            move_reject_timeout_s=float(data.get("move_reject_timeout_s", 2.0)),
            position_tolerance_m=float(data.get("position_tolerance_m", 0.003)),
            orientation_tolerance_rad=float(
                data.get("orientation_tolerance_rad", 0.02)
            ),
            poll_period_s=float(data.get("poll_period_s", 0.05)),
            settle_s_after_move=float(data.get("settle_s_after_move", 0.25)),
            convergence_frames=max(1, int(data.get("convergence_frames", 2))),
            camera_to_tool=matrix,
        )
        if cfg.max_step_m <= 0.0 or cfg.max_step_m > 0.05:
            raise ValueError("servo.max_step_m must be in (0, 0.05]")
        if cfg.max_total_travel_m <= 0.0:
            raise ValueError("servo.max_total_travel_m must be positive")
        if not 0.0 < cfg.speed_mps <= 0.05:
            raise ValueError("servo.speed_mps must be in (0, 0.05]")
        if not 0.0 < cfg.angular_speed_rad_s <= 0.5:
            raise ValueError("servo.angular_speed_rad_s must be in (0, 0.5]")
        if cfg.desired_standoff_m <= cfg.min_depth_m:
            raise ValueError("desired_standoff_m must exceed min_depth_m")
        if cfg.max_steps < 1:
            raise ValueError("servo.max_steps must be >= 1")
        if cfg.move_timeout_s <= 0.0 or cfg.poll_period_s <= 0.0:
            raise ValueError("servo move timeout and poll period must be positive")
        if not 0.1 <= cfg.move_reject_timeout_s <= 10.0:
            raise ValueError("servo.move_reject_timeout_s must be in [0.1, 10]")
        if not 0.0 < cfg.orientation_tolerance_rad <= 0.2:
            raise ValueError(
                "servo.orientation_tolerance_rad must be in (0, 0.2]"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("servo.camera_to_tool contains non-finite values")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4):
            raise ValueError("servo.camera_to_tool must be orthonormal")
        if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-4):
            raise ValueError("servo.camera_to_tool must be a proper rotation (det=+1)")
        return cfg


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool
    yaw_offsets_deg: tuple[float, ...]
    pitch_down_offsets_deg: tuple[float, ...]
    settle_s: float
    observe_frames: int
    joint_max_step_deg: float
    joint_move_timeout_s: float
    joint_reject_timeout_s: float
    return_to_initial_on_miss: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            yaw_offsets_deg=tuple(
                float(v) for v in data.get("yaw_offsets_deg", [0.0, -12.0, 12.0])
            ),
            pitch_down_offsets_deg=tuple(
                float(v) for v in data.get("pitch_down_offsets_deg", [0.0])
            ),
            settle_s=float(data.get("settle_s", 0.8)),
            observe_frames=max(1, int(data.get("observe_frames", 3))),
            joint_max_step_deg=float(data.get("joint_max_step_deg", 1.0)),
            joint_move_timeout_s=float(data.get("joint_move_timeout_s", 8.0)),
            joint_reject_timeout_s=float(
                data.get("joint_reject_timeout_s", 2.0)
            ),
            return_to_initial_on_miss=bool(
                data.get("return_to_initial_on_miss", True)
            ),
        )


@dataclass(frozen=True)
class WorkspaceConfig:
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    margin: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceConfig":
        low = np.asarray(data.get("min_xyz", [-0.4, -0.4, 0.05]), dtype=float)
        high = np.asarray(data.get("max_xyz", [0.4, 0.4, 0.8]), dtype=float)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError("workspace min/max must each contain 3 numbers")
        if np.any(high <= low):
            raise ValueError("workspace max must exceed min")
        margin = float(data.get("margin", 0.01))
        if margin < 0.0:
            raise ValueError("workspace.margin must be non-negative")
        return cls(low, high, margin)


@dataclass(frozen=True)
class ProbeConfig:
    vision: VisionConfig
    servo: ServoConfig
    search: SearchConfig
    workspace: WorkspaceConfig
    robot_info_timeout_s: float
    robotmove_topic: str
    robotinfo_topic: str

    @classmethod
    def load(cls, path: Path) -> "ProbeConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            vision=VisionConfig.from_dict(raw.get("vision", {})),
            servo=ServoConfig.from_dict(raw.get("servo", {})),
            search=SearchConfig.from_dict(raw.get("search", {})),
            workspace=WorkspaceConfig.from_dict(raw.get("workspace", {})),
            robot_info_timeout_s=float(raw.get("robot_info_timeout_s", 2.0)),
            robotmove_topic=str(raw.get("robotmove_topic", "RobotMove")),
            robotinfo_topic=str(raw.get("robotinfo_topic", "RobotInfo")),
        )


# ---------------------------------------------------------------------------
# Geometry and data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]

    def as_list(self) -> list[float]:
        return [*self.xyz, *self.rpy]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    u: float
    v: float
    depth_m: float


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def rotate_180_intrinsics(intrinsics: Intrinsics) -> Intrinsics:
    """Map a camera principal point after rotating an image by 180 degrees."""
    return Intrinsics(
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.width - 1.0 - intrinsics.cx,
        intrinsics.height - 1.0 - intrinsics.cy,
    )


def rotate_180_rgbd(
    color: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
) -> tuple[np.ndarray, np.ndarray, Intrinsics]:
    """Rotate aligned RGB-D frames and principal point by 180 degrees."""
    if color.shape[:2] != depth_m.shape[:2]:
        raise RuntimeError(
            f"RGB/depth shape mismatch: {color.shape[:2]} vs {depth_m.shape[:2]}"
        )
    color = np.ascontiguousarray(color[::-1, ::-1])
    depth_m = np.ascontiguousarray(depth_m[::-1, ::-1])
    return color, depth_m, rotate_180_intrinsics(intrinsics)


@dataclass(frozen=True)
class Observation:
    color: np.ndarray
    detections: tuple[Detection, ...]
    intrinsics: Intrinsics


@dataclass(frozen=True)
class RobotStatus:
    ready: bool
    state: str
    fault_flag: int
    pose: Pose | None
    message: str
    joints: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ServoCommand:
    target: Pose
    delta_base: np.ndarray
    pixel_error: tuple[float, float]
    depth_m: float
    converged: bool
    reason: str


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll), the usual ROS ZYX convention."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a ZYX RPY rotation matrix back to roll, pitch, yaw."""
    horizontal = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    pitch = math.atan2(-float(matrix[2, 0]), horizontal)
    if horizontal > 1e-6:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(matrix[0, 1]), float(matrix[1, 1]))
    return roll, pitch, yaw


def rotation_angle_error(current: np.ndarray, target: np.ndarray) -> float:
    """Smallest rotation angle between two proper rotation matrices, in rad."""
    relative = np.asarray(current, dtype=float).T @ np.asarray(target, dtype=float)
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(cosine))


def axis_angle_matrix(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.eye(3)
    x, y, z = vector / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def controller_fk(joints: Sequence[float]) -> np.ndarray:
    """FK for the controller's modified-DH model, returning a 4x4 matrix."""
    q = np.asarray(joints, dtype=float)
    if q.shape != (6,):
        raise ValueError("controller FK requires exactly six joint angles")

    def rx(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0],
             [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

    def rz(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array(
            [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

    transform = np.eye(4)
    for index, (a_mm, alpha_deg, d_mm, theta_deg) in enumerate(CONTROLLER_DH):
        translate_x = np.eye(4)
        translate_x[0, 3] = a_mm / 1000.0
        translate_z = np.eye(4)
        translate_z[2, 3] = d_mm / 1000.0
        transform = (
            transform
            @ rx(math.radians(alpha_deg))
            @ translate_x
            @ rz(math.radians(theta_deg) + float(q[index]))
            @ translate_z
        )
    return transform


def rotation_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the principal rotation vector for a proper rotation matrix."""
    matrix = np.asarray(matrix, dtype=float)
    cosine = np.clip((float(np.trace(matrix)) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cosine)
    if angle < 1e-8:
        return 0.5 * np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
    if angle < math.pi - 1e-5:
        return (angle / (2.0 * math.sin(angle))) * np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
    values, vectors = np.linalg.eig(matrix)
    axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.zeros(3)
    return axis * (angle / norm)


def solve_orientation_ik(
    q_initial: Sequence[float],
    target_rotation: np.ndarray,
    *,
    max_iterations: int = 600,
) -> np.ndarray:
    """Solve a minimal joint change for a desired tool rotation.

    Only orientation is constrained.  This is intentional: constraining XYZ
    as well makes the controller reject the large scan rotations at the
    current near-vertical posture.  The damped least-squares update and the
    null-space bias keep the solution close to the current joint vector.
    """
    q_initial = np.asarray(q_initial, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)
    if q_initial.shape != (6,):
        raise ValueError("orientation IK requires exactly six joints")
    if target_rotation.shape != (3, 3):
        raise ValueError("target_rotation must be 3x3")

    q = np.clip(q_initial.copy(), JOINT_LOW, JOINT_HIGH)
    epsilon = 1e-6
    damping = 0.02
    home_bias = 1e-4
    max_iteration_step = 0.20
    orientation_error_norm = float("inf")

    for _ in range(max_iterations):
        current_rotation = controller_fk(q)[:3, :3]
        # World-frame rotation that maps current orientation to target.
        error = rotation_matrix_to_vector(target_rotation @ current_rotation.T)
        orientation_error_norm = float(np.linalg.norm(error))

        jacobian = np.zeros((3, 6), dtype=float)
        for joint_index in range(6):
            q_perturbed = q.copy()
            q_perturbed[joint_index] += epsilon
            perturbed_rotation = controller_fk(q_perturbed)[:3, :3]
            delta = rotation_matrix_to_vector(
                perturbed_rotation @ current_rotation.T
            )
            jacobian[:, joint_index] = delta / epsilon

        normal = jacobian.T @ jacobian + damping * np.eye(6)
        rhs = jacobian.T @ error + home_bias * (q_initial - q)
        try:
            delta_q = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("orientation IK singular system") from exc

        largest = float(np.max(np.abs(delta_q)))
        if largest > max_iteration_step:
            delta_q *= max_iteration_step / largest
        q = np.clip(q + delta_q, JOINT_LOW, JOINT_HIGH)
        if orientation_error_norm <= 1e-4 and largest <= 1e-8:
            break

    if orientation_error_norm > 2e-4:
        raise RuntimeError(
            f"orientation IK did not converge (error={orientation_error_norm:.6f} rad)"
        )
    return q


def solve_pose_ik(
    q_initial: Sequence[float],
    target_position: Sequence[float],
    target_rotation: np.ndarray,
    *,
    max_iterations: int = 1000,
    damping: float = 0.01,
    position_tolerance_m: float = 2e-4,
    rotation_tolerance_rad: float = 2e-4,
) -> np.ndarray:
    """Solve a nearby full-pose target with damped least squares.

    The Codroid controller rejects Cartesian trajectories at the current
    near-vertical posture, but accepts absolute joint targets. This solver
    starts from the measured joint vector and keeps a small home bias so the
    returned solution stays on the same arm configuration.
    """
    q_initial = np.asarray(q_initial, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)
    if q_initial.shape != (6,):
        raise ValueError("pose IK requires exactly six joints")
    if target_position.shape != (3,):
        raise ValueError("pose IK target position must contain three values")
    if target_rotation.shape != (3, 3):
        raise ValueError("pose IK target rotation must be 3x3")

    q = np.clip(q_initial.copy(), JOINT_LOW, JOINT_HIGH)
    epsilon = 1e-6
    home_bias = 1e-6
    max_iteration_step = 0.10
    rotation_characteristic_m = 0.10
    weights = np.array(
        [1.0, 1.0, 1.0,
         rotation_characteristic_m,
         rotation_characteristic_m,
         rotation_characteristic_m],
        dtype=float,
    )
    position_error_norm = float("inf")
    rotation_error_norm = float("inf")

    for _ in range(max_iterations):
        current_transform = controller_fk(q)
        current_position = current_transform[:3, 3]
        current_rotation = current_transform[:3, :3]
        position_error = target_position - current_position
        rotation_error = rotation_matrix_to_vector(
            target_rotation @ current_rotation.T
        )
        position_error_norm = float(np.linalg.norm(position_error))
        rotation_error_norm = float(np.linalg.norm(rotation_error))
        if (
            position_error_norm <= position_tolerance_m
            and rotation_error_norm <= rotation_tolerance_rad
        ):
            break

        jacobian = np.zeros((6, 6), dtype=float)
        for joint_index in range(6):
            q_perturbed = q.copy()
            q_perturbed[joint_index] += epsilon
            perturbed_transform = controller_fk(q_perturbed)
            jacobian[:3, joint_index] = (
                perturbed_transform[:3, 3] - current_position
            ) / epsilon
            jacobian[3:, joint_index] = rotation_matrix_to_vector(
                perturbed_transform[:3, :3] @ current_rotation.T
            ) / epsilon

        error = np.concatenate((position_error, rotation_error))
        weighted_jacobian = weights[:, None] * jacobian
        weighted_error = weights * error
        normal = (
            weighted_jacobian.T @ weighted_jacobian
            + damping * np.eye(6)
        )
        rhs = (
            weighted_jacobian.T @ weighted_error
            + home_bias * (q_initial - q)
        )
        try:
            delta_q = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("pose IK singular system") from exc
        largest = float(np.max(np.abs(delta_q)))
        if largest > max_iteration_step:
            delta_q *= max_iteration_step / largest
        q = np.clip(q + delta_q, JOINT_LOW, JOINT_HIGH)

    if (
        position_error_norm > position_tolerance_m * 2
        or rotation_error_norm > rotation_tolerance_rad * 2
    ):
        raise RuntimeError(
            "pose IK did not converge "
            f"(position={position_error_norm:.6f} m, "
            f"rotation={rotation_error_norm:.6f} rad)"
        )
    return q


def joint_scan_target(
    q_initial: Sequence[float],
    camera_to_tool: np.ndarray,
    *,
    pitch_down_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Find joint angles that pitch/yaw the wrist camera from q_initial."""
    q_initial = np.asarray(q_initial, dtype=float)
    initial_rotation = controller_fk(q_initial)[:3, :3]
    camera_forward = initial_rotation @ np.asarray(camera_to_tool)[:, 2]
    world_down = np.array([0.0, 0.0, -1.0])
    pitch_axis = np.cross(camera_forward, world_down)
    if float(np.linalg.norm(pitch_axis)) < 1e-6:
        pitch_axis = np.array([1.0, 0.0, 0.0])
    pitch_rotation = axis_angle_matrix(
        pitch_axis, math.radians(float(pitch_down_deg))
    )
    yaw_rotation = axis_angle_matrix(
        (0.0, 0.0, 1.0), math.radians(float(yaw_deg))
    )
    target_rotation = yaw_rotation @ pitch_rotation @ initial_rotation
    return solve_orientation_ik(q_initial, target_rotation)


def joints_within_limits(joints: Sequence[float], margin_rad: float = 0.03) -> bool:
    q = np.asarray(joints, dtype=float)
    return bool(
        q.shape == (6,)
        and np.all(np.isfinite(q))
        and np.all(q >= JOINT_LOW + margin_rad)
        and np.all(q <= JOINT_HIGH - margin_rad)
    )


def downward_look_pose(
    base_pose: Pose,
    camera_to_tool: np.ndarray,
    *,
    pitch_down_deg: float,
    yaw_deg: float,
) -> Pose:
    """Pitch the camera toward world -Z, then yaw around world +Z."""
    initial_rotation = rpy_to_matrix(base_pose.rpy)
    camera_forward_tool = camera_to_tool[:, 2]
    camera_forward_base = initial_rotation @ camera_forward_tool
    world_down = np.array([0.0, 0.0, -1.0])
    pitch_axis = np.cross(camera_forward_base, world_down)
    if np.linalg.norm(pitch_axis) < 1e-6:
        pitch_axis = np.array([0.0, 1.0, 0.0])
    pitch_rotation = axis_angle_matrix(pitch_axis, math.radians(pitch_down_deg))
    yaw_rotation = axis_angle_matrix((0.0, 0.0, 1.0), math.radians(yaw_deg))
    rotation = yaw_rotation @ pitch_rotation @ initial_rotation
    return Pose(base_pose.xyz, matrix_to_rpy(rotation))


def clamp_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm == 0.0:
        return vector.copy()
    return vector * (limit / norm)


def select_detection(
    observations: Sequence[Observation], target_class: str
) -> tuple[Detection | None, Observation | None]:
    best: Detection | None = None
    best_obs: Observation | None = None
    for observation in observations:
        for detection in observation.detections:
            if target_class and detection.label != target_class:
                continue
            if best is None or detection.confidence > best.confidence:
                best = detection
                best_obs = observation
    if best is None:
        return None, observations[-1] if observations else None
    return best, best_obs


# ---------------------------------------------------------------------------
# Robot backends
# ---------------------------------------------------------------------------


class RobotBackend:
    def status(self) -> RobotStatus:
        raise NotImplementedError

    def wait_for_status(self, timeout_s: float) -> RobotStatus:
        raise NotImplementedError

    def move_to(self, pose: Pose, *, name: str) -> None:
        raise NotImplementedError

    def spin_once(self, timeout_s: float = 0.05) -> None:
        _ = timeout_s


class MockRobot(RobotBackend):
    def __init__(self) -> None:
        # Approximate the real setup: vertical arm, camera looking horizontally
        # along tool +Z and an object below/in front of the wrist camera.
        self._pose = Pose((0.0, 0.0, 0.30), (-math.pi / 2.0, 0.0, 0.0))
        self.moves: list[tuple[str, Pose]] = []

    def status(self) -> RobotStatus:
        return RobotStatus(True, "AutoReady", 0, self._pose, "mock-ready")

    def wait_for_status(self, timeout_s: float) -> RobotStatus:
        _ = timeout_s
        return self.status()

    def move_to(self, pose: Pose, *, name: str) -> None:
        self.moves.append((name, pose))
        self._pose = pose
        print(f"[mock-arm] {name}: xyz={pose.xyz} rpy={pose.rpy}")


class RosRobot(RobotBackend):
    CARTESIAN_NAMES = ["x", "y", "z", "rx", "ry", "rz"]

    def __init__(self, node, config: ProbeConfig, *, execute: bool) -> None:
        from codroid_msgs.msg import RobotInfo
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        self.node = node
        self.config = config
        self.execute = bool(execute)
        self._lock = threading.Lock()
        self._latest = None
        self._stamp = 0.0
        self._trajectory_type = JointTrajectory
        self._point_type = JointTrajectoryPoint

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self._publisher = node.create_publisher(
            JointTrajectory, config.robotmove_topic, qos
        )
        node.create_subscription(
            RobotInfo, config.robotinfo_topic, self._on_info, qos
        )

    def _on_info(self, msg) -> None:
        with self._lock:
            self._latest = msg
            self._stamp = time.monotonic()

    def spin_once(self, timeout_s: float = 0.05) -> None:
        import rclpy

        rclpy.spin_once(self.node, timeout_sec=timeout_s)

    def status(self) -> RobotStatus:
        with self._lock:
            msg = self._latest
            age = time.monotonic() - self._stamp if self._stamp else float("inf")
        if msg is None or age > self.config.robot_info_timeout_s:
            return RobotStatus(False, "", 0, None, "no fresh /RobotInfo")
        ends = [float(v) for v in msg.end_positions]
        pose = Pose(tuple(ends[:3]), tuple(ends[3:6])) if len(ends) >= 6 else None
        raw_joints = [float(v) for v in msg.joint_positions]
        joints = tuple(raw_joints[:6]) if len(raw_joints) >= 6 else None
        state = str(msg.state)
        fault = int(msg.fault_flag)
        ready = (
            state == "AutoReady"
            and fault == 0
            and pose is not None
            and joints is not None
        )
        return RobotStatus(
            ready,
            state,
            fault,
            pose,
            f"state={state or 'unknown'} fault={fault}",
            joints,
        )

    def wait_for_status(self, timeout_s: float) -> RobotStatus:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.spin_once(0.05)
            status = self.status()
            if status.pose is not None:
                return status
        return self.status()

    def move_to(self, pose: Pose, *, name: str) -> None:
        # Perception/model startup can block the ROS executor long enough that
        # the last callback is stale. Refresh before the safety gate and send.
        status = self.wait_for_status(1.0)
        if not self.execute:
            print(
                f"[dry-run] {name}: would send xyz={pose.xyz} rpy={pose.rpy} "
                f"v={self.config.servo.speed_mps:.3f} m/s"
            )
            return
        if not status.ready:
            raise RuntimeError(f"cannot move {name!r}: {status.message}")
        if status.pose is None or status.joints is None:
            raise RuntimeError(f"cannot move {name!r}: no current pose/joints")

        # The controller silently drops Cartesian trajectories at this
        # near-vertical posture, but absolute joint targets work reliably.
        # Convert the requested Cartesian pose to a nearby joint solution.
        q_current = np.asarray(status.joints, dtype=float)
        target_rotation = rpy_to_matrix(pose.rpy)
        try:
            q_target = solve_pose_ik(
                q_current,
                pose.xyz,
                target_rotation,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"cannot move {name!r}: pose IK failed: {exc}"
            ) from exc
        if not joints_within_limits(q_target):
            raise RuntimeError(
                f"cannot move {name!r}: IK target violates joint limits"
            )

        solved = controller_fk(q_target)
        position_error = float(
            np.linalg.norm(solved[:3, 3] - np.asarray(pose.xyz, dtype=float))
        )
        orientation_error = rotation_angle_error(
            solved[:3, :3], target_rotation
        )
        print(
            f"[robot] plan {name}: xyz={np.round(pose.xyz, 4).tolist()} "
            f"-> joints={np.round(np.degrees(q_target), 2).tolist()} deg "
            f"ik_error=({position_error * 1000:.3f} mm, "
            f"{orientation_error:.5f} rad)"
        )
        self.move_joints(q_target, name=name)

    def move_joints(
        self,
        target: Sequence[float],
        *,
        name: str,
        max_step_deg: float | None = None,
        timeout_s: float | None = None,
        reject_timeout_s: float | None = None,
    ) -> np.ndarray:
        """Move to an absolute joint vector using small interpolated steps."""
        target_q = np.asarray(target, dtype=float)
        if target_q.shape != (6,):
            raise ValueError("joint target must contain six values")
        status = self.wait_for_status(1.0)
        if not self.execute:
            print(
                f"[dry-run] {name}: would send joints="
                f"{np.round(np.degrees(target_q), 2).tolist()} deg"
            )
            return target_q
        if not status.ready:
            raise RuntimeError(f"cannot move {name!r}: {status.message}")
        if status.joints is None:
            raise RuntimeError(f"cannot move {name!r}: no joint positions")
        if not joints_within_limits(target_q):
            raise RuntimeError(
                f"cannot move {name!r}: target is outside joint limits"
            )

        start_q = np.asarray(status.joints, dtype=float)
        max_step_rad = math.radians(
            float(
                max_step_deg
                if max_step_deg is not None
                else self.config.search.joint_max_step_deg
            )
        )
        if max_step_rad <= 0.0:
            raise ValueError("joint max step must be positive")
        largest_delta = float(np.max(np.abs(target_q - start_q)))
        steps = max(1, int(math.ceil(largest_delta / max_step_rad)))
        print(
            f"[robot] {name}: {steps} joint step(s), "
            f"max_delta={math.degrees(largest_delta):.2f} deg"
        )

        current_q = start_q.copy()
        for index in range(1, steps + 1):
            waypoint = start_q + (target_q - start_q) * (index / steps)
            current_q = self._send_joint_waypoint(
                waypoint,
                name=f"{name}_step_{index:02d}",
                timeout_s=(
                    self.config.search.joint_move_timeout_s
                    if timeout_s is None
                    else float(timeout_s)
                ),
                reject_timeout_s=(
                    self.config.search.joint_reject_timeout_s
                    if reject_timeout_s is None
                    else float(reject_timeout_s)
                ),
            )
        return current_q

    def _send_joint_waypoint(
        self,
        target: np.ndarray,
        *,
        name: str,
        timeout_s: float,
        reject_timeout_s: float,
    ) -> np.ndarray:
        status = self.wait_for_status(1.0)
        if not status.ready or status.joints is None:
            raise RuntimeError(f"cannot move {name!r}: {status.message}")
        start_q = np.asarray(status.joints, dtype=float)
        tolerance = math.radians(0.35)
        start_tolerance = math.radians(0.05)

        message = self._trajectory_type()
        message.joint_names = list(JOINT_NAMES)
        point = self._point_type()
        point.positions = [float(v) for v in target]
        message.points.append(point)
        sent_stamp = self._stamp
        start_time = time.monotonic()
        saw_running = False
        self._publisher.publish(message)
        print(
            f"[robot] sent {name}: joints="
            f"{np.round(np.degrees(target), 2).tolist()} deg"
        )

        deadline = start_time + float(timeout_s)
        while time.monotonic() < deadline:
            self.spin_once(self.config.servo.poll_period_s)
            current = self.status()
            if current.fault_flag:
                raise RuntimeError(f"controller fault during {name!r}")
            if current.state == "AutoRunning":
                saw_running = True
            if current.joints is None:
                continue
            actual = np.asarray(current.joints, dtype=float)
            error = float(np.max(np.abs(actual - target)))
            moved = float(np.max(np.abs(actual - start_q))) > start_tolerance
            if (
                current.state == "AutoReady"
                and error <= tolerance
                and (saw_running or moved)
            ):
                return actual
            elapsed = time.monotonic() - start_time
            if (
                self._stamp > sent_stamp
                and current.state == "AutoReady"
                and not saw_running
                and not moved
                and elapsed >= float(reject_timeout_s)
            ):
                raise RuntimeError(
                    f"controller ignored joint move {name!r}; "
                    f"state stayed AutoReady and q did not change "
                    f"(error={math.degrees(error):.3f} deg)"
                )
        raise RuntimeError(f"joint move {name!r} timed out")


# ---------------------------------------------------------------------------
# Perception backends
# ---------------------------------------------------------------------------


class PerceptionBackend:
    def capture(self, pose: Pose | None) -> Observation:
        raise NotImplementedError

    def close(self) -> None:
        return None


class RealSensePerception(PerceptionBackend):
    def __init__(self, config: ProbeConfig) -> None:
        from taskA_grasp.config import CameraConfig
        from taskA_grasp.detector import RealSenseStream, YoloDetector

        self.config = config
        if not config.vision.weights:
            raise RuntimeError("vision.weights is empty")
        if not Path(config.vision.weights).exists():
            raise RuntimeError(f"weights not found: {config.vision.weights}")

        camera = CameraConfig(
            width=config.vision.width,
            height=config.vision.height,
            fps=config.vision.fps,
            serial=config.vision.serial,
            depth_window=config.vision.depth_window,
            warmup_frames=config.vision.warmup_frames,
            anchor=config.vision.anchor,
        )
        self._stream = RealSenseStream(camera, log=print)
        intrinsics = self._stream.start()
        for _ in range(max(1, config.vision.warmup_frames)):
            self._stream.read()
        detector_config = SimpleNamespace(
            weights=config.vision.weights,
            imgsz=config.vision.imgsz,
            confidence=config.vision.confidence,
            device=config.vision.device,
            classes=[],
            camera=camera,
        )
        self._detector = YoloDetector(detector_config, self._stream, log=print)
        self._intrinsics = Intrinsics(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
        if config.vision.rotate_180:
            self._intrinsics = rotate_180_intrinsics(self._intrinsics)
        print(
            "[camera] RealSense ready "
            f"{self._intrinsics.width}x{self._intrinsics.height} "
            f"rotate180={config.vision.rotate_180} "
            f"fx={self._intrinsics.fx:.1f} fy={self._intrinsics.fy:.1f} "
            f"cx={self._intrinsics.cx:.1f} cy={self._intrinsics.cy:.1f}"
        )

    def capture(self, pose: Pose | None) -> Observation:
        _ = pose
        color, depth_m, _intrinsics = self._stream.read()
        if self.config.vision.rotate_180:
            color, depth_m, _intrinsics = rotate_180_rgbd(
                color, depth_m, _intrinsics
            )
        raw = self._detector.detect(color, depth_m, _intrinsics)
        detections = tuple(
            Detection(
                label=item.class_name,
                confidence=float(item.confidence),
                bbox=tuple(int(v) for v in item.bbox),
                u=float(item.u),
                v=float(item.v),
                depth_m=float(item.depth_m),
            )
            for item in raw
        )
        return Observation(color, detections, self._intrinsics)

    def close(self) -> None:
        self._stream.stop()


class MockPerception(PerceptionBackend):
    """Object fixed in base coordinates; view changes as the virtual arm moves."""

    def __init__(self, config: ProbeConfig) -> None:
        self.intrinsics = Intrinsics(640, 480, 600.0, 600.0, 320.0, 240.0)
        # Object is almost exactly under the vertical wrist, not in front.
        # The horizontal camera therefore needs the 80 deg downward view.
        self.object_base = np.array([0.03, 0.01, 0.10], dtype=float)
        self.label = config.vision.target_class or "6-Foam_Brick"
        self.camera_to_tool = config.servo.camera_to_tool.copy()

    def capture(self, pose: Pose | None) -> Observation:
        if pose is None:
            raise RuntimeError("mock perception needs a robot pose")
        rotation = rpy_to_matrix(pose.rpy)
        relative_tool = rotation.T @ (self.object_base - np.asarray(pose.xyz))
        # camera_to_tool is R_tool_camera, so invert it to express the object
        # in the virtual camera's optical frame.
        relative = self.camera_to_tool.T @ relative_tool
        depth = float(relative[2])
        image = np.full((480, 640, 3), 210, dtype=np.uint8)
        if depth <= 0.05:
            return Observation(image, (), self.intrinsics)
        u = self.intrinsics.cx + self.intrinsics.fx * relative[0] / depth
        v = self.intrinsics.cy + self.intrinsics.fy * relative[1] / depth
        if not (30 <= u <= 610 and 30 <= v <= 450):
            return Observation(image, (), self.intrinsics)
        half_w = max(18, int(1800.0 / max(depth, 0.1)))
        half_h = max(24, int(2200.0 / max(depth, 0.1)))
        bbox = (
            max(0, int(u - half_w)),
            max(0, int(v - half_h)),
            min(639, int(u + half_w)),
            min(479, int(v + half_h)),
        )
        detection = Detection(
            label=self.label,
            confidence=0.95,
            bbox=bbox,
            u=float(u),
            v=float(v),
            depth_m=depth,
        )
        try:
            import cv2

            cv2.rectangle(image, bbox[:2], bbox[2:], (40, 180, 40), 2)
            cv2.drawMarker(
                image, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS
            )
            cv2.putText(
                image,
                f"{self.label} depth={depth:.3f}m",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 30, 30),
                2,
            )
        except ImportError:
            pass
        return Observation(image, (detection,), self.intrinsics)


# ---------------------------------------------------------------------------
# Visual servo and search
# ---------------------------------------------------------------------------


def compute_command(
    status: RobotStatus,
    detection: Detection,
    intrinsics: Intrinsics,
    config: ProbeConfig,
) -> ServoCommand:
    if status.pose is None:
        raise RuntimeError("cannot compute servo command without robot pose")
    if detection.depth_m < config.servo.min_depth_m:
        raise RuntimeError(
            f"target depth {detection.depth_m:.3f} m is below safety minimum "
            f"{config.servo.min_depth_m:.3f} m"
        )

    pose = status.pose
    error_u = detection.u - intrinsics.cx
    error_v = detection.v - intrinsics.cy
    normalized_x = error_u / intrinsics.fx
    normalized_y = error_v / intrinsics.fy
    forward_error = detection.depth_m - config.servo.desired_standoff_m

    delta_camera = np.array(
        [
            config.servo.lateral_gain * normalized_x * detection.depth_m,
            config.servo.lateral_gain * normalized_y * detection.depth_m,
            config.servo.forward_gain * forward_error,
        ],
        dtype=float,
    )
    delta_tool = config.servo.camera_to_tool @ delta_camera
    delta_tool = clamp_norm(delta_tool, config.servo.max_step_m)
    delta_base = rpy_to_matrix(pose.rpy) @ delta_tool
    target_xyz = np.asarray(pose.xyz, dtype=float) + delta_base

    converged = (
        abs(error_u) <= config.servo.center_tolerance_px
        and abs(error_v) <= config.servo.center_tolerance_px
        and abs(forward_error) <= config.servo.depth_tolerance_m
    )
    return ServoCommand(
        target=Pose(tuple(float(v) for v in target_xyz), pose.rpy),
        delta_base=delta_base,
        pixel_error=(float(error_u), float(error_v)),
        depth_m=float(detection.depth_m),
        converged=converged,
        reason="centered and within standoff tolerance" if converged else "servo step",
    )


def annotate_and_save(
    observation: Observation,
    detection: Detection,
    config: ProbeConfig,
    index: int,
    tag: str,
) -> Path | None:
    try:
        import cv2
    except ImportError:
        return None
    image = observation.color.copy()
    x0, y0, x1, y1 = detection.bbox
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 180, 0), 2)
    cv2.drawMarker(
        image,
        (int(round(detection.u)), int(round(detection.v))),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        18,
        2,
    )
    cv2.drawMarker(
        image,
        (int(round(observation.intrinsics.cx)), int(round(observation.intrinsics.cy))),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        26,
        2,
    )
    text = (
        f"{detection.label} conf={detection.confidence:.2f} "
        f"depth={detection.depth_m:.3f}m"
    )
    cv2.putText(image, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    config.vision.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.vision.output_dir / f"{tag}_{index:02d}.jpg"
    cv2.imwrite(str(path), image)
    if config.vision.show:
        cv2.imshow("visual_arm_probe", image)
        cv2.waitKey(1)
    return path


def save_observation(
    observation: Observation,
    config: ProbeConfig,
    index: int,
    tag: str,
) -> Path | None:
    """Save a scan frame even when the detector returned nothing."""
    try:
        import cv2
    except ImportError:
        return None
    image = observation.color.copy()
    cv2.putText(
        image,
        f"scan frame {index}: no target",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
    )
    config.vision.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.vision.output_dir / f"{tag}_{index:02d}_nodet.jpg"
    cv2.imwrite(str(path), image)
    if config.vision.show:
        cv2.imshow("visual_arm_probe", image)
        cv2.waitKey(1)
    return path


def grab_best(
    perception: PerceptionBackend,
    pose: Pose,
    config: ProbeConfig,
    frames: int,
    interval_s: float = 0.1,
) -> tuple[Detection | None, Observation | None]:
    observations: list[Observation] = []
    for _ in range(max(1, frames)):
        try:
            observations.append(perception.capture(pose))
        except RuntimeError as exc:
            print(f"[vision] frame failed: {exc}")
        time.sleep(max(0.0, interval_s))
    return select_detection(observations, config.vision.target_class)


def scan_joint_space(
    robot: RosRobot,
    perception: PerceptionBackend,
    config: ProbeConfig,
    *,
    frame_counter: list[int],
) -> tuple[RobotStatus, Detection | None, Observation | None]:
    """Pitch the wrist downward through small joint-space waypoints."""
    initial = robot.wait_for_status(3.0)
    if not initial.ready or initial.joints is None:
        raise RuntimeError(f"cannot start joint scan: {initial.message}")
    q_initial = np.asarray(initial.joints, dtype=float)
    initial_rotation = controller_fk(q_initial)[:3, :3]
    camera_forward = initial_rotation @ config.servo.camera_to_tool[:, 2]
    print(
        "[search] joint-space scan; camera forward in base frame: "
        f"{np.round(camera_forward, 4).tolist()}"
    )

    yaw_offsets = config.search.yaw_offsets_deg if config.search.enabled else (0.0,)
    pitch_offsets = (
        config.search.pitch_down_offsets_deg if config.search.enabled else (0.0,)
    )
    if not yaw_offsets:
        yaw_offsets = (0.0,)
    if not pitch_offsets:
        pitch_offsets = (0.0,)
    views = [(pitch, yaw) for pitch in pitch_offsets for yaw in yaw_offsets]
    print(f"[search] joint-space views (pitch,yaw deg)={views}")

    for index, (pitch_deg, yaw_deg) in enumerate(views):
        if index > 0:
            try:
                q_target = joint_scan_target(
                    q_initial,
                    config.servo.camera_to_tool,
                    pitch_down_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                )
            except RuntimeError as exc:
                print(f"[search] stop before view {index + 1}: {exc}")
                break
            if not joints_within_limits(q_target):
                print(
                    f"[search] stop before view {index + 1}: target joints "
                    "are outside the configured limits"
                )
                break
            target_rotation = controller_fk(q_target)[:3, :3]
            target_forward = target_rotation @ config.servo.camera_to_tool[:, 2]
            print(
                f"[search] view {index + 1}/{len(views)} move "
                f"pitch_down={pitch_deg:+.1f} deg yaw={yaw_deg:+.1f} deg "
                f"forward={np.round(target_forward, 4).tolist()}"
            )
            robot.move_joints(
                q_target,
                name=(
                    f"scan_{index:02d}_pitch_{pitch_deg:+.0f}deg_"
                    f"yaw_{yaw_deg:+.0f}deg"
                ),
            )
            time.sleep(max(0.0, config.search.settle_s))

        status = robot.wait_for_status(3.0)
        if status.pose is None:
            raise RuntimeError(f"lost robot pose while scanning: {status.message}")
        frame_counter[0] += 1
        detection, observation = grab_best(
            perception,
            status.pose,
            config,
            max(1, config.search.observe_frames),
        )
        if observation is not None:
            best_for_image = detection or (
                observation.detections[0] if observation.detections else None
            )
            if best_for_image is not None:
                annotate_and_save(
                    observation,
                    best_for_image,
                    config,
                    frame_counter[0],
                    "scan",
                )
            else:
                save_observation(
                    observation,
                    config,
                    frame_counter[0],
                    "scan",
                )
        print(
            f"[search] view {index + 1}/{len(views)} "
            f"pitch_down={pitch_deg:+.1f} deg yaw={yaw_deg:+.1f} deg "
            f"detections={len(observation.detections) if observation else 0}"
        )
        if detection is not None:
            return robot.status(), detection, observation

    if config.search.return_to_initial_on_miss:
        print("[search] target not found, returning to initial joint pose")
        try:
            robot.move_joints(q_initial, name="scan_return_joint")
        except RuntimeError as exc:
            print(f"[search] return move failed: {exc}")
    return robot.status(), None, None


def scan_for_target(
    robot: RobotBackend,
    perception: PerceptionBackend,
    config: ProbeConfig,
    *,
    execute: bool,
    frame_counter: list[int],
) -> tuple[RobotStatus, Detection | None, Observation | None]:
    initial = robot.wait_for_status(3.0)
    if initial.pose is None:
        raise RuntimeError(f"no robot pose for scanning: {initial.message}")
    if execute and isinstance(robot, RosRobot):
        return scan_joint_space(
            robot,
            perception,
            config,
            frame_counter=frame_counter,
        )
    initial_pose = initial.pose
    camera_forward = rpy_to_matrix(initial_pose.rpy) @ config.servo.camera_to_tool[:, 2]
    print(
        "[search] assumed camera forward in base frame: "
        f"{np.round(camera_forward, 4).tolist()}"
    )

    yaw_offsets = config.search.yaw_offsets_deg if config.search.enabled else (0.0,)
    pitch_offsets = (
        config.search.pitch_down_offsets_deg if config.search.enabled else (0.0,)
    )
    if not yaw_offsets:
        yaw_offsets = (0.0,)
    if not pitch_offsets:
        pitch_offsets = (0.0,)
    views = [(pitch, yaw) for pitch in pitch_offsets for yaw in yaw_offsets]
    if not execute and len(views) > 1:
        print(
            "[search] dry-run: only the current view is checked; planned pitch/yaw "
            f"sweep (deg)={views}"
        )

    for index, (pitch_deg, yaw_deg) in enumerate(views):
        if not execute and index > 0:
            break
        look_pose = downward_look_pose(
            initial_pose,
            config.servo.camera_to_tool,
            pitch_down_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )
        if index > 0:
            if not execute:
                print(
                    f"[search] would look pitch_down={pitch_deg:+.1f} deg "
                    f"yaw={yaw_deg:+.1f} deg"
                )
                continue
            robot.move_to(
                look_pose,
                name=(
                    f"scan_{index:02d}_pitch_{pitch_deg:+.0f}deg_"
                    f"yaw_{yaw_deg:+.0f}deg"
                ),
            )
            time.sleep(max(0.0, config.search.settle_s))

        status = robot.status()
        if status.pose is None:
            raise RuntimeError(f"lost robot pose while scanning: {status.message}")

        frame_counter[0] += 1
        detection, observation = grab_best(
            perception,
            status.pose,
            config,
            max(1, config.search.observe_frames),
        )
        if observation is not None:
            best_for_image = detection or (
                observation.detections[0] if observation.detections else None
            )
            if best_for_image is not None:
                annotate_and_save(
                    observation,
                    best_for_image,
                    config,
                    frame_counter[0],
                    "scan",
                )
        print(
            f"[search] view {index + 1}/{len(views)} "
            f"pitch_down={pitch_deg:+.1f} deg yaw={yaw_deg:+.1f} deg "
            f"detections={len(observation.detections) if observation else 0}"
        )
        if detection is not None:
            return robot.status(), detection, observation

    if execute and len(views) > 1:
        print("[search] target not found, returning to initial pose")
        robot.move_to(initial_pose, name="scan_return")
    elif not execute:
        print(
            "[hint] the target may be outside the current view; after switching "
            "to AutoReady run --execute --scan-only first"
        )
    return robot.status(), None, None


def run_servo(
    robot: RobotBackend,
    perception: PerceptionBackend,
    config: ProbeConfig,
    *,
    execute: bool,
    approach: bool = True,
) -> bool:
    frame_counter = [0]
    status, detection, observation = scan_for_target(
        robot,
        perception,
        config,
        execute=execute,
        frame_counter=frame_counter,
    )
    if detection is None or observation is None:
        print("[result] target not found")
        return False
    if not approach:
        print(
            "[scan-only] target found; stopping before approach "
            f"label={detection.label} conf={detection.confidence:.2f} "
            f"depth={detection.depth_m:.3f}m"
        )
        print("[scan-only] robot remains at the detecting view")
        return True

    if status.pose is None:
        status = robot.wait_for_status(3.0)
    if status.pose is None:
        raise RuntimeError("lost current pose before servo")
    start_position = np.asarray(status.pose.xyz, dtype=float)
    converged_frames = 0

    for step in range(config.servo.max_steps):
        status = robot.wait_for_status(3.0)
        if status.pose is None:
            raise RuntimeError(f"lost robot pose during servo: {status.message}")
        detection, observation = grab_best(
            perception,
            status.pose,
            config,
            max(1, config.search.observe_frames),
        )
        if detection is None or observation is None:
            print(f"[servo] step {step:02d}: target lost")
            return False

        try:
            command = compute_command(status, detection, observation.intrinsics, config)
        except RuntimeError as exc:
            print(f"[abort] {exc}")
            return False
        frame_counter[0] += 1
        annotate_and_save(
            observation, detection, config, frame_counter[0], "servo"
        )
        print(
            f"[servo] step={step:02d} label={detection.label} "
            f"conf={detection.confidence:.2f} "
            f"pixel_error=({command.pixel_error[0]:+.1f},{command.pixel_error[1]:+.1f}) "
            f"depth={command.depth_m:.3f}m "
            f"delta_base={np.round(command.delta_base, 4).tolist()}"
        )
        if command.converged:
            converged_frames += 1
            print(
                f"[servo] converged frame {converged_frames}/"
                f"{config.servo.convergence_frames}"
            )
            if converged_frames >= config.servo.convergence_frames or not execute:
                print("[result] object centered and approach standoff reached")
                return True
            time.sleep(max(0.05, config.servo.settle_s_after_move))
            continue
        converged_frames = 0

        travel = float(np.linalg.norm(np.asarray(command.target.xyz) - start_position))
        if travel > config.servo.max_total_travel_m:
            print(
                f"[abort] total travel {travel:.3f} m exceeds "
                f"{config.servo.max_total_travel_m:.3f} m"
            )
            return False

        low = config.workspace.min_xyz + config.workspace.margin
        high = config.workspace.max_xyz - config.workspace.margin
        target = np.asarray(command.target.xyz)
        if np.any(target < low) or np.any(target > high):
            print(
                f"[abort] target {target.tolist()} outside workspace "
                f"low={low.tolist()} high={high.tolist()}"
            )
            return False
        print(f"[servo] travel={travel:.3f}m target={np.round(target, 4).tolist()}")

        if not execute:
            print(
                "[dry-run] one visual-servo step computed; not publishing. "
                "Run with --mock for a full simulated approach, or --execute on hardware."
            )
            return False
        robot.move_to(command.target, name=f"servo_{step:02d}")
        time.sleep(max(0.0, config.servo.settle_s_after_move))

    print(f"[abort] no convergence after {config.servo.max_steps} steps")
    return False


# ---------------------------------------------------------------------------
# Camera-only test
# ---------------------------------------------------------------------------


def run_camera_only(config: ProbeConfig, frames: int) -> int:
    perception = RealSensePerception(config)
    found = 0
    try:
        for index in range(frames):
            observation = perception.capture(None)
            detection, best_observation = select_detection(
                [observation], config.vision.target_class
            )
            print(
                f"[camera-only] frame={index + 1}/{frames} "
                f"detections={len(observation.detections)}"
            )
            for item in observation.detections:
                print(
                    f"  {item.label} conf={item.confidence:.2f} "
                    f"bbox={item.bbox} center=({item.u:.1f},{item.v:.1f}) "
                    f"depth={item.depth_m:.3f}m"
                )
            if detection is not None and best_observation is not None:
                found += 1
                path = annotate_and_save(
                    best_observation, detection, config, index + 1, "camera"
                )
                if path is not None:
                    print(f"[camera-only] saved {path}")
    finally:
        perception.close()
    print(f"[camera-only] finished, {found}/{frames} frames had a target")
    return 0 if found else 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="YAML configuration",
    )
    parser.add_argument("--mock", action="store_true", help="run without hardware")
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="open RealSense + YOLO and never initialize ROS",
    )
    parser.add_argument(
        "--robot-status",
        action="store_true",
        help="read and print /RobotInfo without opening the camera or moving",
    )
    parser.add_argument("--frames", type=int, default=10, help="camera-only frame count")
    parser.add_argument(
        "--target-class",
        help="override vision.target_class for this run (empty is not allowed)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show annotated frames in a window (requires a desktop display)",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="with --execute, stop after the search finds the target; do not approach",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow small real robot moves (default: dry-run)",
    )
    return parser


def run_mock(config: ProbeConfig, *, approach: bool = True) -> int:
    robot = MockRobot()
    perception = MockPerception(config)
    try:
        ok = run_servo(
            robot, perception, config, execute=True, approach=approach
        )
    finally:
        perception.close()
    print(f"[mock] moves={len(robot.moves)} success={ok}")
    return 0 if ok else 1


def run_real(config: ProbeConfig, *, execute: bool, approach: bool = True) -> int:
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("arm_visual_probe")
    robot = None
    perception = None
    try:
        robot = RosRobot(node, config, execute=execute)
        perception = RealSensePerception(config)
        status = robot.wait_for_status(3.0)
        print(f"[robot] {status.message} pose={status.pose}")
        if execute and not status.ready:
            print("[abort] --execute requires controller state AutoReady")
            return 2
        ok = run_servo(
            robot, perception, config, execute=execute, approach=approach
        )
        return 0 if ok else 1
    finally:
        if perception is not None:
            perception.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except ImportError:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_robot_status(config: ProbeConfig) -> int:
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("arm_visual_probe_status")
    try:
        robot = RosRobot(node, config, execute=False)
        status = robot.wait_for_status(3.0)
        print(f"[robot] {status.message}")
        print(f"[robot] ready={status.ready} pose={status.pose}")
        if status.pose is not None:
            camera_forward = (
                rpy_to_matrix(status.pose.rpy) @ config.servo.camera_to_tool[:, 2]
            )
            print(
                "[robot] assumed camera forward in base frame: "
                f"{np.round(camera_forward, 4).tolist()}"
            )
        return 0 if status.pose is not None else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ProbeConfig.load(Path(args.config).expanduser())
    if args.target_class is not None:
        config = replace(
            config,
            vision=replace(config.vision, target_class=args.target_class),
        )
    if args.show:
        config = replace(config, vision=replace(config.vision, show=True))
    if args.mock:
        return run_mock(config, approach=not bool(args.scan_only))
    if args.camera_only:
        try:
            return run_camera_only(config, max(1, args.frames))
        except (RuntimeError, ValueError) as exc:
            print(f"[error] camera test failed: {exc}")
            print("[hint] check RealSense USB connection/power and /dev/video* nodes")
            return 2
    if args.robot_status:
        try:
            return run_robot_status(config)
        except (RuntimeError, ValueError) as exc:
            print(f"[error] {exc}")
            return 2
    try:
        return run_real(
            config,
            execute=bool(args.execute),
            approach=not bool(args.scan_only),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
