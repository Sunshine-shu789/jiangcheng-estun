"""Eye-in-hand target selection and small-step visual servoing maths.

This module contains no ROS or camera imports so the safety-critical command
calculation can be unit tested without hardware.  The camera is mounted on the
tool flange; a detection is converted from image/pixel space into a bounded
translation in the robot base frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import MotionConfig, ServoConfig
from .domain import Detection, Frame, Pose, RobotState
from .kinematics import rpy_to_matrix


@dataclass(frozen=True)
class ServoCommand:
    """One bounded Cartesian command produced from one camera frame."""

    target: Pose
    delta_base: np.ndarray
    pixel_error: tuple[float, float]
    depth_m: float
    converged: bool
    reason: str


def clamp_norm(vector: Sequence[float], limit: float) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= limit or norm == 0.0:
        return value.copy()
    return value * (float(limit) / norm)


def select_target(
    frame: Frame,
    *,
    target_class: str = "",
    min_confidence: float = 0.35,
) -> Detection | None:
    """Return the most confident usable target in a frame."""

    candidates = [
        detection
        for detection in frame.detections
        if detection.confidence >= float(min_confidence)
        and (not target_class or detection.label == target_class)
        and detection.depth_m is not None
        and detection.camera_xyz is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda detection: detection.score())


def detection_intrinsics(detection: Detection) -> tuple[float, float, float, float]:
    """Read the pinhole parameters attached by the live vision adapter."""

    try:
        return tuple(
            float(detection.attributes[name])
            for name in ("fx", "fy", "cx", "cy")
        )  # type: ignore[return-value]
    except KeyError as exc:
        raise RuntimeError(
            "live detection is missing fx/fy/cx/cy camera intrinsics"
        ) from exc


def compute_servo_command(
    state: RobotState,
    detection: Detection,
    servo: ServoConfig,
    motion: MotionConfig | None = None,
) -> ServoCommand:
    """Convert one target observation into a bounded base-frame translation.

    Command convention: image +X/+Y/+Z -> tool +X/+Y/+Z according to the
    configured R_tool_camera, then rotate from the current tool orientation
    into the base frame.  Orientation is intentionally held constant.
    """

    if state.pose is None:
        raise RuntimeError("cannot compute a servo command without a robot pose")
    if detection.depth_m is None or detection.depth_m <= 0.0:
        raise RuntimeError("target depth is missing or non-positive")
    depth = float(detection.depth_m)
    if depth < servo.min_depth_m:
        raise RuntimeError(
            f"target depth {depth:.3f} m is below safety minimum "
            f"{servo.min_depth_m:.3f} m"
        )

    u = _optional_float(detection.attributes.get("u"))
    v = _optional_float(detection.attributes.get("v"))
    if u is None or v is None:
        center_x = 0.5 * (float(detection.bbox[0]) + float(detection.bbox[2]))
        center_y = 0.5 * (float(detection.bbox[1]) + float(detection.bbox[3]))
        u = center_x if u is None else u
        v = center_y if v is None else v

    fx, fy, cx, cy = detection_intrinsics(detection)
    error_u = float(u) - cx
    error_v = float(v) - cy
    forward_error = depth - float(servo.desired_standoff_m)
    delta_camera = np.asarray(
        [
            float(servo.lateral_gain) * error_u / fx * depth,
            float(servo.lateral_gain) * error_v / fy * depth,
            float(servo.forward_gain) * forward_error,
        ],
        dtype=float,
    )
    delta_tool = np.asarray(servo.camera_to_tool, dtype=float) @ delta_camera
    delta_tool = clamp_norm(delta_tool, float(servo.max_step_m))
    delta_base = rpy_to_matrix(state.pose.rpy) @ delta_tool
    target_xyz = np.asarray(state.pose.xyz, dtype=float) + delta_base
    target = Pose(tuple(float(v) for v in target_xyz), state.pose.rpy)

    converged = (
        abs(error_u) <= float(servo.center_tolerance_px)
        and abs(error_v) <= float(servo.center_tolerance_px)
        and abs(forward_error) <= float(servo.depth_tolerance_m)
    )
    return ServoCommand(
        target=target,
        delta_base=delta_base,
        pixel_error=(error_u, error_v),
        depth_m=depth,
        converged=converged,
        reason=(
            "centered and within standoff tolerance"
            if converged
            else "servo step"
        ),
    )


def pose_inside_workspace(
    pose: Pose,
    motion: MotionConfig,
) -> tuple[bool, np.ndarray, np.ndarray]:
    low = np.asarray(motion.workspace_min, dtype=float) + float(
        motion.workspace_margin
    )
    high = np.asarray(motion.workspace_max, dtype=float) - float(
        motion.workspace_margin
    )
    point = np.asarray(pose.xyz, dtype=float)
    return bool(np.all(point >= low) and np.all(point <= high)), low, high


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
