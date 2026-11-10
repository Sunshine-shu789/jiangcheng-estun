"""Controller-matched kinematics for the Estun/Codroid arm.

The live controller accepts absolute joint targets but silently ignores the
Cartesian ``x,y,z,rx,ry,rz`` trajectories that the old adapter published.
The modified-DH model below was verified against ``/RobotInfo`` on the real
cell, so Cartesian waypoints are converted to nearby joint solutions before
they are sent to ``/RobotMove``.

Keep this model aligned with the controller configuration.  It is deliberately
independent of the URDF on the workstation and of any ROS message package so
the IK helpers remain unit-testable without hardware.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


JOINT_NAMES = tuple(f"Joint{i}" for i in range(1, 7))

#: (a_mm, alpha_deg, d_mm, theta_offset_deg), controller modified-DH.
CONTROLLER_DH = (
    (0.0, 0.0, 176.0, 0.0),
    (0.0, -90.0, 158.964, -90.0),
    (249.388, 0.0, -158.0, 0.0),
    (220.792, 180.0, -106.0, -90.0),
    (0.0, 90.0, -105.546, 0.0),
    (0.0, 90.0, 123.5, 0.0),
)

JOINT_LOW = np.radians(np.asarray([-200.0, -200.0, -163.0, -200.0, -200.0, -200.0]))
JOINT_HIGH = -JOINT_LOW


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (ROS ZYX convention)."""

    if len(rpy) != 3:
        raise ValueError("rpy must contain three values")
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a ZYX rotation matrix back to roll, pitch and yaw."""

    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    pitch = math.atan2(-float(rotation[2, 0]), horizontal)
    if horizontal > 1e-6:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return roll, pitch, yaw


def rotation_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the principal rotation vector of a proper rotation matrix."""

    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    cosine = np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    if angle < 1e-8:
        return 0.5 * skew
    if angle < math.pi - 1e-5:
        return (angle / (2.0 * math.sin(angle))) * skew

    values, vectors = np.linalg.eig(rotation)
    axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.zeros(3, dtype=float)
    return axis * (angle / norm)


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
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    one_minus = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
            [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
            [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
        ],
        dtype=float,
    )


def controller_fk(joints: Sequence[float]) -> np.ndarray:
    """Forward kinematics for the controller modified-DH model.

    Returns a homogeneous 4x4 transform from the robot base to the tool.
    """

    q = np.asarray(joints, dtype=float)
    if q.shape != (6,):
        raise ValueError("controller FK requires exactly six joint angles")

    def rx(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def rz(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
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


def joints_within_limits(joints: Sequence[float], margin_rad: float = 0.03) -> bool:
    q = np.asarray(joints, dtype=float)
    return bool(
        q.shape == (6,)
        and np.all(np.isfinite(q))
        and np.all(q >= JOINT_LOW + margin_rad)
        and np.all(q <= JOINT_HIGH - margin_rad)
    )


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

    The home bias keeps the solution on the same arm configuration as the
    measured joint vector.  It is intentionally a local IK solver: callers
    must reject large changes and move in small joint-space increments.
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
    weights = np.asarray(
        [1.0, 1.0, 1.0, rotation_characteristic_m, rotation_characteristic_m, rotation_characteristic_m],
        dtype=float,
    )
    position_error_norm = float("inf")
    rotation_error_norm = float("inf")

    for _ in range(max_iterations):
        current_transform = controller_fk(q)
        current_position = current_transform[:3, 3]
        current_rotation = current_transform[:3, :3]
        position_error = target_position - current_position
        rotation_error = rotation_matrix_to_vector(target_rotation @ current_rotation.T)
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
        normal = weighted_jacobian.T @ weighted_jacobian + damping * np.eye(6)
        rhs = weighted_jacobian.T @ weighted_error + home_bias * (q_initial - q)
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


def solve_orientation_ik(
    q_initial: Sequence[float],
    target_rotation: np.ndarray,
    *,
    max_iterations: int = 600,
) -> np.ndarray:
    """Solve a nearby tool rotation without constraining tool position."""

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
        error = rotation_matrix_to_vector(target_rotation @ current_rotation.T)
        orientation_error_norm = float(np.linalg.norm(error))

        jacobian = np.zeros((3, 6), dtype=float)
        for joint_index in range(6):
            q_perturbed = q.copy()
            q_perturbed[joint_index] += epsilon
            perturbed_rotation = controller_fk(q_perturbed)[:3, :3]
            jacobian[:, joint_index] = rotation_matrix_to_vector(
                perturbed_rotation @ current_rotation.T
            ) / epsilon

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


def joint_scan_target(
    q_initial: Sequence[float],
    camera_to_tool: np.ndarray,
    *,
    pitch_down_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Find a nearby joint vector that pitches/yaws the wrist camera."""

    q_initial = np.asarray(q_initial, dtype=float)
    camera_to_tool = np.asarray(camera_to_tool, dtype=float)
    if camera_to_tool.shape != (3, 3):
        raise ValueError("camera_to_tool must be 3x3")
    initial_rotation = controller_fk(q_initial)[:3, :3]
    camera_forward = initial_rotation @ camera_to_tool[:, 2]
    world_down = np.asarray([0.0, 0.0, -1.0])
    pitch_axis = np.cross(camera_forward, world_down)
    if float(np.linalg.norm(pitch_axis)) < 1e-6:
        pitch_axis = np.asarray([1.0, 0.0, 0.0])
    pitch_rotation = axis_angle_matrix(pitch_axis, math.radians(float(pitch_down_deg)))
    yaw_rotation = axis_angle_matrix((0.0, 0.0, 1.0), math.radians(float(yaw_deg)))
    target_rotation = yaw_rotation @ pitch_rotation @ initial_rotation
    return solve_orientation_ik(q_initial, target_rotation)
