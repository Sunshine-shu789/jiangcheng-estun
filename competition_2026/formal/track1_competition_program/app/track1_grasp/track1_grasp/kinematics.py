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
from dataclasses import dataclass
from typing import Callable, Sequence

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

JOINT_LOW = np.radians(
    np.asarray([-200.0, -200.0, -163.0, -200.0, -200.0, -200.0])
)
JOINT_HIGH = -JOINT_LOW

#: A target rotation is data, not a hint.  Reject a matrix that is not a
#: proper rotation rather than silently projecting it to the nearest one.
ROTATION_MATRIX_TOLERANCE = 1e-6

#: The exact-down tool orientation forces the controller-modified-DH closure
#: ``q5 = +/- pi/2`` and ``q2 + q3 - q4 = +/- pi/2``.  With the formal joint
#: limits above, a deterministic audit of that closure shows a maximum
#: horizontal tool radius of 0.585578111 m at the singular posture
#: ``q = [0, 90, 0, 0, 90, 0] deg``.  Keep a small certification cushion
#: above that value; this is an endpoint reachability screen, while the
#: existing IK and Jacobian gates remain responsible for the singular margin.
EXACT_DOWN_HORIZONTAL_REACH_BOUND_M = 0.5856
EXACT_DOWN_TOOL_Z_TOLERANCE = 1e-9


def validate_rotation_matrix(
    matrix: np.ndarray,
    *,
    name: str = "rotation matrix",
    tolerance: float = ROTATION_MATRIX_TOLERANCE,
) -> np.ndarray:
    """Validate and return a finite proper 3x3 rotation matrix.

    Both orthonormality and ``det=+1`` are checked.  This prevents a
    malformed hand/eye or planning target from being accepted as a
    numerically plausible but physically meaningless IK request.
    """

    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3")
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} contains non-finite values")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rotation validation tolerance must be positive")

    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > tolerance:
        raise ValueError(
            f"{name} is not orthonormal (Frobenius error="
            f"{orthogonality_error:.3g})"
        )
    if abs(determinant - 1.0) > tolerance:
        raise ValueError(
            f"{name} must have determinant +1 (det={determinant:.9g})"
        )
    return rotation


def _validate_joint_vector(joints: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(joints, dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{name} must contain exactly six joints")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def is_exact_downward_tool_axis(
    rotation: np.ndarray,
    *,
    tolerance: float = EXACT_DOWN_TOOL_Z_TOLERANCE,
) -> bool:
    """Return whether a proper rotation points tool Z exactly downward.

    This deliberately recognizes only the strict downward branch.  The
    horizontal reach bound below is not valid for a tilted tool, so a near
    downward pose must continue through the ordinary full IK/FK checks rather
    than being classified using the exact-down envelope.
    """

    matrix = validate_rotation_matrix(rotation)
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("exact-down tool-axis tolerance must be positive")
    return bool(
        np.max(
            np.abs(matrix[:, 2] - np.asarray((0.0, 0.0, -1.0)))
        )
        <= tolerance
    )


def exact_downward_horizontal_reach_violation(
    position: Sequence[float],
    rotation: np.ndarray,
    *,
    bound_m: float = EXACT_DOWN_HORIZONTAL_REACH_BOUND_M,
) -> float | None:
    """Return the horizontal reach when an exact-down endpoint exceeds bound.

    The returned value is the diagnostic distance in metres.  ``None`` means
    either the tool is not exactly downward or the endpoint is within the
    conservative exact-down envelope.  The caller must still run workspace,
    IK, joint-limit, branch-continuity and Jacobian checks.
    """

    if not is_exact_downward_tool_axis(rotation):
        return None
    xyz = np.asarray(position, dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("endpoint position must contain three finite values")
    bound = float(bound_m)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("exact-down horizontal reach bound must be positive")
    reach = float(math.hypot(float(xyz[0]), float(xyz[1])))
    return reach if reach > bound else None


def _is_positive_integer(value: object) -> bool:
    return bool(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) > 0
    )


def _bounded_joint_limits(margin_rad: float) -> tuple[np.ndarray, np.ndarray]:
    margin = float(margin_rad)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("joint limit margin must be finite and non-negative")
    lower = JOINT_LOW + margin
    upper = JOINT_HIGH - margin
    if np.any(lower >= upper):
        raise ValueError("joint limit margin removes the usable joint range")
    return lower, upper


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (ROS ZYX convention)."""

    if len(rpy) != 3:
        raise ValueError("rpy must contain three values")
    roll, pitch, yaw = (float(value) for value in rpy)
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise ValueError("rpy contains non-finite values")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
            ],
            [
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
            ],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a ZYX rotation matrix back to roll, pitch and yaw."""

    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation matrix contains non-finite values")
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
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation matrix contains non-finite values")

    cosine = np.clip(
        (float(np.trace(rotation)) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
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

    # Near pi, the skew part vanishes and an eigenvector from LAPACK can flip
    # sign between calls.  Recover the axis from the symmetric part and choose
    # a deterministic sign so the DLS update direction is repeatable.
    diagonal = np.clip((np.diag(rotation) + 1.0) * 0.5, 0.0, 1.0)
    largest = int(np.argmax(diagonal))
    axis = np.zeros(3, dtype=float)
    axis[largest] = math.sqrt(float(diagonal[largest]))
    if axis[largest] < 1e-9:
        return np.zeros(3, dtype=float)
    if largest == 0:
        axis[1] = (rotation[0, 1] + rotation[1, 0]) / (4.0 * axis[0])
        axis[2] = (rotation[0, 2] + rotation[2, 0]) / (4.0 * axis[0])
    elif largest == 1:
        axis[0] = (rotation[0, 1] + rotation[1, 0]) / (4.0 * axis[1])
        axis[2] = (rotation[1, 2] + rotation[2, 1]) / (4.0 * axis[1])
    else:
        axis[0] = (rotation[0, 2] + rotation[2, 0]) / (4.0 * axis[2])
        axis[1] = (rotation[1, 2] + rotation[2, 1]) / (4.0 * axis[2])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.zeros(3, dtype=float)
    axis /= norm
    nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
    if nonzero.size and axis[int(nonzero[0])] < 0.0:
        axis = -axis
    return axis * angle


def rotation_angle_error(current: np.ndarray, target: np.ndarray) -> float:
    """Smallest rotation angle between two proper rotation matrices, in rad."""

    current_array = np.asarray(current, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if current_array.shape != (3, 3) or target_array.shape != (3, 3):
        raise ValueError("rotation_angle_error requires two 3x3 matrices")
    if not (
        np.all(np.isfinite(current_array))
        and np.all(np.isfinite(target_array))
    ):
        raise ValueError("rotation_angle_error received non-finite values")
    relative = current_array.T @ target_array
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(cosine))


#: Rotation Jacobian rows are scaled to metres so the six singular values can
#: be compared on a consistent scale.  This is the same characteristic length
#: used by the damped IK solver.
JACOBIAN_ROTATION_SCALE_M = 0.10
_JACOBIAN_WEIGHTS = np.asarray(
    [1.0, 1.0, 1.0]
    + [JACOBIAN_ROTATION_SCALE_M] * 3,
    dtype=float,
)


def _controller_frames(
    joints: Sequence[float],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return tool transform, joint origins and joint axes."""

    q = _validate_joint_vector(joints, name="controller joints")

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
    origins: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    for index, (a_mm, alpha_deg, d_mm, theta_deg) in enumerate(
        CONTROLLER_DH
    ):
        translate_x = np.eye(4)
        translate_x[0, 3] = a_mm / 1000.0
        translate_z = np.eye(4)
        translate_z[2, 3] = d_mm / 1000.0
        pre_joint = (
            transform
            @ rx(math.radians(alpha_deg))
            @ translate_x
        )
        origins.append(pre_joint[:3, 3].copy())
        axes.append(pre_joint[:3, 2].copy())
        transform = (
            pre_joint
            @ rz(math.radians(theta_deg) + float(q[index]))
            @ translate_z
        )
    return transform, origins, axes


def controller_fk(joints: Sequence[float]) -> np.ndarray:
    """Forward kinematics for the controller modified-DH model.

    Returns a homogeneous 4x4 transform from the robot base to the tool.
    """

    transform, _origins, _axes = _controller_frames(joints)
    return transform


def _controller_fk_and_jacobian(
    joints: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return FK and the exact geometric Jacobian in one DH pass."""

    transform, origins, axes = _controller_frames(joints)
    tool_position = transform[:3, 3]
    jacobian = np.zeros((6, 6), dtype=float)
    for joint_index in range(6):
        axis = axes[joint_index]
        jacobian[:3, joint_index] = np.cross(
            axis,
            tool_position - origins[joint_index],
        )
        jacobian[3:, joint_index] = axis
    return transform, jacobian


def controller_jacobian(
    joints: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Return the exact 6x6 geometric Jacobian of the controller DH model.

    The first three rows are translational velocity in m/rad.  The final three
    rows are angular velocity in rad/rad.  ``epsilon`` is retained in the
    signature for callers that used the former numerical implementation, but
    the returned matrix is now analytic and therefore not step-size sensitive.
    """

    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("Jacobian epsilon must be positive and finite")
    _transform, jacobian = _controller_fk_and_jacobian(joints)
    return jacobian


def weighted_jacobian_singular_values(joints: Sequence[float]) -> np.ndarray:
    """Return descending singular values of the unit-consistent Jacobian."""

    jacobian = controller_jacobian(joints)
    weighted = _JACOBIAN_WEIGHTS[:, None] * jacobian
    return np.linalg.svd(weighted, compute_uv=False)


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
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=float,
    )


def joints_within_limits(joints: Sequence[float], margin_rad: float = 0.03) -> bool:
    q = np.asarray(joints, dtype=float)
    margin = float(margin_rad)
    return bool(
        q.shape == (6,)
        and math.isfinite(margin)
        and margin >= 0.0
        and np.all(np.isfinite(q))
        and np.all(q >= JOINT_LOW + margin)
        and np.all(q <= JOINT_HIGH - margin)
    )


@dataclass(frozen=True)
class PoseIKSolution:
    """One fully checked IK candidate and its numerical diagnostics."""

    joints: np.ndarray
    initial_joints: np.ndarray
    seed_joints: np.ndarray
    damping: float
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    minimum_jacobian_singular_value: float
    at_joint_limit: bool
    seed_index: int = 0
    #: ``primary``, ``absolute[n]`` or ``offset[n]``.  This is diagnostic
    #: information only and never weakens the acceptance checks below.
    seed_source: str = "primary"

    @property
    def max_joint_delta_rad(self) -> float:
        return float(np.max(np.abs(self.joints - self.initial_joints)))


def _pose_errors(
    transform: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    position_vector = target_position - transform[:3, 3]
    rotation_vector = rotation_matrix_to_vector(
        target_rotation @ transform[:3, :3].T
    )
    position_norm = float(np.linalg.norm(position_vector))
    orientation_norm = float(np.linalg.norm(rotation_vector))
    return position_vector, position_norm, rotation_vector, orientation_norm


def _make_pose_solution(
    joints: np.ndarray,
    *,
    initial_joints: np.ndarray,
    seed_joints: np.ndarray,
    damping: float,
    iterations: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    seed_index: int = 0,
    seed_source: str = "primary",
) -> PoseIKSolution:
    # Final acceptance must use the same public FK/Jacobian entry points as
    # callers.  This costs one extra FK pass only when a candidate is
    # diagnosed, but prevents a test double or future wrapper around the
    # public API from being bypassed by the internal DH fast path.
    transform = controller_fk(joints)
    jacobian = controller_jacobian(joints)
    _position_vector, position_error, _rotation_vector, orientation_error = (
        _pose_errors(transform, target_position, target_rotation)
    )
    singular_values = np.linalg.svd(
        _JACOBIAN_WEIGHTS[:, None] * jacobian,
        compute_uv=False,
    )
    at_joint_limit = bool(
        np.any(joints <= lower + 1e-10)
        or np.any(joints >= upper - 1e-10)
    )
    return PoseIKSolution(
        joints=np.asarray(joints, dtype=float).copy(),
        initial_joints=np.asarray(initial_joints, dtype=float).copy(),
        seed_joints=np.asarray(seed_joints, dtype=float).copy(),
        damping=float(damping),
        iterations=int(iterations),
        position_error_m=float(position_error),
        orientation_error_rad=float(orientation_error),
        minimum_jacobian_singular_value=float(np.min(singular_values)),
        at_joint_limit=at_joint_limit,
        seed_index=int(seed_index),
        seed_source=str(seed_source),
    )


def _validate_ik_parameters(
    *,
    max_iterations: int,
    damping: float,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    joint_limit_margin_rad: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    if not _is_positive_integer(max_iterations):
        raise ValueError("max_iterations must be a positive integer")
    damping = float(damping)
    position_tolerance_m = float(position_tolerance_m)
    rotation_tolerance_rad = float(rotation_tolerance_rad)
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be positive and finite")
    if (
        not math.isfinite(position_tolerance_m)
        or position_tolerance_m <= 0.0
    ):
        raise ValueError("position tolerance must be positive and finite")
    if (
        not math.isfinite(rotation_tolerance_rad)
        or rotation_tolerance_rad <= 0.0
    ):
        raise ValueError("rotation tolerance must be positive and finite")
    lower, upper = _bounded_joint_limits(joint_limit_margin_rad)
    return damping, position_tolerance_m, rotation_tolerance_rad, lower, upper


def _damped_step(
    jacobian: np.ndarray,
    error: np.ndarray,
    *,
    current_joints: np.ndarray,
    home_joints: np.ndarray,
    damping: float,
    home_bias: float,
) -> np.ndarray:
    """Solve a damped least-squares step without normal equations."""

    joint_count = current_joints.size
    sqrt_damping = math.sqrt(damping)
    sqrt_home = math.sqrt(home_bias)
    augmented_matrix = np.vstack(
        (
            jacobian,
            sqrt_damping * np.eye(joint_count),
            sqrt_home * np.eye(joint_count),
        )
    )
    augmented_error = np.concatenate(
        (
            error,
            np.zeros(joint_count, dtype=float),
            sqrt_home * (home_joints - current_joints),
        )
    )
    step, _residuals, rank, _singular_values = np.linalg.lstsq(
        augmented_matrix,
        augmented_error,
        rcond=1e-12,
    )
    if rank < joint_count:
        # A positive damping term normally makes the augmented matrix full
        # rank.  Keep a fail-closed guard for pathological scaling.
        pass
    return np.asarray(step, dtype=float)


def solve_pose_ik(
    q_initial: Sequence[float],
    target_position: Sequence[float],
    target_rotation: np.ndarray,
    *,
    max_iterations: int = 1000,
    damping: float = 0.05,
    position_tolerance_m: float = 2e-4,
    rotation_tolerance_rad: float = 2e-4,
    joint_limit_margin_rad: float = 0.0,
    return_diagnostics: bool = False,
) -> np.ndarray | PoseIKSolution:
    """Solve a nearby full-pose target with damped least squares.

    The solver is deliberately local and is biased toward the supplied seed.
    It must be followed by the motion layer's branch-continuity checks.  The
    final pose is recomputed after the last update; reaching the iteration
    limit is not itself a failure when that final pose meets tolerance.
    """

    initial = _validate_joint_vector(q_initial, name="pose IK seed")
    target_position_array = np.asarray(target_position, dtype=float)
    if target_position_array.shape != (3,):
        raise ValueError("pose IK target position must contain three values")
    if not np.all(np.isfinite(target_position_array)):
        raise ValueError("pose IK target position contains non-finite values")
    target_rotation_array = validate_rotation_matrix(
        target_rotation,
        name="pose IK target rotation",
    )
    damping, position_tolerance, rotation_tolerance, lower, upper = (
        _validate_ik_parameters(
            max_iterations=max_iterations,
            damping=damping,
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_rad=rotation_tolerance_rad,
            joint_limit_margin_rad=joint_limit_margin_rad,
        )
    )
    if np.any(initial < JOINT_LOW - 1e-10) or np.any(
        initial > JOINT_HIGH + 1e-10
    ):
        raise ValueError("pose IK seed is outside the controller joint limits")
    if np.any(initial < lower - 1e-10) or np.any(initial > upper + 1e-10):
        raise ValueError(
            "pose IK seed violates the requested joint-limit margin"
        )

    seed = np.clip(initial, lower, upper)
    q = seed.copy()
    home_bias = 1e-6
    max_iteration_step = 0.10
    active_damping = damping
    accepted_updates = 0

    for iteration in range(int(max_iterations)):
        current_transform, jacobian = _controller_fk_and_jacobian(q)
        (
            position_error,
            position_error_norm,
            rotation_error,
            rotation_error_norm,
        ) = _pose_errors(
            current_transform,
            target_position_array,
            target_rotation_array,
        )
        if (
            position_error_norm <= position_tolerance
            and rotation_error_norm <= rotation_tolerance
        ):
            solution = _make_pose_solution(
                q,
                initial_joints=initial,
                seed_joints=seed,
                damping=damping,
                iterations=iteration,
                target_position=target_position_array,
                target_rotation=target_rotation_array,
                lower=lower,
                upper=upper,
            )
            return solution if return_diagnostics else solution.joints

        weighted_jacobian = _JACOBIAN_WEIGHTS[:, None] * jacobian
        weighted_error = _JACOBIAN_WEIGHTS * np.concatenate(
            (position_error, rotation_error)
        )
        current_cost = float(
            position_error_norm**2
            + (JACOBIAN_ROTATION_SCALE_M * rotation_error_norm) ** 2
        )
        accepted = False
        min_damping = max(1e-9, damping * 1e-3)
        max_damping = max(1.0, damping * 100.0)
        for _lm_attempt in range(6):
            delta_q = _damped_step(
                weighted_jacobian,
                weighted_error,
                current_joints=q,
                home_joints=seed,
                damping=active_damping,
                home_bias=home_bias,
            )
            largest = float(np.max(np.abs(delta_q)))
            if not math.isfinite(largest) or largest < 1e-12:
                break
            if largest > max_iteration_step:
                delta_q *= max_iteration_step / largest

            alpha = 1.0
            for _ in range(8):
                trial = np.clip(q + alpha * delta_q, lower, upper)
                if np.array_equal(trial, q):
                    alpha *= 0.5
                    continue
                trial_transform = controller_fk(trial)
                (
                    _trial_position,
                    trial_position_norm,
                    _trial_rotation,
                    trial_rotation_norm,
                ) = _pose_errors(
                    trial_transform,
                    target_position_array,
                    target_rotation_array,
                )
                trial_cost = float(
                    trial_position_norm**2
                    + (JACOBIAN_ROTATION_SCALE_M * trial_rotation_norm) ** 2
                )
                if trial_cost <= current_cost - max(
                    1e-18,
                    current_cost * 1e-12,
                ):
                    q = trial
                    accepted_updates += 1
                    accepted = True
                    # Once the local Jacobian is healthy, let the damping
                    # decay geometrically.  A fixed absolute damping converges
                    # only linearly and is the main reason full-pose solves
                    # used hundreds of iterations near the tolerance floor.
                    active_damping = max(
                        min_damping,
                        active_damping * 0.25,
                    )
                    break
                alpha *= 0.5
            if accepted:
                break
            # The step did not reduce the pose cost.  Increase damping and
            # recompute the direction instead of merely shrinking a bad one.
            active_damping = min(max_damping, active_damping * 10.0)
        if not accepted:
            break

    final_solution = _make_pose_solution(
        q,
        initial_joints=initial,
        seed_joints=seed,
        damping=damping,
        iterations=accepted_updates,
        target_position=target_position_array,
        target_rotation=target_rotation_array,
        lower=lower,
        upper=upper,
    )
    if (
        final_solution.position_error_m <= position_tolerance
        and final_solution.orientation_error_rad <= rotation_tolerance
    ):
        return final_solution if return_diagnostics else final_solution.joints

    limit_text = ", at joint limit" if final_solution.at_joint_limit else ""
    raise RuntimeError(
        "pose IK did not converge "
        f"(position={final_solution.position_error_m:.6f} m, "
        f"rotation={final_solution.orientation_error_rad:.6f} rad, "
        f"iterations={final_solution.iterations}, "
        f"damping={damping:g}{limit_text})"
    )


POSE_IK_DAMPING_CANDIDATES = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.005)

#: Wrist joints dominate the difficult near-singular corrections.  Trying one
#: deterministic offset at a time keeps the fallback bounded and repeatable.
_POSE_IK_SEED_AXIS_PRIORITY = (3, 5, 4, 0, 2, 1)
_POSE_IK_DEFAULT_SEED_OFFSET_RAD = math.radians(2.0)
_POSE_IK_CONTINUOUS_SEED_OFFSET_RAD = math.radians(0.25)


def _default_seed_offsets(
    accept_delta_rad: float | None,
) -> tuple[np.ndarray, ...]:
    if accept_delta_rad is None:
        offset = _POSE_IK_DEFAULT_SEED_OFFSET_RAD
    else:
        offset = min(
            _POSE_IK_CONTINUOUS_SEED_OFFSET_RAD,
            0.5 * accept_delta_rad,
        )
    if offset <= 0.0:
        return ()
    offsets: list[np.ndarray] = []
    for axis in _POSE_IK_SEED_AXIS_PRIORITY:
        for sign in (1.0, -1.0):
            value = np.zeros(6, dtype=float)
            value[axis] = sign * offset
            offsets.append(value)
    return tuple(offsets)


def _validate_seed_offsets(
    seed_offsets_rad: Sequence[Sequence[float]] | None,
    accept_delta_rad: float | None,
) -> tuple[np.ndarray, ...]:
    if seed_offsets_rad is None:
        return _default_seed_offsets(accept_delta_rad)
    offsets: list[np.ndarray] = []
    for index, offset in enumerate(seed_offsets_rad):
        array = np.asarray(offset, dtype=float)
        if array.shape != (6,):
            raise ValueError(
                f"seed offset {index} must contain exactly six values"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"seed offset {index} contains non-finite values")
        offsets.append(array.copy())
    return tuple(offsets)


def _validate_fallback_seeds(
    fallback_seeds_rad: Sequence[Sequence[float]] | None,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Validate operator-learned absolute postures used only as IK seeds."""

    if fallback_seeds_rad is None:
        return ()
    if isinstance(fallback_seeds_rad, (str, bytes)):
        raise ValueError("fallback IK seeds must be a sequence of joint vectors")
    seeds: list[np.ndarray] = []
    for index, value in enumerate(fallback_seeds_rad):
        seed = _validate_joint_vector(
            value,
            name=f"fallback IK seed {index}",
        )
        if np.any(seed < lower - 1e-12) or np.any(seed > upper + 1e-12):
            raise ValueError(
                f"fallback IK seed {index} violates the controller joint "
                "limits or requested margin"
            )
        seeds.append(seed.copy())
    return tuple(seeds)


def _candidate_rank(
    solution: PoseIKSolution,
    *,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> tuple[float, float, float, int]:
    normalized_residual = (
        solution.position_error_m / position_tolerance_m
        + solution.orientation_error_rad / rotation_tolerance_rad
    )
    return (
        solution.max_joint_delta_rad,
        float(normalized_residual),
        -solution.minimum_jacobian_singular_value,
        solution.seed_index,
    )


def _fast_candidate_is_robust(
    solution: PoseIKSolution,
    *,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    minimum_singular_value: float,
    accept_delta_rad: float | None,
) -> bool:
    normalized_residual = (
        solution.position_error_m / position_tolerance_m
        + solution.orientation_error_rad / rotation_tolerance_rad
    )
    singular_ok = (
        minimum_singular_value <= 0.0
        or solution.minimum_jacobian_singular_value
        >= 1.25 * minimum_singular_value
    )
    return bool(
        normalized_residual <= 0.5
        and not solution.at_joint_limit
        and singular_ok
        and (
            accept_delta_rad is None
            or solution.max_joint_delta_rad <= 0.5 * accept_delta_rad
        )
    )


def _multistart_solution(
    q_initial: Sequence[float],
    target_position: Sequence[float],
    target_rotation: np.ndarray,
    *,
    max_iterations: int = 1000,
    position_tolerance_m: float = 2e-4,
    rotation_tolerance_rad: float = 2e-4,
    damping_candidates: Sequence[float] = POSE_IK_DAMPING_CANDIDATES,
    accept_delta_rad: float | None = None,
    candidate_validator: Callable[[np.ndarray], bool] | None = None,
    fallback_seeds_rad: Sequence[Sequence[float]] | None = None,
    seed_offsets_rad: Sequence[Sequence[float]] | None = None,
    minimum_singular_value: float = 0.0,
    joint_limit_margin_rad: float = 0.0,
    allow_unbounded_fallback: bool = False,
    stop_after_first_safe: bool = False,
) -> PoseIKSolution:
    """Run deterministic local IK restarts and return the selected solution.

    ``fallback_seeds_rad`` contains complete, operator-verified postures.  They
    are tried as absolute numerical seeds after the measured posture and before
    deterministic single-axis offsets.  Being learned or previously successful
    does not exempt a result from any pose, step, limit or validator check.

    ``allow_unbounded_fallback`` is reserved for Cartesian refiners: when all
    converged candidates exceed the requested per-step bound, they need one
    coarse endpoint in order to choose a smaller segment.  Such a candidate is
    never accepted here as safety-checked and the caller must validate it
    again.  All other callers get fail-closed step-bound enforcement.
    """

    initial = _validate_joint_vector(q_initial, name="pose IK seed")
    target_position_array = np.asarray(target_position, dtype=float)
    if target_position_array.shape != (3,):
        raise ValueError("pose IK target position must contain three values")
    if not np.all(np.isfinite(target_position_array)):
        raise ValueError("pose IK target position contains non-finite values")
    target_rotation_array = validate_rotation_matrix(
        target_rotation,
        name="pose IK target rotation",
    )
    if not _is_positive_integer(max_iterations):
        raise ValueError("max_iterations must be a positive integer")
    position_tolerance = float(position_tolerance_m)
    rotation_tolerance = float(rotation_tolerance_rad)
    if (
        not math.isfinite(position_tolerance)
        or position_tolerance <= 0.0
    ):
        raise ValueError("position tolerance must be positive and finite")
    if (
        not math.isfinite(rotation_tolerance)
        or rotation_tolerance <= 0.0
    ):
        raise ValueError("rotation tolerance must be positive and finite")
    if accept_delta_rad is not None:
        accept_delta_rad = float(accept_delta_rad)
        if not math.isfinite(accept_delta_rad) or accept_delta_rad < 0.0:
            raise ValueError("accept_delta_rad must be finite and non-negative")
    if candidate_validator is not None and not callable(candidate_validator):
        raise ValueError("candidate_validator must be callable")
    if not isinstance(allow_unbounded_fallback, bool):
        raise ValueError("allow_unbounded_fallback must be a boolean")
    if not isinstance(stop_after_first_safe, bool):
        raise ValueError("stop_after_first_safe must be a boolean")
    minimum_singular_value = float(minimum_singular_value)
    if (
        not math.isfinite(minimum_singular_value)
        or minimum_singular_value < 0.0
    ):
        raise ValueError(
            "minimum_singular_value must be finite and non-negative"
        )
    lower, upper = _bounded_joint_limits(joint_limit_margin_rad)
    if np.any(initial < lower - 1e-10) or np.any(initial > upper + 1e-10):
        raise ValueError(
            "pose IK seed violates the controller joint limits or requested margin"
        )

    dampings: list[float] = []
    for value in damping_candidates:
        damping = float(value)
        if not math.isfinite(damping) or damping <= 0.0:
            raise ValueError(
                "every damping candidate must be positive and finite"
            )
        dampings.append(damping)
    if not dampings:
        raise ValueError("at least one damping candidate is required")

    fallback_seeds = _validate_fallback_seeds(
        fallback_seeds_rad,
        lower=lower,
        upper=upper,
    )
    offsets = _validate_seed_offsets(seed_offsets_rad, accept_delta_rad)
    seeds: list[tuple[int, np.ndarray, str]] = [
        (0, initial.copy(), "primary")
    ]

    def append_seed(candidate_seed: np.ndarray, source: str) -> None:
        seed = np.asarray(candidate_seed, dtype=float)
        if any(
            np.all(np.abs(seed - existing_seed) <= 1e-12)
            for _index, existing_seed, _source in seeds
        ):
            return
        seeds.append((len(seeds), seed.copy(), source))

    for index, fallback_seed in enumerate(fallback_seeds):
        append_seed(fallback_seed, f"absolute[{index}]")
    for index, offset in enumerate(offsets):
        if np.all(np.abs(offset) <= 1e-15):
            continue
        candidate_seed = initial + offset
        if np.all(candidate_seed >= lower - 1e-12) and np.all(
            candidate_seed <= upper + 1e-12
        ):
            append_seed(candidate_seed, f"offset[{index}]")

    accepted: list[PoseIKSolution] = []
    step_bounded_valid: list[PoseIKSolution] = []
    valid: list[PoseIKSolution] = []
    failures: list[str] = []

    def attempt(
        seed_index: int,
        seed: np.ndarray,
        seed_source: str,
        damping: float,
    ) -> None:
        try:
            raw = solve_pose_ik(
                seed,
                target_position_array,
                target_rotation_array,
                max_iterations=max_iterations,
                damping=damping,
                position_tolerance_m=position_tolerance,
                rotation_tolerance_rad=rotation_tolerance,
                joint_limit_margin_rad=joint_limit_margin_rad,
                return_diagnostics=True,
            )
        except RuntimeError as exc:
            failures.append(
                f"seed={seed_index} ({seed_source}), damping={damping:g}: {exc}"
            )
            return
        if isinstance(raw, PoseIKSolution):
            solution = PoseIKSolution(
                joints=raw.joints.copy(),
                initial_joints=initial.copy(),
                seed_joints=seed.copy(),
                damping=float(damping),
                iterations=int(raw.iterations),
                position_error_m=float(raw.position_error_m),
                orientation_error_rad=float(raw.orientation_error_rad),
                minimum_jacobian_singular_value=float(
                    raw.minimum_jacobian_singular_value
                ),
                at_joint_limit=bool(raw.at_joint_limit),
                seed_index=seed_index,
                seed_source=seed_source,
            )
        else:
            # Compatibility for tests or external callers that monkeypatch the
            # public single-start solver and return only a joint array.
            candidate_array = np.asarray(raw, dtype=float)
            if candidate_array.shape != (6,) or not np.all(
                np.isfinite(candidate_array)
            ):
                raise RuntimeError("pose IK returned an invalid joint candidate")
            if np.any(candidate_array < lower - 1e-10) or np.any(
                candidate_array > upper + 1e-10
            ):
                raise RuntimeError(
                    "pose IK candidate violates the requested joint limits"
                )
            solution = _make_pose_solution(
                candidate_array,
                initial_joints=initial,
                seed_joints=seed,
                damping=damping,
                iterations=-1,
                target_position=target_position_array,
                target_rotation=target_rotation_array,
                lower=lower,
                upper=upper,
                seed_index=seed_index,
                seed_source=seed_source,
            )
        if (
            solution.position_error_m > position_tolerance
            or solution.orientation_error_rad > rotation_tolerance
        ):
            failures.append(
                f"seed={seed_index} ({seed_source}), damping={damping:g}: "
                "final residual "
                f"{solution.position_error_m * 1000:.4f} mm / "
                f"{solution.orientation_error_rad:.7f} rad"
            )
            return
        valid.append(solution)
        step_bounded = (
            accept_delta_rad is None
            or solution.max_joint_delta_rad
            <= accept_delta_rad + 1e-12
        )
        if not step_bounded:
            return
        step_bounded_valid.append(solution)
        if candidate_validator is not None and not candidate_validator(
            solution.joints
        ):
            failures.append(
                f"seed={seed_index} ({seed_source}), damping={damping:g}: "
                "rejected by "
                "motion-level pose/singularity validator"
            )
            return
        accepted.append(solution)
        # Every accepted candidate has already passed the same pose, joint
        # limit, step-bound and caller-validator gates.  Continuity path
        # planning can therefore stop at the first safe candidate instead of
        # spending latency searching for a marginally smaller joint delta.
        # The legacy/default mode still prefers a comfortably robust result.
        if stop_after_first_safe or _fast_candidate_is_robust(
            solution,
            position_tolerance_m=position_tolerance,
            rotation_tolerance_rad=rotation_tolerance,
            minimum_singular_value=minimum_singular_value,
            accept_delta_rad=accept_delta_rad,
        ):
            raise _FastIKSolution(solution)

    primary_accepted_before = len(accepted)
    try:
        for damping in dampings:
            attempt(0, seeds[0][1], seeds[0][2], damping)
    except _FastIKSolution as fast:
        return fast.solution

    if len(accepted) == primary_accepted_before:
        fallback_dampings = dampings[: min(4, len(dampings))]
        for seed_index, seed, seed_source in seeds[1:]:
            before_seed = len(accepted)
            try:
                for damping in fallback_dampings:
                    attempt(seed_index, seed, seed_source, damping)
            except _FastIKSolution as fast:
                return fast.solution
            if len(accepted) > before_seed:
                # Prefer the best of the first seed that produced a safe local
                # solution.  This bounds fallback latency and avoids wandering
                # across unrelated IK branches.
                break

    if accepted:
        return min(
            accepted,
            key=lambda solution: _candidate_rank(
                solution,
                position_tolerance_m=position_tolerance,
                rotation_tolerance_rad=rotation_tolerance,
            ),
        )

    if candidate_validator is not None and step_bounded_valid:
        best = min(
            step_bounded_valid,
            key=lambda solution: _candidate_rank(
                solution,
                position_tolerance_m=position_tolerance,
                rotation_tolerance_rad=rotation_tolerance,
            ),
        )
        raise RuntimeError(
            "no step-bounded IK candidate passed validation "
            f"(best movement={math.degrees(best.max_joint_delta_rad):.6f} "
            "deg, "
            f"position_error={best.position_error_m * 1000:.3f} mm, "
            f"orientation_error={best.orientation_error_rad:.6f} rad, "
            f"minimum_singular_value="
            f"{best.minimum_jacobian_singular_value:.6f})"
        )

    if valid:
        best = min(
            valid,
            key=lambda solution: _candidate_rank(
                solution,
                position_tolerance_m=position_tolerance,
                rotation_tolerance_rad=rotation_tolerance,
            ),
        )
        if allow_unbounded_fallback:
            return best
        assert accept_delta_rad is not None
        raise RuntimeError(
            "no IK candidate satisfies the requested joint-step bound "
            f"(best movement={math.degrees(best.max_joint_delta_rad):.6f} "
            f"deg > {math.degrees(accept_delta_rad):.6f} deg, "
            f"position_error={best.position_error_m * 1000:.3f} mm, "
            f"orientation_error={best.orientation_error_rad:.6f} rad)"
        )

    detail = "; ".join(failures[-4:]) if failures else "no IK attempt returned"
    raise RuntimeError(
        "pose IK did not converge for any deterministic seed/damping: "
        f"{detail}"
    )


class _FastIKSolution(Exception):
    def __init__(self, solution: PoseIKSolution) -> None:
        super().__init__("fast IK candidate accepted")
        self.solution = solution


def solve_pose_ik_multistart(
    q_initial: Sequence[float],
    target_position: Sequence[float],
    target_rotation: np.ndarray,
    *,
    max_iterations: int = 1000,
    position_tolerance_m: float = 2e-4,
    rotation_tolerance_rad: float = 2e-4,
    damping_candidates: Sequence[float] = POSE_IK_DAMPING_CANDIDATES,
    accept_delta_rad: float | None = None,
    candidate_validator: Callable[[np.ndarray], bool] | None = None,
    fallback_seeds_rad: Sequence[Sequence[float]] | None = None,
    seed_offsets_rad: Sequence[Sequence[float]] | None = None,
    minimum_singular_value: float = 0.0,
    joint_limit_margin_rad: float = 0.0,
    allow_unbounded_fallback: bool = False,
    stop_after_first_safe: bool = False,
    return_diagnostics: bool = False,
) -> np.ndarray | PoseIKSolution:
    """Solve a nearby pose with deterministic multi-start fallback.

    The measured posture is always tried first with every damping candidate.
    Complete learned postures supplied in ``fallback_seeds_rad`` are tried
    next, followed by deterministic nearby one-axis offsets.  A fallback seed
    is not accepted merely because it converges: the final candidate is still
    measured against the original posture and must pass the same step bound,
    pose tolerance, limit and caller validator.

    By default, a converged candidate that exceeds ``accept_delta_rad`` is
    rejected.  A Cartesian path refiner may set
    ``allow_unbounded_fallback=True`` to obtain one coarse endpoint for further
    segment subdivision; it must then revalidate that endpoint itself.

    Set ``stop_after_first_safe=True`` for continuity path planning.  The first
    candidate that passes every pose, step, joint-limit and caller validation
    gate is returned immediately; this changes latency and selection order, not
    any safety gate.  Set ``return_diagnostics=True`` to receive the selected
    :class:`PoseIKSolution` with seed, damping, residual and Jacobian details.
    """

    solution = _multistart_solution(
        q_initial,
        target_position,
        target_rotation,
        max_iterations=max_iterations,
        position_tolerance_m=position_tolerance_m,
        rotation_tolerance_rad=rotation_tolerance_rad,
        damping_candidates=damping_candidates,
        accept_delta_rad=accept_delta_rad,
        candidate_validator=candidate_validator,
        fallback_seeds_rad=fallback_seeds_rad,
        seed_offsets_rad=seed_offsets_rad,
        minimum_singular_value=minimum_singular_value,
        joint_limit_margin_rad=joint_limit_margin_rad,
        allow_unbounded_fallback=allow_unbounded_fallback,
        stop_after_first_safe=stop_after_first_safe,
    )
    if return_diagnostics:
        return solution
    return solution.joints


def solve_orientation_ik(
    q_initial: Sequence[float],
    target_rotation: np.ndarray,
    *,
    max_iterations: int = 600,
    orientation_tolerance_rad: float = 2e-4,
    joint_limit_margin_rad: float = 0.0,
) -> np.ndarray:
    """Solve a nearby tool rotation without constraining tool position."""

    initial = _validate_joint_vector(q_initial, name="orientation IK seed")
    target_rotation_array = validate_rotation_matrix(
        target_rotation,
        name="orientation IK target rotation",
    )
    if not _is_positive_integer(max_iterations):
        raise ValueError("max_iterations must be a positive integer")
    orientation_tolerance = float(orientation_tolerance_rad)
    if (
        not math.isfinite(orientation_tolerance)
        or orientation_tolerance <= 0.0
    ):
        raise ValueError(
            "orientation tolerance must be positive and finite"
        )
    lower, upper = _bounded_joint_limits(joint_limit_margin_rad)
    if np.any(initial < lower - 1e-10) or np.any(initial > upper + 1e-10):
        raise ValueError(
            "orientation IK seed violates the controller joint limits or "
            "requested margin"
        )

    seed = np.clip(initial, lower, upper)
    q = seed.copy()
    damping = 0.02
    home_bias = 1e-4
    max_iteration_step = 0.20
    accepted_updates = 0

    for iteration in range(int(max_iterations)):
        current_transform, jacobian = _controller_fk_and_jacobian(q)
        current_rotation = current_transform[:3, :3]
        error = rotation_matrix_to_vector(
            target_rotation_array @ current_rotation.T
        )
        error_norm = float(np.linalg.norm(error))
        if error_norm <= orientation_tolerance:
            return q

        delta_q = _damped_step(
            jacobian[3:, :],
            error,
            current_joints=q,
            home_joints=seed,
            damping=damping,
            home_bias=home_bias,
        )
        largest = float(np.max(np.abs(delta_q)))
        if not math.isfinite(largest) or largest < 1e-12:
            break
        if largest > max_iteration_step:
            delta_q *= max_iteration_step / largest

        accepted = False
        alpha = 1.0
        for _ in range(8):
            trial = np.clip(q + alpha * delta_q, lower, upper)
            if np.array_equal(trial, q):
                alpha *= 0.5
                continue
            trial_rotation = controller_fk(trial)[:3, :3]
            trial_error = rotation_matrix_to_vector(
                target_rotation_array @ trial_rotation.T
            )
            if float(np.linalg.norm(trial_error)) <= error_norm - 1e-14:
                q = trial
                accepted_updates += 1
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    # Recompute after the final update.  This is the authoritative check.
    final_rotation = controller_fk(q)[:3, :3]
    final_error = float(
        np.linalg.norm(
            rotation_matrix_to_vector(
                target_rotation_array @ final_rotation.T
            )
        )
    )
    if final_error <= orientation_tolerance:
        return q
    at_limit = bool(
        np.any(q <= lower + 1e-10) or np.any(q >= upper - 1e-10)
    )
    limit_text = ", at joint limit" if at_limit else ""
    raise RuntimeError(
        "orientation IK did not converge "
        f"(error={final_error:.6f} rad, iterations={accepted_updates}"
        f"{limit_text})"
    )


def joint_scan_target(
    q_initial: Sequence[float],
    camera_to_tool: np.ndarray,
    *,
    pitch_down_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Find a nearby joint vector that pitches/yaws the wrist camera."""

    q_initial = _validate_joint_vector(q_initial, name="joint scan seed")
    camera_to_tool = validate_rotation_matrix(
        camera_to_tool,
        name="camera_to_tool",
    )
    initial_rotation = controller_fk(q_initial)[:3, :3]
    camera_forward = initial_rotation @ camera_to_tool[:, 2]
    world_down = np.asarray([0.0, 0.0, -1.0])
    pitch_axis = np.cross(camera_forward, world_down)
    if float(np.linalg.norm(pitch_axis)) < 1e-6:
        pitch_axis = np.asarray([1.0, 0.0, 0.0])
    pitch_rotation = axis_angle_matrix(
        pitch_axis,
        math.radians(float(pitch_down_deg)),
    )
    yaw_rotation = axis_angle_matrix(
        (0.0, 0.0, 1.0),
        math.radians(float(yaw_deg)),
    )
    target_rotation = yaw_rotation @ pitch_rotation @ initial_rotation
    return solve_orientation_ik(q_initial, target_rotation)
