"""Motion backends.

The mock backend records the complete path without hardware.  The RobotMove
backend targets the Estun/Codroid ROS bridge already used by taskA_grasp:
``/RobotInfo`` in, ``/RobotMove`` out.  A future MoveIt backend can implement
the same interface without changing the task state machine.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import MotionConfig
from .domain import Pose, RobotState
from .kinematics import (
    EXACT_DOWN_HORIZONTAL_REACH_BOUND_M,
    JOINT_NAMES,
    axis_angle_matrix,
    controller_fk,
    exact_downward_horizontal_reach_violation,
    joints_within_limits,
    matrix_to_rpy,
    rotation_angle_error,
    rotation_matrix_to_vector,
    rpy_to_matrix,
    solve_pose_ik_multistart,
    weighted_jacobian_singular_values,
)
from .speed_limit import (
    DEFAULT_SPEED_SCALE,
    SpeedLimitStore,
    bounded_cartesian_speed_mps,
    bounded_joint_speed_rad_s,
    normalize_backend_speed_scale,
    normalize_profile,
)

# The current UDP bridge serializes all trajectory points into one IPv4 UDP
# datagram.  Keep a hard guard here so a long Cartesian path cannot fail inside
# sendto() after the controller has already accepted the segment pre-check.
UDP_MAX_DATAGRAM_BYTES = 65507

# The live controller accepted a 41-point / 7414-byte command but silently
# ignored a 94-point / 16864-byte command.  The vendor bridge also uses an
# 8192-byte receive buffer, so keep a deliberate margin below that boundary.
# Long pre-checked paths are split into sequential controller-safe chunks.
ROBOT_CONTROLLER_MAX_TRAJECTORY_POINTS = 40
ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES = 7600

#: A speed override must never turn a slow trajectory into an unbounded wait.
MAX_TRAJECTORY_TIMEOUT_S = 900.0

#: Task A may drive these zero-based ROS joint indices (J1/J2/J3/J4).
TASK_A_MOVABLE_JOINT_INDICES = frozenset({0, 1, 2, 3})


@dataclass(frozen=True)
class JointPathMetrics:
    """Per-axis continuity metrics for one planned joint path."""

    max_step_deg: tuple[float, ...]
    max_acceleration_deg: tuple[float, ...]
    travel_deg: tuple[float, ...]


@dataclass(frozen=True)
class LinearSegmentPlan:
    """One straight TCP segment, pre-checked before anything is published."""

    target: Pose
    waypoints: tuple[Pose, ...]
    joint_waypoints: tuple[tuple[float, ...], ...] = ()
    planning_step_m: float = 0.0
    max_joint_step_deg: float = 0.0
    max_joint_acceleration_deg: float = 0.0
    min_jacobian_singular_value: float = 0.0
    joint_metrics: JointPathMetrics | None = None

    @property
    def distance_m(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        return _pose_distance(self.waypoints[0], self.waypoints[-1])


@dataclass(frozen=True)
class LinearRoutePlan:
    """A sequence of straight segments that has passed the full pre-check."""

    start_pose: Pose
    segments: tuple[LinearSegmentPlan, ...]

    @property
    def targets(self) -> tuple[Pose, ...]:
        return tuple(segment.target for segment in self.segments)

    @property
    def distances_m(self) -> tuple[float, ...]:
        return tuple(segment.distance_m for segment in self.segments)


@dataclass(frozen=True)
class JointRoutePlan:
    """Fully pre-checked absolute joint targets for the Task A base arc."""

    start_joints: tuple[float, ...]
    targets: tuple[tuple[float, ...], ...]
    waypoints: tuple[tuple[tuple[float, ...], ...], ...]
    tcp_targets: tuple[Pose, ...]
    distances_m: tuple[float, ...]
    fixed_indices: tuple[int, ...]

    @property
    def start_joints_deg(self) -> tuple[float, ...]:
        return tuple(math.degrees(value) for value in self.start_joints)

    @property
    def targets_deg(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(math.degrees(value) for value in target)
            for target in self.targets
        )


class _CartesianPathError(RuntimeError):
    """Internal planning error with an explicit retry classification."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)


def _pose_distance(start: Pose, end: Pose) -> float:
    delta = np.asarray(end.xyz, dtype=float) - np.asarray(start.xyz, dtype=float)
    return float(np.linalg.norm(delta))


def _joint_path_travel_rad(
    start: Sequence[float],
    targets: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Return the conservative summed per-point maximum joint travel."""

    previous = np.asarray(start, dtype=float)
    total = 0.0
    for target in np.asarray(targets, dtype=float):
        current = np.asarray(target, dtype=float)
        total += float(np.max(np.abs(current - previous)))
        previous = current
    return total


def _interpolate_rotation(
    start_rotation: np.ndarray,
    target_rotation: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate orientation along the shortest rotational path."""

    relative = target_rotation @ start_rotation.T
    vector = rotation_matrix_to_vector(relative)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return start_rotation.copy()
    return axis_angle_matrix(vector, float(fraction) * angle) @ start_rotation


def _pose_at_fraction(start: Pose, target: Pose, fraction: float) -> Pose:
    """Interpolate position and shortest-path orientation at one fraction."""

    value = float(fraction)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("pose interpolation fraction must be in [0, 1]")
    start_xyz = np.asarray(start.xyz, dtype=float)
    target_xyz = np.asarray(target.xyz, dtype=float)
    xyz = start_xyz + (target_xyz - start_xyz) * value
    rotation = _interpolate_rotation(
        rpy_to_matrix(start.rpy),
        rpy_to_matrix(target.rpy),
        value,
    )
    return Pose(tuple(float(v) for v in xyz), matrix_to_rpy(rotation))


def linear_pose_waypoints(
    start: Pose,
    target: Pose,
    *,
    step_m: float,
    orientation_step_rad: float | None = None,
) -> tuple[Pose, ...]:
    """Return a straight-line TCP path with no spacing larger than ``step_m``."""

    step = float(step_m)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("linear step must be a positive finite distance")
    orientation_step = (
        None
        if orientation_step_rad is None
        else float(orientation_step_rad)
    )
    if orientation_step is not None and (
        not math.isfinite(orientation_step) or orientation_step <= 0.0
    ):
        raise ValueError("orientation step must be positive and finite")

    start_xyz = np.asarray(start.xyz, dtype=float)
    target_xyz = np.asarray(target.xyz, dtype=float)
    if not np.all(np.isfinite(start_xyz)) or not np.all(np.isfinite(target_xyz)):
        raise ValueError("linear path endpoints must be finite")

    distance = float(np.linalg.norm(target_xyz - start_xyz))
    rotation_angle = rotation_angle_error(
        rpy_to_matrix(start.rpy),
        rpy_to_matrix(target.rpy),
    )
    if distance < 1e-12 and rotation_angle < 1e-12:
        return (start,)
    # Avoid an extra waypoint when a decimal step divides the distance
    # exactly but binary floating point produces 4.000000000000001.
    translation_steps = max(1, int(math.ceil(distance / step - 1e-12)))
    rotation_steps = 1
    if orientation_step is not None:
        rotation_steps = max(
            1,
            int(math.ceil(rotation_angle / orientation_step - 1e-12)),
        )
    steps = max(translation_steps, rotation_steps)
    waypoints: list[Pose] = [start]
    for index in range(1, steps + 1):
        fraction = index / steps
        waypoints.append(_pose_at_fraction(start, target, fraction))
    return tuple(waypoints)


def _validate_workspace(config: MotionConfig, pose: Pose, label: str) -> None:
    low = np.asarray(config.workspace_min, dtype=float) + float(
        config.workspace_margin
    )
    high = np.asarray(config.workspace_max, dtype=float) - float(
        config.workspace_margin
    )
    xyz = np.asarray(pose.xyz, dtype=float)
    if np.any(xyz < low - 1e-9) or np.any(xyz > high + 1e-9):
        raise RuntimeError(
            f"{label} is outside the configured workspace: "
            f"xyz={tuple(round(float(v), 4) for v in xyz)}"
        )


def _validate_exact_downward_reach(pose: Pose, *, label: str) -> None:
    """Reject an exact-down endpoint beyond the controller DH reach envelope.

    The ordinary workspace is intentionally larger than the physical
    exact-down reach because tilted grasps have a different envelope.  This
    early endpoint screen applies only when tool Z is strictly downward.  It
    must run before IK/route subdivision so an unreachable visual target cannot
    consume the full planning retry budget or be mistaken for an IK/singularity
    tuning problem.
    """

    violation = exact_downward_horizontal_reach_violation(
        pose.xyz,
        rpy_to_matrix(pose.rpy),
    )
    if violation is None:
        return
    raise RuntimeError(
        f"{label} downward tool reach {violation:.6f} m exceeds validated "
        f"exact-down bound {EXACT_DOWN_HORIZONTAL_REACH_BOUND_M:.6f} m; "
        "move object/box inward or re-teach observation; adding waypoints "
        "cannot make this endpoint reachable"
    )


def _validate_joint_waypoint_workspace(
    config: MotionConfig,
    joint_waypoints: Sequence[np.ndarray],
    *,
    name: str,
) -> None:
    """Reject a joint-space interpolation before any waypoint is published.

    Joint interpolation is not guaranteed to be a straight Cartesian path.
    Checking every generated joint waypoint with forward kinematics closes the
    gap between an endpoint workspace check and the actual path sent to the
    controller.
    """

    low = np.asarray(config.workspace_min, dtype=float) + float(
        config.workspace_margin
    )
    high = np.asarray(config.workspace_max, dtype=float) - float(
        config.workspace_margin
    )
    count = len(joint_waypoints)
    for index, joint_waypoint in enumerate(joint_waypoints, start=1):
        xyz = np.asarray(controller_fk(joint_waypoint)[:3, 3], dtype=float)
        if np.any(xyz < low - 1e-9) or np.any(xyz > high + 1e-9):
            formatted = tuple(round(float(value), 4) for value in xyz)
            raise RuntimeError(
                f"joint move {name!r} waypoint {index}/{count} would leave "
                f"the configured workspace: xyz={formatted}, "
                f"low={tuple(round(float(v), 4) for v in low)}, "
                f"high={tuple(round(float(v), 4) for v in high)}"
            )


def _joint_vector(value: Sequence[float], *, label: str) -> tuple[float, ...]:
    vector = tuple(float(item) for item in value)
    if len(vector) != 6:
        raise ValueError(f"{label} must contain six joint values")
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{label} must contain six finite joint values")
    return vector


def _pose_from_joints(joints: Sequence[float]) -> Pose:
    transform = controller_fk(joints)
    xyz = tuple(float(value) for value in transform[:3, 3])
    rpy = matrix_to_rpy(transform[:3, :3])
    return Pose(xyz, rpy)


def _joint_error_deg(
    left: Sequence[float],
    right: Sequence[float],
    *,
    indices: Sequence[int] | None = None,
) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    selected = (
        tuple(range(6))
        if indices is None
        else tuple(int(index) for index in indices)
    )
    if not selected:
        return 0.0
    return math.degrees(
        float(np.max(np.abs(left_array[list(selected)] - right_array[list(selected)])))
    )


def _robot_state_detail(state: RobotState) -> str:
    """Format controller feedback for an actionable motion error."""

    joints = (
        tuple(round(math.degrees(value), 3) for value in state.joints)
        if state.joints is not None
        else None
    )
    return (
        f"controller_state={state.controller_state or 'unknown'!r} "
        f"fault_flag={state.fault_flag} ready={state.ready} "
        f"message={state.message!r} joints_deg={joints}"
    )


def _build_joint_route_plan(
    config: MotionConfig,
    state: RobotState,
    targets: Sequence[Sequence[float]],
    *,
    allowed_joints: Sequence[int],
    max_initial_move_deg: float,
    max_target_move_deg: float,
    max_step_deg: float,
    min_endpoint_distance_m: float,
    name_prefix: str,
) -> JointRoutePlan:
    """Validate every target and every interpolated waypoint before motion."""

    if state.fault:
        raise RuntimeError(f"cannot pre-check joint route: {state.message}")
    if not state.ready or state.joints is None:
        raise RuntimeError(
            "cannot pre-check joint route without ready six-axis feedback: "
            f"{state.message}"
        )
    start = _joint_vector(state.joints, label="startup joint feedback")
    allowed = tuple(dict.fromkeys(int(index) for index in allowed_joints))
    if not allowed or any(index < 0 or index >= 6 for index in allowed):
        raise ValueError("allowed_joints must contain one or more indexes 0..5")
    invalid_allowed = set(allowed) - TASK_A_MOVABLE_JOINT_INDICES
    if invalid_allowed:
        raise ValueError(
            "Task A may move only Joint1, Joint2, Joint3 and Joint4; "
            "Joint5 and Joint6 must remain locked; invalid indexes="
            f"{sorted(invalid_allowed)}"
        )
    fixed_indices = tuple(index for index in range(6) if index not in allowed)

    initial_limit = float(max_initial_move_deg)
    target_limit = float(max_target_move_deg)
    step_deg = float(max_step_deg)
    minimum_distance = float(min_endpoint_distance_m)
    for label, value in (
        ("max_initial_move_deg", initial_limit),
        ("max_target_move_deg", target_limit),
        ("max_step_deg", step_deg),
        ("min_endpoint_distance_m", minimum_distance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be positive and finite")
    if not targets:
        raise ValueError("joint route must contain at least one target")

    start_array = np.asarray(start, dtype=float)
    current = start_array
    previous_pose = _pose_from_joints(start)
    margin_rad = math.radians(float(config.joint_limit_margin_deg))
    fixed_tolerance_rad = math.radians(1e-7)
    planned_targets: list[tuple[float, ...]] = []
    planned_waypoints: list[tuple[tuple[float, ...], ...]] = []
    tcp_targets: list[Pose] = []
    distances: list[float] = []
    step_rad = math.radians(step_deg)

    for target_index, raw_target in enumerate(targets, start=1):
        target = _joint_vector(
            raw_target,
            label=f"patrol target {target_index}",
        )
        target_array = np.asarray(target, dtype=float)
        if not joints_within_limits(target_array, margin_rad):
            raise RuntimeError(
                f"patrol target {target_index} violates the configured "
                f"{config.joint_limit_margin_deg:g} degree joint-limit margin"
            )
        fixed_error = _joint_error_deg(
            target,
            start,
            indices=fixed_indices,
        )
        if math.radians(fixed_error) > fixed_tolerance_rad:
            raise RuntimeError(
                f"patrol target {target_index} changes a locked joint by "
                f"{fixed_error:.6f} degrees; Task A may move only "
                "Joint1, Joint2, Joint3 and Joint4"
            )

        move_deg = _joint_error_deg(target, current)
        limit_deg = initial_limit if target_index == 1 else target_limit
        if move_deg > limit_deg + 1e-9:
            suffix = "initial approach" if target_index == 1 else "adjacent patrol move"
            raise RuntimeError(
                f"patrol target {target_index} requires {move_deg:.3f} degrees "
                f"on one axis for the {suffix}; limit is {limit_deg:.3f} degrees"
            )

        largest_delta_rad = float(np.max(np.abs(target_array - current)))
        steps = max(1, int(math.ceil(largest_delta_rad / step_rad - 1e-12)))
        segment_waypoints = tuple(
            tuple(
                float(value)
                for value in (
                    current + (target_array - current) * (index / steps)
                )
            )
            for index in range(1, steps + 1)
        )
        _validate_joint_waypoint_workspace(
            config,
            tuple(np.asarray(item, dtype=float) for item in segment_waypoints),
            name=f"{name_prefix}-{target_index:02d}",
        )

        target_pose = _pose_from_joints(target)
        distance = _pose_distance(previous_pose, target_pose)
        if distance + 1e-9 < minimum_distance:
            raise RuntimeError(
                f"patrol target {target_index} is only {distance:.3f} m from "
                "the previous endpoint; Task A needs at least "
                f"{minimum_distance:.3f} m"
            )

        planned_targets.append(target)
        planned_waypoints.append(segment_waypoints)
        tcp_targets.append(target_pose)
        distances.append(distance)
        current = target_array
        previous_pose = target_pose

    return JointRoutePlan(
        start_joints=start,
        targets=tuple(planned_targets),
        waypoints=tuple(planned_waypoints),
        tcp_targets=tuple(tcp_targets),
        distances_m=tuple(distances),
        fixed_indices=fixed_indices,
    )


class MockMotionBackend:
    """Deterministic motion simulator for framework testing."""

    def __init__(
        self,
        config: MotionConfig,
        log=None,
        *,
        profile: str | None = None,
        speed_store: SpeedLimitStore | None = None,
    ) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self.profile = normalize_profile(profile) if profile else None
        self._speed_store = speed_store or SpeedLimitStore()
        self._pose = Pose((0.0, 0.0, 0.30), config.fixed_rpy)
        self._joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.moves: list[tuple[str, Pose]] = []
        #: Kept separate from ``moves`` so existing pick-task tests retain the
        #: historical one-entry-per-waypoint behaviour.
        self.linear_moves: list[tuple[str, tuple[Pose, ...]]] = []
        self.joint_moves: list[tuple[str, tuple[float, ...]]] = []

    @property
    def dry_run(self) -> bool:
        return bool(self.config.dry_run)

    def get_state(self) -> RobotState:
        return RobotState(
            ready=True,
            pose=self._pose,
            message="mock-ready",
            joints=self._joints,
            controller_state="MockReady",
            fault_flag=0,
        )

    def get_feedback_state(self) -> RobotState:
        """Mock feedback and simulated planning state are intentionally equal."""

        return self.get_state()

    def _speed_scale(self) -> float:
        # Several tests and compatibility callers construct a backend via
        # ``__new__`` and do not initialize the speed override members.  Such
        # callers retain the historical 100% configuration cap.
        profile = getattr(self, "profile", None)
        store = getattr(self, "_speed_store", None)
        if profile is None or store is None:
            return DEFAULT_SPEED_SCALE
        try:
            # Re-validate at the consumer boundary as well.  A broken custom
            # store must never pass zero or NaN through to a trajectory.
            return normalize_backend_speed_scale(store.scale_for(profile))
        except Exception as exc:
            log = getattr(self, "_log", None)
            if callable(log):
                log(
                    f"[motion:speed-limit] could not read profile "
                    f"{profile}: {exc}; using 100% configuration cap"
                )
            return DEFAULT_SPEED_SCALE

    def move_to(self, pose: Pose, *, name: str, speed_mps: float) -> None:
        effective_speed = bounded_cartesian_speed_mps(
            speed_mps,
            self.config.speed_mps,
            self._speed_scale(),
        )
        self.moves.append((name, pose))
        self._pose = pose
        self._log(
            f"[motion:mock] {name}: xyz={pose.xyz} rpy={pose.rpy} "
            f"v={effective_speed:.3f}"
        )

    def move_joints(
        self,
        target: Sequence[float],
        *,
        name: str,
        max_step_deg: float | None = None,
        timeout_s: float | None = None,
        reject_timeout_s: float | None = None,
    ) -> np.ndarray:
        target_q = np.asarray(target, dtype=float)
        if target_q.shape != (6,) or not np.all(np.isfinite(target_q)):
            raise ValueError("joint target must contain six finite values")
        if not joints_within_limits(target_q):
            raise RuntimeError(f"joint target for {name!r} violates limits")
        self.joint_moves.append(
            (str(name), tuple(float(value) for value in target_q))
        )
        self._joints = tuple(float(value) for value in target_q)
        joint_speed = bounded_joint_speed_rad_s(
            self.config.joint_speed_rad_s,
            self.config.joint_speed_safety_limit_rad_s,
            self._speed_scale(),
        )
        self._log(
            f"[motion:mock-joint] {name}: joints="
            f"{np.round(np.degrees(target_q), 2).tolist()} deg "
            f"joint_velocity={joint_speed:.3f} rad/s"
        )
        return target_q

    def precheck_pose_condition(self, pose: Pose, *, label: str) -> float | None:
        """Return no Jacobian value for the kinematic-free mock backend."""

        return None

    def precheck_linear_targets(
        self, targets: Sequence[Pose], *, step_m: float
    ) -> LinearRoutePlan:
        start_pose = self._pose
        current = start_pose
        segments: list[LinearSegmentPlan] = []
        for target_index, target in enumerate(targets, start=1):
            waypoints = linear_pose_waypoints(current, target, step_m=step_m)
            for waypoint_index, waypoint in enumerate(waypoints, start=1):
                _validate_workspace(
                    self.config,
                    waypoint,
                    f"patrol target {target_index} waypoint {waypoint_index}",
                )
            segments.append(LinearSegmentPlan(target=target, waypoints=waypoints))
            current = target
        return LinearRoutePlan(start_pose=start_pose, segments=tuple(segments))

    def precheck_joint_route(
        self,
        targets: Sequence[Sequence[float]],
        *,
        allowed_joints: Sequence[int],
        max_initial_move_deg: float,
        max_target_move_deg: float,
        max_step_deg: float | None = None,
        min_endpoint_distance_m: float,
        name_prefix: str = "task-a-patrol",
    ) -> JointRoutePlan:
        """Validate an absolute joint route without publishing a command."""

        return _build_joint_route_plan(
            self.config,
            self.get_state(),
            targets,
            allowed_joints=allowed_joints,
            max_initial_move_deg=max_initial_move_deg,
            max_target_move_deg=max_target_move_deg,
            max_step_deg=(
                self.config.joint_max_step_deg
                if max_step_deg is None
                else max_step_deg
            ),
            min_endpoint_distance_m=min_endpoint_distance_m,
            name_prefix=name_prefix,
        )

    def execute_joint_route(
        self,
        route: JointRoutePlan,
        *,
        names: Sequence[str],
    ) -> None:
        """Execute a route that was already fully pre-checked."""

        if len(names) != len(route.targets):
            raise ValueError("one name is required for every joint target")
        for index, name in enumerate(names):
            self.execute_joint_route_step(route, index=index, name=str(name))

    def execute_joint_route_step(
        self,
        route: JointRoutePlan,
        *,
        index: int,
        name: str,
    ) -> None:
        """Execute one point from a route that was fully pre-checked."""

        if index < 0 or index >= len(route.targets):
            raise IndexError(
                f"joint route step index {index} is outside "
                f"0..{len(route.targets) - 1}"
            )
        movable = tuple(
            joint_index
            for joint_index in range(6)
            if joint_index not in route.fixed_indices
        )
        if not movable or not set(movable).issubset(
            TASK_A_MOVABLE_JOINT_INDICES
        ):
            raise RuntimeError(
                f"refusing mock joint route {name!r}: Task A may move only "
                "Joint1/Joint2/Joint3/Joint4; "
                f"movable_indices={movable}"
            )
        state = self.get_state()
        if not state.ready or state.joints is None:
            raise RuntimeError(
                f"mock joint route {name!r} is not ready: {state.message}"
            )
        fixed_error_deg = _joint_error_deg(
            state.joints,
            route.start_joints,
            indices=route.fixed_indices,
        )
        if fixed_error_deg > 0.10:
            raise RuntimeError(
                f"mock joint route {name!r} fixed-joint drift "
                f"{fixed_error_deg:.3f} deg exceeds 0.100 deg"
            )
        self.move_joints(
            route.targets[index],
            name=name,
            max_step_deg=float(self.config.joint_max_step_deg),
        )

    def execute_linear_route(
        self, route: LinearRoutePlan, *, names: Sequence[str], speed_mps: float
    ) -> None:
        if len(names) != len(route.segments):
            raise ValueError("one name is required for every linear segment")
        for name, segment in zip(names, route.segments):
            effective_speed = bounded_cartesian_speed_mps(
                speed_mps,
                self.config.speed_mps,
                self._speed_scale(),
            )
            self.linear_moves.append((str(name), segment.waypoints))
            self._pose = segment.target
            self._log(
                f"[motion:mock-linear] {name}: distance={segment.distance_m:.3f} m "
                f"waypoints={max(0, len(segment.waypoints) - 1)} "
                f"v={effective_speed:.3f}"
            )

    def execute_prechecked_segment(
        self,
        segment: LinearSegmentPlan,
        *,
        name: str,
        speed_mps: float,
    ) -> None:
        """Record one segment that was already validated by a route precheck."""

        effective_speed = bounded_cartesian_speed_mps(
            speed_mps,
            self.config.speed_mps,
            self._speed_scale(),
        )
        self.moves.append((str(name), segment.target))
        self._pose = segment.target
        self._log(
            f"[motion:mock-prechecked] {name}: "
            f"distance={segment.distance_m:.3f} m "
            f"waypoints={max(0, len(segment.waypoints) - 1)} "
            f"v={effective_speed:.3f}"
        )

    def move_linear_to(
        self,
        pose: Pose,
        *,
        name: str,
        speed_mps: float,
        step_m: float | None = None,
    ) -> None:
        route = self.precheck_linear_targets(
            (pose,),
            step_m=(
                self.config.cartesian_step_m
                if step_m is None
                else float(step_m)
            ),
        )
        self.execute_linear_route(route, names=(name,), speed_mps=speed_mps)

    def close(self) -> None:
        return None


class RobotMoveMotionBackend:
    """Absolute-joint publisher for the Codroid ``RobotMove`` bridge.

    The live controller silently ignores Cartesian ``x,y,z,rx,ry,rz``
    trajectories in the current installation.  Cartesian moves are therefore
    subdivided into small straight TCP steps, converted to a continuous local
    full-pose IK solution, checked as a complete six-joint route, and only then
    sent as absolute ``Joint1..Joint6`` positions.
    """

    CARTESIAN_NAMES = ["x", "y", "z", "rx", "ry", "rz"]

    def __init__(
        self,
        config: MotionConfig,
        node,
        log=None,
        *,
        profile: str | None = None,
        speed_store: SpeedLimitStore | None = None,
    ) -> None:
        self.config = config
        self.node = node
        self._log = log or (lambda message: None)
        self.profile = normalize_profile(profile) if profile else None
        self._speed_store = speed_store or SpeedLimitStore()
        self._lock = threading.Lock()
        self._state = ""
        self._end_positions: list[float] = []
        self._joint_positions: list[float] = []
        self._dry_run_joint_positions: list[float] | None = None
        self._fault_flag = 0
        self._stamp = 0.0
        self._feedback_sequence = 0

        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        try:
            from codroid_msgs.msg import RobotInfo
        except ImportError as exc:
            raise RuntimeError(
                "motion.backend=robotmove needs codroid_msgs on AMENT_PREFIX_PATH"
            ) from exc

        self._trajectory_type = JointTrajectory
        self._point_type = JointTrajectoryPoint
        self._publisher = node.create_publisher(
            JointTrajectory, config.robotmove_topic, 10
        )
        node.create_subscription(RobotInfo, config.robotinfo_topic, self._on_info, 10)

    @property
    def dry_run(self) -> bool:
        return bool(self.config.dry_run)

    def _speed_scale(self) -> float:
        # Several tests and compatibility callers construct a backend via
        # ``__new__`` and do not initialize the speed override members.  Such
        # callers retain the historical 100% configuration cap.
        profile = getattr(self, "profile", None)
        store = getattr(self, "_speed_store", None)
        if profile is None or store is None:
            return DEFAULT_SPEED_SCALE
        try:
            # Re-validate at the consumer boundary as well.  A broken custom
            # store must never pass zero or NaN through to a trajectory.
            return normalize_backend_speed_scale(store.scale_for(profile))
        except Exception as exc:
            log = getattr(self, "_log", None)
            if callable(log):
                log(
                    f"[motion:speed-limit] could not read profile "
                    f"{profile}: {exc}; using 100% configuration cap"
                )
            return DEFAULT_SPEED_SCALE

    def _on_info(self, msg) -> None:
        with self._lock:
            self._state = str(msg.state)
            self._end_positions = [float(v) for v in msg.end_positions]
            self._joint_positions = [
                float(v) for v in getattr(msg, "joint_positions", ())
            ]
            self._fault_flag = int(msg.fault_flag)
            self._stamp = time.monotonic()
            self._feedback_sequence = (
                int(getattr(self, "_feedback_sequence", 0)) + 1
            )

    def _advance_dry_run_joints(self, target: Sequence[float]) -> None:
        """Advance the virtual joint state after a non-actuating move.

        ``RobotInfo`` continues to refresh the real controller state while a
        dry-run is active.  Keeping the simulated position in a separate slot
        prevents those feedback messages from rewinding the virtual arm after
        each pre-checked move.
        """

        target_q = np.asarray(target, dtype=float)
        if target_q.shape != (6,) or not np.all(np.isfinite(target_q)):
            raise ValueError(
                "dry-run joint target must contain six finite values"
            )
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._dry_run_joint_positions = [
                float(value) for value in target_q
            ]
            return
        with lock:
            self._dry_run_joint_positions = [
                float(value) for value in target_q
            ]

    def _state_from_feedback(self, *, use_dry_run_override: bool) -> RobotState:
        """Build a state from controller feedback with optional dry-run pose.

        ``get_state`` intentionally exposes the stateful virtual posture while
        a dry-run is active so consecutive motion prechecks behave like a live
        route.  Camera images, however, always come from the physical robot.
        Their detection-to-base transform therefore needs
        ``get_feedback_state``, which never applies the virtual override.
        """

        with self._lock:
            state_name = self._state
            end_positions = tuple(self._end_positions)
            joint_positions = tuple(self._joint_positions)
            simulated_joint_positions = tuple(
                getattr(self, "_dry_run_joint_positions", None) or ()
            )
            fault_flag = self._fault_flag
            stamp = self._stamp
        age = time.monotonic() - stamp if stamp else float("inf")
        pose_values = end_positions[:6]
        joint_values = joint_positions[:6]
        pose_finite = (
            len(pose_values) >= 6
            and all(math.isfinite(float(value)) for value in pose_values)
        )
        joints_finite = (
            len(joint_values) >= 6
            and all(math.isfinite(float(value)) for value in joint_values)
        )
        pose = (
            Pose(tuple(pose_values[:3]), tuple(pose_values[3:6]))
            if pose_finite
            else None
        )
        joints = tuple(joint_values) if joints_finite else None

        # A dry-run is a stateful simulation.  Every mocked move advances the
        # virtual posture so subsequent route prechecks begin where the
        # preceding dry-run move ended, while readiness still comes from live
        # controller feedback and this override can never publish a command.
        simulated_finite = (
            len(simulated_joint_positions) >= 6
            and all(
                math.isfinite(float(value))
                for value in simulated_joint_positions[:6]
            )
        )
        if (
            use_dry_run_override
            and self.dry_run
            and simulated_finite
        ):
            joints = tuple(simulated_joint_positions[:6])
            simulated_transform = controller_fk(joints)
            pose = Pose(
                tuple(float(value) for value in simulated_transform[:3, 3]),
                matrix_to_rpy(simulated_transform[:3, :3]),
            )
            pose_finite = True
            joints_finite = True
        fault = fault_flag != 0
        fresh = age <= self.config.robot_info_timeout_s
        ready = (
            fresh
            and not fault
            and state_name == "AutoReady"
            and pose is not None
            and joints is not None
        )
        if not fresh:
            message = "no fresh RobotInfo"
        elif fault:
            message = f"controller fault_flag={fault_flag}"
        elif not pose_finite or not joints_finite:
            message = (
                "RobotInfo is missing end/joint positions or contains "
                "non-finite values"
            )
        else:
            message = f"controller state={state_name or 'unknown'}"
        return RobotState(
            ready=ready,
            pose=pose,
            fault=fault,
            message=message,
            joints=joints,
            controller_state=state_name,
            fault_flag=fault_flag,
        )

    def get_state(self) -> RobotState:
        """Return the virtual dry-run posture when one has been simulated."""

        return self._state_from_feedback(use_dry_run_override=True)

    def get_feedback_state(self) -> RobotState:
        """Return unmodified ``RobotInfo`` feedback for camera geometry.

        This state must be used to transform an RGB-D frame captured before a
        dry-run-only observation move has physically happened.
        """

        return self._state_from_feedback(use_dry_run_override=False)

    def wait_for_state(self, timeout_s: float) -> RobotState:
        """Wait until a fresh RobotInfo with pose and joints is available."""

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            state = self.get_state()
            fresh_feedback = state.message != "no fresh RobotInfo"
            if (
                fresh_feedback
                and state.pose is not None
                and state.joints is not None
            ):
                return state
            if time.monotonic() >= deadline:
                return state
            time.sleep(min(self.config.poll_period_s, 0.05))

    def get_feedback_sequence(self) -> int:
        """Return the monotonic count of received physical RobotInfo samples."""

        with self._lock:
            return int(getattr(self, "_feedback_sequence", 0))

    def wait_for_feedback_state(
        self,
        timeout_s: float,
        *,
        newer_than: int | None = None,
    ) -> RobotState:
        """Wait for fresh physical feedback, optionally newer than a sample.

        ``get_state()`` may expose a virtual dry-run posture, so capture-time
        safety checks must use this method instead.  A capture can block the
        pipeline thread for several seconds while the first YOLO model is
        loaded; requiring ``newer_than`` ensures the post-capture pose is
        actually received after that blocking work rather than merely
        re-reading the last pre-capture sample.
        """

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        minimum_sequence = (
            None if newer_than is None else int(newer_than)
        )
        while True:
            # Read the sequence before the state.  This orders the lock
            # acquisitions so a returned state cannot predate a sample that
            # already satisfies ``newer_than``.
            with self._lock:
                sequence = int(getattr(self, "_feedback_sequence", 0))
            state = self._state_from_feedback(use_dry_run_override=False)
            fresh_feedback = state.message != "no fresh RobotInfo"
            has_new_feedback = (
                minimum_sequence is None or sequence > minimum_sequence
            )
            if (
                fresh_feedback
                and has_new_feedback
                and state.pose is not None
                and state.joints is not None
            ):
                return state
            if time.monotonic() >= deadline:
                return state
            time.sleep(min(self.config.poll_period_s, 0.05))

    def move_to(self, pose: Pose, *, name: str, speed_mps: float) -> None:
        state = self.wait_for_state(1.0)
        if self.config.motion_mode == "cartesian":
            if self.dry_run:
                effective_speed = bounded_cartesian_speed_mps(
                    speed_mps,
                    self.config.speed_mps,
                    self._speed_scale(),
                )
                self._log(
                    f"[motion:robotmove:dry-run-cartesian] {name}: "
                    f"xyz={pose.xyz} rpy={pose.rpy} v={effective_speed:.3f}"
                )
                return
            if not state.ready:
                raise RuntimeError(
                    f"robot not ready before move {name!r}: {state.message}"
                )
            if state.joints is None or state.pose is None:
                raise RuntimeError(
                    f"robot state incomplete before move {name!r}"
                )
            self._send_cartesian(pose, name=name, speed_mps=speed_mps)
            return

        # Default path: subdivide the straight Cartesian segment, solve a
        # continuous local IK solution at every waypoint, and validate the
        # entire joint trajectory before publishing the first command.
        route = self.precheck_linear_targets(
            (pose,),
            step_m=self.config.cartesian_step_m,
        )
        self.execute_linear_route(
            route,
            names=(name,),
            speed_mps=speed_mps,
        )

    def _validate_joint_pose(
        self,
        joints: Sequence[float],
        pose: Pose,
        *,
        label: str,
        position_tolerance_m: float | None = None,
        orientation_tolerance_rad: float | None = None,
        enforce_singularity: bool = True,
    ) -> float:
        """Validate one IK solution against limits, pose and manipulability."""

        q = np.asarray(joints, dtype=float)
        margin_rad = math.radians(float(self.config.joint_limit_margin_deg))
        if not joints_within_limits(q, margin_rad):
            raise _CartesianPathError(
                f"IK solution for {label} violates the configured joint-limit "
                f"margin ({self.config.joint_limit_margin_deg:g} deg)",
                retryable=False,
            )

        solved = controller_fk(q)
        solved_position = solved[:3, 3]
        target_position = np.asarray(pose.xyz, dtype=float)
        target_rotation = rpy_to_matrix(pose.rpy)
        position_error = float(np.linalg.norm(solved_position - target_position))
        orientation_error = rotation_angle_error(
            solved[:3, :3], target_rotation
        )
        allowed_position = (
            float(self.config.ik_position_tolerance_m)
            if position_tolerance_m is None
            else float(position_tolerance_m)
        )
        allowed_orientation = (
            float(self.config.ik_orientation_tolerance_rad)
            if orientation_tolerance_rad is None
            else float(orientation_tolerance_rad)
        )
        if position_error > allowed_position:
            raise _CartesianPathError(
                f"IK position error for {label} is "
                f"{position_error * 1000:.2f} mm "
                f"(limit {allowed_position * 1000:.2f} mm)",
                retryable=False,
            )
        if orientation_error > allowed_orientation:
            raise _CartesianPathError(
                f"IK orientation error for {label} is "
                f"{orientation_error:.5f} rad "
                f"(limit {allowed_orientation:.5f} rad)",
                retryable=False,
            )

        singular_values = weighted_jacobian_singular_values(q)
        if singular_values.size != 6 or not np.all(np.isfinite(singular_values)):
            raise _CartesianPathError(
                f"could not evaluate the Jacobian for {label}",
                retryable=False,
            )
        minimum = float(np.min(singular_values))
        if (
            enforce_singularity
            and minimum
            < float(self.config.cartesian_min_jacobian_singular_value)
        ):
            raise _CartesianPathError(
                f"near-singular posture at {label}: minimum weighted Jacobian "
                f"singular value={minimum:.6f} < "
                f"{self.config.cartesian_min_jacobian_singular_value:.6f}",
                retryable=False,
            )
        return minimum

    def _solve_checked_waypoint(
        self,
        q_initial: np.ndarray,
        waypoint: Pose,
        *,
        label: str,
    ) -> tuple[np.ndarray, float]:
        validated_candidate: np.ndarray | None = None
        validated_singular: float | None = None
        rejection_reasons: list[str] = []
        initial_joints = np.asarray(q_initial, dtype=float)
        relative_seeds_rad = tuple(
            initial_joints
            + np.radians(np.asarray(offset, dtype=float))
            for offset in self.config.ik_branch_seed_offsets_deg
        )

        def candidate_is_safe(candidate: np.ndarray) -> bool:
            nonlocal validated_candidate, validated_singular
            try:
                singular = self._validate_joint_pose(
                    candidate,
                    waypoint,
                    label=label,
                )
            except _CartesianPathError as exc:
                rejection_reasons.append(str(exc))
                return False
            validated_candidate = np.asarray(candidate, dtype=float).copy()
            validated_singular = float(singular)
            return True

        try:
            ik_solution = solve_pose_ik_multistart(
                q_initial,
                waypoint.xyz,
                rpy_to_matrix(waypoint.rpy),
                # IK acceptance is intentionally tighter than the controller
                # feedback tolerance.  The pre-check must prove the generated
                # joint target itself hits the Cartesian waypoint.
                position_tolerance_m=float(
                    self.config.ik_position_tolerance_m
                ),
                rotation_tolerance_rad=float(
                    self.config.ik_orientation_tolerance_rad
                ),
                accept_delta_rad=math.radians(
                    float(self.config.cartesian_joint_step_deg)
                ),
                candidate_validator=candidate_is_safe,
                # Operator-verified successful postures are numerical seeds
                # only.  They never become trajectory waypoints and cannot
                # bypass the step, tolerance, limit or singularity checks.
                fallback_seeds_rad=(
                    tuple(
                        np.radians(np.asarray(seed, dtype=float))
                        for seed in self.config.ik_fallback_seeds_deg
                    )
                    + relative_seeds_rad
                ),
                minimum_singular_value=float(
                    self.config.cartesian_min_jacobian_singular_value
                ),
                joint_limit_margin_rad=math.radians(
                    float(self.config.joint_limit_margin_deg)
                ),
                # A coarse endpoint is needed to decide how to subdivide a
                # Cartesian segment.  It is revalidated below before use.
                allow_unbounded_fallback=True,
                # A smooth Cartesian path does not need the numerically
                # smallest joint delta once a candidate has passed every
                # safety gate.  Stopping at the first safe local solution cuts
                # the typical 7-damping/12-seed search to the minimum needed.
                stop_after_first_safe=True,
                return_diagnostics=True,
            )
            solved_q = np.asarray(ik_solution.joints, dtype=float)
            if int(ik_solution.seed_index) > 0:
                self._log(
                    "[motion:ik-restart-selected] "
                    f"{label}: seed={ik_solution.seed_index}, "
                    f"source={ik_solution.seed_source}, "
                    f"damping={ik_solution.damping:g}, "
                    f"iterations={ik_solution.iterations}, "
                    f"step={math.degrees(ik_solution.max_joint_delta_rad):.3f} deg, "
                    f"pos_err={ik_solution.position_error_m * 1000:.4f} mm, "
                    f"rot_err={ik_solution.orientation_error_rad:.6f} rad, "
                    f"min_sigma="
                    f"{ik_solution.minimum_jacobian_singular_value:.6f}"
                )
        except RuntimeError as exc:
            detail = str(exc)
            if rejection_reasons:
                detail = f"{detail}; last rejection: {rejection_reasons[-1]}"
            raise _CartesianPathError(
                f"IK failed for {label}: {detail}",
                retryable=True,
            ) from exc

        if (
            validated_candidate is not None
            and np.array_equal(solved_q, validated_candidate)
            and validated_singular is not None
        ):
            return solved_q, validated_singular

        # A candidate above the per-step bound is returned so the caller can
        # refine the Cartesian segment first.  Do not reject that coarse
        # endpoint merely because its temporary posture is near a singularity.
        largest_delta = float(np.max(np.abs(solved_q - q_initial)))
        enforce_singularity = (
            largest_delta
            <= math.radians(float(self.config.cartesian_joint_step_deg)) + 1e-12
        )
        minimum = self._validate_joint_pose(
            solved_q,
            waypoint,
            label=label,
            enforce_singularity=enforce_singularity,
        )
        return solved_q, minimum

    @staticmethod
    def _joint_path_metrics(
        joint_waypoints: Sequence[Sequence[float]],
    ) -> JointPathMetrics:
        """Return per-axis step, acceleration and cumulative travel metrics."""

        path = np.asarray(joint_waypoints, dtype=float)
        if path.size == 0:
            zeros = (0.0,) * len(JOINT_NAMES)
            return JointPathMetrics(zeros, zeros, zeros)
        if path.ndim != 2 or path.shape[1] != len(JOINT_NAMES):
            raise ValueError("joint path must contain six values per waypoint")
        if not np.all(np.isfinite(path)):
            raise ValueError("joint path contains non-finite values")
        if len(path) < 2:
            zeros = (0.0,) * len(JOINT_NAMES)
            return JointPathMetrics(zeros, zeros, zeros)

        delta_deg = np.degrees(np.diff(path, axis=0))
        max_step_deg = tuple(
            float(value) for value in np.max(np.abs(delta_deg), axis=0)
        )
        travel_deg = tuple(
            float(value) for value in np.sum(np.abs(delta_deg), axis=0)
        )
        if len(delta_deg) < 2:
            zeros = (0.0,) * len(JOINT_NAMES)
            return JointPathMetrics(max_step_deg, zeros, travel_deg)

        acceleration_deg = np.diff(delta_deg, axis=0)
        max_acceleration_deg = tuple(
            float(value)
            for value in np.max(np.abs(acceleration_deg), axis=0)
        )
        return JointPathMetrics(
            max_step_deg=max_step_deg,
            max_acceleration_deg=max_acceleration_deg,
            travel_deg=travel_deg,
        )

    @staticmethod
    def _axis_metric_list(
        segment: LinearSegmentPlan,
        field_name: str,
    ) -> str:
        """Format one per-axis metric as comma-separated Joint1..Joint6 data."""

        if segment.joint_metrics is None:
            return "-"
        if field_name not in {
            "max_step_deg",
            "max_acceleration_deg",
            "travel_deg",
        }:
            raise ValueError(f"unknown joint path metric {field_name!r}")
        values = getattr(segment.joint_metrics, field_name)
        if len(values) != len(JOINT_NAMES):
            raise ValueError(f"joint path metric {field_name!r} is not six-axis")
        return ",".join(f"{float(value):.3f}" for value in values)

    @staticmethod
    def _reverse_prechecked_segment(
        segment: LinearSegmentPlan,
    ) -> LinearSegmentPlan:
        """Return the same validated joint path traversed end-to-start."""

        return LinearSegmentPlan(
            target=segment.waypoints[0],
            waypoints=tuple(reversed(segment.waypoints)),
            joint_waypoints=tuple(reversed(segment.joint_waypoints)),
            planning_step_m=segment.planning_step_m,
            max_joint_step_deg=segment.max_joint_step_deg,
            max_joint_acceleration_deg=segment.max_joint_acceleration_deg,
            min_jacobian_singular_value=(
                segment.min_jacobian_singular_value
            ),
            joint_metrics=segment.joint_metrics,
        )

    def _plan_cartesian_segment(
        self,
        start_pose: Pose,
        start_q: np.ndarray,
        target: Pose,
        *,
        name: str,
        step_m: float,
    ) -> LinearSegmentPlan:
        """Plan a continuous IK path and reject every unsafe waypoint.

        The requested spacing is only an upper bound.  If one Cartesian step
        needs too much joint motion, or the discrete acceleration between
        neighbouring IK solutions grows, the route is refined and re-solved
        from the same start posture before any command is published.
        """

        plan_started = time.perf_counter()
        planning_attempts = 0
        requested_step = float(step_m)
        if not math.isfinite(requested_step) or requested_step <= 0.0:
            raise ValueError("Cartesian planning step must be positive and finite")
        _validate_exact_downward_reach(target, label=f"{name} target endpoint")
        planning_step = min(requested_step, float(self.config.cartesian_step_m))
        minimum_step = min(float(self.config.cartesian_min_step_m), planning_step)
        planning_orientation_step = math.radians(
            float(self.config.cartesian_orientation_step_deg)
        )
        minimum_orientation_step = math.radians(
            float(self.config.cartesian_min_orientation_step_deg)
        )
        joint_step_limit = math.radians(
            float(self.config.cartesian_joint_step_deg)
        )
        branch_jump_limit = math.radians(
            float(self.config.cartesian_joint_branch_jump_deg)
        )
        acceleration_limit = float(
            self.config.cartesian_joint_acceleration_limit_deg
        )

        if (
            _pose_distance(start_pose, target) < 1e-12
            and rotation_angle_error(
                rpy_to_matrix(start_pose.rpy),
                rpy_to_matrix(target.rpy),
            )
            < 1e-12
        ):
            minimum_singular = self._validate_joint_pose(
                start_q,
                start_pose,
                label=f"{name} unchanged posture",
            )
            joint_waypoints = (tuple(float(value) for value in start_q),)
            return LinearSegmentPlan(
                target=target,
                waypoints=(start_pose,),
                joint_waypoints=joint_waypoints,
                planning_step_m=planning_step,
                max_joint_step_deg=0.0,
                max_joint_acceleration_deg=0.0,
                min_jacobian_singular_value=minimum_singular,
                joint_metrics=self._joint_path_metrics(joint_waypoints),
            )

        def plan_once(
            coarse_step: float,
            coarse_orientation_step: float,
        ) -> tuple[
            tuple[Pose, ...],
            tuple[tuple[float, ...], ...],
            float,
        ]:
            base_waypoints = linear_pose_waypoints(
                start_pose,
                target,
                step_m=coarse_step,
                orientation_step_rad=coarse_orientation_step,
            )
            max_block_waypoints = int(self.config.cartesian_max_waypoints)
            max_total_waypoints = int(
                self.config.cartesian_max_total_waypoints
            )
            if len(base_waypoints) > max_total_waypoints:
                raise _CartesianPathError(
                    f"Cartesian plan for {name!r} needs {len(base_waypoints)} "
                    "base waypoints, above the configured total maximum "
                    f"{max_total_waypoints}",
                    retryable=False,
                )

            # A logical straight segment can be longer than the per-block
            # refinement budget (for example, a large wrist yaw).  Build
            # overlapping blocks from the exact same Cartesian path.  The
            # shared endpoint is already checked at the end of one block and
            # becomes the IK seed for the next, so no requested pose is
            # skipped while each block remains independently bounded.
            block_span = max_block_waypoints - 1
            blocks: list[tuple[Pose, ...]] = []
            block_start = 0
            while block_start < len(base_waypoints) - 1:
                block_end = min(
                    block_start + block_span,
                    len(base_waypoints) - 1,
                )
                blocks.append(base_waypoints[block_start : block_end + 1])
                block_start = block_end
            if len(blocks) > 1:
                self._log(
                    f"[motion:cartesian-block-split] {name}: "
                    f"{len(base_waypoints) - 1} base step(s) in "
                    f"{len(blocks)} block(s), max_block_waypoints="
                    f"{max_block_waypoints}, total_limit="
                    f"{max_total_waypoints}"
                )

            poses: list[Pose] = [start_pose]
            joints: list[tuple[float, ...]] = [
                tuple(float(value) for value in start_q)
            ]
            minimum_singular = float("inf")

            def append_waypoint(waypoint: Pose, joint_waypoint: np.ndarray) -> None:
                if len(poses) >= max_total_waypoints:
                    raise _CartesianPathError(
                        f"refined Cartesian plan for {name!r} exceeds the "
                        "configured total maximum of "
                        f"{max_total_waypoints} waypoints",
                        retryable=False,
                    )
                poses.append(waypoint)
                joints.append(tuple(float(value) for value in joint_waypoint))

            for block_index, block in enumerate(blocks, start=1):
                block_start_pose = block[0]
                if block_start_pose != poses[-1]:
                    raise _CartesianPathError(
                        f"internal Cartesian block {block_index} for {name!r} "
                        "does not start at the previous block endpoint",
                        retryable=False,
                    )
                block_start_q = np.asarray(joints[-1], dtype=float)
                local_poses: list[Pose] = [block_start_pose]
                local_joints: list[tuple[float, ...]] = [
                    tuple(float(value) for value in block_start_q)
                ]
                local_minimum_singular = float("inf")

                def append_local_waypoint(
                    waypoint: Pose,
                    joint_waypoint: np.ndarray,
                ) -> None:
                    if len(local_poses) >= max_block_waypoints:
                        raise _CartesianPathError(
                            f"refined Cartesian block {block_index} for "
                            f"{name!r} exceeds {max_block_waypoints} "
                            "waypoints; retry with a finer base spacing",
                            retryable=True,
                        )
                    local_poses.append(waypoint)
                    local_joints.append(
                        tuple(float(value) for value in joint_waypoint)
                    )

                def visit(
                    segment_start_pose: Pose,
                    segment_start_q: np.ndarray,
                    segment_end_pose: Pose,
                    *,
                    depth: int,
                ) -> None:
                    nonlocal local_minimum_singular
                    label = f"{name} path waypoint"
                    _validate_exact_downward_reach(
                        segment_end_pose,
                        label=label,
                    )
                    _validate_workspace(self.config, segment_end_pose, label)
                    end_q, singular = self._solve_checked_waypoint(
                        segment_start_q,
                        segment_end_pose,
                        label=label,
                    )
                    largest_delta = float(
                        np.max(np.abs(end_q - segment_start_q))
                    )
                    if largest_delta <= joint_step_limit + 1e-12:
                        append_local_waypoint(segment_end_pose, end_q)
                        local_minimum_singular = min(
                            local_minimum_singular,
                            singular,
                        )
                        return

                    distance = _pose_distance(
                        segment_start_pose,
                        segment_end_pose,
                    )
                    orientation_delta = rotation_angle_error(
                        rpy_to_matrix(segment_start_pose.rpy),
                        rpy_to_matrix(segment_end_pose.rpy),
                    )
                    can_refine = (
                        distance > minimum_step + 1e-12
                        or orientation_delta
                        > minimum_orientation_step + 1e-12
                    )
                    if not can_refine:
                        if largest_delta > branch_jump_limit:
                            reason = (
                                "possible IK branch jump: one minimum-step "
                                f"segment needs {math.degrees(largest_delta):.3f} "
                                "deg on one joint"
                            )
                        else:
                            reason = (
                                "joint-step limit cannot be satisfied at the "
                                "minimum spacing "
                                f"({math.degrees(largest_delta):.3f} deg > "
                                f"{math.degrees(joint_step_limit):.3f} deg)"
                            )
                        raise _CartesianPathError(
                            f"{reason} at {label}, depth={depth}",
                            retryable=False,
                        )

                    midpoint = _pose_at_fraction(
                        segment_start_pose,
                        segment_end_pose,
                        0.5,
                    )
                    if (
                        _pose_distance(segment_start_pose, midpoint) < 1e-12
                        and rotation_angle_error(
                            rpy_to_matrix(segment_start_pose.rpy),
                            rpy_to_matrix(midpoint.rpy),
                        )
                        < 1e-12
                    ):
                        raise _CartesianPathError(
                            f"Cartesian refinement made no progress at {label}",
                            retryable=False,
                        )
                    visit(
                        segment_start_pose,
                        segment_start_q,
                        midpoint,
                        depth=depth + 1,
                    )
                    # The midpoint is already the final pose of the left
                    # branch; solve the right branch from that exact posture.
                    midpoint_q = np.asarray(local_joints[-1], dtype=float)
                    visit(
                        midpoint,
                        midpoint_q,
                        segment_end_pose,
                        depth=depth + 1,
                    )

                for waypoint in block[1:]:
                    visit(
                        local_poses[-1],
                        np.asarray(local_joints[-1], dtype=float),
                        waypoint,
                        depth=1,
                    )

                if len(poses) + len(local_poses) - 1 > max_total_waypoints:
                    raise _CartesianPathError(
                        f"refined Cartesian plan for {name!r} exceeds the "
                        "configured total maximum of "
                        f"{max_total_waypoints} waypoints",
                        retryable=False,
                    )
                for waypoint, joint_waypoint in zip(
                    local_poses[1:],
                    local_joints[1:],
                ):
                    append_waypoint(waypoint, np.asarray(joint_waypoint))
                minimum_singular = min(
                    minimum_singular,
                    local_minimum_singular,
                )

            return tuple(poses), tuple(joints), minimum_singular

        while True:
            planning_attempts += 1
            try:
                waypoints, joint_waypoints, minimum_singular = plan_once(
                    planning_step,
                    planning_orientation_step,
                )
            except _CartesianPathError as exc:
                next_step = max(minimum_step, planning_step * 0.5)
                next_orientation = max(
                    minimum_orientation_step,
                    planning_orientation_step * 0.5,
                )
                can_retry = (
                    exc.retryable
                    and (
                        next_step < planning_step - 1e-15
                        or next_orientation
                        < planning_orientation_step - 1e-15
                    )
                )
                if can_retry:
                    planning_step = next_step
                    planning_orientation_step = next_orientation
                    continue
                raise RuntimeError(str(exc)) from exc

            joint_metrics = self._joint_path_metrics(joint_waypoints)
            max_step = max(joint_metrics.max_step_deg, default=0.0)
            max_acceleration = max(
                joint_metrics.max_acceleration_deg,
                default=0.0,
            )
            if max_step > math.degrees(joint_step_limit) + 1e-9:
                raise RuntimeError(
                    f"internal joint-step check failed for {name!r}: "
                    f"{max_step:.3f} deg"
                )
            if max_acceleration <= acceleration_limit + 1e-9:
                self._log(
                    f"[motion:plan-timing] {name}: "
                    f"total_s={time.perf_counter() - plan_started:.3f}, "
                    f"attempts={planning_attempts}, "
                    f"waypoints={max(0, len(waypoints) - 1)}, "
                    f"planning_step_mm={planning_step * 1000.0:.3f}, "
                    f"max_joint_step={max_step:.3f} deg, "
                    f"max_joint_accel={max_acceleration:.3f} deg, "
                    f"min_sigma={minimum_singular:.6f}"
                )
                return LinearSegmentPlan(
                    target=target,
                    waypoints=waypoints,
                    joint_waypoints=joint_waypoints,
                    planning_step_m=planning_step,
                    max_joint_step_deg=max_step,
                    max_joint_acceleration_deg=max_acceleration,
                    min_jacobian_singular_value=minimum_singular,
                    joint_metrics=joint_metrics,
                )

            next_step = max(minimum_step, planning_step * 0.5)
            next_orientation = max(
                minimum_orientation_step,
                planning_orientation_step * 0.5,
            )
            if (
                next_step >= planning_step - 1e-15
                and next_orientation >= planning_orientation_step - 1e-15
            ):
                raise RuntimeError(
                    f"joint acceleration guard failed for {name!r}: "
                    f"{max_acceleration:.3f} deg per waypoint > "
                    f"{acceleration_limit:.3f} deg even at minimum spacing"
                )
            planning_step = next_step
            planning_orientation_step = next_orientation

    def precheck_pose_condition(self, pose: Pose, *, label: str) -> float:
        """Cheaply reject an unreachable or ill-conditioned candidate endpoint.

        This is an early candidate filter, not a replacement for the complete
        six-segment route check.  It solves the first high waypoint from the
        current measured posture and returns its Jacobian margin.  A candidate
        that fails here cannot provide a safe continuous route and would
        otherwise spend tens of seconds refining a singular Cartesian path.
        """

        state = self.wait_for_state(1.0)
        if state.fault:
            raise RuntimeError(f"cannot screen candidate endpoint: {state.message}")
        if not state.ready or state.pose is None or state.joints is None:
            raise RuntimeError(
                "cannot screen candidate endpoint without a ready robot state: "
                f"{state.message}"
            )
        _validate_exact_downward_reach(
            pose,
            label=f"{label} candidate endpoint",
        )
        _validate_workspace(self.config, pose, f"{label} candidate endpoint")
        _solved_q, minimum_sigma = self._solve_checked_waypoint(
            np.asarray(state.joints, dtype=float),
            pose,
            label=f"{label} candidate endpoint",
        )
        return float(minimum_sigma)

    def precheck_linear_targets(
        self, targets: Sequence[Pose], *, step_m: float
    ) -> LinearRoutePlan:
        """Resolve and validate every straight-path waypoint without moving."""

        state = self.wait_for_state(1.0)
        if state.fault:
            raise RuntimeError(f"cannot pre-check path: {state.message}")
        if not state.ready or state.pose is None or state.joints is None:
            raise RuntimeError(
                "cannot pre-check path without a ready robot state: "
                f"{state.message}"
            )

        route_started = time.perf_counter()
        start_pose = state.pose
        current_pose = start_pose
        current_q = np.asarray(state.joints, dtype=float)
        _validate_workspace(self.config, start_pose, "start pose")
        start_singular = self._validate_joint_pose(
            current_q,
            start_pose,
            label="start posture",
            # Five recorded /RobotInfo postures match controller FK within
            # 0.001 mm and 0.001 mrad.  A larger feedback/model disagreement
            # must therefore stop route planning before any waypoint is sent.
            position_tolerance_m=float(self.config.ik_position_tolerance_m),
            orientation_tolerance_rad=float(
                self.config.ik_orientation_tolerance_rad
            ),
        )

        # The Task B route descends and lifts along the same grasp column more
        # than once.  An already validated path can be reused only when both
        # Cartesian endpoints and both endpoint joint postures match exactly.
        # A reversed path contains the same poses, IK solutions, joint steps,
        # acceleration deltas and singularity measurements in reverse order,
        # so no safety check is weakened by reusing it.
        validated_paths: dict[
            tuple[Pose, tuple[float, ...], Pose], LinearSegmentPlan
        ] = {}
        segments: list[LinearSegmentPlan] = []
        for target_index, target in enumerate(targets, start=1):
            name = f"target {target_index}"
            _validate_exact_downward_reach(
                target,
                label=f"{name} target endpoint",
            )
            current_joints = tuple(float(value) for value in current_q)
            path_key = (current_pose, current_joints, target)
            segment = validated_paths.get(path_key)
            if segment is None:
                segment = self._plan_cartesian_segment(
                    current_pose,
                    current_q,
                    target,
                    name=name,
                    step_m=step_m,
                )
                end_joints = tuple(
                    float(value) for value in segment.joint_waypoints[-1]
                )
                validated_paths[path_key] = segment
                validated_paths[(target, end_joints, current_pose)] = (
                    self._reverse_prechecked_segment(segment)
                )
            else:
                self._log(
                    f"[motion:cartesian-precheck-reuse] {name}: "
                    "exact endpoint and joint posture match"
                )
            segments.append(segment)
            current_pose = target
            current_q = np.asarray(segment.joint_waypoints[-1], dtype=float)
            self._log(
                f"[motion:cartesian-precheck] {name}: "
                f"{len(segment.waypoints) - 1} waypoint(s), "
                f"planning_step={segment.planning_step_m * 1000:.3f} mm, "
                f"max_joint_step={segment.max_joint_step_deg:.3f} deg, "
                f"max_joint_accel={segment.max_joint_acceleration_deg:.3f} deg, "
                f"min_sigma={segment.min_jacobian_singular_value:.5f}, "
                f"axis_step_deg={self._axis_metric_list(segment, 'max_step_deg')}, "
                f"axis_travel_deg={self._axis_metric_list(segment, 'travel_deg')}"
            )

        minimum_singular = min(
            [start_singular]
            + [segment.min_jacobian_singular_value for segment in segments]
        )
        self._log(
            f"[motion:linear-precheck] validated {len(segments)} segment(s), "
            f"{sum(max(0, len(item.waypoints) - 1) for item in segments)} "
            f"Cartesian waypoint(s), min_sigma={minimum_singular:.5f}, "
            f"total_s={time.perf_counter() - route_started:.3f}"
        )
        return LinearRoutePlan(start_pose=start_pose, segments=tuple(segments))

    def precheck_joint_route(
        self,
        targets: Sequence[Sequence[float]],
        *,
        allowed_joints: Sequence[int],
        max_initial_move_deg: float,
        max_target_move_deg: float,
        max_step_deg: float | None = None,
        min_endpoint_distance_m: float,
        name_prefix: str = "task-a-patrol",
    ) -> JointRoutePlan:
        """Validate all Task A joint targets before publishing any command."""

        state = self.wait_for_state(float(self.config.robot_info_timeout_s))
        return _build_joint_route_plan(
            self.config,
            state,
            targets,
            allowed_joints=allowed_joints,
            max_initial_move_deg=max_initial_move_deg,
            max_target_move_deg=max_target_move_deg,
            max_step_deg=(
                self.config.joint_max_step_deg
                if max_step_deg is None
                else max_step_deg
            ),
            min_endpoint_distance_m=min_endpoint_distance_m,
            name_prefix=name_prefix,
        )

    def execute_joint_route_step(
        self,
        route: JointRoutePlan,
        *,
        index: int,
        name: str,
    ) -> None:
        """Execute one point from a fully pre-checked Task A joint route.

        The vision pipeline calls this method once per position so the camera
        can observe while the arm is stopped.  J5 and J6 are never accepted as
        movable joints, and every candidate target has already been checked by
        :meth:`precheck_joint_route` before this method can publish anything.
        """

        if index < 0 or index >= len(route.targets):
            raise IndexError(
                f"joint route step index {index} is outside "
                f"0..{len(route.targets) - 1}"
            )
        movable = tuple(
            joint_index
            for joint_index in range(6)
            if joint_index not in route.fixed_indices
        )
        if not movable or not set(movable).issubset(
            TASK_A_MOVABLE_JOINT_INDICES
        ):
            raise RuntimeError(
                f"refusing joint route {name!r}: Task A may move only "
                "Joint1/Joint2/Joint3/Joint4; "
                f"movable_indices={movable}"
            )

        target = route.targets[index]
        expected_start = (
            route.start_joints if index == 0 else route.targets[index - 1]
        )
        state = self.wait_for_state(float(self.config.robot_info_timeout_s))
        if not state.ready or state.joints is None or state.pose is None:
            raise RuntimeError(
                f"robot is not ready before joint patrol {name!r}: "
                f"{_robot_state_detail(state)}"
            )

        fixed_drift_deg = _joint_error_deg(
            state.joints,
            route.start_joints,
            indices=route.fixed_indices,
        )
        fixed_limit_deg = max(
            0.10,
            2.0 * float(self.config.joint_start_tolerance_deg),
        )
        if fixed_drift_deg > fixed_limit_deg + 1e-9:
            raise RuntimeError(
                f"refusing joint route {name!r}: locked-joint drift "
                f"{fixed_drift_deg:.3f} deg exceeds {fixed_limit_deg:.3f} deg; "
                f"{_robot_state_detail(state)}"
            )

        if not self.dry_run:
            start_error_deg = _joint_error_deg(
                state.joints,
                expected_start,
                indices=movable,
            )
            start_limit_deg = max(
                0.25,
                2.0 * float(self.config.joint_tolerance_deg),
            )
            if start_error_deg > start_limit_deg + 1e-9:
                raise RuntimeError(
                    f"pre-checked joint route {name!r} is stale at step "
                    f"{index + 1}/{len(route.targets)}: actual movable joints "
                    f"are {start_error_deg:.3f} deg from the previous target "
                    f"(limit {start_limit_deg:.3f} deg); "
                    f"{_robot_state_detail(state)}"
                )

        try:
            actual = self.move_joints(
                target,
                name=name,
                max_step_deg=float(self.config.joint_max_step_deg),
            )
        except Exception as exc:
            failure = self.get_state()
            raise RuntimeError(
                f"joint patrol command {name!r} failed at step "
                f"{index + 1}/{len(route.targets)}; target_deg="
                f"{tuple(round(math.degrees(value), 3) for value in target)}; "
                f"{_robot_state_detail(failure)}; cause={exc}"
            ) from exc

        if not self.dry_run:
            actual_array = np.asarray(actual, dtype=float)
            fixed_after_deg = _joint_error_deg(
                actual_array,
                route.start_joints,
                indices=route.fixed_indices,
            )
            if fixed_after_deg > fixed_limit_deg + 1e-9:
                raise RuntimeError(
                    f"joint patrol {name!r} changed a locked joint by "
                    f"{fixed_after_deg:.3f} deg (limit {fixed_limit_deg:.3f} deg)"
                )

    def execute_joint_route(
        self,
        route: JointRoutePlan,
        *,
        names: Sequence[str],
    ) -> None:
        """Execute a full pre-checked route, one observable point at a time."""

        if len(names) != len(route.targets):
            raise ValueError("one name is required for every joint target")
        for index, name in enumerate(names):
            self.execute_joint_route_step(route, index=index, name=str(name))

    def execute_linear_route(
        self, route: LinearRoutePlan, *, names: Sequence[str], speed_mps: float
    ) -> None:
        if len(names) != len(route.segments):
            raise ValueError("one name is required for every linear segment")

        for name, segment in zip(names, route.segments):
            self.execute_prechecked_segment(
                segment,
                name=name,
                speed_mps=speed_mps,
            )

    def execute_prechecked_segment(
        self,
        segment: LinearSegmentPlan,
        *,
        name: str,
        speed_mps: float,
    ) -> None:
        """Execute one unchanged segment from a route that passed precheck."""

        if len(segment.joint_waypoints) != len(segment.waypoints):
            raise RuntimeError(f"linear route {name!r} has no joint plan")
        speed_scale = self._speed_scale()
        effective_speed_mps = bounded_cartesian_speed_mps(
            speed_mps,
            self.config.speed_mps,
            speed_scale,
        )
        velocity_rad_s = self._cartesian_joint_velocity_rad_s(
            segment,
            # Apply the operator scale exactly once.  The helper computes the
            # effective Cartesian speed from the requested speed and scale;
            # passing the already-scaled value here would slow the joint
            # trajectory twice at 10--99% settings.
            speed_mps=speed_mps,
            speed_scale=speed_scale,
        )

        if self.dry_run:
            self._log(
                f"[motion:robotmove:dry-run-linear] {name}: "
                f"distance={segment.distance_m:.3f} m "
                f"waypoints={max(0, len(segment.waypoints) - 1)} "
                f"target_xyz={segment.target.xyz} "
                f"v={effective_speed_mps:.3f} "
                f"speed_scale={speed_scale:.2f} "
                f"joint_velocity={velocity_rad_s:.3f} rad/s "
                f"max_joint_step={segment.max_joint_step_deg:.3f} deg "
                f"max_joint_accel={segment.max_joint_acceleration_deg:.3f} deg "
                f"min_sigma={segment.min_jacobian_singular_value:.5f} "
                f"axis_step_deg={self._axis_metric_list(segment, 'max_step_deg')} "
                f"axis_travel_deg={self._axis_metric_list(segment, 'travel_deg')}"
            )
            if segment.joint_waypoints:
                self._advance_dry_run_joints(segment.joint_waypoints[-1])
            return

        state = self.wait_for_state(1.0)
        if not state.ready or state.joints is None:
            raise RuntimeError(
                f"robot not ready before linear route {name!r}: "
                f"{state.message}"
            )
        actual_q = np.asarray(state.joints, dtype=float)
        expected_start = np.asarray(segment.joint_waypoints[0], dtype=float)
        start_error_deg = math.degrees(
            float(np.max(np.abs(actual_q - expected_start)))
        )
        allowed_start_error = max(
            0.10,
            2.0 * float(self.config.joint_start_tolerance_deg),
        )
        if start_error_deg > allowed_start_error:
            raise RuntimeError(
                f"pre-checked route {name!r} is stale: robot moved "
                f"{start_error_deg:.3f} deg from the planned start "
                f"(limit {allowed_start_error:.3f} deg)"
            )

        original_commanded_waypoints = segment.joint_waypoints[1:]
        if original_commanded_waypoints:
            first_target = np.asarray(original_commanded_waypoints[0], dtype=float)
            commanded_delta_deg = math.degrees(
                float(np.max(np.abs(first_target - actual_q)))
            )
            allowed_delta = (
                float(self.config.cartesian_joint_step_deg)
                + float(self.config.joint_tolerance_deg)
            )
            if commanded_delta_deg > allowed_delta + 1e-9:
                raise RuntimeError(
                    f"refusing {name!r} first waypoint: actual joint feedback "
                    f"would require {commanded_delta_deg:.3f} deg in one step "
                    f"(limit {allowed_delta:.3f} deg)"
                )
        commanded_waypoints = self._compress_joint_trajectory(
            original_commanded_waypoints,
            name=name,
        )
        if commanded_waypoints.size:
            # Send the complete IK path in one JointTrajectory.  The controller
            # can then plan continuously through every point instead of the arm
            # stopping and waiting for AutoReady after each 2 mm increment.
            actual_q = self._send_joint_trajectory(
                commanded_waypoints,
                name=name,
                velocity_rad_s=velocity_rad_s,
                timeout_s=self._trajectory_timeout_s(
                    len(commanded_waypoints),
                    joint_travel_rad=_joint_path_travel_rad(
                        actual_q,
                        commanded_waypoints,
                    ),
                    velocity_rad_s=velocity_rad_s,
                ),
                reject_timeout_s=self._reject_confirmation_timeout_s(
                    len(commanded_waypoints)
                ),
            )
        state = self.wait_for_state(1.0)
        if state.pose is None:
            raise RuntimeError(f"no pose feedback after linear move {name!r}")
        endpoint_error = _pose_distance(state.pose, segment.target)
        if endpoint_error > max(0.005, 2.0 * self.config.position_tolerance_m):
            raise RuntimeError(
                f"linear move {name!r} endpoint error is "
                f"{endpoint_error * 1000:.1f} mm"
            )
        self._log(
            f"[motion:linear] {name}: completed "
            f"{segment.distance_m:.3f} m with "
            f"{len(segment.waypoints) - 1} waypoint(s), "
            f"cartesian_speed={effective_speed_mps:.3f} m/s, "
            f"speed_scale={speed_scale:.2f}, "
            f"joint_velocity={velocity_rad_s:.3f} rad/s, "
            f"max_joint_step={segment.max_joint_step_deg:.3f} deg"
        )

    def _compress_joint_trajectory(
        self,
        targets: Sequence[Sequence[float]] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """Thin a fully prechecked joint path without moving its endpoints.

        The planner still checks every dense waypoint for workspace, IK,
        joint-step, branch, singularity, acceleration and route clearance.
        This helper only reduces controller hand-offs.  A retained chord is
        accepted only when its largest joint step and the deviations of every
        skipped validated point from that chord remain below explicit bounds.
        """

        target_array = np.asarray(targets, dtype=float)
        if target_array.size == 0:
            return target_array.reshape(0, 6)
        if target_array.ndim != 2 or target_array.shape[1] != 6:
            raise ValueError("joint trajectory targets must have shape (N, 6)")
        if not np.all(np.isfinite(target_array)):
            raise ValueError("joint trajectory targets must be finite")
        if (
            not self.config.trajectory_compression_enabled
            or len(target_array) <= 2
        ):
            return target_array

        max_step_rad = math.radians(
            float(self.config.trajectory_compression_max_joint_step_deg)
        )
        max_deviation_rad = math.radians(
            float(self.config.trajectory_compression_max_chord_deviation_deg)
        )
        kept_indices = [0]
        while kept_indices[-1] < len(target_array) - 1:
            start_index = kept_indices[-1]
            best_end = start_index + 1
            for end_index in range(start_index + 1, len(target_array)):
                chord_step = float(
                    np.max(
                        np.abs(
                            target_array[end_index]
                            - target_array[start_index]
                        )
                    )
                )
                if chord_step > max_step_rad + 1e-12:
                    break

                chord_is_safe = True
                chord = (
                    target_array[end_index] - target_array[start_index]
                )
                span = float(end_index - start_index)
                for skipped_index in range(start_index + 1, end_index):
                    alpha = float(skipped_index - start_index) / span
                    interpolated = (
                        target_array[start_index] + alpha * chord
                    )
                    deviation = float(
                        np.max(
                            np.abs(
                                target_array[skipped_index] - interpolated
                            )
                        )
                    )
                    if deviation > max_deviation_rad + 1e-12:
                        chord_is_safe = False
                        break
                if not chord_is_safe:
                    break
                best_end = end_index
            selected_step = float(
                np.max(
                    np.abs(
                        target_array[best_end]
                        - target_array[start_index]
                    )
                )
            )
            if selected_step > max_step_rad + 1e-12:
                raise RuntimeError(
                    f"cannot compress {name!r}: validated joint waypoints "
                    "contain a step larger than the configured compression "
                    "limit"
                )
            kept_indices.append(best_end)

        compressed = target_array[kept_indices]
        if len(compressed) < len(target_array):
            if not np.array_equal(compressed[0], target_array[0]):
                raise RuntimeError("compressed trajectory changed its first point")
            if not np.array_equal(compressed[-1], target_array[-1]):
                raise RuntimeError("compressed trajectory changed its endpoint")
            compressed_step_deg = math.degrees(
                float(np.max(np.abs(np.diff(compressed, axis=0))))
            )
            max_skipped_deviation_deg = 0.0
            for left, right in zip(kept_indices, kept_indices[1:]):
                chord = target_array[right] - target_array[left]
                span = float(right - left)
                for skipped_index in range(left + 1, right):
                    alpha = float(skipped_index - left) / span
                    interpolated = target_array[left] + alpha * chord
                    max_skipped_deviation_deg = max(
                        max_skipped_deviation_deg,
                        math.degrees(
                            float(
                                np.max(
                                    np.abs(
                                        target_array[skipped_index]
                                        - interpolated
                                    )
                                )
                            )
                        ),
                    )
            self._log(
                f"[motion:robotmove] {name}: compressed fully prechecked "
                f"trajectory {len(target_array)}->{len(compressed)} points; "
                f"max_step={compressed_step_deg:.3f} deg "
                f"max_skipped_chord_deviation="
                f"{max_skipped_deviation_deg:.3f} deg"
            )
        return compressed

    def _cartesian_joint_velocity_rad_s(
        self,
        segment: LinearSegmentPlan,
        *,
        speed_mps: float,
        speed_scale: float | None = None,
    ) -> float:
        """Derive the vendor bridge's single scalar joint velocity.

        A Cartesian waypoint is nominally reached in ``distance / speed``
        seconds.  The largest joint increment divided by that time is the
        minimum scalar velocity that preserves the requested Cartesian speed
        over the complete route.  The configured safety limit caps the result;
        acceleration and per-axis step checks have already passed in precheck.
        Pure rotational paths have no translation distance and therefore use
        the explicit joint-space speed.
        """

        scale = (
            self._speed_scale()
            if speed_scale is None
            else normalize_backend_speed_scale(speed_scale)
        )
        speed = bounded_cartesian_speed_mps(
            speed_mps,
            self.config.speed_mps,
            scale,
        )
        joint_speed_cap = bounded_joint_speed_rad_s(
            self.config.joint_speed_rad_s,
            self.config.joint_speed_safety_limit_rad_s,
            scale,
        )
        if len(segment.joint_waypoints) != len(segment.waypoints):
            raise RuntimeError("linear segment has no matching joint plan")
        if len(segment.waypoints) < 2:
            return joint_speed_cap

        required: list[float] = []
        total_distance = 0.0
        for index in range(1, len(segment.waypoints)):
            distance_m = _pose_distance(
                segment.waypoints[index - 1],
                segment.waypoints[index],
            )
            total_distance += distance_m
            joint_delta_rad = float(
                np.max(
                    np.abs(
                        np.asarray(segment.joint_waypoints[index], dtype=float)
                        - np.asarray(
                            segment.joint_waypoints[index - 1],
                            dtype=float,
                        )
                    )
                )
            )
            if distance_m > 1e-12:
                duration_s = distance_m / speed
                required.append(joint_delta_rad / duration_s)

        if total_distance <= 1e-12:
            return joint_speed_cap
        if not required:
            return joint_speed_cap
        velocity_rad_s = min(max(required), joint_speed_cap)
        if velocity_rad_s <= 0.0:
            return joint_speed_cap
        return float(velocity_rad_s)

    def move_linear_to(
        self,
        pose: Pose,
        *,
        name: str,
        speed_mps: float,
        step_m: float = 0.01,
    ) -> None:
        route = self.precheck_linear_targets((pose,), step_m=step_m)
        self.execute_linear_route(route, names=(name,), speed_mps=speed_mps)

    def _send_cartesian(self, pose: Pose, *, name: str, speed_mps: float) -> None:
        """Legacy path kept only for controllers that accept Cartesian input."""

        effective_speed_mps = bounded_cartesian_speed_mps(
            speed_mps,
            self.config.speed_mps,
            self._speed_scale(),
        )
        message = self._trajectory_type()
        message.joint_names = list(self.CARTESIAN_NAMES)
        point = self._point_type()
        point.positions = pose.as_list()
        point.velocities = [effective_speed_mps] * 6
        message.points.append(point)
        state_before = self.get_state()
        estimated_motion_s = 0.0
        if state_before.pose is not None:
            estimated_motion_s = _pose_distance(state_before.pose, pose) / (
                effective_speed_mps
            )
        timeout_s = min(
            MAX_TRAJECTORY_TIMEOUT_S,
            max(
                float(self.config.move_timeout_s),
                15.0 + 2.0 * estimated_motion_s,
            ),
        )
        self._publisher.publish(message)
        self._log(
            f"[motion:cartesian] sent {name}: xyz={pose.xyz} rpy={pose.rpy} "
            f"v={effective_speed_mps:.3f} timeout={timeout_s:.1f}s"
        )

        deadline = time.monotonic() + timeout_s
        target = pose.xyz
        while time.monotonic() < deadline:
            state = self.get_state()
            if state.fault:
                raise RuntimeError(f"fault during move {name!r}: {state.message}")
            if state.pose is not None:
                error = max(
                    abs(float(a) - float(b))
                    for a, b in zip(state.pose.xyz, target)
                )
                if error <= self.config.position_tolerance_m:
                    return
            time.sleep(self.config.poll_period_s)
        raise RuntimeError(f"move {name!r} timed out after {timeout_s:.1f}s")

    def move_joints(
        self,
        target: Sequence[float],
        *,
        name: str,
        max_step_deg: float | None = None,
        timeout_s: float | None = None,
        reject_timeout_s: float | None = None,
    ) -> np.ndarray:
        """Move to an absolute joint vector using bounded interpolation."""

        target_q = np.asarray(target, dtype=float)
        if target_q.shape != (6,) or not np.all(np.isfinite(target_q)):
            raise ValueError("joint target must contain six finite values")
        speed_scale = self._speed_scale()
        joint_velocity_rad_s = bounded_joint_speed_rad_s(
            self.config.joint_speed_rad_s,
            self.config.joint_speed_safety_limit_rad_s,
            speed_scale,
        )
        if self.dry_run:
            self._log(
                f"[motion:robotmove:dry-run] {name}: would send joints="
                f"{np.round(np.degrees(target_q), 2).tolist()} deg "
                f"joint_velocity={joint_velocity_rad_s:.3f} rad/s "
                f"speed_scale={speed_scale:.2f}"
            )
            self._advance_dry_run_joints(target_q)
            return target_q

        state = self.wait_for_state(1.0)
        if not state.ready or state.joints is None:
            raise RuntimeError(f"robot not ready before {name!r}: {state.message}")
        if not joints_within_limits(
            target_q,
            math.radians(float(self.config.joint_limit_margin_deg)),
        ):
            raise RuntimeError(f"joint target for {name!r} violates limits")

        start_q = np.asarray(state.joints, dtype=float)
        max_step_rad = math.radians(
            float(
                self.config.joint_max_step_deg
                if max_step_deg is None
                else max_step_deg
            )
        )
        if max_step_rad <= 0.0:
            raise ValueError("joint max step must be positive")
        largest_delta = float(np.max(np.abs(target_q - start_q)))
        if largest_delta <= math.radians(self.config.joint_tolerance_deg):
            self._log(f"[motion:robotmove] {name}: already at target")
            return start_q

        steps = max(1, int(math.ceil(largest_delta / max_step_rad)))
        self._log(
            f"[motion:robotmove] {name}: {steps} step(s), "
            f"max_delta={math.degrees(largest_delta):.2f} deg, "
            f"joint_velocity={joint_velocity_rad_s:.3f} rad/s, "
            f"speed_scale={speed_scale:.2f}"
        )
        joint_waypoints = [
            start_q + (target_q - start_q) * (index / steps)
            for index in range(1, steps + 1)
        ]
        _validate_joint_waypoint_workspace(
            self.config,
            joint_waypoints,
            name=name,
        )
        effective_timeout_s = (
            self._trajectory_timeout_s(
                len(joint_waypoints),
                joint_travel_rad=_joint_path_travel_rad(
                    start_q,
                    joint_waypoints,
                ),
                velocity_rad_s=joint_velocity_rad_s,
            )
            if timeout_s is None
            else float(timeout_s)
        )
        return self._send_joint_trajectory(
            joint_waypoints,
            name=name,
            velocity_rad_s=joint_velocity_rad_s,
            timeout_s=effective_timeout_s,
            reject_timeout_s=(
                self._reject_confirmation_timeout_s(len(joint_waypoints))
                if reject_timeout_s is None
                else float(reject_timeout_s)
            ),
        )

    def _trajectory_timeout_s(
        self,
        point_count: int,
        *,
        joint_travel_rad: float | None = None,
        velocity_rad_s: float | None = None,
    ) -> float:
        """Return a conservative whole-trajectory timeout.

        A continuous segment is no longer bounded by the historical 8 s
        single-waypoint timeout.  The controller may need dozens of seconds to
        traverse a long pre-checked path, so allow 15 s overhead plus 1.0 s per
        commanded point while retaining the configured full-move timeout as a
        floor.  The per-point allowance is intentionally conservative: the S3
        controller has been observed to spend roughly 0.6-0.7 s per queued
        point on a long joint trajectory, so a 0.5 s budget could expire just
        before the final settled feedback was published.
        """

        count = max(1, int(point_count))
        baseline_s = max(
            float(self.config.joint_move_timeout_s),
            float(self.config.move_timeout_s),
            15.0 + 1.0 * float(count),
        )
        estimate_s = baseline_s
        if joint_travel_rad is not None and velocity_rad_s is not None:
            travel = float(joint_travel_rad)
            velocity = float(velocity_rad_s)
            if (
                math.isfinite(travel)
                and travel > 0.0
                and math.isfinite(velocity)
                and velocity > 0.0
            ):
                # Two times the theoretical joint travel plus fixed overhead
                # covers controller planning, settling and low-speed feedback.
                estimate_s = max(
                    baseline_s,
                    20.0 + 2.0 * travel / velocity,
                )
        return min(MAX_TRAJECTORY_TIMEOUT_S, estimate_s)

    def _reject_confirmation_timeout_s(self, point_count: int) -> float:
        """Return how long to wait for the controller to accept a trajectory.

        A long ``JointTrajectory`` can remain ``AutoReady`` while the
        controller plans it.  Keep the configured two-second fast-fail window
        for short commands, then add a bounded planning allowance of 0.25 s per
        commanded point beyond the first 16.  This never extends the overall
        trajectory timeout and caps the confirmation wait at 30 s.
        """

        count = max(1, int(point_count))
        base_s = float(self.config.joint_reject_timeout_s)
        planning_window_s = base_s + 0.25 * max(0, count - 16)
        return min(
            max(base_s, planning_window_s),
            30.0,
            self._trajectory_timeout_s(count),
        )

    def _send_joint_waypoint(
        self,
        target: np.ndarray,
        *,
        name: str,
        velocity_rad_s: float,
        timeout_s: float,
        reject_timeout_s: float,
    ) -> np.ndarray:
        return self._send_joint_trajectory(
            (target,),
            name=name,
            velocity_rad_s=velocity_rad_s,
            timeout_s=timeout_s,
            reject_timeout_s=reject_timeout_s,
        )

    @staticmethod
    def _joint_trajectory_payload_size(
        targets: Sequence[Sequence[float]] | np.ndarray,
        *,
        velocity_rad_s: float,
    ) -> int:
        """Return the exact JSON size emitted by the current UDP bridge."""

        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim == 1:
            target_array = target_array.reshape(1, 6)
        velocity = float(velocity_rad_s)
        payload = {
            "joint_names": list(JOINT_NAMES),
            "points": [
                {
                    "positions": [float(value) for value in target_q],
                    "velocities": [velocity],
                    "accelerations": [],
                }
                for target_q in target_array
            ],
        }
        return len(json.dumps(payload).encode("utf-8"))

    def _split_joint_trajectory(
        self,
        targets: np.ndarray,
        *,
        name: str,
        velocity_rad_s: float,
    ) -> tuple[np.ndarray, ...]:
        """Split a path into datagrams the controller has accepted before."""

        chunks: list[np.ndarray] = []
        start = 0
        total = len(targets)
        while start < total:
            end = min(total, start + ROBOT_CONTROLLER_MAX_TRAJECTORY_POINTS)
            chunk = targets[start:end]
            while (
                len(chunk) > 1
                and self._joint_trajectory_payload_size(
                    chunk,
                    velocity_rad_s=velocity_rad_s,
                )
                > ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES
            ):
                end -= 1
                chunk = targets[start:end]
            payload_bytes = self._joint_trajectory_payload_size(
                chunk,
                velocity_rad_s=velocity_rad_s,
            )
            if payload_bytes > ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES:
                raise RuntimeError(
                    f"joint trajectory {name!r} chunk at point {start + 1} "
                    f"would create a {payload_bytes}-byte payload, above the "
                    f"controller-safe {ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES}-"
                    "byte limit"
                )
            chunks.append(chunk.copy())
            start = end
        return tuple(chunks)

    def _send_joint_trajectory(
        self,
        targets: Sequence[Sequence[float]] | np.ndarray,
        *,
        name: str,
        velocity_rad_s: float,
        timeout_s: float,
        reject_timeout_s: float,
    ) -> np.ndarray:
        """Publish a pre-checked path through controller-safe chunks.

        Every target has already passed the caller's IK, limit, workspace,
        singularity, joint-step and acceleration checks.  Chunks remain
        sequential: the next datagram is sent only after the controller has
        completed the previous chunk and feedback has settled.  Any missing
        acknowledgement, fault, stale boundary or timeout stops the path.
        """

        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim == 1:
            target_array = target_array.reshape(1, 6)
        if target_array.ndim != 2 or target_array.shape[1] != 6:
            raise ValueError("joint trajectory targets must have shape (N, 6)")
        if target_array.shape[0] < 1 or not np.all(np.isfinite(target_array)):
            raise ValueError("joint trajectory needs at least one finite target")
        velocity = float(velocity_rad_s)
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError(
                "joint trajectory velocity must be positive and finite"
            )
        margin_rad = math.radians(float(self.config.joint_limit_margin_deg))
        for index, target_q in enumerate(target_array, start=1):
            if not joints_within_limits(target_q, margin_rad):
                raise RuntimeError(
                    f"joint trajectory {name!r} target {index}/"
                    f"{len(target_array)} violates the configured joint-limit "
                    f"margin ({self.config.joint_limit_margin_deg:g} deg)"
                )

        chunks = self._split_joint_trajectory(
            target_array,
            name=name,
            velocity_rad_s=velocity,
        )
        if len(chunks) > 1:
            self._log(
                f"[motion:robotmove] {name}: split {len(target_array)} points "
                f"into {len(chunks)} controller-safe chunks "
                f"points={[len(chunk) for chunk in chunks]} "
                f"max_points={ROBOT_CONTROLLER_MAX_TRAJECTORY_POINTS} "
                f"max_payload_bytes={ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES}"
            )

        start_time = time.monotonic()
        deadline = start_time + float(timeout_s)
        actual = np.asarray(target_array[-1], dtype=float)
        for index, chunk in enumerate(chunks, start=1):
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError(
                    f"joint move {name!r} timed out after {timeout_s}s "
                    f"before chunk {index}/{len(chunks)}"
                )

            if index > 1:
                state = self.wait_for_state(1.0)
                if state.fault:
                    raise RuntimeError(
                        f"controller fault between {name!r} chunks "
                        f"{index - 1}/{len(chunks)} and {index}/{len(chunks)}"
                    )
                if not state.ready or state.joints is None:
                    raise RuntimeError(
                        f"robot not ready before {name!r} chunk "
                        f"{index}/{len(chunks)}: {state.message}"
                    )
                actual = np.asarray(state.joints, dtype=float)
                next_start_delta_deg = math.degrees(
                    float(np.max(np.abs(actual - chunk[0])))
                )
                allowed_delta_deg = (
                    max(
                        float(self.config.joint_max_step_deg),
                        float(self.config.cartesian_joint_step_deg),
                    )
                    + float(self.config.joint_tolerance_deg)
                )
                if next_start_delta_deg > allowed_delta_deg + 1e-9:
                    raise RuntimeError(
                        f"refusing {name!r} chunk {index}/{len(chunks)}: "
                        f"feedback is {next_start_delta_deg:.3f} deg from its "
                        f"first target (limit {allowed_delta_deg:.3f} deg)"
                    )

            chunk_name = (
                name
                if len(chunks) == 1
                else f"{name} chunk {index}/{len(chunks)}"
            )
            chunk_timeout_s = min(
                self._trajectory_timeout_s(len(chunk)),
                max(0.0, remaining_s),
            )
            chunk_reject_timeout_s = min(
                float(reject_timeout_s),
                self._reject_confirmation_timeout_s(len(chunk)),
            )
            actual = self._send_joint_trajectory_once(
                chunk,
                name=chunk_name,
                velocity_rad_s=velocity,
                timeout_s=chunk_timeout_s,
                reject_timeout_s=chunk_reject_timeout_s,
            )
        return actual

    def _send_joint_trajectory_once(
        self,
        targets: Sequence[Sequence[float]] | np.ndarray,
        *,
        name: str,
        velocity_rad_s: float,
        timeout_s: float,
        reject_timeout_s: float,
    ) -> np.ndarray:
        """Publish one controller-safe multi-point joint trajectory datagram."""

        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim == 1:
            target_array = target_array.reshape(1, 6)
        if target_array.ndim != 2 or target_array.shape[1] != 6:
            raise ValueError("joint trajectory targets must have shape (N, 6)")
        if target_array.shape[0] < 1 or not np.all(np.isfinite(target_array)):
            raise ValueError("joint trajectory needs at least one finite target")
        velocity = float(velocity_rad_s)
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError(
                "joint trajectory velocity must be positive and finite"
            )
        if len(target_array) > ROBOT_CONTROLLER_MAX_TRAJECTORY_POINTS:
            raise RuntimeError(
                f"joint trajectory {name!r} has {len(target_array)} points, "
                f"above the controller-safe "
                f"{ROBOT_CONTROLLER_MAX_TRAJECTORY_POINTS}-point limit"
            )

        state = self.wait_for_state(1.0)
        if not state.ready or state.joints is None:
            raise RuntimeError(f"cannot move {name!r}: {state.message}")

        start_q = np.asarray(state.joints, dtype=float)
        tolerance_rad = math.radians(self.config.joint_tolerance_deg)
        start_tolerance_rad = math.radians(
            self.config.joint_start_tolerance_deg
        )
        path_delta_rad = float(
            np.max(np.abs(target_array - start_q[None, :]))
        )
        if path_delta_rad <= start_tolerance_rad:
            self._log(
                f"[motion:robotmove] {name}: already within startup tolerance "
                f"({math.degrees(path_delta_rad):.3f} deg <= "
                f"{self.config.joint_start_tolerance_deg:.3f} deg); "
                "no joint command sent"
            )
            return start_q

        payload_bytes = self._joint_trajectory_payload_size(
            target_array,
            velocity_rad_s=velocity,
        )
        if payload_bytes > ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES:
            raise RuntimeError(
                f"joint trajectory {name!r} would create a {payload_bytes}-byte "
                "payload, above the controller-safe "
                f"{ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES}-byte limit"
            )
        if payload_bytes > UDP_MAX_DATAGRAM_BYTES:
            raise RuntimeError(
                f"joint trajectory {name!r} would create a {payload_bytes}-byte "
                f"UDP payload, above the {UDP_MAX_DATAGRAM_BYTES}-byte limit; "
                "reduce Cartesian waypoints before publishing"
            )

        endpoint_q = target_array[-1]
        with self._lock:
            sent_stamp = self._stamp

        message = self._trajectory_type()
        message.joint_names = list(JOINT_NAMES)
        for target_q in target_array:
            point = self._point_type()
            point.positions = [float(v) for v in target_q]
            point.velocities = [velocity]
            point.accelerations = []
            message.points.append(point)
        self._publisher.publish(message)
        start_time = time.monotonic()
        saw_running = False
        self._log(
            f"[motion:robotmove] sent continuous {name}: "
            f"points={len(target_array)} payload_bytes={payload_bytes} "
            f"joint_velocity={velocity:.3f} rad/s endpoint_deg="
            f"{np.round(np.degrees(endpoint_q), 2).tolist()}"
        )

        deadline = start_time + float(timeout_s)
        while time.monotonic() < deadline:
            time.sleep(self.config.poll_period_s)
            current = self.get_state()
            if current.fault:
                raise RuntimeError(f"controller fault during {name!r}")
            with self._lock:
                current_state_name = self._state
                current_stamp = self._stamp
            if current_state_name == "AutoRunning":
                saw_running = True
            if current.joints is None:
                continue

            actual = np.asarray(current.joints, dtype=float)
            error = float(np.max(np.abs(actual - endpoint_q)))
            moved = (
                float(np.max(np.abs(actual - start_q))) > start_tolerance_rad
            )
            if (
                current_state_name == "AutoReady"
                and error <= tolerance_rad
                and (saw_running or moved)
            ):
                return actual

            elapsed = time.monotonic() - start_time
            if (
                current_stamp > sent_stamp
                and current_state_name == "AutoReady"
                and not saw_running
                and not moved
                and elapsed >= float(reject_timeout_s)
            ):
                raise RuntimeError(
                    f"controller ignored joint move {name!r}; state stayed "
                    "AutoReady and joints did not change "
                    f"(error={math.degrees(error):.3f} deg)"
                )
        raise RuntimeError(f"joint move {name!r} timed out after {timeout_s}s")

    def close(self) -> None:
        return None


def build_motion(
    config: MotionConfig,
    node=None,
    log=None,
    *,
    profile: str | None = None,
    speed_store: SpeedLimitStore | None = None,
):
    if config.backend == "mock":
        return MockMotionBackend(
            config,
            log=log,
            profile=profile,
            speed_store=speed_store,
        )
    if config.backend == "robotmove":
        if node is None:
            raise RuntimeError("motion.backend=robotmove requires a running ROS 2 node")
        return RobotMoveMotionBackend(
            config,
            node=node,
            log=log,
            profile=profile,
            speed_store=speed_store,
        )
    raise ValueError(f"unsupported motion backend: {config.backend!r}")
