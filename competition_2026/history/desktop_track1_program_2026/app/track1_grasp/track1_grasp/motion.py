"""Motion backends.

The mock backend records the complete path without hardware.  The RobotMove
backend targets the Estun/Codroid ROS bridge already used by taskA_grasp:
``/RobotInfo`` in, ``/RobotMove`` out.  A future MoveIt backend can implement
the same interface without changing the task state machine.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import MotionConfig
from .domain import Pose, RobotState
from .kinematics import (
    JOINT_NAMES,
    axis_angle_matrix,
    controller_fk,
    joints_within_limits,
    matrix_to_rpy,
    rotation_angle_error,
    rotation_matrix_to_vector,
    rpy_to_matrix,
    solve_pose_ik_multistart,
)


@dataclass(frozen=True)
class LinearSegmentPlan:
    """One straight TCP segment, pre-checked before anything is published."""

    target: Pose
    waypoints: tuple[Pose, ...]
    joint_waypoints: tuple[tuple[float, ...], ...] = ()

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


def _pose_distance(start: Pose, end: Pose) -> float:
    delta = np.asarray(end.xyz, dtype=float) - np.asarray(start.xyz, dtype=float)
    return float(np.linalg.norm(delta))


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


def linear_pose_waypoints(start: Pose, target: Pose, *, step_m: float) -> tuple[Pose, ...]:
    """Return a straight-line TCP path with no spacing larger than ``step_m``."""

    step = float(step_m)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("linear step must be a positive finite distance")

    start_xyz = np.asarray(start.xyz, dtype=float)
    target_xyz = np.asarray(target.xyz, dtype=float)
    if not np.all(np.isfinite(start_xyz)) or not np.all(np.isfinite(target_xyz)):
        raise ValueError("linear path endpoints must be finite")

    distance = float(np.linalg.norm(target_xyz - start_xyz))
    # Avoid an extra waypoint when a decimal step divides the distance
    # exactly but binary floating point produces 4.000000000000001.
    steps = max(1, int(math.ceil(distance / step - 1e-12)))
    start_rotation = rpy_to_matrix(start.rpy)
    target_rotation = rpy_to_matrix(target.rpy)

    waypoints: list[Pose] = [start]
    for index in range(1, steps + 1):
        fraction = index / steps
        xyz = start_xyz + (target_xyz - start_xyz) * fraction
        rotation = _interpolate_rotation(
            start_rotation, target_rotation, fraction
        )
        waypoints.append(Pose(tuple(float(v) for v in xyz), matrix_to_rpy(rotation)))
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


class MockMotionBackend:
    """Deterministic motion simulator for framework testing."""

    def __init__(self, config: MotionConfig, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self._pose = Pose((0.0, 0.0, 0.30), config.fixed_rpy)
        self.moves: list[tuple[str, Pose]] = []
        #: Kept separate from ``moves`` so existing pick-task tests retain the
        #: historical one-entry-per-waypoint behaviour.
        self.linear_moves: list[tuple[str, tuple[Pose, ...]]] = []

    @property
    def dry_run(self) -> bool:
        return bool(self.config.dry_run)

    def get_state(self) -> RobotState:
        return RobotState(ready=True, pose=self._pose, message="mock-ready")

    def move_to(self, pose: Pose, *, name: str, speed_mps: float) -> None:
        self.moves.append((name, pose))
        self._pose = pose
        self._log(
            f"[motion:mock] {name}: xyz={pose.xyz} rpy={pose.rpy} v={speed_mps:.3f}"
        )

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

    def execute_linear_route(
        self, route: LinearRoutePlan, *, names: Sequence[str], speed_mps: float
    ) -> None:
        if len(names) != len(route.segments):
            raise ValueError("one name is required for every linear segment")
        for name, segment in zip(names, route.segments):
            self.linear_moves.append((str(name), segment.waypoints))
            self._pose = segment.target
            self._log(
                f"[motion:mock-linear] {name}: distance={segment.distance_m:.3f} m "
                f"waypoints={max(0, len(segment.waypoints) - 1)} "
                f"v={speed_mps:.3f}"
            )

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

    def close(self) -> None:
        return None


class RobotMoveMotionBackend:
    """Absolute-joint publisher for the Codroid ``RobotMove`` bridge.

    The live controller silently ignores Cartesian ``x,y,z,rx,ry,rz``
    trajectories in the current installation.  Every Cartesian waypoint is
    therefore converted to a nearby full-pose IK solution and sent as absolute
    ``Joint1..Joint6`` positions, interpolated in maximum 1 degree steps.
    """

    CARTESIAN_NAMES = ["x", "y", "z", "rx", "ry", "rz"]

    def __init__(self, config: MotionConfig, node, log=None) -> None:
        self.config = config
        self.node = node
        self._log = log or (lambda message: None)
        self._lock = threading.Lock()
        self._state = ""
        self._end_positions: list[float] = []
        self._joint_positions: list[float] = []
        self._fault_flag = 0
        self._stamp = 0.0

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

    def _on_info(self, msg) -> None:
        with self._lock:
            self._state = str(msg.state)
            self._end_positions = [float(v) for v in msg.end_positions]
            self._joint_positions = [
                float(v) for v in getattr(msg, "joint_positions", ())
            ]
            self._fault_flag = int(msg.fault_flag)
            self._stamp = time.monotonic()

    def get_state(self) -> RobotState:
        with self._lock:
            state_name = self._state
            end_positions = tuple(self._end_positions)
            joint_positions = tuple(self._joint_positions)
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
        )

    def wait_for_state(self, timeout_s: float) -> RobotState:
        """Wait until a fresh RobotInfo with pose and joints is available."""

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        state = self.get_state()
        while time.monotonic() < deadline and state.pose is None:
            time.sleep(min(self.config.poll_period_s, 0.05))
            state = self.get_state()
        return state

    def move_to(self, pose: Pose, *, name: str, speed_mps: float) -> None:
        state = self.wait_for_state(1.0)
        if self.dry_run:
            detail = ""
            if state.joints is not None:
                try:
                    q_target = solve_pose_ik_multistart(
                        state.joints, pose.xyz, rpy_to_matrix(pose.rpy)
                    )
                    detail = (
                        " joints="
                        f"{np.round(np.degrees(q_target), 2).tolist()} deg"
                    )
                except RuntimeError as exc:
                    detail = f" ik_failed={exc}"
            self._log(
                f"[motion:robotmove:dry-run] {name}: "
                f"xyz={pose.xyz} rpy={pose.rpy}{detail}"
            )
            return

        if not state.ready:
            raise RuntimeError(f"robot not ready before move {name!r}: {state.message}")
        if state.joints is None or state.pose is None:
            raise RuntimeError(f"robot state incomplete before move {name!r}")

        if self.config.motion_mode == "cartesian":
            self._send_cartesian(pose, name=name, speed_mps=speed_mps)
            return

        target_rotation = rpy_to_matrix(pose.rpy)
        try:
            q_target = solve_pose_ik_multistart(
                state.joints, pose.xyz, target_rotation
            )
        except RuntimeError as exc:
            raise RuntimeError(f"pose IK failed for {name!r}: {exc}") from exc
        if not joints_within_limits(q_target):
            raise RuntimeError(f"IK target for {name!r} violates joint limits")

        solved = controller_fk(q_target)
        position_error = float(
            np.linalg.norm(solved[:3, 3] - np.asarray(pose.xyz, dtype=float))
        )
        orientation_error = rotation_angle_error(
            solved[:3, :3], target_rotation
        )
        self._log(
            f"[motion:ik] {name}: xyz={pose.xyz} -> "
            f"joints={np.round(np.degrees(q_target), 2).tolist()} deg "
            f"error={position_error * 1000:.3f} mm/"
            f"{orientation_error:.5f} rad"
        )
        self.move_joints(q_target, name=name)

    def precheck_linear_targets(
        self, targets: Sequence[Pose], *, step_m: float
    ) -> LinearRoutePlan:
        """Resolve and validate every straight-path waypoint without moving."""

        state = self.wait_for_state(1.0)
        if state.fault:
            raise RuntimeError(f"cannot pre-check patrol: {state.message}")
        if not state.ready or state.pose is None or state.joints is None:
            raise RuntimeError(
                "cannot pre-check patrol without a ready robot state: "
                f"{state.message}"
            )

        start_pose = state.pose
        current_pose = start_pose
        current_q = np.asarray(state.joints, dtype=float)

        measured = controller_fk(current_q)
        measured_position_error = float(
            np.linalg.norm(measured[:3, 3] - np.asarray(start_pose.xyz, dtype=float))
        )
        measured_rotation_error = rotation_angle_error(
            measured[:3, :3], rpy_to_matrix(start_pose.rpy)
        )
        if measured_position_error > max(0.01, 2.0 * self.config.position_tolerance_m):
            raise RuntimeError(
                "RobotInfo pose and joint feedback disagree by "
                f"{measured_position_error * 1000:.1f} mm"
            )
        if measured_rotation_error > 0.02:
            raise RuntimeError(
                "RobotInfo orientation and joint feedback disagree by "
                f"{measured_rotation_error:.4f} rad"
            )

        segments: list[LinearSegmentPlan] = []
        for target_index, target in enumerate(targets, start=1):
            waypoints = linear_pose_waypoints(current_pose, target, step_m=step_m)
            joint_waypoints: list[tuple[float, ...]] = [tuple(float(v) for v in current_q)]
            for waypoint_index, waypoint in enumerate(waypoints[1:], start=1):
                label = (
                    f"patrol target {target_index} waypoint "
                    f"{waypoint_index}/{len(waypoints) - 1}"
                )
                _validate_workspace(self.config, waypoint, label)
                try:
                    solved_q = solve_pose_ik_multistart(
                        current_q,
                        waypoint.xyz,
                        rpy_to_matrix(waypoint.rpy),
                    )
                except RuntimeError as exc:
                    raise RuntimeError(f"IK failed for {label}: {exc}") from exc
                if not joints_within_limits(solved_q):
                    raise RuntimeError(f"IK solution for {label} violates joint limits")

                solved = controller_fk(solved_q)
                position_error = float(
                    np.linalg.norm(
                        solved[:3, 3] - np.asarray(waypoint.xyz, dtype=float)
                    )
                )
                orientation_error = rotation_angle_error(
                    solved[:3, :3], rpy_to_matrix(waypoint.rpy)
                )
                if position_error > max(0.002, 2.0 * self.config.position_tolerance_m):
                    raise RuntimeError(
                        f"IK position error for {label} is "
                        f"{position_error * 1000:.2f} mm"
                    )
                if orientation_error > 0.002:
                    raise RuntimeError(
                        f"IK orientation error for {label} is "
                        f"{orientation_error:.5f} rad"
                    )
                current_q = solved_q
                joint_waypoints.append(tuple(float(v) for v in current_q))

            segments.append(
                LinearSegmentPlan(
                    target=target,
                    waypoints=waypoints,
                    joint_waypoints=tuple(joint_waypoints),
                )
            )
            current_pose = target

        self._log(
            f"[motion:linear-precheck] validated {len(segments)} segment(s), "
            f"{sum(max(0, len(item.waypoints) - 1) for item in segments)} "
            "Cartesian waypoint(s)"
        )
        return LinearRoutePlan(start_pose=start_pose, segments=tuple(segments))

    def execute_linear_route(
        self, route: LinearRoutePlan, *, names: Sequence[str], speed_mps: float
    ) -> None:
        if len(names) != len(route.segments):
            raise ValueError("one name is required for every linear segment")

        if self.dry_run:
            for name, segment in zip(names, route.segments):
                self._log(
                    f"[motion:robotmove:dry-run-linear] {name}: "
                    f"distance={segment.distance_m:.3f} m "
                    f"waypoints={max(0, len(segment.waypoints) - 1)} "
                    f"target_xyz={segment.target.xyz} "
                    f"v={speed_mps:.3f}"
                )
            return

        for name, segment in zip(names, route.segments):
            if len(segment.joint_waypoints) != len(segment.waypoints):
                raise RuntimeError(f"linear route {name!r} has no joint plan")
            for waypoint_index, target_q in enumerate(
                segment.joint_waypoints[1:], start=1
            ):
                self.move_joints(
                    target_q,
                    name=f"{name}_p{waypoint_index:03d}",
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
                f"{segment.distance_m:.3f} m with {len(segment.waypoints) - 1} waypoint(s)"
            )

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

        message = self._trajectory_type()
        message.joint_names = list(self.CARTESIAN_NAMES)
        point = self._point_type()
        point.positions = pose.as_list()
        point.velocities = [float(speed_mps)] * 6
        message.points.append(point)
        self._publisher.publish(message)
        self._log(
            f"[motion:cartesian] sent {name}: xyz={pose.xyz} rpy={pose.rpy}"
        )

        deadline = time.monotonic() + self.config.move_timeout_s
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
        raise RuntimeError(f"move {name!r} timed out after {self.config.move_timeout_s}s")

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
        if self.dry_run:
            self._log(
                f"[motion:robotmove:dry-run] {name}: would send joints="
                f"{np.round(np.degrees(target_q), 2).tolist()} deg"
            )
            return target_q

        state = self.wait_for_state(1.0)
        if not state.ready or state.joints is None:
            raise RuntimeError(f"robot not ready before {name!r}: {state.message}")
        if not joints_within_limits(target_q):
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
            f"max_delta={math.degrees(largest_delta):.2f} deg"
        )
        current_q = start_q.copy()
        for index in range(1, steps + 1):
            waypoint = start_q + (target_q - start_q) * (index / steps)
            current_q = self._send_joint_waypoint(
                waypoint,
                name=f"{name}_step_{index:02d}",
                timeout_s=(
                    self.config.joint_move_timeout_s
                    if timeout_s is None
                    else float(timeout_s)
                ),
                reject_timeout_s=(
                    self.config.joint_reject_timeout_s
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
        state = self.wait_for_state(1.0)
        if not state.ready or state.joints is None:
            raise RuntimeError(f"cannot move {name!r}: {state.message}")

        start_q = np.asarray(state.joints, dtype=float)
        tolerance_rad = math.radians(self.config.joint_tolerance_deg)
        start_tolerance_rad = math.radians(
            self.config.joint_start_tolerance_deg
        )
        with self._lock:
            sent_stamp = self._stamp

        message = self._trajectory_type()
        message.joint_names = list(JOINT_NAMES)
        point = self._point_type()
        point.positions = [float(v) for v in target]
        message.points.append(point)
        self._publisher.publish(message)
        start_time = time.monotonic()
        saw_running = False
        self._log(
            f"[motion:robotmove] sent {name}: joints="
            f"{np.round(np.degrees(target), 2).tolist()} deg"
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
            error = float(np.max(np.abs(actual - target)))
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


def build_motion(config: MotionConfig, node=None, log=None):
    if config.backend == "mock":
        return MockMotionBackend(config, log=log)
    if config.backend == "robotmove":
        if node is None:
            raise RuntimeError("motion.backend=robotmove requires a running ROS 2 node")
        return RobotMoveMotionBackend(config, node=node, log=log)
    raise ValueError(f"unsupported motion backend: {config.backend!r}")
