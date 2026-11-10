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
from typing import Sequence

import numpy as np

from .config import MotionConfig
from .domain import Pose, RobotState
from .kinematics import (
    JOINT_NAMES,
    controller_fk,
    joints_within_limits,
    rotation_angle_error,
    rpy_to_matrix,
    solve_pose_ik,
)


class MockMotionBackend:
    """Deterministic motion simulator for framework testing."""

    def __init__(self, config: MotionConfig, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self._pose = Pose((0.0, 0.0, 0.30), config.fixed_rpy)
        self.moves: list[tuple[str, Pose]] = []

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

        from trajectory_msgs.msg import JointTrajectory

        try:
            from codroid_msgs.msg import RobotInfo
        except ImportError as exc:
            raise RuntimeError(
                "motion.backend=robotmove needs codroid_msgs on AMENT_PREFIX_PATH"
            ) from exc

        self._trajectory_type = JointTrajectory
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
        pose = (
            Pose(tuple(end_positions[:3]), tuple(end_positions[3:6]))
            if len(end_positions) >= 6
            else None
        )
        joints = tuple(joint_positions[:6]) if len(joint_positions) >= 6 else None
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
        elif pose is None or joints is None:
            message = "RobotInfo is missing end positions or joint positions"
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
                    q_target = solve_pose_ik(
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
            q_target = solve_pose_ik(state.joints, pose.xyz, target_rotation)
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

    def _send_cartesian(self, pose: Pose, *, name: str, speed_mps: float) -> None:
        """Legacy path kept only for controllers that accept Cartesian input."""

        message = self._trajectory_type()
        message.joint_names = list(self.CARTESIAN_NAMES)
        point = message.points.add()
        point.positions = pose.as_list()
        point.velocities = [float(speed_mps)] * 6
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
        point = message.points.add()
        point.positions = [float(v) for v in target]
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
