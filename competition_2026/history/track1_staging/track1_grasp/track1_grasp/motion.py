"""Motion backends.

The mock backend records the complete path without hardware.  The RobotMove
backend targets the Estun/Codroid ROS bridge already used by taskA_grasp:
``/RobotInfo`` in, ``/RobotMove`` out.  A future MoveIt backend can implement
the same interface without changing the task state machine.
"""

from __future__ import annotations

import time

from .config import MotionConfig
from .domain import Pose, RobotState


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
    """Cartesian publisher for the Codroid ``RobotMove`` UDP bridge.

    The robot controller subscribes on its own port; ``codroid_bridge`` is
    responsible for translating ROS 2 ``/RobotMove`` into the controller UDP
    protocol and publishing fresh ``/RobotInfo``.
    """

    CARTESIAN_NAMES = ["x", "y", "z", "rx", "ry", "rz"]

    def __init__(self, config: MotionConfig, node, log=None) -> None:
        self.config = config
        self.node = node
        self._log = log or (lambda message: None)
        self._state = ""
        self._end_positions: list[float] = []
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
        self._state = str(msg.state)
        self._end_positions = [float(v) for v in msg.end_positions]
        self._fault_flag = int(msg.fault_flag)
        self._stamp = time.monotonic()

    def get_state(self) -> RobotState:
        age = time.monotonic() - self._stamp if self._stamp else float("inf")
        pose = (
            Pose(tuple(self._end_positions[:3]), tuple(self._end_positions[3:6]))
            if len(self._end_positions) >= 6
            else None
        )
        fault = self._fault_flag != 0
        fresh = age <= self.config.robot_info_timeout_s
        ready = fresh and not fault and self._state == "AutoReady"
        if not fresh:
            message = "no fresh RobotInfo"
        elif fault:
            message = f"controller fault_flag={self._fault_flag}"
        else:
            message = f"controller state={self._state or 'unknown'}"
        return RobotState(ready=ready, pose=pose, fault=fault, message=message)

    def move_to(self, pose: Pose, *, name: str, speed_mps: float) -> None:
        if self.dry_run:
            self._log(
                f"[motion:robotmove:dry-run] {name}: xyz={pose.xyz} rpy={pose.rpy}"
            )
            return

        state = self.get_state()
        if not state.ready:
            raise RuntimeError(f"robot not ready before move {name!r}: {state.message}")

        message = self._trajectory_type()
        message.joint_names = list(self.CARTESIAN_NAMES)
        point = message.points.add()
        point.positions = pose.as_list()
        point.velocities = [float(speed_mps)] * 6
        self._publisher.publish(message)
        self._log(f"[motion:robotmove] sent {name}: xyz={pose.xyz} rpy={pose.rpy}")

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
