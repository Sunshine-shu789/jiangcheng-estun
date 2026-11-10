"""Protocol boundaries for replaceable vision, motion, gripper and hold checks.

The orchestration layer only knows these Protocols.  Consequently the current
mock implementation can be replaced by a RealSense/YOLO source, a
MoveIt/controller motion backend and a real gripper without changing the task
state machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

from .domain import Detection, Frame, GripperState, MotionPlan, Pose, RobotState

if TYPE_CHECKING:
    from .motion import LinearRoutePlan


class VisionSource(Protocol):
    def capture(self) -> Frame:
        """Return one RGB/depth observation with zero or more detections."""

    def close(self) -> None:
        """Release camera resources."""


class MotionBackend(Protocol):
    @property
    def dry_run(self) -> bool:
        """True when moves are recorded/validated but not sent to hardware."""

    def get_state(self) -> RobotState:
        """Return a cached, fresh-enough robot state."""

    def move_to(self, pose: Pose, *, name: str, speed_mps: float) -> None:
        """Move the tool centre point to one Cartesian target."""

    def precheck_linear_targets(
        self, targets: Sequence[Pose], *, step_m: float
    ) -> "LinearRoutePlan":
        """Validate every straight-path waypoint before any motion is sent."""

    def execute_linear_route(
        self,
        route: "LinearRoutePlan",
        *,
        names: Sequence[str],
        speed_mps: float,
    ) -> None:
        """Execute an already pre-checked sequence of straight TCP segments."""

    def close(self) -> None:
        """Release subscriptions/connections."""


class GripperBackend(Protocol):
    @property
    def state(self) -> GripperState:
        """Last known jaw state."""

    def open(self) -> None:
        """Command the jaws to open."""

    def close(self) -> None:
        """Command the jaws to close."""

    def close_backend(self) -> None:
        """Release serial/socket resources."""


class HoldVerifier(Protocol):
    def verify(self, detection: Detection, plan: MotionPlan) -> tuple[bool, str]:
        """Return whether the object is still held after closing the jaws.""" 
