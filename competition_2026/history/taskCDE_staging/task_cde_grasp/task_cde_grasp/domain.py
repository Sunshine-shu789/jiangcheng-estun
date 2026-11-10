"""Hardware-independent data types for the CDE pick pipeline.

Nothing in this module imports ROS, a camera SDK or a robot SDK.  The same
objects are used by the offline mock, unit tests and the eventual real
adapters, which makes the state machine testable before hardware is mounted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pi
from typing import Any, Mapping


class TaskMode(str, Enum):
    """Competition task variants.

    C: unordered, known target set
    D: unordered, includes previously unseen objects
    E: ordered target list
    """

    C = "C"
    D = "D"
    E = "E"


class PickPhase(str, Enum):
    """Observable state-machine phases."""

    IDLE = "idle"
    OPEN_GRIPPER = "open_gripper"
    SCAN = "scan"
    DETECT = "detect"
    SELECT = "select"
    PLAN = "plan"
    APPROACH = "approach"
    GRASP = "grasp"
    VERIFY_HOLD = "verify_hold"
    LIFT = "lift"
    TRANSFER = "transfer"
    RELEASE = "release"
    RETREAT = "retreat"
    DONE = "done"
    ABORT = "abort"


class GripperState(str, Enum):
    UNKNOWN = "unknown"
    OPEN = "open"
    CLOSED = "closed"
    MOVING = "moving"
    FAULT = "fault"


class PickStatus(str, Enum):
    SUCCESS = "success"
    NO_TARGET = "no_target"
    REJECTED = "rejected"
    ABORTED = "aborted"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class Pose:
    """A Cartesian pose in the robot base frame.

    ``xyz`` is metres and ``rpy`` is radians.  The default orientation points
    the tool Z axis downwards, which is the safe initial convention for the
    framework.  A real calibration can override it per target.
    """

    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float] = (pi, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.xyz) != 3 or len(self.rpy) != 3:
            raise ValueError("Pose needs xyz and rpy triples")

    def as_list(self) -> list[float]:
        return [float(v) for v in (*self.xyz, *self.rpy)]


@dataclass(frozen=True)
class Detection:
    """One object observation produced by a detector.

    ``camera_xyz`` is optional because some detectors only provide a 2D box
    and depth.  ``base_xyz`` is useful in mock mode and for a fixed-camera
    system that publishes a pre-transformed point.  Real deployments should
    normally use ``camera_xyz`` plus the calibrated camera-to-base transform.
    """

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    depth_m: float | None = None
    camera_xyz: tuple[float, float, float] | None = None
    base_xyz: tuple[float, float, float] | None = None
    graspability: float = 1.0
    instance_id: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Detection.label must not be empty")
        if len(self.bbox) != 4:
            raise ValueError("Detection.bbox must be (x0, y0, x1, y1)")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection.confidence must be in [0, 1]")
        if not 0.0 <= self.graspability <= 1.0:
            raise ValueError("Detection.graspability must be in [0, 1]")
        if self.depth_m is not None and self.depth_m <= 0.0:
            raise ValueError("Detection.depth_m must be positive when supplied")
        for value in (self.camera_xyz, self.base_xyz):
            if value is not None and len(value) != 3:
                raise ValueError("Detection xyz fields must be triples")

    def score(self) -> float:
        """A small deterministic ranking score used before motion planning."""
        return float(self.confidence) + 0.20 * float(self.graspability)


@dataclass(frozen=True)
class Frame:
    """A camera observation at one instant."""

    frame_id: str
    detections: tuple[Detection, ...]
    stamp_s: float = 0.0


@dataclass(frozen=True)
class Selection:
    """Result of applying the task-specific target policy."""

    detection: Detection | None
    accepted: bool
    reason: str
    priority: int = 0


@dataclass(frozen=True)
class RobotState:
    ready: bool
    pose: Pose | None = None
    fault: bool = False
    message: str = ""


@dataclass(frozen=True)
class MotionPlan:
    target: Detection
    grasp_pose: Pose
    pregrasp_pose: Pose
    lift_pose: Pose
    transfer_pose: Pose | None = None
    release_pose: Pose | None = None
    retreat_pose: Pose | None = None

    def ordered_waypoints(self) -> tuple[tuple[str, Pose], ...]:
        waypoints: list[tuple[str, Pose]] = [
            ("pregrasp", self.pregrasp_pose),
            ("grasp", self.grasp_pose),
            ("lift", self.lift_pose),
        ]
        if self.transfer_pose is not None:
            waypoints.append(("transfer", self.transfer_pose))
        if self.release_pose is not None:
            waypoints.append(("release", self.release_pose))
        if self.retreat_pose is not None:
            waypoints.append(("retreat", self.retreat_pose))
        return tuple(waypoints)


@dataclass(frozen=True)
class PipelineEvent:
    phase: PickPhase
    message: str


@dataclass(frozen=True)
class PickResult:
    status: PickStatus
    target: Detection | None
    message: str
    events: tuple[PipelineEvent, ...] = ()

    @property
    def success(self) -> bool:
        return self.status == PickStatus.SUCCESS


@dataclass(frozen=True)
class RunReport:
    requested_picks: int
    attempts: int
    successes: int
    results: tuple[PickResult, ...]

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts
