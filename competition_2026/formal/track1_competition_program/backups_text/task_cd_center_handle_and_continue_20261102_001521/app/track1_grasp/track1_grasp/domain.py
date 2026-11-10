"""Hardware-independent data types for the whole Track 1 grasping challenge.

Nothing in this module imports ROS, a camera SDK or a robot SDK.  The same
objects are used by the offline mock, the unit tests and the eventual real
adapters, which keeps the whole challenge testable before any hardware is
mounted.

Track 1 (赛道一 · 机器人自主抓取挑战) is made of five sub-tasks:

* ``A`` detection + annotation demo, scored by IoU (vision only, no motion).
* ``B`` one object picked and placed at five bin positions (four corners plus
  the centre).
* ``C`` unordered bin clearing over the known object set.
* ``D`` unordered bin clearing that also contains unseen objects.
* ``E`` bin clearing in a predefined order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, pi
from typing import Any, Mapping


#: Ordered inner-bottom box corners.  NW->NE is box-local +X and
#: NE->SE is box-local -Y; equivalently SW->NW is +Y.
BOX_CORNER_NAMES = ("corner_nw", "corner_ne", "corner_se", "corner_sw")

#: Safety-ordered object-relative grasp points used by Task B.
BOX_POINT_NAMES = (
    "corner_nw",
    "corner_ne",
    "corner_se",
    "corner_sw",
    "center",
)


class TaskMode(str, Enum):
    """The five sub-tasks of Track 1."""

    A = "A"  # detection + annotation demo (IoU scored)
    B = "B"  # one object pick&place at five bin positions
    C = "C"  # unordered, known target set
    D = "D"  # unordered, includes previously unseen objects
    E = "E"  # ordered target list

    @property
    def needs_motion(self) -> bool:
        """Whether the task may command the arm.

        Task A is still usable as a vision-only demo when patrol is disabled,
        but its competition workflow now supports a ten-position arm patrol.
        """

        return True


class TaskPhase(str, Enum):
    """Observable phases shared by the vision demo and the pick pipelines."""

    IDLE = "idle"
    SCAN = "scan"
    DETECT = "detect"
    ANNOTATE = "annotate"
    EVALUATE = "evaluate"
    SELECT = "select"
    PLAN = "plan"
    OPEN_GRIPPER = "open_gripper"
    APPROACH = "approach"
    GRASP = "grasp"
    VERIFY_HOLD = "verify_hold"
    LIFT = "lift"
    TRANSFER = "transfer"
    RELEASE = "release"
    RETREAT = "retreat"
    REPORT = "report"
    DONE = "done"
    ABORT = "abort"


#: Historical name kept so earlier C/D/E code keeps importing ``PickPhase``.
PickPhase = TaskPhase


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
    the tool Z axis downwards, the safe initial convention for this framework.
    A real calibration can override it per target.
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
    and a depth value.  ``base_xyz`` is useful in mock mode and for a fixed
    camera that publishes an already-transformed point.  Real deployments
    should normally use ``camera_xyz`` plus the calibrated camera->base
    transform.
    """

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    depth_m: float | None = None
    camera_xyz: tuple[float, float, float] | None = None
    base_xyz: tuple[float, float, float] | None = None
    #: Robust object-surface centre computed from the RGB-D point cloud.  The
    #: adaptive inverse path uses this instead of the detector's single-pixel
    #: ``camera_xyz`` anchor so box jitter cannot move the grasp column.
    object_center_camera_xyz: tuple[float, float, float] | None = None
    #: Principal/minor axes in the camera optical frame.  The major axis is
    #: used only to propose grasp headings; the planner still validates every
    #: candidate through full-route IK before motion.
    major_axis_camera: tuple[float, float, float] | None = None
    minor_axis_camera: tuple[float, float, float] | None = None
    major_axis_length_m: float | None = None
    minor_axis_length_m: float | None = None
    pose_confidence: float | None = None
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
        for value in (
            self.camera_xyz,
            self.base_xyz,
            self.object_center_camera_xyz,
            self.major_axis_camera,
            self.minor_axis_camera,
        ):
            if value is not None and len(value) != 3:
                raise ValueError("Detection xyz fields must be triples")
        if self.major_axis_camera is not None and not any(
            float(value) != 0.0 for value in self.major_axis_camera
        ):
            raise ValueError("Detection.major_axis_camera must not be zero")
        if self.minor_axis_camera is not None and not any(
            float(value) != 0.0 for value in self.minor_axis_camera
        ):
            raise ValueError("Detection.minor_axis_camera must not be zero")
        for value in (self.major_axis_length_m, self.minor_axis_length_m):
            if value is not None and (not isfinite(value) or value <= 0.0):
                raise ValueError("Detection axis lengths must be positive")
        if self.pose_confidence is not None and not 0.0 <= self.pose_confidence <= 1.0:
            raise ValueError("Detection.pose_confidence must be in [0, 1]")

    def score(self) -> float:
        """A small deterministic ranking score used before motion planning."""

        return float(self.confidence) + 0.20 * float(self.graspability)

    def as_annotation(self) -> "BoxAnnotation":
        return BoxAnnotation(self.label, self.bbox, self.confidence)


@dataclass(frozen=True)
class BoxAnnotation:
    """A labelled box in pixel coordinates ``(x0, y0, x1, y1)``.

    Used both for the ground truth loaded from a file and for the predictions
    drawn on the annotated image, so IoU matching compares like with like.
    """

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    instance_id: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("BoxAnnotation.label must not be empty")
        if len(self.bbox) != 4:
            raise ValueError("BoxAnnotation.bbox must be (x0, y0, x1, y1)")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("BoxAnnotation.confidence must be in [0, 1]")

    def as_detection(self, **extra: Any) -> Detection:
        return Detection(
            label=self.label,
            confidence=self.confidence,
            bbox=self.bbox,
            instance_id=self.instance_id,
            **extra,
        )


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole RGB-D intrinsics in pixels for one captured frame."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera intrinsics width/height must be positive")
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class BoxPoseEstimate:
    """Auditable box pose fitted from one frame's ordered four corners.

    All XYZ and axis values are unit metres.  ``yaw_rad`` is the base-frame
    angle from base +X to the fitted box-local +X axis.  The estimate is only
    a geometry result; callers must additionally enforce their configured
    quality thresholds before allowing planning.
    """

    frame_id: str
    corner_names: tuple[str, ...]
    center_camera_xyz_m: tuple[float, float, float]
    center_base_xyz_m: tuple[float, float, float]
    yaw_rad: float
    x_axis_base: tuple[float, float, float]
    y_axis_base: tuple[float, float, float]
    normal_base: tuple[float, float, float]
    width_m: float
    depth_m: float
    plane_rms_m: float
    width_error_m: float
    depth_error_m: float
    orthogonality_error_deg: float
    reprojection_rms_px: float
    corner_depth_std_m: float
    normal_z: float
    quality_score: float
    #: Per-corner audit data retained from the exact frame used for the fit.
    corner_camera_xyz_m: tuple[
        tuple[float, float, float], ...
    ] = ()
    corner_base_xyz_m: tuple[tuple[float, float, float], ...] = ()

    @property
    def center_base_xy_m(self) -> tuple[float, float]:
        return (float(self.center_base_xyz_m[0]), float(self.center_base_xyz_m[1]))


@dataclass(frozen=True)
class BoxPoint:
    """One fixed grasp point in the fitted box-local XY frame."""

    name: str
    local_xy_m: tuple[float, float]


@dataclass(frozen=True)
class BoxPointLayout:
    """The immutable five-point layout generated for one box pose."""

    points: tuple[BoxPoint, ...]
    gripper_half_extent_m: float
    wall_clearance_m: float

    def __post_init__(self) -> None:
        names = tuple(point.name for point in self.points)
        if names != BOX_POINT_NAMES:
            raise ValueError(
                "box point layout must contain the ordered five points "
                f"{BOX_POINT_NAMES}, got {names}"
            )
        if self.gripper_half_extent_m <= 0.0:
            raise ValueError("gripper half extent must be positive")
        if self.wall_clearance_m < 0.0:
            raise ValueError("wall clearance must be >= 0")

    def by_name(self, name: str) -> BoxPoint:
        for point in self.points:
            if point.name == name:
                return point
        raise KeyError(name)


@dataclass(frozen=True)
class Frame:
    """A camera observation at one instant."""

    frame_id: str
    detections: tuple[Detection, ...]
    stamp_s: float = 0.0
    #: Optional ground truth for task A scoring, loaded from a sibling file.
    annotations: tuple[BoxAnnotation, ...] = ()
    #: Optional path of the frame's RGB image once real vision is wired in.
    image_path: str = ""
    #: Optional in-memory BGR image.  It is deliberately excluded from dataclass
    #: equality/hash operations because NumPy arrays do not provide a scalar
    #: truth value and can be large.
    image: Any = field(default=None, repr=False, compare=False)
    #: Full-frame RGB-D metadata.  These trailing fields preserve positional
    #: compatibility with older Frame construction sites.
    camera_intrinsics: CameraIntrinsics | None = None
    depth_m: Any = field(default=None, repr=False, compare=False)
    #: Raw ordered camera-frame corners and their pixel observations.  A box
    #: pose must be fitted from these values in the same frame; callers must
    #: not substitute a remembered or configured box centre.
    box_corner_camera_xyz_m: tuple[
        tuple[float, float, float], ...
    ] = ()
    box_corner_uv_px: tuple[tuple[float, float], ...] = ()
    #: Filled only after a successful fit from this frame.
    box_pose_estimate: BoxPoseEstimate | None = None
    #: Robot FK captured before/after the RGB-D frame and verified stable.
    capture_robot_state: "CaptureRobotState | None" = field(
        default=None,
        repr=False,
        compare=False,
    )


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
    #: Measured absolute joint positions in radians, Joint1..Joint6.
    joints: tuple[float, ...] | None = None
    #: Raw controller state and fault word retained for actionable diagnostics.
    controller_state: str = ""
    fault_flag: int = 0


@dataclass(frozen=True)
class CaptureRobotState:
    """Fresh robot feedback captured immediately before and after one frame."""

    before: RobotState
    after: RobotState


@dataclass(frozen=True)
class MotionPlan:
    target: Detection
    grasp_pose: Pose
    pregrasp_pose: Pose
    lift_pose: Pose
    transfer_pose: Pose | None = None
    release_pose: Pose | None = None
    retreat_pose: Pose | None = None
    #: Name of the configured place slot this plan targets (task B / C/D/E).
    place_name: str = ""

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
    phase: TaskPhase
    message: str


@dataclass(frozen=True)
class PickResult:
    status: PickStatus
    target: Detection | None
    message: str
    events: tuple[PipelineEvent, ...] = ()
    place_name: str = ""

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


@dataclass(frozen=True)
class IouMatch:
    """One detection matched (or not) to one ground-truth annotation."""

    detection: "Detection | None"
    annotation: BoxAnnotation | None
    iou: float
    passed: bool

    @property
    def label(self) -> str:
        if self.detection is not None:
            return self.detection.label
        if self.annotation is not None:
            return self.annotation.label
        return "-"


@dataclass(frozen=True)
class VisionReport:
    """Result of a task A detection/annotation/demo run."""

    frames: int
    detections: int
    matched: int
    passed: int
    total_annotations: int
    results: tuple[IouMatch, ...] = ()
    iou_threshold: float = 0.75
    #: Ten commanded TCP targets when task A patrol mode is enabled.
    patrol_targets: tuple[Pose, ...] = ()
    #: Straight-line TCP distance for each commanded patrol segment.
    patrol_distances_m: tuple[float, ...] = ()
    #: Absolute six-joint targets in degrees for the joint-arc patrol.
    patrol_joint_targets_deg: tuple[tuple[float, ...], ...] = ()
    #: One-based logical arc number for each patrol position.
    patrol_arcs: tuple[int, ...] = ()
    #: One-based patrol positions where the configured target was confirmed.
    patrol_visible_positions: tuple[int, ...] = ()
    #: Sorted task-A class labels observed across the patrol positions.
    patrol_detected_classes: tuple[str, ...] = ()
    #: Configured task-A classes not observed anywhere during the patrol.
    patrol_missing_classes: tuple[str, ...] = ()

    @property
    def pass_rate(self) -> float:
        if self.total_annotations == 0:
            return 0.0
        return self.passed / self.total_annotations

    @property
    def mean_iou(self) -> float:
        if not self.results:
            return 0.0
        return sum(match.iou for match in self.results) / len(self.results)
