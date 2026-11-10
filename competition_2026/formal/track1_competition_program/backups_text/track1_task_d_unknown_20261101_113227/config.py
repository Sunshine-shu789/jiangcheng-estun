"""YAML configuration for the hardware-agnostic Track 1 framework.

One file describes the whole challenge.  The ``task.mode`` field selects the
sub-task (A/B/C/D/E) and the matching sections (``task_a``, ``task_b``) pick
up their own parameters; unset sections simply fall back to the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import hypot, isfinite, radians
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .collision import (
    GEOMETRY_SOURCE_FIELD_MEASURED,
    GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC,
    BoxClearanceConfig,
    validate_box_clearance_geometry,
)
from .domain import TaskMode
from .kinematics import joints_within_limits, rpy_to_matrix


def _triple(value: Sequence[float] | None, default: Sequence[float]) -> tuple[float, float, float]:
    source = default if value is None else value
    if len(source) != 3:
        raise ValueError(f"expected a 3-value vector, got {source!r}")
    return tuple(float(v) for v in source)  # type: ignore[return-value]


def _pair(value: Sequence[float] | None) -> tuple[float, float]:
    if (
        value is None
        or isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"expected a 2-value vector, got {value!r}")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _joint_vector_deg(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"expected a six-value joint vector, got {value!r}")
    if len(value) != 6:
        raise ValueError(f"expected a six-value joint vector, got {value!r}")
    return tuple(float(v) for v in value)


def _pose_value(value: Sequence[float] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    if len(value) != 6:
        raise ValueError(f"expected [x, y, z, rx, ry, rz], got {value!r}")
    return tuple(float(v) for v in value)


def _pose_item(item: Any) -> tuple[float, ...] | None:
    """Read one pose entry: ``[x, y, z, rx, ry, rz]`` or ``{pose: [...]}``."""

    if isinstance(item, Mapping):
        item = item.get("pose", item.get("xyz_rpy", item.get("xyz")))
    return _pose_value(item)


def _pose_list(value: Sequence[Any] | None) -> tuple[tuple[float, ...], ...]:
    if not value:
        return ()
    return tuple(pose for pose in (_pose_item(item) for item in value) if pose is not None)


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _str_tuple(value: Sequence[Any] | None, default: Sequence[str] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    return tuple(str(item) for item in value)


def _float_tuple(value: Sequence[Any] | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"expected a numeric sequence, got {value!r}")
    return tuple(float(item) for item in value)


def _int_tuple(value: Sequence[Any] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"expected an integer sequence, got {value!r}")
    return tuple(int(item) for item in value)


def _float_rows(value: Sequence[Any] | None) -> tuple[tuple[float, ...], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"expected a sequence of numeric rows, got {value!r}")
    rows: list[tuple[float, ...]] = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"expected a numeric row, got {row!r}")
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)


def _bool_tuple(value: Sequence[Any] | None) -> tuple[bool, ...]:
    if not value:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"expected a boolean sequence, got {value!r}")
    result: list[bool] = []
    for item in value:
        if not isinstance(item, bool):
            raise ValueError(f"expected boolean entries, got {value!r}")
        result.append(item)
    return tuple(result)


def _load_extrinsics_metadata(path: str) -> tuple[bool, str]:
    """Read the safety metadata attached to an extrinsics document."""

    if not path:
        return False, ""
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"extrinsics document must be a mapping: {path}")
    return bool(payload.get("validated", False)), str(
        payload.get("calibration_id", "")
    ).strip()


@dataclass(frozen=True)
class TaskConfig:
    mode: TaskMode = TaskMode.C
    max_picks: int = 20
    stop_on_error: bool = False
    sample_period_s: float = 0.5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TaskConfig":
        data = dict(data or {})
        try:
            mode = TaskMode(str(data.get("mode", "C")).upper())
        except ValueError as exc:
            raise ValueError("task.mode must be A, B, C, D or E") from exc
        return cls(
            mode=mode,
            max_picks=int(data.get("max_picks", 20)),
            stop_on_error=bool(data.get("stop_on_error", False)),
            sample_period_s=float(data.get("sample_period_s", 0.5)),
        )


@dataclass(frozen=True)
class UnknownObjectProposerConfig:
    """Conservative class-agnostic RGB-D proposals for Task D.

    Enabling this configuration is not enough to accept a proposal: Task D's
    target policy still requires a minimum graspability, and the normal planner
    then applies workspace, IK, collision and route checks.
    """

    enabled: bool = False
    label: str = "unknown"
    min_depth_m: float = 0.08
    max_depth_m: float = 1.20
    plane_max_samples: int = 18000
    plane_ransac_iterations: int = 128
    plane_ransac_threshold_m: float = 0.003
    plane_min_inliers: int = 300
    plane_min_inlier_ratio: float = 0.22
    plane_max_rms_m: float = 0.006
    plane_min_normal_z: float = 0.50
    min_object_height_m: float = 0.008
    max_object_height_m: float = 0.35
    min_area_px: int = 400
    max_area_ratio: float = 0.60
    min_component_width_px: int = 12
    min_component_height_px: int = 8
    min_valid_ratio: float = 0.35
    min_depth_points: int = 120
    morph_open_px: int = 3
    morph_close_px: int = 5
    edge_margin_px: int = 2
    max_proposals: int = 8
    known_overlap_iou: float = 0.25
    known_containment: float = 0.70
    min_pose_confidence: float = 0.55
    graspability_scale: float = 0.90
    seed: int = 20261101

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
    ) -> "UnknownObjectProposerConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            label=str(data.get("label", "unknown")).strip(),
            min_depth_m=float(data.get("min_depth_m", 0.08)),
            max_depth_m=float(data.get("max_depth_m", 1.20)),
            plane_max_samples=int(data.get("plane_max_samples", 18000)),
            plane_ransac_iterations=int(
                data.get("plane_ransac_iterations", 128)
            ),
            plane_ransac_threshold_m=float(
                data.get("plane_ransac_threshold_m", 0.003)
            ),
            plane_min_inliers=int(data.get("plane_min_inliers", 300)),
            plane_min_inlier_ratio=float(
                data.get("plane_min_inlier_ratio", 0.22)
            ),
            plane_max_rms_m=float(data.get("plane_max_rms_m", 0.006)),
            plane_min_normal_z=float(data.get("plane_min_normal_z", 0.50)),
            min_object_height_m=float(
                data.get("min_object_height_m", 0.008)
            ),
            max_object_height_m=float(
                data.get("max_object_height_m", 0.35)
            ),
            min_area_px=int(data.get("min_area_px", 400)),
            max_area_ratio=float(data.get("max_area_ratio", 0.60)),
            min_component_width_px=int(
                data.get("min_component_width_px", 12)
            ),
            min_component_height_px=int(
                data.get("min_component_height_px", 8)
            ),
            min_valid_ratio=float(data.get("min_valid_ratio", 0.35)),
            min_depth_points=int(data.get("min_depth_points", 120)),
            morph_open_px=int(data.get("morph_open_px", 3)),
            morph_close_px=int(data.get("morph_close_px", 5)),
            edge_margin_px=int(data.get("edge_margin_px", 2)),
            max_proposals=int(data.get("max_proposals", 8)),
            known_overlap_iou=float(data.get("known_overlap_iou", 0.25)),
            known_containment=float(data.get("known_containment", 0.70)),
            min_pose_confidence=float(
                data.get("min_pose_confidence", 0.55)
            ),
            graspability_scale=float(data.get("graspability_scale", 0.90)),
            seed=int(data.get("seed", 20261101)),
        )

    def validate(self) -> None:
        if not self.label:
            raise ValueError("vision.unknown_proposer.label must not be empty")
        if (
            not isfinite(self.min_depth_m)
            or not isfinite(self.max_depth_m)
            or self.min_depth_m <= 0.0
            or self.max_depth_m <= self.min_depth_m
        ):
            raise ValueError(
                "vision.unknown_proposer depth range must be positive, finite "
                "and increasing"
            )
        if self.plane_max_samples < 3:
            raise ValueError(
                "vision.unknown_proposer.plane_max_samples must be >= 3"
            )
        if self.plane_ransac_iterations < 1:
            raise ValueError(
                "vision.unknown_proposer.plane_ransac_iterations must be >= 1"
            )
        if (
            not isfinite(self.plane_ransac_threshold_m)
            or self.plane_ransac_threshold_m <= 0.0
        ):
            raise ValueError(
                "vision.unknown_proposer.plane_ransac_threshold_m must be "
                "positive"
            )
        if self.plane_min_inliers < 3:
            raise ValueError(
                "vision.unknown_proposer.plane_min_inliers must be >= 3"
            )
        for name in (
            "plane_min_inlier_ratio",
            "plane_min_normal_z",
            "max_area_ratio",
            "min_valid_ratio",
            "known_overlap_iou",
            "known_containment",
            "min_pose_confidence",
            "graspability_scale",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"vision.unknown_proposer.{name} must be in [0, 1]"
                )
        if self.plane_min_normal_z <= 0.0:
            raise ValueError(
                "vision.unknown_proposer.plane_min_normal_z must be > 0"
            )
        if (
            not isfinite(self.plane_max_rms_m)
            or self.plane_max_rms_m <= 0.0
        ):
            raise ValueError(
                "vision.unknown_proposer.plane_max_rms_m must be positive"
            )
        if (
            not isfinite(self.min_object_height_m)
            or not isfinite(self.max_object_height_m)
            or self.min_object_height_m <= 0.0
            or self.max_object_height_m <= self.min_object_height_m
        ):
            raise ValueError(
                "vision.unknown_proposer object-height range must be positive, "
                "finite and increasing"
            )
        if self.min_area_px < 1 or self.min_depth_points < 3:
            raise ValueError(
                "vision.unknown_proposer area and depth-point thresholds must "
                "be positive"
            )
        if (
            self.min_component_width_px < 1
            or self.min_component_height_px < 1
            or self.edge_margin_px < 0
            or self.max_proposals < 1
        ):
            raise ValueError(
                "vision.unknown_proposer component, edge and proposal limits "
                "are invalid"
            )
        if self.morph_open_px < 0 or self.morph_close_px < 0:
            raise ValueError(
                "vision.unknown_proposer morphology sizes must be >= 0"
            )


@dataclass(frozen=True)
class VisionConfig:
    #: mock | taska_realsense | realsense_yolo | legacy_http
    backend: str = "mock"
    scenario_file: str = ""
    taska_config: str = ""
    #: Read-only adapter for the legacy Task A browser recognizer.  Task A
    #: must use that program instead of the newer grab-recognition runtime.
    legacy_status_url: str = "http://127.0.0.1:9100/status.json"
    legacy_poll_s: float = 0.10
    min_confidence: float = 0.35
    require_depth: bool = False
    loop_scenario: bool = False
    camera_frame_id: str = "camera_color_optical_frame"

    #: Direct RealSense + YOLO backend.  These fields deliberately mirror the
    #: proven arm_visual_probe settings instead of depending on taskA's live
    #: configuration file.
    weights: str = ""
    #: Optional required SHA256 for the YOLO weight file.  When set, the
    #: hash is checked before the model is passed to Ultralytics.
    weights_sha256: str = ""
    width: int = 640
    height: int = 480
    fps: int = 15
    serial: str = ""
    warmup_frames: int = 20
    imgsz: int = 640
    device: str = "cpu"
    target_class: str = ""
    classes: tuple[str, ...] = ()
    depth_window: int = 5
    anchor: str = "center"  # center | top_center | bottom_center
    rotate_180: bool = False
    output_dir: str = ""
    save_annotated: bool = False
    save_interval_s: float = 2.0
    show: bool = False
    #: Keep a real camera/YOLO source running in a background thread.  Mock
    #: sources stay synchronous so scripted unit tests remain deterministic.
    continuous: bool = True
    window_name: str = "Track 1 - D405 + YOLO"
    preview_fps: float = 30.0
    capture_timeout_s: float = 5.0
    #: The first live frame includes lazy camera/model startup and can be much
    #: slower than steady-state inference on the competition PC.
    first_frame_timeout_s: float = 120.0
    stale_after_s: float = 2.0

    #: Lightweight temporal stabilisation for the live YOLO path.  Raw YOLO is
    #: intentionally allowed to run at a lower threshold; the stabiliser is
    #: responsible for confirming new tracks and suppressing one-frame noise.
    stabilize: bool = True
    stabilizer_inference_min_confidence: float = 0.10
    stabilizer_confirm_hits: int = 3
    stabilizer_max_misses: int = 3
    stabilizer_match_iou: float = 0.20
    stabilizer_confidence_alpha: float = 0.25
    stabilizer_box_alpha: float = 0.55
    stabilizer_depth_alpha: float = 0.25
    stabilizer_duplicate_iou: float = 0.55
    stabilizer_containment_ratio: float = 0.90
    stabilizer_cross_class_iou: float = 0.40
    stabilizer_cross_class_containment: float = 0.60
    stabilizer_cross_class_conf_ratio: float = 1.50
    stabilizer_max_per_class: int = 1
    #: Detector labels that represent physical instances rather than a class.
    #: ``max_per_class`` is intentionally bypassed for these labels while IoU
    #: and containment de-duplication remain active.
    stabilizer_instance_tracking_labels: tuple[str, ...] = (
        "unknown",
        "object",
    )
    stabilizer_class_switch_hits: int = 3
    stabilizer_edge_margin_px: int = 8
    stabilizer_edge_suppress_confidence: float = 0.55

    #: Task D-only class-agnostic RGB-D proposals.  The task-mode gate in
    #: :meth:`for_task_mode` enables this for D and disables it for every other
    #: task regardless of the YAML default.
    unknown_proposer: UnknownObjectProposerConfig = field(
        default_factory=UnknownObjectProposerConfig
    )

    #: Optional per-task mock scenarios, e.g. {"A": "mock_task_a_scene.yaml"}.
    scenario_files: Mapping[str, str] = field(default_factory=dict)

    def accepted_target_labels(self) -> tuple[str, ...]:
        """Return the detector labels allowed to confirm the configured target.

        A non-empty ``target_class`` pins one exact object label for tasks that
        follow one item.  When it is empty, ``classes`` is the allow-list; Task
        A uses this mode to recognize all ten competition classes.
        """

        target_class = self.target_class.strip()
        if target_class:
            return (target_class,)
        return tuple(
            label.strip() for label in self.classes if label.strip()
        )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "VisionConfig":
        data = dict(data or {})
        target_class = str(data.get("target_class", ""))
        configured_classes = _str_tuple(data.get("classes"))
        if target_class and not configured_classes:
            configured_classes = (target_class,)
        stabilizer = dict(data.get("stabilizer", {}) or {})
        return cls(
            backend=str(data.get("backend", "mock")),
            scenario_file=_resolve_path(str(data.get("scenario_file", "")), base_dir),
            taska_config=_resolve_path(str(data.get("taska_config", "")), base_dir),
            legacy_status_url=str(
                data.get(
                    "legacy_status_url",
                    "http://127.0.0.1:9100/status.json",
                )
            ),
            legacy_poll_s=float(data.get("legacy_poll_s", 0.10)),
            min_confidence=float(data.get("min_confidence", 0.35)),
            require_depth=bool(data.get("require_depth", False)),
            loop_scenario=bool(data.get("loop_scenario", False)),
            camera_frame_id=str(data.get("camera_frame_id", "camera_color_optical_frame")),
            weights=_resolve_path(str(data.get("weights", "")), base_dir),
            weights_sha256=str(data.get("weights_sha256", "")).strip().lower(),
            width=int(data.get("width", 640)),
            height=int(data.get("height", 480)),
            fps=int(data.get("fps", 15)),
            serial=str(data.get("serial", "")),
            warmup_frames=int(data.get("warmup_frames", 20)),
            imgsz=int(data.get("imgsz", 640)),
            device=str(data.get("device", "cpu")),
            target_class=target_class,
            classes=configured_classes,
            depth_window=int(data.get("depth_window", 5)),
            anchor=str(data.get("anchor", "center")),
            rotate_180=bool(data.get("rotate_180", False)),
            output_dir=_resolve_path(str(data.get("output_dir", "")), base_dir),
            save_annotated=bool(data.get("save_annotated", False)),
            save_interval_s=float(data.get("save_interval_s", 2.0)),
            show=bool(data.get("show", False)),
            continuous=bool(data.get("continuous", True)),
            window_name=str(data.get("window_name", "Track 1 - D405 + YOLO")),
            preview_fps=float(data.get("preview_fps", 30.0)),
            capture_timeout_s=float(data.get("capture_timeout_s", 5.0)),
            first_frame_timeout_s=float(data.get("first_frame_timeout_s", 120.0)),
            stale_after_s=float(data.get("stale_after_s", 2.0)),
            stabilize=bool(data.get("stabilize", stabilizer.get("enabled", True))),
            stabilizer_inference_min_confidence=float(
                stabilizer.get("inference_min_confidence", 0.10)
            ),
            stabilizer_confirm_hits=int(stabilizer.get("confirm_hits", 3)),
            stabilizer_max_misses=int(stabilizer.get("max_misses", 3)),
            stabilizer_match_iou=float(stabilizer.get("match_iou", 0.20)),
            stabilizer_confidence_alpha=float(
                stabilizer.get("confidence_alpha", 0.25)
            ),
            stabilizer_box_alpha=float(stabilizer.get("box_alpha", 0.55)),
            stabilizer_depth_alpha=float(stabilizer.get("depth_alpha", 0.25)),
            stabilizer_duplicate_iou=float(
                stabilizer.get("duplicate_iou", 0.55)
            ),
            stabilizer_containment_ratio=float(
                stabilizer.get("containment_ratio", 0.90)
            ),
            stabilizer_cross_class_iou=float(
                stabilizer.get("cross_class_iou", 0.40)
            ),
            stabilizer_cross_class_containment=float(
                stabilizer.get("cross_class_containment", 0.60)
            ),
            stabilizer_cross_class_conf_ratio=float(
                stabilizer.get("cross_class_conf_ratio", 1.50)
            ),
            stabilizer_max_per_class=int(stabilizer.get("max_per_class", 1)),
            stabilizer_instance_tracking_labels=_str_tuple(
                stabilizer.get("instance_tracking_labels"),
                ("unknown", "object"),
            ),
            stabilizer_class_switch_hits=int(
                stabilizer.get("class_switch_hits", 3)
            ),
            stabilizer_edge_margin_px=int(
                stabilizer.get("edge_margin_px", 8)
            ),
            stabilizer_edge_suppress_confidence=float(
                stabilizer.get("edge_suppress_confidence", 0.55)
            ),
            unknown_proposer=UnknownObjectProposerConfig.from_dict(
                data.get("unknown_proposer")
            ),
            scenario_files={
                str(key).upper(): _resolve_path(str(value), base_dir)
                for key, value in dict(data.get("scenario_files", {}) or {}).items()
            },
        )

    def scenario_for(self, mode: "TaskMode | str") -> str:
        """Scenario file to use for a task, falling back to scenario_file."""

        key = mode.value if isinstance(mode, TaskMode) else str(mode).upper()
        return self.scenario_files.get(key, self.scenario_file)

    def for_task_mode(self, mode: "TaskMode | str") -> "VisionConfig":
        """Return a copy with task-specific detectors enabled or disabled."""

        key = mode.value if isinstance(mode, TaskMode) else str(mode).upper()
        return replace(
            self,
            unknown_proposer=replace(
                self.unknown_proposer,
                enabled=key == TaskMode.D.value,
            ),
        )


@dataclass(frozen=True)
class ObjectVisibilityConfig:
    """Per-position target observation used by the task A patrol.

    The arm stops at every position and observes the target for a bounded
    window.  Consecutive detections confirm that the target is present at that
    position.  If the target is not confirmed before the window expires, the
    patrol records that position as unseen and safely advances to the next
    one.  This is a search patrol, so an unseen position is not an error.
    """

    enabled: bool = False
    visible_confirm_frames: int = 3
    observation_window_s: float = 3.0
    poll_s: float = 0.10

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ObjectVisibilityConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            visible_confirm_frames=int(data.get("visible_confirm_frames", 3)),
            observation_window_s=float(
                data.get(
                    "observation_window_s",
                    data.get("max_wait_visible_s", 3.0),
                )
            ),
            poll_s=float(data.get("poll_s", 0.10)),
        )


@dataclass(frozen=True)
class TaskAPatrolConfig:
    """Ten-position Task A patrol.

    ``joint_arc`` is the gripper-safe route.  Task A moves J1/J2/J3/J4 while
    J5/J6 are copied exactly from startup RobotInfo feedback.  J1 uses the
    measured startup yaw as an anchor and distributes two 270-degree arcs to
    its left and right.  When +/-135 degrees cannot both fit inside the
    controller limits, the constrained side uses all safe available travel and
    the other side receives the remainder.  J2/J3/J4 offsets stay relative to
    startup, so the middle-arm posture is independent of base yaw.

    ``cartesian_box`` is retained only for hardware-free regression tests.  The
    live Task A configuration must use ``joint_arc``.
    """

    enabled: bool = False
    mode: str = "joint_arc"  # joint_arc | cartesian_box
    position_count: int = 10
    #: Optional supervised compatibility mode.  Competition Task A leaves this
    #: false and advances automatically after the recognition dwell.
    manual_step: bool = False
    #: Competition joint patrol.  Entries are one-based ROS joint numbers.
    #: J1-J4 may move; J5 and J6 remain exactly at startup feedback.
    movable_joints: tuple[int, ...] = (1, 2, 3, 4)
    #: Normalized ten-point J1 profile in degrees.  The first five traverse
    #: +135 to -135 on arc 1; the second five traverse -135 to +135 on arc 2.
    #: These values are mapped onto a safe left/right split around startup J1.
    #: If one side cannot provide 135 degrees, it uses all safe available travel
    #: and the opposite side receives the remaining span.
    joint1_relative_targets_deg: tuple[float, ...] = (
        135.0,
        67.5,
        0.0,
        -67.5,
        -135.0,
        -135.0,
        -67.5,
        0.0,
        67.5,
        135.0,
    )
    #: One row per position for J2/J3/J4, relative to startup feedback.
    joint_offsets_deg: tuple[tuple[float, ...], ...] = ()
    #: Logical arc labels (1 or 2) for the ten positions.
    arc_indices: tuple[int, ...] = ()
    joint_max_step_deg: float = 2.0
    max_initial_move_deg: float = 120.0
    max_target_move_deg: float = 120.0
    #: Hold still for this long after an allowed object is confirmed, then
    #: continue automatically to the next patrol position.
    post_detection_dwell_s: float = 3.0
    #: Return to the exact startup joint posture after the tenth observation.
    return_to_start: bool = True
    #: Legacy Cartesian-box fields.
    min_move_distance_m: float = 0.15
    stride_m: float = 0.17
    linear_step_m: float = 0.02
    settle_s: float = 0.5
    axes: tuple[str, ...] = ("x", "y", "z")
    object_visibility: ObjectVisibilityConfig = field(
        default_factory=ObjectVisibilityConfig
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TaskAPatrolConfig":
        data = dict(data or {})
        axes = tuple(
            str(axis).lower()
            for axis in _str_tuple(data.get("axes"), ("x", "y", "z"))
        )
        return cls(
            enabled=bool(data.get("enabled", False)),
            # Historical Task B/C configs contain a legacy Task A patrol
            # section without an explicit mode.  Treat a missing key as the
            # old Cartesian behaviour; the scored Task A configuration always
            # declares ``mode: joint_arc`` explicitly.
            mode=str(data.get("mode", "cartesian_box")).strip().lower(),
            position_count=int(data.get("position_count", 10)),
            manual_step=bool(data.get("manual_step", False)),
            movable_joints=_int_tuple(data.get("movable_joints", (1, 2, 3, 4))),
            joint1_relative_targets_deg=_float_tuple(
                data.get("joint1_relative_targets_deg", (
                    135.0,
                    67.5,
                    0.0,
                    -67.5,
                    -135.0,
                    -135.0,
                    -67.5,
                    0.0,
                    67.5,
                    135.0,
                ))
            ),
            joint_offsets_deg=_float_rows(data.get("joint_offsets_deg")),
            arc_indices=_int_tuple(data.get("arc_indices")),
            joint_max_step_deg=float(data.get("joint_max_step_deg", 2.0)),
            max_initial_move_deg=float(data.get("max_initial_move_deg", 120.0)),
            max_target_move_deg=float(data.get("max_target_move_deg", 120.0)),
            post_detection_dwell_s=float(
                data.get("post_detection_dwell_s", 3.0)
            ),
            return_to_start=bool(data.get("return_to_start", True)),
            min_move_distance_m=float(data.get("min_move_distance_m", 0.15)),
            stride_m=float(data.get("stride_m", 0.17)),
            linear_step_m=float(data.get("linear_step_m", 0.02)),
            settle_s=float(data.get("settle_s", 0.5)),
            axes=axes,
            object_visibility=ObjectVisibilityConfig.from_dict(
                data.get("object_visibility")
            ),
        )


@dataclass(frozen=True)
class TaskAConfig:
    """Task A: patrol, detect, annotate and score IoU from ten positions."""

    iou_threshold: float = 0.75
    ground_truth_file: str = ""
    output_dir: str = ""
    save_annotated: bool = False
    draw_labels: bool = True
    draw_ground_truth: bool = True
    max_frames: int = 0  # 0 = every frame the scenario/camera provides
    min_positions: int = 10  # competition demo: 10 different positions
    patrol: TaskAPatrolConfig = field(default_factory=TaskAPatrolConfig)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "TaskAConfig":
        data = dict(data or {})
        return cls(
            iou_threshold=float(data.get("iou_threshold", 0.75)),
            ground_truth_file=_resolve_path(str(data.get("ground_truth_file", "")), base_dir),
            output_dir=_resolve_path(str(data.get("output_dir", "")), base_dir),
            save_annotated=bool(data.get("save_annotated", False)),
            draw_labels=bool(data.get("draw_labels", True)),
            draw_ground_truth=bool(data.get("draw_ground_truth", True)),
            max_frames=int(data.get("max_frames", 0)),
            min_positions=int(data.get("min_positions", 10)),
            patrol=TaskAPatrolConfig.from_dict(data.get("patrol")),
        )


@dataclass(frozen=True)
class TaskBConfig:
    """Task B: pick five same-class targets and return each to its pick point.

    The default workflow is deliberately an in-place pick-and-return: the
    release pose comes from the observed target, not from a fixed drop slot.
    ``place_poses`` remains as a backwards-compatible metadata source for
    callers that explicitly set ``return_to_pick`` to false.
    """

    place_poses: tuple[tuple[float, ...], ...] = ()
    place_poses_file: str = ""
    #: Captured absolute six-joint arm posture used once before the first
    #: Task B visual search.  An empty path leaves the current pose unchanged.
    initial_pose_file: str = ""
    initial_pose_tolerance_deg: float = 0.5
    rounds: int = 5
    #: Release at the observed grasp location instead of a configured slot.
    return_to_pick: bool = True
    #: Optional class lock.  When empty, the first successful target class is
    #: reused for all remaining rounds.
    target_class: str = ""
    #: Maximum base-frame distance between a detection and the expected point.
    pick_pose_tolerance_m: float = 0.08
    #: Require the current point to be the detection's nearest configured
    #: reference point.  Fixed-point runs can disable this when the camera
    #: detection anchor shifts slightly across adjacent points; the distance
    #: window, class/depth checks and completed-point exclusion remain active.
    require_nearest_reference_point: bool = True
    #: A candidate must be at least this much closer to the current point than
    #: to every previously completed point.  This prevents an object returned
    #: to an old point from triggering the next round when tolerance circles
    #: overlap.
    point_assignment_margin_m: float = 0.01
    #: How long to wait for the target to appear at the next point.  Zero waits
    #: until the task is interrupted.
    next_point_timeout_s: float = 120.0
    next_point_poll_s: float = 0.5
    #: Supervised pause after reaching the lifted pose and before the return
    #: path starts.  A non-zero value makes the hold explicit in live logs.
    lift_hold_s: float = 0.0
    #: Pause after each completed in-place pick-and-return.  The operator can
    #: reposition the target at the next point, then press Enter to let the
    #: next cycle begin its visual search.  No motion is issued while paused.
    operator_pause_after_each_pick: bool = False
    #: When true the same object/class is used for every round (the "same
    #: target class at five positions" wording of the competition).
    keep_same_target: bool = True
    #: Human-readable labels for the five reference points.  With
    #: ``return_to_pick`` enabled these labels are reporting metadata only.
    place_names: tuple[str, ...] = ()
    #: Optional per-reference-point grasp height overrides, in the same order
    #: as ``place_poses``.  ``None`` falls back to ``motion.grasp_z_m``.
    point_grasp_z_m: tuple[float | None, ...] = ()
    #: Optional per-reference-point switches that take grasp/release XY from
    #: the point's taught observation anchor instead of the visual XY.  Visual
    #: confirmation is still mandatory before planning; this only selects a
    #: validated vertical descent column for points that need it.
    point_grasp_xy_from_observation: tuple[bool, ...] = ()
    #: Optional per-reference-point explicit grasp/release XY in the robot base
    #: frame.  This keeps a manually taught grasp column fixed even when the
    #: safe camera observation pose is deliberately offset from it.  Mutually
    #: exclusive with ``point_grasp_xy_from_observation`` for the same point.
    point_grasp_xy_m: tuple[tuple[float, float] | None, ...] = ()
    #: Optional per-reference-point XY correction in the flange/tool frame,
    #: applied after selecting the visual, taught-observation, or explicit
    #: grasp XY.  The final grasp RPY rotates the correction into the base
    #: frame, but only its base X/Y projection is applied so grasp Z remains
    #: fixed.  Target confirmation and point assignment still use the
    #: unshifted coordinates.
    point_grasp_xy_offset_m: tuple[tuple[float, float], ...] = ()
    #: Optional per-reference-point jaw-tip offset expressed in the observed
    #: object's horizontal PCA frame as ``[major, minor]`` metres.  Unlike the
    #: flange-frame correction above, this is learned from a paired successful
    #: grasp sample and rotates with the currently observed object axes.  It
    #: corrects the common case where the detector reports an object centre,
    #: while the gripper must close at an asymmetric point on the object.
    point_grasp_jaw_tip_object_offset_m: tuple[
        tuple[float, float] | None, ...
    ] = ()
    #: Optional per-reference-point grasp orientation overrides, in the same
    #: order as ``place_poses``.  ``None`` falls back to ``motion.grasp_rpy``.
    point_grasp_rpy: tuple[tuple[float, float, float] | None, ...] = ()
    #: Optional per-reference-point switches that lock the high approach,
    #: descent, lift and release to one vertical grasp column.  The high pose
    #: then uses the exact grasp XY and orientation, so once the arm is above
    #: the object the only remaining pre-grasp motion is a straight Z descent.
    point_vertical_grasp_descent: tuple[bool, ...] = ()
    #: Optional per-reference-point override for the locked high approach
    #: height above the final grasp Z.  ``None`` falls back to
    #: ``vertical_grasp_approach_height_m`` so a point-specific clearance can
    #: be tuned without changing every other point's route.
    point_vertical_grasp_approach_height_m: tuple[float | None, ...] = ()
    #: Height of the locked high approach pose above the final grasp Z.  With a
    #: verified grasp height this gives a deterministic vertical descent instead
    #: of inheriting an arbitrary observation-pose height.
    vertical_grasp_approach_height_m: float = 0.08
    #: Optional per-reference-point six-joint observation posture in degrees.
    #: When an entry is present, the arm moves there before its first camera
    #: observation for that point.  This keeps the later Cartesian approach
    #: local to the operator-validated low observation posture instead of
    #: planning a long diagonal move from the high startup/retreat posture.
    point_observation_joints_deg: tuple[tuple[float, ...] | None, ...] = ()
    #: Maximum six-joint residual after moving to an observation posture.
    observation_pose_tolerance_deg: float = 0.5
    #: Return to ``initial_pose_file`` after each point has been picked,
    #: returned in place and retreated successfully.  With the supervised gate
    #: enabled, the arm is already at the initial pose before the operator is
    #: asked to move the banana and press Enter for the next point.
    return_to_initial_after_each_pick: bool = False
    #: Return to the completed point's taught six-joint observation posture
    #: after a successful in-place pick/return and retreat.  This keeps the
    #: next visual acquisition local and repeatable without taking the shared
    #: initial-pose detour.  It is mutually exclusive with
    #: ``return_to_initial_after_each_pick`` and requires a concrete observation
    #: posture for every configured point.
    return_to_observation_after_each_pick: bool = False
    #: Legacy final-only return mode.  Ignored for per-point return decisions
    #: when ``return_to_initial_after_each_pick`` is true.
    return_to_initial_after_task: bool = False

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "TaskBConfig":
        data = dict(data or {})
        point_grasp_z_m = tuple(
            None if value is None else float(value)
            for value in data.get("point_grasp_z_m", ())
        )
        point_grasp_xy_from_observation = _bool_tuple(
            data.get("point_grasp_xy_from_observation", ())
        )
        point_grasp_xy_m = tuple(
            None if value is None else _pair(value)
            for value in data.get("point_grasp_xy_m", ())
        )
        point_grasp_xy_offset_m = tuple(
            _pair(value)
            for value in data.get("point_grasp_xy_offset_m", ())
        )
        point_grasp_jaw_tip_object_offset_m = tuple(
            None if value is None else _pair(value)
            for value in data.get(
                "point_grasp_jaw_tip_object_offset_m", ()
            )
        )
        point_grasp_rpy = tuple(
            None
            if value is None
            else _triple(value, (0.0, 0.0, 0.0))
            for value in data.get("point_grasp_rpy", ())
        )
        point_vertical_grasp_descent = _bool_tuple(
            data.get("point_vertical_grasp_descent", ())
        )
        point_vertical_grasp_approach_height_m = tuple(
            None if value is None else float(value)
            for value in data.get("point_vertical_grasp_approach_height_m", ())
        )
        point_observation_joints_deg = tuple(
            _joint_vector_deg(value)
            for value in data.get("point_observation_joints_deg", ())
        )
        return cls(
            place_poses=_pose_list(data.get("place_poses")),
            place_poses_file=_resolve_path(str(data.get("place_poses_file", "")), base_dir),
            initial_pose_file=_resolve_path(
                str(data.get("initial_pose_file", "")), base_dir
            ),
            initial_pose_tolerance_deg=float(
                data.get("initial_pose_tolerance_deg", 0.5)
            ),
            rounds=int(data.get("rounds", 5)),
            return_to_pick=bool(data.get("return_to_pick", True)),
            target_class=str(data.get("target_class", "")),
            pick_pose_tolerance_m=float(
                data.get("pick_pose_tolerance_m", 0.08)
            ),
            require_nearest_reference_point=bool(
                data.get("require_nearest_reference_point", True)
            ),
            point_assignment_margin_m=float(
                data.get("point_assignment_margin_m", 0.01)
            ),
            next_point_timeout_s=float(
                data.get("next_point_timeout_s", 120.0)
            ),
            next_point_poll_s=float(
                data.get("next_point_poll_s", 0.5)
            ),
            lift_hold_s=float(data.get("lift_hold_s", 0.0)),
            operator_pause_after_each_pick=bool(
                data.get("operator_pause_after_each_pick", False)
            ),
            keep_same_target=bool(data.get("keep_same_target", True)),
            place_names=_str_tuple(data.get("place_names")),
            point_grasp_z_m=point_grasp_z_m,
            point_grasp_xy_from_observation=point_grasp_xy_from_observation,
            point_grasp_xy_m=point_grasp_xy_m,
            point_grasp_xy_offset_m=point_grasp_xy_offset_m,
            point_grasp_jaw_tip_object_offset_m=(
                point_grasp_jaw_tip_object_offset_m
            ),
            point_grasp_rpy=point_grasp_rpy,
            point_vertical_grasp_descent=point_vertical_grasp_descent,
            point_vertical_grasp_approach_height_m=(
                point_vertical_grasp_approach_height_m
            ),
            vertical_grasp_approach_height_m=float(
                data.get("vertical_grasp_approach_height_m", 0.08)
            ),
            point_observation_joints_deg=point_observation_joints_deg,
            observation_pose_tolerance_deg=float(
                data.get("observation_pose_tolerance_deg", 0.5)
            ),
            return_to_initial_after_each_pick=bool(
                data.get("return_to_initial_after_each_pick", False)
            ),
            return_to_observation_after_each_pick=bool(
                data.get("return_to_observation_after_each_pick", False)
            ),
            return_to_initial_after_task=bool(
                data.get("return_to_initial_after_task", False)
            ),
        )


@dataclass(frozen=True)
class TargetConfig:
    # any = C; known_and_unseen = D; ordered = E.  known/any also cover the
    # single-object wording of task B.
    policy: str = "any"
    known_classes: tuple[str, ...] = ()
    unknown_labels: tuple[str, ...] = ("unknown", "object")
    unknown_min_graspability: float = 0.65
    order: tuple[str, ...] = ()
    order_file: str = ""
    max_attempts_per_target: int = 2

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "TargetConfig":
        data = dict(data or {})
        return cls(
            policy=str(data.get("policy", "any")),
            known_classes=_str_tuple(data.get("known_classes")),
            unknown_labels=_str_tuple(data.get("unknown_labels"), ("unknown", "object")),
            unknown_min_graspability=float(data.get("unknown_min_graspability", 0.65)),
            order=_str_tuple(data.get("order")),
            order_file=_resolve_path(str(data.get("order_file", "")), base_dir),
            max_attempts_per_target=int(data.get("max_attempts_per_target", 2)),
        )


@dataclass(frozen=True)
class PoseConfig:
    camera_frame: str = "base"  # base | flange
    extrinsics_file: str = ""
    use_detection_base_xyz: bool = True
    fallback_camera_to_base: bool = False
    fixed_camera_to_base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Coordinate semantics of RobotInfo.end_positions.  "unknown" is safe for
    #: the legacy fixed path, but adaptive eye-in-hand grasping refuses it.
    robot_pose_frame: str = "unknown"  # unknown | flange | tcp
    #: Flange-to-TCP translation in the flange frame.  Used only when the
    #: controller reports TCP poses; camera geometry is always flange based.
    tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Translation from the flange origin to the lowest point of the closed
    #: gripper jaws, expressed in the flange frame.  Visual grasp planning
    #: subtracts this offset (rotated into the selected grasp orientation) so
    #: the measured object point becomes the jaw-tip target rather than the
    #: flange target.  ``None`` is rejected whenever visual grasp Z is enabled.
    gripper_tip_offset_flange_m: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "PoseConfig":
        data = dict(data or {})
        return cls(
            camera_frame=str(data.get("camera_frame", "base")),
            extrinsics_file=_resolve_path(str(data.get("extrinsics_file", "")), base_dir),
            use_detection_base_xyz=bool(data.get("use_detection_base_xyz", True)),
            fallback_camera_to_base=bool(data.get("fallback_camera_to_base", False)),
            fixed_camera_to_base_xyz=_triple(
                data.get("fixed_camera_to_base_xyz"), (0.0, 0.0, 0.0)
            ),
            robot_pose_frame=str(data.get("robot_pose_frame", "unknown")),
            tcp_offset_m=_triple(
                data.get("tcp_offset_m"), (0.0, 0.0, 0.0)
            ),
            gripper_tip_offset_flange_m=(
                None
                if data.get("gripper_tip_offset_flange_m") is None
                else _triple(
                    data.get("gripper_tip_offset_flange_m"),
                    (0.0, 0.0, 0.0),
                )
            ),
        )


@dataclass(frozen=True)
class AdaptivePcaConfig:
    min_valid_points: int = 200
    min_depth_m: float = 0.08
    max_depth_m: float = 1.20
    trim_percentile: float = 5.0
    min_axis_ratio: float = 1.10
    #: A visual grasp must be based on a measured support plane rather than the
    #: top surface of the object.  Keep this disabled for generic point-cloud
    #: tests, but enable it for arbitrary-placement Task B so a failed tabletop
    #: fit rejects the observation instead of falling back to the object top.
    support_plane_required: bool = False
    support_plane_min_points: int = 300
    support_plane_max_rms_m: float = 0.0035
    support_plane_min_inlier_ratio: float = 0.35
    support_plane_min_normal_z: float = 0.80
    support_plane_depth_window_m: float = 0.012
    support_plane_guard_dilation_px: int = 13
    support_plane_ring_padding_ratio: float = 0.45
    support_plane_ransac_threshold_m: float = 0.002
    support_plane_ransac_iterations: int = 64
    support_plane_max_samples: int = 30000

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AdaptivePcaConfig":
        data = dict(data or {})
        return cls(
            min_valid_points=int(data.get("min_valid_points", 200)),
            min_depth_m=float(data.get("min_depth_m", 0.08)),
            max_depth_m=float(data.get("max_depth_m", 1.20)),
            trim_percentile=float(data.get("trim_percentile", 5.0)),
            min_axis_ratio=float(data.get("min_axis_ratio", 1.10)),
            support_plane_required=bool(
                data.get("support_plane_required", False)
            ),
            support_plane_min_points=int(
                data.get("support_plane_min_points", 300)
            ),
            support_plane_max_rms_m=float(
                data.get("support_plane_max_rms_m", 0.0035)
            ),
            support_plane_min_inlier_ratio=float(
                data.get("support_plane_min_inlier_ratio", 0.35)
            ),
            support_plane_min_normal_z=float(
                data.get("support_plane_min_normal_z", 0.80)
            ),
            support_plane_depth_window_m=float(
                data.get("support_plane_depth_window_m", 0.012)
            ),
            support_plane_guard_dilation_px=int(
                data.get("support_plane_guard_dilation_px", 13)
            ),
            support_plane_ring_padding_ratio=float(
                data.get("support_plane_ring_padding_ratio", 0.45)
            ),
            support_plane_ransac_threshold_m=float(
                data.get("support_plane_ransac_threshold_m", 0.002)
            ),
            support_plane_ransac_iterations=int(
                data.get("support_plane_ransac_iterations", 64)
            ),
            support_plane_max_samples=int(
                data.get("support_plane_max_samples", 30000)
            ),
        )


@dataclass(frozen=True)
class AdaptiveCandidatesConfig:
    max_count: int = 5
    #: When true, every visual candidate keeps the operator-taught RPY heading
    #: and does not align itself to the PCA object axis.  The generic yaw and
    #: pitch offset lists must contain exactly one zero entry.  The explicit
    #: ``taught_pitch_offsets_deg`` list may still provide small, tightly
    #: bounded IK branch candidates around that taught orientation.
    lock_taught_orientation: bool = False
    #: Object-relative yaw candidates.  The first candidate aligns the taught
    #: tool heading with the PCA long axis; later entries cover ambiguity in
    #: the gripper closing axis.
    yaw_offsets_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0)
    #: Signed candidate offsets added to the ZYX pitch component after the
    #: object-relative yaw has been applied.  Legacy arbitrary-placement mode
    #: keeps every entry exactly 0.0 so tool Z points straight down.  When
    #: ``adaptive_inverse.tilted_visual_grasp`` is explicitly enabled, non-zero
    #: entries are allowed and each candidate approaches along its own tool Z
    #: axis, while all IK, limit, singularity and collision gates stay active.
    pitch_offsets_deg: tuple[float, ...] = (0.0,)
    #: Small pitch-only branches around the operator-taught RPY when
    #: ``lock_taught_orientation`` is enabled.  These are attempted in order;
    #: every candidate still has to pass the complete IK, workspace, joint
    #: step, acceleration, singularity and Z-floor route prechecks.
    taught_pitch_offsets_deg: tuple[float, ...] = ()
    axis_alignment_offset_deg: float = 0.0
    max_orientation_change_deg: float = 120.0
    #: Apply the observed surface Z instead of a per-point taught grasp Z.
    use_visual_grasp_z: bool = False
    #: Additional base-Z clearance between the fitted support surface and the
    #: commanded jaw-tip point.  This prevents depth noise or an unsupported
    #: tabletop fit from commanding the jaw through the support surface.
    jaw_tip_clearance_m: float = 0.0
    #: Hard lower bound for the commanded grasp flange Z in the robot base
    #: frame.  Every adaptive candidate must satisfy this bound before route
    #: precheck or motion.  Fixed-point live grasping requires this value so a
    #: bad vision depth can never command the fingers through the box floor.
    minimum_grasp_flange_z_m: float | None = None
    approach_height_m: float | None = None
    #: Require every fixed-point adaptive grasp to carry a learned jaw-tip
    #: offset in object major/minor coordinates.  This is intentionally
    #: fail-closed: when enabled, planning will not silently substitute the
    #: detector's object centre for the actual gripper contact point.
    require_jaw_tip_object_offset: bool = False

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None
    ) -> "AdaptiveCandidatesConfig":
        data = dict(data or {})
        raw_offsets = data.get("yaw_offsets_deg", (0.0, 45.0, -45.0, 90.0, -90.0))
        if isinstance(raw_offsets, (str, bytes)) or not isinstance(
            raw_offsets, Sequence
        ):
            raise ValueError(
                "adaptive_inverse.candidates.yaw_offsets_deg must be a sequence"
            )
        raw_pitch_offsets = data.get("pitch_offsets_deg", (0.0,))
        if isinstance(raw_pitch_offsets, (str, bytes)) or not isinstance(
            raw_pitch_offsets, Sequence
        ):
            raise ValueError(
                "adaptive_inverse.candidates.pitch_offsets_deg must be a sequence"
            )
        raw_taught_pitch_offsets = data.get("taught_pitch_offsets_deg", ())
        if isinstance(raw_taught_pitch_offsets, (str, bytes)) or not isinstance(
            raw_taught_pitch_offsets, Sequence
        ):
            raise ValueError(
                "adaptive_inverse.candidates.taught_pitch_offsets_deg must be "
                "a sequence"
            )
        return cls(
            max_count=int(data.get("max_count", 5)),
            lock_taught_orientation=bool(
                data.get("lock_taught_orientation", False)
            ),
            yaw_offsets_deg=tuple(float(value) for value in raw_offsets),
            pitch_offsets_deg=tuple(
                float(value) for value in raw_pitch_offsets
            ),
            taught_pitch_offsets_deg=tuple(
                float(value)
                for value in raw_taught_pitch_offsets
            ),
            axis_alignment_offset_deg=float(
                data.get("axis_alignment_offset_deg", 0.0)
            ),
            max_orientation_change_deg=float(
                data.get("max_orientation_change_deg", 120.0)
            ),
            use_visual_grasp_z=bool(data.get("use_visual_grasp_z", False)),
            jaw_tip_clearance_m=float(
                data.get("jaw_tip_clearance_m", 0.0)
            ),
            minimum_grasp_flange_z_m=(
                None
                if data.get("minimum_grasp_flange_z_m") is None
                else float(data["minimum_grasp_flange_z_m"])
            ),
            approach_height_m=(
                None
                if data.get("approach_height_m") is None
                else float(data["approach_height_m"])
            ),
            require_jaw_tip_object_offset=bool(
                data.get("require_jaw_tip_object_offset", False)
            ),
        )


@dataclass(frozen=True)
class AdaptiveSelectionConfig:
    min_sigma_margin: float = 0.001
    sigma_weight: float = 10.0
    joint_travel_weight: float = 1.0
    orientation_weight: float = 1.0
    #: Candidate order is a safety preference order.  When true, stop after
    #: the first candidate whose complete route passes every motion check.
    #: This avoids spending tens of seconds evaluating lower-priority fallbacks
    #: after a safe route already exists.
    stop_after_first_valid: bool = False

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None
    ) -> "AdaptiveSelectionConfig":
        data = dict(data or {})
        return cls(
            min_sigma_margin=float(data.get("min_sigma_margin", 0.001)),
            sigma_weight=float(data.get("sigma_weight", 10.0)),
            joint_travel_weight=float(data.get("joint_travel_weight", 1.0)),
            orientation_weight=float(data.get("orientation_weight", 1.0)),
            stop_after_first_valid=bool(
                data.get("stop_after_first_valid", False)
            ),
        )


@dataclass(frozen=True)
class BoxRelativeFivePointConfig:
    """Fail-closed controls for dynamic box-pose five-point Task B.

    The box centre and yaw are always fitted from the current frame's ordered
    corners.  No configured or remembered centre/yaw is passed to the planner.
    The five candidate objects are all fully prechecked; only the one closest
    to the observed item may be executed.
    """

    enabled: bool = False
    gripper_max_outer_m: float = 0.140
    boss_axis: str = "x"
    max_capture_pose_translation_m: float = 0.002
    max_capture_pose_rotation_deg: float = 0.5
    max_capture_joint_delta_deg: float = 0.5
    max_corner_plane_rms_m: float = 0.004
    max_corner_edge_error_m: float = 0.015
    max_corner_orthogonality_error_deg: float = 3.0
    max_corner_reprojection_rms_px: float = 3.0
    max_corner_depth_std_m: float = 0.006
    min_plane_normal_z: float = 0.85
    item_match_max_distance_m: float = 0.075
    item_match_min_margin_m: float = 0.015

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None
    ) -> "BoxRelativeFivePointConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            gripper_max_outer_m=float(data.get("gripper_max_outer_m", 0.140)),
            boss_axis=str(data.get("boss_axis", "x")).strip().lower(),
            max_capture_pose_translation_m=float(
                data.get("max_capture_pose_translation_m", 0.002)
            ),
            max_capture_pose_rotation_deg=float(
                data.get("max_capture_pose_rotation_deg", 0.5)
            ),
            max_capture_joint_delta_deg=float(
                data.get("max_capture_joint_delta_deg", 0.5)
            ),
            max_corner_plane_rms_m=float(
                data.get("max_corner_plane_rms_m", 0.004)
            ),
            max_corner_edge_error_m=float(
                data.get("max_corner_edge_error_m", 0.015)
            ),
            max_corner_orthogonality_error_deg=float(
                data.get("max_corner_orthogonality_error_deg", 3.0)
            ),
            max_corner_reprojection_rms_px=float(
                data.get("max_corner_reprojection_rms_px", 3.0)
            ),
            max_corner_depth_std_m=float(
                data.get("max_corner_depth_std_m", 0.006)
            ),
            min_plane_normal_z=float(data.get("min_plane_normal_z", 0.85)),
            item_match_max_distance_m=float(
                data.get("item_match_max_distance_m", 0.075)
            ),
            item_match_min_margin_m=float(
                data.get("item_match_min_margin_m", 0.015)
            ),
        )


@dataclass(frozen=True)
class AdaptiveInverseConfig:
    enabled: bool = False
    #: Task B may disable the five-point workflow completely.  In this mode
    #: every round must use only the current visual observation; no point
    #: coordinates, per-point posture, or previously released XY may be read.
    #: The pipeline and validation keep this path strictly separate from the
    #: legacy five-point path so the latter remains available as a fallback.
    arbitrary_placement: bool = False
    #: Opt-in Task B arbitrary-placement mode: allow each visual candidate to
    #: approach along its own tool Z axis instead of locking every route to a
    #: vertical-down column.  This does not relax any motion or collision gate.
    tilted_visual_grasp: bool = False
    require_validated_extrinsics: bool = True
    min_pose_confidence: float = 0.50
    pca: AdaptivePcaConfig = field(default_factory=AdaptivePcaConfig)
    candidates: AdaptiveCandidatesConfig = field(
        default_factory=AdaptiveCandidatesConfig
    )
    selection: AdaptiveSelectionConfig = field(
        default_factory=AdaptiveSelectionConfig
    )
    box_clearance: BoxClearanceConfig = field(
        default_factory=BoxClearanceConfig
    )
    box_relative_five_point: BoxRelativeFivePointConfig = field(
        default_factory=BoxRelativeFivePointConfig
    )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None
    ) -> "AdaptiveInverseConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            arbitrary_placement=bool(
                data.get("arbitrary_placement", False)
            ),
            tilted_visual_grasp=bool(
                data.get("tilted_visual_grasp", False)
            ),
            require_validated_extrinsics=bool(
                data.get("require_validated_extrinsics", True)
            ),
            min_pose_confidence=float(data.get("min_pose_confidence", 0.50)),
            pca=AdaptivePcaConfig.from_dict(data.get("pca")),
            candidates=AdaptiveCandidatesConfig.from_dict(
                data.get("candidates")
            ),
            selection=AdaptiveSelectionConfig.from_dict(
                data.get("selection")
            ),
            box_clearance=BoxClearanceConfig.from_dict(
                data.get("box_clearance")
            ),
            box_relative_five_point=BoxRelativeFivePointConfig.from_dict(
                data.get("box_relative_five_point")
            ),
        )


@dataclass(frozen=True)
class GraspSafetyConfig:
    """Fail-closed geometric envelope for one tabletop grasp plan.

    These limits are deliberately separate from the generic Cartesian
    workspace.  A dry-run may use provisional values while the cell is being
    brought up, but a non-dry-run C/D configuration must provide every bound
    and a real feedback-capable gripper.
    """

    enabled: bool = False
    require_flange_tip_offset: bool = True
    minimum_grasp_flange_z_m: float | None = None
    maximum_grasp_flange_z_m: float | None = None
    minimum_grasp_tip_z_m: float | None = None
    maximum_grasp_tip_z_m: float | None = None
    #: The same envelope applied to every waypoint, not only the grasp.  This
    #: catches a configured release pose that would put the jaws through the
    #: table after the object has already been lifted.
    minimum_route_flange_z_m: float | None = None
    maximum_route_flange_z_m: float | None = None
    minimum_route_tip_z_m: float | None = None
    maximum_route_tip_z_m: float | None = None
    maximum_horizontal_reach_m: float | None = None
    maximum_tool_tilt_deg: float | None = None
    minimum_approach_height_m: float | None = None
    maximum_approach_height_m: float | None = None
    maximum_approach_xy_error_m: float | None = None
    require_tool_down: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "GraspSafetyConfig":
        data = dict(data or {})

        def optional_float(name: str) -> float | None:
            value = data.get(name)
            return None if value is None else float(value)

        return cls(
            enabled=bool(data.get("enabled", False)),
            require_flange_tip_offset=bool(
                data.get("require_flange_tip_offset", True)
            ),
            minimum_grasp_flange_z_m=optional_float(
                "minimum_grasp_flange_z_m"
            ),
            maximum_grasp_flange_z_m=optional_float(
                "maximum_grasp_flange_z_m"
            ),
            minimum_grasp_tip_z_m=optional_float("minimum_grasp_tip_z_m"),
            maximum_grasp_tip_z_m=optional_float("maximum_grasp_tip_z_m"),
            minimum_route_flange_z_m=optional_float(
                "minimum_route_flange_z_m"
            ),
            maximum_route_flange_z_m=optional_float(
                "maximum_route_flange_z_m"
            ),
            minimum_route_tip_z_m=optional_float("minimum_route_tip_z_m"),
            maximum_route_tip_z_m=optional_float("maximum_route_tip_z_m"),
            maximum_horizontal_reach_m=optional_float(
                "maximum_horizontal_reach_m"
            ),
            maximum_tool_tilt_deg=optional_float("maximum_tool_tilt_deg"),
            minimum_approach_height_m=optional_float(
                "minimum_approach_height_m"
            ),
            maximum_approach_height_m=optional_float(
                "maximum_approach_height_m"
            ),
            maximum_approach_xy_error_m=optional_float(
                "maximum_approach_xy_error_m"
            ),
            require_tool_down=bool(data.get("require_tool_down", True)),
        )


@dataclass(frozen=True)
class MotionConfig:
    backend: str = "mock"  # mock | robotmove
    dry_run: bool = True
    motion_mode: str = "joint_ik"  # cartesian (legacy) | joint_ik
    workspace_min: tuple[float, float, float] = (-0.45, -0.45, 0.0)
    workspace_max: tuple[float, float, float] = (0.45, 0.45, 0.60)
    workspace_margin: float = 0.0
    approach_height: float = 0.08
    grasp_z_offset: float = 0.0
    #: Optional operator-verified TCP height for Task B in-place pick/return.
    #: When set, visual detection supplies target XY but this value supplies
    #: the calibrated grasp Z instead of the raw depth-derived height.
    grasp_z_m: float | None = None
    #: Optional operator-verified TCP orientation for Task B pick/return.
    grasp_rpy: tuple[float, float, float] | None = None
    lift_height: float = 0.12
    place_pose: tuple[float, ...] | None = None
    #: Coordinate semantics of ``place_pose``.  ``flange`` preserves the
    #: historical controller-pose meaning.  ``jaw_tip`` means the configured
    #: XYZ is the desired closed-jaw low point and the planner converts it to
    #: the flange command pose with ``gripper_tip_offset_flange_m``.  C/D
    #: safety-enabled configurations must state this explicitly.
    place_pose_frame: str | None = None  # flange | jaw_tip
    place_approach_height: float = 0.08
    retreat_height: float = 0.08
    speed_mps: float = 0.05
    #: Scalar joint speed used for explicit joint-space moves such as the
    #: initial pose and per-point observation posture.
    joint_speed_rad_s: float = 0.6
    #: Upper bound for the scalar joint speed derived from Cartesian motion.
    #: The vendor bridge accepts one velocity scalar, not six joint rates.
    joint_speed_safety_limit_rad_s: float = 0.6
    orientation: str = "fixed"  # fixed | current
    fixed_rpy: tuple[float, float, float] = (3.141592653589793, 0.0, 0.0)
    move_timeout_s: float = 30.0
    position_tolerance_m: float = 0.003
    #: Tight numerical IK acceptance limits.  These are deliberately separate
    #: from ``position_tolerance_m``, which is the looser controller/feedback
    #: completion tolerance.  A Cartesian route is not published unless every
    #: generated waypoint meets these IK limits.
    ik_position_tolerance_m: float = 2e-4
    ik_orientation_tolerance_rad: float = 2e-4
    #: Complete operator-verified successful postures in degrees.  They are
    #: used only to initialize local IK restarts after the measured posture;
    #: they are never sent as waypoints and remain subject to every motion
    #: continuity, limit, tolerance and singularity guard.
    ik_fallback_seeds_deg: tuple[tuple[float, ...], ...] = ()
    poll_period_s: float = 0.05
    robotmove_topic: str = "RobotMove"
    robotinfo_topic: str = "RobotInfo"
    robot_info_timeout_s: float = 1.0
    joint_max_step_deg: float = 1.0
    joint_tolerance_deg: float = 0.35
    joint_start_tolerance_deg: float = 0.05
    joint_limit_margin_deg: float = 2.0
    joint_move_timeout_s: float = 8.0
    joint_reject_timeout_s: float = 2.0
    #: Planned Cartesian moves are subdivided until every adjacent IK step is
    #: below the joint-step limit.  The complete route is checked before the
    #: first command is published.
    cartesian_step_m: float = 0.002
    cartesian_min_step_m: float = 0.0005
    cartesian_orientation_step_deg: float = 1.0
    cartesian_min_orientation_step_deg: float = 0.1
    cartesian_max_waypoints: int = 600
    cartesian_joint_step_deg: float = 0.5
    cartesian_joint_branch_jump_deg: float = 5.0
    #: Maximum change in the per-waypoint joint increment.  This is a discrete
    #: acceleration guard because the controller bridge accepts absolute
    #: waypoint positions rather than a timed multi-point trajectory.
    cartesian_joint_acceleration_limit_deg: float = 0.25
    cartesian_min_jacobian_singular_value: float = 0.01
    grasp_safety: GraspSafetyConfig = field(
        default_factory=GraspSafetyConfig
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MotionConfig":
        data = dict(data or {})
        return cls(
            backend=str(data.get("backend", "mock")),
            dry_run=bool(data.get("dry_run", True)),
            motion_mode=str(data.get("motion_mode", "joint_ik")),
            workspace_min=_triple(data.get("workspace_min"), (-0.45, -0.45, 0.0)),
            workspace_max=_triple(data.get("workspace_max"), (0.45, 0.45, 0.60)),
            workspace_margin=float(data.get("workspace_margin", 0.0)),
            approach_height=float(data.get("approach_height", 0.08)),
            grasp_z_offset=float(data.get("grasp_z_offset", 0.0)),
            grasp_z_m=(
                None
                if data.get("grasp_z_m") is None
                else float(data["grasp_z_m"])
            ),
            grasp_rpy=(
                None
                if data.get("grasp_rpy") is None
                else _triple(data.get("grasp_rpy"), (0.0, 0.0, 0.0))
            ),
            lift_height=float(data.get("lift_height", 0.12)),
            place_pose=_pose_value(data.get("place_pose")),
            place_pose_frame=(
                None
                if data.get("place_pose_frame") is None
                else str(data.get("place_pose_frame")).strip().lower()
            ),
            place_approach_height=float(data.get("place_approach_height", 0.08)),
            retreat_height=float(data.get("retreat_height", 0.08)),
            speed_mps=float(data.get("speed_mps", 0.05)),
            joint_speed_rad_s=float(data.get("joint_speed_rad_s", 0.6)),
            joint_speed_safety_limit_rad_s=float(
                data.get("joint_speed_safety_limit_rad_s", 0.6)
            ),
            orientation=str(data.get("orientation", "fixed")),
            fixed_rpy=_triple(
                data.get("fixed_rpy"), (3.141592653589793, 0.0, 0.0)
            ),
            move_timeout_s=float(data.get("move_timeout_s", 30.0)),
            position_tolerance_m=float(data.get("position_tolerance_m", 0.003)),
            ik_position_tolerance_m=float(
                data.get("ik_position_tolerance_m", 2e-4)
            ),
            ik_orientation_tolerance_rad=float(
                data.get("ik_orientation_tolerance_rad", 2e-4)
            ),
            ik_fallback_seeds_deg=_float_rows(
                data.get("ik_fallback_seeds_deg")
            ),
            poll_period_s=float(data.get("poll_period_s", 0.05)),
            robotmove_topic=str(data.get("robotmove_topic", "RobotMove")),
            robotinfo_topic=str(data.get("robotinfo_topic", "RobotInfo")),
            robot_info_timeout_s=float(data.get("robot_info_timeout_s", 1.0)),
            joint_max_step_deg=float(data.get("joint_max_step_deg", 1.0)),
            joint_tolerance_deg=float(data.get("joint_tolerance_deg", 0.35)),
            joint_start_tolerance_deg=float(
                data.get("joint_start_tolerance_deg", 0.05)
            ),
            joint_limit_margin_deg=float(
                data.get("joint_limit_margin_deg", 2.0)
            ),
            joint_move_timeout_s=float(data.get("joint_move_timeout_s", 8.0)),
            joint_reject_timeout_s=float(
                data.get("joint_reject_timeout_s", 2.0)
            ),
            cartesian_step_m=float(data.get("cartesian_step_m", 0.002)),
            cartesian_min_step_m=float(
                data.get("cartesian_min_step_m", 0.0005)
            ),
            cartesian_orientation_step_deg=float(
                data.get("cartesian_orientation_step_deg", 1.0)
            ),
            cartesian_min_orientation_step_deg=float(
                data.get("cartesian_min_orientation_step_deg", 0.1)
            ),
            cartesian_max_waypoints=int(
                data.get("cartesian_max_waypoints", 600)
            ),
            cartesian_joint_step_deg=float(
                data.get("cartesian_joint_step_deg", 0.5)
            ),
            cartesian_joint_branch_jump_deg=float(
                data.get("cartesian_joint_branch_jump_deg", 5.0)
            ),
            cartesian_joint_acceleration_limit_deg=float(
                data.get("cartesian_joint_acceleration_limit_deg", 0.25)
            ),
            cartesian_min_jacobian_singular_value=float(
                data.get("cartesian_min_jacobian_singular_value", 0.01)
            ),
            grasp_safety=GraspSafetyConfig.from_dict(
                data.get("grasp_safety")
            ),
        )


@dataclass(frozen=True)
class ServoConfig:
    """Small-step eye-in-hand visual servoing parameters.

    ``camera_to_tool`` is R_tool_camera.  After the D405 RGB-D frame has been
    rotated 180 degrees, identity means image +X/+Y/+Z align with tool
    +X/+Y/+Z.  Keep this matrix configurable and verify signs under supervision.
    """

    desired_standoff_m: float = 0.18
    min_depth_m: float = 0.12
    center_tolerance_px: float = 35.0
    depth_tolerance_m: float = 0.015
    max_steps: int = 36
    max_step_m: float = 0.005
    max_total_travel_m: float = 0.35
    speed_mps: float = 0.01
    lateral_gain: float = 0.8
    forward_gain: float = 0.8
    settle_s_after_move: float = 0.25
    convergence_frames: int = 2
    camera_to_tool: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ServoConfig":
        import numpy as np

        data = dict(data or {})
        matrix = np.asarray(
            data.get("camera_to_tool", np.eye(3).tolist()), dtype=float
        )
        if matrix.shape != (3, 3):
            raise ValueError("servo.camera_to_tool must be a 3x3 matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("servo.camera_to_tool contains non-finite values")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4):
            raise ValueError("servo.camera_to_tool must be orthonormal")
        if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-4:
            raise ValueError("servo.camera_to_tool must have determinant +1")
        config = cls(
            desired_standoff_m=float(data.get("desired_standoff_m", 0.18)),
            min_depth_m=float(data.get("min_depth_m", 0.12)),
            center_tolerance_px=float(data.get("center_tolerance_px", 35.0)),
            depth_tolerance_m=float(data.get("depth_tolerance_m", 0.015)),
            max_steps=int(data.get("max_steps", 36)),
            max_step_m=float(data.get("max_step_m", 0.005)),
            max_total_travel_m=float(data.get("max_total_travel_m", 0.35)),
            speed_mps=float(data.get("speed_mps", 0.01)),
            lateral_gain=float(data.get("lateral_gain", 0.8)),
            forward_gain=float(data.get("forward_gain", 0.8)),
            settle_s_after_move=float(data.get("settle_s_after_move", 0.25)),
            convergence_frames=int(data.get("convergence_frames", 2)),
            camera_to_tool=tuple(tuple(float(v) for v in row) for row in matrix),
        )
        if not 0.0 < config.max_step_m <= 0.05:
            raise ValueError("servo.max_step_m must be in (0, 0.05]")
        if config.min_depth_m <= 0.0:
            raise ValueError("servo.min_depth_m must be positive")
        if config.desired_standoff_m <= config.min_depth_m:
            raise ValueError("servo.desired_standoff_m must exceed min_depth_m")
        if config.max_steps < 1 or config.convergence_frames < 1:
            raise ValueError("servo max_steps/convergence_frames must be >= 1")
        if config.max_total_travel_m <= 0.0:
            raise ValueError("servo.max_total_travel_m must be positive")
        if not 0.0 < config.speed_mps <= 0.05:
            raise ValueError("servo.speed_mps must be in (0, 0.05]")
        return config


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool = True
    yaw_offsets_deg: tuple[float, ...] = (0.0,)
    pitch_down_offsets_deg: tuple[float, ...] = (0.0,)
    settle_s: float = 0.8
    #: Safe Cartesian observation positions around the expected Task B point.
    #: Each entry is an (x, y) offset in metres from that point.  Empty keeps
    #: the legacy wrist-only search used by the hardware-free tests.
    observation_offsets_xy_m: tuple[tuple[float, float], ...] = ()
    #: Height above the expected point's Z used for the active observation
    #: positions.  Keep the camera clear of the table while it scans.
    observation_clearance_m: float = 0.20
    #: Relative-to-current-TCP observation adjustments for object acquisition
    #: when no remembered pick point exists.  Each entry is an (x, y, z)
    #: translation in metres and preserves the current flange orientation.
    #: The pipeline pre-checks every requested view and the return to the
    #: original observation posture before issuing the first motion command.
    arbitrary_observation_offsets_xyz_m: tuple[
        tuple[float, float, float], ...
    ] = ()
    observe_frames: int = 3
    #: Number of consecutive frames that must select the same target before a
    #: pick cycle is allowed to move.  The D405 stabilizer already suppresses
    #: one-frame flicker, but this second gate protects the motion planner from
    #: a single false positive after a search move.
    confirm_hits: int = 2
    return_to_initial_on_miss: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SearchConfig":
        data = dict(data or {})
        offsets: list[tuple[float, float]] = []
        raw_offsets = data.get("observation_offsets_xy_m", ())
        if raw_offsets is None:
            raw_offsets = ()
        for index, raw in enumerate(raw_offsets, start=1):
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(
                    "search.observation_offsets_xy_m entries must be [x, y]"
                )
            try:
                x, y = (float(raw[0]), float(raw[1]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "search.observation_offsets_xy_m entries must be numeric"
                ) from exc
            if not isfinite(x) or not isfinite(y):
                raise ValueError(
                    f"search.observation_offsets_xy_m entry {index} must be finite"
                )
            offsets.append((x, y))
        relative_offsets: list[tuple[float, float, float]] = []
        raw_relative_offsets = data.get("arbitrary_observation_offsets_xyz_m", ())
        if raw_relative_offsets is None:
            raw_relative_offsets = ()
        for index, raw in enumerate(raw_relative_offsets, start=1):
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                raise ValueError(
                    "search.arbitrary_observation_offsets_xyz_m entries "
                    "must be [x, y, z]"
                )
            try:
                dx, dy, dz = (float(raw[0]), float(raw[1]), float(raw[2]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "search.arbitrary_observation_offsets_xyz_m entries "
                    "must be numeric"
                ) from exc
            if not all(isfinite(value) for value in (dx, dy, dz)):
                raise ValueError(
                    "search.arbitrary_observation_offsets_xyz_m entry "
                    f"{index} must be finite"
                )
            distance_m = (dx * dx + dy * dy + dz * dz) ** 0.5
            if distance_m <= 1e-9:
                raise ValueError(
                    "search.arbitrary_observation_offsets_xyz_m entry "
                    f"{index} must not be [0, 0, 0]"
                )
            if distance_m > 0.10:
                raise ValueError(
                    "search.arbitrary_observation_offsets_xyz_m entry "
                    f"{index} exceeds the 0.10 m observation-adjustment limit"
                )
            relative_offsets.append((dx, dy, dz))
        config = cls(
            enabled=bool(data.get("enabled", True)),
            yaw_offsets_deg=tuple(
                float(v) for v in data.get("yaw_offsets_deg", (0.0,))
            ),
            pitch_down_offsets_deg=tuple(
                float(v) for v in data.get("pitch_down_offsets_deg", (0.0,))
            ),
            settle_s=float(data.get("settle_s", 0.8)),
            observation_offsets_xy_m=tuple(offsets),
            observation_clearance_m=float(
                data.get("observation_clearance_m", 0.20)
            ),
            arbitrary_observation_offsets_xyz_m=tuple(relative_offsets),
            observe_frames=int(data.get("observe_frames", 3)),
            confirm_hits=int(data.get("confirm_hits", 2)),
            return_to_initial_on_miss=bool(
                data.get("return_to_initial_on_miss", True)
            ),
        )
        if config.observe_frames < 1:
            raise ValueError("search.observe_frames must be >= 1")
        if not 1 <= config.confirm_hits <= config.observe_frames:
            raise ValueError(
                "search.confirm_hits must be in [1, search.observe_frames]"
            )
        if config.settle_s < 0.0:
            raise ValueError("search.settle_s must be >= 0")
        if config.observation_clearance_m <= 0.0:
            raise ValueError("search.observation_clearance_m must be > 0")
        return config


@dataclass(frozen=True)
class GripperConfig:
    """How to drive the jaws.

    ``backend`` selects the transport.  Everything after it is deliberately
    generic so that a two-finger electric gripper with its own electronic
    controller can be driven by whichever interface it exposes:

    * ``mock``        - no hardware, records commands (default)
    * ``taska``       - reuse the existing pluggable gripper in taskA_grasp
    * ``serial``      - RS485/RS232: open/close command bytes, optional feedback
    * ``tcp``         - Ethernet: same command bytes over a socket
    * ``ros_service`` - call a service exposed by the vendor's ROS driver
    * ``ros_topic``   - publish std_msgs/Bool on a topic consumed by a driver
    * ``ms42dc_ros``  - use the vendor ``step_motor/motor_node`` ROS driver
    """

    backend: str = "mock"
    taska_config: str = ""
    settle_time_s: float = 0.8
    open_before_approach: bool = True

    # --- generic discrete command layer (serial / tcp) -------------------
    open_command: str = ""
    close_command: str = ""
    command_encoding: str = "ascii"  # ascii | hex
    command_terminator: str = ""  # e.g. "\r\n"; escapes are expanded
    expect_response: bool = False
    response_timeout_s: float = 0.3
    response_length: int = 0  # 0 -> read whatever is available
    feedback_command: str = ""  # optional query issued before state is read
    feedback_open_match: str = ""  # substring in the reply that means OPEN
    feedback_closed_match: str = ""  # substring in the reply that means CLOSED

    # --- serial ----------------------------------------------------------
    serial_port: str = ""
    serial_baudrate: int = 115200
    serial_bytesize: int = 8
    serial_parity: str = "N"
    serial_stopbits: int = 1

    # --- tcp -------------------------------------------------------------
    tcp_host: str = ""
    tcp_port: int = 0

    # --- ros drivers -----------------------------------------------------
    ros_service: str = ""
    ros_topic: str = ""
    ros_open_value: bool = True
    ros_close_value: bool = False

    # --- WHEELTEC FAE 2-finger electric gripper (MS42DC stepper) ---------
    #   ``ms42dc`` talks the 11-byte protocol directly over the serial port.
    #   ``ms42dc_ros`` delegates the same protocol to the vendor ROS 2 node,
    #   so only one process owns /dev/motor_serial.
    device_id: int = 1
    subdivide: int = 32
    motion_mode: int = 2  # 2 = relative angle, 4 = absolute angle
    open_angle_deg: float = 1872.0  # 5.2 turns; wire value is this times ten
    close_angle_deg: float = 1872.0
    speed_rad_s: float = 20.0  # wire value is this times ten
    status_timeout_s: float = 1.0
    motion_timeout_s: float = 8.0
    invert_direction: bool = False  # swap open/close when the wiring is reversed
    hold_min_angle_deg: float = 0.0  # > 0 enables the empty-grip angle check
    require_feedback: bool = False
    ignore_empty_close: bool = False

    # --- vendor MS42DC ROS 2 driver --------------------------------------
    motor_control_topic: str = "/motor_control"
    motor_state_topic: str = "/motor_state"
    open_dir: int = 0
    close_dir: int = 1
    ros_connect_timeout_s: float = 3.0

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "GripperConfig":
        data = dict(data or {})
        return cls(
            backend=str(data.get("backend", "mock")),
            taska_config=_resolve_path(str(data.get("taska_config", "")), base_dir),
            settle_time_s=float(data.get("settle_time_s", 0.8)),
            open_before_approach=bool(data.get("open_before_approach", True)),
            open_command=str(data.get("open_command", "")),
            close_command=str(data.get("close_command", "")),
            command_encoding=str(data.get("command_encoding", "ascii")),
            command_terminator=str(data.get("command_terminator", "")),
            expect_response=bool(data.get("expect_response", False)),
            response_timeout_s=float(data.get("response_timeout_s", 0.3)),
            response_length=int(data.get("response_length", 0)),
            feedback_command=str(data.get("feedback_command", "")),
            feedback_open_match=str(data.get("feedback_open_match", "")),
            feedback_closed_match=str(data.get("feedback_closed_match", "")),
            serial_port=str(data.get("serial_port", "")),
            serial_baudrate=int(data.get("serial_baudrate", 115200)),
            serial_bytesize=int(data.get("serial_bytesize", 8)),
            serial_parity=str(data.get("serial_parity", "N")),
            serial_stopbits=int(data.get("serial_stopbits", 1)),
            tcp_host=str(data.get("tcp_host", "")),
            tcp_port=int(data.get("tcp_port", 0)),
            ros_service=str(data.get("ros_service", "")),
            ros_topic=str(data.get("ros_topic", "")),
            ros_open_value=bool(data.get("ros_open_value", True)),
            ros_close_value=bool(data.get("ros_close_value", False)),
            device_id=int(data.get("device_id", 1)),
            subdivide=int(data.get("subdivide", 32)),
            motion_mode=int(data.get("motion_mode", 2)),
            open_angle_deg=float(data.get("open_angle_deg", 1872.0)),
            close_angle_deg=float(data.get("close_angle_deg", 1872.0)),
            speed_rad_s=float(data.get("speed_rad_s", 20.0)),
            status_timeout_s=float(data.get("status_timeout_s", 1.0)),
            motion_timeout_s=float(data.get("motion_timeout_s", 8.0)),
            invert_direction=bool(data.get("invert_direction", False)),
            hold_min_angle_deg=float(data.get("hold_min_angle_deg", 0.0)),
            require_feedback=bool(data.get("require_feedback", False)),
            ignore_empty_close=bool(data.get("ignore_empty_close", False)),
            motor_control_topic=str(data.get("motor_control_topic", "/motor_control")),
            motor_state_topic=str(data.get("motor_state_topic", "/motor_state")),
            open_dir=int(data.get("open_dir", 0)),
            close_dir=int(data.get("close_dir", 1)),
            ros_connect_timeout_s=float(data.get("ros_connect_timeout_s", 3.0)),
        )


@dataclass(frozen=True)
class GraspConsensusGateConfig:
    """Runtime binding for one per-scene offline multi-frame grasp report.

    The defaults are also hard safety ceilings: an enabled gate may make a
    threshold stricter, but it may never relax it.  The report is deliberately
    disabled by default because a consensus generated for one scene must not
    silently authorize a different scene.
    """

    enabled: bool = False
    report_file: str = ""
    minimum_distinct_frames: int = 4
    maximum_translation_tolerance_m: float = 0.025
    maximum_jaw_rotation_tolerance_deg: float = 15.0
    maximum_pose_translation_error_m: float = 0.025
    maximum_pose_jaw_rotation_error_deg: float = 15.0

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "GraspConsensusGateConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            report_file=_resolve_path(str(data.get("report_file", "")), base_dir),
            minimum_distinct_frames=int(data.get("minimum_distinct_frames", 4)),
            maximum_translation_tolerance_m=float(
                data.get("maximum_translation_tolerance_m", 0.025)
            ),
            maximum_jaw_rotation_tolerance_deg=float(
                data.get("maximum_jaw_rotation_tolerance_deg", 15.0)
            ),
            maximum_pose_translation_error_m=float(
                data.get("maximum_pose_translation_error_m", 0.025)
            ),
            maximum_pose_jaw_rotation_error_deg=float(
                data.get("maximum_pose_jaw_rotation_error_deg", 15.0)
            ),
        )


@dataclass(frozen=True)
class Track1Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    task_a: TaskAConfig = field(default_factory=TaskAConfig)
    task_b: TaskBConfig = field(default_factory=TaskBConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    adaptive_inverse: AdaptiveInverseConfig = field(
        default_factory=AdaptiveInverseConfig
    )
    motion: MotionConfig = field(default_factory=MotionConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    grasp_consensus: GraspConsensusGateConfig = field(
        default_factory=GraspConsensusGateConfig
    )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, *, base_dir: Path | None = None
    ) -> "Track1Config":
        data = dict(data or {})
        base_dir = base_dir or Path.cwd()
        config = cls(
            task=TaskConfig.from_dict(data.get("task")),
            vision=VisionConfig.from_dict(data.get("vision"), base_dir),
            task_a=TaskAConfig.from_dict(data.get("task_a"), base_dir),
            task_b=TaskBConfig.from_dict(data.get("task_b"), base_dir),
            target=TargetConfig.from_dict(data.get("target"), base_dir),
            pose=PoseConfig.from_dict(data.get("pose"), base_dir),
            adaptive_inverse=AdaptiveInverseConfig.from_dict(
                data.get("adaptive_inverse")
            ),
            motion=MotionConfig.from_dict(data.get("motion")),
            servo=ServoConfig.from_dict(data.get("servo")),
            search=SearchConfig.from_dict(data.get("search")),
            gripper=GripperConfig.from_dict(data.get("gripper"), base_dir),
            grasp_consensus=GraspConsensusGateConfig.from_dict(
                data.get("grasp_consensus"), base_dir
            ),
        )
        config.validate()
        return config

    @property
    def order(self) -> tuple[str, ...]:
        if self.target.order:
            return self.target.order
        if not self.target.order_file:
            return ()
        return _load_string_list(self.target.order_file, key="order")

    @property
    def place_poses(self) -> tuple[tuple[float, ...], ...]:
        """Legacy task B release slots, from config or an external YAML file."""

        if self.task_b.place_poses:
            return self.task_b.place_poses
        if not self.task_b.place_poses_file:
            return ()
        poses = _load_pose_list(self.task_b.place_poses_file)
        if self.task_b.rounds and len(poses) > self.task_b.rounds:
            return poses[: self.task_b.rounds]
        return poses

    def validate(self) -> None:
        if self.task.max_picks < 1:
            raise ValueError("task.max_picks must be >= 1")
        if self.task.sample_period_s < 0.0:
            raise ValueError("task.sample_period_s must be >= 0")
        if self.vision.backend not in {
            "mock",
            "taska_realsense",
            "realsense_yolo",
            "legacy_http",
        }:
            raise ValueError(
                "vision.backend must be mock, taska_realsense, realsense_yolo "
                "or legacy_http"
            )
        if self.vision.backend == "mock" and not self.vision.scenario_file:
            raise ValueError("mock vision needs vision.scenario_file")
        if self.vision.backend == "taska_realsense" and not self.vision.taska_config:
            raise ValueError("taska_realsense vision needs vision.taska_config")
        if self.vision.backend == "realsense_yolo" and not self.vision.weights:
            raise ValueError("realsense_yolo vision needs vision.weights")
        if self.vision.weights_sha256:
            expected_sha256 = self.vision.weights_sha256.strip().lower()
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            ):
                raise ValueError(
                    "vision.weights_sha256 must be a 64-character hexadecimal SHA256"
                )
        if self.vision.backend == "legacy_http":
            if not self.vision.legacy_status_url:
                raise ValueError(
                    "legacy_http vision needs vision.legacy_status_url"
                )
            if not self.vision.accepted_target_labels():
                raise ValueError(
                    "legacy_http vision needs vision.target_class or vision.classes "
                    "so unrelated detections cannot satisfy the patrol"
                )
        if self.vision.legacy_poll_s <= 0.0:
            raise ValueError("vision.legacy_poll_s must be positive")
        if self.vision.anchor not in {"center", "top_center", "bottom_center"}:
            raise ValueError(
                "vision.anchor must be center, top_center or bottom_center"
            )
        if self.vision.width <= 0 or self.vision.height <= 0 or self.vision.fps <= 0:
            raise ValueError("vision width/height/fps must be positive")
        if self.vision.depth_window < 1:
            raise ValueError("vision.depth_window must be >= 1")
        if self.vision.capture_timeout_s <= 0.0:
            raise ValueError("vision.capture_timeout_s must be positive")
        if self.vision.first_frame_timeout_s <= 0.0:
            raise ValueError("vision.first_frame_timeout_s must be positive")
        if self.vision.stale_after_s <= 0.0:
            raise ValueError("vision.stale_after_s must be positive")
        if not 0.0 <= self.vision.min_confidence <= 1.0:
            raise ValueError("vision.min_confidence must be in [0, 1]")
        if not 0.0 <= self.vision.stabilizer_inference_min_confidence <= 1.0:
            raise ValueError(
                "vision.stabilizer.inference_min_confidence must be in [0, 1]"
            )
        if self.vision.stabilizer_confirm_hits < 1:
            raise ValueError("vision.stabilizer.confirm_hits must be >= 1")
        if self.vision.stabilizer_max_misses < 0:
            raise ValueError("vision.stabilizer.max_misses must be >= 0")
        if self.vision.stabilizer_max_per_class < 0:
            raise ValueError("vision.stabilizer.max_per_class must be >= 0")
        for name in (
            "stabilizer_match_iou",
            "stabilizer_confidence_alpha",
            "stabilizer_box_alpha",
            "stabilizer_depth_alpha",
            "stabilizer_duplicate_iou",
            "stabilizer_containment_ratio",
            "stabilizer_cross_class_iou",
            "stabilizer_cross_class_containment",
            "stabilizer_edge_suppress_confidence",
        ):
            value = float(getattr(self.vision, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"vision.{name} must be in [0, 1]")
        if self.vision.stabilizer_cross_class_conf_ratio < 1.0:
            raise ValueError(
                "vision.stabilizer.cross_class_conf_ratio must be >= 1"
            )
        if self.vision.stabilizer_class_switch_hits < 1:
            raise ValueError("vision.stabilizer.class_switch_hits must be >= 1")
        if self.vision.stabilizer_edge_margin_px < 0:
            raise ValueError("vision.stabilizer.edge_margin_px must be >= 0")
        if not 0.0 <= self.task_a.iou_threshold <= 1.0:
            raise ValueError("task_a.iou_threshold must be in [0, 1]")
        if self.task_a.max_frames < 0:
            raise ValueError("task_a.max_frames must be >= 0")
        if self.task_a.min_positions < 1:
            raise ValueError("task_a.min_positions must be >= 1")
        patrol = self.task_a.patrol
        if patrol.mode not in {"joint_arc", "cartesian_box"}:
            raise ValueError(
                "task_a.patrol.mode must be joint_arc or cartesian_box"
            )
        if patrol.position_count < 2:
            raise ValueError("task_a.patrol.position_count must be >= 2")
        if patrol.enabled and patrol.position_count < self.task_a.min_positions:
            raise ValueError(
                "task_a.patrol.position_count must be at least "
                "task_a.min_positions"
            )
        if patrol.enabled and patrol.position_count != 10:
            raise ValueError(
                "task_a.patrol.position_count must be exactly 10 for "
                "the scored Task A patrol"
            )
        if patrol.min_move_distance_m <= 0.0:
            raise ValueError("task_a.patrol.min_move_distance_m must be positive")
        if patrol.settle_s < 0.0:
            raise ValueError("task_a.patrol.settle_s must be >= 0")
        if patrol.mode == "joint_arc":
            expected_movable = (1, 2, 3, 4)
            if patrol.movable_joints != expected_movable:
                raise ValueError(
                    "task_a.patrol.movable_joints must be exactly "
                    "[Joint1, Joint2, Joint3, Joint4]; Joint5 and Joint6 "
                    "are locked"
                )
            for name in (
                "joint_max_step_deg",
                "max_initial_move_deg",
                "max_target_move_deg",
            ):
                value = float(getattr(patrol, name))
                if not isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"task_a.patrol.{name} must be positive and finite"
                    )
            if patrol.joint_max_step_deg > 5.0:
                raise ValueError(
                    "task_a.patrol.joint_max_step_deg must be <= 5 degrees"
                )
            if (
                not isfinite(patrol.post_detection_dwell_s)
                or patrol.post_detection_dwell_s < 0.0
            ):
                raise ValueError(
                    "task_a.patrol.post_detection_dwell_s must be finite "
                    "and >= 0"
                )

            j1_targets = patrol.joint1_relative_targets_deg
            if patrol.enabled and not j1_targets:
                raise ValueError(
                    "task_a.patrol.joint1_relative_targets_deg must define "
                    "all ten startup-relative J1 targets"
                )
            if j1_targets:
                if len(j1_targets) != patrol.position_count:
                    raise ValueError(
                        "task_a.patrol.joint1_relative_targets_deg must contain "
                        "exactly one angle per patrol position"
                    )
                if not all(isfinite(value) for value in j1_targets):
                    raise ValueError(
                        "task_a.patrol.joint1_relative_targets_deg must contain "
                        "finite values"
                    )
                if min(j1_targets) < -135.0 - 1e-9 or max(j1_targets) > 135.0 + 1e-9:
                    raise ValueError(
                        "task_a.patrol.joint1_relative_targets_deg must stay "
                        "within the relative 270-degree range [-135, +135]"
                    )
                if patrol.enabled:
                    first_arc = j1_targets[:5]
                    second_arc = j1_targets[5:10]
                    if abs(first_arc[0] - 135.0) > 1e-9 or abs(first_arc[-1] + 135.0) > 1e-9:
                        raise ValueError(
                            "the first J1 arc must span +135 to -135 degrees"
                        )
                    if any(
                        left <= right
                        for left, right in zip(first_arc, first_arc[1:])
                    ):
                        raise ValueError(
                            "the first J1 arc must decrease monotonically"
                        )
                    if abs(second_arc[0] + 135.0) > 1e-9 or abs(second_arc[-1] - 135.0) > 1e-9:
                        raise ValueError(
                            "the second J1 arc must span -135 to +135 degrees"
                        )
                    if any(
                        left >= right
                        for left, right in zip(second_arc, second_arc[1:])
                    ):
                        raise ValueError(
                            "the second J1 arc must increase monotonically"
                        )
                    if abs(first_arc[-1] - second_arc[0]) > 1e-9:
                        raise ValueError(
                            "J1 relative targets must remain at -135 degrees "
                            "when switching arcs"
                        )

            if patrol.enabled and not patrol.joint_offsets_deg:
                raise ValueError(
                    "task_a.patrol.joint_offsets_deg must define all ten "
                    "J2/J3/J4 offset targets"
                )
            if patrol.joint_offsets_deg:
                if len(patrol.joint_offsets_deg) != patrol.position_count:
                    raise ValueError(
                        "task_a.patrol.joint_offsets_deg must contain exactly "
                        "one row per patrol position"
                    )
                expected_width = 3  # J2, J3, J4
                for index, row in enumerate(patrol.joint_offsets_deg, start=1):
                    if len(row) != expected_width:
                        raise ValueError(
                            "task_a.patrol.joint_offsets_deg row "
                            f"{index} must contain {expected_width} values "
                            "for J2/J3/J4"
                        )
                    if not all(isfinite(value) for value in row):
                        raise ValueError(
                            "task_a.patrol.joint_offsets_deg row "
                            f"{index} must contain finite values"
                        )
            if patrol.arc_indices:
                if len(patrol.arc_indices) != patrol.position_count:
                    raise ValueError(
                        "task_a.patrol.arc_indices must contain exactly one "
                        "entry per patrol position"
                    )
                if set(patrol.arc_indices) - {1, 2}:
                    raise ValueError(
                        "task_a.patrol.arc_indices may only contain 1 or 2"
                    )
                if patrol.enabled and set(patrol.arc_indices) != {1, 2}:
                    raise ValueError(
                        "task_a.patrol.arc_indices must use both of the two arcs"
                    )
                if patrol.enabled:
                    if patrol.arc_indices[:5] != (1, 1, 1, 1, 1):
                        raise ValueError(
                            "the first five patrol positions must use J1 arc 1"
                        )
                    if patrol.arc_indices[5:] != (2, 2, 2, 2, 2):
                        raise ValueError(
                            "the second five patrol positions must use J1 arc 2"
                        )
        else:
            if patrol.stride_m < patrol.min_move_distance_m:
                raise ValueError(
                    "task_a.patrol.stride_m must be >= min_move_distance_m"
                )
            if patrol.linear_step_m <= 0.0:
                raise ValueError("task_a.patrol.linear_step_m must be positive")
            if not patrol.axes:
                raise ValueError("task_a.patrol.axes must not be empty")
            invalid_axes = set(patrol.axes) - {"x", "y", "z"}
            if invalid_axes:
                raise ValueError(
                    "task_a.patrol.axes may only contain x, y and z; got "
                    f"{sorted(invalid_axes)}"
                )
        visibility = patrol.object_visibility
        if patrol.enabled and patrol.manual_step:
            expected_classes = tuple(
                label.strip() for label in self.vision.classes if label.strip()
            )
            if len(expected_classes) != patrol.position_count:
                raise ValueError(
                    "task_a.patrol.manual_step requires exactly one "
                    "vision.classes label per patrol position"
                )
            if len(set(expected_classes)) != len(expected_classes):
                raise ValueError(
                    "task_a.patrol.manual_step requires unique vision.classes "
                    "labels so every point has one expected object"
                )
        if visibility.visible_confirm_frames < 1:
            raise ValueError(
                "task_a.patrol.object_visibility.visible_confirm_frames "
                "must be >= 1"
            )
        if visibility.observation_window_s <= 0.0:
            raise ValueError(
                "task_a.patrol.object_visibility.observation_window_s "
                "must be positive"
            )
        if visibility.poll_s <= 0.0:
            raise ValueError(
                "task_a.patrol.object_visibility.poll_s must be positive"
            )
        if self.task_b.rounds < 1:
            raise ValueError("task_b.rounds must be >= 1")
        adaptive = self.adaptive_inverse
        offline_only = (
            self.vision.backend == "mock"
            and self.motion.backend == "mock"
            and self.motion.dry_run
            and self.gripper.backend == "mock"
        )
        box_relative = adaptive.box_relative_five_point
        if (
            adaptive.box_clearance.required
            or adaptive.box_clearance.has_configured_geometry()
        ):
            validate_box_clearance_geometry(
                adaptive.box_clearance,
                require_pose=not box_relative.enabled,
            )
        if adaptive.arbitrary_placement and box_relative.enabled:
            raise ValueError(
                "adaptive_inverse.arbitrary_placement and "
                "adaptive_inverse.box_relative_five_point are mutually "
                "exclusive"
            )
        if box_relative.enabled:
            if not adaptive.enabled:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point requires "
                    "adaptive_inverse.enabled=true"
                )
            if self.task.mode is not TaskMode.B:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point is restricted "
                    "to Task B"
                )
            if self.vision.continuous:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point requires "
                    "vision.continuous=false so every pose comes from one "
                    "fresh RGB-D frame"
                )
            if not self.task_b.return_to_pick:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point requires "
                    "task_b.return_to_pick=true"
                )
            if not adaptive.candidates.use_visual_grasp_z:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point requires "
                    "adaptive_inverse.candidates.use_visual_grasp_z=true"
                )
            if not adaptive.box_clearance.required:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point requires "
                    "adaptive_inverse.box_clearance.required=true"
                )
            if not adaptive.require_validated_extrinsics:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point requires "
                    "require_validated_extrinsics=true"
                )
            if self.pose.camera_frame != "flange":
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point requires "
                    "pose.camera_frame=flange"
                )
            if box_relative.boss_axis != "x":
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point.boss_axis must "
                    "be 'x' for the measured north/south edge bosses"
                )
            positive_values = {
                "gripper_max_outer_m": box_relative.gripper_max_outer_m,
                "max_capture_pose_translation_m": (
                    box_relative.max_capture_pose_translation_m
                ),
                "max_capture_pose_rotation_deg": (
                    box_relative.max_capture_pose_rotation_deg
                ),
                "max_capture_joint_delta_deg": (
                    box_relative.max_capture_joint_delta_deg
                ),
                "max_corner_plane_rms_m": box_relative.max_corner_plane_rms_m,
                "max_corner_edge_error_m": box_relative.max_corner_edge_error_m,
                "max_corner_orthogonality_error_deg": (
                    box_relative.max_corner_orthogonality_error_deg
                ),
                "max_corner_reprojection_rms_px": (
                    box_relative.max_corner_reprojection_rms_px
                ),
                "max_corner_depth_std_m": box_relative.max_corner_depth_std_m,
                "item_match_max_distance_m": (
                    box_relative.item_match_max_distance_m
                ),
            }
            invalid_positive = [
                name
                for name, value in positive_values.items()
                if not isfinite(float(value)) or float(value) <= 0.0
            ]
            if invalid_positive:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point values must be "
                    "positive and finite: "
                    + ", ".join(invalid_positive)
                )
            if (
                not isfinite(box_relative.min_plane_normal_z)
                or not 0.0 < box_relative.min_plane_normal_z <= 1.0
            ):
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point."
                    "min_plane_normal_z must be in (0, 1]"
                )
            if (
                not isfinite(box_relative.item_match_min_margin_m)
                or box_relative.item_match_min_margin_m < 0.0
            ):
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point."
                    "item_match_min_margin_m must be finite and >= 0"
                )
            if (
                adaptive.box_clearance.geometry_source
                == GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC
                and not offline_only
            ):
                raise ValueError(
                    "adaptive_inverse.box_clearance synthetic geometry is "
                    "restricted to the fully offline Mock pipeline"
                )
            if (
                not offline_only
                and adaptive.box_clearance.geometry_source
                != GEOMETRY_SOURCE_FIELD_MEASURED
            ):
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point outside the "
                    "offline Mock pipeline requires field_measured geometry"
                )
            if adaptive.box_clearance.interior_size_m is not None:
                half_extent = float(box_relative.gripper_max_outer_m) * 0.5
                usable_x = (
                    float(adaptive.box_clearance.interior_size_m[0]) * 0.5
                    - half_extent
                    - float(adaptive.box_clearance.wall_clearance_m)
                )
                usable_y = (
                    float(adaptive.box_clearance.interior_size_m[1]) * 0.5
                    - half_extent
                    - float(adaptive.box_clearance.wall_clearance_m)
                )
                if usable_x <= 0.0 or usable_y <= 0.0:
                    raise ValueError(
                        "adaptive_inverse.box_relative_five_point gripper "
                        "envelope and wall clearance leave no usable box points"
                    )
            legacy_fields = {
                "place_poses": self.task_b.place_poses,
                "place_poses_file": self.task_b.place_poses_file,
                "place_names": self.task_b.place_names,
                "point_grasp_z_m": self.task_b.point_grasp_z_m,
                "point_grasp_xy_from_observation": (
                    self.task_b.point_grasp_xy_from_observation
                ),
                "point_grasp_xy_m": self.task_b.point_grasp_xy_m,
                "point_grasp_xy_offset_m": self.task_b.point_grasp_xy_offset_m,
                "point_grasp_jaw_tip_object_offset_m": (
                    self.task_b.point_grasp_jaw_tip_object_offset_m
                ),
                "point_grasp_rpy": self.task_b.point_grasp_rpy,
                "point_vertical_grasp_descent": (
                    self.task_b.point_vertical_grasp_descent
                ),
                "point_vertical_grasp_approach_height_m": (
                    self.task_b.point_vertical_grasp_approach_height_m
                ),
                "point_observation_joints_deg": (
                    self.task_b.point_observation_joints_deg
                ),
            }
            configured_legacy = [
                name for name, value in legacy_fields.items() if value
            ]
            if configured_legacy:
                raise ValueError(
                    "adaptive_inverse.box_relative_five_point forbids all "
                    "legacy five-point data; remove task_b."
                    + ", task_b.".join(configured_legacy)
                )
        if adaptive.arbitrary_placement:
            if not adaptive.enabled:
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement requires "
                    "adaptive_inverse.enabled=true"
                )
            if self.task.mode is not TaskMode.B:
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement is restricted to "
                    "Task B"
                )
            if not self.task_b.return_to_pick:
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement requires "
                    "task_b.return_to_pick=true"
                )
            if not adaptive.candidates.use_visual_grasp_z:
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement requires "
                    "adaptive_inverse.candidates.use_visual_grasp_z=true"
                )
            if not adaptive.box_clearance.required:
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement requires "
                    "adaptive_inverse.box_clearance.required=true"
                )
            if (
                adaptive.box_clearance.geometry_source
                == GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC
                and not offline_only
            ):
                raise ValueError(
                    "adaptive_inverse.box_clearance synthetic geometry is "
                    "restricted to the fully offline Mock pipeline"
                )
            if (
                not offline_only
                and adaptive.box_clearance.geometry_source
                != GEOMETRY_SOURCE_FIELD_MEASURED
            ):
                raise ValueError(
                    "adaptive_inverse.box_clearance outside the offline Mock "
                    "pipeline requires field_measured geometry"
                )
            legacy_fields = {
                "place_poses": self.task_b.place_poses,
                "place_poses_file": self.task_b.place_poses_file,
                "place_names": self.task_b.place_names,
                "point_grasp_z_m": self.task_b.point_grasp_z_m,
                "point_grasp_xy_from_observation": (
                    self.task_b.point_grasp_xy_from_observation
                ),
                "point_grasp_xy_m": self.task_b.point_grasp_xy_m,
                "point_grasp_xy_offset_m": self.task_b.point_grasp_xy_offset_m,
                "point_grasp_jaw_tip_object_offset_m": (
                    self.task_b.point_grasp_jaw_tip_object_offset_m
                ),
                "point_grasp_rpy": self.task_b.point_grasp_rpy,
                "point_vertical_grasp_descent": (
                    self.task_b.point_vertical_grasp_descent
                ),
                "point_vertical_grasp_approach_height_m": (
                    self.task_b.point_vertical_grasp_approach_height_m
                ),
                "point_observation_joints_deg": (
                    self.task_b.point_observation_joints_deg
                ),
            }
            configured_legacy = [
                name for name, value in legacy_fields.items() if value
            ]
            if configured_legacy:
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement forbids all legacy "
                    "five-point data; remove task_b."
                    + ", task_b.".join(configured_legacy)
                )
        if (
            not isfinite(self.task_b.initial_pose_tolerance_deg)
            or self.task_b.initial_pose_tolerance_deg <= 0.0
        ):
            raise ValueError(
                "task_b.initial_pose_tolerance_deg must be positive and finite"
            )
        if self.task_b.pick_pose_tolerance_m <= 0.0:
            raise ValueError("task_b.pick_pose_tolerance_m must be positive")
        if self.task_b.point_assignment_margin_m < 0.0:
            raise ValueError(
                "task_b.point_assignment_margin_m must be >= 0"
            )
        for index, pose in enumerate(self.place_poses):
            if not all(isfinite(value) for value in pose):
                raise ValueError(
                    f"task_b reference point {index + 1} must contain finite values"
                )
        for first in range(len(self.place_poses)):
            for second in range(first + 1, len(self.place_poses)):
                separation = hypot(
                    self.place_poses[first][0] - self.place_poses[second][0],
                    self.place_poses[first][1] - self.place_poses[second][1],
                )
                minimum = 2.0 * self.task_b.point_assignment_margin_m
                if separation <= minimum:
                    raise ValueError(
                        "task_b reference points "
                        f"{first + 1} and {second + 1} are only "
                        f"{separation:.3f} m apart; point assignment needs "
                        f"more than {minimum:.3f} m separation"
                    )
        if self.task_b.next_point_timeout_s < 0.0:
            raise ValueError("task_b.next_point_timeout_s must be >= 0")
        if self.task_b.next_point_poll_s <= 0.0:
            raise ValueError("task_b.next_point_poll_s must be positive")
        if self.task_b.lift_hold_s < 0.0:
            raise ValueError("task_b.lift_hold_s must be >= 0")
        point_count = len(self.place_poses)
        for name, values in (
            ("point_grasp_z_m", self.task_b.point_grasp_z_m),
            (
                "point_grasp_xy_from_observation",
                self.task_b.point_grasp_xy_from_observation,
            ),
            ("point_grasp_xy_m", self.task_b.point_grasp_xy_m),
            (
                "point_grasp_xy_offset_m",
                self.task_b.point_grasp_xy_offset_m,
            ),
            (
                "point_grasp_jaw_tip_object_offset_m",
                self.task_b.point_grasp_jaw_tip_object_offset_m,
            ),
            ("point_grasp_rpy", self.task_b.point_grasp_rpy),
            (
                "point_vertical_grasp_descent",
                self.task_b.point_vertical_grasp_descent,
            ),
            (
                "point_vertical_grasp_approach_height_m",
                self.task_b.point_vertical_grasp_approach_height_m,
            ),
            ("point_observation_joints_deg", self.task_b.point_observation_joints_deg),
        ):
            if values and len(values) != point_count:
                raise ValueError(
                    f"task_b.{name} must be empty or contain exactly "
                    f"{point_count} entries (one per configured point)"
                )
        for index, value in enumerate(self.task_b.point_grasp_z_m):
            if value is not None and not isfinite(value):
                raise ValueError(
                    f"task_b.point_grasp_z_m entry {index + 1} must be finite"
                )
        for index, value in enumerate(
            self.task_b.point_grasp_xy_from_observation
        ):
            if not isinstance(value, bool):
                raise ValueError(
                    "task_b.point_grasp_xy_from_observation entry "
                    f"{index + 1} must be boolean"
                )
        for index, value in enumerate(self.task_b.point_grasp_xy_m):
            if value is None:
                continue
            if not all(isfinite(item) for item in value):
                raise ValueError(
                    f"task_b.point_grasp_xy_m entry {index + 1} must contain "
                    "finite values"
                )
            if (
                index < len(self.task_b.point_grasp_xy_from_observation)
                and self.task_b.point_grasp_xy_from_observation[index]
            ):
                raise ValueError(
                    f"task_b.point_grasp_xy_m entry {index + 1} conflicts with "
                    "point_grasp_xy_from_observation"
                )
        for index, value in enumerate(self.task_b.point_grasp_xy_offset_m):
            if not all(isfinite(item) for item in value):
                raise ValueError(
                    f"task_b.point_grasp_xy_offset_m entry {index + 1} must "
                    "contain finite values"
                )
        for index, value in enumerate(
            self.task_b.point_grasp_jaw_tip_object_offset_m
        ):
            if value is None:
                continue
            if not all(isfinite(item) for item in value):
                raise ValueError(
                    "task_b.point_grasp_jaw_tip_object_offset_m entry "
                    f"{index + 1} must contain finite values"
                )
            if max(abs(float(item)) for item in value) > 0.10:
                raise ValueError(
                    "task_b.point_grasp_jaw_tip_object_offset_m entry "
                    f"{index + 1} exceeds the 0.10 m object-offset safety limit"
                )
        for index, value in enumerate(self.task_b.point_grasp_rpy):
            if value is not None and not all(isfinite(item) for item in value):
                raise ValueError(
                    f"task_b.point_grasp_rpy entry {index + 1} must contain "
                    "finite values"
                )
        for index, value in enumerate(
            self.task_b.point_vertical_grasp_descent
        ):
            if not isinstance(value, bool):
                raise ValueError(
                    "task_b.point_vertical_grasp_descent entry "
                    f"{index + 1} must be boolean"
                )
        for index, value in enumerate(
            self.task_b.point_vertical_grasp_approach_height_m
        ):
            if value is not None and (
                not isfinite(value) or value <= 0.0
            ):
                raise ValueError(
                    "task_b.point_vertical_grasp_approach_height_m entry "
                    f"{index + 1} must be positive and finite"
                )
        for index, value in enumerate(self.task_b.point_observation_joints_deg):
            if value is not None and not all(isfinite(item) for item in value):
                raise ValueError(
                    "task_b.point_observation_joints_deg entry "
                    f"{index + 1} must contain finite values"
                )
        if (
            not isfinite(self.task_b.vertical_grasp_approach_height_m)
            or self.task_b.vertical_grasp_approach_height_m <= 0.0
        ):
            raise ValueError(
                "task_b.vertical_grasp_approach_height_m must be positive "
                "and finite"
            )
        if self.task_b.observation_pose_tolerance_deg <= 0.0:
            raise ValueError(
                "task_b.observation_pose_tolerance_deg must be > 0"
            )
        if (
            self.task_b.return_to_initial_after_each_pick
            and not self.task_b.initial_pose_file
        ):
            raise ValueError(
                "task_b.return_to_initial_after_each_pick needs "
                "task_b.initial_pose_file"
            )
        if (
            self.task_b.return_to_initial_after_each_pick
            and self.task_b.return_to_observation_after_each_pick
        ):
            raise ValueError(
                "task_b.return_to_initial_after_each_pick and "
                "task_b.return_to_observation_after_each_pick are mutually "
                "exclusive"
            )
        if (
            self.task_b.return_to_observation_after_each_pick
            and not self.task_b.point_observation_joints_deg
        ):
            raise ValueError(
                "task_b.return_to_observation_after_each_pick needs "
                "task_b.point_observation_joints_deg"
            )
        if (
            self.task_b.return_to_observation_after_each_pick
            and any(
                value is None
                for value in self.task_b.point_observation_joints_deg
            )
        ):
            raise ValueError(
                "task_b.return_to_observation_after_each_pick requires a "
                "concrete observation posture for every configured point"
            )
        if (
            self.task_b.return_to_initial_after_task
            and not self.task_b.initial_pose_file
        ):
            raise ValueError(
                "task_b.return_to_initial_after_task needs "
                "task_b.initial_pose_file"
            )
        if self.target.policy not in {"any", "known", "known_and_unseen", "ordered"}:
            raise ValueError("unsupported target.policy")
        if self.target.policy == "ordered" and not (self.target.order or self.target.order_file):
            raise ValueError("ordered policy needs target.order or target.order_file")
        if self.target.policy == "known" and not self.target.known_classes:
            raise ValueError("known policy needs target.known_classes")
        if not 0.0 <= self.target.unknown_min_graspability <= 1.0:
            raise ValueError("unknown_min_graspability must be in [0, 1]")
        if self.target.max_attempts_per_target < 1:
            raise ValueError("max_attempts_per_target must be >= 1")
        if self.pose.camera_frame not in {"base", "flange"}:
            raise ValueError("pose.camera_frame must be base or flange")
        if self.pose.robot_pose_frame not in {"unknown", "flange", "tcp"}:
            raise ValueError("pose.robot_pose_frame must be unknown, flange or tcp")
        if not all(isfinite(value) for value in self.pose.tcp_offset_m):
            raise ValueError("pose.tcp_offset_m must contain finite values")
        if self.pose.gripper_tip_offset_flange_m is not None:
            if not all(
                isfinite(value)
                for value in self.pose.gripper_tip_offset_flange_m
            ):
                raise ValueError(
                    "pose.gripper_tip_offset_flange_m must contain finite "
                    "values"
                )
            if not any(
                abs(value) > 1e-9
                for value in self.pose.gripper_tip_offset_flange_m
            ):
                raise ValueError(
                    "pose.gripper_tip_offset_flange_m must be non-zero"
                )
        if adaptive.enabled and adaptive.candidates.use_visual_grasp_z:
            if self.pose.gripper_tip_offset_flange_m is None:
                raise ValueError(
                    "adaptive visual grasp requires "
                    "pose.gripper_tip_offset_flange_m"
                )
        if not 0.0 <= adaptive.min_pose_confidence <= 1.0:
            raise ValueError(
                "adaptive_inverse.min_pose_confidence must be in [0, 1]"
            )
        if adaptive.pca.min_valid_points < 3:
            raise ValueError(
                "adaptive_inverse.pca.min_valid_points must be >= 3"
            )
        if (
            not isfinite(adaptive.pca.min_depth_m)
            or not isfinite(adaptive.pca.max_depth_m)
            or adaptive.pca.min_depth_m <= 0.0
            or adaptive.pca.max_depth_m <= adaptive.pca.min_depth_m
        ):
            raise ValueError(
                "adaptive_inverse.pca depth range must be positive, finite "
                "and increasing"
            )
        if not 0.0 <= adaptive.pca.trim_percentile < 50.0:
            raise ValueError(
                "adaptive_inverse.pca.trim_percentile must be in [0, 50)"
            )
        if not isfinite(adaptive.pca.min_axis_ratio) or adaptive.pca.min_axis_ratio < 1.0:
            raise ValueError(
                "adaptive_inverse.pca.min_axis_ratio must be >= 1"
            )
        if adaptive.pca.support_plane_min_points < 3:
            raise ValueError(
                "adaptive_inverse.pca.support_plane_min_points must be >= 3"
            )
        if (
            not isfinite(adaptive.pca.support_plane_max_rms_m)
            or adaptive.pca.support_plane_max_rms_m <= 0.0
        ):
            raise ValueError(
                "adaptive_inverse.pca.support_plane_max_rms_m must be "
                "positive and finite"
            )
        if not 0.0 <= adaptive.pca.support_plane_min_inlier_ratio <= 1.0:
            raise ValueError(
                "adaptive_inverse.pca.support_plane_min_inlier_ratio must be "
                "in [0, 1]"
            )
        if not 0.0 < adaptive.pca.support_plane_min_normal_z <= 1.0:
            raise ValueError(
                "adaptive_inverse.pca.support_plane_min_normal_z must be in "
                "(0, 1]"
            )
        if (
            not isfinite(adaptive.pca.support_plane_depth_window_m)
            or adaptive.pca.support_plane_depth_window_m <= 0.0
        ):
            raise ValueError(
                "adaptive_inverse.pca.support_plane_depth_window_m must be "
                "positive and finite"
            )
        if adaptive.pca.support_plane_guard_dilation_px < 0:
            raise ValueError(
                "adaptive_inverse.pca.support_plane_guard_dilation_px must "
                "be >= 0"
            )
        if (
            not isfinite(adaptive.pca.support_plane_ring_padding_ratio)
            or adaptive.pca.support_plane_ring_padding_ratio <= 0.0
        ):
            raise ValueError(
                "adaptive_inverse.pca.support_plane_ring_padding_ratio must "
                "be positive and finite"
            )
        if (
            not isfinite(adaptive.pca.support_plane_ransac_threshold_m)
            or adaptive.pca.support_plane_ransac_threshold_m <= 0.0
        ):
            raise ValueError(
                "adaptive_inverse.pca.support_plane_ransac_threshold_m must "
                "be positive and finite"
            )
        if adaptive.pca.support_plane_ransac_iterations < 1:
            raise ValueError(
                "adaptive_inverse.pca.support_plane_ransac_iterations must "
                "be >= 1"
            )
        if adaptive.pca.support_plane_max_samples < 3:
            raise ValueError(
                "adaptive_inverse.pca.support_plane_max_samples must be >= 3"
            )
        if adaptive.candidates.max_count < 1:
            raise ValueError(
                "adaptive_inverse.candidates.max_count must be >= 1"
            )
        if not adaptive.candidates.yaw_offsets_deg:
            raise ValueError(
                "adaptive_inverse.candidates.yaw_offsets_deg must not be empty"
            )
        if not all(
            isfinite(value) for value in adaptive.candidates.yaw_offsets_deg
        ):
            raise ValueError(
                "adaptive_inverse.candidates.yaw_offsets_deg must contain "
                "finite values"
            )
        if not adaptive.candidates.pitch_offsets_deg:
            raise ValueError(
                "adaptive_inverse.candidates.pitch_offsets_deg must not be empty"
            )
        if not all(
            isfinite(value) for value in adaptive.candidates.pitch_offsets_deg
        ):
            raise ValueError(
                "adaptive_inverse.candidates.pitch_offsets_deg must contain "
                "finite values"
            )
        if any(
            abs(value) > 90.0
            for value in adaptive.candidates.pitch_offsets_deg
        ):
            raise ValueError(
                "adaptive_inverse.candidates.pitch_offsets_deg values must be "
                "in [-90, 90]"
            )
        if adaptive.candidates.lock_taught_orientation and (
            len(adaptive.candidates.yaw_offsets_deg) != 1
            or abs(adaptive.candidates.yaw_offsets_deg[0]) > 1e-12
            or len(adaptive.candidates.pitch_offsets_deg) != 1
            or abs(adaptive.candidates.pitch_offsets_deg[0]) > 1e-12
        ):
            raise ValueError(
                "adaptive_inverse.candidates.lock_taught_orientation=true "
                "requires yaw_offsets_deg=[0.0] and pitch_offsets_deg=[0.0]"
            )
        if adaptive.candidates.taught_pitch_offsets_deg:
            if not adaptive.candidates.lock_taught_orientation:
                raise ValueError(
                    "adaptive_inverse.candidates.taught_pitch_offsets_deg "
                    "requires lock_taught_orientation=true"
                )
            if not all(
                isfinite(value) and abs(value) <= 20.0
                for value in adaptive.candidates.taught_pitch_offsets_deg
            ):
                raise ValueError(
                    "adaptive_inverse.candidates.taught_pitch_offsets_deg "
                    "values must be finite and in [-20, 20]"
                )
        if adaptive.candidates.lock_taught_orientation and (
            adaptive.arbitrary_placement
            or adaptive.box_relative_five_point.enabled
        ):
            raise ValueError(
                "adaptive_inverse.candidates.lock_taught_orientation is only "
                "supported for fixed-point adaptive grasping"
            )
        if adaptive.candidates.require_jaw_tip_object_offset and (
            not adaptive.enabled
            or adaptive.arbitrary_placement
            or adaptive.box_relative_five_point.enabled
        ):
            raise ValueError(
                "adaptive_inverse.candidates.require_jaw_tip_object_offset "
                "is only supported for enabled fixed-point adaptive grasping"
            )
        configured_object_offsets = tuple(
            value
            for value in self.task_b.point_grasp_jaw_tip_object_offset_m
            if value is not None
        )
        if (
            configured_object_offsets
            and self.pose.gripper_tip_offset_flange_m is None
        ):
            raise ValueError(
                "task_b.point_grasp_jaw_tip_object_offset_m requires "
                "pose.gripper_tip_offset_flange_m"
            )
        if not (
            isfinite(adaptive.candidates.axis_alignment_offset_deg)
            and 0.0
            < adaptive.candidates.max_orientation_change_deg
            <= 180.0
        ):
            raise ValueError(
                "adaptive_inverse.candidates max_orientation_change_deg must "
                "be in (0, 180]"
            )
        if adaptive.candidates.approach_height_m is not None and (
            not isfinite(adaptive.candidates.approach_height_m)
            or adaptive.candidates.approach_height_m <= 0.0
        ):
            raise ValueError(
                "adaptive_inverse.candidates.approach_height_m must be "
                "positive and finite when configured"
            )
        if (
            not isfinite(adaptive.candidates.jaw_tip_clearance_m)
            or adaptive.candidates.jaw_tip_clearance_m < 0.0
        ):
            raise ValueError(
                "adaptive_inverse.candidates.jaw_tip_clearance_m must be "
                "finite and >= 0"
            )
        minimum_grasp_flange_z_m = (
            adaptive.candidates.minimum_grasp_flange_z_m
        )
        if minimum_grasp_flange_z_m is not None and not isfinite(
            minimum_grasp_flange_z_m
        ):
            raise ValueError(
                "adaptive_inverse.candidates.minimum_grasp_flange_z_m must "
                "be finite when configured"
            )
        fixed_point_live = (
            adaptive.enabled
            and self.task.mode is TaskMode.B
            and not adaptive.arbitrary_placement
            and not adaptive.box_relative_five_point.enabled
            and not offline_only
        )
        if fixed_point_live:
            if minimum_grasp_flange_z_m is None:
                raise ValueError(
                    "live fixed-point adaptive grasping requires "
                    "adaptive_inverse.candidates.minimum_grasp_flange_z_m"
                )
            if adaptive.candidates.require_jaw_tip_object_offset:
                learned_offsets = (
                    self.task_b.point_grasp_jaw_tip_object_offset_m
                )
                if (
                    len(learned_offsets) != len(self.place_poses)
                    or any(value is None for value in learned_offsets)
                ):
                    raise ValueError(
                        "live fixed-point adaptive grasping requires a concrete "
                        "task_b.point_grasp_jaw_tip_object_offset_m entry for "
                        "every configured point when "
                        "require_jaw_tip_object_offset=true"
                    )
            configured_point_depths = tuple(
                value
                for value in self.task_b.point_grasp_z_m
                if value is not None
            )
            if (
                adaptive.candidates.use_visual_grasp_z
                and configured_point_depths
            ):
                raise ValueError(
                    "live fixed-point adaptive grasping cannot combine "
                    "candidates.use_visual_grasp_z=true with "
                    "task_b.point_grasp_z_m; the taught depth would be silently "
                    "ignored"
                )
            if (
                adaptive.candidates.use_visual_grasp_z
                and not adaptive.pca.support_plane_required
            ):
                raise ValueError(
                    "live fixed-point adaptive visual grasp Z requires "
                    "adaptive_inverse.pca.support_plane_required=true"
                )
            effective_depths = configured_point_depths
            if not effective_depths and self.motion.grasp_z_m is not None:
                effective_depths = (float(self.motion.grasp_z_m),)
            if not effective_depths:
                raise ValueError(
                    "live fixed-point adaptive grasping requires a concrete "
                    "task_b.point_grasp_z_m or motion.grasp_z_m"
                )
            if any(
                float(value) < minimum_grasp_flange_z_m
                for value in effective_depths
            ):
                raise ValueError(
                    "configured fixed grasp flange Z is below "
                    "adaptive_inverse.candidates.minimum_grasp_flange_z_m"
                )
        if adaptive.tilted_visual_grasp:
            if not adaptive.enabled:
                raise ValueError(
                    "adaptive_inverse.tilted_visual_grasp requires "
                    "adaptive_inverse.enabled=true"
                )
            if not adaptive.arbitrary_placement:
                raise ValueError(
                    "adaptive_inverse.tilted_visual_grasp requires "
                    "adaptive_inverse.arbitrary_placement=true"
                )
            if self.task.mode is not TaskMode.B:
                raise ValueError(
                    "adaptive_inverse.tilted_visual_grasp is restricted to "
                    "Task B"
                )
            if not self.task_b.return_to_pick:
                raise ValueError(
                    "adaptive_inverse.tilted_visual_grasp requires "
                    "task_b.return_to_pick=true"
                )
            if not adaptive.candidates.use_visual_grasp_z:
                raise ValueError(
                    "adaptive_inverse.tilted_visual_grasp requires "
                    "adaptive_inverse.candidates.use_visual_grasp_z=true"
                )
            if not any(
                abs(value) > 1e-9
                for value in adaptive.candidates.pitch_offsets_deg
            ):
                raise ValueError(
                    "adaptive_inverse.tilted_visual_grasp requires "
                    "adaptive_inverse.candidates.pitch_offsets_deg to contain "
                    "at least one non-zero tilt"
                )
        if adaptive.arbitrary_placement:
            if (
                not adaptive.tilted_visual_grasp
                and any(
                    abs(value) > 1e-9
                    for value in adaptive.candidates.pitch_offsets_deg
                )
            ):
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement requires "
                    "adaptive_inverse.candidates.pitch_offsets_deg to contain "
                    "only 0.0; set "
                    "adaptive_inverse.tilted_visual_grasp=true to allow tilt"
                )
            if self.motion.grasp_rpy is None:
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement requires an explicit "
                    "motion.grasp_rpy as the taught baseline orientation"
                )
            if not adaptive.tilted_visual_grasp:
                tool_z = rpy_to_matrix(self.motion.grasp_rpy)[:, 2]
                vertical_error = max(
                    abs(float(tool_z[0])),
                    abs(float(tool_z[1])),
                    abs(float(tool_z[2]) + 1.0),
                )
                if vertical_error > 1e-7:
                    raise ValueError(
                        "adaptive_inverse.arbitrary_placement requires "
                        "motion.grasp_rpy tool Z axis vertical down; "
                        f"axis=({tool_z[0]:.9f}, {tool_z[1]:.9f}, "
                        f"{tool_z[2]:.9f})"
                    )
        if (
            adaptive.arbitrary_placement
            and adaptive.candidates.use_visual_grasp_z
            and not adaptive.pca.support_plane_required
        ):
            raise ValueError(
                "adaptive arbitrary-placement visual grasp requires "
                "adaptive_inverse.pca.support_plane_required=true"
            )
        selection = adaptive.selection
        if not isfinite(selection.min_sigma_margin) or selection.min_sigma_margin < 0.0:
            raise ValueError(
                "adaptive_inverse.selection.min_sigma_margin must be >= 0"
            )
        for name in (
            "sigma_weight",
            "joint_travel_weight",
            "orientation_weight",
        ):
            value = float(getattr(selection, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"adaptive_inverse.selection.{name} must be finite and >= 0"
                )
        if adaptive.enabled:
            if self.task.mode is not TaskMode.B:
                raise ValueError(
                    "adaptive_inverse is currently restricted to Task B"
                )
            if (
                adaptive.arbitrary_placement
                and not offline_only
                and not adaptive.require_validated_extrinsics
            ):
                raise ValueError(
                    "adaptive_inverse.arbitrary_placement requires "
                    "require_validated_extrinsics=true outside the offline "
                    "mock pipeline"
                )
            if self.pose.camera_frame != "flange":
                raise ValueError(
                    "adaptive_inverse requires pose.camera_frame=flange"
                )
            if not self.pose.extrinsics_file:
                raise ValueError(
                    "adaptive_inverse requires pose.extrinsics_file"
                )
            if self.pose.robot_pose_frame not in {"flange", "tcp"}:
                raise ValueError(
                    "adaptive_inverse requires pose.robot_pose_frame to be "
                    "explicitly flange or tcp"
                )
            if self.vision.backend not in {"mock", "realsense_yolo"}:
                raise ValueError(
                    "adaptive_inverse needs a depth-capable vision backend "
                    "(mock or realsense_yolo)"
                )
            if adaptive.require_validated_extrinsics:
                validated, calibration_id = _load_extrinsics_metadata(
                    self.pose.extrinsics_file
                )
                if not validated or not calibration_id:
                    raise ValueError(
                        "adaptive_inverse requires validated extrinsics with "
                        "a non-empty calibration_id"
                    )
        if self.motion.backend not in {"mock", "robotmove"}:
            raise ValueError("motion.backend must be mock or robotmove")
        if self.motion.motion_mode not in {"joint_ik", "cartesian"}:
            raise ValueError("motion.motion_mode must be joint_ik or cartesian")
        if any(a >= b for a, b in zip(self.motion.workspace_min, self.motion.workspace_max)):
            raise ValueError("workspace_min must be smaller than workspace_max")
        if self.motion.workspace_margin < 0.0:
            raise ValueError("motion.workspace_margin must be >= 0")
        if self.motion.approach_height < 0.0 or self.motion.lift_height < 0.0:
            raise ValueError("approach_height and lift_height must be >= 0")
        if self.motion.grasp_z_m is not None and not isfinite(self.motion.grasp_z_m):
            raise ValueError("motion.grasp_z_m must be finite when configured")
        if self.motion.grasp_rpy is not None and not all(
            isfinite(value) for value in self.motion.grasp_rpy
        ):
            raise ValueError("motion.grasp_rpy must contain finite values")
        if not isfinite(self.motion.speed_mps) or self.motion.speed_mps <= 0.0:
            raise ValueError("motion.speed_mps must be positive and finite")
        if (
            not isfinite(self.motion.joint_speed_rad_s)
            or self.motion.joint_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "motion.joint_speed_rad_s must be positive and finite"
            )
        if (
            not isfinite(self.motion.joint_speed_safety_limit_rad_s)
            or self.motion.joint_speed_safety_limit_rad_s <= 0.0
        ):
            raise ValueError(
                "motion.joint_speed_safety_limit_rad_s must be positive "
                "and finite"
            )
        if (
            self.motion.joint_speed_rad_s
            > self.motion.joint_speed_safety_limit_rad_s
        ):
            raise ValueError(
                "motion.joint_speed_rad_s must not exceed "
                "motion.joint_speed_safety_limit_rad_s"
            )
        if self.motion.orientation not in {"fixed", "current"}:
            raise ValueError("motion.orientation must be fixed or current")
        if not 0.0 < self.motion.ik_position_tolerance_m <= 0.002:
            raise ValueError(
                "motion.ik_position_tolerance_m must be in (0, 0.002]"
            )
        if not 0.0 < self.motion.ik_orientation_tolerance_rad <= 0.005:
            raise ValueError(
                "motion.ik_orientation_tolerance_rad must be in (0, 0.005]"
            )
        if not 0.0 < self.motion.joint_max_step_deg <= 5.0:
            raise ValueError("motion.joint_max_step_deg must be in (0, 5]")
        if not 0.0 < self.motion.joint_tolerance_deg <= 2.0:
            raise ValueError("motion.joint_tolerance_deg must be in (0, 2]")
        if not 0.0 < self.motion.joint_limit_margin_deg <= 10.0:
            raise ValueError("motion.joint_limit_margin_deg must be in (0, 10]")
        for index, seed in enumerate(self.motion.ik_fallback_seeds_deg):
            if len(seed) != 6:
                raise ValueError(
                    "motion.ik_fallback_seeds_deg"
                    f"[{index}] must contain exactly six joints"
                )
            if not all(isfinite(float(value)) for value in seed):
                raise ValueError(
                    "motion.ik_fallback_seeds_deg"
                    f"[{index}] contains non-finite values"
                )
            seed_rad = tuple(radians(float(value)) for value in seed)
            if not joints_within_limits(
                seed_rad,
                margin_rad=radians(
                    float(self.motion.joint_limit_margin_deg)
                ),
            ):
                raise ValueError(
                    "motion.ik_fallback_seeds_deg"
                    f"[{index}] violates joint limits or the configured "
                    "limit margin"
                )
        if self.motion.joint_move_timeout_s <= 0.0:
            raise ValueError("motion.joint_move_timeout_s must be positive")
        if not 0.1 <= self.motion.joint_reject_timeout_s <= 10.0:
            raise ValueError("motion.joint_reject_timeout_s must be in [0.1, 10]")
        if not 0.0 < self.motion.cartesian_step_m <= 0.02:
            raise ValueError("motion.cartesian_step_m must be in (0, 0.02]")
        if not 0.0 < self.motion.cartesian_min_step_m <= self.motion.cartesian_step_m:
            raise ValueError(
                "motion.cartesian_min_step_m must be in "
                "(0, cartesian_step_m]"
            )
        if not 0.0 < self.motion.cartesian_orientation_step_deg <= 5.0:
            raise ValueError(
                "motion.cartesian_orientation_step_deg must be in (0, 5]"
            )
        if (
            not 0.0
            < self.motion.cartesian_min_orientation_step_deg
            <= self.motion.cartesian_orientation_step_deg
        ):
            raise ValueError(
                "motion.cartesian_min_orientation_step_deg must be in "
                "(0, cartesian_orientation_step_deg]"
            )
        if self.motion.cartesian_max_waypoints < 2:
            raise ValueError("motion.cartesian_max_waypoints must be >= 2")
        if not 0.0 < self.motion.cartesian_joint_step_deg <= 2.0:
            raise ValueError(
                "motion.cartesian_joint_step_deg must be in (0, 2]"
            )
        if (
            self.motion.cartesian_joint_branch_jump_deg
            <= self.motion.cartesian_joint_step_deg
        ):
            raise ValueError(
                "motion.cartesian_joint_branch_jump_deg must exceed "
                "cartesian_joint_step_deg"
            )
        if self.motion.cartesian_joint_acceleration_limit_deg <= 0.0:
            raise ValueError(
                "motion.cartesian_joint_acceleration_limit_deg must be positive"
            )
        if self.motion.cartesian_min_jacobian_singular_value <= 0.0:
            raise ValueError(
                "motion.cartesian_min_jacobian_singular_value must be positive"
            )
        _validate_gripper(self.gripper)
        _validate_grasp_consensus_gate(self)
        _validate_grasp_safety(self)
        if (
            self.task.mode is TaskMode.B
            and not adaptive.arbitrary_placement
            and not box_relative.enabled
        ):
            if not self.place_poses:
                raise ValueError(
                    "task B needs task_b.place_poses or task_b.place_poses_file"
                )
            if len(self.place_poses) < self.task_b.rounds:
                raise ValueError(
                    "task B needs one configured reference pick point per round"
                )


GRIPPER_BACKENDS = (
    "mock",
    "taska",
    "ms42dc",
    "serial",
    "tcp",
    "ros_service",
    "ros_topic",
    "ms42dc_ros",
)


def _validate_grasp_safety(config: Track1Config) -> None:
    """Validate the C/D grasp envelope and fail closed for hardware runs."""

    safety = config.motion.grasp_safety
    if config.motion.place_pose_frame not in {None, "flange", "jaw_tip"}:
        raise ValueError(
            "motion.place_pose_frame must be one of: flange, jaw_tip"
        )
    if (
        config.motion.place_pose_frame == "jaw_tip"
        and config.pose.gripper_tip_offset_flange_m is None
    ):
        raise ValueError(
            "motion.place_pose_frame=jaw_tip requires "
            "pose.gripper_tip_offset_flange_m"
        )
    optional_limits = {
        "minimum_grasp_flange_z_m": safety.minimum_grasp_flange_z_m,
        "maximum_grasp_flange_z_m": safety.maximum_grasp_flange_z_m,
        "minimum_grasp_tip_z_m": safety.minimum_grasp_tip_z_m,
        "maximum_grasp_tip_z_m": safety.maximum_grasp_tip_z_m,
        "minimum_route_flange_z_m": safety.minimum_route_flange_z_m,
        "maximum_route_flange_z_m": safety.maximum_route_flange_z_m,
        "minimum_route_tip_z_m": safety.minimum_route_tip_z_m,
        "maximum_route_tip_z_m": safety.maximum_route_tip_z_m,
        "maximum_horizontal_reach_m": safety.maximum_horizontal_reach_m,
        "maximum_tool_tilt_deg": safety.maximum_tool_tilt_deg,
        "minimum_approach_height_m": safety.minimum_approach_height_m,
        "maximum_approach_height_m": safety.maximum_approach_height_m,
        "maximum_approach_xy_error_m": safety.maximum_approach_xy_error_m,
    }
    for name, value in optional_limits.items():
        if value is not None and not isfinite(float(value)):
            raise ValueError(f"motion.grasp_safety.{name} must be finite")

    for low_name, high_name in (
        ("minimum_grasp_flange_z_m", "maximum_grasp_flange_z_m"),
        ("minimum_grasp_tip_z_m", "maximum_grasp_tip_z_m"),
        ("minimum_route_flange_z_m", "maximum_route_flange_z_m"),
        ("minimum_route_tip_z_m", "maximum_route_tip_z_m"),
        ("minimum_approach_height_m", "maximum_approach_height_m"),
    ):
        low = getattr(safety, low_name)
        high = getattr(safety, high_name)
        if low is not None and high is not None and float(low) > float(high):
            raise ValueError(
                f"motion.grasp_safety.{low_name} must not exceed {high_name}"
            )
    if (
        safety.maximum_horizontal_reach_m is not None
        and safety.maximum_horizontal_reach_m <= 0.0
    ):
        raise ValueError(
            "motion.grasp_safety.maximum_horizontal_reach_m must be positive"
        )
    if (
        safety.maximum_tool_tilt_deg is not None
        and not 0.0 <= safety.maximum_tool_tilt_deg < 90.0
    ):
        raise ValueError(
            "motion.grasp_safety.maximum_tool_tilt_deg must be in [0, 90)"
        )
    if (
        safety.maximum_approach_xy_error_m is not None
        and safety.maximum_approach_xy_error_m < 0.0
    ):
        raise ValueError(
            "motion.grasp_safety.maximum_approach_xy_error_m must be >= 0"
        )
    if (
        safety.minimum_approach_height_m is not None
        and safety.minimum_approach_height_m <= 0.0
    ):
        raise ValueError(
            "motion.grasp_safety.minimum_approach_height_m must be positive"
        )
    if (
        safety.maximum_approach_height_m is not None
        and safety.maximum_approach_height_m <= 0.0
    ):
        raise ValueError(
            "motion.grasp_safety.maximum_approach_height_m must be positive"
        )

    if config.task.mode not in {TaskMode.C, TaskMode.D}:
        return
    if not config.motion.dry_run and not safety.enabled:
        raise ValueError(
            "Task C/D live motion requires motion.grasp_safety.enabled=true"
        )
    if not safety.enabled:
        return
    if config.motion.place_pose_frame is None:
        raise ValueError(
            "Task C/D grasp safety requires an explicit "
            "motion.place_pose_frame=flange|jaw_tip so release XYZ cannot be "
            "silently interpreted in the wrong coordinate frame"
        )
    if config.pose.robot_pose_frame != "flange":
        raise ValueError(
            "Task C/D grasp safety requires pose.robot_pose_frame=flange so "
            "gripper_tip_offset_flange_m is not mixed with a controller TCP frame"
        )
    if (
        safety.require_flange_tip_offset
        and config.pose.gripper_tip_offset_flange_m is None
    ):
        raise ValueError(
            "Task C/D grasp safety requires "
            "pose.gripper_tip_offset_flange_m"
        )
    if config.motion.orientation != "fixed" or config.motion.grasp_rpy is None:
        raise ValueError(
            "Task C/D grasp safety requires motion.orientation=fixed and an "
            "explicit motion.grasp_rpy; current orientation is not admissible"
        )
    if not config.vision.require_depth:
        raise ValueError(
            "Task C/D grasp safety requires vision.require_depth=true"
        )
    if config.motion.dry_run:
        return

    required_live_limits = (
        "minimum_grasp_flange_z_m",
        "maximum_grasp_flange_z_m",
        "minimum_grasp_tip_z_m",
        "maximum_grasp_tip_z_m",
        "minimum_route_flange_z_m",
        "maximum_route_flange_z_m",
        "minimum_route_tip_z_m",
        "maximum_route_tip_z_m",
        "maximum_horizontal_reach_m",
        "maximum_tool_tilt_deg",
        "minimum_approach_height_m",
        "maximum_approach_height_m",
        "maximum_approach_xy_error_m",
    )
    missing = [
        name for name in required_live_limits if getattr(safety, name) is None
    ]
    if missing:
        raise ValueError(
            "Task C/D live motion is missing grasp safety limits: "
            + ", ".join(missing)
        )
    if config.motion.backend == "mock":
        raise ValueError("Task C/D live motion cannot use the mock motion backend")
    if config.gripper.backend == "mock":
        raise ValueError("Task C/D live motion cannot use a mock gripper")
    if not config.gripper.require_feedback:
        raise ValueError(
            "Task C/D live motion requires gripper.require_feedback=true"
        )
    if config.gripper.backend in {"ms42dc", "ms42dc_ros"} and (
        config.gripper.hold_min_angle_deg <= 0.0
    ):
        raise ValueError(
            "Task C/D live MS42DC motion requires "
            "gripper.hold_min_angle_deg > 0 for empty-grasp rejection"
        )
    if not config.pose.extrinsics_file or not Path(
        config.pose.extrinsics_file
    ).is_file():
        raise ValueError(
            "Task C/D live motion requires an existing camera calibration file"
        )


def _validate_grasp_consensus_gate(config: Track1Config) -> None:
    """Validate the fail-closed offline consensus binding.

    Values are checked even while disabled so a later one-line enable cannot
    activate a malformed or deliberately weakened configuration.  Enabling is
    restricted to hardware-idle Task C/D execution: motion remains dry-run,
    the gripper is a mock, and no autonomous search move can precede the gate.
    """

    gate = config.grasp_consensus
    if isinstance(gate.enabled, bool) is False:
        raise ValueError("grasp_consensus.enabled must be boolean")
    if (
        isinstance(gate.minimum_distinct_frames, bool)
        or not isinstance(gate.minimum_distinct_frames, int)
        or gate.minimum_distinct_frames < 4
    ):
        raise ValueError(
            "grasp_consensus.minimum_distinct_frames must be an integer >= 4"
        )

    maximums = {
        "maximum_translation_tolerance_m": (0.025, "m"),
        "maximum_jaw_rotation_tolerance_deg": (15.0, "deg"),
        "maximum_pose_translation_error_m": (0.025, "m"),
        "maximum_pose_jaw_rotation_error_deg": (15.0, "deg"),
    }
    for name, (safe_maximum, unit) in maximums.items():
        value = float(getattr(gate, name))
        if not isfinite(value) or value <= 0.0:
            raise ValueError(
                f"grasp_consensus.{name} must be positive and finite"
            )
        if value > safe_maximum + 1e-12:
            raise ValueError(
                f"grasp_consensus.{name} must not exceed the safe maximum "
                f"{safe_maximum:g} {unit}"
            )

    if not gate.enabled:
        return
    if not gate.report_file:
        raise ValueError(
            "grasp_consensus.report_file is required when the gate is enabled"
        )
    if config.task.mode not in {TaskMode.C, TaskMode.D}:
        raise ValueError("grasp_consensus is restricted to Task C/D")
    if config.motion.dry_run is not True:
        raise ValueError(
            "grasp_consensus requires motion.dry_run=true; actual robot motion "
            "is forbidden"
        )
    if config.gripper.backend != "mock":
        raise ValueError(
            "grasp_consensus requires gripper.backend=mock while the consensus "
            "gate is being validated"
        )
    if config.search.enabled is not False:
        raise ValueError(
            "grasp_consensus requires search.enabled=false so no search motion "
            "can occur before consensus is checked"
        )
    if config.pose.robot_pose_frame not in {"flange", "tcp"}:
        raise ValueError(
            "grasp_consensus requires pose.robot_pose_frame=flange or tcp"
        )
    if not all(isfinite(float(value)) for value in config.pose.tcp_offset_m):
        raise ValueError("grasp_consensus requires finite pose.tcp_offset_m")


def _validate_gripper(cfg: "GripperConfig") -> None:
    if cfg.backend not in GRIPPER_BACKENDS:
        raise ValueError(
            "gripper.backend must be one of " + ", ".join(GRIPPER_BACKENDS)
        )
    if cfg.backend == "taska" and not cfg.taska_config:
        raise ValueError("taska gripper needs gripper.taska_config")
    if cfg.backend in {"serial", "tcp"}:
        if not cfg.open_command or not cfg.close_command:
            raise ValueError(
                f"{cfg.backend} gripper needs gripper.open_command and "
                "gripper.close_command"
            )
        if cfg.command_encoding not in {"ascii", "hex"}:
            raise ValueError("gripper.command_encoding must be ascii or hex")
    if cfg.backend == "serial" and not cfg.serial_port:
        raise ValueError("serial gripper needs gripper.serial_port")
    if cfg.backend == "tcp" and (not cfg.tcp_host or cfg.tcp_port <= 0):
        raise ValueError("tcp gripper needs gripper.tcp_host and gripper.tcp_port")
    if cfg.backend == "ros_service" and not cfg.ros_service:
        raise ValueError("ros_service gripper needs gripper.ros_service")
    if cfg.backend == "ros_topic" and not cfg.ros_topic:
        raise ValueError("ros_topic gripper needs gripper.ros_topic")
    if cfg.backend in {"ms42dc", "ms42dc_ros"}:
        if cfg.backend == "ms42dc" and not cfg.serial_port:
            raise ValueError("ms42dc gripper needs gripper.serial_port")
        if cfg.subdivide not in {2, 4, 8, 16, 32}:
            raise ValueError(
                f"{cfg.backend} gripper.subdivide must be 2, 4, 8, 16 or 32"
            )
        if cfg.motion_mode not in {2, 4}:
            raise ValueError(
                f"{cfg.backend} gripper.motion_mode must be 2 (relative) "
                "or 4 (absolute)"
            )
        for name in ("open_angle_deg", "close_angle_deg"):
            if getattr(cfg, name) <= 0.0:
                raise ValueError(f"{cfg.backend} gripper.{name} must be > 0")
        if cfg.speed_rad_s <= 0.0:
            raise ValueError(f"{cfg.backend} gripper.speed_rad_s must be > 0")
        if not 0 <= cfg.device_id <= 0xFF:
            raise ValueError(f"{cfg.backend} gripper.device_id must fit uint8")
        if int(round(cfg.open_angle_deg * 10.0)) > 0xFFFF:
            raise ValueError(
                f"{cfg.backend} gripper.open_angle_deg is too large for uint16"
            )
        if int(round(cfg.close_angle_deg * 10.0)) > 0xFFFF:
            raise ValueError(
                f"{cfg.backend} gripper.close_angle_deg is too large for uint16"
            )
        if int(round(cfg.speed_rad_s * 10.0)) > 0xFFFF:
            raise ValueError(
                f"{cfg.backend} gripper.speed_rad_s is too large for uint16"
            )
    if cfg.backend == "ms42dc_ros":
        if not cfg.motor_control_topic or not cfg.motor_state_topic:
            raise ValueError(
                "ms42dc_ros gripper needs motor_control_topic and motor_state_topic"
            )
        if cfg.open_dir not in {0, 1} or cfg.close_dir not in {0, 1}:
            raise ValueError("ms42dc_ros open_dir/close_dir must be 0 or 1")
        if cfg.open_dir == cfg.close_dir:
            raise ValueError("ms42dc_ros open_dir and close_dir must differ")
        if cfg.ros_connect_timeout_s <= 0.0:
            raise ValueError("ms42dc_ros ros_connect_timeout_s must be > 0")


#: Backwards-compatible alias for code written against the C/D/E version.
CDEConfig = Track1Config


def _load_string_list(path: str, *, key: str) -> tuple[str, ...]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if isinstance(payload, list):
        return tuple(str(item) for item in payload)
    if isinstance(payload, Mapping) and key in payload:
        return tuple(str(item) for item in payload[key])
    if isinstance(payload, Mapping) and "order" in payload:
        return tuple(str(item) for item in payload["order"])
    raise ValueError(f"unsupported {key} file format: {path}")


def _load_pose_list(path: str) -> tuple[tuple[float, ...], ...]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if isinstance(payload, Mapping):
        for key in ("places", "place_poses", "poses"):
            if key in payload:
                payload = payload[key]
                break
        else:
            payload = payload.get("items", [])
    if isinstance(payload, list):
        return _pose_list(payload)
    raise ValueError(f"unsupported place pose file format: {path}")


def load_joint_pose_file(
    path: str | Path,
    *,
    expected_role: str = "",
) -> tuple[float, ...]:
    """Load and validate an absolute six-joint posture capture.

    Joint values are read from ``joint_positions_rad`` in radians.  The file is
    rejected unless it contains exactly six finite values and, when
    ``expected_role`` is supplied, the YAML ``role`` matches exactly.  Keeping
    this validation in one place prevents a malformed capture from becoming a
    motion target.
    """

    pose_path = Path(path).expanduser()
    if not pose_path.is_file():
        raise ValueError(f"joint pose file does not exist: {pose_path}")
    with open(pose_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"joint pose file must contain a mapping: {pose_path}")

    role = str(payload.get("role", "")).strip()
    if expected_role and role != expected_role:
        raise ValueError(
            f"joint pose role must be {expected_role!r}, got {role!r}: {pose_path}"
        )

    raw_joints = payload.get("joint_positions_rad")
    if not isinstance(raw_joints, (list, tuple)) or len(raw_joints) != 6:
        raise ValueError(
            "joint_positions_rad must contain exactly six values: "
            f"{pose_path}"
        )
    try:
        joints = tuple(float(value) for value in raw_joints)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"joint_positions_rad contains a non-numeric value: {pose_path}"
        ) from exc
    if not all(isfinite(value) for value in joints):
        raise ValueError(
            f"joint_positions_rad contains a non-finite value: {pose_path}"
        )
    return joints


def load_config(path: str | Path) -> Track1Config:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return Track1Config.from_dict(payload, base_dir=config_path.parent)


#: Target policy implied by each competition task.
_MODE_POLICY = {
    TaskMode.A: "any",
    TaskMode.B: "any",
    TaskMode.C: "any",
    TaskMode.D: "known_and_unseen",
    TaskMode.E: "ordered",
}


def apply_mode(config: Track1Config, mode: TaskMode | str) -> Track1Config:
    """Return a copy whose target policy matches the competition task.

    Task A is a detection/patrol demo and task B is a single-object
    pick&place, so both default to the permissive ``any`` policy.  C is
    any-object clearing, D additionally accepts unseen objects and E enforces
    the published order.
    Explicit advanced policies can still be built by calling the dataclasses
    directly.
    """

    selected = mode if isinstance(mode, TaskMode) else TaskMode(str(mode).upper())
    return replace(
        config,
        task=replace(config.task, mode=selected),
        target=replace(config.target, policy=_MODE_POLICY[selected]),
    )
