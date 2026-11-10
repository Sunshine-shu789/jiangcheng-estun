"""YAML configuration for the hardware-agnostic Track 1 framework.

One file describes the whole challenge.  The ``task.mode`` field selects the
sub-task (A/B/C/D/E) and the matching sections (``task_a``, ``task_b``) pick
up their own parameters; unset sections simply fall back to the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .domain import TaskMode


def _triple(value: Sequence[float] | None, default: Sequence[float]) -> tuple[float, float, float]:
    source = default if value is None else value
    if len(source) != 3:
        raise ValueError(f"expected a 3-value vector, got {source!r}")
    return tuple(float(v) for v in source)  # type: ignore[return-value]


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
class VisionConfig:
    backend: str = "mock"  # mock | taska_realsense | realsense_yolo
    scenario_file: str = ""
    taska_config: str = ""
    min_confidence: float = 0.35
    require_depth: bool = False
    loop_scenario: bool = False
    camera_frame_id: str = "camera_color_optical_frame"

    #: Direct RealSense + YOLO backend.  These fields deliberately mirror the
    #: proven arm_visual_probe settings instead of depending on taskA's live
    #: configuration file.
    weights: str = ""
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
    stabilizer_class_switch_hits: int = 3
    stabilizer_edge_margin_px: int = 8
    stabilizer_edge_suppress_confidence: float = 0.55

    #: Optional per-task mock scenarios, e.g. {"A": "mock_task_a_scene.yaml"}.
    scenario_files: Mapping[str, str] = field(default_factory=dict)

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
            min_confidence=float(data.get("min_confidence", 0.35)),
            require_depth=bool(data.get("require_depth", False)),
            loop_scenario=bool(data.get("loop_scenario", False)),
            camera_frame_id=str(data.get("camera_frame_id", "camera_color_optical_frame")),
            weights=_resolve_path(str(data.get("weights", "")), base_dir),
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
            stabilizer_class_switch_hits=int(
                stabilizer.get("class_switch_hits", 3)
            ),
            stabilizer_edge_margin_px=int(
                stabilizer.get("edge_margin_px", 8)
            ),
            stabilizer_edge_suppress_confidence=float(
                stabilizer.get("edge_suppress_confidence", 0.55)
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


@dataclass(frozen=True)
class TaskAConfig:
    """Task A: show the camera image, detect, annotate and score by IoU."""

    iou_threshold: float = 0.75
    ground_truth_file: str = ""
    output_dir: str = ""
    save_annotated: bool = False
    draw_labels: bool = True
    draw_ground_truth: bool = True
    max_frames: int = 0  # 0 = every frame the scenario/camera provides
    min_positions: int = 10  # competition demo: 10 different positions

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
        )


@dataclass(frozen=True)
class TaskBConfig:
    """Task B: one object picked and placed at five bin positions.

    ``place_poses`` are ``[x, y, z, rx, ry, rz]`` entries in the robot base
    frame, normally the bin's four corners plus its centre.  They may also be
    loaded from ``place_poses_file`` (a YAML list or ``{places: [...]}``).
    """

    place_poses: tuple[tuple[float, ...], ...] = ()
    place_poses_file: str = ""
    rounds: int = 5
    #: When true the same object/class is used for every round (the "same
    #: target object at five positions" wording of the competition).
    keep_same_target: bool = True
    place_names: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "TaskBConfig":
        data = dict(data or {})
        return cls(
            place_poses=_pose_list(data.get("place_poses")),
            place_poses_file=_resolve_path(str(data.get("place_poses_file", "")), base_dir),
            rounds=int(data.get("rounds", 5)),
            keep_same_target=bool(data.get("keep_same_target", True)),
            place_names=_str_tuple(data.get("place_names")),
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
    lift_height: float = 0.12
    place_pose: tuple[float, ...] | None = None
    place_approach_height: float = 0.08
    retreat_height: float = 0.08
    speed_mps: float = 0.05
    orientation: str = "fixed"  # fixed | current
    fixed_rpy: tuple[float, float, float] = (3.141592653589793, 0.0, 0.0)
    move_timeout_s: float = 30.0
    position_tolerance_m: float = 0.003
    poll_period_s: float = 0.05
    robotmove_topic: str = "RobotMove"
    robotinfo_topic: str = "RobotInfo"
    robot_info_timeout_s: float = 1.0
    joint_max_step_deg: float = 1.0
    joint_tolerance_deg: float = 0.35
    joint_start_tolerance_deg: float = 0.05
    joint_move_timeout_s: float = 8.0
    joint_reject_timeout_s: float = 2.0

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
            lift_height=float(data.get("lift_height", 0.12)),
            place_pose=_pose_value(data.get("place_pose")),
            place_approach_height=float(data.get("place_approach_height", 0.08)),
            retreat_height=float(data.get("retreat_height", 0.08)),
            speed_mps=float(data.get("speed_mps", 0.05)),
            orientation=str(data.get("orientation", "fixed")),
            fixed_rpy=_triple(
                data.get("fixed_rpy"), (3.141592653589793, 0.0, 0.0)
            ),
            move_timeout_s=float(data.get("move_timeout_s", 30.0)),
            position_tolerance_m=float(data.get("position_tolerance_m", 0.003)),
            poll_period_s=float(data.get("poll_period_s", 0.05)),
            robotmove_topic=str(data.get("robotmove_topic", "RobotMove")),
            robotinfo_topic=str(data.get("robotinfo_topic", "RobotInfo")),
            robot_info_timeout_s=float(data.get("robot_info_timeout_s", 1.0)),
            joint_max_step_deg=float(data.get("joint_max_step_deg", 1.0)),
            joint_tolerance_deg=float(data.get("joint_tolerance_deg", 0.35)),
            joint_start_tolerance_deg=float(
                data.get("joint_start_tolerance_deg", 0.05)
            ),
            joint_move_timeout_s=float(data.get("joint_move_timeout_s", 8.0)),
            joint_reject_timeout_s=float(
                data.get("joint_reject_timeout_s", 2.0)
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
    observe_frames: int = 3
    return_to_initial_on_miss: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SearchConfig":
        data = dict(data or {})
        config = cls(
            enabled=bool(data.get("enabled", True)),
            yaw_offsets_deg=tuple(
                float(v) for v in data.get("yaw_offsets_deg", (0.0,))
            ),
            pitch_down_offsets_deg=tuple(
                float(v) for v in data.get("pitch_down_offsets_deg", (0.0,))
            ),
            settle_s=float(data.get("settle_s", 0.8)),
            observe_frames=int(data.get("observe_frames", 3)),
            return_to_initial_on_miss=bool(
                data.get("return_to_initial_on_miss", True)
            ),
        )
        if config.observe_frames < 1:
            raise ValueError("search.observe_frames must be >= 1")
        if config.settle_s < 0.0:
            raise ValueError("search.settle_s must be >= 0")
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
    #   11-byte protocol over TTL serial or USB-C; see docs/GRIPPER_INTEGRATION.md
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
        )


@dataclass(frozen=True)
class Track1Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    task_a: TaskAConfig = field(default_factory=TaskAConfig)
    task_b: TaskBConfig = field(default_factory=TaskBConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)

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
            motion=MotionConfig.from_dict(data.get("motion")),
            servo=ServoConfig.from_dict(data.get("servo")),
            search=SearchConfig.from_dict(data.get("search")),
            gripper=GripperConfig.from_dict(data.get("gripper"), base_dir),
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
        """Task B place slots, from the config or an external YAML file."""

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
        if self.vision.backend not in {"mock", "taska_realsense", "realsense_yolo"}:
            raise ValueError(
                "vision.backend must be mock, taska_realsense or realsense_yolo"
            )
        if self.vision.backend == "mock" and not self.vision.scenario_file:
            raise ValueError("mock vision needs vision.scenario_file")
        if self.vision.backend == "taska_realsense" and not self.vision.taska_config:
            raise ValueError("taska_realsense vision needs vision.taska_config")
        if self.vision.backend == "realsense_yolo" and not self.vision.weights:
            raise ValueError("realsense_yolo vision needs vision.weights")
        if self.vision.anchor not in {"center", "top_center", "bottom_center"}:
            raise ValueError(
                "vision.anchor must be center, top_center or bottom_center"
            )
        if self.vision.width <= 0 or self.vision.height <= 0 or self.vision.fps <= 0:
            raise ValueError("vision width/height/fps must be positive")
        if self.vision.depth_window < 1:
            raise ValueError("vision.depth_window must be >= 1")
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
        if self.task_b.rounds < 1:
            raise ValueError("task_b.rounds must be >= 1")
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
        if self.motion.backend not in {"mock", "robotmove"}:
            raise ValueError("motion.backend must be mock or robotmove")
        if self.motion.motion_mode not in {"joint_ik", "cartesian"}:
            raise ValueError("motion.motion_mode must be joint_ik or cartesian")
        if any(a >= b for a, b in zip(self.motion.workspace_min, self.motion.workspace_max)):
            raise ValueError("workspace_min must be smaller than workspace_max")
        if self.motion.approach_height < 0.0 or self.motion.lift_height < 0.0:
            raise ValueError("approach_height and lift_height must be >= 0")
        if self.motion.speed_mps <= 0.0:
            raise ValueError("motion.speed_mps must be positive")
        if self.motion.orientation not in {"fixed", "current"}:
            raise ValueError("motion.orientation must be fixed or current")
        if not 0.0 < self.motion.joint_max_step_deg <= 5.0:
            raise ValueError("motion.joint_max_step_deg must be in (0, 5]")
        if not 0.0 < self.motion.joint_tolerance_deg <= 2.0:
            raise ValueError("motion.joint_tolerance_deg must be in (0, 2]")
        if self.motion.joint_move_timeout_s <= 0.0:
            raise ValueError("motion.joint_move_timeout_s must be positive")
        if not 0.1 <= self.motion.joint_reject_timeout_s <= 10.0:
            raise ValueError("motion.joint_reject_timeout_s must be in [0.1, 10]")
        _validate_gripper(self.gripper)
        if self.task.mode is TaskMode.B and not self.place_poses:
            raise ValueError("task B needs task_b.place_poses or task_b.place_poses_file")


GRIPPER_BACKENDS = (
    "mock",
    "taska",
    "ms42dc",
    "serial",
    "tcp",
    "ros_service",
    "ros_topic",
)


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
    if cfg.backend == "ms42dc":
        if not cfg.serial_port:
            raise ValueError("ms42dc gripper needs gripper.serial_port")
        if cfg.subdivide not in {2, 4, 8, 16, 32}:
            raise ValueError("ms42dc gripper.subdivide must be 2, 4, 8, 16 or 32")
        if cfg.motion_mode not in {2, 4}:
            raise ValueError("ms42dc gripper.motion_mode must be 2 (relative) or 4 (absolute)")
        for name in ("open_angle_deg", "close_angle_deg"):
            if getattr(cfg, name) <= 0.0:
                raise ValueError(f"ms42dc gripper.{name} must be > 0")
        if cfg.speed_rad_s <= 0.0:
            raise ValueError("ms42dc gripper.speed_rad_s must be > 0")


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

    Task A is vision-only and task B is a single-object pick&place, so both
    default to the permissive ``any`` policy.  C is any-object clearing, D
    additionally accepts unseen objects and E enforces the published order.
    Explicit advanced policies can still be built by calling the dataclasses
    directly.
    """

    selected = mode if isinstance(mode, TaskMode) else TaskMode(str(mode).upper())
    return replace(
        config,
        task=replace(config.task, mode=selected),
        target=replace(config.target, policy=_MODE_POLICY[selected]),
    )
