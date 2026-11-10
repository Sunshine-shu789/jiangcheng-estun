"""YAML configuration for the hardware-agnostic CDE pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
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


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


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
            raise ValueError("task.mode must be C, D or E") from exc
        return cls(
            mode=mode,
            max_picks=int(data.get("max_picks", 20)),
            stop_on_error=bool(data.get("stop_on_error", False)),
            sample_period_s=float(data.get("sample_period_s", 0.5)),
        )


@dataclass(frozen=True)
class VisionConfig:
    backend: str = "mock"  # mock | taska_realsense
    scenario_file: str = ""
    taska_config: str = ""
    min_confidence: float = 0.35
    require_depth: bool = False
    loop_scenario: bool = False

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, base_dir: Path
    ) -> "VisionConfig":
        data = dict(data or {})
        return cls(
            backend=str(data.get("backend", "mock")),
            scenario_file=_resolve_path(str(data.get("scenario_file", "")), base_dir),
            taska_config=_resolve_path(str(data.get("taska_config", "")), base_dir),
            min_confidence=float(data.get("min_confidence", 0.35)),
            require_depth=bool(data.get("require_depth", False)),
            loop_scenario=bool(data.get("loop_scenario", False)),
        )


@dataclass(frozen=True)
class TargetConfig:
    # any = C; known_and_unseen = D; ordered = E.  known/any are also useful
    # for B-style single-object tests.
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
            known_classes=tuple(str(v) for v in data.get("known_classes", ())),
            unknown_labels=tuple(str(v) for v in data.get("unknown_labels", ("unknown", "object"))),
            unknown_min_graspability=float(data.get("unknown_min_graspability", 0.65)),
            order=tuple(str(v) for v in data.get("order", ())),
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MotionConfig":
        data = dict(data or {})
        return cls(
            backend=str(data.get("backend", "mock")),
            dry_run=bool(data.get("dry_run", True)),
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
        )


@dataclass(frozen=True)
class GripperConfig:
    backend: str = "mock"  # mock | taska
    taska_config: str = ""
    settle_time_s: float = 0.8
    open_before_approach: bool = True

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
        )


@dataclass(frozen=True)
class CDEConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, *, base_dir: Path | None = None
    ) -> "CDEConfig":
        data = dict(data or {})
        base_dir = base_dir or Path.cwd()
        config = cls(
            task=TaskConfig.from_dict(data.get("task")),
            vision=VisionConfig.from_dict(data.get("vision"), base_dir),
            target=TargetConfig.from_dict(data.get("target"), base_dir),
            pose=PoseConfig.from_dict(data.get("pose"), base_dir),
            motion=MotionConfig.from_dict(data.get("motion")),
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
        with open(self.target.order_file, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if isinstance(payload, list):
            return tuple(str(item) for item in payload)
        if isinstance(payload, Mapping) and "order" in payload:
            return tuple(str(item) for item in payload["order"])
        raise ValueError(f"unsupported order file format: {self.target.order_file}")

    def validate(self) -> None:
        if self.task.max_picks < 1:
            raise ValueError("task.max_picks must be >= 1")
        if self.task.sample_period_s < 0.0:
            raise ValueError("task.sample_period_s must be >= 0")
        if self.vision.backend not in {"mock", "taska_realsense"}:
            raise ValueError("vision.backend must be mock or taska_realsense")
        if self.vision.backend == "mock" and not self.vision.scenario_file:
            raise ValueError("mock vision needs vision.scenario_file")
        if not 0.0 <= self.vision.min_confidence <= 1.0:
            raise ValueError("vision.min_confidence must be in [0, 1]")
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
        if any(a >= b for a, b in zip(self.motion.workspace_min, self.motion.workspace_max)):
            raise ValueError("workspace_min must be smaller than workspace_max")
        if self.motion.approach_height < 0.0 or self.motion.lift_height < 0.0:
            raise ValueError("approach_height and lift_height must be >= 0")
        if self.motion.speed_mps <= 0.0:
            raise ValueError("motion.speed_mps must be positive")
        if self.motion.orientation not in {"fixed", "current"}:
            raise ValueError("motion.orientation must be fixed or current")
        if self.gripper.backend not in {"mock", "taska"}:
            raise ValueError("gripper.backend must be mock or taska")
        if self.gripper.backend == "taska" and not self.gripper.taska_config:
            raise ValueError("taska gripper needs gripper.taska_config")


def load_config(path: str | Path) -> CDEConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return CDEConfig.from_dict(payload, base_dir=config_path.parent)


def apply_mode(config: CDEConfig, mode: TaskMode | str) -> CDEConfig:
    """Return a copy whose target policy matches competition mode C/D/E.

    Explicit advanced policies can still be built by calling the dataclasses
    directly; this helper is for the standard competition semantics.
    """
    from dataclasses import replace

    selected = mode if isinstance(mode, TaskMode) else TaskMode(str(mode).upper())
    policy = {
        TaskMode.C: "any",
        TaskMode.D: "known_and_unseen",
        TaskMode.E: "ordered",
    }[selected]
    return replace(
        config,
        task=replace(config.task, mode=selected),
        target=replace(config.target, policy=policy),
    )
