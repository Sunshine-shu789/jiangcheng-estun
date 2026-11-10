"""Vision sources and detector-output adapters.

The orchestration consumes :class:`Frame` objects.  Mock frames make the full
pick cycle runnable before the camera is mounted; the RealSense adapter is
kept behind a lazy import so this file remains testable without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import json
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
import yaml

from .adaptive import estimate_object_pose
from .config import AdaptiveInverseConfig, VisionConfig
from .detection_stabilizer import DetectionStabilizer
from .domain import (
    BoxAnnotation,
    CameraIntrinsics,
    CaptureRobotState,
    Detection,
    Frame,
    Pose,
    RobotState,
)
from .vision_runtime import LiveVisionSource


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a model file without loading the model."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """Return the ROS ZYX rotation matrix without importing the planner module."""

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class _MockWorldDetection:
    """Detector geometry fixed in the robot base frame for dynamic mock frames."""

    xyz: tuple[float, float, float]
    object_center_xyz: tuple[float, float, float]
    major_axis: tuple[float, float, float] | None
    minor_axis: tuple[float, float, float] | None


class MockVisionSource:
    """Read scripted frames from YAML or JSON.

    Example frame::

        frames:
          - frame_id: scene_0
            detections:
              - label: mug
                confidence: 0.94
                bbox: [120, 80, 250, 250]
                depth_m: 0.42
                camera_xyz: [0.02, 0.03, 0.42]
                base_xyz: [0.25, -0.10, 0.04]
                graspability: 1.0
                instance_id: mug_1
    """

    def __init__(self, config: VisionConfig, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        with open(config.scenario_file, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        raw_frames = payload.get("frames", payload if isinstance(payload, list) else [])
        if not raw_frames:
            raise ValueError(f"mock scenario has no frames: {config.scenario_file}")
        self._frames: list[Frame] = [self._parse_frame(item) for item in raw_frames]
        self._world_detections: list[tuple[_MockWorldDetection | None, ...]] = [
            self._parse_world_detections(item) for item in raw_frames
        ]
        self._robot_state_provider = None
        self._flange_camera: np.ndarray | None = None
        self._index = 0
        self._closed = False
        dynamic_count = sum(
            detection is not None
            for geometry in self._world_detections
            for detection in geometry
        )
        self._uses_world_geometry = dynamic_count > 0
        self._log(
            f"[vision:mock] loaded {len(self._frames)} frame(s), "
            f"{dynamic_count} dynamically projected detection(s)"
        )

    @staticmethod
    def _vector3(
        raw: object,
        *,
        field: str,
        allow_zero: bool = False,
    ) -> tuple[float, float, float]:
        values = tuple(float(value) for value in raw)  # type: ignore[arg-type]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"mock {field} must contain three finite values")
        if not allow_zero and float(np.linalg.norm(values)) <= 1e-12:
            raise ValueError(f"mock {field} must be non-zero")
        return values

    @classmethod
    def _parse_world_detections(
        cls,
        item: dict,
    ) -> tuple[_MockWorldDetection | None, ...]:
        geometry: list[_MockWorldDetection | None] = []
        for raw in item.get("detections", ()):
            has_world = any(
                key in raw
                for key in (
                    "world_xyz",
                    "world_object_center_xyz",
                    "world_major_axis",
                    "world_minor_axis",
                )
            )
            if not has_world:
                geometry.append(None)
                continue
            if "world_xyz" not in raw:
                raise ValueError(
                    "mock dynamic detection needs world_xyz together with its axes"
                )
            xyz = cls._vector3(raw["world_xyz"], field="world_xyz")
            center = cls._vector3(
                raw.get("world_object_center_xyz", raw["world_xyz"]),
                field="world_object_center_xyz",
            )
            major = (
                None
                if raw.get("world_major_axis") is None
                else cls._vector3(
                    raw["world_major_axis"],
                    field="world_major_axis",
                )
            )
            minor = (
                None
                if raw.get("world_minor_axis") is None
                else cls._vector3(
                    raw["world_minor_axis"],
                    field="world_minor_axis",
                )
            )
            geometry.append(
                _MockWorldDetection(
                    xyz=xyz,
                    object_center_xyz=center,
                    major_axis=major,
                    minor_axis=minor,
                )
            )
        return tuple(geometry)

    def bind_robot_state_provider(
        self,
        provider,
        *,
        extrinsics_file: str,
    ) -> None:
        """Project world-fixed mock targets through the current flange pose.

        Static scenarios remain source-compatible.  A scenario that declares
        ``world_xyz`` must bind a live pose provider before capture, otherwise it
        fails closed instead of serving stale camera coordinates.
        """

        if not callable(provider):
            raise TypeError("robot state provider must be callable")
        if not self._uses_world_geometry:
            # Static scenarios keep their historical behaviour and need neither
            # robot feedback nor extrinsics.
            self._robot_state_provider = provider
            return
        if not extrinsics_file:
            raise ValueError("dynamic mock projection needs an extrinsics file")
        from .planner import load_extrinsics

        transform = load_extrinsics(extrinsics_file)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("dynamic mock projection extrinsics must be finite 4x4")
        if abs(float(np.linalg.det(transform[:3, :3]))) <= 1e-12:
            raise ValueError("dynamic mock projection extrinsics are singular")
        self._robot_state_provider = provider
        self._flange_camera = transform
        self._log(
            "[vision:mock] bound live flange pose provider for world-fixed targets"
        )

    def _project_dynamic_frame(self, frame: Frame) -> Frame:
        geometry_index = self._index
        geometry = self._world_detections[geometry_index]
        if not any(item is not None for item in geometry):
            return frame
        if self._robot_state_provider is None or self._flange_camera is None:
            raise RuntimeError(
                "mock scenario uses world_xyz geometry but no robot pose provider "
                "and extrinsics are bound"
            )

        state = self._robot_state_provider()
        if state is None or state.pose is None:
            raise RuntimeError(
                "dynamic mock camera pose is unavailable from robot feedback"
            )
        pose_values = np.asarray((*state.pose.xyz, *state.pose.rpy), dtype=float)
        if pose_values.shape != (6,) or not np.all(np.isfinite(pose_values)):
            raise RuntimeError("dynamic mock camera pose is malformed")

        base_flange = np.eye(4)
        base_flange[:3, :3] = _rpy_to_matrix(state.pose.rpy)
        base_flange[:3, 3] = np.asarray(state.pose.xyz, dtype=float)
        base_camera = base_flange @ self._flange_camera
        camera_from_base = np.linalg.inv(base_camera)
        rotation_camera_from_base = camera_from_base[:3, :3]

        projected: list[Detection] = []
        for detection, world in zip(frame.detections, geometry):
            if world is None:
                projected.append(detection)
                continue
            camera_xyz = (
                rotation_camera_from_base @ np.asarray(world.xyz, dtype=float)
                + camera_from_base[:3, 3]
            )
            center_camera_xyz = (
                rotation_camera_from_base
                @ np.asarray(world.object_center_xyz, dtype=float)
                + camera_from_base[:3, 3]
            )
            major_axis_camera = (
                None
                if world.major_axis is None
                else rotation_camera_from_base
                @ np.asarray(world.major_axis, dtype=float)
            )
            minor_axis_camera = (
                None
                if world.minor_axis is None
                else rotation_camera_from_base
                @ np.asarray(world.minor_axis, dtype=float)
            )
            attributes = dict(detection.attributes)
            attributes["mock_world_projection"] = True
            attributes["mock_flange_xyz"] = tuple(float(v) for v in state.pose.xyz)
            attributes["mock_flange_rpy"] = tuple(float(v) for v in state.pose.rpy)
            projected.append(
                replace(
                    detection,
                    depth_m=float(camera_xyz[2]),
                    camera_xyz=tuple(float(v) for v in camera_xyz),
                    base_xyz=world.xyz,
                    object_center_camera_xyz=tuple(
                        float(v) for v in center_camera_xyz
                    ),
                    major_axis_camera=(
                        None
                        if major_axis_camera is None
                        else tuple(float(v) for v in major_axis_camera)
                    ),
                    minor_axis_camera=(
                        None
                        if minor_axis_camera is None
                        else tuple(float(v) for v in minor_axis_camera)
                    ),
                    attributes=attributes,
                )
            )
        return replace(
            frame,
            detections=tuple(projected),
            stamp_s=time.time(),
        )

    @staticmethod
    def _parse_annotations(item: dict) -> tuple[BoxAnnotation, ...]:
        annotations: list[BoxAnnotation] = []
        for raw in item.get("annotations", ()):
            bbox = raw.get("bbox")
            if bbox is None:
                raise ValueError("mock annotation is missing bbox")
            annotations.append(
                BoxAnnotation(
                    label=str(raw["label"]),
                    bbox=tuple(float(v) for v in bbox),
                    confidence=float(raw.get("confidence", 1.0)),
                    instance_id=str(raw.get("instance_id", "")),
                )
            )
        return tuple(annotations)

    @classmethod
    def _parse_frame(cls, item: dict) -> Frame:
        detections: list[Detection] = []
        for raw in item.get("detections", ()):
            bbox = raw.get("bbox")
            if bbox is None:
                raise ValueError("mock detection is missing bbox")
            detection = Detection(
                label=str(raw["label"]),
                confidence=float(raw["confidence"]),
                bbox=tuple(float(v) for v in bbox),
                depth_m=(None if raw.get("depth_m") is None else float(raw["depth_m"])),
                camera_xyz=(
                    None
                    if raw.get("camera_xyz") is None
                    else tuple(float(v) for v in raw["camera_xyz"])
                ),
                base_xyz=(
                    None
                    if raw.get("base_xyz") is None
                    else tuple(float(v) for v in raw["base_xyz"])
                ),
                object_center_camera_xyz=(
                    None
                    if raw.get("object_center_camera_xyz") is None
                    else tuple(
                        float(v) for v in raw["object_center_camera_xyz"]
                    )
                ),
                major_axis_camera=(
                    None
                    if raw.get("major_axis_camera") is None
                    else tuple(float(v) for v in raw["major_axis_camera"])
                ),
                minor_axis_camera=(
                    None
                    if raw.get("minor_axis_camera") is None
                    else tuple(float(v) for v in raw["minor_axis_camera"])
                ),
                major_axis_length_m=(
                    None
                    if raw.get("major_axis_length_m") is None
                    else float(raw["major_axis_length_m"])
                ),
                minor_axis_length_m=(
                    None
                    if raw.get("minor_axis_length_m") is None
                    else float(raw["minor_axis_length_m"])
                ),
                pose_confidence=(
                    None
                    if raw.get("pose_confidence") is None
                    else float(raw["pose_confidence"])
                ),
                graspability=float(raw.get("graspability", 1.0)),
                instance_id=str(raw.get("instance_id", "")),
                attributes=dict(raw.get("attributes", {})),
            )
            detections.append(detection)
        intrinsics_raw = item.get("camera_intrinsics")
        camera_intrinsics = (
            None
            if intrinsics_raw is None
            else CameraIntrinsics(
                width=int(intrinsics_raw["width"]),
                height=int(intrinsics_raw["height"]),
                fx=float(intrinsics_raw["fx"]),
                fy=float(intrinsics_raw["fy"]),
                cx=float(intrinsics_raw["cx"]),
                cy=float(intrinsics_raw["cy"]),
            )
        )
        depth_raw = item.get("depth_m")
        depth_m = None if depth_raw is None else np.asarray(depth_raw, dtype=float).copy()
        capture_raw = item.get("capture_robot_state")
        capture_robot_state = None
        if capture_raw is not None:
            def parse_robot_state(raw: dict) -> RobotState:
                pose_raw = raw.get("pose")
                return RobotState(
                    ready=bool(raw.get("ready", False)),
                    pose=(
                        None
                        if pose_raw is None
                        else Pose(
                            tuple(float(v) for v in pose_raw["xyz"]),
                            tuple(
                                float(v)
                                for v in pose_raw.get(
                                    "rpy",
                                    (3.141592653589793, 0.0, 0.0),
                                )
                            ),
                        )
                    ),
                    fault=bool(raw.get("fault", False)),
                    message=str(raw.get("message", "")),
                    joints=(
                        None
                        if raw.get("joints") is None
                        else tuple(float(v) for v in raw["joints"])
                    ),
                    controller_state=str(raw.get("controller_state", "")),
                    fault_flag=int(raw.get("fault_flag", 0)),
                )

            capture_robot_state = CaptureRobotState(
                before=parse_robot_state(dict(capture_raw["before"])),
                after=parse_robot_state(dict(capture_raw["after"])),
            )
        return Frame(
            frame_id=str(item.get("frame_id", "")),
            detections=tuple(detections),
            stamp_s=float(item.get("stamp_s", 0.0)),
            annotations=cls._parse_annotations(item),
            image_path=str(item.get("image", "") or item.get("image_path", "")),
            camera_intrinsics=camera_intrinsics,
            depth_m=depth_m,
            box_corner_camera_xyz_m=tuple(
                tuple(float(v) for v in point)
                for point in item.get("box_corner_camera_xyz_m", ())
            ),
            box_corner_uv_px=tuple(
                tuple(float(v) for v in point)
                for point in item.get("box_corner_uv_px", ())
            ),
            capture_robot_state=capture_robot_state,
        )

    def capture(self) -> Frame:
        if self._closed:
            raise RuntimeError("mock vision source is closed")
        if self._index >= len(self._frames):
            if not self.config.loop_scenario:
                raise StopIteration("mock scenario exhausted")
            self._index = 0
        frame = self._project_dynamic_frame(self._frames[self._index])
        self._index += 1
        return frame

    def close(self) -> None:
        self._closed = True


class TaskARealSenseVision:
    """Adapter over the existing D405 + YOLO implementation in taskA_grasp."""

    def __init__(self, config: VisionConfig, log=None) -> None:
        if not config.taska_config:
            raise ValueError("taska_realsense vision needs vision.taska_config")
        self.config = config
        self._log = log or (lambda message: None)
        self._stream = None
        self._detector = None
        self._started = False

    def _start(self) -> None:
        if self._started:
            return
        from taskA_grasp.config import load_config
        from taskA_grasp.detector import RealSenseStream, YoloDetector

        taska = load_config(self.config.taska_config)
        self._stream = RealSenseStream(taska.camera, log=self._log)
        self._stream.start()
        self._detector = YoloDetector(taska, self._stream, log=self._log)
        self._detector.load()
        self._started = True
        self._log("[vision:taska] RealSense + YOLO ready")

    def capture(self) -> Frame:
        self._start()
        assert self._detector is not None
        raw_detections, color, _intrinsics = self._detector.grab_targets()
        detections = tuple(
            Detection(
                label=str(item.class_name),
                confidence=float(item.confidence),
                bbox=tuple(float(v) for v in item.bbox),
                depth_m=float(item.depth_m),
                camera_xyz=tuple(float(v) for v in item.cam_xyz),
                graspability=1.0,
                instance_id="",
                attributes={},
            )
            for item in raw_detections
        )
        return Frame(
            frame_id=f"taska-live-{time.time_ns()}",
            detections=detections,
            stamp_s=time.time(),
            image=color,
        )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
        self._started = False


class LegacyHttpVision:
    """Read-only adapter over the old Task A browser recognizer.

    The legacy program owns the RealSense camera and publishes its latest
    detections at ``/status.json``.  This adapter never starts, stops or
    reconfigures it, so the browser's MJPEG stream keeps refreshing while the
    arm is moving.  The legacy preview must expose ``frameSequence`` so one
    unchanged status payload cannot be mistaken for multiple confirmation
    frames.
    """

    def __init__(self, config: VisionConfig, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self._last_sequence: int | None = None
        self._closed = False

    def _fetch_status(self) -> dict:
        request = urllib.request.Request(
            self.config.legacy_status_url,
            headers={"Cache-Control": "no-store"},
        )
        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
                payload = json.load(response)
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "旧视觉识别服务不可用，请先打开“任务A 视觉识别”并等待模型就绪"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("旧视觉识别返回的 status.json 格式不正确")
        return payload

    def _frame_from_status(self, payload: dict, sequence: int) -> Frame:
        accepted_labels = set(self.config.accepted_target_labels())
        detections: list[Detection] = []
        for raw in payload.get("detections", ()):
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("class", raw.get("label", "")))
            if accepted_labels and label not in accepted_labels:
                continue
            try:
                confidence = float(
                    raw.get(
                        "stableConfidence",
                        raw.get("confidence", 0.0),
                    )
                )
                bbox_raw = raw["bbox"]
                if len(bbox_raw) != 4:
                    continue
                bbox = tuple(float(value) for value in bbox_raw)
            except (KeyError, TypeError, ValueError):
                continue
            confidence = min(1.0, max(0.0, confidence))
            if confidence < self.config.min_confidence:
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    graspability=1.0,
                    attributes={
                        "source": "taskA-legacy",
                        "frameSequence": sequence,
                        "rawConfidence": raw.get("confidence"),
                        "stableConfidence": raw.get("stableConfidence"),
                    },
                )
            )
        return Frame(
            frame_id=f"legacy-{sequence:08d}",
            detections=tuple(detections),
            stamp_s=time.time(),
            image=None,
        )

    def capture(self) -> Frame:
        if self._closed:
            raise RuntimeError("legacy vision source is closed")
        first_frame = self._last_sequence is None
        timeout_s = float(
            self.config.first_frame_timeout_s
            if first_frame
            else self.config.capture_timeout_s
        )
        deadline = time.monotonic() + max(0.05, timeout_s)
        last_error = "旧视觉识别尚未发布画面"
        while True:
            try:
                payload = self._fetch_status()
            except RuntimeError as exc:
                last_error = str(exc)
            else:
                if not payload.get("ready"):
                    last_error = "旧视觉识别正在加载模型或相机尚未就绪"
                else:
                    raw_sequence = payload.get("frameSequence")
                    if raw_sequence is None:
                        last_error = (
                            "正在运行的旧视觉识别缺少 frameSequence；"
                            "请重新打开“任务A 视觉识别”"
                        )
                    else:
                        try:
                            sequence = int(raw_sequence)
                        except (TypeError, ValueError):
                            last_error = "旧视觉识别返回了无效的 frameSequence"
                        else:
                            if sequence != self._last_sequence:
                                self._last_sequence = sequence
                                return self._frame_from_status(payload, sequence)
                            last_error = "等待旧视觉识别发布下一帧"
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"旧视觉识别在 {timeout_s:.1f} 秒内没有产生新帧：{last_error}"
                )
            time.sleep(min(max(0.01, self.config.legacy_poll_s), remaining))

    def set_status(self, _status: str) -> None:
        """The legacy browser page owns its own status display."""

    def close(self) -> None:
        self._closed = True


def rotate_180_intrinsics(intrinsics):
    """Return pinhole intrinsics for an RGB-D frame rotated by 180 degrees."""

    return replace(
        intrinsics,
        cx=float(intrinsics.width) - 1.0 - float(intrinsics.cx),
        cy=float(intrinsics.height) - 1.0 - float(intrinsics.cy),
    )


def rotate_180_rgbd(color, depth_m, intrinsics):
    """Rotate aligned colour/depth arrays and their principal point together."""

    color_array = np.asarray(color)
    depth_array = np.asarray(depth_m)
    if color_array.shape[:2] != depth_array.shape[:2]:
        raise RuntimeError(
            "RGB/depth shape mismatch: "
            f"{color_array.shape[:2]} vs {depth_array.shape[:2]}"
        )
    rotated_color = np.ascontiguousarray(color_array[::-1, ::-1])
    rotated_depth = np.ascontiguousarray(depth_array[::-1, ::-1])
    return rotated_color, rotated_depth, rotate_180_intrinsics(intrinsics)


def target_support_guard_mask(color, label: str):
    """Return a target-colour mask used only outside support-plane samples."""

    if "banana" not in str(label).lower():
        return None
    try:
        import cv2
    except ImportError:
        return None
    image = np.asarray(color)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 45)
        & (hsv[:, :, 1] >= 70)
        & (hsv[:, :, 2] >= 50)
    )


class RealSenseYoloVision:
    """Direct D405 + Ultralytics YOLO adapter for live Track 1 tests.

    The implementation is intentionally self-contained: the RGB frame, depth
    frame and principal point are rotated together when the camera is mounted
    upside down.  It does not use taskA's preconfigured live detector, whose
    image path does not always apply that rotation.
    """

    def __init__(
        self,
        config: VisionConfig,
        log=None,
        adaptive_inverse: AdaptiveInverseConfig | None = None,
    ) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self._adaptive_inverse = adaptive_inverse
        self._stream = None
        self._detector = None
        self._intrinsics = None
        self._started = False
        self._frame_index = 0
        self._stabilizer = (
            DetectionStabilizer(config) if config.stabilize else None
        )
        self._output_dir = Path(config.output_dir).expanduser() if config.output_dir else None
        self._last_save_s = 0.0
        self._last_pose_error_log_s = 0.0
        self._last_pose_error_message = ""

    def _start(self) -> None:
        if self._started:
            return
        if not self.config.weights:
            raise RuntimeError("realsense_yolo vision needs vision.weights")
        weights_path = Path(self.config.weights)
        if not weights_path.is_file():
            raise RuntimeError(f"YOLO weights not found: {self.config.weights}")
        expected_sha256 = self.config.weights_sha256.strip().lower()
        if expected_sha256:
            actual_sha256 = _sha256_file(weights_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "YOLO weights SHA256 mismatch: "
                    f"expected={expected_sha256} actual={actual_sha256} "
                    f"path={weights_path}"
                )
            self._log(
                "[vision:realsense_yolo] verified weights "
                f"sha256={actual_sha256}"
            )

        from taskA_grasp.config import CameraConfig
        from taskA_grasp.detector import RealSenseStream, YoloDetector

        camera = CameraConfig(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            serial=self.config.serial,
            depth_window=self.config.depth_window,
            warmup_frames=self.config.warmup_frames,
            anchor=self.config.anchor,
        )
        self._stream = RealSenseStream(camera, log=self._log)
        intrinsics = self._stream.start()
        for _ in range(max(1, self.config.warmup_frames)):
            self._stream.read()
        self._intrinsics = rotate_180_intrinsics(intrinsics) if self.config.rotate_180 else intrinsics

        detector_config = SimpleNamespace(
            weights=self.config.weights,
            imgsz=self.config.imgsz,
            confidence=(
                min(
                    self.config.min_confidence,
                    self.config.stabilizer_inference_min_confidence,
                )
                if self.config.stabilize
                else self.config.min_confidence
            ),
            device=self.config.device,
            classes=list(self.config.classes),
            camera=camera,
        )
        self._detector = YoloDetector(detector_config, self._stream, log=self._log)
        self._detector.load()
        self._started = True
        self._log(
            "[vision:realsense_yolo] ready "
            f"{self._intrinsics.width}x{self._intrinsics.height} "
            f"rotate180={self.config.rotate_180} anchor={self.config.anchor} "
            f"cx={self._intrinsics.cx:.1f} cy={self._intrinsics.cy:.1f} "
            f"stabilizer={'on' if self.config.stabilize else 'off'}"
        )

    def capture(self) -> Frame:
        self._start()
        assert self._stream is not None
        assert self._detector is not None
        assert self._intrinsics is not None

        color, depth_m, intrinsics = self._stream.read()
        if self.config.rotate_180:
            color, depth_m, intrinsics = rotate_180_rgbd(color, depth_m, intrinsics)
        raw_detections = self._detector.detect(color, depth_m, intrinsics)
        raw_threshold = (
            min(
                self.config.min_confidence,
                self.config.stabilizer_inference_min_confidence,
            )
            if self.config.stabilize
            else self.config.min_confidence
        )
        detected = tuple(
            self._make_detection(item, depth_m, intrinsics, color)
            for item in raw_detections
            if item.confidence >= raw_threshold
        )
        if self._stabilizer is None:
            detections = detected
        else:
            detections = self._stabilizer.update(
                detected,
                frame_width=int(color.shape[1]),
                frame_height=int(color.shape[0]),
            )
        self._frame_index += 1
        image = np.asarray(color).copy()
        image_path = self._save_frame(color, detections, intrinsics)
        return Frame(
            frame_id=f"realsense-{self._frame_index:06d}",
            detections=detections,
            stamp_s=time.time(),
            image_path=str(image_path) if image_path is not None else "",
            image=image,
            camera_intrinsics=CameraIntrinsics(
                width=int(intrinsics.width),
                height=int(intrinsics.height),
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                cx=float(intrinsics.cx),
                cy=float(intrinsics.cy),
            ),
            depth_m=np.asarray(depth_m).copy(),
            box_corner_camera_xyz_m=(),
            box_corner_uv_px=(),
        )

    def _make_detection(self, item, depth_m, intrinsics, color) -> Detection:
        bbox = tuple(float(value) for value in item.bbox)
        attributes = {
            "u": float(item.u),
            "v": float(item.v),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
        }
        pose = None
        adaptive = self._adaptive_inverse
        if adaptive is not None and adaptive.enabled:
            try:
                pose = estimate_object_pose(
                    depth_m,
                    bbox,
                    intrinsics,
                    adaptive.pca,
                    object_mask=target_support_guard_mask(
                        color,
                        item.class_name,
                    ),
                )
            except (TypeError, ValueError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                attributes["adaptive_pose_error"] = message
                now_s = time.monotonic()
                if message != self._last_pose_error_message:
                    self._log(
                        "[vision:adaptive] PCA rejection for "
                        f"{item.class_name!r}: {message}"
                    )
                    self._last_pose_error_message = message
                    self._last_pose_error_log_s = now_s
                elif now_s - self._last_pose_error_log_s >= 2.0:
                    self._log(
                        "[vision:adaptive] PCA still rejected for "
                        f"{item.class_name!r}: {message}"
                    )
                    self._last_pose_error_log_s = now_s
            else:
                attributes["adaptive_pose_valid"] = True
                attributes["adaptive_support_plane_valid"] = (
                    pose.support_plane_rms_m is not None
                )
                attributes["adaptive_support_plane_rms_m"] = (
                    pose.support_plane_rms_m
                )
                attributes["adaptive_support_plane_points"] = (
                    pose.support_plane_valid_points
                )
                attributes["adaptive_support_plane_inlier_ratio"] = (
                    pose.support_plane_inlier_ratio
                )
                self._last_pose_error_message = ""

        return Detection(
            label=str(item.class_name),
            confidence=float(item.confidence),
            bbox=bbox,
            depth_m=float(item.depth_m),
            camera_xyz=tuple(float(v) for v in item.cam_xyz),
            object_center_camera_xyz=(
                None if pose is None else pose.center_camera_xyz
            ),
            major_axis_camera=(
                None if pose is None else pose.major_axis_camera
            ),
            minor_axis_camera=(
                None if pose is None else pose.minor_axis_camera
            ),
            major_axis_length_m=(
                None if pose is None else pose.major_axis_length_m
            ),
            minor_axis_length_m=(
                None if pose is None else pose.minor_axis_length_m
            ),
            pose_confidence=(
                None if pose is None else pose.pose_confidence
            ),
            graspability=1.0,
            instance_id="",
            attributes=attributes,
        )

    def _save_frame(self, color, detections: tuple[Detection, ...], intrinsics):
        if not self.config.save_annotated or self._output_dir is None:
            return None
        now_s = time.monotonic()
        if now_s - self._last_save_s < max(0.0, float(self.config.save_interval_s)):
            return None
        try:
            import cv2
        except ImportError:
            return None

        image = np.asarray(color).copy()
        cv2.drawMarker(
            image,
            (int(round(float(intrinsics.cx))), int(round(float(intrinsics.cy)))),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            24,
            2,
        )
        for detection in detections:
            x0, y0, x1, y1 = (int(round(v)) for v in detection.bbox)
            u = float(detection.attributes.get("u", (x0 + x1) * 0.5))
            v = float(detection.attributes.get("v", (y0 + y1) * 0.5))
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 180, 0), 2)
            cv2.drawMarker(
                image,
                (int(round(u)), int(round(v))),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                18,
                2,
            )
            label = (
                f"{detection.label} {detection.confidence:.2f} "
                f"{float(detection.depth_m or 0.0):.3f}m"
            )
            cv2.putText(
                image,
                label,
                (x0, max(18, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                2,
            )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"frame_{self._frame_index:06d}.jpg"
        if not cv2.imwrite(str(path), image):
            self._log(f"[vision:realsense_yolo] failed to write {path}")
            return None
        self._last_save_s = now_s
        return path

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
        self._started = False


def build_vision(
    config: VisionConfig,
    log=None,
    adaptive_inverse: AdaptiveInverseConfig | None = None,
):
    if config.backend == "mock":
        return MockVisionSource(config, log=log)
    if config.backend == "taska_realsense":
        source = TaskARealSenseVision(config, log=log)
    elif config.backend == "realsense_yolo":
        # Keep the legacy constructor call unchanged when adaptive inverse is
        # disabled.  This matters for injected/third-party adapters that only
        # implement the original ``(config, log=...)`` interface.
        if adaptive_inverse is None:
            source = RealSenseYoloVision(config, log=log)
        else:
            source = RealSenseYoloVision(
                config,
                log=log,
                adaptive_inverse=adaptive_inverse,
            )
    elif config.backend == "legacy_http":
        source = LegacyHttpVision(config, log=log)
    else:
        raise ValueError(f"unsupported vision backend: {config.backend!r}")
    if config.continuous:
        return LiveVisionSource(source, config, log=log)
    return source
