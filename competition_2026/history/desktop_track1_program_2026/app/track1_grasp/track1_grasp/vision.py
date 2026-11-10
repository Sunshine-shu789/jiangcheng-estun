"""Vision sources and detector-output adapters.

The orchestration consumes :class:`Frame` objects.  Mock frames make the full
pick cycle runnable before the camera is mounted; the RealSense adapter is
kept behind a lazy import so this file remains testable without a camera.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
import yaml

from .config import VisionConfig
from .detection_stabilizer import DetectionStabilizer
from .domain import BoxAnnotation, Detection, Frame
from .vision_runtime import LiveVisionSource


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
        self._index = 0
        self._closed = False
        self._log(f"[vision:mock] loaded {len(self._frames)} frame(s)")

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
                graspability=float(raw.get("graspability", 1.0)),
                instance_id=str(raw.get("instance_id", "")),
                attributes=dict(raw.get("attributes", {})),
            )
            detections.append(detection)
        return Frame(
            frame_id=str(item.get("frame_id", "")),
            detections=tuple(detections),
            stamp_s=float(item.get("stamp_s", 0.0)),
            annotations=cls._parse_annotations(item),
            image_path=str(item.get("image", "") or item.get("image_path", "")),
        )

    def capture(self) -> Frame:
        if self._closed:
            raise RuntimeError("mock vision source is closed")
        if self._index >= len(self._frames):
            if not self.config.loop_scenario:
                raise StopIteration("mock scenario exhausted")
            self._index = 0
        frame = self._frames[self._index]
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
        target_class = self.config.target_class.strip()
        detections: list[Detection] = []
        for raw in payload.get("detections", ()):
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("class", raw.get("label", "")))
            if target_class and label != target_class:
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


class RealSenseYoloVision:
    """Direct D405 + Ultralytics YOLO adapter for live Track 1 tests.

    The implementation is intentionally self-contained: the RGB frame, depth
    frame and principal point are rotated together when the camera is mounted
    upside down.  It does not use taskA's preconfigured live detector, whose
    image path does not always apply that rotation.
    """

    def __init__(self, config: VisionConfig, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
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

    def _start(self) -> None:
        if self._started:
            return
        if not self.config.weights:
            raise RuntimeError("realsense_yolo vision needs vision.weights")
        if not Path(self.config.weights).is_file():
            raise RuntimeError(f"YOLO weights not found: {self.config.weights}")

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
            Detection(
                label=str(item.class_name),
                confidence=float(item.confidence),
                bbox=tuple(float(v) for v in item.bbox),
                depth_m=float(item.depth_m),
                camera_xyz=tuple(float(v) for v in item.cam_xyz),
                graspability=1.0,
                instance_id="",
                attributes={
                    "u": float(item.u),
                    "v": float(item.v),
                    "fx": float(intrinsics.fx),
                    "fy": float(intrinsics.fy),
                    "cx": float(intrinsics.cx),
                    "cy": float(intrinsics.cy),
                },
            )
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


def build_vision(config: VisionConfig, log=None):
    if config.backend == "mock":
        return MockVisionSource(config, log=log)
    if config.backend == "taska_realsense":
        source = TaskARealSenseVision(config, log=log)
    elif config.backend == "realsense_yolo":
        source = RealSenseYoloVision(config, log=log)
    elif config.backend == "legacy_http":
        source = LegacyHttpVision(config, log=log)
    else:
        raise ValueError(f"unsupported vision backend: {config.backend!r}")
    if config.continuous:
        return LiveVisionSource(source, config, log=log)
    return source
