"""Vision sources and detector-output adapters.

The orchestration consumes :class:`Frame` objects.  Mock frames make the full
pick cycle runnable before the camera is mounted; the RealSense adapter is
kept behind a lazy import so this file remains testable without a camera.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import VisionConfig
from .domain import Detection, Frame


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
    def _parse_frame(item: dict) -> Frame:
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
        raw_detections, _color, _intrinsics = self._detector.grab_targets()
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
        return Frame(frame_id="taska-live", detections=detections)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
        self._started = False


def build_vision(config: VisionConfig, log=None):
    if config.backend == "mock":
        return MockVisionSource(config, log=log)
    if config.backend == "taska_realsense":
        return TaskARealSenseVision(config, log=log)
    raise ValueError(f"unsupported vision backend: {config.backend!r}")
