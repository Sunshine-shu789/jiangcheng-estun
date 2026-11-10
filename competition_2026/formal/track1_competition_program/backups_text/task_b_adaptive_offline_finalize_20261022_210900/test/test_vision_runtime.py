from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from track1_grasp.annotator import Annotator
from track1_grasp.config import TaskAConfig, VisionConfig
from track1_grasp.domain import Detection, Frame
import track1_grasp.vision as vision_module
from track1_grasp.vision_runtime import (
    LiveVisionSource,
    VisionRuntimeError,
    render_preview,
)


def runtime_config(**overrides) -> VisionConfig:
    values = {
        "backend": "mock",
        "show": False,
        "continuous": True,
        "capture_timeout_s": 1.0,
    }
    values.update(overrides)
    return VisionConfig(**values)


class CountingSource:
    def __init__(self, delay_s: float = 0.02) -> None:
        self.delay_s = delay_s
        self.count = 0
        self.closed = False
        self._lock = threading.Lock()

    def capture(self) -> Frame:
        time.sleep(self.delay_s)
        with self._lock:
            self.count += 1
            count = self.count
        return Frame(
            frame_id=f"fake-{count:04d}",
            detections=(),
            image=np.zeros((24, 32, 3), dtype=np.uint8),
        )

    def close(self) -> None:
        self.closed = True


class FailingSource:
    def __init__(self) -> None:
        self.closed = False

    def capture(self) -> Frame:
        raise RuntimeError("camera unavailable")

    def close(self) -> None:
        self.closed = True


def test_background_vision_keeps_running_without_consumer_calls():
    source = CountingSource(delay_s=0.02)
    runtime = LiveVisionSource(source, runtime_config(), log=lambda _message: None)
    try:
        first = runtime.capture()
        time.sleep(0.18)
        latest = runtime.latest()
        assert latest is not None
        assert latest.frame_id != first.frame_id
        assert source.count >= 4
    finally:
        runtime.close()


def test_capture_waits_for_a_frame_after_the_call():
    source = CountingSource(delay_s=0.03)
    runtime = LiveVisionSource(source, runtime_config(), log=lambda _message: None)
    try:
        runtime.capture()
        time.sleep(0.12)
        barrier = runtime.snapshot().sequence
        frame = runtime.capture()
        returned_sequence = int(frame.frame_id.rsplit("-", 1)[1])
        assert returned_sequence > barrier
    finally:
        runtime.close()


def test_latest_never_returns_a_consumed_old_frame_after_motion():
    source = CountingSource(delay_s=0.02)
    runtime = LiveVisionSource(source, runtime_config(), log=lambda _message: None)
    try:
        first = runtime.capture()
        time.sleep(0.16)  # simulate a blocking arm move
        latest = runtime.latest()
        assert latest is not None
        assert latest.frame_id != first.frame_id
        assert latest.detections == ()
    finally:
        runtime.close()


def test_background_error_is_raised_in_task_thread():
    source = FailingSource()
    runtime = LiveVisionSource(source, runtime_config(), log=lambda _message: None)
    try:
        with pytest.raises(VisionRuntimeError, match="camera unavailable"):
            runtime.capture()
        snapshot = runtime.snapshot()
        assert "VISION ERROR" in snapshot.status
        assert "camera unavailable" in snapshot.error
    finally:
        runtime.close()
    assert source.closed is True


def test_close_stops_worker_and_releases_source():
    source = CountingSource(delay_s=0.01)
    runtime = LiveVisionSource(source, runtime_config(), log=lambda _message: None)
    runtime.capture()
    runtime.close()
    assert runtime.is_alive is False
    assert source.closed is True


def test_frame_image_is_ignored_for_dataclass_equality():
    first = Frame("same", (), image=np.zeros((2, 2, 3), dtype=np.uint8))
    second = Frame("same", (), image=np.ones((2, 2, 3), dtype=np.uint8))
    assert first == second


def test_annotator_prefers_in_memory_bgr_image():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    frame = Frame("memory", (), image=image)
    rendered = Annotator(TaskAConfig()).render(frame)
    assert rendered.shape == image.shape
    assert tuple(rendered[2, 2]) == (10, 20, 30)


def test_preview_renderer_shows_target_boxes_and_runtime_state():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    frame = Frame(
        "preview",
        (
            Detection(
                "6-Foam_Brick",
                0.92,
                (80.0, 100.0, 180.0, 200.0),
                depth_m=0.33,
            ),
        ),
        image=image,
    )
    rendered = render_preview(
        frame,
        status="MOVING: approach_step_01",
        target_class="6-Foam_Brick",
        fps=8.5,
        capture_ms=117.0,
        age_s=0.1,
    )
    assert rendered.shape == image.shape
    assert np.any(np.all(rendered == (40, 210, 40), axis=2))
    assert image.max() == 0  # renderer must not mutate the stored camera frame


def test_real_vision_backend_is_wrapped_in_continuous_runtime(monkeypatch):
    class FakeAdapter:
        def __init__(self, config, log=None) -> None:
            self.config = config
            self.closed = False

        def capture(self) -> Frame:
            return Frame(
                "adapter-frame",
                (),
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(vision_module, "RealSenseYoloVision", FakeAdapter)
    config = VisionConfig(
        backend="realsense_yolo",
        continuous=True,
        show=False,
        capture_timeout_s=1.0,
    )
    runtime = vision_module.build_vision(config, log=lambda _message: None)
    try:
        assert isinstance(runtime, LiveVisionSource)
        assert runtime.capture().frame_id == "adapter-frame"
    finally:
        runtime.close()
