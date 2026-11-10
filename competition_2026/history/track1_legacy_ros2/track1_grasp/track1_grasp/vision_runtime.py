"""Continuous vision and an independent live preview window.

The real camera/YOLO adapters are synchronous: one ``capture`` call acquires a
frame, runs inference and returns.  That is fine for the mock pipeline, but it
means the arm control thread and the camera cannot both make progress during a
blocking ``move_to`` call.  :class:`LiveVisionSource` puts the adapter behind a
single background worker and exposes only the newest frame to consumers.

Preview rendering runs in a second thread.  OpenCV GUI calls therefore never
hold the inference lock and the window keeps refreshing while a robot motion is
in progress.  The preview is observability-only; if no display is available the
vision worker continues normally.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from .config import VisionConfig
from .domain import Detection, Frame


class VisionRuntimeError(RuntimeError):
    """Raised in the task thread when the background vision worker fails."""


@dataclass(frozen=True)
class VisionRuntimeSnapshot:
    """Consistent view of the background source for the preview thread."""

    frame: Frame | None
    sequence: int
    published_at_s: float
    status: str
    error: str
    capture_fps: float
    capture_ms: float
    age_s: float


class LiveVisionSource:
    """Wrap a synchronous vision source with a newest-frame background worker.

    ``capture()`` deliberately waits for a frame produced *after* the call.
    This is important for visual servoing: immediately after a motion finishes,
    an image captured before or during that motion must not be used to compute
    the next command.  ``latest()`` never consumes a sequence and is intended
    for the preview window.
    """

    def __init__(self, source: Any, config: VisionConfig, log=None) -> None:
        self.source = source
        self.config = config
        self._log = log or (lambda message: None)
        self._condition = threading.Condition()
        self._latest_frame: Frame | None = None
        self._latest_sequence = 0
        self._consumed_sequence = 0
        self._published_at_s = 0.0
        self._status = "STARTING VISION"
        self._error: BaseException | None = None
        self._capture_fps = 0.0
        self._capture_ms = 0.0
        self._closed = False
        self._source_closed = False
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="track1-vision-capture",
            daemon=True,
        )
        self._capture_thread.start()
        self._preview = _PreviewWindow(self, config, self._log)
        self._preview.start()

    @property
    def is_alive(self) -> bool:
        return self._capture_thread.is_alive()

    def set_status(self, status: str) -> None:
        """Update the task/motion/gripper state shown in the preview window."""

        with self._condition:
            self._status = str(status)
        self._preview.wake()

    def latest(self) -> Frame | None:
        """Return the newest frame without consuming it."""

        with self._condition:
            return self._latest_frame

    def snapshot(self) -> VisionRuntimeSnapshot:
        """Return all preview data atomically, including the frame age."""

        now = time.monotonic()
        with self._condition:
            return VisionRuntimeSnapshot(
                frame=self._latest_frame,
                sequence=self._latest_sequence,
                published_at_s=self._published_at_s,
                status=self._status,
                error=self._format_error(self._error),
                capture_fps=self._capture_fps,
                capture_ms=self._capture_ms,
                age_s=(now - self._published_at_s if self._published_at_s else float("inf")),
            )

    def capture(self) -> Frame:
        """Wait for and return a frame produced after this call.

        The very first call may consume the first available frame.  Every later
        call records the current sequence as a barrier and waits for the next
        publication, so stale detections cannot be used after a blocking move.
        """

        timeout_s = max(0.05, float(self.config.capture_timeout_s))
        deadline = time.monotonic() + timeout_s
        with self._condition:
            if self._consumed_sequence == 0:
                while self._latest_sequence == 0:
                    self._raise_if_unavailable_locked()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError(
                            f"vision produced no frame within {timeout_s:.1f}s"
                        )
                    self._condition.wait(remaining)
            else:
                barrier = self._latest_sequence
                while self._latest_sequence <= barrier:
                    self._raise_if_unavailable_locked()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError(
                            "vision produced no new frame within "
                            f"{timeout_s:.1f}s"
                        )
                    self._condition.wait(remaining)

            if self._latest_frame is None:  # defensive; the loops above prevent it
                raise VisionRuntimeError("vision worker has no frame")
            frame = self._latest_frame
            self._consumed_sequence = self._latest_sequence
            return frame

    def close(self) -> None:
        if self._closed:
            return
        with self._condition:
            self._closed = True
            self._condition.notify_all()

        self._preview.stop()
        if self._capture_thread.is_alive() and threading.current_thread() is not self._capture_thread:
            self._capture_thread.join(timeout=3.0)

        self._close_source()

        if self._capture_thread.is_alive() and threading.current_thread() is not self._capture_thread:
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                self._log("[vision:live] capture worker did not stop cleanly")

    def _capture_loop(self) -> None:
        last_tick_s: float | None = None
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                started_s = time.monotonic()
                try:
                    frame = self.source.capture()
                except Exception as exc:
                    with self._condition:
                        if not self._closed:
                            self._error = exc
                            self._status = f"VISION ERROR: {exc}"
                            self._condition.notify_all()
                    if not self._closed:
                        self._log(f"[vision:live] background capture failed: {exc}")
                    return

                finished_s = time.monotonic()
                if last_tick_s is not None:
                    instantaneous_fps = 1.0 / max(1e-6, finished_s - last_tick_s)
                    self._capture_fps = (
                        instantaneous_fps
                        if self._capture_fps <= 0.0
                        else 0.85 * self._capture_fps + 0.15 * instantaneous_fps
                    )
                last_tick_s = finished_s
                capture_ms = 1000.0 * (finished_s - started_s)

                with self._condition:
                    if self._closed:
                        return
                    self._latest_frame = frame
                    self._latest_sequence += 1
                    self._published_at_s = finished_s
                    self._capture_ms = capture_ms
                    if self._status == "STARTING VISION":
                        self._status = "LIVE"
                    self._condition.notify_all()
        except BaseException as exc:  # pragma: no cover - defensive thread guard
            with self._condition:
                if not self._closed:
                    self._error = exc
                    self._status = f"VISION ERROR: {exc}"
                    self._condition.notify_all()

    def _raise_if_unavailable_locked(self) -> None:
        if self._error is not None:
            message = self._format_error(self._error)
            raise VisionRuntimeError(f"background vision failed: {message}") from self._error
        if self._closed:
            raise VisionRuntimeError("vision source is closed")

    @staticmethod
    def _format_error(error: BaseException | None) -> str:
        if error is None:
            return ""
        return f"{type(error).__name__}: {error}"

    def _close_source(self) -> None:
        if self._source_closed:
            return
        self._source_closed = True
        close = getattr(self.source, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception as exc:  # pragma: no cover - hardware dependent
            self._log(f"[vision:live] source close failed: {exc}")


class _PreviewWindow:
    """Dedicated OpenCV GUI thread for the newest vision frame."""

    def __init__(self, owner: LiveVisionSource, config: VisionConfig, log) -> None:
        self.owner = owner
        self.config = config
        self._log = log
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="track1-vision-preview",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if not self.config.show:
            return
        if os.name != "nt" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self._log(
                "[preview] show=true but DISPLAY/WAYLAND_DISPLAY is unset; "
                "continuous recognition will run without a window"
            )
            return
        self._started = True
        self._thread.start()

    def wake(self) -> None:
        self._wake_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import cv2
        except ImportError:
            self._log("[preview] OpenCV is unavailable; window disabled")
            return

        window_name = self.config.window_name or "Track 1 - D405 + YOLO"
        period_s = 1.0 / max(1.0, min(120.0, float(self.config.preview_fps)))
        opened = False
        try:
            while not self._stop_event.is_set():
                snapshot = self.owner.snapshot()
                image = render_preview(
                    snapshot.frame,
                    status="STOPPING" if self._stop_event.is_set() else snapshot.status,
                    target_class=self.config.target_class,
                    fps=snapshot.capture_fps,
                    capture_ms=snapshot.capture_ms,
                    age_s=snapshot.age_s,
                    stale_after_s=self.config.stale_after_s,
                    error=snapshot.error,
                )
                if not opened:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
                    opened = True
                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self._log("[preview] window hidden; continuous vision is still running")
                    break
                if key == ord("s"):
                    self._save_snapshot(cv2, image, snapshot)
                self._wake_event.wait(period_s)
                self._wake_event.clear()
        except Exception as exc:  # GUI backend failures must not stop recognition
            self._log(f"[preview] window disabled: {exc}")
        finally:
            if opened:
                try:
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)
                except Exception:
                    pass

    def _save_snapshot(self, cv2, image, snapshot: VisionRuntimeSnapshot) -> None:
        output_dir = (
            Path(self.config.output_dir).expanduser()
            if self.config.output_dir
            else Path.cwd() / "track1_snapshots"
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_id = snapshot.frame.frame_id if snapshot.frame is not None else "no_frame"
            safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in frame_id)
            path = output_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}_{safe_id}.jpg"
            if cv2.imwrite(str(path), image):
                self._log(f"[preview] snapshot saved: {path}")
            else:
                self._log(f"[preview] failed to save snapshot: {path}")
        except Exception as exc:  # pragma: no cover - filesystem dependent
            self._log(f"[preview] snapshot error: {exc}")


def render_preview(
    frame: Frame | None,
    *,
    status: str,
    target_class: str = "",
    fps: float = 0.0,
    capture_ms: float = 0.0,
    age_s: float = 0.0,
    stale_after_s: float = 2.0,
    error: str = "",
) -> np.ndarray:
    """Render the live camera image, detections and runtime state as BGR.

    This is a pure function apart from OpenCV drawing calls, which makes the
    overlay path unit-testable without creating a GUI window.
    """

    import cv2

    image = _frame_image(frame, cv2)
    detections = tuple(frame.detections) if frame is not None else ()
    selected = _select_preview_target(detections, target_class)
    age_known = age_s != float("inf")
    stale = bool(error) or (age_known and age_s > max(0.1, float(stale_after_s)))
    red = (20, 20, 230)
    green = (40, 210, 40)
    orange = (0, 165, 255)

    for detection in detections:
        is_selected = detection is selected
        held = bool(detection.attributes.get("held", False))
        color = red if stale else (orange if held else (green if is_selected else orange))
        _draw_detection(cv2, image, detection, color, thickness=3 if is_selected else 2)

    banner_height = min(max(96, image.shape[0] // 6), max(1, image.shape[0] - 1))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], banner_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)

    status_text = str(status or "LIVE").replace("\n", " ")
    if error:
        status_text = f"VISION ERROR: {error}"
    elif stale and age_s != float("inf"):
        status_text = f"STALE {age_s:.1f}s | {status_text}"
    state_color = red if stale or error else green
    cv2.circle(image, (16, 20), 7, state_color, -1, cv2.LINE_AA)
    _put_text(cv2, image, status_text[:110], (30, 26), 0.62, (245, 245, 245), 2)

    age_text = f"{age_s:.2f}s" if age_known else "waiting"
    metrics = (
        f"Vision {fps:.1f} FPS | capture+infer {capture_ms:.0f} ms "
        f"| age {age_text}"
    )
    _put_text(cv2, image, metrics, (14, 53), 0.53, (210, 210, 210), 1)
    target_line = _target_line(target_class, selected, len(detections))
    _put_text(cv2, image, target_line, (14, 78), 0.53, (225, 225, 225), 1)
    _put_text(
        cv2,
        image,
        "q: hide preview   s: save snapshot   Ctrl-C: stop task",
        (14, image.shape[0] - 12),
        0.48,
        (235, 235, 235),
        1,
    )
    return image


def _frame_image(frame: Frame | None, cv2) -> np.ndarray:
    raw = frame.image if frame is not None else None
    if raw is None and frame is not None and frame.image_path:
        raw = cv2.imread(frame.image_path)
    if raw is None:
        return np.full((540, 960, 3), 28, dtype=np.uint8)

    image = np.asarray(raw)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        return np.full((540, 960, 3), 28, dtype=np.uint8)
    return np.ascontiguousarray(image).copy()


def _select_preview_target(
    detections: tuple[Detection, ...], target_class: str
) -> Detection | None:
    candidates = (
        tuple(d for d in detections if d.label == target_class)
        if target_class
        else detections
    )
    if not candidates:
        return None
    return max(candidates, key=lambda detection: detection.score())


def _draw_detection(cv2, image: np.ndarray, detection: Detection, color, thickness: int) -> None:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in detection.bbox)
    x0, x1 = max(0, min(width - 1, x0)), max(0, min(width - 1, x1))
    y0, y1 = max(0, min(height - 1, y0)), max(0, min(height - 1, y1))
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    u = float(detection.attributes.get("u", (x0 + x1) * 0.5))
    v = float(detection.attributes.get("v", (y0 + y1) * 0.5))
    cv2.drawMarker(
        image,
        (int(round(u)), int(round(v))),
        color,
        cv2.MARKER_CROSS,
        18,
        2,
    )
    depth = f" {detection.depth_m:.3f}m" if detection.depth_m is not None else ""
    stable = float(detection.attributes.get("stable_confidence", detection.confidence))
    raw = float(detection.attributes.get("raw_confidence", detection.confidence))
    track_id = detection.attributes.get("track_id")
    hits = detection.attributes.get("hits")
    missed = detection.attributes.get("missed")
    held = bool(detection.attributes.get("held", False))
    state = " HOLD" if held else ""
    track = f" #{track_id}" if track_id is not None else ""
    history = (
        f" h{hits} m{missed}"
        if hits is not None and missed is not None
        else ""
    )
    label = (
        f"{detection.label}{track} stable={stable:.2f} raw={raw:.2f}"
        f"{depth}{history}{state}"
    )
    text_y = max(100, y0 - 7)
    _put_text(cv2, image, label, (x0 + 2, text_y), 0.46, color, 2)


def _target_line(target_class: str, selected: Detection | None, count: int) -> str:
    wanted = target_class or "best target"
    if selected is None:
        return f"Target: {wanted} | NOT VISIBLE | detections={count}"
    depth = f" depth={selected.depth_m:.3f}m" if selected.depth_m is not None else ""
    stable = float(selected.attributes.get("stable_confidence", selected.confidence))
    raw = float(selected.attributes.get("raw_confidence", selected.confidence))
    track_id = selected.attributes.get("track_id")
    hits = selected.attributes.get("hits")
    missed = selected.attributes.get("missed")
    held = " HOLD" if selected.attributes.get("held", False) else ""
    track = f" track={track_id}" if track_id is not None else ""
    history = (
        f" hits={hits} missed={missed}"
        if hits is not None and missed is not None
        else ""
    )
    return (
        f"Target: {selected.label} stable={stable:.2f} raw={raw:.2f}"
        f"{depth}{track}{history}{held} | tracks={count}"
    )


def _put_text(cv2, image: np.ndarray, text: str, origin: tuple[int, int], scale: float, color, thickness: int) -> None:
    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
