"""Task A pipeline: capture -> detect -> annotate -> score, with no motion.

Task A of Track 1 only requires the camera image to be shown with the objects
detected and annotated, then the same object tested from ten different
positions with IoU >= 0.75 per position.  There is no arm or gripper in this
path, so it stays usable before any hardware is mounted.
"""

from __future__ import annotations

from pathlib import Path

from .annotator import Annotator, load_ground_truth, merge_reports, score_frame
from .config import Track1Config
from .domain import Frame, IouMatch, TaskPhase, VisionReport


class VisionDemoPipeline:
    """Run the task A detection/annotation/IoU loop over a vision source."""

    def __init__(
        self,
        config: Track1Config,
        *,
        vision,
        annotator: Annotator | None = None,
        log=None,
    ) -> None:
        self.config = config
        self.vision = vision
        self.annotator = annotator or Annotator(config.task_a, log=log)
        self._ground_truth = (
            load_ground_truth(config.task_a.ground_truth_file)
            if config.task_a.ground_truth_file
            else {}
        )
        if self._ground_truth:
            self._log(
                f"[scan] loaded {len(self._ground_truth)} ground-truth entr(ies) "
                f"from {config.task_a.ground_truth_file}"
            )
        self._log = log or (lambda message: None)
        self._closed = False

    def run(self) -> VisionReport:
        threshold = float(self.config.task_a.iou_threshold)
        reports: list[VisionReport] = []
        index = 0
        self._log(
            f"[run] task A detection demo: iou_threshold={threshold:.2f} "
            f"max_frames={self.config.task_a.max_frames or 'all'}"
        )
        self._set_vision_status("TASK A: scanning")
        while True:
            if self.config.task_a.max_frames and index >= self.config.task_a.max_frames:
                break
            try:
                frame, matches, report = self.run_once(index)
            except StopIteration as exc:
                self._log(f"[scan] {exc}")
                self._set_vision_status(f"TASK A DONE: {exc}")
                break
            reports.append(report)
            if self.config.task_a.save_annotated:
                image = self.annotator.render(frame)
                path = self.annotator.save(frame, image, index)
                if path:
                    self._log(f"[annotate] wrote {path}")
            else:
                self.annotator.render(frame)
            self._log_match_summary(frame, matches)
            index += 1
            if self.config.task.sample_period_s:
                import time

                time.sleep(0.0)  # placeholder: real sources pace themselves

        summary = merge_reports(reports, threshold=threshold)
        self._log(
            f"[report] frames={summary.frames} detections={summary.detections} "
            f"matched={summary.matched}/{summary.total_annotations} "
            f"passed={summary.passed} pass_rate={summary.pass_rate:.1%} "
            f"mean_iou={summary.mean_iou:.3f}"
        )
        return summary

    def run_once(self, index: int = 0) -> tuple[Frame, tuple[IouMatch, ...], VisionReport]:
        """Capture, filter, annotate and score exactly one frame."""

        self._log(f"[{TaskPhase.SCAN.value}] capturing frame {index}")
        self._set_vision_status(f"TASK A: capturing frame {index + 1}")
        frame = self.vision.capture()
        frame = self._apply_ground_truth(frame)
        usable = tuple(
            detection
            for detection in frame.detections
            if detection.confidence >= self.config.vision.min_confidence
        )
        scored_frame = Frame(
            frame_id=frame.frame_id,
            detections=usable,
            stamp_s=frame.stamp_s,
            annotations=frame.annotations,
            image_path=frame.image_path,
            image=frame.image,
        )
        self._log(
            f"[{TaskPhase.DETECT.value}] frame {frame.frame_id!r}: "
            f"{len(frame.detections)} raw, {len(usable)} above min_confidence"
        )
        matches, report = score_frame(
            scored_frame, threshold=self.config.task_a.iou_threshold
        )
        self._log(
            f"[{TaskPhase.ANNOTATE.value}] drew {len(usable)} prediction(s) "
            f"and {len(frame.annotations)} ground-truth box(es)"
        )
        self._log(
            f"[{TaskPhase.EVALUATE.value}] passed={report.passed}/"
            f"{report.total_annotations} mean_iou={report.mean_iou:.3f}"
        )
        self._set_vision_status(
            f"TASK A: frame {index + 1}, {len(usable)} detection(s), "
            f"IoU passed {report.passed}/{report.total_annotations}"
        )
        return scored_frame, matches, report

    def _set_vision_status(self, text: str) -> None:
        set_status = getattr(self.vision, "set_status", None)
        if set_status is None:
            return
        try:
            set_status(text)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._log(f"[vision:live] status update failed: {exc}")

    def _apply_ground_truth(self, frame: Frame) -> Frame:
        """Attach externally loaded ground truth when the source has none."""

        if not self._ground_truth or frame.annotations:
            return frame
        boxes = self._ground_truth.get(frame.frame_id)
        if boxes is None and frame.image_path:
            boxes = self._ground_truth.get(Path(frame.image_path).name) or self._ground_truth.get(
                Path(frame.image_path).stem
            )
        if boxes is None:
            return frame
        return Frame(
            frame_id=frame.frame_id,
            detections=frame.detections,
            stamp_s=frame.stamp_s,
            annotations=boxes,
            image_path=frame.image_path,
            image=frame.image,
        )

    def _log_match_summary(self, frame: Frame, matches: tuple[IouMatch, ...]) -> None:
        for match in matches:
            if match.annotation is None:
                label = match.detection.label if match.detection else "-"
                self._log(f"    {label:12s} iou=0.000 (no ground truth) miss")
                continue
            label = match.label
            if match.detection is None:
                self._log(f"    {label:12s} iou=0.000 miss")
                continue
            flag = "ok" if match.passed else "low"
            self._log(f"    {label:12s} iou={match.iou:.3f} {flag}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.vision, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # pragma: no cover - defensive
                self._log(f"[shutdown] vision: {exc}")
