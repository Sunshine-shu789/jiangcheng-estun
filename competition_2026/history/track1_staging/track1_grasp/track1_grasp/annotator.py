"""Task A helpers: IoU scoring, detection/ground-truth matching and drawing.

The geometry helpers are pure Python so they can be unit-tested without a
camera or OpenCV.  Only :class:`Annotator` needs ``cv2`` and it imports it
lazily, which keeps the rest of the framework importable on a bare machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .config import TaskAConfig
from .domain import BoxAnnotation, Detection, Frame, IouMatch, VisionReport

Box = tuple[float, float, float, float]


def _as_box(box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError("a box needs (x0, y0, x1, y1)")
    x0, y0, x1, y1 = (float(v) for v in box)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x0, y0, x1, y1)`` boxes."""

    ax0, ay0, ax1, ay1 = _as_box(box_a)
    bx0, by0, bx1, by1 = _as_box(box_b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _labels_compatible(detection_label: str, annotation_label: str) -> bool:
    """A label match is required, but generic detector labels are forgiven.

    Task C/D/E detectors may only emit ``object``/``unknown`` while the ground
    truth carries a class name; scoring still uses the box overlap there.
    """

    if detection_label == annotation_label:
        return True
    generic = {"object", "unknown", ""}
    return detection_label in generic or annotation_label in generic


def match_frame(
    detections: Iterable[Detection],
    annotations: Iterable[BoxAnnotation],
    *,
    threshold: float = 0.75,
    require_label: bool = True,
) -> tuple[IouMatch, ...]:
    """Greedily match each annotation to its best unused detection by IoU.

    Produces one :class:`IouMatch` per ground-truth annotation (with
    ``detection=None`` when nothing overlapped) followed by the detections no
    annotation claimed, so both missed and spurious boxes are visible.
    """

    detection_list = list(detections)
    annotation_list = list(annotations)
    used: set[int] = set()
    matches: list[IouMatch] = []

    for annotation in annotation_list:
        best_index = -1
        best_iou = 0.0
        for index, detection in enumerate(detection_list):
            if index in used:
                continue
            if require_label and not _labels_compatible(detection.label, annotation.label):
                continue
            value = iou(detection.bbox, annotation.bbox)
            if value > best_iou:
                best_iou = value
                best_index = index
        if best_index < 0:
            matches.append(IouMatch(None, annotation, 0.0, False))
            continue
        used.add(best_index)
        matches.append(
            IouMatch(detection_list[best_index], annotation, best_iou, best_iou >= threshold)
        )

    for index, detection in enumerate(detection_list):
        if index not in used:
            matches.append(IouMatch(detection, None, 0.0, False))

    return tuple(matches)


def score_frame(
    frame: Frame, *, threshold: float = 0.75
) -> tuple[tuple[IouMatch, ...], VisionReport]:
    """Score one frame against its ground-truth annotations."""

    matches = match_frame(frame.detections, frame.annotations, threshold=threshold)
    matched = sum(1 for match in matches if match.annotation is not None and match.detection is not None)
    passed = sum(1 for match in matches if match.passed)
    report = VisionReport(
        frames=1,
        detections=len(frame.detections),
        matched=matched,
        passed=passed,
        total_annotations=len(frame.annotations),
        results=matches,
        iou_threshold=threshold,
    )
    return matches, report


def merge_reports(reports: Sequence[VisionReport], *, threshold: float) -> VisionReport:
    """Aggregate per-frame reports into one run report."""

    results = tuple(match for report in reports for match in report.results)
    return VisionReport(
        frames=sum(report.frames for report in reports),
        detections=sum(report.detections for report in reports),
        matched=sum(report.matched for report in reports),
        passed=sum(report.passed for report in reports),
        total_annotations=sum(report.total_annotations for report in reports),
        results=results,
        iou_threshold=threshold,
    )


def load_ground_truth(path: str | Path) -> dict[str, tuple[BoxAnnotation, ...]]:
    """Load a task A ground-truth file into ``{frame_or_image: boxes}``.

    Entries are keyed by both ``frame_id`` and the basename of ``image`` so a
    real capture set can be matched either way.
    """

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("frames", [])
    table: dict[str, tuple[BoxAnnotation, ...]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        boxes = tuple(
            BoxAnnotation(
                label=str(raw["label"]),
                bbox=tuple(float(v) for v in raw["bbox"]),
                confidence=float(raw.get("confidence", 1.0)),
                instance_id=str(raw.get("instance_id", "")),
            )
            for raw in item.get("annotations", ())
        )
        frame_id = str(item.get("frame_id", ""))
        if frame_id:
            table[frame_id] = boxes
        image = item.get("image") or item.get("image_path")
        if image:
            table[Path(str(image)).stem] = boxes
            table[Path(str(image)).name] = boxes
    return table


class Annotator:
    """Render detections (and optional ground truth) onto the frame image.

    When the vision source does not provide pixels yet (mock mode before the
    camera is mounted) the annotator still produces a blank canvas so the
    pipeline and file output can be exercised.
    """

    def __init__(self, config: TaskAConfig, log: Any = None) -> None:
        self.config = config
        self._log = log or (lambda message: None)

    def _load_image(self, frame: Frame) -> Any:
        import numpy as np

        if frame.image_path:
            try:
                import cv2

                image = cv2.imread(frame.image_path)
                if image is not None:
                    return image
            except Exception as exc:  # pragma: no cover - depends on install
                self._log(f"[annotate] could not read {frame.image_path}: {exc}")
        width = height = 640
        for detection in frame.detections:
            width = max(width, int(detection.bbox[2]) + 20)
            height = max(height, int(detection.bbox[3]) + 20)
        for annotation in frame.annotations:
            width = max(width, int(annotation.bbox[2]) + 20)
            height = max(height, int(annotation.bbox[3]) + 20)
        return np.full((height, width, 3), 32, dtype=np.uint8)

    def render(self, frame: Frame) -> Any:
        """Return a BGR image with ground truth and predictions drawn."""

        import cv2

        image = self._load_image(frame)

        if self.config.draw_ground_truth:
            for annotation in frame.annotations:
                x0, y0, x1, y1 = (int(round(v)) for v in annotation.bbox)
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
                if self.config.draw_labels:
                    cv2.putText(
                        image,
                        f"GT:{annotation.label}",
                        (x0, max(12, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        for detection in frame.detections:
            x0, y0, x1, y1 = (int(round(v)) for v in detection.bbox)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 165, 255), 2)
            if self.config.draw_labels:
                cv2.putText(
                    image,
                    f"{detection.label} {detection.confidence:.2f}",
                    (x0 + 2, min(y1 - 4, max(14, y0 + 16))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )
        return image

    def save(self, frame: Frame, image: Any, index: int) -> str:
        if not self.config.output_dir:
            return ""
        import cv2

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = frame.frame_id or f"frame_{index:04d}"
        path = out_dir / f"{index:04d}_{stem}.png"
        cv2.imwrite(str(path), image)
        return str(path)
