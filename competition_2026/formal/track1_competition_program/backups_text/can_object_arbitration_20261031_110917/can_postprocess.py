#!/usr/bin/env python3
"""Targeted post-processing for the three competition cans.

The YOLO model remains unchanged. This module resolves four observed failures:

* ``2-Starkist_Tuna`` and ``12-Baseball`` describe the same physical can;
* ``5-Tomato_Soup`` is hidden by an overlapping ``29-Paper_Towels`` box;
* ``10-meat_can`` can be mislabeled as an adapter/other class at this view;
* classes and boxes flicker between neighbouring inference frames.

Only the three can classes receive special arbitration. Detections outside that
small set still follow the normal confidence threshold. The gold-can recovery
is gated by fused colour, geometry, texture and scene-context evidence, so a
generic yellow rectangle is not enough on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


TUNA_ID = 1
TOMATO_ID = 4
MEAT_CAN_ID = 9
CAN_IDS = {TUNA_ID, TOMATO_ID, MEAT_CAN_ID}

# These are the classes repeatedly observed on the same physical can.
CAN_CONFUSERS: dict[int, set[int]] = {
    TUNA_ID: {11},
    TOMATO_ID: {11, 17, 20, 22, 28},
    MEAT_CAN_ID: {5, 6, 11, 15, 16, 17, 18, 22, 23, 27, 28, 37, 39, 40},
}


@dataclass(frozen=True)
class CanPostProcessConfig:
    enabled: bool = True
    inference_conf: float = 0.005
    tomato_min_conf: float = 0.012
    tomato_direct_red_min_ratio: float = 0.06
    tomato_red_min_ratio: float = 0.10
    tomato_low_conf_context_iou: float = 0.42
    gold_min_quality: float = 0.78
    gold_min_support_conf: float = 0.005
    gold_min_area_ratio: float = 0.0012
    temporal_min_hits: int = 3
    temporal_hold_inferences: int = 3
    temporal_match_iou: float = 0.16

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CanPostProcessConfig":
        raw = dict(value or {})
        return cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            inference_conf=float(raw.get("inference_conf", cls.inference_conf)),
            tomato_min_conf=float(raw.get("tomato_min_conf", cls.tomato_min_conf)),
            tomato_direct_red_min_ratio=float(
                raw.get("tomato_direct_red_min_ratio", cls.tomato_direct_red_min_ratio)
            ),
            tomato_red_min_ratio=float(raw.get("tomato_red_min_ratio", cls.tomato_red_min_ratio)),
            tomato_low_conf_context_iou=float(
                raw.get("tomato_low_conf_context_iou", cls.tomato_low_conf_context_iou)
            ),
            gold_min_quality=float(raw.get("gold_min_quality", cls.gold_min_quality)),
            gold_min_support_conf=float(
                raw.get("gold_min_support_conf", cls.gold_min_support_conf)
            ),
            gold_min_area_ratio=float(
                raw.get("gold_min_area_ratio", cls.gold_min_area_ratio)
            ),
            temporal_min_hits=max(1, int(raw.get("temporal_min_hits", cls.temporal_min_hits))),
            temporal_hold_inferences=max(
                0, int(raw.get("temporal_hold_inferences", cls.temporal_hold_inferences))
            ),
            temporal_match_iou=float(raw.get("temporal_match_iou", cls.temporal_match_iou)),
        )


def _box(item: Mapping[str, Any] | Sequence[float]) -> tuple[float, float, float, float]:
    value = item.get("bbox") if isinstance(item, Mapping) else item
    if value is None or len(value) != 4:
        raise ValueError("bbox 必须是 4 个数")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _intersection(a: Sequence[float], b: Sequence[float]) -> float:
    width = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    height = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return width * height


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = _intersection(a, b)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def intersection_over_smaller(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return intersection / smaller if smaller > 0 else 0.0


def weighted_bbox_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the stronger of IoU and intersection-over-smaller-box."""
    return max(bbox_iou(a, b), intersection_over_smaller(a, b))


def _center(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def _center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def _red_label_ratio(frame: np.ndarray, box: Sequence[float]) -> float:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(math.floor(box[0]))))
    y1 = max(0, min(height - 1, int(math.floor(box[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(box[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(box[3]))))
    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(roi, np.array([0, 70, 40]), np.array([12, 255, 255]))
    high_red = cv2.inRange(roi, np.array([165, 60, 35]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(low_red, high_red)
    return float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0


def _gold_component_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([22, 35, 45]), np.array([78, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
        iterations=2,
    )


def _gold_mask_metrics(
    frame: np.ndarray,
    mask: np.ndarray,
    component_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, float] | None:
    component = (mask[y : y + height, x : x + width] == component_index).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0:
        return None

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    rectangle = cv2.minAreaRect(contour)
    rectangle_area = float(rectangle[1][0] * rectangle[1][1])
    bbox_area = float(width * height)
    aspect = float(max(width, height)) / float(max(1, min(width, height)))

    roi_bgr = frame[y : y + height, x : x + width]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    strict_gold = cv2.inRange(
        roi_hsv, np.array([22, 35, 45]), np.array([78, 255, 255])
    )
    strict_ratio = float(np.count_nonzero(strict_gold)) / max(1.0, bbox_area)
    component_hues = roi_hsv[:, :, 0][component > 0]
    hue_median = float(np.median(component_hues)) if component_hues.size else 0.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 120)
    edge_density = float(np.count_nonzero(edges)) / max(1.0, bbox_area)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=25,
        minLineLength=max(15, min(width, height) // 4),
        maxLineGap=4,
    )
    line_count = 0 if lines is None else int(len(lines))

    extent = contour_area / max(1.0, bbox_area)
    solidity = contour_area / hull_area if hull_area > 0 else 0.0
    rectangle_fill = contour_area / rectangle_area if rectangle_area > 0 else 0.0
    quality = (
        0.25 * min(1.0, extent)
        + 0.25 * min(1.0, solidity)
        + 0.20 * min(1.0, rectangle_fill)
        + 0.15 * min(1.0, strict_ratio / 0.45)
        + 0.15 * min(1.0, edge_density / 0.03)
    )
    return {
        "quality": float(quality),
        "extent": float(extent),
        "solidity": float(solidity),
        "rectangleFill": float(rectangle_fill),
        "strictGoldRatio": float(strict_ratio),
        "edgeDensity": float(edge_density),
        "lineCount": line_count,
        "aspect": aspect,
        "area": contour_area,
        "hueMedian": hue_median,
    }


@dataclass
class _CanTrack:
    class_id: int
    bbox: list[float]
    confidence: float
    source: str
    hits: int = 1
    misses: int = 0
    support_classes: list[int] = field(default_factory=list)


class CanTemporalStabilizer:
    """Small frame-to-frame tracker for the three can classes only."""

    def __init__(self, min_hits: int, hold_inferences: int, match_iou: float) -> None:
        self.min_hits = max(1, int(min_hits))
        self.hold_inferences = max(0, int(hold_inferences))
        self.match_iou = float(match_iou)
        self.tracks: list[_CanTrack] = []
        self.frame_index = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_index = 0

    def update(self, candidates: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.frame_index += 1
        used: set[int] = set()

        for candidate in sorted(
            candidates, key=lambda item: -float(item.get("confidence", 0.0))
        ):
            class_id = int(candidate["id"])
            box = [float(v) for v in candidate["bbox"]]
            best_index: int | None = None
            best_match = -1.0
            for index, track in enumerate(self.tracks):
                if index in used or track.class_id != class_id:
                    continue
                iou = bbox_iou(box, track.bbox)
                max_side = max(
                    track.bbox[2] - track.bbox[0],
                    track.bbox[3] - track.bbox[1],
                    box[2] - box[0],
                    box[3] - box[1],
                    1.0,
                )
                nearby = _center_distance(box, track.bbox) <= 0.42 * max_side
                score = max(iou, 0.5 if nearby else 0.0)
                if score > best_match and (iou >= self.match_iou or nearby):
                    best_index = index
                    best_match = score

            if best_index is None:
                self.tracks.append(
                    _CanTrack(
                        class_id=class_id,
                        bbox=box,
                        confidence=float(candidate.get("confidence", 0.0)),
                        source=str(candidate.get("source", "model")),
                        support_classes=[int(v) for v in candidate.get("supportClasses", [])],
                    )
                )
                used.add(len(self.tracks) - 1)
                continue

            track = self.tracks[best_index]
            used.add(best_index)
            alpha = 0.45
            track.bbox = [
                (1.0 - alpha) * track.bbox[index] + alpha * box[index]
                for index in range(4)
            ]
            track.confidence = (
                (1.0 - alpha) * track.confidence
                + alpha * float(candidate.get("confidence", 0.0))
            )
            track.source = str(candidate.get("source", track.source))
            track.support_classes = [
                int(v) for v in candidate.get("supportClasses", track.support_classes)
            ]
            track.hits += 1
            track.misses = 0

        for index, track in enumerate(self.tracks):
            if index not in used:
                track.misses += 1

        output: list[dict[str, Any]] = []
        live_tracks: list[_CanTrack] = []
        for track in self.tracks:
            if track.misses <= self.hold_inferences:
                live_tracks.append(track)
            if track.hits >= self.min_hits and track.misses <= self.hold_inferences:
                output.append(
                    {
                        "id": track.class_id,
                        "officialId": track.class_id + 1,
                        "name": "",  # filled by CanPostProcessor
                        "confidence": round(track.confidence, 4),
                        "bbox": [round(value, 1) for value in track.bbox],
                        "source": track.source,
                        "trackHits": track.hits,
                        "supportClasses": list(track.support_classes),
                    }
                )
        self.tracks = live_tracks
        debug = {
            "frameIndex": self.frame_index,
            "activeTracks": len(self.tracks),
            "emittedTracks": len(output),
        }
        return output, debug


class CanPostProcessor:
    def __init__(
        self,
        names: Sequence[str],
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.names = list(names)
        self.config = CanPostProcessConfig.from_mapping(config)
        self.stabilizer = CanTemporalStabilizer(
            min_hits=self.config.temporal_min_hits,
            hold_inferences=self.config.temporal_hold_inferences,
            match_iou=self.config.temporal_match_iou,
        )

    def reset(self) -> None:
        self.stabilizer.reset()

    def _name(self, class_id: int) -> str:
        return self.names[class_id] if 0 <= class_id < len(self.names) else str(class_id)

    def _detection(
        self,
        class_id: int,
        confidence: float,
        box: Sequence[float],
        source: str,
        **extra: Any,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": int(class_id),
            "officialId": int(class_id) + 1,
            "name": self._name(class_id),
            "confidence": round(float(confidence), 4),
            "bbox": [round(float(v), 1) for v in box],
            "source": source,
        }
        value.update(extra)
        return value

    @staticmethod
    def _best_for_class(
        detections: Sequence[Mapping[str, Any]], class_id: int
    ) -> Mapping[str, Any] | None:
        candidates = [item for item in detections if int(item.get("id", -1)) == class_id]
        return max(candidates, key=lambda item: float(item.get("confidence", 0.0))) if candidates else None

    def _gold_candidates(
        self,
        frame: np.ndarray,
        raw_detections: Sequence[Mapping[str, Any]],
        selected_cans: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_width * frame_height)
        mask = _gold_component_mask(frame)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for component_index in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[component_index])
            if width < 28 or height < 22 or width > frame_width * 0.72 or height > frame_height * 0.55:
                continue
            if float(area) < max(600.0, self.config.gold_min_area_ratio * frame_area):
                continue
            if float(area) / frame_area > 0.18:
                continue
            if x <= 0 or y <= 0 or x + width >= frame_width - 1 or y + height >= frame_height - 1:
                continue
            aspect = max(width, height) / max(1, min(width, height))
            if not 1.10 <= aspect <= 3.20:
                continue

            metrics = _gold_mask_metrics(frame, labels, component_index, x, y, width, height)
            if metrics is None:
                continue
            if (
                metrics["extent"] < 0.60
                or metrics["solidity"] < 0.82
                or metrics["rectangleFill"] < 0.68
                or metrics["strictGoldRatio"] < 0.22
                or not 0.010 <= metrics["edgeDensity"] <= 0.20
                or metrics["lineCount"] < 1
                or metrics["quality"] < self.config.gold_min_quality
            ):
                continue

            box = [float(x), float(y), float(x + width), float(y + height)]
            supporting_detections: list[Mapping[str, Any]] = []
            for detection in raw_detections:
                if int(detection.get("id", -1)) in CAN_IDS:
                    continue
                if float(detection.get("confidence", 0.0)) < self.config.gold_min_support_conf:
                    continue
                detection_box = _box(detection)
                if bbox_iou(box, detection_box) >= 0.20 or intersection_over_smaller(box, detection_box) >= 0.48:
                    supporting_detections.append(detection)

            nearby_other_cans: list[Mapping[str, Any]] = []
            for detection in selected_cans:
                class_id = int(detection.get("id", -1))
                if class_id == MEAT_CAN_ID:
                    continue
                other_box = _box(detection)
                if bbox_iou(box, other_box) >= 0.12:
                    continue
                radius = max(220.0, 2.25 * max(width, height))
                if _center_distance(box, other_box) <= radius:
                    nearby_other_cans.append(detection)

            context_ok = len(nearby_other_cans) >= 2
            support_classes = sorted(
                {int(item.get("id", -1)) for item in supporting_detections}
            )
            proxy_support_classes = {
                value
                for value in support_classes
                if value in {5, 6, 11, 18, 21, 22, 23, 27, 37, 39, 40}
            }
            detector_support_ok = bool(proxy_support_classes)
            # Eggs_Plush is the one ambiguous proxy: its warm plush can produce
            # a gold blob. Only accept that proxy for the dull/orange meat-can
            # variant, which has a lower hue and a much more rectangular shape.
            if (
                not context_ok
                and not detector_support_ok
                and 16 in support_classes
                and metrics["hueMedian"] <= 32.0
                and metrics["rectangleFill"] >= 0.78
                and metrics["extent"] >= 0.65
            ):
                detector_support_ok = True
            support_conf = max(
                (float(item.get("confidence", 0.0)) for item in supporting_detections),
                default=0.0,
            )
            record = {
                "bbox": [round(v, 1) for v in box],
                "quality": round(metrics["quality"], 4),
                "extent": round(metrics["extent"], 4),
                "solidity": round(metrics["solidity"], 4),
                "rectangleFill": round(metrics["rectangleFill"], 4),
                "strictGoldRatio": round(metrics["strictGoldRatio"], 4),
                "edgeDensity": round(metrics["edgeDensity"], 4),
                "lineCount": metrics["lineCount"],
                "hueMedian": round(metrics["hueMedian"], 2),
                "contextCanIds": [int(item["id"]) for item in nearby_other_cans],
                "supportClasses": support_classes,
                "supportConfidence": round(support_conf, 4),
            }
            if not (context_ok or detector_support_ok):
                rejected.append({**record, "reason": "no_can_context_or_detector_support"})
                continue
            accepted.append(
                self._detection(
                    MEAT_CAN_ID,
                    metrics["quality"],
                    box,
                    "gold_shape_rule",
                    ruleConfidence=round(metrics["quality"], 4),
                    supportConfidence=round(support_conf, 4),
                    supportClasses=support_classes,
                    goldMetrics=record,
                )
            )

        accepted.sort(key=lambda item: -float(item["confidence"]))
        return accepted[:1], rejected[:4]

    def process(
        self,
        frame: np.ndarray,
        raw_detections: Sequence[Mapping[str, Any]],
        final_conf: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        final_conf = float(final_conf)
        if not self.config.enabled:
            output = [
                dict(item)
                for item in raw_detections
                if float(item.get("confidence", 0.0)) >= final_conf
            ]
            output.sort(key=lambda item: -float(item.get("confidence", 0.0)))
            return output, {"enabled": False}

        selected_cans: list[dict[str, Any]] = []

        tuna = self._best_for_class(raw_detections, TUNA_ID)
        if tuna is not None and float(tuna.get("confidence", 0.0)) >= final_conf:
            selected_cans.append(self._detection(
                TUNA_ID,
                float(tuna["confidence"]),
                _box(tuna),
                "model",
            ))

        tomato = self._best_for_class(raw_detections, TOMATO_ID)
        tomato_support: Mapping[str, Any] | None = None
        tomato_red_ratio = 0.0
        if tomato is not None:
            tomato_box = _box(tomato)
            tomato_red_ratio = _red_label_ratio(frame, tomato_box)
            if (
                float(tomato.get("confidence", 0.0)) >= final_conf
                and tomato_red_ratio >= self.config.tomato_direct_red_min_ratio
            ):
                tomato_support = tomato
            else:
                overlapping_confusers = [
                    item
                    for item in raw_detections
                    if int(item.get("id", -1)) in CAN_CONFUSERS[TOMATO_ID]
                    and weighted_bbox_overlap(tomato_box, _box(item))
                    >= self.config.tomato_low_conf_context_iou
                    and float(item.get("confidence", 0.0)) >= final_conf
                ]
                if (
                    float(tomato.get("confidence", 0.0)) >= self.config.tomato_min_conf
                    and tomato_red_ratio >= self.config.tomato_red_min_ratio
                    and overlapping_confusers
                ):
                    tomato_support = tomato
            if tomato_support is not None:
                selected_cans.append(
                    self._detection(
                        TOMATO_ID,
                        float(tomato_support["confidence"]),
                        tomato_box,
                        "model" if float(tomato_support["confidence"]) >= final_conf else "can_arbitration",
                        redLabelRatio=round(tomato_red_ratio, 4),
                    )
                )

        meat_can = self._best_for_class(raw_detections, MEAT_CAN_ID)
        if meat_can is not None and float(meat_can.get("confidence", 0.0)) >= final_conf:
            selected_cans.append(
                self._detection(
                    MEAT_CAN_ID,
                    float(meat_can["confidence"]),
                    _box(meat_can),
                    "model",
                )
            )

        gold_candidates, rejected_gold = self._gold_candidates(
            frame, raw_detections, selected_cans
        )
        gold_fallback = None
        if gold_candidates:
            candidate = gold_candidates[0]
            if not any(
                int(item["id"]) == MEAT_CAN_ID
                and weighted_bbox_overlap(_box(item), _box(candidate)) >= 0.35
                for item in selected_cans
            ):
                selected_cans.append(candidate)
                gold_fallback = candidate

        # Suppress only known same-object confusers that overlap a selected can.
        suppressed: list[dict[str, Any]] = []
        keep_non_can: list[dict[str, Any]] = []
        for detection in raw_detections:
            class_id = int(detection.get("id", -1))
            detection_box = _box(detection)
            suppress = False
            for can in selected_cans:
                can_id = int(can["id"])
                can_box = _box(can)
                known_confuser = class_id in CAN_CONFUSERS[can_id]
                strong_same_object = (
                    bbox_iou(detection_box, can_box) >= 0.30
                    or intersection_over_smaller(detection_box, can_box) >= 0.68
                )
                if known_confuser and strong_same_object:
                    suppress = True
                    break
                if (
                    can_id == MEAT_CAN_ID
                    and can.get("source") == "gold_shape_rule"
                    and class_id not in CAN_IDS
                    and strong_same_object
                ):
                    suppress = True
                    break
            if suppress:
                suppressed.append(dict(detection))
                continue
            if class_id in CAN_IDS:
                continue
            if float(detection.get("confidence", 0.0)) >= final_conf:
                keep_non_can.append(dict(detection))

        stabilized, tracker_debug = self.stabilizer.update(selected_cans)
        for item in stabilized:
            item["name"] = self._name(int(item["id"]))

        output = keep_non_can + stabilized
        output.sort(key=lambda item: -float(item.get("confidence", 0.0)))
        debug = {
            "enabled": True,
            "inferenceConf": self.config.inference_conf,
            "finalConf": round(final_conf, 4),
            "rawDetectionCount": len(raw_detections),
            "selectedCans": [
                {
                    "id": int(item["id"]),
                    "confidence": round(float(item["confidence"]), 4),
                    "bbox": [round(float(v), 1) for v in item["bbox"]],
                    "source": str(item.get("source", "model")),
                }
                for item in selected_cans
            ],
            "suppressed": [
                {
                    "id": int(item.get("id", -1)),
                    "confidence": round(float(item.get("confidence", 0.0)), 4),
                }
                for item in suppressed
            ],
            "tomatoRedRatio": round(tomato_red_ratio, 4),
            "goldFallbackActive": gold_fallback is not None,
            "goldFallback": (
                {
                    key: value
                    for key, value in gold_fallback.items()
                    if key not in {"name"}
                }
                if gold_fallback is not None
                else None
            ),
            "goldRejected": rejected_gold,
            "tracker": tracker_debug,
        }
        return output, debug
