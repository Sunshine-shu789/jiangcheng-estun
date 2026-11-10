#!/usr/bin/env python3
"""Conservative fusion of a 40-class detector and a dedicated 3-class can detector.

The secondary detector is the only source of can-class boxes.  The primary
detector remains the sole source of every non-can class.  A primary detection
is removed only when it is a can class or when it is a known can-confuser that
strongly overlaps a trusted secondary can box.  This keeps the blast radius of
the correction limited to the three can objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


SECONDARY_TO_PRIMARY = {0: 1, 1: 4, 2: 9}
PRIMARY_CAN_IDS = frozenset(SECONDARY_TO_PRIMARY.values())
DEFAULT_SECONDARY_NAMES = (
    "2-Starkist_Tuna",
    "5-Tomato_Soup",
    "10-meat_can",
)
# Classes repeatedly produced on the same pixels as a specific physical can.
# Suppression remains per-can and still requires high geometric overlap with a
# trusted secondary prediction, so a global class name alone can never delete
# a real object elsewhere in the frame.
DEFAULT_CONFUSERS_BY_CAN_ID: dict[int, frozenset[int]] = {
    1: frozenset({11}),
    4: frozenset({11, 14, 17, 20, 22, 26, 28}),
    9: frozenset({5, 6, 11, 15, 16, 17, 18, 21, 22, 23, 27, 28, 37, 39}),
}


def _bbox(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
    value = item.get("bbox")
    if value is None or len(value) != 4:
        raise ValueError("bbox must contain four numbers")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def bbox_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    width = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    height = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return width * height


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    overlap = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - overlap
    return overlap / union if union > 0.0 else 0.0


def intersection_over_smaller(a: Sequence[float], b: Sequence[float]) -> float:
    overlap = intersection_area(a, b)
    smaller = min(bbox_area(a), bbox_area(b))
    return overlap / smaller if smaller > 0.0 else 0.0


def same_physical_box(
    a: Sequence[float],
    b: Sequence[float],
    *,
    min_iou: float,
    min_ios: float,
) -> bool:
    return bbox_iou(a, b) >= min_iou or intersection_over_smaller(a, b) >= min_ios


def color_evidence_ratios(
    frame: Any | None,
    box: Sequence[float],
) -> tuple[float, float] | None:
    """Return the red-label and low-saturation bright-body ratios for one box."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return None
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height <= 0 or width <= 0:
        return None
    x1 = max(0, min(width - 1, int(np.floor(float(box[0])))))
    y1 = max(0, min(height - 1, int(np.floor(float(box[1])))))
    x2 = max(x1 + 1, min(width, int(np.ceil(float(box[2])))))
    y2 = max(y1 + 1, min(height, int(np.ceil(float(box[3])))))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([12, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([165, 60, 35]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(low_red, high_red)
    white = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([179, 85, 255]))
    red_ratio = float(np.count_nonzero(red)) / float(red.size)
    white_ratio = float(np.count_nonzero(white)) / float(white.size)
    return red_ratio, white_ratio


@dataclass(frozen=True)
class CanFusionConfig:
    enabled: bool = True
    secondary_conf: float = 0.15
    strong_conf: float = 0.30
    base_fallback_conf: float = 0.75
    match_iou: float = 0.30
    match_ios: float = 0.62
    suppress_iou: float = 0.42
    suppress_ios: float = 0.70
    min_box_area_ratio: float = 0.0015
    min_box_side: float = 24.0
    max_box_aspect: float = 3.4
    max_det_per_class: int = 3
    temporal_min_hits: int = 2
    temporal_hold_inferences: int = 6
    temporal_match_iou: float = 0.16
    primary_inference_conf: float = 0.01
    raw_accept_conf: float = 0.40
    primary_evidence_weight: float = 0.35
    primary_tuna_evidence_conf: float = 0.10
    primary_tomato_evidence_conf: float = 0.05
    evidence_group_iou: float = 0.55
    evidence_group_ios: float = 0.80
    evidence_group_max_area_ratio: float = 1.65
    tomato_red_min_ratio: float = 0.06
    tomato_red_full_ratio: float = 0.15
    tomato_color_bonus: float = 0.30
    tuna_red_max_ratio: float = 0.025
    tuna_white_min_ratio: float = 0.18
    tuna_color_bonus: float = 0.12
    primary_color_rescue_enabled: bool = True
    primary_color_rescue_min_white_ratio: float = 0.20
    primary_override_conf: float = 0.75
    primary_override_margin: float = 0.10
    primary_override_iou: float = 0.42
    primary_override_ios: float = 0.70
    confusers_by_can_id: Mapping[int, frozenset[int]] = field(
        default_factory=lambda: dict(DEFAULT_CONFUSERS_BY_CAN_ID)
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CanFusionConfig":
        raw = dict(value or {})
        raw_by_can = raw.get("confusers_by_can_id")
        if raw_by_can is None:
            legacy = raw.get("confuser_ids")
            if legacy is None:
                confusers_by_can_id = dict(DEFAULT_CONFUSERS_BY_CAN_ID)
            else:
                if not isinstance(legacy, Iterable) or isinstance(
                    legacy, (str, bytes)
                ):
                    raise ValueError("can_secondary.confuser_ids must be a list")
                shared = frozenset(int(v) for v in legacy)
                confusers_by_can_id = {class_id: shared for class_id in (1, 4, 9)}
        else:
            if not isinstance(raw_by_can, Mapping):
                raise ValueError(
                    "can_secondary.confusers_by_can_id must be a mapping"
                )
            confusers_by_can_id = {}
            for class_id in (1, 4, 9):
                values = raw_by_can.get(
                    class_id,
                    raw_by_can.get(str(class_id), DEFAULT_CONFUSERS_BY_CAN_ID[class_id]),
                )
                if not isinstance(values, Iterable) or isinstance(
                    values, (str, bytes)
                ):
                    raise ValueError(
                        f"can_secondary.confusers_by_can_id[{class_id}] must be a list"
                    )
                confusers_by_can_id[class_id] = frozenset(int(v) for v in values)
        return cls(
            enabled=bool(raw.get("enabled", True)),
            secondary_conf=float(raw.get("secondary_conf", 0.15)),
            strong_conf=float(raw.get("strong_conf", 0.30)),
            base_fallback_conf=float(raw.get("base_fallback_conf", 0.75)),
            match_iou=float(raw.get("match_iou", 0.30)),
            match_ios=float(raw.get("match_ios", 0.62)),
            suppress_iou=float(raw.get("suppress_iou", 0.42)),
            suppress_ios=float(raw.get("suppress_ios", 0.70)),
            min_box_area_ratio=float(raw.get("min_box_area_ratio", 0.0015)),
            min_box_side=float(raw.get("min_box_side", 24.0)),
            max_box_aspect=float(raw.get("max_box_aspect", 3.4)),
            max_det_per_class=max(1, int(raw.get("max_det_per_class", 3))),
            temporal_min_hits=max(1, int(raw.get("temporal_min_hits", 2))),
            temporal_hold_inferences=max(
                0, int(raw.get("temporal_hold_inferences", 6))
            ),
            temporal_match_iou=float(raw.get("temporal_match_iou", 0.16)),
            primary_inference_conf=float(raw.get("primary_inference_conf", 0.01)),
            raw_accept_conf=float(raw.get("raw_accept_conf", 0.40)),
            primary_evidence_weight=float(raw.get("primary_evidence_weight", 0.35)),
            primary_tuna_evidence_conf=float(
                raw.get("primary_tuna_evidence_conf", 0.10)
            ),
            primary_tomato_evidence_conf=float(
                raw.get("primary_tomato_evidence_conf", 0.05)
            ),
            evidence_group_iou=float(raw.get("evidence_group_iou", 0.55)),
            evidence_group_ios=float(raw.get("evidence_group_ios", 0.80)),
            evidence_group_max_area_ratio=float(
                raw.get("evidence_group_max_area_ratio", 1.65)
            ),
            tomato_red_min_ratio=float(raw.get("tomato_red_min_ratio", 0.06)),
            tomato_red_full_ratio=float(raw.get("tomato_red_full_ratio", 0.15)),
            tomato_color_bonus=float(raw.get("tomato_color_bonus", 0.30)),
            tuna_red_max_ratio=float(raw.get("tuna_red_max_ratio", 0.025)),
            tuna_white_min_ratio=float(raw.get("tuna_white_min_ratio", 0.18)),
            tuna_color_bonus=float(raw.get("tuna_color_bonus", 0.12)),
            primary_color_rescue_enabled=bool(
                raw.get("primary_color_rescue_enabled", True)
            ),
            primary_color_rescue_min_white_ratio=float(
                raw.get("primary_color_rescue_min_white_ratio", 0.20)
            ),
            primary_override_conf=float(raw.get("primary_override_conf", 0.75)),
            primary_override_margin=float(raw.get("primary_override_margin", 0.10)),
            primary_override_iou=float(raw.get("primary_override_iou", 0.42)),
            primary_override_ios=float(raw.get("primary_override_ios", 0.70)),
            confusers_by_can_id={
                class_id: frozenset(values)
                for class_id, values in confusers_by_can_id.items()
            },
        )


@dataclass
class _Track:
    class_id: int
    bbox: list[float]
    confidence: float
    name: str
    source: str
    hits: int = 1
    misses: int = 0


class CanTemporalStabilizer:
    """A small class-aware tracker; it never invents a box without evidence."""

    def __init__(self, config: CanFusionConfig) -> None:
        self.config = config
        self.tracks: list[_Track] = []
        self.frame_index = 0

    def reset(self) -> None:
        self.tracks = []
        self.frame_index = 0

    def _nearby(self, a: Sequence[float], b: Sequence[float]) -> bool:
        ax = (float(a[0]) + float(a[2])) * 0.5
        ay = (float(a[1]) + float(a[3])) * 0.5
        bx = (float(b[0]) + float(b[2])) * 0.5
        by = (float(b[1]) + float(b[3])) * 0.5
        max_side = max(
            float(a[2]) - float(a[0]),
            float(a[3]) - float(a[1]),
            float(b[2]) - float(b[0]),
            float(b[3]) - float(b[1]),
            1.0,
        )
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= 0.35 * max_side

    def update(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        self.frame_index += 1
        used: set[int] = set()
        for candidate in sorted(
            candidates, key=lambda item: -float(item.get("confidence", 0.0))
        ):
            class_id = int(candidate["id"])
            box = [float(v) for v in candidate["bbox"]]
            best_index: int | None = None
            best_score = -1.0
            for index, track in enumerate(self.tracks):
                if index in used:
                    continue
                if track.class_id != class_id:
                    continue
                iou = bbox_iou(box, track.bbox)
                if iou < self.config.temporal_match_iou and not self._nearby(
                    box, track.bbox
                ):
                    continue
                if iou > best_score:
                    best_index = index
                    best_score = iou
            if best_index is None:
                self.tracks.append(
                    _Track(
                        class_id=class_id,
                        bbox=box,
                        confidence=float(candidate.get("confidence", 0.0)),
                        name=str(candidate.get("name", "")),
                        source=str(candidate.get("source", "secondary_model")),
                    )
                )
                used.add(len(self.tracks) - 1)
                continue
            track = self.tracks[best_index]
            used.add(best_index)
            alpha = 0.45
            track.bbox = [
                (1.0 - alpha) * track.bbox[i] + alpha * box[i] for i in range(4)
            ]
            track.confidence = (
                (1.0 - alpha) * track.confidence
                + alpha * float(candidate.get("confidence", 0.0))
            )
            track.name = str(candidate.get("name", track.name))
            track.source = str(candidate.get("source", track.source))
            track.hits += 1
            track.misses = 0

        for index, track in enumerate(self.tracks):
            if index not in used:
                track.misses += 1

        output: list[dict[str, Any]] = []
        live: list[_Track] = []
        for track in self.tracks:
            if track.misses <= self.config.temporal_hold_inferences:
                live.append(track)
            if (
                track.hits >= self.config.temporal_min_hits
                and track.misses <= self.config.temporal_hold_inferences
            ):
                output.append(
                    {
                        "id": track.class_id,
                        "officialId": track.class_id + 1,
                        "name": track.name,
                        "confidence": round(track.confidence, 4),
                        "bbox": [round(v, 1) for v in track.bbox],
                        "source": track.source,
                        "trackHits": track.hits,
                        "trackMisses": track.misses,
                    }
                )
        self.tracks = live
        return output


class CanFusion:
    def __init__(
        self,
        primary_names: Sequence[str],
        config: CanFusionConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.primary_names = [str(v) for v in primary_names]
        self.config = (
            config
            if isinstance(config, CanFusionConfig)
            else CanFusionConfig.from_mapping(config)
        )
        self.stabilizer = CanTemporalStabilizer(self.config)

    def reset(self) -> None:
        self.stabilizer.reset()

    def _geometry_valid(self, item: Mapping[str, Any], frame_shape: Sequence[int]) -> bool:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        box = _bbox(item)
        box_width = max(0.0, box[2] - box[0])
        box_height = max(0.0, box[3] - box[1])
        if min(box_width, box_height) < self.config.min_box_side:
            return False
        area_ratio = bbox_area(box) / float(width * height)
        if area_ratio < self.config.min_box_area_ratio:
            return False
        aspect = max(box_width, box_height) / max(1.0, min(box_width, box_height))
        return aspect <= self.config.max_box_aspect

    def _same_evidence_group(
        self,
        a: Sequence[float],
        b: Sequence[float],
    ) -> bool:
        overlap_iou = bbox_iou(a, b)
        overlap_smaller = intersection_over_smaller(a, b)
        area_ratio = max(bbox_area(a), bbox_area(b)) / max(
            1.0, min(bbox_area(a), bbox_area(b))
        )
        return overlap_iou >= self.config.evidence_group_iou or (
            overlap_smaller >= self.config.evidence_group_ios
            and area_ratio <= self.config.evidence_group_max_area_ratio
        )

    def _primary_can_support(
        self,
        primary_items: Sequence[Mapping[str, Any]],
        class_id: int,
        group: Sequence[Mapping[str, Any]],
    ) -> float:
        best = 0.0
        for item in primary_items:
            if int(item["id"]) != class_id:
                continue
            item_box = _bbox(item)
            if any(
                same_physical_box(
                    item_box,
                    _bbox(candidate),
                    min_iou=self.config.match_iou,
                    min_ios=self.config.match_ios,
                )
                for candidate in group
            ):
                best = max(best, float(item.get("confidence", 0.0)))
        return best

    def _color_bonus(
        self,
        class_id: int,
        raw_confidence: float,
        primary_confidence: float,
        ratios: tuple[float, float] | None,
    ) -> tuple[float, str]:
        if ratios is None:
            return 0.0, ""
        red_ratio, white_ratio = ratios
        if (
            class_id == 4
            and red_ratio >= self.config.tomato_red_min_ratio
        ):
            scaled = min(
                1.0,
                red_ratio / max(1e-6, self.config.tomato_red_full_ratio),
            )
            return self.config.tomato_color_bonus * scaled, "red_label"
        if (
            class_id == 1
            and red_ratio <= self.config.tuna_red_max_ratio
            and white_ratio >= self.config.tuna_white_min_ratio
            and (
                raw_confidence >= 0.25
                or primary_confidence >= self.config.primary_tuna_evidence_conf
            )
        ):
            return self.config.tuna_color_bonus, "plain_metal"
        return 0.0, ""

    def _primary_color_rescues(
        self,
        primary_items: Sequence[Mapping[str, Any]],
        frame_shape: Sequence[int],
        frame: Any | None,
        current: Sequence[Mapping[str, Any]],
        final_conf: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Recover a Tomato can when the dedicated detector omits its box.

        This is intentionally narrower than a generic low-threshold primary
        fallback.  It requires the primary Tomato class, a red label, a bright
        low-saturation can body, and no dedicated can box on the same object.
        A strong non-confuser detection on the same pixels vetoes the rescue,
        which prevents red bottles and other unrelated red objects from being
        relabelled as Tomato soup.
        """
        if not self.config.primary_color_rescue_enabled or frame is None:
            return [], []

        accepted: list[dict[str, Any]] = []
        assessments: list[dict[str, Any]] = []
        allowed_confusers = self.config.confusers_by_can_id.get(4, frozenset())
        for item in primary_items:
            if int(item.get("id", -1)) != 4:
                continue
            confidence = float(item.get("confidence", 0.0))
            if confidence < self.config.primary_tomato_evidence_conf:
                continue
            box = _bbox(item)
            ratios = color_evidence_ratios(frame, box)
            if ratios is None:
                continue
            red_ratio, white_ratio = ratios
            if red_ratio < self.config.tomato_red_min_ratio:
                continue

            assessment = {
                "bbox": [round(value, 1) for value in box],
                "primaryConfidence": round(confidence, 4),
                "redLabelRatio": round(red_ratio, 4),
                "whiteBodyRatio": round(white_ratio, 4),
                "accepted": False,
            }
            if not self._geometry_valid(item, frame_shape):
                assessments.append({**assessment, "reason": "geometry"})
                continue
            if white_ratio < self.config.primary_color_rescue_min_white_ratio:
                assessments.append({**assessment, "reason": "insufficient_white_body"})
                continue
            if any(
                same_physical_box(
                    box,
                    _bbox(anchor),
                    min_iou=self.config.match_iou,
                    min_ios=self.config.match_ios,
                )
                for anchor in current
            ):
                assessments.append(
                    {**assessment, "reason": "secondary_can_candidate_present"}
                )
                continue

            blockers = [
                candidate
                for candidate in primary_items
                if int(candidate.get("id", -1)) not in PRIMARY_CAN_IDS
                and int(candidate.get("id", -1)) not in allowed_confusers
                and float(candidate.get("confidence", 0.0)) >= final_conf
                and same_physical_box(
                    box,
                    _bbox(candidate),
                    min_iou=self.config.suppress_iou,
                    min_ios=self.config.suppress_ios,
                )
            ]
            if blockers:
                assessments.append(
                    {
                        **assessment,
                        "reason": "non_confuser_overlap",
                        "blockers": blockers,
                    }
                )
                continue

            scaled_red = min(
                1.0,
                red_ratio / max(1e-6, self.config.tomato_red_full_ratio),
            )
            evidence_confidence = min(
                0.999,
                confidence + self.config.tomato_color_bonus * scaled_red,
            )
            if evidence_confidence < final_conf:
                assessments.append(
                    {**assessment, "reason": "insufficient_evidence"}
                )
                continue
            rescue = {
                "id": 4,
                "officialId": 5,
                "name": self.primary_names[4],
                "confidence": round(evidence_confidence, 4),
                "bbox": [round(value, 1) for value in box],
                "source": "primary_color_fallback",
                "primaryConfidence": round(confidence, 4),
                "evidenceSource": "primary_and_color",
                "redLabelRatio": round(red_ratio, 4),
                "whiteBodyRatio": round(white_ratio, 4),
            }
            accepted.append(rescue)
            assessments.append(
                {
                    **assessment,
                    "accepted": True,
                    "reason": "red_label_white_body",
                    "evidenceConfidence": rescue["confidence"],
                }
            )
        return accepted, assessments

    def _secondary_candidates(
        self,
        secondary: Sequence[Mapping[str, Any]],
        frame_shape: Sequence[int],
        primary_items: Sequence[Mapping[str, Any]],
        final_conf: float,
        frame: Any | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Keep ambiguous can evidence until it can be arbitrated as an object.

        The dedicated detector occasionally emits Tuna and Tomato for the same
        silver can in one inference.  Keeping only the highest score makes the
        output depend on a few thousandths of confidence.  Here every class is
        retained until the physical box is grouped, then colour and low-conf
        primary-can evidence select exactly one identity for that object.
        """
        rejected: list[dict[str, Any]] = []
        by_class: dict[int, list[dict[str, Any]]] = {1: [], 4: [], 9: []}
        for item in secondary:
            secondary_id = int(item.get("class_id", item.get("id", -1)))
            primary_id = SECONDARY_TO_PRIMARY.get(secondary_id)
            confidence = float(item.get("confidence", 0.0))
            if primary_id is None:
                continue
            candidate = {
                "id": primary_id,
                "officialId": primary_id + 1,
                "name": self.primary_names[primary_id],
                "confidence": round(confidence, 4),
                "bbox": [round(float(v), 1) for v in _bbox(item)],
                "source": "secondary_model",
            }
            if confidence < self.config.secondary_conf:
                continue
            if not self._geometry_valid(candidate, frame_shape):
                rejected.append({**candidate, "reason": "geometry"})
                continue
            by_class[primary_id].append(candidate)

        candidates: list[dict[str, Any]] = []
        for class_id in (1, 4, 9):
            ranked = sorted(
                by_class[class_id], key=lambda item: -float(item["confidence"])
            )
            candidates.extend(ranked[: self.config.max_det_per_class])

        # Union overlapping boxes into physical objects.  Chaining is allowed
        # only through the high overlap thresholds above, so the three nearby
        # cans in this scene remain three separate groups.
        parent = list(range(len(candidates)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                if self._same_evidence_group(
                    candidates[left]["bbox"], candidates[right]["bbox"]
                ):
                    union(left, right)

        grouped: dict[int, list[dict[str, Any]]] = {}
        for index, candidate in enumerate(candidates):
            grouped.setdefault(find(index), []).append(candidate)

        accepted: list[dict[str, Any]] = []
        arbitrations: list[dict[str, Any]] = []
        for group in grouped.values():
            group.sort(key=lambda item: -float(item["confidence"]))
            representative = group[0]
            ratios = color_evidence_ratios(frame, representative["bbox"])
            red_ratio = ratios[0] if ratios is not None else None
            white_ratio = ratios[1] if ratios is not None else None
            primary_support = {
                class_id: self._primary_can_support(
                    primary_items, class_id, group
                )
                for class_id in (1, 4, 9)
            }

            records_by_class: dict[int, dict[str, Any]] = {}
            all_records: list[dict[str, Any]] = []
            for candidate in group:
                class_id = int(candidate["id"])
                raw_confidence = float(candidate["confidence"])
                primary_confidence = primary_support[class_id]
                color_bonus, color_source = self._color_bonus(
                    class_id,
                    raw_confidence,
                    primary_confidence,
                    ratios,
                )
                evidence = (
                    raw_confidence
                    + self.config.primary_evidence_weight * primary_confidence
                    + color_bonus
                )
                if primary_confidence > 0.0 and color_bonus > 0.0:
                    evidence_source = "primary_and_color"
                elif primary_confidence > 0.0:
                    evidence_source = "primary"
                elif color_bonus > 0.0:
                    evidence_source = "color"
                else:
                    evidence_source = "model"
                record = {
                    **candidate,
                    "confidence": round(min(0.999, max(0.0, evidence)), 4),
                    "modelConfidence": round(raw_confidence, 4),
                    "primaryEvidenceConfidence": round(primary_confidence, 4),
                    "colorEvidence": color_source,
                    "evidenceSource": evidence_source,
                    "secondaryClassMatched": True,
                    "redLabelRatio": (
                        round(red_ratio, 4) if red_ratio is not None else None
                    ),
                    "whiteBodyRatio": (
                        round(white_ratio, 4) if white_ratio is not None else None
                    ),
                }
                all_records.append(record)
                previous = records_by_class.get(class_id)
                if previous is None or (
                    float(record["confidence"]), raw_confidence
                ) > (
                    float(previous["confidence"]),
                    float(previous["modelConfidence"]),
                ):
                    records_by_class[class_id] = record

            # If one class is missing but the primary detector and can body
            # colour identify it, use the objectness box from the other class.
            # This is deliberately limited to Tuna/Tomato and to a strong body
            # colour contradiction; arbitrary can relabelling is not allowed.
            max_raw = float(representative["confidence"])
            for class_id, minimum_primary in (
                (1, self.config.primary_tuna_evidence_conf),
                (4, self.config.primary_tomato_evidence_conf),
            ):
                if class_id in records_by_class:
                    continue
                primary_confidence = primary_support[class_id]
                if primary_confidence < minimum_primary:
                    continue
                alternative_raw = max_raw * 0.90
                color_bonus, color_source = self._color_bonus(
                    class_id,
                    alternative_raw,
                    primary_confidence,
                    ratios,
                )
                if color_bonus <= 0.0:
                    continue
                evidence = (
                    alternative_raw
                    + self.config.primary_evidence_weight * primary_confidence
                    + color_bonus
                )
                record = {
                    "id": class_id,
                    "officialId": class_id + 1,
                    "name": self.primary_names[class_id],
                    "confidence": round(min(0.999, max(0.0, evidence)), 4),
                    "bbox": list(representative["bbox"]),
                    "source": "secondary_color_arbitrated",
                    "modelConfidence": round(alternative_raw, 4),
                    "primaryEvidenceConfidence": round(primary_confidence, 4),
                    "colorEvidence": color_source,
                    "evidenceSource": "primary_and_color",
                    "secondaryClassMatched": False,
                    "rewrittenFrom": int(representative["id"]),
                    "redLabelRatio": (
                        round(red_ratio, 4) if red_ratio is not None else None
                    ),
                    "whiteBodyRatio": (
                        round(white_ratio, 4) if white_ratio is not None else None
                    ),
                }
                all_records.append(record)
                records_by_class[class_id] = record

            ranked = sorted(
                records_by_class.values(),
                key=lambda item: (
                    -float(item["confidence"]),
                    -float(item["modelConfidence"]),
                    int(item["id"]),
                ),
            )
            if not ranked:
                continue
            winner = ranked[0]
            decision = "highest_evidence"

            # Strong body-colour contradictions take precedence over small
            # class-score differences.  The primary evidence requirement keeps
            # a rotated, low-red Tomato can from being forced to Tuna merely
            # because its label is out of view.
            if ratios is not None:
                primary_tuna = primary_support[1]
                primary_tomato = primary_support[4]
                if (
                    red_ratio is not None
                    and white_ratio is not None
                    and red_ratio <= self.config.tuna_red_max_ratio
                    and white_ratio >= self.config.tuna_white_min_ratio
                    and primary_tuna >= self.config.primary_tuna_evidence_conf
                    and primary_tuna >= primary_tomato
                    and 1 in records_by_class
                ):
                    winner = records_by_class[1]
                    decision = "plain_metal_primary_support"
                elif (
                    red_ratio is not None
                    and red_ratio >= self.config.tomato_red_min_ratio
                    and primary_tomato >= self.config.primary_tomato_evidence_conf
                    and primary_tomato >= primary_tuna
                    and 4 in records_by_class
                ):
                    winner = records_by_class[4]
                    decision = "red_label_primary_support"

            supported = (
                float(winner["modelConfidence"]) >= self.config.raw_accept_conf
                or float(winner["primaryEvidenceConfidence"])
                >= (
                    self.config.primary_tuna_evidence_conf
                    if int(winner["id"]) == 1
                    else self.config.primary_tomato_evidence_conf
                )
                or bool(winner.get("colorEvidence"))
            )
            evidence_accepted = (
                float(winner["confidence"]) >= final_conf and supported
            )
            arbitration = {
                "bbox": list(representative["bbox"]),
                "decision": decision,
                "accepted": evidence_accepted,
                "chosenClass": int(winner["id"]),
                "chosenModelClassMatched": bool(
                    winner.get("secondaryClassMatched", False)
                ),
                "redLabelRatio": (
                    round(red_ratio, 4) if red_ratio is not None else None
                ),
                "whiteBodyRatio": (
                    round(white_ratio, 4) if white_ratio is not None else None
                ),
                "primarySupport": {
                    self.primary_names[class_id]: round(confidence, 4)
                    for class_id, confidence in primary_support.items()
                    if confidence > 0.0
                },
                "candidates": [
                    {
                        "id": int(record["id"]),
                        "name": record["name"],
                        "modelConfidence": record["modelConfidence"],
                        "evidenceConfidence": record["confidence"],
                        "secondaryClassMatched": record["secondaryClassMatched"],
                    }
                    for record in ranked
                ],
            }
            if (
                len(records_by_class) > 1
                or not bool(winner.get("secondaryClassMatched", False))
                or ratios is not None
            ):
                arbitrations.append(arbitration)

            if evidence_accepted:
                accepted.append(winner)
                for record in all_records:
                    if record is winner:
                        continue
                    rejected.append(
                        {
                            **record,
                            "reason": "physical_group_loser",
                            "chosenId": int(winner["id"]),
                        }
                    )
            else:
                for record in all_records:
                    rejected.append(
                        {
                            **record,
                            "reason": "insufficient_evidence",
                        }
                    )

        accepted.sort(key=lambda item: -float(item["confidence"]))
        return accepted, rejected, arbitrations

    def fuse(
        self,
        frame_shape: Sequence[int],
        primary_detections: Sequence[Mapping[str, Any]],
        secondary_detections: Sequence[Mapping[str, Any]],
        *,
        final_conf: float,
        use_temporal: bool = True,
        frame: Any | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        primary_items: list[dict[str, Any]] = []
        for raw in primary_detections:
            item = dict(raw)
            # The acceptance server emits canonical `id` fields, while the
            # offline evaluator stores raw YOLO results with `class_id`.
            # Accept both forms so production and evaluation exercise the
            # exactly same fusion implementation.
            class_id = int(item.get("id", item.get("class_id", -1)))
            item["id"] = class_id
            item.setdefault("officialId", class_id + 1)
            if 0 <= class_id < len(self.primary_names):
                item.setdefault("name", self.primary_names[class_id])
            primary_items.append(item)

        current, rejected, secondary_arbitrations = self._secondary_candidates(
            secondary_detections,
            frame_shape,
            primary_items,
            final_conf,
            frame,
        )
        secondary_current = list(current)
        primary_color_rescues, primary_color_assessments = (
            self._primary_color_rescues(
                primary_items,
                frame_shape,
                frame,
                secondary_current,
                final_conf,
            )
        )
        current = secondary_current + primary_color_rescues
        stabilized = self.stabilizer.update(current) if use_temporal else list(current)
        anchors = [
            item for item in stabilized if float(item["confidence"]) >= final_conf
        ]

        # A confident primary can box may arbitrate only when it is
        # substantially stronger than a *different-class* secondary anchor on
        # the same pixels.  This is deliberately narrow: it recovers a stable
        # Tuna/Tomato identity after a tracker flip without allowing the
        # primary model to redefine ordinary secondary-can output.
        arbitration_primary_indexes: set[int] = set()
        overridden_anchor_indexes: set[int] = set()
        primary_arbitrations: list[dict[str, Any]] = []
        for primary_index, item in enumerate(primary_items):
            class_id = int(item["id"])
            confidence = float(item.get("confidence", 0.0))
            if (
                class_id not in PRIMARY_CAN_IDS
                or confidence < self.config.primary_override_conf
            ):
                continue
            box = _bbox(item)
            has_same_class_anchor = any(
                int(anchor["id"]) == class_id
                and same_physical_box(
                    box,
                    anchor["bbox"],
                    min_iou=self.config.match_iou,
                    min_ios=self.config.match_ios,
                )
                for anchor in anchors
            )
            conflicting = [
                (index, anchor)
                for index, anchor in enumerate(anchors)
                if int(anchor["id"]) != class_id
                and same_physical_box(
                    box,
                    anchor["bbox"],
                    min_iou=self.config.primary_override_iou,
                    min_ios=self.config.primary_override_ios,
                )
            ]
            if has_same_class_anchor or not conflicting:
                continue
            strongest_conflict = max(
                float(anchor.get("confidence", 0.0))
                for _, anchor in conflicting
            )
            if confidence < strongest_conflict + self.config.primary_override_margin:
                continue
            arbitration_primary_indexes.add(primary_index)
            overridden_anchor_indexes.update(index for index, _ in conflicting)
            primary_arbitrations.append(
                {
                    "primary": item,
                    "overriddenSecondary": [anchor for _, anchor in conflicting],
                    "confidenceMargin": round(confidence - strongest_conflict, 4),
                }
            )

        active_anchors = [
            anchor
            for index, anchor in enumerate(anchors)
            if index not in overridden_anchor_indexes
        ]
        trusted_anchors = [
            item
            for item in active_anchors
            if float(item["confidence"]) >= self.config.strong_conf
            or item.get("source") == "primary_color_fallback"
        ]
        suppression_anchors = list(trusted_anchors)

        output: list[dict[str, Any]] = []
        removed_primary_cans: list[dict[str, Any]] = []
        removed_secondary_anchors: list[dict[str, Any]] = []
        suppressed_confusers: list[dict[str, Any]] = []
        retained_confusers: list[dict[str, Any]] = []
        primary_can_outputs: list[dict[str, Any]] = []

        for primary_index, item in enumerate(primary_items):
            class_id = int(item["id"])
            confidence = float(item.get("confidence", 0.0))
            box = _bbox(item)
            if class_id in PRIMARY_CAN_IDS:
                if primary_index in arbitration_primary_indexes:
                    item["source"] = "primary_can_arbitrated"
                    primary_can_outputs.append(item)
                    suppression_anchors.append(item)
                    continue
                # Outside the narrow arbitration above, a trusted secondary
                # instance remains authoritative on the same pixels.  A very
                # strong primary can box is retained only when no trusted
                # secondary can covers it.
                overlaps_anchor = any(
                    same_physical_box(
                        box,
                        anchor["bbox"],
                        min_iou=self.config.match_iou,
                        min_ios=self.config.match_ios,
                    )
                    for anchor in trusted_anchors
                )
                if (
                    confidence >= self.config.base_fallback_conf
                    and not overlaps_anchor
                ):
                    item["source"] = "primary_can_fallback"
                    primary_can_outputs.append(item)
                    suppression_anchors.append(item)
                else:
                    removed_primary_cans.append(item)
                continue
            if confidence >= final_conf:
                overlapping_anchor = any(
                    class_id
                    in self.config.confusers_by_can_id.get(
                        int(anchor["id"]), frozenset()
                    )
                    and same_physical_box(
                        box,
                        anchor["bbox"],
                        min_iou=self.config.suppress_iou,
                        min_ios=self.config.suppress_ios,
                    )
                    for anchor in suppression_anchors
                )
                if overlapping_anchor:
                    suppressed_confusers.append(item)
                    continue
                retained_confusers.append(item)
            if confidence >= final_conf:
                output.append(item)

        removed_secondary_anchors.extend(
            {
                **anchor,
                "reason": "primary_arbitration",
            }
            for index, anchor in enumerate(anchors)
            if index in overridden_anchor_indexes
        )
        output.extend(primary_can_outputs)
        output.extend(active_anchors)

        # NMS across can boxes.  Same-class duplicates use the normal overlap
        # threshold.  Cross-class duplicates require near-identical geometry,
        # so separate cans that merely touch are never merged.  Live tracks
        # outrank held ghosts, and a primary arbitration outranks a conflicting
        # secondary track.
        def can_rank(item: Mapping[str, Any]) -> tuple[int, int, int, float]:
            source_rank = 0 if item.get("source") == "primary_can_arbitrated" else 1
            return (
                int(item.get("trackMisses", 0)),
                source_rank,
                -int(item.get("trackHits", 0)),
                -float(item.get("confidence", 0.0)),
            )

        final_cans: list[dict[str, Any]] = []
        for item in sorted(
            (v for v in output if int(v["id"]) in PRIMARY_CAN_IDS),
            key=can_rank,
        ):
            if any(
                (
                    int(item["id"]) == int(kept["id"])
                    and same_physical_box(
                        item["bbox"],
                        kept["bbox"],
                        min_iou=0.60,
                        min_ios=0.82,
                    )
                )
                or (
                    int(item["id"]) != int(kept["id"])
                    and same_physical_box(
                        item["bbox"],
                        kept["bbox"],
                        min_iou=0.72,
                        min_ios=0.88,
                    )
                )
                for kept in final_cans
            ):
                continue
            final_cans.append(item)
        noncans = [v for v in output if int(v["id"]) not in PRIMARY_CAN_IDS]
        fused = noncans + final_cans
        fused.sort(key=lambda item: -float(item.get("confidence", 0.0)))

        debug = {
            "enabled": True,
            "mode": "dedicated_can_detector",
            "secondaryRaw": len(secondary_detections),
            "secondaryCurrent": len(secondary_current),
            "secondaryAnchors": len(anchors),
            "trustedAnchors": len(trusted_anchors),
            "secondaryAnchorsAfterArbitration": len(active_anchors),
            "effectiveSuppressionAnchors": len(suppression_anchors),
            "selectedCans": final_cans,
            "primaryCanArbitrations": primary_arbitrations,
            "secondaryEvidenceArbitrations": secondary_arbitrations,
            "primaryColorRescues": primary_color_rescues,
            "primaryColorAssessments": primary_color_assessments,
            "removedSecondaryAnchors": removed_secondary_anchors,
            "removedPrimaryCans": removed_primary_cans,
            "suppressedConfusers": suppressed_confusers,
            "retainedConfuserCount": len(retained_confusers),
            "rejectedSecondary": rejected,
            "trackerFrame": self.stabilizer.frame_index,
            "tracks": len(self.stabilizer.tracks),
        }
        return fused, debug
