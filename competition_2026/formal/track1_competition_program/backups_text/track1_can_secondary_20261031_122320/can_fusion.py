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


SECONDARY_TO_PRIMARY = {0: 1, 1: 4, 2: 9}
PRIMARY_CAN_IDS = frozenset(SECONDARY_TO_PRIMARY.values())
DEFAULT_SECONDARY_NAMES = (
    "2-Starkist_Tuna",
    "5-Tomato_Soup",
    "10-meat_can",
)
# Classes repeatedly produced on the same pixels as one of the three cans in
# the self-collected validation set.  Suppression still requires geometric
# overlap with a trusted secondary prediction.
DEFAULT_CONFUSER_IDS = frozenset({11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 27, 28})


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
    confuser_ids: frozenset[int] = DEFAULT_CONFUSER_IDS

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CanFusionConfig":
        raw = dict(value or {})
        confusers = raw.get("confuser_ids", sorted(DEFAULT_CONFUSER_IDS))
        if not isinstance(confusers, Iterable) or isinstance(confusers, (str, bytes)):
            raise ValueError("can_secondary.confuser_ids must be a list")
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
            confuser_ids=frozenset(int(v) for v in confusers),
        )


@dataclass
class _Track:
    class_id: int
    bbox: list[float]
    confidence: float
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
                if index in used or track.class_id != class_id:
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
                        "confidence": round(track.confidence, 4),
                        "bbox": [round(v, 1) for v in track.bbox],
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

    def _secondary_candidates(
        self,
        secondary: Sequence[Mapping[str, Any]],
        frame_shape: Sequence[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

        # A single physical can must not be emitted under two classes.  This is
        # applied only to cross-class overlaps and therefore cannot remove
        # separate cans that merely touch each other.
        deduped: list[dict[str, Any]] = []
        cross_class_removed: list[dict[str, Any]] = []
        for candidate in sorted(
            candidates, key=lambda item: -float(item["confidence"])
        ):
            duplicate = False
            for kept in deduped:
                if int(kept["id"]) == int(candidate["id"]):
                    continue
                if same_physical_box(
                    candidate["bbox"],
                    kept["bbox"],
                    min_iou=0.70,
                    min_ios=0.84,
                ):
                    duplicate = True
                    break
            if duplicate:
                cross_class_removed.append(candidate)
            else:
                deduped.append(candidate)
        return deduped, rejected + cross_class_removed

    def fuse(
        self,
        frame_shape: Sequence[int],
        primary_detections: Sequence[Mapping[str, Any]],
        secondary_detections: Sequence[Mapping[str, Any]],
        *,
        final_conf: float,
        use_temporal: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        current, rejected = self._secondary_candidates(
            secondary_detections, frame_shape
        )
        stabilized = self.stabilizer.update(current) if use_temporal else list(current)
        anchors = [
            item for item in stabilized if float(item["confidence"]) >= final_conf
        ]
        trusted_anchors = [
            item
            for item in anchors
            if float(item["confidence"]) >= self.config.strong_conf
        ]
        anchor_ids = {int(item["id"]) for item in anchors}

        output: list[dict[str, Any]] = []
        removed_primary_cans: list[dict[str, Any]] = []
        suppressed_confusers: list[dict[str, Any]] = []
        retained_confusers: list[dict[str, Any]] = []
        fallback_cans: list[dict[str, Any]] = []

        for raw in primary_detections:
            item = dict(raw)
            class_id = int(item.get("id", -1))
            confidence = float(item.get("confidence", 0.0))
            box = _bbox(item)
            if class_id in PRIMARY_CAN_IDS:
                # Never let the primary model decide a can class.  A very
                # strong primary box is retained only as a fallback if the
                # secondary model has no trusted instance on those pixels.
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
                    and class_id not in anchor_ids
                    and not overlaps_anchor
                ):
                    item["source"] = "primary_can_fallback"
                    fallback_cans.append(item)
                else:
                    removed_primary_cans.append(item)
                continue
            if class_id in self.config.confuser_ids and confidence >= final_conf:
                overlapping_anchor = any(
                    same_physical_box(
                        box,
                        anchor["bbox"],
                        min_iou=self.config.suppress_iou,
                        min_ios=self.config.suppress_ios,
                    )
                    for anchor in trusted_anchors
                )
                if overlapping_anchor:
                    suppressed_confusers.append(item)
                    continue
                retained_confusers.append(item)
            if confidence >= final_conf:
                output.append(item)

        output.extend(fallback_cans)
        output.extend(anchors)

        # NMS across final can boxes from tracker/held tracks.
        final_cans: list[dict[str, Any]] = []
        for item in sorted(
            (v for v in output if int(v["id"]) in PRIMARY_CAN_IDS),
            key=lambda v: -float(v["confidence"]),
        ):
            if any(
                int(item["id"]) == int(kept["id"])
                and same_physical_box(
                    item["bbox"],
                    kept["bbox"],
                    min_iou=0.60,
                    min_ios=0.82,
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
            "secondaryCurrent": len(current),
            "secondaryAnchors": len(anchors),
            "trustedAnchors": len(trusted_anchors),
            "selectedCans": final_cans,
            "removedPrimaryCans": removed_primary_cans,
            "suppressedConfusers": suppressed_confusers,
            "retainedConfuserCount": len(retained_confusers),
            "rejectedSecondary": rejected,
            "trackerFrame": self.stabilizer.frame_index,
            "tracks": len(self.stabilizer.tracks),
        }
        return fused, debug
