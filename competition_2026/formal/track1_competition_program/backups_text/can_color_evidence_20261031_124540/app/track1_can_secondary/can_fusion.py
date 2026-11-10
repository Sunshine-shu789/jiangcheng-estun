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

        # A confident primary can box may arbitrate only when it is
        # substantially stronger than a *different-class* secondary anchor on
        # the same pixels.  This is deliberately narrow: it recovers a stable
        # Tuna/Tomato identity after a tracker flip without allowing the
        # primary model to redefine ordinary secondary-can output.
        arbitration_primary_indexes: set[int] = set()
        overridden_anchor_indexes: set[int] = set()
        arbitrations: list[dict[str, Any]] = []
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
            arbitrations.append(
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
            "secondaryCurrent": len(current),
            "secondaryAnchors": len(anchors),
            "trustedAnchors": len(trusted_anchors),
            "secondaryAnchorsAfterArbitration": len(active_anchors),
            "effectiveSuppressionAnchors": len(suppression_anchors),
            "selectedCans": final_cans,
            "primaryCanArbitrations": arbitrations,
            "removedSecondaryAnchors": removed_secondary_anchors,
            "removedPrimaryCans": removed_primary_cans,
            "suppressedConfusers": suppressed_confusers,
            "retainedConfuserCount": len(retained_confusers),
            "rejectedSecondary": rejected,
            "trackerFrame": self.stabilizer.frame_index,
            "tracks": len(self.stabilizer.tracks),
        }
        return fused, debug
