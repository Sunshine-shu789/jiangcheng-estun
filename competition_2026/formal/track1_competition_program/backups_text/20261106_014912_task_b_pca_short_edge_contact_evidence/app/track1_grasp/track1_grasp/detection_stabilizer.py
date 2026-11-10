"""Temporal stabilisation for real-time object detections.

Raw YOLO output is deliberately single-frame: boxes jitter, classes can switch
for one or two frames, and confidence often falls just below the task threshold
even when the object is clearly visible.  This module keeps lightweight tracks
for the live camera path and applies the same practical rules used by the
earlier Task A preview:

* require several hits before a new track is visible;
* suppress duplicate and cross-class boxes around the same object;
* smooth boxes, depth, pixel anchor and confidence;
* delay class changes until a challenger wins consistently;
* keep a confirmed track for a few missed frames, marked as held.

The stabiliser is stateful but otherwise independent from RealSense and YOLO,
so its behaviour is covered by deterministic unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .config import VisionConfig
from .domain import Detection

if TYPE_CHECKING:
    from .grasp_geometry import GraspGeometryEvidence

Box = tuple[float, float, float, float]


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(first: Box, second: Box) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0.0:
        return 0.0
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def box_containment(first: Box, second: Box) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = min(box_area(first), box_area(second))
    return intersection / smaller if smaller > 0.0 else 0.0


def box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def boxes_are_duplicates(
    first_box: Box,
    first_label: str,
    first_confidence: float,
    second_box: Box,
    second_label: str,
    second_confidence: float,
    *,
    duplicate_iou: float,
    containment_ratio: float,
    cross_class_iou: float,
    cross_class_containment: float,
    cross_class_conf_ratio: float,
) -> bool:
    """Return whether two detector boxes describe the same physical instance."""

    overlap = box_iou(first_box, second_box)
    containment = box_containment(first_box, second_box)
    if first_label == second_label:
        return (
            overlap >= float(duplicate_iou)
            or containment >= float(containment_ratio)
        )
    if not (
        overlap >= float(cross_class_iou)
        or containment >= float(cross_class_containment)
    ):
        return False
    stronger = max(float(first_confidence), float(second_confidence))
    weaker = min(float(first_confidence), float(second_confidence))
    if weaker <= 0.0:
        return True
    return stronger >= weaker * max(1.0, float(cross_class_conf_ratio))


def select_non_duplicate_box_indices(
    candidates: Sequence[tuple[Box, str, float]],
    *,
    duplicate_iou: float = 0.55,
    containment_ratio: float = 0.90,
    cross_class_iou: float = 0.40,
    cross_class_containment: float = 0.60,
    cross_class_conf_ratio: float = 1.50,
) -> tuple[int, ...]:
    """Select one box per physical instance, preserving input order.

    The ranking and overlap rules are shared with :class:`DetectionStabilizer`.
    This helper deliberately leaves edge filtering and per-class retention to
    their owning callers; scene-plane sampling only needs duplicate suppression.
    """

    if not candidates:
        return ()
    ordered_indices = sorted(
        range(len(candidates)),
        key=lambda index: float(candidates[index][2]),
        reverse=True,
    )
    kept: list[int] = []
    for index in ordered_indices:
        box, label, confidence = candidates[index]
        if any(
            boxes_are_duplicates(
                box,
                label,
                confidence,
                candidates[accepted][0],
                candidates[accepted][1],
                candidates[accepted][2],
                duplicate_iou=duplicate_iou,
                containment_ratio=containment_ratio,
                cross_class_iou=cross_class_iou,
                cross_class_containment=cross_class_containment,
                cross_class_conf_ratio=cross_class_conf_ratio,
            )
            for accepted in kept
        ):
            continue
        kept.append(index)
    return tuple(sorted(kept))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class DetectionTrack:
    """One temporally linked object observation."""

    track_id: int
    label: str
    bbox: Box
    confidence: float
    stable_confidence: float
    depth_m: float | None
    camera_xyz: tuple[float, float, float] | None
    object_center_camera_xyz: tuple[float, float, float] | None
    major_axis_camera: tuple[float, float, float] | None
    minor_axis_camera: tuple[float, float, float] | None
    major_axis_length_m: float | None
    minor_axis_length_m: float | None
    pose_confidence: float | None
    geometry_evidence: "GraspGeometryEvidence | None"
    graspability: float
    attributes: dict[str, object]
    hits: int = 1
    missed: int = 0
    class_votes: dict[str, float] = field(default_factory=dict)
    challenger_label: str | None = None
    challenger_hits: int = 0

class DetectionStabilizer:
    """Convert noisy per-frame detections into confirmed live tracks."""

    def __init__(self, config: VisionConfig) -> None:
        self.confirm_hits = max(1, int(config.stabilizer_confirm_hits))
        self.max_misses = max(0, int(config.stabilizer_max_misses))
        self.match_iou = _clamp01(config.stabilizer_match_iou)
        self.confidence_alpha = _clamp01(config.stabilizer_confidence_alpha)
        self.box_alpha = _clamp01(config.stabilizer_box_alpha)
        self.depth_alpha = _clamp01(config.stabilizer_depth_alpha)
        self.duplicate_iou = _clamp01(config.stabilizer_duplicate_iou)
        self.containment_ratio = _clamp01(config.stabilizer_containment_ratio)
        self.cross_class_iou = _clamp01(config.stabilizer_cross_class_iou)
        self.cross_class_containment = _clamp01(
            config.stabilizer_cross_class_containment
        )
        self.cross_class_conf_ratio = max(
            1.0, float(config.stabilizer_cross_class_conf_ratio)
        )
        self.max_per_class = int(config.stabilizer_max_per_class)
        self.instance_tracking_labels = frozenset(
            label.strip()
            for label in config.stabilizer_instance_tracking_labels
            if label.strip()
        )
        self.class_switch_hits = max(1, int(config.stabilizer_class_switch_hits))
        self.edge_margin_px = max(0, int(config.stabilizer_edge_margin_px))
        self.edge_suppress_confidence = _clamp01(
            config.stabilizer_edge_suppress_confidence
        )
        self.new_track_confidence = _clamp01(config.min_confidence)
        self.keep_track_confidence = max(
            _clamp01(config.stabilizer_inference_min_confidence),
            self.new_track_confidence * 0.60,
        )
        self.tracks: dict[int, DetectionTrack] = {}
        self.next_track_id = 1

    def update(
        self,
        detections: Sequence[Detection],
        frame_width: int,
        frame_height: int,
    ) -> tuple[Detection, ...]:
        """Update tracks with one frame and return stable visible detections."""

        candidates = self._suppress_duplicates(
            detections,
            max(1, int(frame_width)),
            max(1, int(frame_height)),
        )
        existing_track_ids = set(self.tracks)
        diagonal = max(float(np.hypot(frame_width, frame_height)), 1.0)

        pairs: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(candidates):
            for track_id, track in self.tracks.items():
                score = self._match_score(
                    detection.bbox,
                    detection,
                    track,
                    diagonal,
                )
                if score is not None:
                    pairs.append((score, detection_index, track_id))

        assigned_detections: set[int] = set()
        assigned_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(pairs, reverse=True):
            if (
                detection_index in assigned_detections
                or track_id in assigned_tracks
            ):
                continue
            assigned_detections.add(detection_index)
            assigned_tracks.add(track_id)
            self._update_track(
                self.tracks[track_id],
                candidates[detection_index],
            )

        for detection_index, detection in enumerate(candidates):
            if detection_index in assigned_detections:
                continue
            if detection.confidence < self.new_track_confidence:
                continue
            self._create_track(detection)

        for track_id in existing_track_ids:
            track = self.tracks.get(track_id)
            if track is None or track_id in assigned_tracks:
                continue
            track.missed += 1
            track.confidence *= 0.82
            track.stable_confidence *= 0.90
            # Held detections do not contain a fresh point-cloud measurement.
            # A rim or sidewall from an older frame is never safe to command.
            track.geometry_evidence = None
            if (
                track.hits < self.confirm_hits and track.missed >= 1
            ) or track.missed > self.max_misses:
                del self.tracks[track_id]

        visible = self._visible_tracks()
        return tuple(
            detection
            for detection in (
                self._to_detection(track) for track in visible
            )
        )

    def reset(self) -> None:
        """Discard all temporal state, for example after a camera reconnect."""

        self.tracks.clear()
        self.next_track_id = 1

    def _suppress_duplicates(
        self,
        detections: Sequence[Detection],
        frame_width: int,
        frame_height: int,
    ) -> list[Detection]:
        ordered = sorted(
            (
                detection
                for detection in detections
                if detection.confidence >= self.keep_track_confidence
            ),
            key=lambda detection: detection.confidence,
            reverse=True,
        )
        kept: list[Detection] = []
        class_counts: dict[str, int] = {}
        for candidate in ordered:
            if self._is_edge_false_positive(
                candidate.bbox,
                candidate.confidence,
                frame_width,
                frame_height,
            ):
                continue
            if any(
                self._boxes_are_duplicates(
                    candidate.bbox,
                    candidate.label,
                    candidate.confidence,
                    accepted.bbox,
                    accepted.label,
                    accepted.confidence,
                )
                for accepted in kept
            ):
                continue
            if (
                self.max_per_class > 0
                and candidate.label not in self.instance_tracking_labels
                and class_counts.get(candidate.label, 0) >= self.max_per_class
            ):
                continue
            kept.append(candidate)
            if candidate.label not in self.instance_tracking_labels:
                class_counts[candidate.label] = (
                    class_counts.get(candidate.label, 0) + 1
                )
        return kept

    def _boxes_are_duplicates(
        self,
        first_box: Box,
        first_label: str,
        first_confidence: float,
        second_box: Box,
        second_label: str,
        second_confidence: float,
    ) -> bool:
        return boxes_are_duplicates(
            first_box,
            first_label,
            first_confidence,
            second_box,
            second_label,
            second_confidence,
            duplicate_iou=self.duplicate_iou,
            containment_ratio=self.containment_ratio,
            cross_class_iou=self.cross_class_iou,
            cross_class_containment=self.cross_class_containment,
            cross_class_conf_ratio=self.cross_class_conf_ratio,
        )

    def _is_edge_false_positive(
        self,
        bbox: Box,
        confidence: float,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        if confidence >= self.edge_suppress_confidence:
            return False
        margin = float(self.edge_margin_px)
        edge_count = sum(
            (
                bbox[0] <= margin,
                bbox[1] <= margin,
                bbox[2] >= frame_width - 1 - margin,
                bbox[3] >= frame_height - 1 - margin,
            )
        )
        return edge_count >= 2

    def _match_score(
        self,
        bbox: Box,
        detection: Detection,
        track: DetectionTrack,
        diagonal: float,
    ) -> float | None:
        overlap = box_iou(bbox, track.bbox)
        containment = box_containment(bbox, track.bbox)
        if detection.label == track.label:
            if overlap >= self.match_iou:
                return 1.0 + overlap
            current_center = box_center(bbox)
            track_center = box_center(track.bbox)
            center_distance = float(
                np.hypot(
                    current_center[0] - track_center[0],
                    current_center[1] - track_center[1],
                )
            )
            area_ratio = max(box_area(bbox), 1.0) / max(box_area(track.bbox), 1.0)
            if (
                center_distance / diagonal <= 0.08
                and 0.35 <= area_ratio <= 2.85
            ):
                return 0.5 + max(0.0, 1.0 - center_distance / diagonal)
            return None
        if overlap >= max(0.45, self.match_iou) or containment >= 0.75:
            return 0.25 + max(overlap, containment)
        return None

    def _update_track(self, track: DetectionTrack, detection: Detection) -> None:
        raw_confidence = float(detection.confidence)
        if detection.label == track.label:
            track.challenger_label = None
            track.challenger_hits = 0
        elif detection.label == track.challenger_label:
            track.challenger_hits += 1
        else:
            track.challenger_label = detection.label
            track.challenger_hits = 1

        for label in list(track.class_votes):
            track.class_votes[label] *= 0.65
            if track.class_votes[label] < 0.01:
                del track.class_votes[label]
        track.class_votes[detection.label] = (
            track.class_votes.get(detection.label, 0.0) + raw_confidence
        )
        best_label, best_score = max(
            track.class_votes.items(), key=lambda item: item[1]
        )
        current_score = track.class_votes.get(track.label, 0.0)
        if (
            best_label != track.label
            and best_label == track.challenger_label
            and track.challenger_hits >= self.class_switch_hits
            and best_score >= current_score * 1.35
            and raw_confidence >= self.new_track_confidence
        ):
            track.label = best_label
            track.class_votes = {best_label: best_score}
            track.stable_confidence = raw_confidence
            # A class switch can represent a different physical object.  Never
            # blend the previous class's PCA pose into the new observation.
            track.object_center_camera_xyz = None
            track.major_axis_camera = None
            track.minor_axis_camera = None
            track.major_axis_length_m = None
            track.minor_axis_length_m = None
            track.pose_confidence = None
            track.geometry_evidence = None
            track.challenger_label = None
            track.challenger_hits = 0

        previous_center = box_center(track.bbox)
        current_center = box_center(detection.bbox)
        center_distance = float(
            np.hypot(
                current_center[0] - previous_center[0],
                current_center[1] - previous_center[1],
            )
        )
        box_diagonal = max(
            float(
                np.hypot(
                    detection.bbox[2] - detection.bbox[0],
                    detection.bbox[3] - detection.bbox[1],
                )
            ),
            1.0,
        )
        alpha = 0.85 if center_distance >= 0.10 * box_diagonal else self.box_alpha
        track.bbox = tuple(
            old * (1.0 - alpha) + new * alpha
            for old, new in zip(track.bbox, detection.bbox)
        )
        track.confidence = (
            track.confidence * (1.0 - self.confidence_alpha)
            + raw_confidence * self.confidence_alpha
        )
        track.stable_confidence = (
            track.stable_confidence * (1.0 - self.confidence_alpha)
            + raw_confidence * self.confidence_alpha
        )

        if detection.depth_m is not None:
            track.depth_m = self._smooth_optional(
                track.depth_m,
                float(detection.depth_m),
                self.depth_alpha,
            )
        if detection.camera_xyz is not None:
            track.camera_xyz = tuple(
                self._smooth_optional(
                    None if track.camera_xyz is None else track.camera_xyz[index],
                    float(value),
                    self.depth_alpha,
                )
                for index, value in enumerate(detection.camera_xyz)
            )
        if detection.object_center_camera_xyz is not None:
            track.object_center_camera_xyz = self._smooth_vector(
                track.object_center_camera_xyz,
                detection.object_center_camera_xyz,
                self.depth_alpha,
            )
        if detection.major_axis_camera is not None:
            track.major_axis_camera = self._smooth_axis(
                track.major_axis_camera,
                detection.major_axis_camera,
                self.depth_alpha,
            )
        if detection.minor_axis_camera is not None:
            track.minor_axis_camera = self._smooth_axis(
                track.minor_axis_camera,
                detection.minor_axis_camera,
                self.depth_alpha,
            )
        if detection.major_axis_length_m is not None:
            track.major_axis_length_m = self._smooth_optional(
                track.major_axis_length_m,
                float(detection.major_axis_length_m),
                self.depth_alpha,
            )
        if detection.minor_axis_length_m is not None:
            track.minor_axis_length_m = self._smooth_optional(
                track.minor_axis_length_m,
                float(detection.minor_axis_length_m),
                self.depth_alpha,
            )
        if detection.pose_confidence is not None:
            track.pose_confidence = self._smooth_optional(
                track.pose_confidence,
                float(detection.pose_confidence),
                self.confidence_alpha,
            )
        # Geometry evidence is a full-frame measurement, not a temporal scalar.
        # Replace it atomically and clear it when the current frame lacks it.
        # Mixing a new footprint with an old rim would fabricate a mode.
        track.geometry_evidence = detection.geometry_evidence

        attributes = dict(detection.attributes)
        # Keep the newest raw YOLO score separately from the smoothed value.
        # The preview uses this to show whether a bad display is caused by the
        # model or by temporal smoothing.
        attributes["raw_confidence"] = _clamp01(raw_confidence)
        for key in ("u", "v"):
            if key in detection.attributes:
                old_value = track.attributes.get(key)
                attributes[key] = self._smooth_optional(
                    None if old_value is None else float(old_value),
                    float(detection.attributes[key]),
                    self.box_alpha,
                )
        track.attributes = attributes
        track.graspability = float(detection.graspability)
        track.hits += 1
        track.missed = 0

    @staticmethod
    def _smooth_optional(
        previous: float | None,
        current: float,
        alpha: float,
    ) -> float:
        if previous is None:
            return current
        return float(previous) * (1.0 - alpha) + current * alpha

    @classmethod
    def _smooth_vector(
        cls,
        previous: tuple[float, float, float] | None,
        current: tuple[float, float, float],
        alpha: float,
    ) -> tuple[float, float, float]:
        if previous is None:
            return tuple(float(value) for value in current)
        smoothed = tuple(
            cls._smooth_optional(float(old), float(new), alpha)
            for old, new in zip(previous, current)
        )
        return smoothed  # type: ignore[return-value]

    @classmethod
    def _smooth_axis(
        cls,
        previous: tuple[float, float, float] | None,
        current: tuple[float, float, float],
        alpha: float,
    ) -> tuple[float, float, float]:
        """Smooth a direction while resolving the eigenvector sign ambiguity."""

        current_vector = np.asarray(current, dtype=float)
        current_norm = float(np.linalg.norm(current_vector))
        if not np.isfinite(current_norm) or current_norm <= 1e-12:
            if previous is not None:
                return previous
            raise ValueError("adaptive axis must be finite and non-zero")
        current_vector = current_vector / current_norm
        if previous is None:
            return tuple(float(value) for value in current_vector)

        previous_vector = np.asarray(previous, dtype=float)
        previous_norm = float(np.linalg.norm(previous_vector))
        if not np.isfinite(previous_norm) or previous_norm <= 1e-12:
            return tuple(float(value) for value in current_vector)
        previous_vector = previous_vector / previous_norm
        if float(np.dot(previous_vector, current_vector)) < 0.0:
            current_vector = -current_vector
        blended = (
            (1.0 - alpha) * previous_vector + alpha * current_vector
        )
        blended_norm = float(np.linalg.norm(blended))
        if not np.isfinite(blended_norm) or blended_norm <= 1e-12:
            # Exactly opposite axes can only occur with an invalid estimate;
            # retain the deterministic new direction rather than emitting a
            # zero vector downstream.
            return tuple(float(value) for value in current_vector)
        blended /= blended_norm
        return tuple(float(value) for value in blended)

    def _create_track(self, detection: Detection) -> None:
        track = DetectionTrack(
            track_id=self.next_track_id,
            label=detection.label,
            bbox=tuple(float(value) for value in detection.bbox),
            confidence=float(detection.confidence),
            stable_confidence=float(detection.confidence),
            depth_m=None if detection.depth_m is None else float(detection.depth_m),
            camera_xyz=(
                None
                if detection.camera_xyz is None
                else tuple(float(value) for value in detection.camera_xyz)
            ),
            object_center_camera_xyz=(
                None
                if detection.object_center_camera_xyz is None
                else tuple(
                    float(value)
                    for value in detection.object_center_camera_xyz
                )
            ),
            major_axis_camera=(
                None
                if detection.major_axis_camera is None
                else tuple(float(value) for value in detection.major_axis_camera)
            ),
            minor_axis_camera=(
                None
                if detection.minor_axis_camera is None
                else tuple(float(value) for value in detection.minor_axis_camera)
            ),
            major_axis_length_m=(
                None
                if detection.major_axis_length_m is None
                else float(detection.major_axis_length_m)
            ),
            minor_axis_length_m=(
                None
                if detection.minor_axis_length_m is None
                else float(detection.minor_axis_length_m)
            ),
            pose_confidence=(
                None
                if detection.pose_confidence is None
                else float(detection.pose_confidence)
            ),
            geometry_evidence=detection.geometry_evidence,
            graspability=float(detection.graspability),
            attributes={
                **dict(detection.attributes),
                "raw_confidence": _clamp01(detection.confidence),
            },
            class_votes={detection.label: float(detection.confidence)},
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1

    def _visible_tracks(self) -> list[DetectionTrack]:
        visible = [
            track
            for track in self.tracks.values()
            if (
                track.hits >= self.confirm_hits
                and track.missed <= self.max_misses
                and track.confidence >= self.new_track_confidence
            )
        ]
        visible.sort(key=self._track_rank, reverse=True)
        deduplicated: list[DetectionTrack] = []
        for track in visible:
            if any(
                self._boxes_are_duplicates(
                    track.bbox,
                    track.label,
                    track.confidence,
                    accepted.bbox,
                    accepted.label,
                    accepted.confidence,
                )
                for accepted in deduplicated
            ):
                continue
            deduplicated.append(track)

        if self.max_per_class <= 0:
            return deduplicated
        best_by_class: dict[str, DetectionTrack] = {}
        instance_tracks: list[DetectionTrack] = []
        for track in deduplicated:
            if track.label in self.instance_tracking_labels:
                instance_tracks.append(track)
                continue
            current = best_by_class.get(track.label)
            if current is None or self._track_rank(track) > self._track_rank(current):
                best_by_class[track.label] = track
        return sorted(
            (*best_by_class.values(), *instance_tracks),
            key=self._track_rank,
            reverse=True,
        )

    @staticmethod
    def _track_rank(track: DetectionTrack) -> tuple[float, int, float]:
        return (
            track.confidence
            + min(track.hits, 30) * 0.002
            - track.missed * 0.12,
            track.hits,
            track.confidence,
        )

    @staticmethod
    def _raw_confidence(track: DetectionTrack) -> float:
        value = track.attributes.get("raw_confidence")
        if value is None:
            return float(track.confidence)
        return float(value)

    def _to_detection(self, track: DetectionTrack) -> Detection:
        held = track.missed > 0
        attributes = dict(track.attributes)
        attributes.update(
            {
                "track_id": track.track_id,
                "hits": track.hits,
                "missed": track.missed,
                "held": held,
                "stable_confidence": _clamp01(track.stable_confidence),
                "raw_confidence": (
                    0.0 if held else _clamp01(self._raw_confidence(track))
                ),
            }
        )
        return Detection(
            label=track.label,
            confidence=_clamp01(track.stable_confidence),
            bbox=tuple(float(value) for value in track.bbox),
            depth_m=track.depth_m,
            camera_xyz=track.camera_xyz,
            object_center_camera_xyz=track.object_center_camera_xyz,
            major_axis_camera=track.major_axis_camera,
            minor_axis_camera=track.minor_axis_camera,
            major_axis_length_m=track.major_axis_length_m,
            minor_axis_length_m=track.minor_axis_length_m,
            pose_confidence=track.pose_confidence,
            geometry_evidence=track.geometry_evidence,
            graspability=track.graspability,
            instance_id=f"track-{track.track_id}",
            attributes=attributes,
        )
