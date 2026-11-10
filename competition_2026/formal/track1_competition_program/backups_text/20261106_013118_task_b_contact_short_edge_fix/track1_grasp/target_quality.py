"""Deterministic target filtering and quality ordering for Tasks C/D.

The live C/D path no longer invents a fixed rectangular middle/corner split.
When ``mode="adaptive"`` it derives a continuous centre-first score from the
targets visible in the same frame.  The target nearest the visual centre is the
anchor; every other target is ordered by its relative distance from that
anchor.  This keeps ordering stable when the bin or camera shifts.

Small peripheral clutter is identified from two independent signals: a broad
corner geometry check and a size comparison against clearly larger targets in
the same frame.  Large peripheral objects remain grasp candidates.  A singleton
target uses a deliberately loose image-scale fallback so a lone corner
strawberry can still terminate an otherwise empty box scan.

The legacy fixed-zone implementation remains available for compatibility when
``mode="legacy"`` is requested explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, sqrt
from typing import Any, Sequence

from .config import CornerSmallObjectFilterConfig
from .domain import Detection, Frame


@dataclass(frozen=True)
class _BBoxMetrics:
    x0: float
    y0: float
    x1: float
    y1: float
    center_x: float
    center_y: float
    width: float
    height: float
    max_dimension: float


def _finite_positive_dimension(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def frame_dimensions(
    frame: Frame,
    fallback_width: int,
    fallback_height: int,
) -> tuple[int, int]:
    """Return actual frame dimensions with safe configured fallbacks."""

    intrinsics = frame.camera_intrinsics
    if intrinsics is not None:
        width = _finite_positive_dimension(intrinsics.width)
        height = _finite_positive_dimension(intrinsics.height)
        if width is not None and height is not None:
            return width, height

    image = frame.image
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        height = _finite_positive_dimension(shape[0])
        width = _finite_positive_dimension(shape[1])
        if width is not None and height is not None:
            return width, height

    width = _finite_positive_dimension(fallback_width)
    height = _finite_positive_dimension(fallback_height)
    if width is None or height is None:
        raise ValueError("fallback frame dimensions must be positive integers")
    return width, height


def _bbox_metrics(
    bbox: tuple[float, float, float, float] | list[float] | Sequence[float],
) -> _BBoxMetrics | None:
    if len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    width = x1 - x0
    height = y1 - y0
    return _BBoxMetrics(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        center_x=(x0 + x1) * 0.5,
        center_y=(y0 + y1) * 0.5,
        width=width,
        height=height,
        max_dimension=max(width, height),
    )


def _box_values(
    bboxes: Sequence[Sequence[float]] | None,
) -> list[_BBoxMetrics]:
    return [
        metrics
        for bbox in (bboxes or ())
        if (metrics := _bbox_metrics(bbox)) is not None
    ]


def _same_box(left: _BBoxMetrics, right: _BBoxMetrics) -> bool:
    return (
        abs(left.x0 - right.x0) <= 1e-6
        and abs(left.y0 - right.y0) <= 1e-6
        and abs(left.x1 - right.x1) <= 1e-6
        and abs(left.y1 - right.y1) <= 1e-6
    )


def _frame_corner_metrics(metrics: _BBoxMetrics, frame_width: int, frame_height: int) -> tuple[float, float]:
    """Return radial outwardness and corner balance without a fixed box.

    ``radial`` is zero at the image centre and approximately one in a physical
    image corner.  ``balance`` rejects targets near only one image edge, which
    would have a misleadingly large radial value but are not in a four-corner
    region.  Both values are continuous and are combined with relative target
    distribution ordering below.
    """

    half_width = max(1.0, frame_width * 0.5)
    half_height = max(1.0, frame_height * 0.5)
    normalised_x = min(1.0, abs(metrics.center_x - half_width) / half_width)
    normalised_y = min(1.0, abs(metrics.center_y - half_height) / half_height)
    radial = hypot(normalised_x, normalised_y) / sqrt(2.0)
    smaller = min(normalised_x, normalised_y)
    larger = max(normalised_x, normalised_y)
    balance = smaller / larger if larger > 1e-9 else 0.0
    return radial, balance


def _adaptive_corner_region(
    metrics: _BBoxMetrics,
    frame_width: int,
    frame_height: int,
) -> bool:
    radial, balance = _frame_corner_metrics(metrics, frame_width, frame_height)
    # These are broad geometric guards, not a tuneable 30% rectangle: a target
    # must lie on the outer radial half of the image and be meaningfully close
    # to both adjacent edges.  Ordering itself remains continuous below.
    return radial >= 0.45 and balance >= 0.35


def _adaptive_center_score(
    metrics: _BBoxMetrics,
    scene: list[_BBoxMetrics],
    frame_width: int,
    frame_height: int,
) -> float:
    """Return 1 at the visual centre anchor and 0 at the outermost target."""

    if len(scene) <= 1:
        return 1.0
    anchor = min(
        scene,
        key=lambda item: _frame_corner_metrics(
            item,
            frame_width,
            frame_height,
        )[0],
    )
    distances = [
        hypot(
            (item.center_x - anchor.center_x) / max(1.0, float(frame_width)),
            (item.center_y - anchor.center_y) / max(1.0, float(frame_height)),
        )
        for item in scene
    ]
    maximum = max(distances)
    if maximum <= 1e-12:
        return 1.0
    current = min(
        distances,
        key=lambda distance: (
            abs(
                distance
                - hypot(
                    (metrics.center_x - anchor.center_x)
                    / max(1.0, float(frame_width)),
                    (metrics.center_y - anchor.center_y)
                    / max(1.0, float(frame_height)),
                )
            )
        ),
    )
    return max(0.0, min(1.0, 1.0 - current / maximum))


def _adaptive_small_object(
    metrics: _BBoxMetrics,
    scene: list[_BBoxMetrics],
    frame_width: int,
    frame_height: int,
) -> bool:
    """Use same-frame relative size first, with a loose image-scale fallback."""

    others = list(scene)
    for index, item in enumerate(others):
        if _same_box(item, metrics):
            del others[index]
            break
    if others:
        largest_peer_dimension = max(item.max_dimension for item in others)
        if metrics.max_dimension <= 0.60 * largest_peer_dimension:
            return True
    # A lone corner item has no peer distribution.  This fallback is intentionally
    # much looser than the old 0.12/0.14 limits and only catches clearly small
    # image objects (less than one quarter of the shorter image dimension).
    return metrics.max_dimension <= 0.20 * min(frame_width, frame_height)


def _legacy_corner_region_object(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
) -> bool:
    metrics = _bbox_metrics(bbox)
    if metrics is None:
        return False
    zone = float(config.corner_zone_ratio)
    if not isfinite(zone) or not 0.0 < zone < 0.5:
        return False
    center_x_ratio = metrics.center_x / float(frame_width)
    center_y_ratio = metrics.center_y / float(frame_height)
    in_horizontal_corner = center_x_ratio <= zone or center_x_ratio >= 1.0 - zone
    in_vertical_corner = center_y_ratio <= zone or center_y_ratio >= 1.0 - zone
    return in_horizontal_corner and in_vertical_corner


def _legacy_exceeds_corner_small_limits(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
) -> bool:
    metrics = _bbox_metrics(bbox)
    if metrics is None:
        return True
    box_width_ratio = metrics.width / float(frame_width)
    box_height_ratio = metrics.height / float(frame_height)
    return (
        box_width_ratio > float(config.max_box_width_ratio)
        or box_height_ratio > float(config.max_box_height_ratio)
    )


def is_corner_small_object(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
    scene_bboxes: Sequence[Sequence[float]] | None = None,
) -> bool:
    """Return true only for small peripheral clutter in the current frame.

    ``scene_bboxes`` should contain every usable detection from the same frame,
    including ``bbox`` itself.  Relative ordering and relative size then adapt to
    the actual bin/camera placement.  Omitting it remains valid for diagnostics
    and uses the adaptive singleton fallback.
    """

    metrics = _bbox_metrics(bbox)
    if not config.enabled or metrics is None:
        return False
    if config.mode == "adaptive":
        if not _adaptive_corner_region(metrics, frame_width, frame_height):
            return False
        scene = _box_values(scene_bboxes)
        if not any(_same_box(item, metrics) for item in scene):
            scene.append(metrics)
        return _adaptive_small_object(
            metrics,
            scene,
            frame_width,
            frame_height,
        )

    return (
        _legacy_corner_region_object(
            bbox,
            frame_width,
            frame_height,
            config,
        )
        and not _legacy_exceeds_corner_small_limits(
            bbox,
            frame_width,
            frame_height,
            config,
        )
    )


def is_corner_region_object(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
) -> bool:
    """Return whether a valid bbox is in the configured broad corner geometry."""

    metrics = _bbox_metrics(bbox)
    if metrics is None:
        return False
    if config.mode == "adaptive":
        return _adaptive_corner_region(metrics, frame_width, frame_height)
    return _legacy_corner_region_object(
        bbox,
        frame_width,
        frame_height,
        config,
    )


def candidate_region_priority(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
    scene_bboxes: Sequence[Sequence[float]] | None = None,
) -> float:
    """Return a continuous centre-first score in ``[0, 1]``.

    In adaptive mode a target closest to the frame centre is the anchor and the
    remaining targets are ranked by relative distance from it.  A singleton is
    assigned 1.0 because there is no within-frame ordering decision to make.
    Legacy mode keeps the old middle=1/corner=0 buckets.
    """

    metrics = _bbox_metrics(bbox)
    if not config.enabled or metrics is None:
        return 1.0
    if config.mode == "legacy":
        return (
            0.0
            if _legacy_corner_region_object(
                bbox,
                frame_width,
                frame_height,
                config,
            )
            else 1.0
        )
    scene = _box_values(scene_bboxes)
    if not any(_same_box(item, metrics) for item in scene):
        scene.append(metrics)
    return _adaptive_center_score(
        metrics,
        scene,
        frame_width,
        frame_height,
    )


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _geometry_scores(detection: Detection) -> tuple[float, float, float]:
    evidence = detection.geometry_evidence
    if evidence is None:
        return 0.0, 0.0, 0.0
    support_plane_valid = 1.0 if evidence.support_plane_valid else 0.0
    support_plane_quality = _clamp01(evidence.support_plane_quality)
    surface_confidences = [_clamp01(evidence.short_edge_confidence)]
    if evidence.rim is not None:
        surface_confidences.append(_clamp01(evidence.rim.confidence))
    if evidence.sidewall is not None:
        surface_confidences.append(_clamp01(evidence.sidewall.confidence))
    return support_plane_valid, support_plane_quality, max(surface_confidences)


def _edge_margin_ratio(
    detection: Detection,
    frame_width: int,
    frame_height: int,
) -> float:
    x0, y0, x1, y1 = (float(value) for value in detection.bbox)
    if not all(isfinite(value) for value in (x0, y0, x1, y1)):
        return 0.0
    margin_px = min(x0, y0, frame_width - x1, frame_height - y1)
    return _clamp01(margin_px / max(1.0, float(min(frame_width, frame_height))))


def detection_quality_key(
    detection: Detection,
    frame_width: int,
    frame_height: int,
) -> tuple[float, ...]:
    """Return a lexicographic key where larger values are better.

    ``graspability`` remains the primary signal.  For the live V11 path it is
    commonly 1.0 for every physical candidate, so same-frame 3-D support-plane
    and grasp-mode evidence deliberately rank ahead of raw detector confidence.
    Image-edge margin prefers a well-centred usable object when geometry and
    confidence are otherwise tied.
    """

    geometry_valid, support_quality, geometry_confidence = _geometry_scores(
        detection
    )
    return (
        _clamp01(detection.graspability),
        geometry_valid,
        support_quality,
        geometry_confidence,
        _clamp01(detection.pose_confidence),
        _clamp01(detection.confidence),
        _edge_margin_ratio(detection, frame_width, frame_height),
    )
