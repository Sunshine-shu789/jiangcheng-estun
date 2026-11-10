"""Deterministic target filtering and quality ordering for Tasks C/D.

A four-corner rule is deliberately separate from detector confidence.  Small
corner detections are retained as diagnostic observations, but they must not
become grasp candidates or shared support-plane samples.  Large corner objects
remain valid candidates.  Candidate ordering always prefers the middle image
region before the four corner regions, then prefers objects with stronger
same-frame 3-D geometry before detector confidence and image-edge margin.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from .config import CornerSmallObjectFilterConfig
from .domain import Detection, Frame


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


def is_corner_small_object(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
) -> bool:
    """Return true only for a small bbox centred in an image corner.

    Malformed geometry is not silently excluded: returning ``False`` keeps the
    normal fail-closed depth/geometry checks in charge of rejecting malformed
    detections rather than allowing them to disappear from physical evidence.
    """

    if not config.enabled or not is_corner_region_object(
        bbox,
        frame_width,
        frame_height,
        config,
    ):
        return False
    return not _exceeds_corner_small_limits(
        bbox,
        frame_width,
        frame_height,
        config,
    )


def is_corner_region_object(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
) -> bool:
    """Return true when the bbox centre lies in one of the four corners.

    This is a region-only test and deliberately ignores bbox size.  It is used
    both by the small-object filter and by candidate ordering, so large corner
    objects remain graspable but are always considered after middle objects.
    """

    if len(bbox) != 4:
        return False
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(isfinite(value) for value in (x0, y0, x1, y1)):
        return False
    if x1 <= x0 or y1 <= y0:
        return False
    width = _finite_positive_dimension(frame_width)
    height = _finite_positive_dimension(frame_height)
    if width is None or height is None:
        return False

    center_x_ratio = ((x0 + x1) * 0.5) / float(width)
    center_y_ratio = ((y0 + y1) * 0.5) / float(height)
    zone = float(config.corner_zone_ratio)
    if not isfinite(zone) or not 0.0 < zone < 0.5:
        return False
    in_horizontal_corner = center_x_ratio <= zone or center_x_ratio >= 1.0 - zone
    in_vertical_corner = center_y_ratio <= zone or center_y_ratio >= 1.0 - zone
    return in_horizontal_corner and in_vertical_corner


def _exceeds_corner_small_limits(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
) -> bool:
    """Return true when either bbox dimension is too large to be clutter."""

    x0, y0, x1, y1 = (float(value) for value in bbox)
    width = _finite_positive_dimension(frame_width)
    height = _finite_positive_dimension(frame_height)
    if width is None or height is None:
        return True
    box_width_ratio = (x1 - x0) / float(width)
    box_height_ratio = (y1 - y0) / float(height)
    return (
        box_width_ratio > float(config.max_box_width_ratio)
        or box_height_ratio > float(config.max_box_height_ratio)
    )


def candidate_region_priority(
    bbox: tuple[float, float, float, float] | list[float],
    frame_width: int,
    frame_height: int,
    config: CornerSmallObjectFilterConfig,
) -> int:
    """Return an absolute priority bucket: middle=1, corner=0."""

    return (
        0
        if is_corner_region_object(
            bbox,
            frame_width,
            frame_height,
            config,
        )
        else 1
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
