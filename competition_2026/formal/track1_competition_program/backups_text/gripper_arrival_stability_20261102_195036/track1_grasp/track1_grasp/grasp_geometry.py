"""Geometry-only grasp-mode estimation from an object point cloud.

This module deliberately has no ROS, detector-label, camera-SDK or robot-SDK
dependency.  It turns one already segmented RGB-D point cloud plus an optional
support plane into the evidence needed to choose among three physical grasp
modes:

``vertical_short_edge``
    Top-down grasp across the short edge of an elongated footprint.
``vertical_rim``
    Top-down grasp at the measured rim of an open, roughly circular container.
``horizontal_sidewall``
    Side approach at the measured wall of a tall, roughly circular object.

Labels are never accepted as geometric evidence.  A caller may use an object
label only as a weak ordering hint after a mode has independently passed every
geometric threshold.  This keeps an incorrect detector class from creating a
rim grasp out of a closed box or a side grasp out of a flat plate.

All coordinates are metres in the camera optical frame.  The support-plane
normal is oriented away from the camera (towards the table), matching the
adaptive point-cloud estimator.  ``up_camera`` is its opposite direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, isfinite, pi, sin, sqrt
from typing import Any, Iterable, Sequence

import numpy as np

_EPS = 1e-12
_ANGULAR_BINS = 24


@dataclass(frozen=True)
class CircleFit:
    """Robust two-dimensional circle fit in a support-plane tangent frame."""

    center_xy_m: tuple[float, float]
    radius_m: float
    rms_m: float
    circularity: float
    coverage: float
    inlier_ratio: float
    valid_points: int


@dataclass(frozen=True)
class RimGeometry:
    """Visible top-rim evidence for an open container."""

    center_camera_xyz: tuple[float, float, float]
    center_xy_m: tuple[float, float]
    radius_m: float
    depression_m: float
    circularity: float
    coverage: float
    fit_error_m: float
    inlier_ratio: float
    valid_points: int
    confidence: float


@dataclass(frozen=True)
class SidewallGeometry:
    """Visible side-wall evidence for a tall object."""

    center_camera_xyz: tuple[float, float, float]
    center_xy_m: tuple[float, float]
    radius_m: float
    height_m: float
    outward_direction_camera: tuple[float, float, float]
    circularity: float
    coverage: float
    fit_error_m: float
    inlier_ratio: float
    valid_points: int
    confidence: float


@dataclass(frozen=True)
class GraspGeometryEvidence:
    """Geometry evidence retained with one stabilised detection."""

    object_center_camera_xyz: tuple[float, float, float]
    support_plane_valid: bool
    support_plane_quality: float
    support_plane_center_camera_xyz: tuple[float, float, float] | None
    support_plane_normal_camera: tuple[float, float, float] | None
    up_camera: tuple[float, float, float] | None
    visible_height_m: float | None
    footprint_center_camera_xyz: tuple[float, float, float] | None
    footprint_major_length_m: float | None
    footprint_minor_length_m: float | None
    footprint_axis_ratio: float | None
    footprint_major_axis_camera: tuple[float, float, float] | None
    footprint_minor_axis_camera: tuple[float, float, float] | None
    top_center_camera_xyz: tuple[float, float, float] | None
    top_height_m: float | None
    short_edge_confidence: float
    rim: RimGeometry | None
    sidewall: SidewallGeometry | None
    valid_points: int
    method_version: str = "geometry-policy-v1"


@dataclass(frozen=True)
class GraspGeometryCandidate:
    """One physically grounded grasp-mode proposal."""

    mode: str
    target_camera_xyz: tuple[float, float, float]
    tool_axis_camera: tuple[float, float, float]
    closure_axis_camera: tuple[float, float, float]
    confidence: float
    evidence: str


@dataclass(frozen=True)
class GraspGeometryDecision:
    """Ordered candidates plus a fail-closed rejection reason."""

    candidates: tuple[GraspGeometryCandidate, ...]
    reason: str = ""


def _unit(vector: Sequence[float], *, name: str = "vector") -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(value))
    if norm <= _EPS:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Choose the world axis least aligned with the normal.  Cross products then
    # form a deterministic right-handed basis without a fragile Euler choice.
    axes = np.eye(3, dtype=float)
    seed = axes[int(np.argmin(np.abs(normal)))]
    first = np.cross(normal, seed)
    first /= max(float(np.linalg.norm(first)), _EPS)
    second = np.cross(normal, first)
    second /= max(float(np.linalg.norm(second)), _EPS)
    return first, second


def _clamp01(value: float) -> float:
    if not isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _robust_axis_length(projected: np.ndarray, trim_percentile: float = 2.0) -> float:
    if projected.size == 0:
        return 0.0
    low, high = np.percentile(
        projected,
        [float(trim_percentile), 100.0 - float(trim_percentile)],
    )
    return max(0.0, float(high - low))


def _fit_circle_2d(xy_m: np.ndarray) -> CircleFit | None:
    points = np.asarray(xy_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 12:
        return None

    origin = np.median(points, axis=0)
    centered = points - origin
    design = np.column_stack(
        (
            2.0 * centered[:, 0],
            2.0 * centered[:, 1],
            np.ones(len(centered), dtype=float),
        )
    )
    rhs = np.sum(centered * centered, axis=1)
    try:
        solution, _residuals, rank, _singular = np.linalg.lstsq(
            design,
            rhs,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None
    if rank < 3 or not np.all(np.isfinite(solution)):
        return None
    offset_x, offset_y, constant = (float(value) for value in solution)
    radius_squared = float(offset_x * offset_x + offset_y * offset_y + constant)
    if not isfinite(radius_squared) or radius_squared <= _EPS:
        return None
    center = origin + np.asarray((offset_x, offset_y), dtype=float)
    radius = sqrt(radius_squared)

    for _ in range(3):
        distances = np.linalg.norm(points - center, axis=1)
        residual = distances - radius
        tolerance = max(0.0015, 0.20 * radius)
        keep = np.abs(residual) <= tolerance
        if int(np.count_nonzero(keep)) < 12 or np.all(keep):
            break
        points = points[keep]
        # Re-fit the trimmed points after moving the origin to their median.
        origin = np.median(points, axis=0)
        centered = points - origin
        design = np.column_stack(
            (
                2.0 * centered[:, 0],
                2.0 * centered[:, 1],
                np.ones(len(centered), dtype=float),
            )
        )
        rhs = np.sum(centered * centered, axis=1)
        try:
            solution, _residuals, rank, _singular = np.linalg.lstsq(
                design,
                rhs,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            return None
        if rank < 3 or not np.all(np.isfinite(solution)):
            return None
        offset_x, offset_y, constant = (float(value) for value in solution)
        radius_squared = float(
            offset_x * offset_x + offset_y * offset_y + constant
        )
        if not isfinite(radius_squared) or radius_squared <= _EPS:
            return None
        center = origin + np.asarray((offset_x, offset_y), dtype=float)
        radius = sqrt(radius_squared)

    distances = np.linalg.norm(points - center, axis=1)
    residual = distances - radius
    tolerance = max(0.0015, 0.20 * radius)
    inliers = np.abs(residual) <= tolerance
    if int(np.count_nonzero(inliers)) < 8:
        return None
    fit_points = points[inliers]
    fit_residual = residual[inliers]
    rms = float(np.sqrt(np.mean(fit_residual * fit_residual)))
    if not isfinite(rms):
        return None

    relative = fit_points - center
    covariance = np.cov(relative, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(np.max(eigenvalues))
    smallest = float(np.maximum(0.0, np.min(eigenvalues)))
    circularity = _clamp01(sqrt(smallest / largest)) if largest > _EPS else 0.0

    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * pi)
    bins = np.minimum(
        _ANGULAR_BINS - 1,
        np.floor(angles * (_ANGULAR_BINS / (2.0 * pi))).astype(int),
    )
    occupied = np.bincount(bins, minlength=_ANGULAR_BINS) > 0
    coverage = float(np.count_nonzero(occupied)) / float(_ANGULAR_BINS)
    inlier_ratio = float(np.count_nonzero(inliers)) / max(1.0, float(len(xy_m)))

    return CircleFit(
        center_xy_m=(float(center[0]), float(center[1])),
        radius_m=float(radius),
        rms_m=rms,
        circularity=float(circularity),
        coverage=float(coverage),
        inlier_ratio=float(inlier_ratio),
        valid_points=int(np.count_nonzero(inliers)),
    )


def _point_from_tangent_height(
    support_center: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    up: np.ndarray,
    xy_m: Sequence[float],
    height_m: float,
) -> tuple[float, float, float]:
    u, v = (float(xy_m[0]), float(xy_m[1]))
    support_u = float(support_center @ first)
    support_v = float(support_center @ second)
    xyz = (
        support_center
        + (u - support_u) * first
        + (v - support_v) * second
        + float(height_m) * up
    )
    return tuple(float(value) for value in xyz)


def _directional_coverage(
    xy_m: np.ndarray,
    center_xy_m: Sequence[float],
    outward_xy: np.ndarray,
) -> float:
    if len(xy_m) == 0:
        return 0.0
    delta = np.asarray(xy_m, dtype=float) - np.asarray(center_xy_m, dtype=float)
    norms = np.linalg.norm(delta, axis=1)
    keep = norms > 1e-6
    if not np.any(keep):
        return 0.0
    directions = delta[keep] / norms[keep, None]
    # Fraction of visible wall directions within +/- 45 degrees of the chosen
    # approach azimuth.  This is stronger evidence than mere angular occupancy:
    # two disconnected depth islands cannot look like a graspable local wall.
    dots = np.clip(directions @ outward_xy, -1.0, 1.0)
    return float(np.count_nonzero(dots >= cos(pi / 4.0))) / max(
        1.0, float(len(xy_m))
    )


def _select_outward_direction(
    xy_m: np.ndarray,
    center_xy_m: Sequence[float],
) -> tuple[np.ndarray | None, float]:
    if len(xy_m) == 0:
        return None, 0.0
    delta = np.asarray(xy_m, dtype=float) - np.asarray(center_xy_m, dtype=float)
    norms = np.linalg.norm(delta, axis=1)
    keep = norms > 1e-6
    if not np.any(keep):
        return None, 0.0
    directions = delta[keep] / norms[keep, None]
    angles = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * pi)
    bins = np.minimum(
        _ANGULAR_BINS - 1,
        np.floor(angles * (_ANGULAR_BINS / (2.0 * pi))).astype(int),
    )
    counts = np.bincount(bins, minlength=_ANGULAR_BINS)
    best_bin = int(np.argmax(counts))
    center_angle = (best_bin + 0.5) * (2.0 * pi / _ANGULAR_BINS)
    outward = np.asarray((cos(center_angle), sin(center_angle)), dtype=float)
    sector = np.abs(
        np.arctan2(
            np.sin(angles - center_angle),
            np.cos(angles - center_angle),
        )
    ) <= (pi / 6.0)
    if not np.any(sector):
        return outward, 0.0
    weighted = np.sum(directions[sector], axis=0)
    norm = float(np.linalg.norm(weighted))
    if norm <= _EPS:
        return outward, 0.0
    outward = weighted / norm
    coverage = _directional_coverage(xy_m, center_xy_m, outward)
    return outward, coverage


def analyze_grasp_geometry(
    points_camera_xyz: np.ndarray,
    *,
    object_center_camera_xyz: Sequence[float],
    support_plane_center_camera_xyz: Sequence[float] | None = None,
    support_plane_normal_camera: Sequence[float] | None = None,
    support_plane_rms_m: float | None = None,
    support_plane_inlier_ratio: float = 0.0,
    support_plane_max_rms_m: float = 0.0035,
    support_plane_min_normal_z: float = 0.80,
    top_band_m: float = 0.010,
    rim_inner_radius_fraction: float = 0.45,
    rim_outer_radius_fraction: float = 0.55,
    sidewall_height_band: tuple[float, float] = (0.20, 0.80),
) -> GraspGeometryEvidence:
    """Measure mode-independent geometry from one object point cloud.

    Insufficient support-plane or object evidence is represented by ``None``
    fields; it is never replaced with a label-specific prior.
    """

    points = np.asarray(points_camera_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("object point cloud must have shape (N, 3)")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        raise ValueError("object point cloud has fewer than three finite points")
    center = np.asarray(object_center_camera_xyz, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("object center must be a finite 3-vector")
    center = center.copy()

    low_fraction, high_fraction = (
        float(sidewall_height_band[0]),
        float(sidewall_height_band[1]),
    )
    if not 0.0 <= low_fraction < high_fraction <= 1.0:
        raise ValueError(
            "sidewall_height_band must be an increasing fraction inside [0, 1]"
        )

    centroid = np.median(points, axis=0)
    tangent_covariance = np.cov(points - centroid, rowvar=False)
    if not np.all(np.isfinite(tangent_covariance)):
        raise ValueError("object point cloud covariance is non-finite")
    if float(np.max(np.linalg.eigvalsh(tangent_covariance))) <= _EPS:
        raise ValueError("object point cloud is spatially degenerate")

    support_valid = (
        support_plane_center_camera_xyz is not None
        and support_plane_normal_camera is not None
        and support_plane_rms_m is not None
    )
    support_center: np.ndarray | None = None
    normal: np.ndarray | None = None
    up: np.ndarray | None = None
    first: np.ndarray | None = None
    second: np.ndarray | None = None
    support_quality = 0.0
    if support_valid:
        support_center = np.asarray(
            support_plane_center_camera_xyz,
            dtype=float,
        )
        normal = _unit(support_plane_normal_camera, name="support normal")
        if support_center.shape != (3,) or not np.all(np.isfinite(support_center)):
            raise ValueError("support-plane center must be a finite 3-vector")
        # Confirm that the normal points from the object towards the support
        # surface.  Flip only on a strong signed-depth disagreement.
        signed = float((support_center - center) @ normal)
        if signed < 0.0:
            normal = -normal
            signed = -signed
        if signed <= _EPS:
            support_valid = False
        else:
            up = -normal
            first, second = _tangent_basis(normal)
            rms = float(support_plane_rms_m)
            if not isfinite(rms) or rms < 0.0:
                support_valid = False
            else:
                max_rms = max(float(support_plane_max_rms_m), 1e-6)
                rms_score = _clamp01(1.0 - rms / max_rms)
                inlier_score = _clamp01(float(support_plane_inlier_ratio))
                normal_score = _clamp01(
                    abs(float(normal[2])) / max(float(support_plane_min_normal_z), 1e-6)
                )
                support_quality = rms_score * inlier_score * normal_score

    footprint_center_camera: tuple[float, float, float] | None = None
    footprint_major: float | None = None
    footprint_minor: float | None = None
    footprint_ratio: float | None = None
    footprint_major_axis_camera: tuple[float, float, float] | None = None
    footprint_minor_axis_camera: tuple[float, float, float] | None = None
    top_center_camera: tuple[float, float, float] | None = None
    visible_height: float | None = None
    top_height: float | None = None
    short_edge_confidence = 0.0
    rim_geometry: RimGeometry | None = None
    sidewall_geometry: SidewallGeometry | None = None

    if support_valid and support_center is not None and up is not None and first is not None and second is not None:
        tangent = np.column_stack((points @ first, points @ second))
        tangent_center = np.median(tangent, axis=0)
        centered_tangent = tangent - tangent_center
        covariance = np.cov(centered_tangent, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if float(eigenvalues[0]) > _EPS:
            major_vector = eigenvectors[:, 0]
            minor_vector = eigenvectors[:, 1]
            footprint_major = _robust_axis_length(centered_tangent @ major_vector)
            footprint_minor = _robust_axis_length(centered_tangent @ minor_vector)
            if (
                footprint_major is not None
                and footprint_minor is not None
                and footprint_major > _EPS
                and footprint_minor > _EPS
            ):
                footprint_ratio = float(footprint_major / footprint_minor)
                major_camera = (
                    float(major_vector[0]) * first
                    + float(major_vector[1]) * second
                )
                minor_camera = (
                    float(minor_vector[0]) * first
                    + float(minor_vector[1]) * second
                )
                major_camera /= max(float(np.linalg.norm(major_camera)), _EPS)
                minor_camera /= max(float(np.linalg.norm(minor_camera)), _EPS)
                footprint_major_axis_camera = tuple(
                    float(value) for value in major_camera
                )
                footprint_minor_axis_camera = tuple(
                    float(value) for value in minor_camera
                )
            else:
                footprint_major = None
                footprint_minor = None
        if footprint_major is not None and footprint_minor is not None:
            footprint_center_camera = _point_from_tangent_height(
                support_center,
                first,
                second,
                up,
                tangent_center,
                0.0,
            )
            axis_score = _clamp01((float(footprint_ratio) - 1.0) / 0.60)
            short_edge_confidence = _clamp01(
                0.70 * axis_score + 0.30 * support_quality
            )

        heights = (support_center - points) @ normal
        finite_heights = heights[np.isfinite(heights)]
        if len(finite_heights) >= 3:
            positive_heights = finite_heights[finite_heights >= -0.002]
            if len(positive_heights) >= 3:
                top_height = float(np.percentile(positive_heights, 97.5))
                visible_height = max(0.0, top_height)
                band = max(0.0015, min(float(top_band_m), 0.25 * visible_height + 0.0015))
                top_mask = heights >= max(0.0, top_height - band)
                top_points = points[top_mask]
                top_tangent = tangent[top_mask]
                if len(top_points) >= 6:
                    top_center_xy = np.median(top_tangent, axis=0)
                    top_center_camera = _point_from_tangent_height(
                        support_center,
                        first,
                        second,
                        up,
                        top_center_xy,
                        top_height,
                    )

                    # First identify the outer-top ring around the robust top
                    # centre, then fit a circle.  A closed cylinder contains a
                    # filled top disk and therefore fails the later depression
                    # gate even if the outer perimeter itself is circular.
                    radial = np.linalg.norm(top_tangent - top_center_xy, axis=1)
                    if len(radial) >= 12:
                        outer_radius = float(np.percentile(radial, 90.0))
                        if outer_radius > 0.004:
                            ring_mask = radial >= max(
                                0.001,
                                float(rim_outer_radius_fraction) * outer_radius,
                            )
                            ring_points = top_tangent[ring_mask]
                            circle = _fit_circle_2d(ring_points)
                            if circle is not None:
                                circle_center = np.asarray(
                                    circle.center_xy_m,
                                    dtype=float,
                                )
                                circle_radius = float(circle.radius_m)
                                all_radial = np.linalg.norm(
                                    tangent - circle_center,
                                    axis=1,
                                )
                                inner_mask = (
                                    all_radial
                                    <= max(
                                        0.002,
                                        float(rim_inner_radius_fraction)
                                        * circle_radius,
                                    )
                                ) & (heights < top_height - max(0.0015, band * 0.35))
                                rim_band = (
                                    all_radial
                                    >= max(
                                        0.002,
                                        float(rim_outer_radius_fraction)
                                        * circle_radius,
                                    )
                                ) & (heights >= top_height - band)
                                if int(np.count_nonzero(inner_mask)) >= 6 and int(
                                    np.count_nonzero(rim_band)
                                ) >= 8:
                                    interior_height = float(
                                        np.median(heights[inner_mask])
                                    )
                                    rim_height = float(np.median(heights[rim_band]))
                                    depression = max(0.0, rim_height - interior_height)
                                    fit_score = _clamp01(
                                        1.0 - circle.rms_m / max(0.0025, circle.radius_m * 0.15)
                                    )
                                    depression_score = _clamp01(
                                        depression / max(0.008, circle.radius_m * 0.25)
                                    )
                                    confidence = _clamp01(
                                        0.25 * circle.circularity
                                        + 0.20 * circle.coverage
                                        + 0.20 * fit_score
                                        + 0.20 * depression_score
                                        + 0.15 * support_quality
                                    )
                                    rim_center_xyz = _point_from_tangent_height(
                                        support_center,
                                        first,
                                        second,
                                        up,
                                        circle_center,
                                        rim_height,
                                    )
                                    rim_geometry = RimGeometry(
                                        center_camera_xyz=rim_center_xyz,
                                        center_xy_m=(
                                            float(circle_center[0]),
                                            float(circle_center[1]),
                                        ),
                                        radius_m=circle_radius,
                                        depression_m=float(depression),
                                        circularity=float(circle.circularity),
                                        coverage=float(circle.coverage),
                                        fit_error_m=float(circle.rms_m),
                                        inlier_ratio=float(circle.inlier_ratio),
                                        valid_points=int(circle.valid_points),
                                        confidence=float(confidence),
                                    )

                # The side-wall model uses mid-height points only and is
                # deliberately independent of the top-rim fit.  A tall closed
                # cylinder or an occluded opening must still be eligible for a
                # diagnostic side grasp when its visible wall is circular.
                if visible_height is not None and visible_height > 0.006:
                    mid_mask = heights >= low_fraction * visible_height
                    mid_mask &= heights <= high_fraction * visible_height
                    mid_tangent = tangent[mid_mask]
                    side_circle = _fit_circle_2d(mid_tangent)
                    if side_circle is not None:
                        side_center = np.asarray(
                            side_circle.center_xy_m,
                            dtype=float,
                        )
                        outward_xy, direction_coverage = _select_outward_direction(
                            mid_tangent,
                            side_center,
                        )
                        if outward_xy is not None:
                            outward_camera = (
                                float(outward_xy[0]) * first
                                + float(outward_xy[1]) * second
                            )
                            outward_camera /= max(
                                float(np.linalg.norm(outward_camera)),
                                _EPS,
                            )
                            fit_score = _clamp01(
                                1.0
                                - side_circle.rms_m
                                / max(0.0025, side_circle.radius_m * 0.15)
                            )
                            height_score = _clamp01(visible_height / 0.10)
                            side_confidence = _clamp01(
                                0.25 * side_circle.circularity
                                + 0.20 * side_circle.coverage
                                + 0.20 * direction_coverage
                                + 0.15 * fit_score
                                + 0.10 * height_score
                                + 0.10 * support_quality
                            )
                            side_center_xyz = _point_from_tangent_height(
                                support_center,
                                first,
                                second,
                                up,
                                side_center,
                                0.50 * visible_height,
                            )
                            sidewall_geometry = SidewallGeometry(
                                center_camera_xyz=side_center_xyz,
                                center_xy_m=(
                                    float(side_center[0]),
                                    float(side_center[1]),
                                ),
                                radius_m=float(side_circle.radius_m),
                                height_m=float(visible_height),
                                outward_direction_camera=tuple(
                                    float(value) for value in outward_camera
                                ),
                                circularity=float(side_circle.circularity),
                                coverage=float(side_circle.coverage),
                                fit_error_m=float(side_circle.rms_m),
                                inlier_ratio=float(side_circle.inlier_ratio),
                                valid_points=int(side_circle.valid_points),
                                confidence=float(side_confidence),
                            )

    return GraspGeometryEvidence(
        object_center_camera_xyz=tuple(float(value) for value in center),
        support_plane_valid=bool(support_valid),
        support_plane_quality=float(support_quality),
        support_plane_center_camera_xyz=(
            None
            if not support_valid or support_center is None
            else tuple(float(value) for value in support_center)
        ),
        support_plane_normal_camera=(
            None
            if not support_valid or normal is None
            else tuple(float(value) for value in normal)
        ),
        up_camera=(
            None if not support_valid or up is None else tuple(float(value) for value in up)
        ),
        visible_height_m=(
            None if visible_height is None else float(visible_height)
        ),
        footprint_center_camera_xyz=footprint_center_camera,
        footprint_major_length_m=(
            None if footprint_major is None else float(footprint_major)
        ),
        footprint_minor_length_m=(
            None if footprint_minor is None else float(footprint_minor)
        ),
        footprint_axis_ratio=(
            None if footprint_ratio is None else float(footprint_ratio)
        ),
        footprint_major_axis_camera=footprint_major_axis_camera,
        footprint_minor_axis_camera=footprint_minor_axis_camera,
        top_center_camera_xyz=top_center_camera,
        top_height_m=None if top_height is None else float(top_height),
        short_edge_confidence=float(short_edge_confidence),
        rim=rim_geometry,
        sidewall=sidewall_geometry,
        valid_points=int(len(points)),
    )


def _label_hint_bonus(
    mode: str,
    label: str,
    rim_label_hints: Iterable[str],
) -> float:
    """Return a strictly bounded ordering hint, never an acceptance gate."""

    hints = frozenset(str(value).strip() for value in rim_label_hints if str(value).strip())
    if mode == "vertical_rim" and str(label) in hints:
        return 0.03
    return 0.0


def select_grasp_geometry(
    evidence: GraspGeometryEvidence | None,
    policy: Any,
    *,
    label: str = "",
) -> GraspGeometryDecision:
    """Apply fail-closed thresholds and order physically valid candidates.

    ``policy`` is the validated configuration object.  It is intentionally
    duck-typed so this module remains independent from YAML/configuration code.
    """

    if evidence is None:
        return GraspGeometryDecision((), "geometry evidence is missing")
    if not evidence.support_plane_valid:
        return GraspGeometryDecision((), "a reliable support plane is required")
    if evidence.support_plane_quality < float(policy.min_support_plane_quality):
        return GraspGeometryDecision(
            (),
            "support-plane quality "
            f"{evidence.support_plane_quality:.3f} is below "
            f"{float(policy.min_support_plane_quality):.3f}",
        )
    if evidence.up_camera is None:
        return GraspGeometryDecision((), "support-plane up direction is missing")

    candidates: list[GraspGeometryCandidate] = []
    up = np.asarray(evidence.up_camera, dtype=float)
    up /= max(float(np.linalg.norm(up)), _EPS)

    footprint_major_axis: np.ndarray | None = None
    footprint_minor_axis: np.ndarray | None = None
    if (
        evidence.footprint_major_axis_camera is not None
        and evidence.footprint_minor_axis_camera is not None
    ):
        try:
            footprint_major_axis = _unit(
                evidence.footprint_major_axis_camera,
                name="footprint major axis",
            )
            footprint_minor_axis = _unit(
                evidence.footprint_minor_axis_camera,
                name="footprint minor axis",
            )
        except ValueError:
            footprint_major_axis = None
            footprint_minor_axis = None

    if (
        evidence.footprint_axis_ratio is not None
        and evidence.footprint_center_camera_xyz is not None
        and footprint_major_axis is not None
        and footprint_minor_axis is not None
        and evidence.footprint_axis_ratio + 1e-12
        >= float(policy.short_edge_min_axis_ratio)
        and evidence.short_edge_confidence + 1e-12
        >= float(policy.short_edge_min_confidence)
    ):
        candidates.append(
            GraspGeometryCandidate(
                mode="vertical_short_edge",
                target_camera_xyz=evidence.footprint_center_camera_xyz,
                tool_axis_camera=tuple(float(value) for value in -up),
                # Parallel jaws close along the narrow footprint direction,
                # crossing the two short edges.  The major axis is the object's
                # long direction and remains reserved for endpoint fallbacks.
                closure_axis_camera=tuple(float(value) for value in footprint_minor_axis),
                confidence=float(evidence.short_edge_confidence),
                evidence=(
                    f"axis_ratio={evidence.footprint_axis_ratio:.4f}; "
                    f"support_quality={evidence.support_plane_quality:.4f}; "
                    "closure_axis=pca_minor"
                ),
            )
        )

    rim = evidence.rim
    if (
        policy.allow_rim
        and rim is not None
        and float(policy.rim_min_radius_m)
        <= rim.radius_m
        <= float(policy.rim_max_radius_m)
        and rim.depression_m + 1e-12 >= float(policy.rim_min_depression_m)
        and rim.circularity + 1e-12 >= float(policy.rim_min_circularity)
        and rim.coverage + 1e-12 >= float(policy.rim_min_coverage)
        and rim.fit_error_m <= float(policy.rim_max_fit_error_m) + 1e-12
        and rim.confidence + 1e-12 >= float(policy.rim_min_confidence)
    ):
        # The top-down tool axis points at the support surface.  The closure
        # direction is arbitrary around a circular rim, so use the deterministic
        # camera-up tangent projection as a weak starting heading; downstream IK
        # and wrist-equivalent candidates still validate it.
        tangent_heading = up - float(up @ np.asarray((0.0, 0.0, 1.0))) * np.asarray(
            (0.0, 0.0, 1.0)
        )
        if float(np.linalg.norm(tangent_heading)) <= 1e-6:
            tangent_heading = np.asarray((1.0, 0.0, 0.0), dtype=float)
        tangent_heading /= max(float(np.linalg.norm(tangent_heading)), _EPS)
        rim_target = np.asarray(rim.center_camera_xyz, dtype=float)
        rim_target -= float(policy.rim_insertion_depth_m) * up
        candidates.append(
            GraspGeometryCandidate(
                mode="vertical_rim",
                target_camera_xyz=tuple(float(value) for value in rim_target),
                tool_axis_camera=tuple(float(value) for value in -up),
                closure_axis_camera=tuple(
                    float(value) for value in tangent_heading
                ),
                confidence=float(rim.confidence),
                evidence=(
                    f"radius={rim.radius_m:.4f}; depression={rim.depression_m:.4f}; "
                    f"circularity={rim.circularity:.3f}; coverage={rim.coverage:.3f}; "
                    f"fit_rms={rim.fit_error_m:.4f}"
                ),
            )
        )

    sidewall = evidence.sidewall
    if (
        policy.allow_horizontal_sidewall
        and sidewall is not None
        and float(policy.sidewall_min_height_m)
        <= sidewall.height_m
        and float(policy.sidewall_min_radius_m)
        <= sidewall.radius_m
        <= float(policy.sidewall_max_radius_m)
        and sidewall.circularity + 1e-12
        >= float(policy.sidewall_min_circularity)
        and sidewall.coverage + 1e-12 >= float(policy.sidewall_min_coverage)
        and sidewall.fit_error_m
        <= float(policy.sidewall_max_fit_error_m) + 1e-12
        and sidewall.confidence + 1e-12
        >= float(policy.sidewall_min_confidence)
    ):
        outward = np.asarray(sidewall.outward_direction_camera, dtype=float)
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm > _EPS:
            outward /= outward_norm
            # Approach from the observed free side towards the wall.  The
            # closure axis lies in the table plane and perpendicular to the
            # approach, so the fingers close across the radial diameter.
            tool_axis = -outward
            closure = np.cross(up, tool_axis)
            closure_norm = float(np.linalg.norm(closure))
            if closure_norm > _EPS:
                closure /= closure_norm
                # Re-orthogonalise against tool Z to remove any numerical drift.
                closure -= float(closure @ tool_axis) * tool_axis
                closure /= max(float(np.linalg.norm(closure)), _EPS)
                target = np.asarray(sidewall.center_camera_xyz, dtype=float)
                target -= float(policy.sidewall_axis_offset_m) * outward
                candidates.append(
                    GraspGeometryCandidate(
                        mode="horizontal_sidewall",
                        target_camera_xyz=tuple(float(value) for value in target),
                        tool_axis_camera=tuple(float(value) for value in tool_axis),
                        closure_axis_camera=tuple(float(value) for value in closure),
                        confidence=float(sidewall.confidence),
                        evidence=(
                            f"radius={sidewall.radius_m:.4f}; "
                            f"height={sidewall.height_m:.4f}; "
                            f"circularity={sidewall.circularity:.3f}; "
                            f"coverage={sidewall.coverage:.3f}; "
                            f"fit_rms={sidewall.fit_error_m:.4f}"
                        ),
                    )
                )

    if not candidates:
        reasons: list[str] = []
        if evidence.footprint_axis_ratio is None:
            reasons.append("no elongated footprint")
        elif evidence.footprint_axis_ratio < float(policy.short_edge_min_axis_ratio):
            reasons.append(
                f"axis ratio {evidence.footprint_axis_ratio:.3f} below "
                f"{float(policy.short_edge_min_axis_ratio):.3f}"
            )
        if evidence.rim is None:
            reasons.append("no measured open rim")
        if (
            evidence.sidewall is None
            and bool(policy.allow_horizontal_sidewall)
        ):
            reasons.append("no measured side wall")
        return GraspGeometryDecision(
            (),
            "no geometry mode passed all evidence gates"
            + (": " + "; ".join(reasons) if reasons else ""),
        )

    def rank(candidate: GraspGeometryCandidate) -> tuple[float, str]:
        # Confidence is primary.  The label bonus is only 0.03 and therefore
        # cannot promote a candidate with a materially weaker geometric fit.
        score = candidate.confidence + _label_hint_bonus(
            candidate.mode,
            label,
            policy.rim_label_hints,
        )
        return (-round(float(score), 12), candidate.mode)

    candidates.sort(key=rank)
    return GraspGeometryDecision(tuple(candidates))


def rotation_matrix_from_tool_axes(
    tool_axis_camera: Sequence[float],
    closure_axis_camera: Sequence[float],
) -> np.ndarray:
    """Build a right-handed camera-frame rotation from grasp axes.

    Column 2 is the approach/tool Z direction.  Column 1 is the jaw-closure
    direction, projected perpendicular to tool Z.  Column 0 completes the
    right-handed frame.  The caller still converts this camera-frame rotation
    into the robot base frame and validates it as an IK candidate.
    """

    tool_z = _unit(tool_axis_camera, name="tool axis")
    closure = np.asarray(closure_axis_camera, dtype=float)
    if closure.shape != (3,) or not np.all(np.isfinite(closure)):
        raise ValueError("closure axis must be a finite 3-vector")
    closure -= float(closure @ tool_z) * tool_z
    closure_norm = float(np.linalg.norm(closure))
    if closure_norm <= _EPS:
        raise ValueError("closure axis must not be parallel to tool axis")
    closure /= closure_norm
    tool_x = np.cross(closure, tool_z)
    tool_x_norm = float(np.linalg.norm(tool_x))
    if tool_x_norm <= _EPS:
        raise ValueError("grasp axes cannot form a right-handed frame")
    tool_x /= tool_x_norm
    # Recompute Y to guarantee exact orthonormality.
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= max(float(np.linalg.norm(tool_y)), _EPS)
    return np.column_stack((tool_x, tool_y, tool_z))
