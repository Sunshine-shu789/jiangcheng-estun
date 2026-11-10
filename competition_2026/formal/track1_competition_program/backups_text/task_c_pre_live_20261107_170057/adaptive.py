"""RGB-D object-pose estimation for adaptive inverse grasp planning.

The detector supplies a bounding box and a depth image.  Adaptive grasping
needs a stable object centre and principal axes, so this module converts the
depth pixels inside the box into a camera-frame point cloud, removes obvious
background/outlier points, and uses a two-pass PCA to estimate the long and
short axes.  It deliberately has no ROS or robot dependency and is therefore
safe to exercise entirely offline.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import isfinite
from typing import Any

import cv2
import numpy as np

from .config import AdaptivePcaConfig
from .grasp_geometry import GraspGeometryEvidence, analyze_grasp_geometry


# A broken colour mask can leave a thin bridge through an otherwise thick
# banana silhouette.  The skeleton then follows that bridge and reports an
# implausibly narrow "waist".  Compare the local distance-transform width with
# the area-averaged silhouette width so this case fails closed instead of
# producing a plausible-looking but unsafe grasp point.
_WAIST_MIN_WIDTH_TO_MEAN_RATIO = 0.65


@dataclass(frozen=True)
class ObjectPoseEstimate:
    """A robust object pose estimate in the camera optical frame."""

    center_camera_xyz: tuple[float, float, float]
    major_axis_camera: tuple[float, float, float]
    minor_axis_camera: tuple[float, float, float]
    major_axis_length_m: float
    minor_axis_length_m: float
    pose_confidence: float
    valid_points: int
    depth_inlier_ratio: float
    #: Original point-cloud covariance ratio.  Unlike the support-projected
    #: centre, this value is never rewritten by a tabletop-plane fit and is
    #: therefore safe to retain as diagnostic evidence for downstream policy.
    axis_ratio: float = 1.0
    geometry_evidence: GraspGeometryEvidence | None = None
    support_plane_center_camera_xyz: tuple[float, float, float] | None = None
    support_plane_normal_camera: tuple[float, float, float] | None = None
    support_plane_rms_m: float | None = None
    support_plane_valid_points: int = 0
    support_plane_candidate_points: int = 0
    support_plane_inlier_ratio: float = 0.0
    #: ``waist_midline`` means ``center_camera_xyz`` is the arc-length midpoint
    #: of the target-colour centreline and ``minor_axis_camera`` is the local
    #: in-plane normal there.  ``global_pca`` is the legacy whole-object fit.
    axis_source: str = "global_pca"
    waist_midpoint_uv_px: tuple[float, float] | None = None
    waist_path_length_px: float | None = None
    waist_half_width_px: float | None = None


@dataclass(frozen=True)
class SupportPlaneEstimate:
    """Robust support-plane fit with the diagnostics needed for safe rejection."""

    center_camera_xyz: tuple[float, float, float]
    normal_camera: tuple[float, float, float]
    rms_m: float
    valid_points: int
    candidate_points: int
    inlier_ratio: float
    median_residual_m: float


@dataclass(frozen=True)
class _SupportSurfaceComponent:
    """One disconnected, locally flat patch in the support-plane candidate mask."""

    label: int
    points: np.ndarray
    centroid_uv: tuple[float, float]

    @property
    def pixel_count(self) -> int:
        return int(len(self.points))


@dataclass(frozen=True)
class _WaistMidlineEstimate:
    """Pixel-space banana waist geometry before camera-frame projection."""

    center_uv: tuple[float, float]
    tangent_uv: tuple[float, float]
    path_length_px: float
    half_width_px: float


def _as_depth_array(depth_m: Any) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    return depth


def _axis_span(points: np.ndarray, axis: np.ndarray, percentile: float) -> float:
    projection = points @ axis
    if percentile <= 0.0:
        low = float(np.min(projection))
        high = float(np.max(projection))
    else:
        low, high = np.percentile(
            projection, [float(percentile), 100.0 - float(percentile)]
        )
    return max(0.0, float(high) - float(low))


def _dominant_colour_component_mask(
    mask: np.ndarray,
    *,
    overlap_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Keep the target-colour component best supported by the depth mode.

    Colour segmentation intentionally has a broad banana-yellow range.  A
    detector box can therefore contain a separate same-coloured object at the
    same depth.  Picking one connected component by depth-mode overlap excludes
    that clutter, while retaining the complete depth range of the selected
    curved object instead of truncating it at the Otsu threshold.
    """

    colour = np.asarray(mask, dtype=bool)
    if colour.ndim != 2 or not np.any(colour):
        return None
    overlap = None
    if overlap_mask is not None:
        overlap = np.asarray(overlap_mask, dtype=bool)
        if overlap.shape != colour.shape:
            return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        colour.astype(np.uint8, copy=False),
        connectivity=8,
    )
    best_index = 0
    best_score: tuple[int, int] | None = None
    for component_index in range(1, count):
        component = labels == component_index
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        supported = (
            int(np.count_nonzero(component & overlap))
            if overlap is not None
            else area
        )
        score = (supported, area)
        if best_score is None or score > best_score:
            best_score = score
            best_index = component_index
    if best_index <= 0:
        return None
    return labels == best_index


def _largest_binary_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected component of a two-dimensional mask."""

    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        return np.zeros(binary.shape, dtype=bool)
    component_index = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )
    return labels == component_index


def _longest_skeleton_path(skeleton: np.ndarray) -> np.ndarray:
    """Return the longest geodesic skeleton polyline as ``(N, 2)`` x/y pixels."""

    skeleton = np.asarray(skeleton, dtype=bool)
    ys, xs = np.nonzero(skeleton)
    if len(xs) < 2:
        raise ValueError("waist midline skeleton is too small")

    pixels = tuple(zip(ys.tolist(), xs.tolist(), strict=True))
    pixel_to_index = {pixel: index for index, pixel in enumerate(pixels)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in pixels]
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for index, (v, u) in enumerate(pixels):
        for dv, du in offsets:
            neighbour = pixel_to_index.get((v + dv, u + du))
            if neighbour is None:
                continue
            adjacency[index].append(
                (neighbour, 1.4142135623730951 if dv and du else 1.0)
            )

    def shortest_paths(
        start: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.full(len(pixels), np.inf, dtype=float)
        parents = np.full(len(pixels), -1, dtype=int)
        distances[start] = 0.0
        queue: list[tuple[float, int]] = [(0.0, start)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for neighbour, step in adjacency[node]:
                candidate = distance + step
                if candidate >= distances[neighbour]:
                    continue
                distances[neighbour] = candidate
                parents[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
        return distances, parents

    # Two graph-diameter sweeps are exact for the tree-like Zhang-Suen
    # centreline and remain deterministic when small tip loops create cycles.
    first_distances, _ = shortest_paths(0)
    if not np.any(np.isfinite(first_distances)):
        raise ValueError("waist midline skeleton has no finite path")
    first_far = int(np.argmax(first_distances))
    distances, parents = shortest_paths(first_far)
    second_far = int(np.argmax(distances))
    if not isfinite(float(distances[second_far])):
        raise ValueError("waist midline skeleton path is not finite")

    path_indices: list[int] = []
    current = second_far
    while current >= 0:
        path_indices.append(current)
        if current == first_far:
            break
        current = int(parents[current])
    if not path_indices or path_indices[-1] != first_far:
        raise ValueError("waist midline skeleton path could not be reconstructed")
    path_indices.reverse()
    return np.asarray(
        [(float(xs[index]), float(ys[index])) for index in path_indices],
        dtype=float,
    )


def _smooth_skeleton_path(path: np.ndarray) -> np.ndarray:
    """Suppress one-pixel skeleton stair-steps without shortening the tips."""

    points = np.asarray(path, dtype=float)
    if len(points) < 5:
        return points
    padded = np.pad(points, ((2, 2), (0, 0)), mode="edge")
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), dtype=float) / 9.0
    return np.column_stack(
        (
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        )
    )


def _zhang_suen_thin(binary: np.ndarray) -> np.ndarray:
    """Thin a binary image without requiring OpenCV contrib.

    The live runtime ships a plain OpenCV build without ``ximgproc``.  This is
    the standard vectorised two-subiteration Zhang-Suen skeletonisation
    algorithm, used as a deterministic fallback to ``ximgproc.thinning``.
    """

    pixels = np.asarray(binary, dtype=bool).astype(np.uint8)
    if pixels.ndim != 2:
        raise ValueError("thinning mask must be two-dimensional")
    if not np.any(pixels):
        return pixels.astype(bool)

    while True:
        changed = False
        for step in (0, 1):
            padded = np.pad(pixels, 1, mode="constant")
            centre = padded[1:-1, 1:-1]
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = (p2, p3, p4, p5, p6, p7, p8, p9)

            neighbour_count = sum(
                neighbour.astype(np.uint16) for neighbour in neighbours
            )
            transitions = sum(
                ((neighbour == 0) & (neighbours[(index + 1) % 8] == 1))
                for index, neighbour in enumerate(neighbours)
            )
            if step == 0:
                topology = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                topology = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (
                (centre == 1)
                & (neighbour_count >= 2)
                & (neighbour_count <= 6)
                & (transitions == 1)
                & topology
            )
            if np.any(remove):
                pixels[remove] = 0
                changed = True
        if not changed:
            return pixels.astype(bool)


def _thin_binary(binary: np.ndarray) -> np.ndarray:
    """Return a one-pixel skeleton with OpenCV contrib when available."""

    pixels = np.asarray(binary, dtype=bool)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return (
            cv2.ximgproc.thinning(
                pixels.astype(np.uint8) * 255,
                thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
            )
            > 0
        )
    return _zhang_suen_thin(pixels)


def _estimate_waist_midline_pixel(
    mask: np.ndarray,
    config: AdaptivePcaConfig,
) -> _WaistMidlineEstimate:
    """Locate a curved object's centreline waist and local tangent.

    The long axis of a curved banana is not a grasp direction.  This routine
    follows the target-colour silhouette centreline instead, selects the point
    at half of its arc length, and measures the tangent from a short symmetric
    window.  Its image-plane normal is therefore the local short axis at the
    waist rather than the whole-object PCA minor axis.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("waist midline mask must be two-dimensional")
    if int(np.count_nonzero(binary)) < int(config.waist_midline_min_mask_points):
        raise ValueError(
            "target-colour mask has too few pixels for a waist midline"
        )
    cleaned = (
        cv2.morphologyEx(
            binary.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        > 0
    )
    cleaned = _largest_binary_component(cleaned)
    if int(np.count_nonzero(cleaned)) < int(config.waist_midline_min_mask_points):
        raise ValueError("waist midline component is too small")

    skeleton = _thin_binary(cleaned)
    skeleton = _largest_binary_component(skeleton)
    path = _smooth_skeleton_path(_longest_skeleton_path(skeleton))
    if len(path) < 7:
        raise ValueError("waist midline path has too few points")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=float), np.cumsum(segment_lengths))
    )
    path_length = float(cumulative[-1])
    if (
        not isfinite(path_length)
        or path_length < float(config.waist_midline_min_path_px)
    ):
        raise ValueError(
            "waist midline path is too short for a stable local axis"
        )

    halfway = 0.5 * path_length
    upper = int(np.searchsorted(cumulative, halfway, side="left"))
    upper = max(1, min(len(path) - 1, upper))
    lower = upper - 1
    segment_length = float(cumulative[upper] - cumulative[lower])
    if segment_length <= 1e-9:
        waist = path[lower]
    else:
        fraction = (halfway - float(cumulative[lower])) / segment_length
        waist = path[lower] + fraction * (path[upper] - path[lower])

    window = max(2.0, float(config.waist_midline_tangent_window_px))
    local = path[np.abs(cumulative - halfway) <= window]
    if len(local) < 5:
        distances = np.linalg.norm(path - waist, axis=1)
        nearest = np.argsort(distances)[: min(9, len(path))]
        local = path[np.sort(nearest)]
    if len(local) < 5:
        raise ValueError("waist local tangent has too few centreline points")

    local_centered = local - np.mean(local, axis=0)
    covariance = np.cov(local_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    tangent = np.asarray(eigenvectors[:, order[0]], dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if not isfinite(tangent_norm) or tangent_norm <= 1e-12:
        raise ValueError("waist local tangent is degenerate")
    tangent /= tangent_norm
    dominant = int(np.argmax(np.abs(tangent)))
    if tangent[dominant] < 0.0:
        tangent = -tangent

    local_span = float(np.ptp(local_centered @ tangent))
    linearity = float(eigenvalues[0] / max(float(eigenvalues[1]), 1e-12))
    if (
        not isfinite(local_span)
        or local_span < max(4.0, 0.75 * window)
        or not isfinite(linearity)
        or linearity < float(config.waist_midline_min_tangent_linearity)
    ):
        raise ValueError("waist local tangent is not stable enough")

    distance = cv2.distanceTransform(cleaned.astype(np.uint8), cv2.DIST_L2, 5)
    local_pixels = np.rint(local).astype(int)
    local_pixels[:, 0] = np.clip(
        local_pixels[:, 0], 0, cleaned.shape[1] - 1
    )
    local_pixels[:, 1] = np.clip(
        local_pixels[:, 1], 0, cleaned.shape[0] - 1
    )
    local_widths = distance[local_pixels[:, 1], local_pixels[:, 0]]
    half_width = float(np.median(local_widths))
    mean_half_width = float(np.count_nonzero(cleaned)) / (2.0 * path_length)
    if (
        not isfinite(half_width)
        or not isfinite(mean_half_width)
        or half_width < 2.0
    ):
        raise ValueError("waist centreline has no measurable local width")
    if half_width < _WAIST_MIN_WIDTH_TO_MEAN_RATIO * mean_half_width:
        raise ValueError(
            "waist centreline width is inconsistent with the banana silhouette"
        )

    return _WaistMidlineEstimate(
        center_uv=(float(waist[0]), float(waist[1])),
        tangent_uv=(float(tangent[0]), float(tangent[1])),
        path_length_px=path_length,
        half_width_px=half_width,
    )


def _waist_camera_axes(
    tangent_uv: tuple[float, float],
    intrinsics: Any,
    support_plane_normal_camera: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map an image-space waist tangent/normal onto the support plane."""

    tangent_pixel = np.asarray(tangent_uv, dtype=float)
    pixel_norm = float(np.linalg.norm(tangent_pixel))
    if not isfinite(pixel_norm) or pixel_norm <= 1e-12:
        raise ValueError("waist tangent must be finite and non-zero")
    tangent_pixel /= pixel_norm

    fx = float(getattr(intrinsics, "fx"))
    fy = float(getattr(intrinsics, "fy"))
    if support_plane_normal_camera is None:
        plane_normal = np.asarray((0.0, 0.0, 1.0), dtype=float)
    else:
        plane_normal = np.asarray(support_plane_normal_camera, dtype=float)
        plane_normal /= max(float(np.linalg.norm(plane_normal)), 1e-12)
    if not np.all(np.isfinite(plane_normal)):
        raise ValueError("waist support-plane normal is not finite")

    tangent_camera = np.asarray(
        (
            float(tangent_pixel[0]) / fx,
            float(tangent_pixel[1]) / fy,
            0.0,
        ),
        dtype=float,
    )
    tangent_camera -= plane_normal * float(plane_normal @ tangent_camera)
    tangent_norm = float(np.linalg.norm(tangent_camera))
    if not isfinite(tangent_norm) or tangent_norm <= 1e-12:
        raise ValueError("waist tangent has no component on the support plane")
    tangent_camera /= tangent_norm

    short_axis = np.cross(plane_normal, tangent_camera)
    short_norm = float(np.linalg.norm(short_axis))
    if not isfinite(short_norm) or short_norm <= 1e-12:
        raise ValueError("waist local short axis is degenerate")
    short_axis /= short_norm

    for axis in (tangent_camera, short_axis):
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            axis *= -1.0
    return tangent_camera, short_axis


def _otsu_depth_split(
    values: np.ndarray,
    min_cluster_points: int,
) -> tuple[float, float] | None:
    """Return an Otsu threshold and cluster separation for a depth patch.

    A tabletop object normally produces two depth modes: the nearer object
    surface and the farther support surface.  Otsu's threshold separates those
    modes without assuming that the object fills most of its detector box.
    Unimodal patches deliberately return ``None`` so the caller can use the
    median-band fallback.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) < 2 * min_cluster_points:
        return None
    low_percentile, high_percentile = np.percentile(values, [1.0, 99.0])
    span = float(high_percentile - low_percentile)
    if not isfinite(span) or span < 0.020:
        return None

    histogram, edges = np.histogram(
        np.clip(values, low_percentile, high_percentile),
        bins=256,
        range=(float(low_percentile), float(high_percentile)),
    )
    histogram = histogram.astype(float)
    centers = (edges[:-1] + edges[1:]) * 0.5
    total = float(histogram.sum())
    if total <= 0.0:
        return None

    probability = histogram / total
    omega = np.cumsum(probability)
    weighted_mean = np.cumsum(probability * centers)
    total_mean = float(weighted_mean[-1])
    denominator = omega * (1.0 - omega)
    between_class_variance = np.divide(
        (total_mean * omega - weighted_mean) ** 2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    # Keep both Otsu classes non-empty.  With two perfectly separated synthetic
    # modes, every empty bin between them has the same maximum between-class
    # variance and ``argmax`` would otherwise choose the first histogram bin,
    # leaving no samples to find the left mode.
    split_index = 1 + int(np.argmax(between_class_variance[1:-1]))
    threshold = float(edges[split_index + 1])
    lower = values[values <= threshold]
    upper = values[values > threshold]
    if len(lower) < min_cluster_points or len(upper) < min_cluster_points:
        return None
    separation = float(np.median(upper) - np.median(lower))
    if separation < 0.020:
        return None

    # Otsu always finds a split, even for a single sloped surface.  Require a
    # genuine histogram valley between two supported depth modes so a
    # continuous object surface is not mistaken for object plus tabletop.
    radius = 9
    sigma = 3.0
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= float(kernel.sum())
    smoothed = np.convolve(histogram, kernel, mode="same")
    left_peak = int(np.argmax(smoothed[:split_index]))
    right_peak = split_index + 1 + int(
        np.argmax(smoothed[split_index + 1 :])
    )
    valley = left_peak + int(
        np.argmin(smoothed[left_peak : right_peak + 1])
    )
    peak_floor = min(float(smoothed[left_peak]), float(smoothed[right_peak]))
    if peak_floor <= 0.0:
        return None
    if float(smoothed[valley]) / peak_floor > 0.55:
        return None
    mode_separation = float(centers[right_peak] - centers[left_peak])
    if mode_separation < 0.020:
        return None

    # With perfectly separated synthetic modes Otsu can place its threshold at
    # the first empty bin (an arbitrary tie).  The histogram valley is the
    # robust class boundary, so use it for the actual foreground split.
    threshold = float(centers[valley])
    lower = values[values <= threshold]
    upper = values[values > threshold]
    if len(lower) < min_cluster_points or len(upper) < min_cluster_points:
        return None
    separation = float(np.median(upper) - np.median(lower))
    if separation < 0.020:
        return None
    return threshold, separation


def _fit_support_plane(
    rays_u: np.ndarray,
    rays_v: np.ndarray,
    depth: np.ndarray,
) -> tuple[float, float, float] | None:
    """Fit an inverse-depth plane to support-plane samples.

    For a pinhole camera, every pixel ray is ``(ray_u, ray_v, 1)`` and a 3-D
    plane satisfies ``1 / depth = a*ray_u + b*ray_v + c``.  Fitting inverse
    depth is linear in these ray coordinates and avoids the bias introduced by
    fitting ordinary depth as a function of pixel coordinates.
    """

    samples = np.column_stack((rays_u, rays_v, np.ones_like(depth)))
    inverse_depth = 1.0 / depth
    keep = np.ones(len(depth), dtype=bool)
    coefficients = None
    for _ in range(4):
        if int(np.count_nonzero(keep)) < 3:
            return None
        coefficients, *_ = np.linalg.lstsq(
            samples[keep], inverse_depth[keep], rcond=None
        )
        residual = np.abs(samples @ coefficients - inverse_depth)
        median_residual = float(np.median(residual[keep]))
        # 1e-4 m^-1 is roughly a 7 mm depth residual at 0.27 m.  The floor
        # prevents sub-millimetre depth quantisation from rejecting the plane.
        limit = max(1e-4, 3.0 * median_residual)
        next_keep = residual <= limit
        if int(np.count_nonzero(next_keep)) < 3 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    if coefficients is None or not np.all(np.isfinite(coefficients)):
        return None
    return tuple(float(value) for value in coefficients)


def _fit_support_plane_robust(
    points: np.ndarray,
    config: AdaptivePcaConfig,
) -> SupportPlaneEstimate | None:
    """Fit the dominant near-horizontal plane and report its quality.

    A support fit cannot use ordinary least squares: a transparent box wall or
    the gripper can form a second, visually large plane.  RANSAC first finds the
    dominant thin layer, then the selected inliers are refined with SVD.  The
    fit is deterministic because a fixed-seed generator is used for the small
    candidate subsample.
    """

    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    candidate_points = int(len(points))
    if candidate_points < 3:
        return None

    max_samples = int(config.support_plane_max_samples)
    if candidate_points > max_samples:
        rng = np.random.default_rng(20261025)
        sample_indices = rng.choice(candidate_points, max_samples, replace=False)
        fit_points = points[sample_indices]
    else:
        fit_points = points

    threshold = float(config.support_plane_ransac_threshold_m)
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    rng = np.random.default_rng(20261025)
    for iteration in range(int(config.support_plane_ransac_iterations) + 1):
        if iteration == 0:
            # Seed with an all-point SVD fit.  When the support dominates this
            # usually finds it immediately; RANSAC then protects against walls.
            center = np.median(fit_points, axis=0)
            _, _, vh = np.linalg.svd(fit_points - center, full_matrices=False)
            normal = vh[-1]
        else:
            selected = fit_points[rng.choice(len(fit_points), 3, replace=False)]
            normal = np.cross(selected[1] - selected[0], selected[2] - selected[0])
            norm = float(np.linalg.norm(normal))
            if not isfinite(norm) or norm < 1e-10:
                continue
            normal = normal / norm
            center = np.mean(selected, axis=0)
        if not np.all(np.isfinite(normal)):
            continue
        residual = np.abs((fit_points - center) @ normal)
        mask = residual <= threshold
        count = int(np.count_nonzero(mask))
        if count < 3:
            continue
        score = (count, -float(np.median(residual[mask])))
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        return None

    keep = best_mask
    for _ in range(8):
        selected_points = fit_points[keep]
        if len(selected_points) < 3:
            return None
        center = np.median(selected_points, axis=0)
        _, _, vh = np.linalg.svd(
            selected_points - center,
            full_matrices=False,
        )
        normal = vh[-1]
        if normal[2] < 0.0:
            normal = -normal
        residual = np.abs((fit_points - center) @ normal)
        next_keep = residual <= threshold
        if int(np.count_nonzero(next_keep)) < 3 or np.array_equal(next_keep, keep):
            break
        keep = next_keep

    # Refine once on all candidate points, not only the bounded RANSAC sample.
    selected_points = fit_points[keep]
    center = np.median(selected_points, axis=0)
    _, _, vh = np.linalg.svd(selected_points - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    all_residual = np.abs((points - center) @ normal)
    all_keep = all_residual <= threshold
    if int(np.count_nonzero(all_keep)) >= 3:
        selected_points = points[all_keep]
        center = np.median(selected_points, axis=0)
        _, _, vh = np.linalg.svd(
            selected_points - center,
            full_matrices=False,
        )
        normal = vh[-1]
        if normal[2] < 0.0:
            normal = -normal
        all_residual = np.abs((points - center) @ normal)
        all_keep = all_residual <= threshold

    inliers = points[all_keep]
    if len(inliers) < 3:
        return None
    residual = (inliers - center) @ normal
    rms = float(np.sqrt(np.mean(residual * residual)))
    inlier_ratio = float(len(inliers)) / max(1.0, float(candidate_points))
    return SupportPlaneEstimate(
        center_camera_xyz=tuple(float(value) for value in center),
        normal_camera=tuple(float(value) for value in normal),
        rms_m=rms,
        valid_points=int(len(inliers)),
        candidate_points=candidate_points,
        inlier_ratio=inlier_ratio,
        median_residual_m=float(np.median(np.abs(residual))),
    )


def _support_plane_rejection_reason(
    estimate: SupportPlaneEstimate | None,
    config: AdaptivePcaConfig,
) -> str:
    if estimate is None:
        return "support plane could not be sampled"
    if estimate.valid_points < int(config.support_plane_min_points):
        return (
            "support plane has only "
            f"{estimate.valid_points} inliers; need "
            f"{int(config.support_plane_min_points)}"
        )
    if estimate.inlier_ratio < float(config.support_plane_min_inlier_ratio):
        return (
            "support plane inlier ratio "
            f"{estimate.inlier_ratio:.3f} is below "
            f"{float(config.support_plane_min_inlier_ratio):.3f}"
        )
    if estimate.rms_m > float(config.support_plane_max_rms_m):
        return (
            "support plane RMS "
            f"{estimate.rms_m * 1000.0:.3f} mm exceeds "
            f"{float(config.support_plane_max_rms_m) * 1000.0:.3f} mm"
        )
    if abs(estimate.normal_camera[2]) < float(
        config.support_plane_min_normal_z
    ):
        return (
            "support plane normal is too vertical in the image "
            f"(|nz|={abs(estimate.normal_camera[2]):.3f})"
        )
    return ""


def _masked_support_plane(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: Any,
    config: AdaptivePcaConfig,
) -> SupportPlaneEstimate | None:
    v, u = np.nonzero(mask)
    if len(v) < int(config.support_plane_min_points):
        return None
    z = depth[v, u].astype(float, copy=False)
    x = (u.astype(float) - float(intrinsics.cx)) / float(intrinsics.fx) * z
    y = (v.astype(float) - float(intrinsics.cy)) / float(intrinsics.fy) * z
    points = np.column_stack((x, y, z))
    return _fit_support_plane_robust(points, config)


def _support_ring_mask(
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    guard: np.ndarray,
    config: AdaptivePcaConfig,
) -> np.ndarray:
    """Return a local ring around the target, excluding a dilated guard mask."""

    height, width = depth.shape
    x0, y0, x1, y1 = bbox
    dilation = int(config.support_plane_guard_dilation_px)
    guard = np.asarray(guard, dtype=bool)
    if guard.shape != depth.shape:
        raise ValueError("support guard mask must match the depth image")
    if dilation > 0:
        kernel_size = 2 * dilation + 1
        guard = (
            cv2.dilate(
                guard.astype(np.uint8),
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
            )
            > 0
        )

    box_width = float(x1 - x0)
    box_height = float(y1 - y0)
    padding_ratio = float(config.support_plane_ring_padding_ratio)
    pad_x = max(35, int(round(padding_ratio * box_width)))
    pad_y = max(30, int(round(padding_ratio * box_height)))
    outer_x0 = max(0, x0 - pad_x)
    outer_y0 = max(0, y0 - pad_y)
    outer_x1 = min(width, x1 + pad_x)
    outer_y1 = min(height, y1 + pad_y)
    ring = np.zeros((height, width), dtype=bool)
    ring[outer_y0:outer_y1, outer_x0:outer_x1] = True
    inner = np.zeros((height, width), dtype=bool)
    inner[
        max(0, y0 - 10) : min(height, y1 + 10),
        max(0, x0 - 10) : min(width, x1 + 10),
    ] = True
    ring &= ~inner
    ring &= ~guard
    return ring


def _scene_support_regions(
    depth: np.ndarray,
    bboxes: tuple[tuple[float, float, float, float], ...],
    config: AdaptivePcaConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the detector union and its expanded scene ROI.

    Invalid or fully out-of-frame boxes are ignored.  Values are still
    validated here because the fallback candidate sampler is a safety-critical
    path and must fail closed on malformed detector output.
    """

    height, width = depth.shape
    detection_union = np.zeros((height, width), dtype=np.uint8)
    scene_roi = np.zeros((height, width), dtype=np.uint8)
    margin_x = int(config.support_plane_scene_margin_x_px)
    margin_y = int(config.support_plane_scene_margin_y_px)
    for raw_bbox in bboxes:
        if len(raw_bbox) != 4:
            raise ValueError("scene support bbox must contain four values")
        if not all(isfinite(float(value)) for value in raw_bbox):
            raise ValueError("scene support bbox values must be finite")
        x0 = max(0, min(width, int(round(float(raw_bbox[0])))))
        y0 = max(0, min(height, int(round(float(raw_bbox[1])))))
        x1 = max(0, min(width, int(round(float(raw_bbox[2])))))
        y1 = max(0, min(height, int(round(float(raw_bbox[3])))))
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(detection_union, (x0, y0), (x1, y1), 1, -1)
        cv2.rectangle(
            scene_roi,
            (max(0, x0 - margin_x), max(0, y0 - margin_y)),
            (min(width - 1, x1 + margin_x), min(height - 1, y1 + margin_y)),
            1,
            -1,
        )
    return detection_union > 0, scene_roi > 0


def _scene_local_surface_features(
    depth: np.ndarray,
    intrinsics: Any,
    config: AdaptivePcaConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return depth validity and a locally planar surface description."""

    height, width = depth.shape
    depth_valid = (
        np.isfinite(depth)
        & (depth >= float(config.min_depth_m))
        & (depth <= float(config.max_depth_m))
        & (depth > 0.0)
    )
    radius = int(config.support_plane_normal_radius_px)
    local_kernel_size = 2 * radius + 1
    local_kernel = np.ones(
        (local_kernel_size, local_kernel_size),
        dtype=np.uint8,
    )
    local_depth_valid = (
        cv2.erode(
            depth_valid.astype(np.uint8),
            local_kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )

    safe_depth = np.where(depth_valid, depth, 0.0)
    rows, cols = np.indices((height, width), dtype=float)
    points = np.dstack(
        (
            (cols - float(intrinsics.cx)) / float(intrinsics.fx) * safe_depth,
            (rows - float(intrinsics.cy)) / float(intrinsics.fy) * safe_depth,
            safe_depth,
        )
    )
    delta_u = np.zeros_like(points)
    delta_v = np.zeros_like(points)
    delta_u[:, radius:-radius, :] = (
        points[:, 2 * radius :, :] - points[:, : -2 * radius, :]
    )
    delta_v[radius:-radius, :, :] = (
        points[2 * radius :, :, :] - points[: -2 * radius, :, :]
    )
    normal = np.cross(delta_u, delta_v)
    normal_length = np.linalg.norm(normal, axis=2)
    normal_z = np.divide(
        normal[:, :, 2],
        normal_length,
        out=np.zeros_like(normal_length),
        where=normal_length > 1e-12,
    )

    sentinel_far = float(config.max_depth_m) + 1.0
    sentinel_near = -1.0
    local_min = cv2.erode(
        np.where(depth_valid, depth, sentinel_far).astype(np.float32),
        local_kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=sentinel_far,
    )
    local_max = cv2.dilate(
        np.where(depth_valid, depth, sentinel_near).astype(np.float32),
        local_kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=sentinel_near,
    )
    local_depth_span = local_max - local_min
    return depth_valid, local_depth_valid, normal_z, local_depth_span


def _scene_support_candidate_mask(
    depth: np.ndarray,
    bboxes: tuple[tuple[float, float, float, float], ...],
    intrinsics: Any,
    config: AdaptivePcaConfig,
) -> np.ndarray:
    """Return locally planar support samples from one complete RGB-D scene.

    Detector boxes are excluded conservatively because their interiors contain
    object, transparent-wall and (often) gripper points.  The surrounding box
    margins provide only narrow tabletop annuli.  A candidate must belong to a
    fully valid local-depth window and have a near-horizontal local surface
    normal; the subsequent RANSAC fit still applies every existing RMS,
    inlier-ratio and coverage gate.
    """

    height, width = depth.shape
    depth_valid, local_depth_valid, normal_z, local_depth_span = (
        _scene_local_surface_features(depth, intrinsics, config)
    )
    detection_union, scene_roi = _scene_support_regions(depth, bboxes, config)
    if not np.any(detection_union):
        return np.zeros((height, width), dtype=bool)

    guard_dilation = int(config.support_plane_guard_dilation_px)
    if guard_dilation > 0:
        guard_kernel_size = 2 * guard_dilation + 1
        detection_guard = (
            cv2.dilate(
                detection_union.astype(np.uint8),
                np.ones((guard_kernel_size, guard_kernel_size), dtype=np.uint8),
            )
            > 0
        )
    else:
        detection_guard = detection_union

    return (
        scene_roi
        & ~detection_guard
        & depth_valid
        & local_depth_valid
        & (normal_z >= float(config.support_plane_min_local_normal_z))
        & (
            local_depth_span
            <= float(config.support_plane_max_local_depth_span_m)
        )
    )


def _scene_support_roi_far_candidate_mask(
    depth: np.ndarray,
    bboxes: tuple[tuple[float, float, float, float], ...],
    intrinsics: Any,
    config: AdaptivePcaConfig,
) -> np.ndarray:
    """Sample far, locally flat surfaces inside transparent-object ROIs.

    Strict support sampling excludes every detector box because objects and
    grippers may dominate those pixels.  A transparent bin can leave only a
    few dozen valid support pixels outside its boxes, however.  This fallback
    keeps the expanded ROI, selects depths at or beyond a high quantile of the
    detector union, and relies on the same connected-component consensus,
    quality, clearance and normal gates before a plane can be accepted.
    """

    height, width = depth.shape
    depth_valid, local_depth_valid, normal_z, local_depth_span = (
        _scene_local_surface_features(depth, intrinsics, config)
    )
    detection_union, scene_roi = _scene_support_regions(depth, bboxes, config)
    detector_depths = depth[detection_union & depth_valid]
    if len(detector_depths) < int(config.support_plane_min_points):
        return np.zeros((height, width), dtype=bool)

    far_threshold = float(
        np.quantile(
            detector_depths,
            float(config.support_plane_roi_far_depth_quantile),
        )
    )
    if not isfinite(far_threshold):
        return np.zeros((height, width), dtype=bool)
    return (
        scene_roi
        & depth_valid
        & (depth >= far_threshold)
        & local_depth_valid
        & (
            normal_z
            >= float(config.support_plane_roi_far_min_local_normal_z)
        )
        & (
            local_depth_span
            <= float(config.support_plane_roi_far_max_local_depth_span_m)
        )
    )


def _fit_component_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the least-squares plane of one connected surface component."""

    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return None
    center = np.median(points, axis=0)
    try:
        _, singular_values, vh = np.linalg.svd(
            points - center,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return None
    if len(singular_values) < 3 or singular_values[1] <= 1e-9:
        return None
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if not isfinite(norm) or norm <= 1e-12:
        return None
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    if not np.all(np.isfinite(normal)):
        return None
    return center, normal


def _support_component_plane_hypotheses(
    components: tuple[_SupportSurfaceComponent, ...],
    config: AdaptivePcaConfig,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Generate deterministic planes from patches, not from arbitrary pixels.

    Each component contributes its own least-squares plane.  Additional RANSAC
    hypotheses always sample three *different* components, which prevents three
    nearby points from one object surface from defining a false "consensus".
    """

    hypotheses: list[tuple[np.ndarray, np.ndarray]] = []
    for component in components:
        fitted = _fit_component_plane(component.points)
        if fitted is None:
            continue
        _center, normal = fitted
        if normal[2] >= float(config.support_plane_min_normal_z):
            hypotheses.append(fitted)

    if len(components) < 3:
        return tuple(hypotheses)

    rng = np.random.default_rng(20261104)
    iterations = int(config.support_plane_component_ransac_iterations)
    for _ in range(iterations):
        selected_components = rng.choice(len(components), 3, replace=False)
        samples = []
        for component_index in selected_components:
            component = components[int(component_index)]
            point_index = int(rng.integers(0, len(component.points)))
            samples.append(component.points[point_index])
        samples_array = np.asarray(samples, dtype=float)
        normal = np.cross(
            samples_array[1] - samples_array[0],
            samples_array[2] - samples_array[0],
        )
        normal_norm = float(np.linalg.norm(normal))
        if not isfinite(normal_norm) or normal_norm <= 1e-10:
            continue
        normal = normal / normal_norm
        if normal[2] < 0.0:
            normal = -normal
        if normal[2] < float(config.support_plane_min_normal_z):
            continue
        center = np.mean(samples_array, axis=0)
        if np.all(np.isfinite(center)) and np.all(np.isfinite(normal)):
            hypotheses.append((center, normal))
    return tuple(hypotheses)


def _select_supporting_components(
    components: tuple[_SupportSurfaceComponent, ...],
    center: np.ndarray,
    normal: np.ndarray,
    config: AdaptivePcaConfig,
) -> tuple[
    tuple[int, ...],
    np.ndarray | None,
]:
    """Return component indices and their combined inlier point cloud."""

    selected_indices: list[int] = []
    inlier_clouds: list[np.ndarray] = []
    threshold = float(config.support_plane_ransac_threshold_m)
    min_component_points = int(config.support_plane_component_min_points)
    min_component_ratio = float(
        config.support_plane_component_min_inlier_ratio
    )
    for index, component in enumerate(components):
        residual = np.abs(component.points @ normal - float(center @ normal))
        component_inliers = component.points[residual <= threshold]
        component_ratio = len(component_inliers) / max(1, component.pixel_count)
        if (
            len(component_inliers) >= min_component_points
            and component_ratio >= min_component_ratio
        ):
            selected_indices.append(index)
            inlier_clouds.append(component_inliers)
    if not inlier_clouds:
        return (), None
    return tuple(selected_indices), np.vstack(inlier_clouds)


def _scene_support_plane_quality(
    estimate: SupportPlaneEstimate | None,
    config: AdaptivePcaConfig,
) -> float:
    """Use the same quality product as the downstream geometry policy."""

    if estimate is None:
        return 0.0
    max_rms = max(float(config.support_plane_max_rms_m), 1e-6)
    rms_score = max(0.0, min(1.0, 1.0 - estimate.rms_m / max_rms))
    inlier_score = max(0.0, min(1.0, estimate.inlier_ratio))
    normal_score = max(
        0.0,
        min(
            1.0,
            abs(estimate.normal_camera[2])
            / max(float(config.support_plane_min_normal_z), 1e-6),
        ),
    )
    return rms_score * inlier_score * normal_score


def _support_plane_center_depth(
    center: np.ndarray,
    normal: np.ndarray,
    u: float,
    v: float,
    intrinsics: Any,
) -> float | None:
    """Intersect a camera ray through ``(u, v)`` with the fitted plane."""

    denominator = (
        float(normal[0]) * (float(u) - float(intrinsics.cx))
        / float(intrinsics.fx)
        + float(normal[1]) * (float(v) - float(intrinsics.cy))
        / float(intrinsics.fy)
        + float(normal[2])
    )
    if not isfinite(denominator) or abs(denominator) <= 1e-12:
        return None
    depth = float(center @ normal) / denominator
    if not isfinite(depth) or depth <= 0.0:
        return None
    return depth


def _support_plane_target_clearances(
    depth: np.ndarray,
    bboxes: tuple[tuple[float, float, float, float], ...],
    center: np.ndarray,
    normal: np.ndarray,
    intrinsics: Any,
    config: AdaptivePcaConfig,
) -> tuple[float, ...] | None:
    """Require the plane to sit below/behind every detector target.

    A robust median from the central portion of each detector box estimates the
    target depth without allowing near, unsupported clutter at a box edge or
    corner to masquerade as the object.  The support plane must still lie
    measurably behind every target at the box centre.  This keeps an object top
    from being accepted as the support surface, while rejecting a distant
    background plane.  Any missing/invalid box or insufficient central depth
    support makes the scene fail closed.
    """

    height, width = depth.shape
    valid_depth = (
        np.isfinite(depth)
        & (depth >= float(config.min_depth_m))
        & (depth <= float(config.max_depth_m))
        & (depth > 0.0)
    )
    center_fraction = float(
        config.support_plane_target_depth_center_fraction
    )
    if (
        not isfinite(center_fraction)
        or center_fraction <= 0.0
        or center_fraction > 1.0
    ):
        return None
    inset_fraction = 0.5 * (1.0 - center_fraction)
    clearances: list[float] = []
    for raw_bbox in bboxes:
        if len(raw_bbox) != 4 or not all(
            isfinite(float(value)) for value in raw_bbox
        ):
            return None
        x0 = max(0, min(width, int(np.floor(float(raw_bbox[0])))))
        y0 = max(0, min(height, int(np.floor(float(raw_bbox[1])))))
        x1 = max(0, min(width, int(np.ceil(float(raw_bbox[2])))))
        y1 = max(0, min(height, int(np.ceil(float(raw_bbox[3])))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None

        box_width = float(x1 - x0)
        box_height = float(y1 - y0)
        center_x0 = max(
            0,
            min(width, int(np.floor(float(x0) + inset_fraction * box_width))),
        )
        center_x1 = max(
            0,
            min(width, int(np.ceil(float(x1) - inset_fraction * box_width))),
        )
        center_y0 = max(
            0,
            min(height, int(np.floor(float(y0) + inset_fraction * box_height))),
        )
        center_y1 = max(
            0,
            min(height, int(np.ceil(float(y1) - inset_fraction * box_height))),
        )
        if center_x1 - center_x0 < 2 or center_y1 - center_y0 < 2:
            return None
        patch_mask = valid_depth[center_y0:center_y1, center_x0:center_x1]
        values = depth[center_y0:center_y1, center_x0:center_x1][patch_mask]
        if len(values) < 3:
            return None
        observed_depth = float(np.median(values))
        center_u = 0.5 * (float(x0) + float(x1 - 1))
        center_v = 0.5 * (float(y0) + float(y1 - 1))
        predicted_depth = _support_plane_center_depth(
            center,
            normal,
            center_u,
            center_v,
            intrinsics,
        )
        if predicted_depth is None:
            return None
        clearance = predicted_depth - observed_depth
        if (
            not isfinite(clearance)
            or clearance < float(config.support_plane_min_target_clearance_m)
            or clearance > float(config.support_plane_max_target_clearance_m)
        ):
            return None
        clearances.append(clearance)
    return tuple(clearances)


def _component_consensus_support_plane(
    depth: np.ndarray,
    mask: np.ndarray,
    bboxes: tuple[tuple[float, float, float, float], ...],
    intrinsics: Any,
    config: AdaptivePcaConfig,
) -> SupportPlaneEstimate | None:
    """Select a support surface agreed on by separated image components."""

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    min_component_points = int(config.support_plane_component_min_points)
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.cx)
    cy = float(intrinsics.cy)
    components: list[_SupportSurfaceComponent] = []
    for label in range(1, component_count):
        pixel_count = int(stats[label, cv2.CC_STAT_AREA])
        if pixel_count < min_component_points:
            continue
        v, u = np.nonzero(labels == label)
        z = depth[v, u].astype(float, copy=False)
        points = np.column_stack(
            (
                (u.astype(float) - cx) / fx * z,
                (v.astype(float) - cy) / fy * z,
                z,
            )
        )
        components.append(
            _SupportSurfaceComponent(
                label=label,
                points=points,
                centroid_uv=(
                    float(centroids[label, 0]),
                    float(centroids[label, 1]),
                ),
            )
        )
    minimum_components = int(config.support_plane_min_support_components)
    if len(components) < minimum_components:
        return None

    hypotheses = _support_component_plane_hypotheses(tuple(components), config)
    if not hypotheses:
        return None

    best: SupportPlaneEstimate | None = None
    best_rank: tuple[float, int, int, float] | None = None
    seen_hypotheses: set[tuple[float, float, float, float]] = set()
    for center, normal in hypotheses:
        key = (
            round(float(normal[0]), 4),
            round(float(normal[1]), 4),
            round(float(normal[2]), 4),
            round(float(center @ normal), 4),
        )
        if key in seen_hypotheses:
            continue
        seen_hypotheses.add(key)

        selected_labels, inliers = _select_supporting_components(
            tuple(components),
            center,
            normal,
            config,
        )
        if inliers is None or len(selected_labels) < minimum_components:
            continue

        # Refit only points from the components that support the current plane,
        # then re-select against the new plane.  Every loop iteration therefore
        # leaves ``inliers`` matched to the latest ``center``/``normal``.  A
        # non-convergent hypothesis or any mid-loop fit failure is rejected
        # rather than silently reusing state from an earlier iteration.
        previous_labels = selected_labels
        converged = False
        for _ in range(6):
            fitted = _fit_component_plane(inliers)
            if fitted is None:
                break
            center, normal = fitted
            next_labels, next_inliers = _select_supporting_components(
                tuple(components),
                center,
                normal,
                config,
            )
            if next_inliers is None or len(next_labels) < minimum_components:
                break
            selected_labels = next_labels
            inliers = next_inliers
            if next_labels == previous_labels:
                converged = True
                break
            previous_labels = next_labels
        if not converged:
            continue

        selected_components = tuple(
            components[index] for index in selected_labels
        )
        candidate_points = sum(
            component.pixel_count for component in selected_components
        )
        residual = np.abs(inliers @ normal - float(center @ normal))
        rms = float(np.sqrt(np.mean(residual * residual)))
        estimate = SupportPlaneEstimate(
            center_camera_xyz=tuple(float(value) for value in center),
            normal_camera=tuple(float(value) for value in normal),
            rms_m=rms,
            valid_points=int(len(inliers)),
            candidate_points=int(candidate_points),
            inlier_ratio=float(len(inliers)) / max(1.0, float(candidate_points)),
            median_residual_m=float(np.median(residual)),
        )
        if _support_plane_rejection_reason(estimate, config):
            continue
        if _support_plane_target_clearances(
            depth,
            bboxes,
            center,
            normal,
            intrinsics,
            config,
        ) is None:
            continue
        component_centers = np.asarray(
            [component.centroid_uv for component in selected_components],
            dtype=float,
        )
        spans = np.ptp(component_centers, axis=0)
        required_span_x = (
            float(config.support_plane_component_min_span_x_ratio)
            * float(depth.shape[1])
        )
        required_span_y = (
            float(config.support_plane_component_min_span_y_ratio)
            * float(depth.shape[0])
        )
        if spans[0] < required_span_x or spans[1] < required_span_y:
            continue

        quality = _scene_support_plane_quality(estimate, config)
        rank = (
            quality,
            len(selected_labels),
            int(len(inliers)),
            float(spans[0] * spans[1]),
        )
        if best_rank is None or rank > best_rank:
            best = estimate
            best_rank = rank
    return best


def estimate_scene_support_plane(
    depth_m: Any,
    bboxes: Any,
    intrinsics: Any,
    config: AdaptivePcaConfig,
) -> SupportPlaneEstimate | None:
    """Fit one quality-gated support plane shared by all detections in a frame."""

    depth = _as_depth_array(depth_m)
    if depth.ndim != 2 or depth.shape[0] == 0 or depth.shape[1] == 0:
        raise ValueError("depth image is empty")
    fx = float(getattr(intrinsics, "fx"))
    fy = float(getattr(intrinsics, "fy"))
    cx = float(getattr(intrinsics, "cx"))
    cy = float(getattr(intrinsics, "cy"))
    if not all(isfinite(value) for value in (fx, fy, cx, cy)):
        raise ValueError("camera intrinsics contain non-finite values")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    normalized_bboxes = tuple(
        tuple(float(value) for value in bbox) for bbox in bboxes
    )
    if not normalized_bboxes:
        return None
    strict_mask = _scene_support_candidate_mask(
        depth,
        normalized_bboxes,
        intrinsics,
        config,
    )
    if int(np.count_nonzero(strict_mask)) >= int(
        config.support_plane_min_points
    ):
        strict_estimate = _component_consensus_support_plane(
            depth,
            strict_mask,
            normalized_bboxes,
            intrinsics,
            config,
        )
        if strict_estimate is not None:
            return strict_estimate

    # Transparent bins often expose the real support surface only through, or
    # close to, the detector boxes.  Try the far-depth ROI sampler only after
    # the conservative outside-box path has failed.  Its output is subject to
    # exactly the same component, fit-quality, clearance and coverage gates.
    fallback_mask = _scene_support_roi_far_candidate_mask(
        depth,
        normalized_bboxes,
        intrinsics,
        config,
    )
    if int(np.count_nonzero(fallback_mask)) < int(
        config.support_plane_min_points
    ):
        return None
    return _component_consensus_support_plane(
        depth,
        fallback_mask,
        normalized_bboxes,
        intrinsics,
        config,
    )


def estimate_object_pose(
    depth_m: Any,
    bbox: tuple[float, float, float, float],
    intrinsics: Any,
    config: AdaptivePcaConfig,
    object_mask: Any | None = None,
    scene_support_plane: SupportPlaneEstimate | None = None,
) -> ObjectPoseEstimate:
    """Estimate one object's robust 3-D centre and principal axes.

    ``bbox`` is in the same RGB-D image as ``depth_m`` and ``intrinsics``.  All
    returned coordinates are metres in the camera optical frame.  A
    ``scene_support_plane`` is an optional frame-level plane fitted from all
    detector boxes.  When supplied it is reused verbatim, which prevents a
    neighbouring object from changing one target's support estimate.

    ``ValueError`` is raised when the box/depth patch cannot provide a stable
    pose; callers should treat that as a rejected detection, never as a reason
    to move the arm.
    """

    depth = _as_depth_array(depth_m)
    color_object_mask = None
    if object_mask is not None:
        color_object_mask = np.asarray(object_mask, dtype=bool)
        if color_object_mask.shape != depth.shape:
            raise ValueError("object mask must match the depth image")
    waist_midline_enabled = bool(
        getattr(config, "waist_midline_enabled", False)
    )
    if waist_midline_enabled:
        if color_object_mask is None:
            raise ValueError(
                "waist-midline pose requires the target-colour object mask"
            )
        if not bool(config.support_plane_required):
            raise ValueError(
                "waist-midline pose requires a required support plane"
            )
    if depth.shape[0] == 0 or depth.shape[1] == 0:
        raise ValueError("depth image is empty")
    if len(bbox) != 4:
        raise ValueError("bbox must contain four values")

    fx = float(getattr(intrinsics, "fx"))
    fy = float(getattr(intrinsics, "fy"))
    cx = float(getattr(intrinsics, "cx"))
    cy = float(getattr(intrinsics, "cy"))
    if not all(isfinite(value) for value in (fx, fy, cx, cy)):
        raise ValueError("camera intrinsics contain non-finite values")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    height, width = depth.shape
    x0 = max(0, min(width, int(np.floor(float(bbox[0])))))
    y0 = max(0, min(height, int(np.floor(float(bbox[1])))))
    x1 = max(0, min(width, int(np.ceil(float(bbox[2])))))
    y1 = max(0, min(height, int(np.ceil(float(bbox[3])))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("bbox has no usable RGB-D pixels")

    patch = depth[y0:y1, x0:x1]
    patch_height, patch_width = patch.shape
    u = np.arange(x0, x0 + patch_width, dtype=float)[None, :]
    v = np.arange(y0, y0 + patch_height, dtype=float)[:, None]

    finite = np.isfinite(patch)
    in_range = finite & (patch >= float(config.min_depth_m)) & (
        patch <= float(config.max_depth_m)
    )
    if int(np.count_nonzero(in_range)) < int(config.min_valid_points):
        raise ValueError(
            "depth patch has fewer than "
            f"{int(config.min_valid_points)} valid points"
        )

    depth_values = patch[in_range]
    median_depth = float(np.median(depth_values))
    mad = float(np.median(np.abs(depth_values - median_depth)))
    # The MAD band removes gross depth outliers when the object fills the box
    # or the scene is effectively unimodal.  The absolute 10 mm floor handles
    # a perfectly planar view where the MAD can collapse to nearly zero.
    band_m = max(0.010, 3.0 * 1.4826 * mad)
    depth_mask = in_range & (np.abs(patch - median_depth) <= band_m)
    color_patch_mask = (
        None
        if color_object_mask is None
        else color_object_mask[y0:y1, x0:x1]
    )
    colour_component_mask = color_patch_mask
    support_plane = scene_support_plane
    support_rejection = (
        "shared scene support plane rejected"
        if scene_support_plane is not None
        else "no distinct object/support depth modes"
    )
    if scene_support_plane is not None:
        scene_rejection = _support_plane_rejection_reason(
            scene_support_plane,
            config,
        )
        if scene_rejection:
            raise ValueError(
                "shared scene support plane failed quality gates: "
                f"{scene_rejection}"
            )
    split_used = False
    split = _otsu_depth_split(
        depth_values,
        max(
            int(config.min_valid_points),
            int(np.ceil(0.05 * len(depth_values))),
        ),
    )
    if split is not None:
        threshold, _ = split
        foreground_mask = in_range & (patch <= threshold)
        background_mask = in_range & (patch > threshold)
        if (
            int(np.count_nonzero(foreground_mask)) >= int(config.min_valid_points)
            and int(np.count_nonzero(background_mask)) >= int(config.min_valid_points)
        ):
            split_used = True
            # Use only the nearer mode for object centring and PCA.  The farther
            # mode is the support surface and supplies the jaw-tip plane under
            # the object.  A supplied colour object mask is a hard foreground
            # constraint: the Otsu near mode may include yellow/background
            # clutter with the same depth, but that clutter must never enter
            # the target point cloud or distort its measured closure width.
            if color_patch_mask is not None:
                colour_component_mask = _dominant_colour_component_mask(
                    color_patch_mask,
                    overlap_mask=foreground_mask,
                )
                if colour_component_mask is None:
                    colour_component_mask = color_patch_mask
                color_foreground = in_range & colour_component_mask
                if int(np.count_nonzero(color_foreground)) < int(
                    config.min_valid_points
                ):
                    raise ValueError(
                        "colour object mask has too few valid depth points in "
                        "the Otsu foreground mode"
                    )
                depth_mask = color_foreground
            else:
                depth_mask = foreground_mask
            if support_plane is None:
                background_v, background_u = np.nonzero(background_mask)
                background_z = patch[background_v, background_u].astype(float)
                support_depth = float(np.median(background_z))

                # Sample a guard-free ring outside the detector box.  The bbox is
                # deliberately not used as the support ROI: a wide object leaves
                # too little tabletop inside it, while transparent box walls can
                # occupy a large fraction of its corners.
                guard = np.zeros((height, width), dtype=bool)
                guard[y0:y1, x0:x1] = foreground_mask
                if color_object_mask is not None:
                    guard |= color_object_mask
                ring = _support_ring_mask(
                    depth,
                    (x0, y0, x1, y1),
                    guard,
                    config,
                )

                depth_window = float(config.support_plane_depth_window_m)
                lower_depth = max(float(threshold) + 0.002, support_depth - depth_window)
                upper_depth = support_depth + depth_window
                full_valid = (
                    np.isfinite(depth)
                    & (depth >= float(config.min_depth_m))
                    & (depth <= float(config.max_depth_m))
                )
                ring &= (
                    full_valid
                    & (depth >= lower_depth)
                    & (depth <= upper_depth)
                )
                support_plane = _masked_support_plane(
                    depth,
                    ring,
                    intrinsics,
                    config,
                )
                support_rejection = _support_plane_rejection_reason(
                    support_plane,
                    config,
                )

                if support_plane is None or support_rejection:
                    # A tightly cropped detector box may leave no tabletop in the
                    # ring.  Retain the old in-box background fallback, but apply
                    # the same RANSAC and normal/RMS safety checks.
                    fallback_background_mask = np.zeros(
                        (height, width), dtype=bool
                    )
                    fallback_background_mask[y0:y1, x0:x1] = background_mask
                    fallback = _masked_support_plane(
                        depth,
                        fallback_background_mask,
                        intrinsics,
                        config,
                    )
                    fallback_rejection = _support_plane_rejection_reason(
                        fallback,
                        config,
                    )
                    if fallback is not None and not fallback_rejection:
                        support_plane = fallback
                        support_rejection = ""
                    elif support_plane is None:
                        support_rejection = fallback_rejection
                if support_plane is not None and _support_plane_rejection_reason(
                    support_plane, config
                ):
                    support_plane = None

    if not split_used:
        # A low-profile object can be only a few millimetres above the support
        # plane, so its depth histogram is intentionally unimodal.  Colour
        # segmentation and an object-free local ring still provide the two
        # independent observations needed for a safe fit: object points for
        # PCA, and a nearby plane for the commanded jaw-tip Z.
        if color_patch_mask is not None:
            colour_component_mask = _dominant_colour_component_mask(
                color_patch_mask,
                overlap_mask=in_range,
            )
            if colour_component_mask is None:
                colour_component_mask = color_patch_mask
            color_foreground = in_range & colour_component_mask
            if int(np.count_nonzero(color_foreground)) >= int(
                config.min_valid_points
            ):
                depth_mask = color_foreground
            else:
                raise ValueError(
                    "colour object mask has too few valid depth points for a "
                    "reliable object pose"
                )

        if support_plane is None:
            guard = np.zeros((height, width), dtype=bool)
            if color_object_mask is not None:
                guard |= color_object_mask
            else:
                guard[y0:y1, x0:x1] = depth_mask
            ring = _support_ring_mask(
                depth,
                (x0, y0, x1, y1),
                guard,
                config,
            )
            full_valid = (
                np.isfinite(depth)
                & (depth >= float(config.min_depth_m))
                & (depth <= float(config.max_depth_m))
            )
            ring &= full_valid
            support_plane = _masked_support_plane(
                depth,
                ring,
                intrinsics,
                config,
            )
            support_rejection = _support_plane_rejection_reason(
                support_plane,
                config,
            )

            if (support_plane is None or support_rejection) and color_patch_mask is not None:
                # If the object is close to an image border the local ring may be
                # too small.  The pixels inside the detector box that are not the
                # colour-labelled target are a second support candidate, but they
                # must still pass the same RANSAC, RMS, inlier-ratio and normal
                # gates as the preferred ring.
                object_guard = np.zeros((height, width), dtype=bool)
                object_guard[y0:y1, x0:x1] = color_patch_mask
                dilation = int(config.support_plane_guard_dilation_px)
                if dilation > 0:
                    kernel_size = 2 * dilation + 1
                    object_guard = (
                        cv2.dilate(
                            object_guard.astype(np.uint8),
                            np.ones((kernel_size, kernel_size), dtype=np.uint8),
                        )
                        > 0
                    )
                fallback_background_mask = np.zeros(
                    (height, width), dtype=bool
                )
                fallback_background_mask[y0:y1, x0:x1] = (
                    in_range & ~object_guard[y0:y1, x0:x1]
                )
                fallback = _masked_support_plane(
                    depth,
                    fallback_background_mask,
                    intrinsics,
                    config,
                )
                fallback_rejection = _support_plane_rejection_reason(
                    fallback,
                    config,
                )
                if fallback is not None and not fallback_rejection:
                    support_plane = fallback
                    support_rejection = ""
                elif support_plane is None:
                    support_rejection = fallback_rejection
            if support_plane is not None and _support_plane_rejection_reason(
                support_plane, config
            ):
                support_plane = None
    if config.support_plane_required and support_plane is None:
        raise ValueError(
            "no reliable support plane for visual grasp: "
            f"{support_rejection}"
        )

    waist_pixel_estimate = None
    waist_midpoint_uv_global = None
    waist_depth_m = None
    waist_surface_center = None
    if waist_midline_enabled:
        waist_patch_mask = color_patch_mask
        if waist_patch_mask is None:
            raise ValueError(
                "waist-midline pose requires target pixels inside the bbox"
            )
        waist_patch_mask = np.asarray(waist_patch_mask, dtype=bool)
        if int(np.count_nonzero(waist_patch_mask)) < int(
            config.waist_midline_min_mask_points
        ):
            raise ValueError(
                "target-colour mask inside the bbox has too few pixels for a "
                "waist midline"
            )
        try:
            waist_pixel_estimate = _estimate_waist_midline_pixel(
                waist_patch_mask,
                config,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"waist-midline geometry rejected: {exc}"
            ) from exc

        waist_midpoint_uv_global = (
            x0 + float(waist_pixel_estimate.center_uv[0]),
            y0 + float(waist_pixel_estimate.center_uv[1]),
        )
        radius = int(config.waist_midline_depth_radius_px)
        patch_v, patch_u = np.mgrid[
            0:patch_height,
            0:patch_width,
        ]
        waist_pixel_u = int(round(float(waist_pixel_estimate.center_uv[0])))
        waist_pixel_v = int(round(float(waist_pixel_estimate.center_uv[1])))
        waist_pixel_u = max(0, min(patch_width - 1, waist_pixel_u))
        waist_pixel_v = max(0, min(patch_height - 1, waist_pixel_v))
        depth_disk = (
            (patch_u - waist_pixel_u) ** 2
            + (patch_v - waist_pixel_v) ** 2
            <= radius * radius
        )
        depth_disk &= waist_patch_mask & in_range
        waist_depths = patch[depth_disk]
        if len(waist_depths) < 3:
            raise ValueError(
                "waist-midline centre has too few valid depth samples"
            )
        waist_depth_m = float(np.median(waist_depths))
        if not isfinite(waist_depth_m) or waist_depth_m <= 0.0:
            raise ValueError("waist-midline depth is invalid")
        waist_u = float(waist_midpoint_uv_global[0])
        waist_v = float(waist_midpoint_uv_global[1])
        waist_surface_center = np.asarray(
            (
                (waist_u - cx) / fx * waist_depth_m,
                (waist_v - cy) / fy * waist_depth_m,
                waist_depth_m,
            ),
            dtype=float,
        )

    if int(np.count_nonzero(depth_mask)) < int(config.min_valid_points):
        if color_patch_mask is not None:
            raise ValueError(
                "colour-constrained object points are insufficient for a "
                "reliable pose"
            )
        # Without a colour object constraint, fall back to the
        # percentile-trimmed depth range rather than turning a reasonable
        # object into a rejection solely because of sensor noise.
        low, high = np.percentile(depth_values, [5.0, 95.0])
        depth_mask = in_range & (patch >= low) & (patch <= high)
    if int(np.count_nonzero(depth_mask)) < int(config.min_valid_points):
        raise ValueError("not enough depth inliers inside the bbox")

    valid_v, valid_u = np.nonzero(depth_mask)
    z = patch[valid_v, valid_u]
    x = (valid_u + x0 - cx) / fx * z
    y = (valid_v + y0 - cy) / fy * z
    points = np.column_stack((x, y, z)).astype(float, copy=False)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < int(config.min_valid_points):
        raise ValueError("camera-frame point cloud is too small")

    # Remove gross radial outliers around the median point before PCA.  This
    # catches isolated depth spikes that survived the depth band.
    median_point = np.median(points, axis=0)
    radial = np.linalg.norm(points - median_point, axis=1)
    trim = float(config.trim_percentile)
    if trim > 0.0:
        radial_limit = float(np.percentile(radial, 100.0 - trim))
        radial_mask = radial <= radial_limit
        if int(np.count_nonzero(radial_mask)) >= int(config.min_valid_points):
            points = points[radial_mask]

    if len(points) < 3:
        raise ValueError("not enough points for PCA")

    # Two pass PCA: the first pass estimates the object axes, then projections
    # outside the configured percentile band are dropped before the final fit.
    center = np.median(points, axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[1] <= 1e-12:
        raise ValueError("object point cloud has no stable planar extent")

    if trim > 0.0:
        major_coord = centered @ eigenvectors[:, 0]
        minor_coord = centered @ eigenvectors[:, 1]
        major_low, major_high = np.percentile(
            major_coord, [trim, 100.0 - trim]
        )
        minor_low, minor_high = np.percentile(
            minor_coord, [trim, 100.0 - trim]
        )
        projected_mask = (
            (major_coord >= major_low)
            & (major_coord <= major_high)
            & (minor_coord >= minor_low)
            & (minor_coord <= minor_high)
        )
        if int(np.count_nonzero(projected_mask)) >= int(config.min_valid_points):
            points = points[projected_mask]
            center = np.median(points, axis=0)
            centered = points - center
            covariance = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

    if eigenvalues[1] <= 1e-12:
        raise ValueError("object point cloud has no stable minor axis")
    axis_ratio = float(np.sqrt(max(eigenvalues[0], 0.0) / eigenvalues[1]))
    defer_axis_ratio_rejection = bool(
        getattr(config, "defer_axis_ratio_rejection", False)
    )
    if axis_ratio < float(config.min_axis_ratio) and not defer_axis_ratio_rejection:
        raise ValueError(
            "object point cloud is not elongated enough for a stable grasp "
            f"heading (axis ratio {axis_ratio:.3f} < "
            f"{float(config.min_axis_ratio):.3f})"
        )

    major_axis = np.asarray(eigenvectors[:, 0], dtype=float)
    minor_axis = np.asarray(eigenvectors[:, 1], dtype=float)
    major_axis /= max(float(np.linalg.norm(major_axis)), 1e-12)
    minor_axis /= max(float(np.linalg.norm(minor_axis)), 1e-12)
    # PCA eigenvectors have arbitrary sign.  Make the sign deterministic so
    # the stabilizer and test traces do not flip between equivalent frames.
    major_dominant = int(np.argmax(np.abs(major_axis)))
    if major_axis[major_dominant] < 0.0:
        major_axis = -major_axis
    minor_dominant = int(np.argmax(np.abs(minor_axis)))
    if minor_axis[minor_dominant] < 0.0:
        minor_axis = -minor_axis

    major_length = _axis_span(points, major_axis, trim)
    minor_length = _axis_span(points, minor_axis, trim)
    if not isfinite(major_length) or major_length <= 0.0:
        raise ValueError("object major-axis length is not positive")

    axis_source = "global_pca"
    waist_midpoint_uv_px = None
    waist_path_length_px = None
    waist_half_width_px = None
    waist_tangent_uv = None
    waist_surface_raw_center = None
    if waist_pixel_estimate is not None:
        assert waist_depth_m is not None
        assert waist_midpoint_uv_global is not None
        assert waist_surface_center is not None
        waist_tangent_uv = waist_pixel_estimate.tangent_uv
        waist_short_uv = (
            -float(waist_tangent_uv[1]),
            float(waist_tangent_uv[0]),
        )
        try:
            major_axis, minor_axis = _waist_camera_axes(
                waist_tangent_uv,
                intrinsics,
                (
                    None
                    if support_plane is None
                    else np.asarray(support_plane.normal_camera, dtype=float)
                ),
            )
        except ValueError as exc:
            raise ValueError(
                f"waist-midline axes rejected: {exc}"
            ) from exc

        # Convert the silhouette half-width at the waist into metres along the
        # local image-space normal.  This is the actual jaw opening estimate;
        # the global PCA minor span remains unsuitable for a curved banana.
        metres_per_pixel = waist_depth_m * float(
            np.hypot(
                waist_short_uv[0] / fx,
                waist_short_uv[1] / fy,
            )
        )
        minor_length = (
            2.0
            * float(waist_pixel_estimate.half_width_px)
            * metres_per_pixel
        )
        if not isfinite(minor_length) or minor_length <= 0.0:
            raise ValueError("waist-midline local width is invalid")

        axis_source = "waist_midline"
        waist_midpoint_uv_px = waist_midpoint_uv_global
        waist_path_length_px = float(waist_pixel_estimate.path_length_px)
        waist_half_width_px = float(waist_pixel_estimate.half_width_px)
        waist_surface_raw_center = waist_surface_center.copy()

    # ``center`` is intentionally projected onto the support plane for the
    # legacy jaw-tip contract below.  Geometry policy needs the unmodified
    # surface/point-cloud centre so rim and sidewall heights remain measured
    # relative to the actual object, not to a zero-height tabletop point.
    raw_center = (
        center.copy()
        if waist_surface_raw_center is None
        else waist_surface_raw_center.copy()
    )
    geometry_evidence = None
    if defer_axis_ratio_rejection:
        geometry_evidence = analyze_grasp_geometry(
            points,
            object_center_camera_xyz=raw_center,
            support_plane_center_camera_xyz=(
                None
                if support_plane is None
                else support_plane.center_camera_xyz
            ),
            support_plane_normal_camera=(
                None if support_plane is None else support_plane.normal_camera
            ),
            support_plane_rms_m=(
                None if support_plane is None else support_plane.rms_m
            ),
            support_plane_inlier_ratio=(
                0.0 if support_plane is None else support_plane.inlier_ratio
            ),
            support_plane_max_rms_m=float(config.support_plane_max_rms_m),
            support_plane_min_normal_z=float(
                config.support_plane_min_normal_z
            ),
        )

    if support_plane is not None:
        # Project the object's centre ray onto the fitted support plane.  The
        # resulting XYZ is the commanded jaw-tip point: XY follows the object,
        # while Z remains on the tabletop rather than on the object's top face.
        if waist_surface_raw_center is None:
            ray_u = float(np.median(points[:, 0] / points[:, 2]))
            ray_v = float(np.median(points[:, 1] / points[:, 2]))
        else:
            ray_u = float(
                waist_surface_raw_center[0] / waist_surface_raw_center[2]
            )
            ray_v = float(
                waist_surface_raw_center[1] / waist_surface_raw_center[2]
            )
        ray = np.asarray((ray_u, ray_v, 1.0), dtype=float)
        plane_normal = np.asarray(
            support_plane.normal_camera,
            dtype=float,
        )
        plane_center = np.asarray(
            support_plane.center_camera_xyz,
            dtype=float,
        )
        denominator = float(plane_normal @ ray)
        if abs(denominator) > 1e-9:
            support_z = float(plane_normal @ plane_center / denominator)
            if isfinite(support_z) and support_z > 0.0:
                center = np.asarray(
                    (ray_u * support_z, ray_v * support_z, support_z),
                    dtype=float,
                )

    valid_ratio = min(1.0, len(points) / max(1.0, float(x1 - x0) * float(y1 - y0)))
    depth_inliers = int(np.count_nonzero(depth_mask))
    depth_ratio = min(1.0, depth_inliers / max(1.0, float(x1 - x0) * float(y1 - y0)))
    axis_score = min(1.0, max(0.0, (axis_ratio - 1.0) / 1.0))
    pose_confidence = min(
        1.0,
        max(
            0.0,
            0.45 * valid_ratio + 0.35 * axis_score + 0.20 * depth_ratio,
        ),
    )

    return ObjectPoseEstimate(
        center_camera_xyz=tuple(float(value) for value in center),
        major_axis_camera=tuple(float(value) for value in major_axis),
        minor_axis_camera=tuple(float(value) for value in minor_axis),
        major_axis_length_m=float(major_length),
        minor_axis_length_m=max(0.0, float(minor_length)),
        pose_confidence=float(pose_confidence),
        valid_points=int(len(points)),
        depth_inlier_ratio=float(depth_ratio),
        axis_ratio=float(axis_ratio),
        geometry_evidence=geometry_evidence,
        support_plane_center_camera_xyz=(
            None
            if support_plane is None
            else support_plane.center_camera_xyz
        ),
        support_plane_normal_camera=(
            None
            if support_plane is None
            else support_plane.normal_camera
        ),
        support_plane_rms_m=(
            None if support_plane is None else support_plane.rms_m
        ),
        support_plane_valid_points=(
            0 if support_plane is None else support_plane.valid_points
        ),
        support_plane_candidate_points=(
            0 if support_plane is None else support_plane.candidate_points
        ),
        support_plane_inlier_ratio=(
            0.0 if support_plane is None else support_plane.inlier_ratio
        ),
        axis_source=axis_source,
        waist_midpoint_uv_px=waist_midpoint_uv_px,
        waist_path_length_px=waist_path_length_px,
        waist_half_width_px=waist_half_width_px,
    )
