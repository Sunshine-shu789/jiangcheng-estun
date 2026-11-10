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
from math import isfinite
from typing import Any

import cv2
import numpy as np

from .config import AdaptivePcaConfig
from .grasp_geometry import GraspGeometryEvidence, analyze_grasp_geometry


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
    split_index = int(np.argmax(between_class_variance))
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


def estimate_object_pose(
    depth_m: Any,
    bbox: tuple[float, float, float, float],
    intrinsics: Any,
    config: AdaptivePcaConfig,
    object_mask: Any | None = None,
) -> ObjectPoseEstimate:
    """Estimate one object's robust 3-D centre and principal axes.

    ``bbox`` is in the same RGB-D image as ``depth_m`` and ``intrinsics``.  All
    returned coordinates are metres in the camera optical frame.  A
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
    support_plane = None
    support_rejection = "no distinct object/support depth modes"
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
            # the object.  This avoids a large tabletop background patch
            # dominating the median and making the principal axis follow the
            # detector box instead of the object.
            depth_mask = foreground_mask
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
        color_patch_mask = None
        if color_object_mask is not None:
            color_patch_mask = color_object_mask[y0:y1, x0:x1]
            color_foreground = in_range & color_patch_mask
            if int(np.count_nonzero(color_foreground)) >= int(
                config.min_valid_points
            ):
                depth_mask = color_foreground

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
    if int(np.count_nonzero(depth_mask)) < int(config.min_valid_points):
        # Fall back to the percentile-trimmed depth range rather than turning a
        # reasonable object into a rejection solely because of sensor noise.
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

    # ``center`` is intentionally projected onto the support plane for the
    # legacy jaw-tip contract below.  Geometry policy needs the unmodified
    # surface/point-cloud centre so rim and sidewall heights remain measured
    # relative to the actual object, not to a zero-height tabletop point.
    raw_center = center.copy()
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
        ray_u = float(np.median(points[:, 0] / points[:, 2]))
        ray_v = float(np.median(points[:, 1] / points[:, 2]))
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
    )
