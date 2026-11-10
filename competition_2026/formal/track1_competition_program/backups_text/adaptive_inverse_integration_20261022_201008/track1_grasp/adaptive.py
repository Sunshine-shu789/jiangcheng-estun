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

import numpy as np

from .config import AdaptivePcaConfig


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


def estimate_object_pose(
    depth_m: Any,
    bbox: tuple[float, float, float, float],
    intrinsics: Any,
    config: AdaptivePcaConfig,
) -> ObjectPoseEstimate:
    """Estimate one object's robust 3-D centre and principal axes.

    ``bbox`` is in the same RGB-D image as ``depth_m`` and ``intrinsics``.  All
    returned coordinates are metres in the camera optical frame.  A
    ``ValueError`` is raised when the box/depth patch cannot provide a stable
    pose; callers should treat that as a rejected detection, never as a reason
    to move the arm.
    """

    depth = _as_depth_array(depth_m)
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
    # The MAD band removes the table/background while retaining sloped object
    # surfaces.  The absolute 10 mm floor handles a perfectly planar view where
    # the MAD can collapse to nearly zero.
    band_m = max(0.010, 3.0 * 1.4826 * mad)
    depth_mask = in_range & (np.abs(patch - median_depth) <= band_m)
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
    if axis_ratio < float(config.min_axis_ratio):
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
    )
