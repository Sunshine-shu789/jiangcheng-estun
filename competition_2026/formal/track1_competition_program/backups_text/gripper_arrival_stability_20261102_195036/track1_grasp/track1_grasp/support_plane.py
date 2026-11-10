"""Deterministic support-plane estimation in the robot base frame.

The grasp scorer only knows about object and obstacle points.  Whole-gripper
clearance additionally needs the support surface, but assuming ``base z=0`` is
not safe: hand-eye and robot-base offsets can put the real table above or below
that plane.  This module estimates a near-horizontal support plane directly
from a local obstacle point cloud and fails closed when the estimate is not
supported by enough inliers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


# The all-point fit is deliberately iterated.  In a real RGB-D cloud a single
# SVD is biased by objects resting on the table, while thresholded re-fitting
# converges to the dominant thin table layer.
_DEFAULT_REFINE_ITERATIONS = 12


def _finite_points(points: np.ndarray, name: str) -> np.ndarray:
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    return cloud


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if len(values) == 0:
        return None
    return float(np.percentile(values, float(percentile)))


@dataclass(frozen=True)
class SupportPlaneFitConfig:
    """Configuration for deterministic RGB-D support-plane estimation."""

    enabled: bool = True
    local_xy_radius_m: float = 0.160
    ransac_threshold_m: float = 0.002
    ransac_iterations: int = 5000
    max_fit_points: int = 30000
    min_inliers: int = 300
    min_inlier_ratio: float = 0.35
    max_rms_m: float = 0.0035
    min_normal_z: float = 0.80
    seed: int = 20261030

    def __post_init__(self) -> None:
        radius = float(self.local_xy_radius_m)
        threshold = float(self.ransac_threshold_m)
        iterations = int(self.ransac_iterations)
        max_fit_points = int(self.max_fit_points)
        min_inliers = int(self.min_inliers)
        ratio = float(self.min_inlier_ratio)
        max_rms = float(self.max_rms_m)
        min_normal_z = float(self.min_normal_z)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("local_xy_radius_m must be finite and positive")
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("ransac_threshold_m must be finite and positive")
        if iterations < 1:
            raise ValueError("ransac_iterations must be positive")
        if max_fit_points < 3:
            raise ValueError("max_fit_points must be at least three")
        if min_inliers < 3:
            raise ValueError("min_inliers must be at least three")
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between zero and one")
        if not math.isfinite(max_rms) or max_rms <= 0.0:
            raise ValueError("max_rms_m must be finite and positive")
        if not math.isfinite(min_normal_z) or not 0.0 < min_normal_z <= 1.0:
            raise ValueError("min_normal_z must be in (0, 1]")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "local_xy_radius_m", radius)
        object.__setattr__(self, "ransac_threshold_m", threshold)
        object.__setattr__(self, "ransac_iterations", iterations)
        object.__setattr__(self, "max_fit_points", max_fit_points)
        object.__setattr__(self, "min_inliers", min_inliers)
        object.__setattr__(self, "min_inlier_ratio", ratio)
        object.__setattr__(self, "max_rms_m", max_rms)
        object.__setattr__(self, "min_normal_z", min_normal_z)
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SupportPlaneFitConfig":
        world = payload.get("world_clearance", payload)
        if not isinstance(world, Mapping):
            raise ValueError("world_clearance must be a mapping")
        data = world.get("support_plane_estimation", world)
        if not isinstance(data, Mapping):
            raise ValueError("support_plane_estimation must be a mapping")
        defaults = cls()
        values = {
            name: data[name]
            for name in defaults.__dataclass_fields__
            if name in data
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class SupportPlaneEstimate:
    """Result of a support-plane fit, including fail-closed diagnostics."""

    valid: bool
    reason_codes: tuple[str, ...]
    origin_base_m: tuple[float, float, float] | None
    normal_base_m: tuple[float, float, float] | None
    plane_offset_m: float | None
    center_base_m: tuple[float, float, float] | None
    candidate_points: int
    inlier_points: int
    inlier_ratio: float
    rms_m: float | None
    median_abs_residual_m: float | None
    p95_abs_residual_m: float | None
    tilt_deg: float | None
    fit_sample_points: int
    seed: int
    threshold_m: float
    ransac_iterations: int
    inlier_mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "reason_codes": list(self.reason_codes),
            "origin_base_m": (
                None if self.origin_base_m is None else list(self.origin_base_m)
            ),
            "normal_base_m": (
                None if self.normal_base_m is None else list(self.normal_base_m)
            ),
            "plane_equation": (
                None
                if self.normal_base_m is None or self.plane_offset_m is None
                else "n_dot_p_plus_d_equals_zero"
            ),
            "plane_offset_m": self.plane_offset_m,
            "center_base_m": (
                None if self.center_base_m is None else list(self.center_base_m)
            ),
            "candidate_points": int(self.candidate_points),
            "inlier_points": int(self.inlier_points),
            "inlier_ratio": float(self.inlier_ratio),
            "rms_m": self.rms_m,
            "median_abs_residual_m": self.median_abs_residual_m,
            "p95_abs_residual_m": self.p95_abs_residual_m,
            "tilt_deg": self.tilt_deg,
            "fit_sample_points": int(self.fit_sample_points),
            "ransac": {
                "seed": int(self.seed),
                "threshold_m": float(self.threshold_m),
                "iterations": int(self.ransac_iterations),
            },
        }


def select_local_support_points(
    obstacle_points_base: np.ndarray,
    target_points_base: np.ndarray,
    *,
    local_xy_radius_m: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Select obstacle points in a base-frame XY disc around the target."""

    obstacles = _finite_points(obstacle_points_base, "obstacle_points_base")
    targets = _finite_points(target_points_base, "target_points_base")
    radius = float(local_xy_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("local_xy_radius_m must be finite and positive")
    if len(targets) == 0:
        raise ValueError("target_points_base must contain a finite point")
    center_xy = np.median(targets[:, :2], axis=0)
    distances = np.linalg.norm(obstacles[:, :2] - center_xy, axis=1)
    selected = obstacles[distances <= radius]
    return np.ascontiguousarray(selected, dtype=float), tuple(
        float(value) for value in center_xy
    )


def _plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("support plane normal is degenerate")
    normal /= normal_norm
    if normal[2] < 0.0:
        normal = -normal
    return center, normal


def _residual_mask(
    points: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return np.abs((points - center) @ normal) <= float(threshold)


def _refine_plane(
    points: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    *,
    threshold: float,
    iterations: int = _DEFAULT_REFINE_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Iteratively refit a near-horizontal plane on all candidate points."""

    plane_center = np.asarray(center, dtype=float)
    plane_normal = np.asarray(normal, dtype=float)
    mask = _residual_mask(points, plane_center, plane_normal, threshold)
    for _ in range(int(iterations)):
        if int(np.count_nonzero(mask)) < 3:
            return None
        plane_center, plane_normal = _plane_svd(points[mask])
        next_mask = _residual_mask(points, plane_center, plane_normal, threshold)
        if np.array_equal(next_mask, mask):
            mask = next_mask
            break
        mask = next_mask
    if int(np.count_nonzero(mask)) < 3:
        return None
    plane_center, plane_normal = _plane_svd(points[mask])
    mask = _residual_mask(points, plane_center, plane_normal, threshold)
    if int(np.count_nonzero(mask)) < 3:
        return None
    return plane_center, plane_normal, mask


def _invalid_estimate(
    *,
    reasons: list[str],
    candidate_points: int,
    fit_sample_points: int,
    config: SupportPlaneFitConfig,
    inlier_points: int = 0,
    inlier_ratio: float = 0.0,
    inlier_mask: np.ndarray | None = None,
) -> SupportPlaneEstimate:
    if inlier_mask is None:
        inlier_mask = np.zeros(candidate_points, dtype=bool)
    return SupportPlaneEstimate(
        valid=False,
        reason_codes=tuple(dict.fromkeys(reasons)),
        origin_base_m=None,
        normal_base_m=None,
        plane_offset_m=None,
        center_base_m=None,
        candidate_points=int(candidate_points),
        inlier_points=int(inlier_points),
        inlier_ratio=float(inlier_ratio),
        rms_m=None,
        median_abs_residual_m=None,
        p95_abs_residual_m=None,
        tilt_deg=None,
        fit_sample_points=int(fit_sample_points),
        seed=int(config.seed),
        threshold_m=float(config.ransac_threshold_m),
        ransac_iterations=int(config.ransac_iterations),
        inlier_mask=np.asarray(inlier_mask, dtype=bool),
    )


def fit_support_plane_base(
    points_base: np.ndarray,
    config: SupportPlaneFitConfig | None = None,
) -> SupportPlaneEstimate:
    """Fit a deterministic near-horizontal support plane in base coordinates.

    The returned normal is oriented toward positive base Z.  The support origin
    is the closest point on the plane to the robot-base origin, and the plane
    satisfies ``normal.dot(point) + offset == 0``.  A low-quality or wrongly
    oriented fit returns ``valid=False``; there is intentionally no fallback to
    an assumed ``z=0`` surface.
    """

    config = config or SupportPlaneFitConfig()
    points = _finite_points(points_base, "points_base")
    candidate_points = int(len(points))
    if not config.enabled:
        return _invalid_estimate(
            reasons=["support_plane_estimation_disabled"],
            candidate_points=candidate_points,
            fit_sample_points=0,
            config=config,
        )
    if candidate_points < 3:
        return _invalid_estimate(
            reasons=["support_plane_too_few_points"],
            candidate_points=candidate_points,
            fit_sample_points=candidate_points,
            config=config,
        )

    rng = np.random.default_rng(int(config.seed))
    if candidate_points > config.max_fit_points:
        sample_indices = rng.choice(
            candidate_points,
            config.max_fit_points,
            replace=False,
        )
        fit_points = points[sample_indices]
    else:
        fit_points = points
    fit_sample_points = int(len(fit_points))

    best_center: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    best_score: tuple[int, float] | None = None

    # Candidate 0: thresholded all-point SVD, followed by iterative refinement
    # against every local point.  This recovers the dominant table layer even
    # when the one-shot SVD is pulled upward by objects resting on it.
    seed_mask = np.ones(fit_sample_points, dtype=bool)
    seed_center, seed_normal = _plane_svd(fit_points[seed_mask])
    for _ in range(_DEFAULT_REFINE_ITERATIONS):
        if abs(float(seed_normal[2])) < config.min_normal_z:
            break
        next_mask = _residual_mask(
            fit_points,
            seed_center,
            seed_normal,
            config.ransac_threshold_m,
        )
        if int(np.count_nonzero(next_mask)) < 3:
            break
        seed_center, seed_normal = _plane_svd(fit_points[next_mask])
        if np.array_equal(next_mask, seed_mask):
            seed_mask = next_mask
            break
        seed_mask = next_mask
    if abs(float(seed_normal[2])) >= config.min_normal_z:
        refined = _refine_plane(
            points,
            seed_center,
            seed_normal,
            threshold=config.ransac_threshold_m,
        )
        if refined is not None:
            plane_center, plane_normal, plane_mask = refined
            residuals = np.abs((points[plane_mask] - plane_center) @ plane_normal)
            best_score = (
                int(np.count_nonzero(plane_mask)),
                -float(np.mean(residuals)),
            )
            best_center = plane_center
            best_normal = plane_normal

    # Deterministic RANSAC protects the estimate when a vertical box wall or a
    # large arbitrary surface dominates the all-point SVD.
    for _ in range(config.ransac_iterations):
        selected = fit_points[rng.choice(fit_sample_points, 3, replace=False)]
        normal = np.cross(selected[1] - selected[0], selected[2] - selected[0])
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            continue
        normal /= normal_norm
        if normal[2] < 0.0:
            normal = -normal
        if abs(float(normal[2])) < config.min_normal_z:
            continue
        center = np.mean(selected, axis=0)
        mask = _residual_mask(
            fit_points,
            center,
            normal,
            config.ransac_threshold_m,
        )
        count = int(np.count_nonzero(mask))
        if count < 3:
            continue
        residual = np.abs((fit_points[mask] - center) @ normal)
        score = (count, -float(np.mean(residual)))
        if best_score is None or score > best_score:
            best_score = score
            best_center = center
            best_normal = normal

    if best_center is None or best_normal is None:
        return _invalid_estimate(
            reasons=["support_plane_no_near_horizontal_plane"],
            candidate_points=candidate_points,
            fit_sample_points=fit_sample_points,
            config=config,
        )

    refined = _refine_plane(
        points,
        best_center,
        best_normal,
        threshold=config.ransac_threshold_m,
    )
    if refined is None:
        return _invalid_estimate(
            reasons=["support_plane_refinement_failed"],
            candidate_points=candidate_points,
            fit_sample_points=fit_sample_points,
            config=config,
        )
    center, normal, inlier_mask = refined
    inlier_points = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inlier_points) / max(1.0, float(candidate_points))
    inliers = points[inlier_mask]
    residuals = (inliers - center) @ normal
    abs_residuals = np.abs(residuals)
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))
    offset = float(-normal @ center)
    origin = -offset * normal

    reasons: list[str] = []
    if abs(float(normal[2])) < config.min_normal_z:
        reasons.append("support_plane_too_vertical")
    if inlier_points < config.min_inliers:
        reasons.append("support_plane_too_few_inliers")
    if inlier_ratio < config.min_inlier_ratio:
        reasons.append("support_plane_low_inlier_ratio")
    if rms > config.max_rms_m:
        reasons.append("support_plane_high_rms")

    return SupportPlaneEstimate(
        valid=not reasons,
        reason_codes=tuple(reasons),
        origin_base_m=tuple(float(value) for value in origin),
        normal_base_m=tuple(float(value) for value in normal),
        plane_offset_m=offset,
        center_base_m=tuple(float(value) for value in center),
        candidate_points=candidate_points,
        inlier_points=inlier_points,
        inlier_ratio=inlier_ratio,
        rms_m=rms,
        median_abs_residual_m=float(np.median(abs_residuals)),
        p95_abs_residual_m=float(np.percentile(abs_residuals, 95.0)),
        tilt_deg=tilt_deg,
        fit_sample_points=fit_sample_points,
        seed=int(config.seed),
        threshold_m=float(config.ransac_threshold_m),
        ransac_iterations=int(config.ransac_iterations),
        inlier_mask=np.asarray(inlier_mask, dtype=bool),
    )


__all__ = [
    "SupportPlaneEstimate",
    "SupportPlaneFitConfig",
    "fit_support_plane_base",
    "select_local_support_points",
]
