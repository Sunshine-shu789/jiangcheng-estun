"""Offline GraspNet-style parallel-jaw grasp quality scoring.

This module borrows the useful parts of GraspNet's post-processing without
requiring PyTorch, Open3D, CUDA extensions or the GraspNet checkpoint:

* local gripper-frame voxel collision estimates for left/right finger, palm and
  approach corridor;
* an empty-grasp check inside the jaw cavity;
* opposite-contact (antipodal) support from target points;
* width fit and maximum-opening checks;
* tolerance-aware quality ranking;
* 3-D translation + rotation NMS.

The coordinate convention is explicit and follows the useful part of
GraspNet's collision detector: local +X is approach/depth, local +Y is the
jaw opening axis and local +Z is grasp height.  Candidate rotations map local
coordinates to the point-cloud frame.

The score is a ranking score, not a calibrated probability of real-world
success.  The supplied MS42DC geometry is deliberately marked unvalidated and
must be replaced by measured/CAD geometry before any live use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


_EPS = 1e-12


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(min(maximum, max(minimum, value)))


def _as_float_tuple(
    values: Sequence[float] | np.ndarray,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in np.asarray(values).reshape(-1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric values") from exc
    if len(result) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain only finite values")
    return result


def _rotation_matrix(flat_values: Sequence[float]) -> np.ndarray:
    rotation = np.asarray(flat_values, dtype=float).reshape(3, 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4, rtol=0.0):
        raise ValueError("candidate rotation must be orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2e-4:
        raise ValueError("candidate rotation determinant must be +1")
    return rotation


def rotation_angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """Return the geodesic rotation angle between two rotation matrices."""

    cosine = (float(np.trace(first.T @ second)) - 1.0) * 0.5
    return float(math.acos(_clamp(cosine, -1.0, 1.0)))


def _points_array(points: Sequence[Sequence[float]] | np.ndarray | None, name: str) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=float)
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must be an N x 3 array")
    finite = np.all(np.isfinite(array), axis=1)
    return np.ascontiguousarray(array[finite], dtype=float)


def _box_mask(
    points_local: np.ndarray,
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> np.ndarray:
    if len(points_local) == 0:
        return np.zeros(0, dtype=bool)
    minimum_array = np.asarray(minimum, dtype=float)
    maximum_array = np.asarray(maximum, dtype=float)
    return np.all(
        (points_local >= minimum_array) & (points_local <= maximum_array),
        axis=1,
    )


@dataclass(frozen=True)
class ParallelJawGeometry:
    """Local geometric envelope used by the offline scorer.

    ``finger_length_m`` is measured along approach X, ``finger_thickness_m``
    along jaw Y and ``finger_height_m`` along Z.  ``palm_length_m`` extends
    behind the fingers along -X.  These are conservative scoring primitives,
    not a replacement for the real MS42DC CAD mesh.
    """

    max_opening_m: float = 0.140
    min_opening_m: float = 0.005
    finger_length_m: float = 0.050
    finger_thickness_m: float = 0.010
    finger_height_m: float = 0.025
    palm_length_m: float = 0.025
    palm_height_m: float = 0.035
    approach_distance_m: float = 0.050
    contact_band_m: float = 0.008
    collision_skin_m: float = 0.001
    closure_clearance_m: float = 0.002
    opening_reserve_m: float = 0.005

    def __post_init__(self) -> None:
        positive = {
            "max_opening_m": self.max_opening_m,
            "finger_length_m": self.finger_length_m,
            "finger_thickness_m": self.finger_thickness_m,
            "finger_height_m": self.finger_height_m,
            "palm_length_m": self.palm_length_m,
            "palm_height_m": self.palm_height_m,
            "approach_distance_m": self.approach_distance_m,
            "contact_band_m": self.contact_band_m,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"gripper.{name} must be positive and finite")
        non_negative = {
            "min_opening_m": self.min_opening_m,
            "collision_skin_m": self.collision_skin_m,
            "closure_clearance_m": self.closure_clearance_m,
            "opening_reserve_m": self.opening_reserve_m,
        }
        for name, value in non_negative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"gripper.{name} must be non-negative and finite")
        if self.min_opening_m >= self.max_opening_m:
            raise ValueError("gripper.min_opening_m must be below max_opening_m")
        if self.opening_reserve_m >= self.max_opening_m:
            raise ValueError("gripper.opening_reserve_m must be below max_opening_m")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ParallelJawGeometry":
        mapping = dict(data or {})
        defaults = cls()
        values: dict[str, float] = {}
        for name in defaults.__dataclass_fields__:
            values[name] = float(mapping.get(name, getattr(defaults, name)))
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class GraspScoringConfig:
    """Thresholds and weights for :class:`ParallelJawGraspScorer`."""

    name: str = "ms42dc_estimated_offline_v1"
    validated: bool = False
    geometry: ParallelJawGeometry = field(default_factory=ParallelJawGeometry)
    voxel_size_m: float = 0.002
    collision_iou_threshold: float = 0.01
    collision_voxel_hits: int = 1
    target_penetration_iou_threshold: float = 0.01
    target_penetration_voxel_hits: int = 3
    min_inner_points: int = 8
    min_contact_points: int = 3
    width_tolerance_m: float = 0.008
    good_tolerance_m: float = 0.020
    desired_obstacle_clearance_m: float = 0.008
    nms_translation_m: float = 0.030
    nms_rotation_deg: float = 30.0
    max_results: int = 50
    contact_weight: float = 0.30
    width_weight: float = 0.20
    engagement_weight: float = 0.15
    collision_weight: float = 0.15
    obstacle_clearance_weight: float = 0.10
    tolerance_weight: float = 0.05
    opening_margin_weight: float = 0.05

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scoring.name must not be empty")
        if self.voxel_size_m <= 0.0 or not math.isfinite(self.voxel_size_m):
            raise ValueError("scoring.voxel_size_m must be positive and finite")
        if self.collision_iou_threshold <= 0.0:
            raise ValueError("scoring.collision_iou_threshold must be positive")
        if self.collision_voxel_hits < 1:
            raise ValueError("scoring.collision_voxel_hits must be >= 1")
        if self.target_penetration_iou_threshold <= 0.0:
            raise ValueError(
                "scoring.target_penetration_iou_threshold must be positive"
            )
        if self.target_penetration_voxel_hits < 1:
            raise ValueError(
                "scoring.target_penetration_voxel_hits must be >= 1"
            )
        if self.min_inner_points < 1:
            raise ValueError("scoring.min_inner_points must be >= 1")
        if self.min_contact_points < 1:
            raise ValueError("scoring.min_contact_points must be >= 1")
        if self.width_tolerance_m <= 0.0:
            raise ValueError("scoring.width_tolerance_m must be positive")
        if self.good_tolerance_m <= 0.0:
            raise ValueError("scoring.good_tolerance_m must be positive")
        if self.desired_obstacle_clearance_m <= 0.0:
            raise ValueError(
                "scoring.desired_obstacle_clearance_m must be positive"
            )
        if self.nms_translation_m < 0.0:
            raise ValueError("scoring.nms_translation_m must be non-negative")
        if not 0.0 <= self.nms_rotation_deg <= 180.0:
            raise ValueError("scoring.nms_rotation_deg must be in [0, 180]")
        if self.max_results < 1:
            raise ValueError("scoring.max_results must be >= 1")
        weights = self.weights
        if any(value < 0.0 or not math.isfinite(value) for value in weights.values()):
            raise ValueError("scoring weights must be finite and non-negative")
        if sum(weights.values()) <= _EPS:
            raise ValueError("at least one scoring weight must be positive")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "contact": float(self.contact_weight),
            "width": float(self.width_weight),
            "engagement": float(self.engagement_weight),
            "collision": float(self.collision_weight),
            "obstacle_clearance": float(
                self.obstacle_clearance_weight
            ),
            "tolerance": float(self.tolerance_weight),
            "opening_margin": float(self.opening_margin_weight),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "GraspScoringConfig":
        payload = dict(data or {})
        geometry = ParallelJawGeometry.from_mapping(
            payload.get("gripper", payload.get("geometry"))
        )
        raw_scoring = payload.get("scoring", payload)
        scoring = dict(raw_scoring or {})

        def number(name: str, default: float) -> float:
            return float(scoring.get(name, default))

        defaults = cls()
        return cls(
            name=str(payload.get("name", defaults.name)),
            validated=bool(payload.get("validated", defaults.validated)),
            geometry=geometry,
            voxel_size_m=number("voxel_size_m", defaults.voxel_size_m),
            collision_iou_threshold=number(
                "collision_iou_threshold",
                defaults.collision_iou_threshold,
            ),
            collision_voxel_hits=int(
                scoring.get("collision_voxel_hits", defaults.collision_voxel_hits)
            ),
            target_penetration_iou_threshold=number(
                "target_penetration_iou_threshold",
                defaults.target_penetration_iou_threshold,
            ),
            target_penetration_voxel_hits=int(
                scoring.get(
                    "target_penetration_voxel_hits",
                    defaults.target_penetration_voxel_hits,
                )
            ),
            min_inner_points=int(
                scoring.get("min_inner_points", defaults.min_inner_points)
            ),
            min_contact_points=int(
                scoring.get("min_contact_points", defaults.min_contact_points)
            ),
            width_tolerance_m=number(
                "width_tolerance_m", defaults.width_tolerance_m
            ),
            good_tolerance_m=number(
                "good_tolerance_m", defaults.good_tolerance_m
            ),
            desired_obstacle_clearance_m=number(
                "desired_obstacle_clearance_m",
                defaults.desired_obstacle_clearance_m,
            ),
            nms_translation_m=number(
                "nms_translation_m", defaults.nms_translation_m
            ),
            nms_rotation_deg=number(
                "nms_rotation_deg", defaults.nms_rotation_deg
            ),
            max_results=int(scoring.get("max_results", defaults.max_results)),
            contact_weight=number("contact_weight", defaults.contact_weight),
            width_weight=number("width_weight", defaults.width_weight),
            engagement_weight=number(
                "engagement_weight", defaults.engagement_weight
            ),
            collision_weight=number(
                "collision_weight", defaults.collision_weight
            ),
            obstacle_clearance_weight=number(
                "obstacle_clearance_weight",
                defaults.obstacle_clearance_weight,
            ),
            tolerance_weight=number(
                "tolerance_weight", defaults.tolerance_weight
            ),
            opening_margin_weight=number(
                "opening_margin_weight", defaults.opening_margin_weight
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "validated": self.validated,
            "gripper": self.geometry.to_dict(),
            "scoring": {
                name: value
                for name, value in self.__dict__.items()
                if name
                not in {
                    "name",
                    "validated",
                    "geometry",
                }
            },
        }


def load_scoring_config(path: str | Path) -> GraspScoringConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("grasp scoring config must contain a mapping")
    return GraspScoringConfig.from_dict(payload)


@dataclass(frozen=True)
class GraspCandidate:
    """One 6-D parallel-jaw candidate in a point-cloud coordinate frame."""

    candidate_id: str
    translation_m: tuple[float, float, float]
    rotation: tuple[float, ...]
    width_m: float
    height_m: float
    depth_m: float
    tolerance_m: float = 0.0
    network_score: float = 1.0
    object_id: int = -1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        translation = _as_float_tuple(self.translation_m, 3, "translation_m")
        rotation = _as_float_tuple(self.rotation, 9, "rotation")
        _rotation_matrix(rotation)
        width = float(self.width_m)
        height = float(self.height_m)
        depth = float(self.depth_m)
        tolerance = float(self.tolerance_m)
        network_score = float(self.network_score)
        for name, value in (
            ("width_m", width),
            ("height_m", height),
            ("depth_m", depth),
            ("tolerance_m", tolerance),
            ("network_score", network_score),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if tolerance < 0.0:
            raise ValueError("tolerance_m must be non-negative")
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            candidate_id = (
                "grasp_"
                f"{translation[0]:+.4f}_"
                f"{translation[1]:+.4f}_"
                f"{translation[2]:+.4f}"
            )
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "width_m", width)
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "tolerance_m", tolerance)
        object.__setattr__(self, "network_score", network_score)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "object_id", int(self.object_id))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def translation(self) -> np.ndarray:
        return np.asarray(self.translation_m, dtype=float)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return _rotation_matrix(self.rotation)

    @classmethod
    def from_graspnet_array(
        cls,
        grasp: Sequence[float] | np.ndarray,
        *,
        candidate_id: str = "",
        tolerance_m: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "GraspCandidate":
        """Parse GraspNet's 17-float array.

        Layout: score, width, height, depth, rotation(9), translation(3),
        object_id.
        """

        values = np.asarray(grasp, dtype=float).reshape(-1)
        if values.size != 17:
            raise ValueError("GraspNet candidate array must contain 17 floats")
        rotation = tuple(float(value) for value in values[4:13])
        translation = tuple(float(value) for value in values[13:16])
        return cls(
            candidate_id=candidate_id,
            translation_m=translation,
            rotation=rotation,
            width_m=float(values[1]),
            height_m=float(values[2]),
            depth_m=float(values[3]),
            tolerance_m=float(tolerance_m),
            network_score=float(values[0]),
            object_id=int(round(float(values[16]))),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "translation_m": list(self.translation_m),
            "rotation": list(self.rotation),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "depth_m": self.depth_m,
            "tolerance_m": self.tolerance_m,
            "network_score": self.network_score,
            "object_id": self.object_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraspQualityResult:
    candidate: GraspCandidate
    feasible: bool
    quality: float
    collision_iou: float
    left_finger_iou: float
    right_finger_iou: float
    palm_iou: float
    approach_iou: float
    collision_voxels: int
    target_penetration_iou: float
    target_penetration_voxels: int
    inner_points: int
    empty_ratio: float
    contact_left_points: int
    contact_right_points: int
    antipodal_score: float
    measured_width_m: float | None
    width_error_m: float | None
    opening_margin_m: float
    obstacle_clearance_m: float | None
    reason_codes: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "feasible": self.feasible,
            "quality": self.quality,
            "collision_iou": self.collision_iou,
            "left_finger_iou": self.left_finger_iou,
            "right_finger_iou": self.right_finger_iou,
            "palm_iou": self.palm_iou,
            "approach_iou": self.approach_iou,
            "collision_voxels": self.collision_voxels,
            "target_penetration_iou": self.target_penetration_iou,
            "target_penetration_voxels": self.target_penetration_voxels,
            "inner_points": self.inner_points,
            "empty_ratio": self.empty_ratio,
            "contact_left_points": self.contact_left_points,
            "contact_right_points": self.contact_right_points,
            "antipodal_score": self.antipodal_score,
            "measured_width_m": self.measured_width_m,
            "width_error_m": self.width_error_m,
            "opening_margin_m": self.opening_margin_m,
            "obstacle_clearance_m": self.obstacle_clearance_m,
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class NmsSuppression:
    candidate_id: str
    kept_candidate_id: str
    translation_distance_m: float
    rotation_difference_rad: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kept_candidate_id": self.kept_candidate_id,
            "translation_distance_m": self.translation_distance_m,
            "rotation_difference_rad": self.rotation_difference_rad,
        }


@dataclass(frozen=True)
class _Region:
    name: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def volume(self) -> float:
        size = np.asarray(self.maximum) - np.asarray(self.minimum)
        if np.any(size <= 0.0):
            return 0.0
        return float(np.prod(size))

    def mask(self, points_local: np.ndarray) -> np.ndarray:
        return _box_mask(points_local, self.minimum, self.maximum)

    def strict_mask(self, points_local: np.ndarray) -> np.ndarray:
        if len(points_local) == 0:
            return np.zeros(0, dtype=bool)
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)
        return np.all(
            (points_local > minimum) & (points_local < maximum),
            axis=1,
        )

    def distance(self, points_local: np.ndarray) -> np.ndarray:
        if len(points_local) == 0:
            return np.empty(0, dtype=float)
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)
        outside = np.maximum(
            np.maximum(minimum - points_local, points_local - maximum),
            0.0,
        )
        return np.linalg.norm(outside, axis=1)


def _occupancy(
    points_local: np.ndarray,
    mask: np.ndarray,
    expected_volume_m3: float,
    voxel_size_m: float,
) -> tuple[float, int]:
    selected = points_local[mask] if len(points_local) else np.empty((0, 3))
    if len(selected) == 0:
        return 0.0, 0
    voxel_indices = np.floor(selected / float(voxel_size_m)).astype(np.int64)
    occupied = int(len(np.unique(voxel_indices, axis=0)))
    expected = max(1.0, float(expected_volume_m3) / float(voxel_size_m) ** 3)
    return float(min(1.0, occupied / expected)), occupied


class ParallelJawGraspScorer:
    """Score parallel-jaw candidates against target and obstacle point clouds."""

    def __init__(self, config: GraspScoringConfig | None = None) -> None:
        self.config = config or GraspScoringConfig()

    def _invalid_result(
        self,
        candidate: GraspCandidate,
        reason_codes: Sequence[str],
        reasons: Sequence[str],
        *,
        opening_margin_m: float,
    ) -> GraspQualityResult:
        return GraspQualityResult(
            candidate=candidate,
            feasible=False,
            quality=0.0,
            collision_iou=0.0,
            left_finger_iou=0.0,
            right_finger_iou=0.0,
            palm_iou=0.0,
            approach_iou=0.0,
            collision_voxels=0,
            target_penetration_iou=0.0,
            target_penetration_voxels=0,
            inner_points=0,
            empty_ratio=1.0,
            contact_left_points=0,
            contact_right_points=0,
            antipodal_score=0.0,
            measured_width_m=None,
            width_error_m=None,
            opening_margin_m=float(opening_margin_m),
            obstacle_clearance_m=None,
            reason_codes=tuple(reason_codes),
            reasons=tuple(reasons),
        )

    def score(
        self,
        candidate: GraspCandidate,
        target_points: Sequence[Sequence[float]] | np.ndarray,
        obstacle_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> GraspQualityResult:
        config = self.config
        geometry = config.geometry
        target_world = _points_array(target_points, "target_points")
        obstacle_world = _points_array(obstacle_points, "obstacle_points")
        opening_margin = geometry.max_opening_m - candidate.width_m

        if len(target_world) == 0:
            return self._invalid_result(
                candidate,
                ("no_target_points",),
                ("no finite target points were supplied",),
                opening_margin_m=opening_margin,
            )
        if candidate.width_m <= 0.0:
            return self._invalid_result(
                candidate,
                ("invalid_width",),
                ("candidate width must be positive",),
                opening_margin_m=opening_margin,
            )
        if candidate.height_m <= 0.0:
            return self._invalid_result(
                candidate,
                ("invalid_height",),
                ("candidate height must be positive",),
                opening_margin_m=opening_margin,
            )
        if candidate.depth_m <= 0.0:
            return self._invalid_result(
                candidate,
                ("invalid_depth",),
                ("candidate depth must be positive",),
                opening_margin_m=opening_margin,
            )

        rotation = candidate.rotation_matrix
        translation = candidate.translation
        target_local = (target_world - translation) @ rotation
        obstacle_local = (obstacle_world - translation) @ rotation

        width_half = candidate.width_m * 0.5
        cavity_half_height = candidate.height_m * 0.5
        finger_x_min = candidate.depth_m - geometry.finger_length_m
        finger_x_max = candidate.depth_m
        finger_half_height = max(
            cavity_half_height,
            geometry.finger_height_m * 0.5,
        )
        skin = geometry.collision_skin_m

        left_finger = _Region(
            "left_finger",
            (
                finger_x_min,
                -width_half - geometry.finger_thickness_m - skin,
                -finger_half_height,
            ),
            (
                finger_x_max,
                -width_half - skin,
                finger_half_height,
            ),
        )
        right_finger = _Region(
            "right_finger",
            (
                finger_x_min,
                width_half + skin,
                -finger_half_height,
            ),
            (
                finger_x_max,
                width_half + geometry.finger_thickness_m + skin,
                finger_half_height,
            ),
        )
        palm_x_max = finger_x_min
        palm_x_min = palm_x_max - geometry.palm_length_m
        palm_half_height = max(
            finger_half_height,
            geometry.palm_height_m * 0.5,
        )
        palm = _Region(
            "palm",
            (
                palm_x_min,
                -width_half - geometry.finger_thickness_m - skin,
                -palm_half_height,
            ),
            (
                palm_x_max,
                width_half + geometry.finger_thickness_m + skin,
                palm_half_height,
            ),
        )
        approach_x_max = palm_x_min
        approach_x_min = approach_x_max - geometry.approach_distance_m
        approach = _Region(
            "approach",
            (
                approach_x_min,
                -width_half - geometry.finger_thickness_m - skin,
                -finger_half_height,
            ),
            (
                approach_x_max,
                width_half + geometry.finger_thickness_m + skin,
                finger_half_height,
            ),
        )
        regions = (left_finger, right_finger, palm, approach)

        region_iou: dict[str, float] = {}
        region_hits: dict[str, int] = {}
        # Obstacle points determine physical collisions.  Target points are
        # intentionally excluded here because a valid grasp is supposed to
        # touch the target at the jaw surfaces.
        for region in regions:
            iou, hits = _occupancy(
                obstacle_local,
                region.mask(obstacle_local),
                region.volume,
                config.voxel_size_m,
            )
            region_iou[region.name] = iou
            region_hits[region.name] = hits

        union_mask = np.zeros(len(obstacle_local), dtype=bool)
        for region in regions:
            union_mask |= region.mask(obstacle_local)
        collision_iou, collision_voxels = _occupancy(
            obstacle_local,
            union_mask,
            sum(region.volume for region in regions),
            config.voxel_size_m,
        )

        # Target points slightly beyond the ideal contact planes indicate that
        # the finger/palm envelope cuts into the object.  Strict masks exclude
        # points exactly on the nominal contact boundary.
        target_penetration_mask = np.zeros(len(target_local), dtype=bool)
        for region in regions:
            target_penetration_mask |= region.strict_mask(target_local)
        (
            target_penetration_iou,
            target_penetration_voxels,
        ) = _occupancy(
            target_local,
            target_penetration_mask,
            sum(region.volume for region in regions),
            config.voxel_size_m,
        )

        inner_mask = _box_mask(
            target_local,
            (finger_x_min, -width_half, -cavity_half_height),
            (finger_x_max, width_half, cavity_half_height),
        )
        inner_points = int(np.count_nonzero(inner_mask))
        empty_ratio = 1.0 - _clamp(
            inner_points / float(config.min_inner_points)
        )

        contact_mask = _box_mask(
            target_local,
            (finger_x_min, -width_half - geometry.contact_band_m, -cavity_half_height),
            (finger_x_max, width_half + geometry.contact_band_m, cavity_half_height),
        )
        left_contact_mask = (
            contact_mask
            & (
                np.abs(target_local[:, 1] + width_half)
                <= geometry.contact_band_m
            )
        )
        right_contact_mask = (
            contact_mask
            & (
                np.abs(target_local[:, 1] - width_half)
                <= geometry.contact_band_m
            )
        )
        contact_left_points = int(np.count_nonzero(left_contact_mask))
        contact_right_points = int(np.count_nonzero(right_contact_mask))
        left_coverage = _clamp(
            contact_left_points / float(config.min_contact_points)
        )
        right_coverage = _clamp(
            contact_right_points / float(config.min_contact_points)
        )
        if min(contact_left_points, contact_right_points) <= 0:
            antipodal_score = 0.0
        else:
            balance = min(contact_left_points, contact_right_points) / float(
                max(contact_left_points, contact_right_points)
            )
            antipodal_score = min(left_coverage, right_coverage) * (
                0.75 + 0.25 * balance
            )

        span_mask = _box_mask(
            target_local,
            (
                finger_x_min,
                -width_half - geometry.finger_thickness_m - geometry.contact_band_m,
                -cavity_half_height,
            ),
            (
                finger_x_max,
                width_half + geometry.finger_thickness_m + geometry.contact_band_m,
                cavity_half_height,
            ),
        )
        span_points = target_local[span_mask]
        measured_width_m: float | None = None
        width_error_m: float | None = None
        if len(span_points) >= 5:
            low, high = np.percentile(span_points[:, 1], (5.0, 95.0))
            measured = float(high - low)
            if measured > 1e-5:
                measured_width_m = measured
                desired_width = measured + geometry.closure_clearance_m
                width_error_m = candidate.width_m - desired_width

        obstacle_clearance_m: float | None = None
        if len(obstacle_local):
            distances = [
                float(np.min(region.distance(obstacle_local)))
                for region in regions
                if len(obstacle_local)
            ]
            if distances:
                obstacle_clearance_m = min(distances)

        reason_codes: list[str] = []
        reasons: list[str] = []
        available_opening = geometry.max_opening_m - geometry.opening_reserve_m
        if candidate.width_m < geometry.min_opening_m:
            reason_codes.append("below_min_opening")
            reasons.append(
                f"width {candidate.width_m:.4f} m is below minimum opening "
                f"{geometry.min_opening_m:.4f} m"
            )
        if candidate.width_m > available_opening:
            reason_codes.append("exceeds_usable_opening")
            reasons.append(
                f"width {candidate.width_m:.4f} m leaves less than "
                f"{geometry.opening_reserve_m:.4f} m opening reserve"
            )
        if empty_ratio >= 1.0 or inner_points < config.min_inner_points:
            reason_codes.append("empty_grasp")
            reasons.append(
                f"jaw cavity contains {inner_points} target point(s); need "
                f"{config.min_inner_points}"
            )
        if contact_left_points < config.min_contact_points:
            reason_codes.append("missing_left_contact")
            reasons.append(
                f"left jaw contact has {contact_left_points} point(s); need "
                f"{config.min_contact_points}"
            )
        if contact_right_points < config.min_contact_points:
            reason_codes.append("missing_right_contact")
            reasons.append(
                f"right jaw contact has {contact_right_points} point(s); need "
                f"{config.min_contact_points}"
            )
        for region in regions:
            if (
                region_hits[region.name] >= config.collision_voxel_hits
                or region_iou[region.name] > config.collision_iou_threshold
            ):
                reason_codes.append(f"{region.name}_collision")
                reasons.append(
                    f"{region.name} collision: {region_hits[region.name]} "
                    f"voxel(s), IoU={region_iou[region.name]:.4f}"
                )
        if (
            target_penetration_voxels >= config.target_penetration_voxel_hits
            or target_penetration_iou
            > config.target_penetration_iou_threshold
        ):
            reason_codes.append("target_penetration")
            reasons.append(
                "target points penetrate the gripper envelope: "
                f"{target_penetration_voxels} voxel(s), "
                f"IoU={target_penetration_iou:.4f}"
            )
        if (
            measured_width_m is not None
            and measured_width_m
            > candidate.width_m + 2.0 * geometry.contact_band_m
        ):
            reason_codes.append("grasp_too_narrow")
            reasons.append(
                f"measured width {measured_width_m:.4f} m cannot fit the "
                f"{candidate.width_m:.4f} m candidate"
            )

        feasible = not reason_codes
        if not feasible:
            quality = 0.0
        else:
            width_fit = (
                0.5
                if width_error_m is None
                else float(
                    math.exp(
                        -0.5
                        * (width_error_m / config.width_tolerance_m) ** 2
                    )
                )
            )
            collision_clearance = 1.0 - _clamp(
                collision_iou / config.collision_iou_threshold
            )
            obstacle_quality = (
                1.0
                if obstacle_clearance_m is None
                else _clamp(
                    obstacle_clearance_m
                    / config.desired_obstacle_clearance_m
                )
            )
            tolerance_quality = (
                0.5
                if candidate.tolerance_m <= 0.0
                else _clamp(candidate.tolerance_m / config.good_tolerance_m)
            )
            opening_quality = _clamp(
                opening_margin
                / max(4.0 * geometry.opening_reserve_m, 1e-6)
            )
            engagement = _clamp(
                inner_points / float(config.min_inner_points)
            )
            weighted = {
                "contact": antipodal_score,
                "width": width_fit,
                "engagement": engagement,
                "collision": collision_clearance,
                "obstacle_clearance": obstacle_quality,
                "tolerance": tolerance_quality,
                "opening_margin": opening_quality,
            }
            weights = config.weights
            denominator = sum(weights.values())
            quality = sum(
                weights[name] * weighted[name] for name in weights
            ) / denominator
            quality = _clamp(quality)

        return GraspQualityResult(
            candidate=candidate,
            feasible=feasible,
            quality=float(quality),
            collision_iou=float(collision_iou),
            left_finger_iou=float(region_iou["left_finger"]),
            right_finger_iou=float(region_iou["right_finger"]),
            palm_iou=float(region_iou["palm"]),
            approach_iou=float(region_iou["approach"]),
            collision_voxels=int(collision_voxels),
            target_penetration_iou=float(target_penetration_iou),
            target_penetration_voxels=int(target_penetration_voxels),
            inner_points=inner_points,
            empty_ratio=float(empty_ratio),
            contact_left_points=contact_left_points,
            contact_right_points=contact_right_points,
            antipodal_score=float(antipodal_score),
            measured_width_m=measured_width_m,
            width_error_m=width_error_m,
            opening_margin_m=float(opening_margin),
            obstacle_clearance_m=obstacle_clearance_m,
            reason_codes=tuple(reason_codes),
            reasons=tuple(reasons),
        )

    def score_many(
        self,
        candidates: Sequence[GraspCandidate],
        target_points: Sequence[Sequence[float]] | np.ndarray,
        obstacle_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> tuple[GraspQualityResult, ...]:
        return tuple(
            self.score(candidate, target_points, obstacle_points)
            for candidate in candidates
        )


def nms_grasps(
    results: Sequence[GraspQualityResult],
    config: GraspScoringConfig | None = None,
) -> tuple[tuple[GraspQualityResult, ...], tuple[NmsSuppression, ...]]:
    """Return feasible results after score-ordered 3-D NMS."""

    active_config = config or GraspScoringConfig()
    pooled = [item for item in results if item.feasible]
    pooled.sort(
        key=lambda item: (
            item.quality,
            item.candidate.network_score,
            item.candidate.candidate_id,
        ),
        reverse=True,
    )
    kept: list[GraspQualityResult] = []
    suppressed: list[NmsSuppression] = []
    rotation_threshold = math.radians(active_config.nms_rotation_deg)
    for result in pooled:
        candidate = result.candidate
        duplicate_of: GraspQualityResult | None = None
        for existing in kept:
            translation_distance = float(
                np.linalg.norm(
                    candidate.translation - existing.candidate.translation
                )
            )
            if translation_distance > active_config.nms_translation_m:
                continue
            rotation_difference = rotation_angle_between(
                candidate.rotation_matrix,
                existing.candidate.rotation_matrix,
            )
            if rotation_difference <= rotation_threshold:
                duplicate_of = existing
                break
        if duplicate_of is None:
            if len(kept) < active_config.max_results:
                kept.append(result)
            continue
        translation_distance = float(
            np.linalg.norm(
                candidate.translation - duplicate_of.candidate.translation
            )
        )
        rotation_difference = rotation_angle_between(
            candidate.rotation_matrix,
            duplicate_of.candidate.rotation_matrix,
        )
        suppressed.append(
            NmsSuppression(
                candidate_id=candidate.candidate_id,
                kept_candidate_id=duplicate_of.candidate.candidate_id,
                translation_distance_m=translation_distance,
                rotation_difference_rad=rotation_difference,
            )
        )
    return tuple(kept), tuple(suppressed)


def score_candidates_with_nms(
    candidates: Sequence[GraspCandidate],
    target_points: Sequence[Sequence[float]] | np.ndarray,
    obstacle_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    config: GraspScoringConfig | None = None,
) -> tuple[
    tuple[GraspQualityResult, ...],
    tuple[GraspQualityResult, ...],
    tuple[NmsSuppression, ...],
]:
    """Convenience wrapper returning all results, NMS winners and suppressions."""

    active_config = config or GraspScoringConfig()
    scorer = ParallelJawGraspScorer(active_config)
    all_results = scorer.score_many(candidates, target_points, obstacle_points)
    kept, suppressed = nms_grasps(all_results, active_config)
    return all_results, kept, suppressed
