"""CAD jaw-profile-aware scoring for GraspNet candidates.

The scorer keeps the semantics used by GraspNet's model-free collision
detector (local +X approach, +Y jaw opening, +Z finger height) but replaces
the fixed finger boxes with the measured/derived FAE2MR86M-D jaw profile.

Important coordinate facts
---------------------------
* A candidate translation is GraspNet's object-surface seed/reference point.
* The profile x coordinate runs from the physical finger root toward the tip.
* ``x_contact = finger_tip_x - graspnet_depth`` is the reference plane in the
  physical finger coordinate.
* The physical URDF ``grasp_tcp`` is at ``tcp_x_m`` from that same root, so its
  x coordinate in the candidate's contact frame is
  ``tcp_x_m - x_contact``.

The score is an offline ranking score, not a calibrated probability of grasp
success.  The bundled configuration is explicitly ``validated: false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from track1_grasp.graspnet_inference import GraspNetInferenceResult
from track1_grasp.jaw_profile import JawProfile, JawSection

_EPS = 1e-12
_JAW_FLIP_X = np.diag((1.0, -1.0, -1.0))


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _finite_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    value = float(mapping.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _proper_rotation(values: Sequence[float] | np.ndarray) -> np.ndarray:
    rotation = np.asarray(values, dtype=float)
    if rotation.shape == (9,):
        rotation = rotation.reshape(3, 3)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3 or a length-9 array")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2e-5:
        raise ValueError("rotation determinant must be +1")
    return rotation


def _points(points: Sequence[Sequence[float]] | np.ndarray | None, name: str) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=float)
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return np.ascontiguousarray(array[np.all(np.isfinite(array), axis=1)], dtype=float)


def _aabb_signed_distance(points: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    center = 0.5 * (minimum + maximum)
    half = 0.5 * (maximum - minimum)
    delta = np.abs(points - center) - half
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.minimum(np.max(delta, axis=1), 0.0)
    return outside + inside


def rotation_angle_between(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(first.T @ second)) - 1.0) * 0.5
    return float(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def jaw_symmetric_rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Distance modulo a 180-degree rotation about the approach axis.

    A parallel jaw rotated by 180 degrees about local X has the same opening
    and finger-height geometry, so treating those two rotations as distinct
    would leave near-duplicate candidates after NMS.
    """

    direct = rotation_angle_between(first, second)
    flipped = rotation_angle_between(first @ _JAW_FLIP_X, second)
    return min(direct, flipped)


@dataclass(frozen=True)
class JawScoringConfig:
    name: str = "fae2mr86m_ms42dc_cad_offline_v1"
    validated: bool = False
    contact_band_m: float = 0.007
    reference_width_band_m: float = 0.012
    penetration_skin_m: float = 0.0008
    collision_skin_m: float = 0.0015
    approach_distance_m: float = 0.035
    min_inner_points: int = 12
    min_contact_points_per_side: int = 4
    max_target_penetration_points: int = 3
    max_collision_points: int = 2
    width_tolerance_m: float = 0.010
    require_measured_width: bool = True
    opening_reserve_m: float = 0.002
    minimum_closure_margin_m: float = 0.001
    desired_obstacle_clearance_m: float = 0.008
    nms_translation_m: float = 0.025
    nms_rotation_deg: float = 30.0
    contact_weight: float = 0.30
    width_weight: float = 0.20
    engagement_weight: float = 0.15
    penetration_weight: float = 0.10
    obstacle_clearance_weight: float = 0.10
    tolerance_weight: float = 0.10
    opening_margin_weight: float = 0.05

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        positive = {
            "contact_band_m": self.contact_band_m,
            "reference_width_band_m": self.reference_width_band_m,
            "collision_skin_m": self.collision_skin_m,
            "approach_distance_m": self.approach_distance_m,
            "width_tolerance_m": self.width_tolerance_m,
            "desired_obstacle_clearance_m": self.desired_obstacle_clearance_m,
            "nms_translation_m": self.nms_translation_m,
            "nms_rotation_deg": self.nms_rotation_deg,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        non_negative = {
            "penetration_skin_m": self.penetration_skin_m,
            "opening_reserve_m": self.opening_reserve_m,
            "minimum_closure_margin_m": self.minimum_closure_margin_m,
        }
        for name, value in non_negative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        integer_non_negative = {
            "max_target_penetration_points": self.max_target_penetration_points,
            "max_collision_points": self.max_collision_points,
        }
        for name, value in integer_non_negative.items():
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.min_inner_points) < 1:
            raise ValueError("min_inner_points must be positive")
        if int(self.min_contact_points_per_side) < 1:
            raise ValueError("min_contact_points_per_side must be positive")
        weights = (
            self.contact_weight,
            self.width_weight,
            self.engagement_weight,
            self.penetration_weight,
            self.obstacle_clearance_weight,
            self.tolerance_weight,
            self.opening_margin_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights):
            raise ValueError("scoring weights must be finite and non-negative")
        if sum(float(value) for value in weights) <= 0.0:
            raise ValueError("at least one scoring weight must be positive")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JawScoringConfig":
        data = payload.get("scoring", payload)
        if not isinstance(data, Mapping):
            raise ValueError("scoring config must be a mapping")
        defaults = cls()
        values: dict[str, Any] = {
            "name": str(payload.get("name", defaults.name)),
            "validated": bool(payload.get("validated", defaults.validated)),
        }
        for name in defaults.__dataclass_fields__:
            if name in {"name", "validated"}:
                continue
            if name in data:
                values[name] = data[name]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def load_jaw_scoring_config(path: str | Path) -> JawScoringConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("jaw scoring config must contain a mapping")
    return JawScoringConfig.from_mapping(payload)


@dataclass(frozen=True)
class JawGraspCandidate:
    candidate_id: str
    translation_m: tuple[float, float, float]
    rotation: tuple[float, ...]
    #: GraspNet's predicted reference/upper-bound jaw opening.  This is not
    #: necessarily the opening at which the physical jaws make contact.
    width_m: float
    depth_m: float
    #: Optional CAD-consistent pre-grasp opening selected after evaluating the
    #: visible target geometry.  All physical geometry checks use this value
    #: when present and fall back to ``width_m`` for legacy callers.
    selected_opening_m: float | None = None
    height_m: float = 0.02
    tolerance_m: float = 0.0
    network_score: float = 1.0
    source_point_index: int = -1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_m, dtype=float)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation_m must contain three finite values")
        rotation = _proper_rotation(self.rotation)
        width = float(self.width_m)
        depth = float(self.depth_m)
        selected_opening = (
            None if self.selected_opening_m is None else float(self.selected_opening_m)
        )
        height = float(self.height_m)
        tolerance = float(self.tolerance_m)
        score = float(self.network_score)
        for name, value in (
            ("width_m", width),
            ("depth_m", depth),
            ("height_m", height),
            ("tolerance_m", tolerance),
            ("network_score", score),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if selected_opening is not None:
            if not math.isfinite(selected_opening):
                raise ValueError("selected_opening_m must be finite")
            if selected_opening <= 0.0:
                raise ValueError("selected_opening_m must be positive")
            if selected_opening > width + 1e-9:
                raise ValueError(
                    "selected_opening_m must not exceed the GraspNet reference width"
                )
        object.__setattr__(self, "translation_m", tuple(float(v) for v in translation))
        object.__setattr__(self, "rotation", tuple(float(v) for v in rotation.reshape(-1)))
        object.__setattr__(self, "width_m", width)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "selected_opening_m", selected_opening)
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "tolerance_m", max(0.0, tolerance))
        object.__setattr__(self, "network_score", score)
        object.__setattr__(self, "source_point_index", int(self.source_point_index))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if not str(self.candidate_id).strip():
            raise ValueError("candidate_id must not be empty")

    @property
    def translation(self) -> np.ndarray:
        return np.asarray(self.translation_m, dtype=float)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=float).reshape(3, 3)

    @property
    def jaw_opening_m(self) -> float:
        """Opening used for CAD contact, collision, clearance, and IK pose."""

        return (
            float(self.width_m)
            if self.selected_opening_m is None
            else float(self.selected_opening_m)
        )

    @classmethod
    def from_graspnet_result(
        cls,
        result: GraspNetInferenceResult,
        index: int,
    ) -> "JawGraspCandidate":
        index = int(index)
        if index < 0 or index >= len(result):
            raise IndexError("GraspNet candidate index is out of range")
        return cls(
            candidate_id=f"graspnet_{index:05d}",
            translation_m=tuple(float(v) for v in result.translations[index]),
            rotation=tuple(float(v) for v in result.rotations[index].reshape(-1)),
            width_m=float(result.widths[index]),
            depth_m=float(result.depths[index]),
            height_m=float(result.heights[index]),
            tolerance_m=float(result.tolerances[index]),
            network_score=float(result.network_scores[index]),
            source_point_index=int(result.source_point_indices[index]),
            metadata={
                "adjusted_network_score": float(result.scores[index]),
                "approach": result.approaches[index].astype(float).tolist(),
                "angle_rad": float(result.angles_rad[index]),
                "graspnet_seed_index": int(result.seed_indices[index]),
                "graspnet_translation_semantics": "surface seed/reference, not grasp_tcp",
            },
        )


@dataclass(frozen=True)
class JawGraspScore:
    candidate: JawGraspCandidate
    feasible: bool
    quality: float
    reason_codes: tuple[str, ...]
    measured_width_m: float | None
    inner_target_points: int
    contact_left_points: int
    contact_right_points: int
    target_penetration_points: int
    obstacle_finger_points: int
    obstacle_cavity_points: int
    obstacle_body_points: int
    obstacle_approach_points: int
    obstacle_clearance_m: float
    opening_margin_m: float
    section: JawSection
    contact_transform: np.ndarray
    body_transform: np.ndarray
    tcp_transform: np.ndarray
    quality_components: Mapping[str, float] = field(default_factory=dict)
    #: Optional base-frame whole-gripper support/workspace diagnostics.  This
    #: remains absent for camera-frame-only scoring; the offline integration
    #: applies it before any candidate can be treated as feasible.
    world_clearance: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": {
                "candidate_id": self.candidate.candidate_id,
                "translation_m": list(self.candidate.translation_m),
                "rotation": list(self.candidate.rotation),
                "width_m": self.candidate.width_m,
                "selected_opening_m": self.candidate.jaw_opening_m,
                "depth_m": self.candidate.depth_m,
                "height_m": self.candidate.height_m,
                "tolerance_m": self.candidate.tolerance_m,
                "network_score": self.candidate.network_score,
                "source_point_index": self.candidate.source_point_index,
                "metadata": dict(self.candidate.metadata),
            },
            "feasible": self.feasible,
            "quality": self.quality,
            "reason_codes": list(self.reason_codes),
            "measured_width_m": self.measured_width_m,
            "counts": {
                "inner_target_points": self.inner_target_points,
                "contact_left_points": self.contact_left_points,
                "contact_right_points": self.contact_right_points,
                "target_penetration_points": self.target_penetration_points,
                "obstacle_finger_points": self.obstacle_finger_points,
                "obstacle_cavity_points": self.obstacle_cavity_points,
                "obstacle_body_points": self.obstacle_body_points,
                "obstacle_approach_points": self.obstacle_approach_points,
            },
            "obstacle_clearance_m": self.obstacle_clearance_m,
            "opening_margin_m": self.opening_margin_m,
            "section": self.section.to_dict(),
            "frames": {
                "contact_transform": self.contact_transform.tolist(),
                "body_transform": self.body_transform.tolist(),
                "tcp_transform": self.tcp_transform.tolist(),
            },
            "quality_components": dict(self.quality_components),
            "world_clearance": (
                None
                if self.world_clearance is None
                else dict(self.world_clearance)
            ),
        }


class JawAwareGraspScorer:
    """Score one grasp against target-only and obstacle-only point clouds."""

    def __init__(self, profile: JawProfile, config: JawScoringConfig | None = None):
        self.profile = profile
        self.config = config or JawScoringConfig()
        depths, inner, outer = profile.profile_arrays(state_fraction=0.0)
        self._profile_depths = depths
        self._profile_inner_closed = inner
        self._profile_outer_closed = outer

    def _profile_at(self, state_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.profile.profile_arrays(state_fraction=state_fraction)

    def score(
        self,
        candidate: JawGraspCandidate,
        target_points: Sequence[Sequence[float]] | np.ndarray | None,
        obstacle_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> JawGraspScore:
        target = _points(target_points, "target_points")
        obstacle = _points(obstacle_points, "obstacle_points")
        rotation = candidate.rotation_matrix
        translation = candidate.translation
        contact_x = self.profile.contact_plane_x_m(candidate.depth_m)
        opening_m = candidate.jaw_opening_m
        section = self.profile.section(contact_x, opening_m)
        width_ok, width_reasons = self.profile.width_feasibility(
            contact_x,
            opening_m,
            opening_reserve_m=self.config.opening_reserve_m,
            minimum_closure_margin_m=self.config.minimum_closure_margin_m,
        )
        contact_transform = np.eye(4, dtype=float)
        contact_transform[:3, :3] = rotation
        contact_transform[:3, 3] = translation
        body_transform = contact_transform.copy()
        body_transform[:3, 3] = translation + rotation @ np.asarray((-contact_x, 0.0, 0.0))
        tcp_transform = self.profile.graspnet_contact_to_tcp_transform(
            translation,
            rotation,
            candidate.depth_m,
        )

        reasons: list[str] = list(width_reasons)
        target_local = (target - translation) @ rotation
        obstacle_local = (obstacle - translation) @ rotation
        profile_depths, inner_half, outer_half = self._profile_at(section.state_fraction)
        half_height = self.profile.finger_height_m * 0.5
        skin = self.config.collision_skin_m
        penetration_skin = self.config.penetration_skin_m

        if len(target_local) == 0:
            reasons.append("no_target_points")
            inner_points = left_contacts = right_contacts = penetration_points = 0
            measured_width = None
            contact_score = 0.0
            engagement_score = 0.0
            width_score = 0.0
            penetration_score = 0.0
        else:
            target_x_root = target_local[:, 0] + contact_x
            target_in_profile = (
                (target_x_root >= float(profile_depths[0]) - 1e-9)
                & (target_x_root <= float(profile_depths[-1]) + 1e-9)
            )
            target_in_height = np.abs(target_local[:, 2]) <= half_height + 1e-9
            target_inner = np.interp(target_x_root, profile_depths, inner_half)
            target_outer = np.interp(target_x_root, profile_depths, outer_half)
            target_abs_y = np.abs(target_local[:, 1])
            target_cavity = (
                target_in_profile
                & target_in_height
                & (target_abs_y < target_inner)
            )
            inner_points = int(np.count_nonzero(target_cavity))

            contact_distance = np.abs(target_abs_y - target_inner)
            contact_mask = (
                target_in_profile
                & target_in_height
                & (contact_distance <= self.config.contact_band_m)
                & (
                    target_abs_y
                    >= np.maximum(
                        0.0,
                        target_inner - self.config.contact_band_m,
                    )
                )
            )
            left_contacts = int(np.count_nonzero(contact_mask & (target_local[:, 1] > 0.0)))
            right_contacts = int(np.count_nonzero(contact_mask & (target_local[:, 1] < 0.0)))

            penetration_mask = (
                target_in_profile
                & target_in_height
                & (target_abs_y >= target_inner + penetration_skin)
                & (target_abs_y <= target_outer + penetration_skin)
            )
            penetration_points = int(np.count_nonzero(penetration_mask))

            reference_mask = (
                target_in_profile
                & target_in_height
                & (np.abs(target_local[:, 0]) <= self.config.reference_width_band_m)
            )
            if np.any(reference_mask):
                measured_width = float(
                    2.0 * np.percentile(target_abs_y[reference_mask], 90.0)
                )
            elif np.any(target_cavity):
                measured_width = float(
                    2.0 * np.percentile(target_abs_y[target_cavity], 90.0)
                )
            else:
                measured_width = None

            if inner_points < self.config.min_inner_points:
                reasons.append("empty_grasp")
            if left_contacts < self.config.min_contact_points_per_side:
                reasons.append("missing_left_contact")
            if right_contacts < self.config.min_contact_points_per_side:
                reasons.append("missing_right_contact")
            if penetration_points > self.config.max_target_penetration_points:
                reasons.append("target_penetration")

            contact_score = min(
                left_contacts / self.config.min_contact_points_per_side,
                right_contacts / self.config.min_contact_points_per_side,
            )
            contact_score *= min(left_contacts, right_contacts) / max(
                1, max(left_contacts, right_contacts)
            )
            contact_score = _clip01(contact_score)
            engagement_score = _clip01(
                inner_points / max(1, 2 * self.config.min_inner_points)
            )
            penetration_score = _clip01(
                1.0
                - penetration_points
                / max(1.0, float(self.config.max_target_penetration_points))
            )
            if measured_width is None:
                width_score = 0.0
            else:
                width_delta = abs(measured_width - opening_m)
                width_score = _clip01(
                    1.0
                    - width_delta
                    / self.config.width_tolerance_m
                )
                # The network width and the visible-cloud width are independent
                # observations.  A disagreement beyond tolerance means the
                # candidate is not safe to rank as grasp-feasible, even though
                # it can still be retained for offline diagnosis.
                if width_delta > self.config.width_tolerance_m:
                    reasons.append("measured_width_conflict")

        if measured_width is None and self.config.require_measured_width:
            reasons.append("width_unmeasured")

        if not width_ok:
            width_score = 0.0

        obstacle_finger, obstacle_cavity, obstacle_body, obstacle_approach, clearance = (
            self._obstacle_metrics(
                obstacle_local,
                contact_x=contact_x,
                state_fraction=section.state_fraction,
            )
        )
        collision_reasons: list[str] = []
        if obstacle_finger >= self.config.max_collision_points:
            reasons.append("finger_collision")
            collision_reasons.append("finger_collision")
        if obstacle_cavity >= self.config.max_collision_points:
            reasons.append("cavity_obstruction")
            collision_reasons.append("cavity_obstruction")
        if obstacle_body >= self.config.max_collision_points:
            reasons.append("body_collision")
            collision_reasons.append("body_collision")
        if obstacle_approach >= self.config.max_collision_points:
            reasons.append("approach_collision")
            collision_reasons.append("approach_collision")
        clearance_score = _clip01(
            clearance / self.config.desired_obstacle_clearance_m
        )
        if collision_reasons:
            # The configured skin is part of the collision boundary.  Once it
            # is violated, report effective clearance as zero even when the
            # nearest raw solid surface is still a few millimetres away.
            clearance = 0.0
            clearance_score = 0.0

        open_width, _ = self.profile._state_at_depth(contact_x, state="open")
        opening_margin = float(open_width - opening_m)
        opening_score = _clip01(
            opening_margin / max(self.config.opening_reserve_m, 1e-6)
        )
        tolerance_score = _clip01(candidate.tolerance_m / 0.05)
        weights = self.config
        weight_sum = sum(
            (
                weights.contact_weight,
                weights.width_weight,
                weights.engagement_weight,
                weights.penetration_weight,
                weights.obstacle_clearance_weight,
                weights.tolerance_weight,
                weights.opening_margin_weight,
            )
        )
        components = {
            "contact": contact_score,
            "width": width_score,
            "engagement": engagement_score,
            "penetration": penetration_score,
            "obstacle_clearance": clearance_score,
            "tolerance": tolerance_score,
            "opening_margin": opening_score,
        }
        quality = (
            weights.contact_weight * contact_score
            + weights.width_weight * width_score
            + weights.engagement_weight * engagement_score
            + weights.penetration_weight * penetration_score
            + weights.obstacle_clearance_weight * clearance_score
            + weights.tolerance_weight * tolerance_score
            + weights.opening_margin_weight * opening_score
        ) / weight_sum
        feasible = not reasons
        # De-duplicate reasons while preserving deterministic order.
        reason_codes = tuple(dict.fromkeys(reasons))
        return JawGraspScore(
            candidate=candidate,
            feasible=feasible,
            quality=float(quality),
            reason_codes=reason_codes,
            measured_width_m=measured_width,
            inner_target_points=inner_points,
            contact_left_points=left_contacts,
            contact_right_points=right_contacts,
            target_penetration_points=penetration_points,
            obstacle_finger_points=obstacle_finger,
            obstacle_cavity_points=obstacle_cavity,
            obstacle_body_points=obstacle_body,
            obstacle_approach_points=obstacle_approach,
            obstacle_clearance_m=float(clearance),
            opening_margin_m=opening_margin,
            section=section,
            contact_transform=contact_transform,
            body_transform=body_transform,
            tcp_transform=tcp_transform,
            quality_components=components,
        )

    def _obstacle_metrics(
        self,
        obstacle_local: np.ndarray,
        *,
        contact_x: float,
        state_fraction: float,
    ) -> tuple[int, int, int, int, float]:
        if len(obstacle_local) == 0:
            return 0, 0, 0, 0, self.config.desired_obstacle_clearance_m
        depths, inner_half, outer_half = self._profile_at(state_fraction)
        half_height = self.profile.finger_height_m * 0.5
        skin = self.config.collision_skin_m
        x_root = obstacle_local[:, 0] + contact_x
        in_profile = (
            (x_root >= float(depths[0]) - 1e-9)
            & (x_root <= float(depths[-1]) + 1e-9)
        )
        inside_height = np.abs(obstacle_local[:, 2]) <= half_height + skin
        inner = np.interp(x_root, depths, inner_half)
        outer = np.interp(x_root, depths, outer_half)
        abs_y = np.abs(obstacle_local[:, 1])
        finger_mask = in_profile & inside_height & (abs_y >= inner - skin) & (abs_y <= outer + skin)
        cavity_mask = in_profile & inside_height & (abs_y < inner - skin)

        body = self.profile.body_envelope
        body_min = np.asarray(
            (body.rear_x_m - contact_x - skin, -body.half_width_m - skin, -body.half_height_m - skin)
        )
        body_max = np.asarray(
            (-contact_x + skin, body.half_width_m + skin, body.half_height_m + skin)
        )
        body_mask = np.all(
            (obstacle_local >= body_min) & (obstacle_local <= body_max), axis=1
        )
        approach_x_max = body_min[0]
        approach_x_min = approach_x_max - self.config.approach_distance_m
        approach_min = np.asarray(
            (approach_x_min, -body.half_width_m - skin, -body.half_height_m - skin)
        )
        approach_max = np.asarray(
            (approach_x_max, body.half_width_m + skin, body.half_height_m + skin)
        )
        approach_mask = np.all(
            (obstacle_local >= approach_min) & (obstacle_local <= approach_max), axis=1
        )

        # Approximate minimum clearance to the CAD finger material.  For points
        # outside the profile range, evaluate the nearest end cross-section and
        # include the axial distance.
        x_clipped_root = np.clip(x_root, float(depths[0]), float(depths[-1]))
        inner_at = np.interp(x_clipped_root, depths, inner_half)
        outer_at = np.interp(x_clipped_root, depths, outer_half)
        axial = np.maximum(
            np.maximum(float(depths[0]) - x_root, x_root - float(depths[-1])),
            0.0,
        )
        radial = np.maximum(
            np.maximum(inner_at - abs_y, abs_y - outer_at), 0.0
        )
        vertical = np.maximum(np.abs(obstacle_local[:, 2]) - half_height, 0.0)
        finger_clearance = np.sqrt(axial**2 + radial**2 + vertical**2)
        body_clearance = np.abs(_aabb_signed_distance(obstacle_local, body_min, body_max))
        approach_clearance = np.abs(
            _aabb_signed_distance(obstacle_local, approach_min, approach_max)
        )
        min_clearance = float(
            min(
                np.min(finger_clearance),
                np.min(body_clearance),
                np.min(approach_clearance),
            )
        )
        return (
            int(np.count_nonzero(finger_mask)),
            int(np.count_nonzero(cavity_mask)),
            int(np.count_nonzero(body_mask)),
            int(np.count_nonzero(approach_mask)),
            min_clearance,
        )


@dataclass(frozen=True)
class JawNmsSuppression:
    candidate_id: str
    kept_candidate_id: str


def nms_jaw_grasp_scores(
    scores: Sequence[JawGraspScore],
    config: JawScoringConfig,
) -> tuple[list[JawGraspScore], list[JawNmsSuppression]]:
    """Greedy feasible-candidate NMS with parallel-jaw rotation symmetry."""

    feasible = [score for score in scores if score.feasible]
    feasible.sort(key=lambda score: score.quality, reverse=True)
    kept: list[JawGraspScore] = []
    suppressed: list[JawNmsSuppression] = []
    max_rotation = math.radians(config.nms_rotation_deg)
    for score in feasible:
        duplicate_of: str | None = None
        for accepted in kept:
            distance = float(
                np.linalg.norm(
                    score.candidate.translation - accepted.candidate.translation
                )
            )
            if distance > config.nms_translation_m:
                continue
            angle = jaw_symmetric_rotation_distance(
                score.candidate.rotation_matrix,
                accepted.candidate.rotation_matrix,
            )
            if angle <= max_rotation:
                duplicate_of = accepted.candidate.candidate_id
                break
        if duplicate_of is None:
            kept.append(score)
        else:
            suppressed.append(
                JawNmsSuppression(
                    candidate_id=score.candidate.candidate_id,
                    kept_candidate_id=duplicate_of,
                )
            )
    return kept, suppressed


__all__ = [
    "JawAwareGraspScorer",
    "JawGraspCandidate",
    "JawGraspScore",
    "JawNmsSuppression",
    "JawScoringConfig",
    "jaw_symmetric_rotation_distance",
    "load_jaw_scoring_config",
    "nms_jaw_grasp_scores",
]
