"""Whole-gripper support-surface clearance diagnostics.

The jaw scorer works in the point-cloud frame and checks object contact plus
local obstacle collisions.  It does not know where the table is.  This module
adds the missing base-frame, fail-closed test by transforming a conservative
set of CAD finger and body samples through the candidate pose.

Only geometry is evaluated here.  A successful result is **not** permission
to move the robot; it merely means that the sampled gripper envelope is above
the configured support surface and that the physical TCP is inside the
configured workspace height.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from track1_grasp.jaw_grasp_scoring import JawGraspScore
from track1_grasp.jaw_profile import JawProfile


def _finite_vector(value: object, name: str, length: int = 3) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite values")
    return vector


def _proper_rotation(value: object, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2e-5:
        raise ValueError(f"{name} determinant must be +1")
    return rotation


@dataclass(frozen=True)
class GripperClearanceConfig:
    """Support-plane and workspace limits expressed in the robot base frame."""

    support_origin_base_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    support_normal_base: tuple[float, float, float] = (0.0, 0.0, 1.0)
    minimum_clearance_m: float = 0.020
    workspace_min_z_m: float = 0.020
    finger_y_samples: int = 3

    def __post_init__(self) -> None:
        origin = _finite_vector(self.support_origin_base_m, "support_origin_base_m")
        normal = _finite_vector(self.support_normal_base, "support_normal_base")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            raise ValueError("support_normal_base must be non-zero")
        clearance = float(self.minimum_clearance_m)
        workspace_min_z = float(self.workspace_min_z_m)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError("minimum_clearance_m must be finite and non-negative")
        if not math.isfinite(workspace_min_z):
            raise ValueError("workspace_min_z_m must be finite")
        y_samples = int(self.finger_y_samples)
        if y_samples < 2:
            raise ValueError("finger_y_samples must be at least two")
        object.__setattr__(self, "support_origin_base_m", tuple(float(v) for v in origin))
        object.__setattr__(
            self,
            "support_normal_base",
            tuple(float(v) for v in normal / normal_norm),
        )
        object.__setattr__(self, "minimum_clearance_m", clearance)
        object.__setattr__(self, "workspace_min_z_m", workspace_min_z)
        object.__setattr__(self, "finger_y_samples", y_samples)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GripperClearanceConfig":
        data = payload.get("world_clearance", payload)
        if not isinstance(data, Mapping):
            raise ValueError("world_clearance must be a mapping")
        defaults = cls()
        values: dict[str, Any] = {}
        for name in defaults.__dataclass_fields__:
            if name in data:
                values[name] = data[name]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def load_gripper_clearance_config(path: str | Path) -> GripperClearanceConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("clearance config must contain a mapping")
    return GripperClearanceConfig.from_mapping(payload)


@dataclass(frozen=True)
class GripperWorldClearanceResult:
    feasible: bool
    reason_codes: tuple[str, ...]
    support_minimum_clearance_m: float
    support_minimum_point_base_m: tuple[float, float, float]
    support_minimum_region: str
    tcp_base_m: tuple[float, float, float]
    tcp_support_clearance_m: float
    tcp_workspace_margin_m: float
    gripper_aabb_min_base_m: tuple[float, float, float]
    gripper_aabb_max_base_m: tuple[float, float, float]
    finger_sample_count: int
    body_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "reason_codes": list(self.reason_codes),
            "support_minimum_clearance_m": self.support_minimum_clearance_m,
            "support_minimum_point_base_m": list(self.support_minimum_point_base_m),
            "support_minimum_region": self.support_minimum_region,
            "tcp_base_m": list(self.tcp_base_m),
            "tcp_support_clearance_m": self.tcp_support_clearance_m,
            "tcp_workspace_margin_m": self.tcp_workspace_margin_m,
            "gripper_aabb_base_m": {
                "min": list(self.gripper_aabb_min_base_m),
                "max": list(self.gripper_aabb_max_base_m),
            },
            "sample_counts": {
                "fingers": self.finger_sample_count,
                "body": self.body_sample_count,
            },
            "geometry_scope": (
                "CAD finger inner/outer surfaces, top/bottom vertices and "
                "conservative body envelope; not a full mesh collision check"
            ),
        }


def sample_gripper_envelope_local(
    profile: JawProfile,
    *,
    state_fraction: float,
    finger_y_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative finger/body samples in the finger-root frame.

    Finger samples cover the full CAD profile from the root to the tip.  At
    each axial station, both inner and outer surfaces are sampled across the
    jaw axis at the top and bottom finger planes.  The body is represented by
    all eight corners of the conservative CAD envelope.
    """

    fraction = float(np.clip(float(state_fraction), 0.0, 1.0))
    sample_count = int(finger_y_samples)
    if sample_count < 2:
        raise ValueError("finger_y_samples must be at least two")

    source_depths, inner_half, outer_half = profile.profile_arrays(
        state_fraction=fraction
    )
    # Include the physical finger root even though the extracted CAD profile
    # starts a few millimetres forward of it.
    depths = np.unique(np.concatenate(([0.0], source_depths)))
    z_values = (-0.5 * profile.finger_height_m, 0.5 * profile.finger_height_m)
    finger_points: list[list[float]] = []
    for depth in depths:
        inner = float(np.interp(depth, source_depths, inner_half))
        outer = float(np.interp(depth, source_depths, outer_half))
        # A taper can make the inner surface slightly wider than the outer one
        # at a noisy CAD station.  The union range remains conservative.
        low = min(inner, outer)
        high = max(inner, outer)
        for sign in (-1.0, 1.0):
            for y_abs in np.linspace(low, high, sample_count):
                for z in z_values:
                    finger_points.append([depth, sign * float(y_abs), float(z)])

    body = profile.body_envelope
    body_points = np.asarray(
        [
            [x, y, z]
            for x in (body.rear_x_m, body.root_x_m)
            for y in (-body.half_width_m, body.half_width_m)
            for z in (-body.half_height_m, body.half_height_m)
        ],
        dtype=float,
    )
    return np.asarray(finger_points, dtype=float), body_points


def evaluate_gripper_world_clearance(
    profile: JawProfile,
    contact_transform_base: np.ndarray,
    *,
    width_m: float,
    depth_m: float,
    config: GripperClearanceConfig | None = None,
) -> GripperWorldClearanceResult:
    """Evaluate whole-gripper support clearance for one candidate pose.

    ``width_m`` is the opening that will actually be commanded for the
    approach/contact pose.  It may be narrower than GraspNet's reference width.
    """

    config = config or GripperClearanceConfig()
    transform = np.asarray(contact_transform_base, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("contact_transform_base must be a finite 4x4 matrix")
    rotation = _proper_rotation(transform[:3, :3], "contact rotation")
    contact_x = profile.contact_plane_x_m(depth_m)
    section = profile.section(contact_x, float(width_m))

    root_transform = transform.copy()
    root_transform[:3, :3] = rotation
    root_transform[:3, 3] = transform[:3, 3] + rotation @ np.asarray(
        (-contact_x, 0.0, 0.0), dtype=float
    )
    tcp_transform = profile.graspnet_contact_to_tcp_transform(
        transform[:3, 3],
        rotation,
        float(depth_m),
    )

    finger_local, body_local = sample_gripper_envelope_local(
        profile,
        state_fraction=section.state_fraction,
        finger_y_samples=config.finger_y_samples,
    )
    finger_base = finger_local @ rotation.T + root_transform[:3, 3]
    body_base = body_local @ rotation.T + root_transform[:3, 3]
    all_points = np.vstack((finger_base, body_base))
    labels = np.concatenate(
        (
            np.full(len(finger_base), "fingers", dtype=object),
            np.full(len(body_base), "body", dtype=object),
        )
    )

    support_origin = np.asarray(config.support_origin_base_m, dtype=float)
    support_normal = np.asarray(config.support_normal_base, dtype=float)
    clearances = (all_points - support_origin) @ support_normal
    minimum_index = int(np.argmin(clearances))
    minimum_clearance = float(clearances[minimum_index])
    minimum_point = all_points[minimum_index]

    tcp_base = tcp_transform[:3, 3]
    tcp_support_clearance = float((tcp_base - support_origin) @ support_normal)
    tcp_workspace_margin = float(tcp_base[2] - config.workspace_min_z_m)

    reasons: list[str] = []
    if tcp_workspace_margin < 0.0:
        reasons.append("below_workspace")
    if minimum_clearance < config.minimum_clearance_m:
        reasons.append("table_clearance")

    return GripperWorldClearanceResult(
        feasible=not reasons,
        reason_codes=tuple(reasons),
        support_minimum_clearance_m=minimum_clearance,
        support_minimum_point_base_m=tuple(float(v) for v in minimum_point),
        support_minimum_region=str(labels[minimum_index]),
        tcp_base_m=tuple(float(v) for v in tcp_base),
        tcp_support_clearance_m=tcp_support_clearance,
        tcp_workspace_margin_m=tcp_workspace_margin,
        gripper_aabb_min_base_m=tuple(float(v) for v in np.min(all_points, axis=0)),
        gripper_aabb_max_base_m=tuple(float(v) for v in np.max(all_points, axis=0)),
        finger_sample_count=int(len(finger_base)),
        body_sample_count=int(len(body_base)),
    )


def apply_gripper_world_clearance(
    score: JawGraspScore,
    profile: JawProfile,
    config: GripperClearanceConfig | None = None,
) -> JawGraspScore:
    """Return a fail-closed score with base-frame support clearance applied."""

    clearance = evaluate_gripper_world_clearance(
        profile,
        score.contact_transform,
        width_m=score.candidate.jaw_opening_m,
        depth_m=score.candidate.depth_m,
        config=config,
    )
    reason_codes = tuple(
        dict.fromkeys((*score.reason_codes, *clearance.reason_codes))
    )
    return replace(
        score,
        feasible=bool(score.feasible and clearance.feasible),
        reason_codes=reason_codes,
        world_clearance=clearance.to_dict(),
    )


__all__ = [
    "GripperClearanceConfig",
    "GripperWorldClearanceResult",
    "apply_gripper_world_clearance",
    "evaluate_gripper_world_clearance",
    "load_gripper_clearance_config",
    "sample_gripper_envelope_local",
]
