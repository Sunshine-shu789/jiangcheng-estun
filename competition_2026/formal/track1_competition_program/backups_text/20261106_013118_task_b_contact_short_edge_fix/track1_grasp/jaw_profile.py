"""CAD-derived jaw-profile geometry for the FAE2MR86M-D gripper.

The STEP-derived profile is the authority for jaw opening as a function of
approach depth.  This module deliberately keeps the state model explicit:
the supplied file contains one geometric ``open`` state and one geometric
``closed`` state.  Intermediate states are *linear approximations* and are
reported as such in every result.

Gripper-local axes follow the competition URDF and the STEP report:

* ``+X``: approach direction, finger root toward fingertip;
* ``+Y``: jaw opening axis;
* ``+Z``: finger height axis.

GraspNet's 17-D translation is the network seed/reference point.  It is a
point sampled from the observed object surface (normally the visible contact
surface), not the gripper body/root origin and not the URDF ``grasp_tcp``.
GraspNet's depth class is measured from that reference along local ``+X`` to
the nominal fingertip plane.  This module therefore maps the network contact
frame to the physical TCP using the actual CAD fingertip length and the URDF
``gripper_body_link -> grasp_tcp`` offset.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JawProfileSample:
    depth_m: float
    opening_m: float
    outer_width_m: float

    def __post_init__(self) -> None:
        values = (self.depth_m, self.opening_m, self.outer_width_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("jaw profile samples must be finite")
        if self.depth_m < 0.0 or self.opening_m < 0.0:
            raise ValueError("jaw profile depth/opening cannot be negative")
        if self.outer_width_m < self.opening_m - 1e-6:
            raise ValueError("outer jaw width must not be below inner opening")


@dataclass(frozen=True)
class JawSection:
    """A linearly interpolated jaw cross-section at one approach depth."""

    depth_m: float
    state_fraction: float
    opening_m: float
    closed_opening_m: float
    open_opening_m: float
    outer_width_m: float
    inner_half_width_m: float
    outer_half_width_m: float
    finger_height_m: float
    approximate: bool = True

    @property
    def finger_thickness_m(self) -> float:
        return max(0.0, self.outer_half_width_m - self.inner_half_width_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_m": self.depth_m,
            "state_fraction": self.state_fraction,
            "opening_m": self.opening_m,
            "closed_opening_m": self.closed_opening_m,
            "open_opening_m": self.open_opening_m,
            "outer_width_m": self.outer_width_m,
            "inner_half_width_m": self.inner_half_width_m,
            "outer_half_width_m": self.outer_half_width_m,
            "finger_height_m": self.finger_height_m,
            "finger_thickness_m": self.finger_thickness_m,
            "approximate": self.approximate,
        }


@dataclass(frozen=True)
class JawOpeningCapacity:
    """Maximum open-state capacity anywhere along the CAD finger profile.

    C/D objects can occupy different heights along the fingers, so a single
    nominal-depth sample must not be used as the complete opening envelope.
    The maximum open-state sample is the final fail-closed authority for
    deciding whether a measured object width can fit at all.
    """

    required_opening_m: float
    maximum_open_opening_m: float
    maximum_open_depth_m: float
    closed_opening_at_maximum_m: float
    profile_depth_min_m: float
    profile_depth_max_m: float

    def __post_init__(self) -> None:
        values = (
            self.required_opening_m,
            self.maximum_open_opening_m,
            self.maximum_open_depth_m,
            self.closed_opening_at_maximum_m,
            self.profile_depth_min_m,
            self.profile_depth_max_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("jaw opening capacity values must be finite")
        if self.required_opening_m <= 0.0:
            raise ValueError("required jaw opening must be positive")
        if self.maximum_open_opening_m <= 0.0:
            raise ValueError("maximum CAD jaw opening must be positive")
        if self.profile_depth_min_m > self.profile_depth_max_m:
            raise ValueError("jaw profile depth range is invalid")

    @property
    def fits(self) -> bool:
        """Return whether the required ``width + margin`` fits the envelope."""

        return self.required_opening_m < self.maximum_open_opening_m

    @property
    def opening_margin_m(self) -> float:
        return self.maximum_open_opening_m - self.required_opening_m

    @property
    def closing_span_m(self) -> float:
        return self.maximum_open_opening_m - self.closed_opening_at_maximum_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_opening_m": self.required_opening_m,
            "maximum_open_opening_m": self.maximum_open_opening_m,
            "maximum_open_depth_m": self.maximum_open_depth_m,
            "closed_opening_at_maximum_m": self.closed_opening_at_maximum_m,
            "profile_depth_min_m": self.profile_depth_min_m,
            "profile_depth_max_m": self.profile_depth_max_m,
            "opening_margin_m": self.opening_margin_m,
            "closing_span_m": self.closing_span_m,
            "fits": self.fits,
        }


@dataclass(frozen=True)
class GripperBodyEnvelope:
    """Conservative rigid envelope around the gripper body/root."""

    rear_x_m: float
    root_x_m: float
    half_width_m: float
    half_height_m: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rear_x_m": self.rear_x_m,
            "root_x_m": self.root_x_m,
            "half_width_m": self.half_width_m,
            "half_height_m": self.half_height_m,
        }


class JawProfile:
    """Load and query the FAE2MR86M-D open/closed jaw profiles."""

    def __init__(
        self,
        open_samples: Sequence[JawProfileSample],
        closed_samples: Sequence[JawProfileSample],
        *,
        finger_tip_x_m: float,
        finger_height_m: float,
        body_envelope: GripperBodyEnvelope,
        tcp_x_m: float = 0.050,
        source_path: str = "",
    ) -> None:
        self.open_samples = tuple(open_samples)
        self.closed_samples = tuple(closed_samples)
        if len(self.open_samples) < 2 or len(self.closed_samples) < 2:
            raise ValueError("jaw profile needs at least two open and closed samples")
        self.finger_tip_x_m = float(finger_tip_x_m)
        self.finger_height_m = float(finger_height_m)
        self.body_envelope = body_envelope
        self.tcp_x_m = float(tcp_x_m)
        self.source_path = str(source_path)
        if self.finger_tip_x_m <= 0.0 or self.finger_height_m <= 0.0:
            raise ValueError("finger tip/height must be positive")
        if not 0.0 <= self.tcp_x_m <= self.finger_tip_x_m:
            raise ValueError("tcp_x_m must lie within the finger length")
        self._open_depths = self._validated_depths(self.open_samples, "open")
        self._closed_depths = self._validated_depths(self.closed_samples, "closed")
        self._open_opening = np.asarray(
            [sample.opening_m for sample in self.open_samples], dtype=float
        )
        self._closed_opening = np.asarray(
            [sample.opening_m for sample in self.closed_samples], dtype=float
        )
        self._open_outer = np.asarray(
            [sample.outer_width_m for sample in self.open_samples], dtype=float
        )
        self._closed_outer = np.asarray(
            [sample.outer_width_m for sample in self.closed_samples], dtype=float
        )

    @staticmethod
    def _validated_depths(
        samples: Sequence[JawProfileSample], state: str
    ) -> np.ndarray:
        depths = np.asarray([sample.depth_m for sample in samples], dtype=float)
        if np.any(np.diff(depths) <= 0.0):
            raise ValueError(f"{state} jaw-profile depths must increase strictly")
        return depths

    @classmethod
    def from_files(
        cls,
        profile_path: str | Path,
        dimensions_path: str | Path | None = None,
        *,
        tcp_x_m: float = 0.050,
    ) -> "JawProfile":
        profile_path = Path(profile_path)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("jaw profile JSON must contain an object")
        if payload.get("units") != "m":
            raise ValueError("jaw profile units must be metres")
        open_samples = tuple(
            JawProfileSample(
                float(row["distance_from_finger_root_m"]),
                float(row["opening_m"]),
                float(row["outer_width_m"]),
            )
            for row in payload.get("open", ())
        )
        closed_samples = tuple(
            JawProfileSample(
                float(row["distance_from_finger_root_m"]),
                float(row["opening_m"]),
                float(row["outer_width_m"]),
            )
            for row in payload.get("closed", ())
        )

        tip_x = max(max(sample.depth_m for sample in open_samples),
                    max(sample.depth_m for sample in closed_samples))
        # The STEP-derived dimensions file contains the actual assembly and
        # finger height.  Fall back to the visible profile depth only when the
        # dimensions document is not supplied.
        height = 0.020714688449880447
        body = GripperBodyEnvelope(
            rear_x_m=-0.0917519242201835,
            root_x_m=0.0,
            half_width_m=0.03150665426204577,
            half_height_m=0.03150665426204577,
        )
        if dimensions_path is not None:
            dimensions = json.loads(Path(dimensions_path).read_text(encoding="utf-8"))
            assembly_bbox = dimensions.get("assembly_bbox_local_m", {})
            bbox_min = assembly_bbox.get("min", ())
            bbox_max = assembly_bbox.get("max", ())
            if len(bbox_min) >= 3 and len(bbox_max) >= 3:
                body = GripperBodyEnvelope(
                    rear_x_m=float(bbox_min[0]),
                    root_x_m=0.0,
                    half_width_m=max(
                        abs(float(bbox_min[1])), abs(float(bbox_max[1]))
                    ),
                    half_height_m=max(
                        abs(float(bbox_min[2])), abs(float(bbox_max[2]))
                    ),
                )
            finger_geometry = dimensions.get("finger_geometry", {})
            if "height_m" in finger_geometry:
                height = float(finger_geometry["height_m"])
        return cls(
            open_samples,
            closed_samples,
            finger_tip_x_m=tip_x,
            finger_height_m=height,
            body_envelope=body,
            tcp_x_m=tcp_x_m,
            source_path=str(profile_path),
        )

    def _state_at_depth(
        self,
        depth_m: float,
        *,
        state: str,
    ) -> tuple[float, float]:
        depth = float(depth_m)
        if not math.isfinite(depth):
            raise ValueError("jaw depth must be finite")
        if state == "open":
            depths, opening, outer = (
                self._open_depths,
                self._open_opening,
                self._open_outer,
            )
        elif state == "closed":
            depths, opening, outer = (
                self._closed_depths,
                self._closed_opening,
                self._closed_outer,
            )
        else:
            raise ValueError(f"unknown jaw state: {state}")
        if depth < float(depths[0]) - 1e-9 or depth > float(depths[-1]) + 1e-9:
            raise ValueError(
                f"depth {depth:.6f} m is outside {state} profile range "
                f"[{depths[0]:.6f}, {depths[-1]:.6f}]"
            )
        return (
            float(np.interp(depth, depths, opening)),
            float(np.interp(depth, depths, outer)),
        )

    def section(self, depth_m: float, opening_m: float) -> JawSection:
        """Return a state consistent with a candidate inner opening."""
        depth = float(depth_m)
        opening = float(opening_m)
        if not math.isfinite(opening) or opening < 0.0:
            raise ValueError("jaw opening must be finite and non-negative")
        open_width, open_outer = self._state_at_depth(depth, state="open")
        closed_width, closed_outer = self._state_at_depth(depth, state="closed")
        span = open_width - closed_width
        if span <= 1e-9:
            fraction = 0.0
        else:
            fraction = (opening - closed_width) / span
        effective_fraction = float(np.clip(fraction, 0.0, 1.0))
        outer_width = closed_outer + effective_fraction * (open_outer - closed_outer)
        return JawSection(
            depth_m=depth,
            state_fraction=effective_fraction,
            opening_m=opening,
            closed_opening_m=closed_width,
            open_opening_m=open_width,
            outer_width_m=float(outer_width),
            inner_half_width_m=0.5 * opening,
            outer_half_width_m=0.5 * float(outer_width),
            finger_height_m=self.finger_height_m,
            approximate=True,
        )

    def width_feasibility(
        self,
        depth_m: float,
        opening_m: float,
        *,
        opening_reserve_m: float = 0.002,
        minimum_closure_margin_m: float = 0.001,
    ) -> tuple[bool, list[str]]:
        """Check whether the CAD profile can reach the requested opening."""
        open_width, _ = self._state_at_depth(depth_m, state="open")
        closed_width, _ = self._state_at_depth(depth_m, state="closed")
        reasons: list[str] = []
        if opening_m > open_width - opening_reserve_m:
            reasons.append("exceeds_profile_opening")
        if opening_m < closed_width + minimum_closure_margin_m:
            reasons.append("below_profile_closed_opening")
        return not reasons, reasons

    def opening_capacity(self, required_opening_m: float) -> JawOpeningCapacity:
        """Return the full-profile capacity for a required open aperture.

        This deliberately checks the largest open-state opening anywhere from
        finger root to fingertip.  It is a feasibility envelope, not a claim
        that every depth can accept the object.  Callers that need an exact
        contact depth must still use :meth:`width_feasibility` at that depth.
        """

        required = float(required_opening_m)
        if not math.isfinite(required) or required <= 0.0:
            raise ValueError("required jaw opening must be finite and positive")
        maximum_index = int(np.argmax(self._open_opening))
        maximum_open_width = float(self._open_opening[maximum_index])
        maximum_open_depth = float(self._open_depths[maximum_index])
        closed_width, _ = self._state_at_depth(
            maximum_open_depth,
            state="closed",
        )
        return JawOpeningCapacity(
            required_opening_m=required,
            maximum_open_opening_m=maximum_open_width,
            maximum_open_depth_m=maximum_open_depth,
            closed_opening_at_maximum_m=closed_width,
            profile_depth_min_m=min(
                float(self._open_depths[0]),
                float(self._closed_depths[0]),
            ),
            profile_depth_max_m=max(
                float(self._open_depths[-1]),
                float(self._closed_depths[-1]),
            ),
        )

    def contact_plane_x_m(self, graspnet_depth_m: float) -> float:
        """Return the reference-plane position from the physical finger root.

        GraspNet's selected depth class is the distance from its seed/reference
        point to the nominal fingertip plane along local ``+X``.  In the CAD
        finger coordinate this places the reference plane at::

            x_contact = finger_tip_x - graspnet_depth

        A depth of one fingertip length would place the reference at the finger
        root; shallower network depths move the reference farther forward.
        """

        depth = float(graspnet_depth_m)
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError("GraspNet depth must be positive and finite")
        if depth > self.finger_tip_x_m + 1e-9:
            raise ValueError(
                "GraspNet depth exceeds the physical finger length "
                f"({depth:.6f} m > {self.finger_tip_x_m:.6f} m)"
            )
        return float(self.finger_tip_x_m - depth)

    def tcp_offset_from_contact_m(
        self,
        graspnet_depth_m: float,
    ) -> np.ndarray:
        """Return the URDF TCP position in the GraspNet contact frame.

        The contact frame is the predicted rotation/translation frame.  Its
        origin is GraspNet's seed/reference point, local ``+X`` points along
        the approach, local ``+Y`` is jaw opening and local ``+Z`` is finger
        height.  The returned vector is expressed in that local frame.
        """

        contact_x = self.contact_plane_x_m(graspnet_depth_m)
        # ``tcp_x_m`` is measured from the physical finger root.  Convert that
        # point back into the contact frame whose origin is at contact_x.
        return np.asarray(
            [self.tcp_x_m - contact_x, 0.0, 0.0],
            dtype=float,
        )

    def graspnet_contact_to_tcp_transform(
        self,
        translation_m: Sequence[float],
        rotation: Sequence[float] | Sequence[Sequence[float]],
        graspnet_depth_m: float,
    ) -> np.ndarray:
        """Map a GraspNet contact frame to a physical ``grasp_tcp`` frame.

        Parameters
        ----------
        translation_m:
            GraspNet 17-D translation in the source point-cloud frame.
        rotation:
            Either a row-major 3x3 matrix or a length-9 flat array mapping the
            GraspNet local frame to the point-cloud frame.
        graspnet_depth_m:
            Selected GraspNet depth class in metres.

        Returns
        -------
        numpy.ndarray
            A 4x4 homogeneous transform ``T_source_tcp`` in the same source
            frame as ``translation_m``.
        """

        translation = np.asarray(translation_m, dtype=float)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation_m must contain three finite values")
        rotation_array = np.asarray(rotation, dtype=float)
        if rotation_array.shape == (9,):
            rotation_array = rotation_array.reshape(3, 3)
        if rotation_array.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 matrix or nine flat values")
        if not np.all(np.isfinite(rotation_array)):
            raise ValueError("rotation contains non-finite values")
        if not np.allclose(
            rotation_array.T @ rotation_array,
            np.eye(3),
            atol=2e-5,
            rtol=0.0,
        ):
            raise ValueError("rotation must be orthonormal")
        if abs(float(np.linalg.det(rotation_array)) - 1.0) > 2e-5:
            raise ValueError("rotation determinant must be +1")

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation_array
        transform[:3, 3] = translation + (
            rotation_array @ self.tcp_offset_from_contact_m(graspnet_depth_m)
        )
        return transform

    def tcp_to_graspnet_contact_transform(
        self,
        tcp_transform: np.ndarray,
        graspnet_depth_m: float,
    ) -> np.ndarray:
        """Inverse of :meth:`graspnet_contact_to_tcp_transform`."""

        transform = np.asarray(tcp_transform, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("tcp_transform must be 4x4")
        if not np.all(np.isfinite(transform)):
            raise ValueError("tcp_transform contains non-finite values")
        rotation = transform[:3, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=2e-5,
            rtol=0.0,
        ):
            raise ValueError("tcp_transform rotation must be orthonormal")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 2e-5:
            raise ValueError("tcp_transform rotation determinant must be +1")
        offset = self.tcp_offset_from_contact_m(graspnet_depth_m)
        contact = transform.copy()
        contact[:3, 3] = transform[:3, 3] - rotation @ offset
        return contact

    def profile_arrays(
        self,
        *,
        state_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return depth, inner half-width, and outer half-width arrays.

        The two source states have different sampling positions, so both are
        evaluated over their union and linearly blended.  Values outside the
        closed-state range use its nearest endpoint; closed overlap at the tip
        is clamped to zero opening by the source data.
        """
        fraction = float(np.clip(state_fraction, 0.0, 1.0))
        depths = np.unique(
            np.concatenate((self._open_depths, self._closed_depths))
        )
        open_opening = np.interp(depths, self._open_depths, self._open_opening)
        closed_opening = np.interp(
            depths, self._closed_depths, self._closed_opening
        )
        open_outer = np.interp(depths, self._open_depths, self._open_outer)
        closed_outer = np.interp(depths, self._closed_depths, self._closed_outer)
        opening = closed_opening + fraction * (open_opening - closed_opening)
        outer = closed_outer + fraction * (open_outer - closed_outer)
        return depths, 0.5 * opening, 0.5 * outer

    def sampled_local_finger_surfaces(
        self,
        *,
        state_fraction: float,
        z_values: Sequence[float] = (-0.010, 0.0, 0.010),
    ) -> np.ndarray:
        """Return sampled CAD-profile inner surfaces for diagnostics/tests."""
        depths, inner_half, _ = self.profile_arrays(state_fraction=state_fraction)
        points: list[list[float]] = []
        for depth, half in zip(depths, inner_half):
            for sign in (-1.0, 1.0):
                for z in z_values:
                    points.append([float(depth), float(sign * half), float(z)])
        return np.asarray(points, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "units": "m",
            "x_definition": "distance forward from the finger root along gripper +X",
            "finger_tip_x_m": self.finger_tip_x_m,
            "finger_height_m": self.finger_height_m,
            "tcp_x_m": self.tcp_x_m,
            "graspnet_reference_semantics": (
                "object-surface seed/reference point; depth runs from this "
                "point to the nominal fingertip plane along local +X; not the "
                "URDF grasp_tcp and not the gripper body origin"
            ),
            "tcp_mapping": (
                "x_tcp_in_contact_frame = tcp_x_m - "
                "(finger_tip_x_m - graspnet_depth_m)"
            ),
            "body_envelope": self.body_envelope.to_dict(),
            "states": {
                "open": [sample.__dict__ for sample in self.open_samples],
                "closed": [sample.__dict__ for sample in self.closed_samples],
            },
            "intermediate_states": "linear approximation; not measured",
        }
