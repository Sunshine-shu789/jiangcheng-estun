#!/usr/bin/env python3
"""Build an offline consensus report from multi-frame grasp candidates.

Only candidates that were already final-feasible, NMS-kept and IK-successful
are considered.  Consensus is an additional fail-closed filter; it never
relaxes the contact, collision, clearance, NMS or IK acceptance thresholds and
it never authorizes robot or gripper motion.

The report is computed entirely from existing ``offline_candidates.json``
files.  This module does not import ROS and cannot command hardware.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from track1_grasp.jaw_grasp_scoring import jaw_symmetric_rotation_distance


FRAME_DIR_RE = re.compile(r"^frame_(\d+)$")
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "offline_grasp_consensus.yaml"
)
_EPS = 1e-9

# The source diagnostic must remain non-actuating and non-authorized.  A
# mismatch makes the whole consensus input invalid rather than being ignored.
_SAFETY_EXPECTATIONS = {
    "offline_only": True,
    "motion_attempted": False,
    "hardware_connected": False,
    "motion_authorized": False,
}


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number



def _vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    return array


def _transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} must have a homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2e-5:
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def _frame_sort_key(path: Path) -> tuple[int, str]:
    match = FRAME_DIR_RE.match(path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


@dataclass(frozen=True)
class ConsensusConfig:
    """Fail-closed multi-frame pose-consensus settings."""

    name: str = "offline_multiframe_grasp_consensus_v1"
    validated: bool = False
    minimum_distinct_frames: int = 4
    translation_tolerance_m: float = 0.025
    jaw_rotation_tolerance_deg: float = 15.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if (
            isinstance(self.minimum_distinct_frames, bool)
            or not isinstance(self.minimum_distinct_frames, int)
            or self.minimum_distinct_frames < 2
        ):
            raise ValueError("minimum_distinct_frames must be an integer >= 2")
        translation = _finite_float(
            self.translation_tolerance_m, "translation_tolerance_m"
        )
        rotation = _finite_float(
            self.jaw_rotation_tolerance_deg, "jaw_rotation_tolerance_deg"
        )
        if translation <= 0.0:
            raise ValueError("translation_tolerance_m must be > 0")
        if rotation <= 0.0 or rotation > 180.0:
            raise ValueError("jaw_rotation_tolerance_deg must be in (0, 180]")

    @property
    def jaw_rotation_tolerance_rad(self) -> float:
        return math.radians(float(self.jaw_rotation_tolerance_deg))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConsensusConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("consensus config must be a mapping")
        nested = payload.get("consensus")
        data = nested if isinstance(nested, Mapping) else payload
        defaults = cls()
        return cls(
            name=str(payload.get("name", data.get("name", defaults.name))),
            validated=bool(payload.get("validated", data.get("validated", False))),
            minimum_distinct_frames=int(
                data.get("minimum_distinct_frames", defaults.minimum_distinct_frames)
            ),
            translation_tolerance_m=float(
                data.get(
                    "translation_tolerance_m", defaults.translation_tolerance_m
                )
            ),
            jaw_rotation_tolerance_deg=float(
                data.get(
                    "jaw_rotation_tolerance_deg",
                    defaults.jaw_rotation_tolerance_deg,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "validated": bool(self.validated),
            "minimum_distinct_frames": int(self.minimum_distinct_frames),
            "translation_tolerance_m": float(self.translation_tolerance_m),
            "jaw_rotation_tolerance_deg": float(self.jaw_rotation_tolerance_deg),
        }


def load_consensus_config(path: str | Path = DEFAULT_CONFIG) -> ConsensusConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("consensus config must contain a mapping")
    return ConsensusConfig.from_mapping(payload)


@dataclass(frozen=True)
class ConsensusCandidate:
    frame: int
    candidate_id: str
    quality: float
    reference_width_m: float
    selected_opening_m: float
    tcp_base_m: tuple[float, float, float]
    rotation_base: np.ndarray
    support_minimum_clearance_m: float | None
    tcp_workspace_margin_m: float | None
    ik_position_error_m: float | None
    ik_orientation_error_rad: float | None
    ik_max_joint_delta_deg: float | None
    joints_rad: tuple[float, ...] | None
    source_path: str

    @property
    def label(self) -> str:
        return f"frame_{self.frame:02d}/{self.candidate_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": int(self.frame),
            "candidate_id": self.candidate_id,
            "quality": float(self.quality),
            "reference_width_m": float(self.reference_width_m),
            "selected_opening_m": float(self.selected_opening_m),
            "tcp_base_m": list(self.tcp_base_m),
            "rotation_base": np.asarray(self.rotation_base, dtype=float).tolist(),
            "support_minimum_clearance_m": self.support_minimum_clearance_m,
            "tcp_workspace_margin_m": self.tcp_workspace_margin_m,
            "ik_position_error_m": self.ik_position_error_m,
            "ik_orientation_error_rad": self.ik_orientation_error_rad,
            "ik_max_joint_delta_deg": self.ik_max_joint_delta_deg,
            "joints_rad": None if self.joints_rad is None else list(self.joints_rad),
            "source_path": self.source_path,
        }


def _candidate_from_record(
    frame: int,
    record: Mapping[str, Any],
    source_path: Path,
) -> ConsensusCandidate:
    candidate_id = str(record.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("candidate_id is empty")

    score = record.get("score_camera")
    graspnet = record.get("graspnet")
    world = record.get("world_clearance")
    transforms = record.get("transforms")
    ik = record.get("ik")
    if not isinstance(score, Mapping):
        raise ValueError("score_camera is missing")
    if not isinstance(graspnet, Mapping):
        raise ValueError("graspnet is missing")
    if not isinstance(world, Mapping):
        raise ValueError("world_clearance is missing")
    if not isinstance(transforms, Mapping):
        raise ValueError("transforms is missing")
    if not isinstance(ik, Mapping):
        raise ValueError("ik is missing")

    t_base_tcp = _transform(transforms.get("T_base_tcp"), "T_base_tcp")
    tcp_from_transform = t_base_tcp[:3, 3]
    tcp_from_world = _vector3(world.get("tcp_base_m"), "world_clearance.tcp_base_m")
    if float(np.linalg.norm(tcp_from_transform - tcp_from_world)) > 1e-5:
        raise ValueError("T_base_tcp and world_clearance.tcp_base_m disagree")

    if ik.get("success") is not True:
        raise ValueError("ik.success must be true")
    if ik.get("at_joint_limit") is not False:
        raise ValueError("ik.at_joint_limit must be explicitly false")
    raw_joints = ik.get("joints_rad")
    joint_array = np.asarray(raw_joints, dtype=float).reshape(-1)
    if joint_array.shape != (6,) or not np.all(np.isfinite(joint_array)):
        raise ValueError("ik.joints_rad must contain six finite values")
    joints = tuple(float(value) for value in joint_array)

    quality = _finite_float(score.get("quality"), "score_camera.quality")
    reference_width = _finite_float(graspnet.get("width_m"), "graspnet.width_m")
    selected_opening = _finite_float(
        score.get("selected_opening_m"), "score_camera.selected_opening_m"
    )
    support_clearance = _finite_float(
        world.get("support_minimum_clearance_m"),
        "world_clearance.support_minimum_clearance_m",
    )
    workspace_margin = _finite_float(
        world.get("tcp_workspace_margin_m"),
        "world_clearance.tcp_workspace_margin_m",
    )
    ik_position_error = _finite_float(
        ik.get("position_error_m"), "ik.position_error_m"
    )
    ik_orientation_error = _finite_float(
        ik.get("orientation_error_rad"), "ik.orientation_error_rad"
    )
    ik_max_joint_delta = _finite_float(
        ik.get("max_joint_delta_deg"), "ik.max_joint_delta_deg"
    )
    if not 0.0 <= quality <= 1.0:
        raise ValueError("score_camera.quality must be in [0, 1]")
    if reference_width <= 0.0 or selected_opening <= 0.0:
        raise ValueError("GraspNet width and selected opening must be > 0")
    if support_clearance < 0.0 or workspace_margin < 0.0:
        raise ValueError("clearance and workspace margin must be >= 0")
    if (
        ik_position_error < 0.0
        or ik_orientation_error < 0.0
        or ik_max_joint_delta < 0.0
    ):
        raise ValueError("IK diagnostics must be >= 0")

    return ConsensusCandidate(
        frame=int(frame),
        candidate_id=candidate_id,
        quality=quality,
        reference_width_m=reference_width,
        selected_opening_m=selected_opening,
        tcp_base_m=tuple(float(value) for value in tcp_from_transform),
        rotation_base=np.asarray(t_base_tcp[:3, :3], dtype=float).copy(),
        support_minimum_clearance_m=support_clearance,
        tcp_workspace_margin_m=workspace_margin,
        ik_position_error_m=ik_position_error,
        ik_orientation_error_rad=ik_orientation_error,
        ik_max_joint_delta_deg=ik_max_joint_delta,
        joints_rad=joints,
        source_path=str(source_path),
    )


def _scan_batch(
    batch_root: Path,
) -> tuple[
    list[ConsensusCandidate],
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
]:
    frame_dirs = sorted(
        (
            path
            for path in batch_root.iterdir()
            if path.is_dir() and FRAME_DIR_RE.match(path.name)
        ),
        key=_frame_sort_key,
    )
    candidates: list[ConsensusCandidate] = []
    frame_results: list[dict[str, Any]] = []
    errors: list[str] = []
    totals = {
        "frames_discovered": len(frame_dirs),
        "frames_loaded": 0,
        "safety_clean_frames": 0,
        "integrity_clean_frames": 0,
        "nms_kept_candidates": 0,
        "ik_evaluated_candidates": 0,
        "ik_success_candidates": 0,
        "ik_failed_candidates": 0,
    }

    for frame_dir in frame_dirs:
        match = FRAME_DIR_RE.match(frame_dir.name)
        assert match is not None
        frame = int(match.group(1))
        source_path = frame_dir / "offline_candidates.json"
        frame_errors: list[str] = []
        safety_clean = True
        integrity_clean = True
        result: dict[str, Any] = {
            "frame": frame,
            "source_path": str(source_path),
            "loaded": False,
            "safety_clean": False,
            "integrity_clean": False,
            "nms_kept_candidates": 0,
            "ik_evaluated_candidates": 0,
            "ik_success_candidates": 0,
            "ik_failed_candidates": 0,
            "eligible_candidates": 0,
            "errors": frame_errors,
        }
        frame_results.append(result)

        if not source_path.is_file():
            frame_errors.append("offline_candidates.json is missing")
            errors.extend(f"frame_{frame:02d}: {item}" for item in frame_errors)
            continue

        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            frame_errors.append(f"cannot read JSON: {type(exc).__name__}: {exc}")
            errors.extend(f"frame_{frame:02d}: {item}" for item in frame_errors)
            continue
        if not isinstance(payload, Mapping):
            frame_errors.append("JSON root must be an object")
            errors.extend(f"frame_{frame:02d}: {item}" for item in frame_errors)
            continue
        result["loaded"] = True
        totals["frames_loaded"] += 1

        for key, expected in _SAFETY_EXPECTATIONS.items():
            if payload.get(key) is not expected:
                safety_clean = False
                frame_errors.append(
                    f"safety flag {key} must be {str(expected).lower()}, "
                    f"got {payload.get(key)!r}"
                )
        result["safety_clean"] = safety_clean
        if safety_clean:
            totals["safety_clean_frames"] += 1

        summary = payload.get("summary")
        raw_candidates = payload.get("candidates")
        if not isinstance(summary, Mapping):
            integrity_clean = False
            frame_errors.append("summary is missing")
            raw_candidates = []
        if not isinstance(raw_candidates, list):
            integrity_clean = False
            frame_errors.append("candidates must be a list")
            raw_candidates = []

        nms_records: list[Mapping[str, Any]] = []
        nms_ids: set[str] = set()
        for index, record in enumerate(raw_candidates):
            if not isinstance(record, Mapping):
                integrity_clean = False
                frame_errors.append(f"candidate index {index} is not an object")
                continue
            if record.get("nms_kept") is True:
                candidate_id = str(record.get("candidate_id") or "")
                if not candidate_id:
                    integrity_clean = False
                    frame_errors.append(
                        f"NMS-kept candidate index {index} has no candidate_id"
                    )
                elif candidate_id in nms_ids:
                    integrity_clean = False
                    frame_errors.append(
                        f"duplicate NMS-kept candidate_id {candidate_id}"
                    )
                else:
                    nms_ids.add(candidate_id)
                nms_records.append(record)

        ik_evaluated = 0
        ik_success = 0
        for record in nms_records:
            candidate_id = str(record.get("candidate_id") or "<missing-id>")
            if record.get("final_feasible") is not True:
                integrity_clean = False
                frame_errors.append(
                    f"NMS-kept candidate {candidate_id} is not final_feasible"
                )
            ik = record.get("ik")
            if not isinstance(ik, Mapping):
                integrity_clean = False
                frame_errors.append(
                    f"NMS-kept candidate {candidate_id} has no IK result"
                )
                continue
            ik_evaluated += 1
            if ik.get("success") is True:
                ik_success += 1
            elif ik.get("success") is False:
                totals["ik_failed_candidates"] += 1
            else:
                integrity_clean = False
                frame_errors.append(
                    f"NMS-kept candidate {candidate_id} has non-boolean IK success"
                )

        try:
            expected_nms = int(summary.get("nms_survivor_count"))
            expected_ik = int(summary.get("ik_evaluated_count"))
            expected_success = int(summary.get("ik_success_count"))
        except (TypeError, ValueError):
            integrity_clean = False
            expected_nms = expected_ik = expected_success = -1
            frame_errors.append("summary IK/NMS counts must be integers")
        if expected_nms != len(nms_records):
            integrity_clean = False
            frame_errors.append(
                "summary.nms_survivor_count does not match candidate records "
                f"({expected_nms} != {len(nms_records)})"
            )
        if expected_ik != ik_evaluated:
            integrity_clean = False
            frame_errors.append(
                "summary.ik_evaluated_count does not match candidate records "
                f"({expected_ik} != {ik_evaluated})"
            )
        if expected_success != ik_success:
            integrity_clean = False
            frame_errors.append(
                "summary.ik_success_count does not match candidate records "
                f"({expected_success} != {ik_success})"
            )

        result.update(
            {
                "nms_kept_candidates": len(nms_records),
                "ik_evaluated_candidates": ik_evaluated,
                "ik_success_candidates": ik_success,
                "ik_failed_candidates": len(nms_records) - ik_success,
            }
        )
        totals["nms_kept_candidates"] += len(nms_records)
        totals["ik_evaluated_candidates"] += ik_evaluated
        totals["ik_success_candidates"] += ik_success

        integrity_counted = integrity_clean
        if integrity_counted:
            totals["integrity_clean_frames"] += 1
        result["integrity_clean"] = integrity_clean
        errors.extend(f"frame_{frame:02d}: {item}" for item in frame_errors)

        # A frame with malformed safety or accounting metadata contributes no
        # candidates.  This prevents partial/unsafe input from forming a
        # seemingly valid consensus.
        if not safety_clean or not integrity_clean:
            continue

        for record in nms_records:
            ik = record.get("ik")
            if not isinstance(ik, Mapping) or ik.get("success") is not True:
                continue
            record_id = str(record.get("candidate_id") or "")
            try:
                candidate = _candidate_from_record(frame, record, source_path)
            except (TypeError, ValueError) as exc:
                if integrity_counted:
                    totals["integrity_clean_frames"] -= 1
                    integrity_counted = False
                integrity_clean = False
                result["integrity_clean"] = False
                message = f"NMS/IK-success candidate {record_id}: {exc}"
                frame_errors.append(message)
                errors.append(f"frame_{frame:02d}: {message}")
                continue
            candidates.append(candidate)
            result["eligible_candidates"] += 1

    return candidates, frame_results, errors, totals


def _pose_distances(first: ConsensusCandidate, second: ConsensusCandidate) -> tuple[float, float]:
    translation = float(
        np.linalg.norm(
            np.asarray(first.tcp_base_m) - np.asarray(second.tcp_base_m)
        )
    )
    rotation = float(
        jaw_symmetric_rotation_distance(first.rotation_base, second.rotation_base)
    )
    return translation, rotation


def _clusters_compatible(
    first: Sequence[ConsensusCandidate],
    second: Sequence[ConsensusCandidate],
    config: ConsensusConfig,
) -> tuple[bool, float, float, float]:
    max_translation = 0.0
    max_rotation = 0.0
    for left in first:
        for right in second:
            translation, rotation = _pose_distances(left, right)
            max_translation = max(max_translation, translation)
            max_rotation = max(max_rotation, rotation)
    compatible = (
        max_translation <= config.translation_tolerance_m + _EPS
        and max_rotation <= config.jaw_rotation_tolerance_rad + _EPS
    )
    normalized = max(
        max_translation / config.translation_tolerance_m,
        max_rotation / config.jaw_rotation_tolerance_rad,
    )
    return compatible, max_translation, max_rotation, normalized


def _pose_medoid(members: Sequence[ConsensusCandidate]) -> ConsensusCandidate:
    def key(candidate: ConsensusCandidate) -> tuple[float, float, int, str]:
        max_translation = 0.0
        max_rotation = 0.0
        for other in members:
            translation, rotation = _pose_distances(candidate, other)
            max_translation = max(max_translation, translation)
            max_rotation = max(max_rotation, rotation)
        return (
            max_translation / 0.001,
            max_rotation / math.radians(1.0),
            -candidate.quality,
            candidate.candidate_id,
        )

    return min(members, key=key)


def cluster_candidates(
    candidates: Sequence[ConsensusCandidate],
    config: ConsensusConfig,
    *,
    input_valid: bool = True,
) -> list[dict[str, Any]]:
    """Cluster candidates using complete-linkage pose compatibility.

    Complete linkage is deliberate: connected-component chaining could put two
    grasps in one consensus even when their outer poses exceed the configured
    tolerance.  Every pair in a reported cluster is within both gates.
    """

    working: list[list[ConsensusCandidate]] = [[candidate] for candidate in candidates]
    while True:
        best: tuple[tuple[float, int, int], int, int] | None = None
        for first_index in range(len(working)):
            for second_index in range(first_index + 1, len(working)):
                compatible, _, _, normalized = _clusters_compatible(
                    working[first_index], working[second_index], config
                )
                if not compatible:
                    continue
                key = (normalized, first_index, second_index)
                if best is None or key < best[0]:
                    best = (key, first_index, second_index)
        if best is None:
            break
        _, first_index, second_index = best
        working[first_index] = sorted(
            working[first_index] + working[second_index],
            key=lambda item: (item.frame, item.candidate_id),
        )
        del working[second_index]

    clusters: list[dict[str, Any]] = []
    for members in working:
        members = sorted(members, key=lambda item: (item.frame, item.candidate_id))
        frames = sorted({item.frame for item in members})
        centroid = np.mean(
            np.asarray([item.tcp_base_m for item in members], dtype=float), axis=0
        )
        representative = _pose_medoid(members)
        from_centroid = [
            float(np.linalg.norm(np.asarray(item.tcp_base_m) - centroid))
            for item in members
        ]
        pairwise_translation = 0.0
        pairwise_rotation = 0.0
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                translation, rotation = _pose_distances(left, right)
                pairwise_translation = max(pairwise_translation, translation)
                pairwise_rotation = max(pairwise_rotation, rotation)
        representative_translation = {
            item.label: _pose_distances(item, representative)[0] for item in members
        }
        representative_rotation = {
            item.label: _pose_distances(item, representative)[1] for item in members
        }
        accepted = input_valid and len(frames) >= config.minimum_distinct_frames
        rejection_reasons: list[str] = []
        if not input_valid:
            rejection_reasons.append("input_invalid_fail_closed")
        if len(frames) < config.minimum_distinct_frames:
            rejection_reasons.append("insufficient_distinct_frames")

        clearances = [
            item.support_minimum_clearance_m
            for item in members
            if item.support_minimum_clearance_m is not None
        ]
        workspace_margins = [
            item.tcp_workspace_margin_m
            for item in members
            if item.tcp_workspace_margin_m is not None
        ]
        ik_position_errors = [
            item.ik_position_error_m
            for item in members
            if item.ik_position_error_m is not None
        ]
        ik_orientation_errors = [
            item.ik_orientation_error_rad
            for item in members
            if item.ik_orientation_error_rad is not None
        ]
        joint_deltas = [
            item.ik_max_joint_delta_deg
            for item in members
            if item.ik_max_joint_delta_deg is not None
        ]
        qualities = [item.quality for item in members]
        openings = [item.selected_opening_m for item in members]
        widths = [item.reference_width_m for item in members]

        clusters.append(
            {
                "cluster_id": "",
                "accepted": bool(accepted),
                "rejection_reasons": rejection_reasons,
                "candidate_count": len(members),
                "distinct_frame_count": len(frames),
                "frames": frames,
                "candidate_labels": [item.label for item in members],
                "median_quality": float(statistics.median(qualities)),
                "minimum_quality": float(min(qualities)),
                "maximum_quality": float(max(qualities)),
                "selected_opening_range_m": [float(min(openings)), float(max(openings))],
                "reference_width_range_m": [float(min(widths)), float(max(widths))],
                "minimum_support_clearance_m": (
                    None if not clearances else float(min(clearances))
                ),
                "minimum_tcp_workspace_margin_m": (
                    None if not workspace_margins else float(min(workspace_margins))
                ),
                "maximum_ik_position_error_m": (
                    None if not ik_position_errors else float(max(ik_position_errors))
                ),
                "maximum_ik_orientation_error_rad": (
                    None
                    if not ik_orientation_errors
                    else float(max(ik_orientation_errors))
                ),
                "maximum_ik_joint_delta_deg": (
                    None if not joint_deltas else float(max(joint_deltas))
                ),
                "centroid_tcp_base_m": [float(value) for value in centroid],
                "maximum_translation_from_centroid_m": float(max(from_centroid)),
                "maximum_pairwise_translation_m": float(pairwise_translation),
                "maximum_pairwise_jaw_rotation_deg": float(
                    math.degrees(pairwise_rotation)
                ),
                "representative_candidate": representative.label,
                "representative_tcp_base_m": list(representative.tcp_base_m),
                "representative_rotation_base": representative.rotation_base.tolist(),
                "representative_joints_rad": (
                    None
                    if representative.joints_rad is None
                    else list(representative.joints_rad)
                ),
                "members": [
                    {
                        **item.to_dict(),
                        "is_representative": item.label == representative.label,
                        "translation_from_centroid_m": float(
                            np.linalg.norm(np.asarray(item.tcp_base_m) - centroid)
                        ),
                        "translation_from_representative_m": float(
                            representative_translation[item.label]
                        ),
                        "jaw_rotation_from_representative_deg": float(
                            math.degrees(
                                representative_rotation[item.label]
                            )
                        ),
                    }
                    for item in members
                ],
            }
        )

    clusters.sort(
        key=lambda item: (
            not item["accepted"],
            -item["distinct_frame_count"],
            -item["median_quality"],
            -item["candidate_count"],
            item["frames"],
            item["candidate_labels"],
        )
    )
    for index, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = f"consensus_{index:03d}"
    return clusters


def summarize_consensus(
    batch_root: str | Path,
    config: ConsensusConfig | None = None,
) -> dict[str, Any]:
    batch_root = Path(batch_root)
    if not batch_root.is_dir():
        raise ValueError(f"batch root is not a directory: {batch_root}")
    resolved_config = config or load_consensus_config()
    candidates, frames, errors, totals = _scan_batch(batch_root)
    input_valid = not errors
    clusters = cluster_candidates(
        candidates, resolved_config, input_valid=input_valid
    )
    accepted = [item for item in clusters if item["accepted"]]
    if not input_valid:
        status = "input_invalid_fail_closed"
    elif accepted:
        status = "consensus_available_offline_only"
    else:
        status = "no_consensus_fail_closed"

    return {
        "schema_version": 1,
        "kind": "offline_multiframe_grasp_consensus",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "offline_only": True,
        "motion_attempted": False,
        "hardware_connected": False,
        "motion_authorized": False,
        "quality_validated": False,
        "batch_root": str(batch_root),
        "config": resolved_config.to_dict(),
        "input_valid": input_valid,
        "consensus_reached": bool(input_valid and accepted),
        "counts": {
            **totals,
            "eligible_nms_final_ik_success_candidates": len(candidates),
            "clusters": len(clusters),
            "accepted_clusters": len(accepted),
            "maximum_distinct_frames_in_cluster": max(
                (item["distinct_frame_count"] for item in clusters), default=0
            ),
        },
        "frames": frames,
        "clusters": clusters,
        "errors": errors,
        "caveats": [
            "Consensus only uses candidates that already passed final CAD/clearance checks, NMS and the existing strict IK acceptance gate.",
            "The pose gate is based on TCP base pose with 180-degree parallel-jaw symmetry; it does not prove the same contact patch or joint branch.",
            "A consensus cluster is an offline stability diagnostic, not a grasp-success probability, motion plan or execution authorization.",
            "No contact, collision, clearance, workspace, NMS or IK threshold was relaxed.",
            "No camera, robot, controller or gripper was connected and no motion or gripper command was attempted.",
        ],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def render_markdown(summary: Mapping[str, Any]) -> str:
    config = summary["config"]
    lines = [
        "# 离线多帧抓取共识报告",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 批次：`{summary['batch_root']}`",
        f"- 输入有效：`{str(summary['input_valid']).lower()}`",
        f"- 达成共识：`{str(summary['consensus_reached']).lower()}`",
        f"- 门禁：至少 `{config['minimum_distinct_frames']}` 个独立帧，"
        f"TCP 平移容差 `{1000.0 * config['translation_tolerance_m']:.1f} mm`，"
        f"夹爪对称旋转容差 `{config['jaw_rotation_tolerance_deg']:.1f}°`",
        "- 安全边界：`offline_only=true`、`motion_attempted=false`、"
        "`hardware_connected=false`、`motion_authorized=false`",
        "",
        "## 输入统计",
        "",
        f"- 发现/成功读取帧：`{summary['counts']['frames_discovered']}` / "
        f"`{summary['counts']['frames_loaded']}`",
        f"- 安全标记通过帧：`{summary['counts']['safety_clean_frames']}`",
        f"- 计数完整性通过帧：`{summary['counts']['integrity_clean_frames']}`",
        f"- NMS 保留候选：`{summary['counts']['nms_kept_candidates']}`",
        f"- 实际 IK 评估：`{summary['counts']['ik_evaluated_candidates']}`",
        f"- IK 成功：`{summary['counts']['ik_success_candidates']}`",
        f"- 纳入共识聚类：`{summary['counts']['eligible_nms_final_ik_success_candidates']}`",
        f"- 共识簇 / 通过簇：`{summary['counts']['clusters']}` / "
        f"`{summary['counts']['accepted_clusters']}`",
        "",
        "## 聚类结果",
        "",
    ]
    if not summary["clusters"]:
        lines.append("**没有 NMS+IK 成功候选，未形成任何共识簇。**")
    else:
        lines.extend(
            [
                "| 簇 | 门禁 | 候选数 | 独立帧 | 帧号 | 中位质量 | 开口范围 mm | 最大平移 mm | 最大旋转 ° |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for cluster in summary["clusters"]:
            opening = cluster["selected_opening_range_m"]
            lines.append(
                "| {cluster} | {status} | {count} | {frames_count} | {frames} | "
                "{quality} | {open_min:.2f}–{open_max:.2f} | {translation:.2f} | {rotation:.2f} |".format(
                    cluster=cluster["cluster_id"],
                    status="通过" if cluster["accepted"] else "拒绝",
                    count=cluster["candidate_count"],
                    frames_count=cluster["distinct_frame_count"],
                    frames=", ".join(str(value) for value in cluster["frames"]),
                    quality=_fmt(cluster["median_quality"], 4),
                    open_min=1000.0 * opening[0],
                    open_max=1000.0 * opening[1],
                    translation=1000.0 * cluster["maximum_pairwise_translation_m"],
                    rotation=cluster["maximum_pairwise_jaw_rotation_deg"],
                )
            )

    accepted = [item for item in summary["clusters"] if item["accepted"]]
    lines.extend(["", "## 最佳共识", ""])
    if not accepted:
        lines.append(
            "**未达到多帧一致门禁，已 fail-closed；不得把单帧偶然成功用于抓取。**"
        )
    else:
        best = accepted[0]
        lines.extend(
            [
                f"- 代表候选：`{best['representative_candidate']}`",
                f"- 独立帧：`{best['distinct_frame_count']}`（{', '.join(str(v) for v in best['frames'])}）",
                f"- 候选：`{best['candidate_count']}`",
                f"- 质心 TCP（基座系，m）："
                f"`[{', '.join(_fmt(value, 6) for value in best['centroid_tcp_base_m'])}]`",
                f"- 共识开口范围：`{1000.0 * best['selected_opening_range_m'][0]:.2f}–"
                f"{1000.0 * best['selected_opening_range_m'][1]:.2f} mm`",
                f"- 簇内最大平移偏差：`{1000.0 * best['maximum_pairwise_translation_m']:.2f} mm`",
                f"- 簇内最大夹爪对称旋转偏差：`{best['maximum_pairwise_jaw_rotation_deg']:.2f}°`",
                f"- 最小支撑面净空：`{1000.0 * best['minimum_support_clearance_m']:.3f} mm`"
                if best["minimum_support_clearance_m"] is not None
                else "- 最小支撑面净空：`n/a`",
                f"- 该门禁只证明离线位姿稳定，不证明实机可抓，也不构成运动授权。",
            ]
        )

    lines.extend(
        [
            "",
            "## 成员候选",
            "",
            "| 簇 | 帧 | 候选 | 代表 | 质量 | 参考宽 mm | 开口 mm | 平移偏差 mm | 旋转偏差 ° | IK 位置误差 mm | IK 姿态误差 ° |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cluster in summary["clusters"]:
        for member in cluster["members"]:
            lines.append(
                "| {cluster} | {frame} | {candidate} | {representative} | "
                "{quality} | {width} | {opening} | {translation} | {rotation} | "
                "{ik_pos} | {ik_rot} |".format(
                    cluster=cluster["cluster_id"],
                    frame=member["frame"],
                    candidate=member["candidate_id"],
                    representative="是" if member["is_representative"] else "",
                    quality=_fmt(member["quality"], 4),
                    width=_fmt(1000.0 * member["reference_width_m"], 2),
                    opening=_fmt(1000.0 * member["selected_opening_m"], 2),
                    translation=_fmt(
                        1000.0 * member["translation_from_representative_m"], 2
                    ),
                    rotation=_fmt(
                        member["jaw_rotation_from_representative_deg"], 2
                    ),
                    ik_pos=(
                        "n/a"
                        if member["ik_position_error_m"] is None
                        else _fmt(1000.0 * member["ik_position_error_m"], 4)
                    ),
                    ik_rot=(
                        "n/a"
                        if member["ik_orientation_error_rad"] is None
                        else _fmt(
                            math.degrees(member["ik_orientation_error_rad"]), 4
                        )
                    ),
                )
            )

    if summary["errors"]:
        lines.extend(["", "## 输入错误", ""])
        lines.extend(f"- {error}" for error in summary["errors"])

    lines.extend(["", "## 安全与限制", ""])
    lines.extend(f"- {caveat}" for caveat in summary["caveats"])
    lines.append("")
    return "\n".join(lines)


def _write_clusters_csv(path: Path, clusters: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "cluster_id",
        "accepted",
        "rejection_reasons",
        "candidate_count",
        "distinct_frame_count",
        "frames",
        "median_quality",
        "selected_opening_min_m",
        "selected_opening_max_m",
        "reference_width_min_m",
        "reference_width_max_m",
        "minimum_support_clearance_m",
        "minimum_tcp_workspace_margin_m",
        "maximum_pairwise_translation_m",
        "maximum_pairwise_jaw_rotation_deg",
        "representative_candidate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for cluster in clusters:
            opening = cluster["selected_opening_range_m"]
            width = cluster["reference_width_range_m"]
            writer.writerow(
                {
                    "cluster_id": cluster["cluster_id"],
                    "accepted": cluster["accepted"],
                    "rejection_reasons": ";".join(cluster["rejection_reasons"]),
                    "candidate_count": cluster["candidate_count"],
                    "distinct_frame_count": cluster["distinct_frame_count"],
                    "frames": ";".join(str(value) for value in cluster["frames"]),
                    "median_quality": cluster["median_quality"],
                    "selected_opening_min_m": opening[0],
                    "selected_opening_max_m": opening[1],
                    "reference_width_min_m": width[0],
                    "reference_width_max_m": width[1],
                    "minimum_support_clearance_m": cluster[
                        "minimum_support_clearance_m"
                    ],
                    "minimum_tcp_workspace_margin_m": cluster[
                        "minimum_tcp_workspace_margin_m"
                    ],
                    "maximum_pairwise_translation_m": cluster[
                        "maximum_pairwise_translation_m"
                    ],
                    "maximum_pairwise_jaw_rotation_deg": cluster[
                        "maximum_pairwise_jaw_rotation_deg"
                    ],
                    "representative_candidate": cluster["representative_candidate"],
                }
            )


def _write_members_csv(path: Path, clusters: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "cluster_id",
        "accepted",
        "frame",
        "candidate_id",
        "is_representative",
        "quality",
        "reference_width_m",
        "selected_opening_m",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
        "translation_from_centroid_m",
        "translation_from_representative_m",
        "jaw_rotation_from_representative_deg",
        "support_minimum_clearance_m",
        "tcp_workspace_margin_m",
        "ik_position_error_m",
        "ik_orientation_error_rad",
        "ik_max_joint_delta_deg",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for cluster in clusters:
            for member in cluster["members"]:
                writer.writerow(
                    {
                        "cluster_id": cluster["cluster_id"],
                        "accepted": cluster["accepted"],
                        "frame": member["frame"],
                        "candidate_id": member["candidate_id"],
                        "is_representative": member["is_representative"],
                        "quality": member["quality"],
                        "reference_width_m": member["reference_width_m"],
                        "selected_opening_m": member["selected_opening_m"],
                        "tcp_x_m": member["tcp_base_m"][0],
                        "tcp_y_m": member["tcp_base_m"][1],
                        "tcp_z_m": member["tcp_base_m"][2],
                        "translation_from_centroid_m": member[
                            "translation_from_centroid_m"
                        ],
                        "translation_from_representative_m": member[
                            "translation_from_representative_m"
                        ],
                        "jaw_rotation_from_representative_deg": member[
                            "jaw_rotation_from_representative_deg"
                        ],
                        "support_minimum_clearance_m": member[
                            "support_minimum_clearance_m"
                        ],
                        "tcp_workspace_margin_m": member[
                            "tcp_workspace_margin_m"
                        ],
                        "ik_position_error_m": member["ik_position_error_m"],
                        "ik_orientation_error_rad": member[
                            "ik_orientation_error_rad"
                        ],
                        "ik_max_joint_delta_deg": member[
                            "ik_max_joint_delta_deg"
                        ],
                        "source_path": member["source_path"],
                    }
                )


def write_consensus_report(
    batch_root: str | Path,
    output_dir: str | Path,
    config: ConsensusConfig | None = None,
) -> dict[str, Any]:
    summary = summarize_consensus(batch_root, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "consensus.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "consensus.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    _write_clusters_csv(output_dir / "consensus_clusters.csv", summary["clusters"])
    _write_members_csv(output_dir / "consensus_members.csv", summary["clusters"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if not args.batch_root.is_dir():
        parser.error(f"batch root is not a directory: {args.batch_root}")
    output_dir = args.output_dir or (args.batch_root / "grasp_consensus")
    config = load_consensus_config(args.config)
    summary = write_consensus_report(args.batch_root, output_dir, config)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "input_valid": summary["input_valid"],
                "consensus_reached": summary["consensus_reached"],
                "accepted_clusters": summary["counts"]["accepted_clusters"],
                "maximum_distinct_frames_in_cluster": summary["counts"][
                    "maximum_distinct_frames_in_cluster"
                ],
                "offline_only": True,
                "motion_attempted": False,
                "hardware_connected": False,
                "motion_authorized": False,
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
