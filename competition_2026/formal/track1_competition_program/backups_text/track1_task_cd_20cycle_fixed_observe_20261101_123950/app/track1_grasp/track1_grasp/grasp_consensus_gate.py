"""Fail-closed runtime gate for offline multi-frame grasp consensus.

The gate consumes the immutable ``consensus.json`` produced by
``summarize_multiframe_grasp_consensus`` and binds its representative TCP pose
to the pose the online planner is about to command.  It never relaxes a
threshold, never authorizes motion, and fails closed on any malformed,
incomplete or unsafe report.

Configuration validation restricts an enabled gate to fully offline Task C/D:
robot motion must be dry-run, the gripper backend must be mock, and autonomous
search motion must be disabled so no arm command can occur before agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import GraspConsensusGateConfig, Track1Config
from .domain import Pose, TaskMode
from .jaw_grasp_scoring import jaw_symmetric_rotation_distance
from .kinematics import rpy_to_matrix


REPORT_KIND = "offline_multiframe_grasp_consensus"
REPORT_STATUS = "consensus_available_offline_only"
_MAX_REPORT_BYTES = 20 * 1024 * 1024


class _GateError(ValueError):
    """A report or planned pose failed a mandatory gate check."""


@dataclass(frozen=True)
class ConsensusDecision:
    """Auditable result of one runtime gate evaluation.

    ``accepted`` only means that the report and the planned TCP pose passed the
    offline gate.  It explicitly does not mean that the arm or gripper is
    authorized to move.
    """

    accepted: bool
    reason: str
    report_path: str = ""
    report_sha256: str = ""
    cluster_id: str | None = None
    representative_candidate: str | None = None
    distinct_frame_count: int = 0
    planned_tcp_base_m: tuple[float, float, float] | None = None
    representative_tcp_base_m: tuple[float, float, float] | None = None
    pose_translation_error_m: float | None = None
    pose_jaw_rotation_error_deg: float | None = None
    report_quality_validated: bool = False
    offline_only: bool = True
    motion_authorized: bool = False


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _GateError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise _GateError(f"{name} must be finite")
    return number


def _vector3(value: Any, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise _GateError(f"{name} must contain three finite values") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise _GateError(f"{name} must contain three finite values")
    return vector


def _rotation(value: Any, name: str) -> np.ndarray:
    try:
        rotation = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise _GateError(f"{name} must be a proper 3x3 rotation") from exc
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise _GateError(f"{name} must be a proper 3x3 rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.0):
        raise _GateError(f"{name} must be orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2e-5:
        raise _GateError(f"{name} determinant must be +1")
    return rotation


def _joints(value: Any, name: str) -> tuple[float, ...]:
    try:
        joints = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise _GateError(f"{name} must contain six finite values") from exc
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise _GateError(f"{name} must contain six finite values")
    return tuple(float(item) for item in joints)


def _require_exact_bool(payload: Mapping[str, Any], key: str, expected: bool) -> None:
    if payload.get(key) is not expected:
        raise _GateError(
            f"report field {key!r} must be {str(expected).lower()}, "
            f"got {payload.get(key)!r}"
        )


def _planned_tcp_transform(
    pose: Pose,
    *,
    pose_frame: str,
    tcp_offset_m: tuple[float, float, float],
) -> np.ndarray:
    """Convert the commanded controller pose to TCP semantics.

    The controller's declared pose frame must be known.  Guessing between a
    flange and a TCP frame could shift the consensus comparison by the entire
    gripper length, so ``unknown`` is rejected.
    """

    base_controller = np.eye(4)
    base_controller[:3, :3] = rpy_to_matrix(pose.rpy)
    base_controller[:3, 3] = np.asarray(pose.xyz, dtype=float)
    if pose_frame == "tcp":
        return base_controller
    if pose_frame == "flange":
        flange_tcp = np.eye(4)
        flange_tcp[:3, 3] = np.asarray(tcp_offset_m, dtype=float)
        return base_controller @ flange_tcp
    raise _GateError(
        "consensus gate requires pose.robot_pose_frame=flange or tcp; "
        f"got {pose_frame!r}"
    )


def _validate_report_thresholds(
    report: Mapping[str, Any], config: GraspConsensusGateConfig
) -> None:
    raw = report.get("config")
    if not isinstance(raw, Mapping):
        raise _GateError("report config is missing")
    minimum = raw.get("minimum_distinct_frames")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise _GateError("report minimum_distinct_frames must be an integer")
    try:
        translation = float(raw["translation_tolerance_m"])
        rotation = float(raw["jaw_rotation_tolerance_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _GateError("report consensus thresholds are malformed") from exc
    if (
        not math.isfinite(translation)
        or translation <= 0.0
        or not math.isfinite(rotation)
        or rotation <= 0.0
    ):
        raise _GateError("report consensus thresholds must be positive and finite")
    if minimum < int(config.minimum_distinct_frames):
        raise _GateError(
            "report minimum_distinct_frames is weaker than the runtime gate "
            f"({minimum} < {config.minimum_distinct_frames})"
        )
    if translation > float(config.maximum_translation_tolerance_m) + 1e-12:
        raise _GateError(
            "report translation tolerance is weaker than the runtime gate "
            f"({translation:.6f} > "
            f"{config.maximum_translation_tolerance_m:.6f} m)"
        )
    if rotation > float(config.maximum_jaw_rotation_tolerance_deg) + 1e-12:
        raise _GateError(
            "report jaw rotation tolerance is weaker than the runtime gate "
            f"({rotation:.6f} > "
            f"{config.maximum_jaw_rotation_tolerance_deg:.6f} deg)"
        )


def _validate_cluster(
    cluster: Mapping[str, Any], config: GraspConsensusGateConfig
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Validate and summarize one accepted cluster without trusting its flags."""

    if cluster.get("accepted") is not True:
        raise _GateError("accepted cluster does not have accepted=true")
    reasons = cluster.get("rejection_reasons")
    if reasons != []:
        raise _GateError("accepted cluster still reports rejection reasons")

    raw_frames = cluster.get("frames")
    if not isinstance(raw_frames, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in raw_frames
    ):
        raise _GateError("cluster frames must be non-negative integers")
    frames = [int(item) for item in raw_frames]
    if len(set(frames)) != len(frames):
        raise _GateError("cluster frames contain duplicates")

    raw_members = cluster.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise _GateError("accepted cluster members are missing")
    candidate_count = cluster.get("candidate_count")
    distinct_frames = cluster.get("distinct_frame_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or isinstance(distinct_frames, bool)
        or not isinstance(distinct_frames, int)
    ):
        raise _GateError("cluster counts must be integers")
    try:
        report_translation = float(cluster["maximum_pairwise_translation_m"])
        report_rotation_deg = float(cluster["maximum_pairwise_jaw_rotation_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _GateError("cluster pairwise metrics are malformed") from exc
    if (
        not math.isfinite(report_translation)
        or report_translation < 0.0
        or not math.isfinite(report_rotation_deg)
        or report_rotation_deg < 0.0
    ):
        raise _GateError("cluster pairwise metrics must be finite and non-negative")
    if candidate_count != len(raw_members):
        raise _GateError("cluster candidate_count does not match members")
    if distinct_frames != len(set(frames)):
        raise _GateError("cluster distinct_frame_count does not match frames")
    if distinct_frames < int(config.minimum_distinct_frames):
        raise _GateError(
            "accepted cluster has too few distinct frames "
            f"({distinct_frames} < {config.minimum_distinct_frames})"
        )

    members: list[dict[str, Any]] = []
    member_frames: list[int] = []
    translations: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    labels: set[str] = set()
    for index, raw_member in enumerate(raw_members):
        if not isinstance(raw_member, Mapping):
            raise _GateError(f"cluster member {index} is not an object")
        raw_frame = raw_member.get("frame")
        if isinstance(raw_frame, bool) or not isinstance(raw_frame, int):
            raise _GateError(f"cluster member {index} has no integer frame")
        candidate_id = raw_member.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise _GateError(f"cluster member {index} has no candidate_id")
        label = f"frame_{raw_frame:02d}/{candidate_id}"
        if label in labels:
            raise _GateError(f"duplicate cluster member {label}")
        labels.add(label)
        member_frames.append(raw_frame)
        translations.append(_vector3(raw_member.get("tcp_base_m"), f"{label} tcp"))
        rotations.append(_rotation(raw_member.get("rotation_base"), f"{label} rotation"))
        _joints(raw_member.get("joints_rad"), f"{label} joints_rad")
        members.append(dict(raw_member))

    if set(member_frames) != set(frames):
        raise _GateError("cluster frames do not match member frames")

    max_translation = 0.0
    max_rotation_rad = 0.0
    for first in range(len(members)):
        for second in range(first + 1, len(members)):
            max_translation = max(
                max_translation,
                float(np.linalg.norm(translations[first] - translations[second])),
            )
            max_rotation_rad = max(
                max_rotation_rad,
                jaw_symmetric_rotation_distance(
                    rotations[first], rotations[second]
                ),
            )
    max_rotation_deg = math.degrees(max_rotation_rad)
    if (
        max_translation
        > float(config.maximum_translation_tolerance_m) + 1e-12
    ):
        raise _GateError(
            "cluster exceeds runtime translation tolerance "
            f"({max_translation:.6f} > "
            f"{config.maximum_translation_tolerance_m:.6f} m)"
        )
    if (
        max_rotation_deg
        > float(config.maximum_jaw_rotation_tolerance_deg) + 1e-12
    ):
        raise _GateError(
            "cluster exceeds runtime jaw rotation tolerance "
            f"({max_rotation_deg:.6f} > "
            f"{config.maximum_jaw_rotation_tolerance_deg:.6f} deg)"
        )
    if not math.isclose(
        report_translation, max_translation, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise _GateError(
            "reported maximum pairwise translation disagrees with members"
        )
    if not math.isclose(
        report_rotation_deg, max_rotation_deg, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise _GateError(
            "reported maximum pairwise jaw rotation disagrees with members"
        )

    representative_label = cluster.get("representative_candidate")
    if not isinstance(representative_label, str) or representative_label not in labels:
        raise _GateError("cluster representative is not one of its members")
    representative_entries = [
        (member, translation, rotation)
        for member, translation, rotation in zip(members, translations, rotations)
        if f"frame_{int(member['frame']):02d}/{member['candidate_id']}"
        == representative_label
    ]
    if len(representative_entries) != 1:
        raise _GateError("cluster representative is ambiguous")
    representative, representative_translation, representative_rotation = (
        representative_entries[0]
    )
    reported_rep_translation = _vector3(
        cluster.get("representative_tcp_base_m"),
        "representative_tcp_base_m",
    )
    reported_rep_rotation = _rotation(
        cluster.get("representative_rotation_base"),
        "representative_rotation_base",
    )
    if not np.allclose(
        representative_translation,
        reported_rep_translation,
        atol=1e-9,
        rtol=0.0,
    ):
        raise _GateError("representative TCP position disagrees with member")
    if not np.allclose(
        representative_rotation,
        reported_rep_rotation,
        atol=2e-6,
        rtol=0.0,
    ):
        raise _GateError("representative rotation disagrees with member")

    cluster_id = cluster.get("cluster_id")
    if not isinstance(cluster_id, str) or not cluster_id.strip():
        raise _GateError("accepted cluster has no non-empty cluster_id")

    summary = {
        "cluster_id": cluster_id,
        "representative_candidate": representative_label,
        "distinct_frame_count": distinct_frames,
        "candidate_count": candidate_count,
    }
    return summary, representative_translation, representative_rotation


def _load_report(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _GateError(f"cannot read consensus report: {exc}") from exc
    if len(raw) > _MAX_REPORT_BYTES:
        raise _GateError("consensus report exceeds the 20 MiB safety limit")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _GateError(f"consensus report is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _GateError("consensus report root must be an object")
    return payload, digest


class GraspConsensusGate:
    """Evaluate one plan against one per-scene offline consensus report."""

    def __init__(
        self,
        config: GraspConsensusGateConfig,
        *,
        pose_frame: str,
        tcp_offset_m: tuple[float, float, float],
        task_mode: TaskMode,
    ) -> None:
        if not config.enabled:
            raise ValueError("cannot construct a disabled consensus gate")
        if task_mode not in {TaskMode.C, TaskMode.D}:
            raise ValueError("offline consensus gate is restricted to Task C/D")
        self.config = config
        self.pose_frame = str(pose_frame)
        self.tcp_offset_m = tuple(float(item) for item in tcp_offset_m)
        self.task_mode = task_mode

    @property
    def report_path(self) -> Path:
        return Path(self.config.report_file).expanduser()

    def evaluate(self, planned_pose: Pose) -> ConsensusDecision:
        """Return a decision; every error is converted into a rejection."""

        path = self.report_path
        sha256 = ""
        nearest_cluster: str | None = None
        nearest_translation: float | None = None
        nearest_rotation_deg: float | None = None
        planned_tcp: tuple[float, float, float] | None = None
        representative_tcp: tuple[float, float, float] | None = None
        try:
            planned_tcp_transform = _planned_tcp_transform(
                planned_pose,
                pose_frame=self.pose_frame,
                tcp_offset_m=self.tcp_offset_m,
            )
            planned_tcp_array = planned_tcp_transform[:3, 3]
            planned_tcp = tuple(float(item) for item in planned_tcp_array)

            report, sha256 = _load_report(path)
            if report.get("schema_version") != 1:
                raise _GateError("report schema_version must be 1")
            if report.get("kind") != REPORT_KIND:
                raise _GateError("report kind is not offline grasp consensus")
            if report.get("status") != REPORT_STATUS:
                raise _GateError(
                    f"report status must be {REPORT_STATUS!r}, "
                    f"got {report.get('status')!r}"
                )
            _require_exact_bool(report, "offline_only", True)
            _require_exact_bool(report, "motion_attempted", False)
            _require_exact_bool(report, "hardware_connected", False)
            _require_exact_bool(report, "motion_authorized", False)
            if not isinstance(report.get("quality_validated"), bool):
                raise _GateError("report quality_validated must be boolean")
            _require_exact_bool(report, "input_valid", True)
            _require_exact_bool(report, "consensus_reached", True)
            errors = report.get("errors")
            if not isinstance(errors, list) or errors:
                raise _GateError("report errors must be an empty list")
            _validate_report_thresholds(report, self.config)

            raw_clusters = report.get("clusters")
            if not isinstance(raw_clusters, list):
                raise _GateError("report clusters must be a list")
            accepted_clusters: list[Mapping[str, Any]] = []
            for index, raw_cluster in enumerate(raw_clusters):
                if not isinstance(raw_cluster, Mapping):
                    raise _GateError(f"cluster {index} is not an object")
                if raw_cluster.get("accepted") is True:
                    accepted_clusters.append(raw_cluster)
            if not accepted_clusters:
                raise _GateError("report has no accepted consensus cluster")

            counts = report.get("counts")
            if not isinstance(counts, Mapping):
                raise _GateError("report counts are missing")
            reported_accepted = counts.get("accepted_clusters")
            reported_clusters = counts.get("clusters")
            if (
                isinstance(reported_accepted, bool)
                or not isinstance(reported_accepted, int)
                or isinstance(reported_clusters, bool)
                or not isinstance(reported_clusters, int)
            ):
                raise _GateError("report cluster counts must be integers")
            if reported_accepted != len(accepted_clusters):
                raise _GateError(
                    "report accepted_clusters count does not match clusters"
                )
            if reported_clusters != len(raw_clusters):
                raise _GateError("report clusters count does not match clusters")

            candidates: list[tuple[float, float, float, dict[str, Any], np.ndarray, np.ndarray]] = []
            for cluster in accepted_clusters:
                summary, rep_translation, rep_rotation = _validate_cluster(
                    cluster, self.config
                )
                translation_error = float(
                    np.linalg.norm(planned_tcp_array - rep_translation)
                )
                rotation_error = float(
                    jaw_symmetric_rotation_distance(
                        planned_tcp_transform[:3, :3], rep_rotation
                    )
                )
                rotation_error_deg = math.degrees(rotation_error)
                score = (
                    translation_error
                    / max(float(self.config.maximum_pose_translation_error_m), 1e-12)
                    + rotation_error_deg
                    / max(
                        float(self.config.maximum_pose_jaw_rotation_error_deg),
                        1e-12,
                    )
                )
                candidates.append(
                    (
                        score,
                        translation_error,
                        rotation_error_deg,
                        summary,
                        rep_translation,
                        rep_rotation,
                    )
                )
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            _score, nearest_translation, nearest_rotation_deg, summary, rep_translation, _rep_rotation = candidates[0]
            nearest_cluster = summary["cluster_id"]
            representative_tcp = tuple(float(item) for item in rep_translation)
            if nearest_translation > float(
                self.config.maximum_pose_translation_error_m
            ) + 1e-12:
                raise _GateError(
                    "planned TCP pose does not match the accepted consensus "
                    f"({nearest_translation:.6f} m > "
                    f"{self.config.maximum_pose_translation_error_m:.6f} m)"
                )
            if nearest_rotation_deg > float(
                self.config.maximum_pose_jaw_rotation_error_deg
            ) + 1e-12:
                raise _GateError(
                    "planned grasp jaw orientation does not match the accepted "
                    f"consensus ({nearest_rotation_deg:.6f} deg > "
                    f"{self.config.maximum_pose_jaw_rotation_error_deg:.6f} deg)"
                )
            return ConsensusDecision(
                accepted=True,
                reason=(
                    "offline multi-frame consensus accepted and bound to the "
                    "planned TCP pose; execution remains dry-run/mock"
                ),
                report_path=str(path),
                report_sha256=sha256,
                cluster_id=summary["cluster_id"],
                representative_candidate=summary["representative_candidate"],
                distinct_frame_count=summary["distinct_frame_count"],
                planned_tcp_base_m=planned_tcp,
                representative_tcp_base_m=representative_tcp,
                pose_translation_error_m=nearest_translation,
                pose_jaw_rotation_error_deg=nearest_rotation_deg,
                report_quality_validated=bool(report["quality_validated"]),
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            return ConsensusDecision(
                accepted=False,
                reason=str(exc),
                report_path=str(path),
                report_sha256=sha256,
                cluster_id=nearest_cluster,
                distinct_frame_count=0,
                planned_tcp_base_m=planned_tcp,
                representative_tcp_base_m=representative_tcp,
                pose_translation_error_m=nearest_translation,
                pose_jaw_rotation_error_deg=nearest_rotation_deg,
                report_quality_validated=False,
            )


def build_grasp_consensus_gate(config: Track1Config) -> GraspConsensusGate | None:
    """Build the configured gate, or return ``None`` when it is disabled."""

    gate = config.grasp_consensus
    if not gate.enabled:
        return None
    return GraspConsensusGate(
        gate,
        pose_frame=config.pose.robot_pose_frame,
        tcp_offset_m=config.pose.tcp_offset_m,
        task_mode=config.task.mode,
    )
