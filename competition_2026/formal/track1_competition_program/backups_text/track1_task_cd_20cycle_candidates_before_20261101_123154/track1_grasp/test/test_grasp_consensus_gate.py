import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from track1_grasp.config import (
    GraspConsensusGateConfig,
    apply_mode,
    load_config,
)
from track1_grasp.domain import PickStatus, Pose, TaskMode, TaskPhase
from track1_grasp.factory import (
    build_grasp_consensus_gate,
    build_pick_pipeline,
    build_pipeline,
)
from track1_grasp.grasp_consensus_gate import GraspConsensusGate
from track1_grasp.tools.summarize_multiframe_grasp_consensus import (
    ConsensusConfig,
    write_consensus_report,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(__file__).resolve().parents[3]
MOCK_CONFIG = ROOT / "config" / "track1_mock.yaml"
ARBITRARY_MOCK_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_b_arbitrary_mock.yaml"
)


def _candidate_payload(tcp):
    rotation = np.eye(3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tcp, dtype=float)
    return {
        "offline_only": True,
        "motion_attempted": False,
        "motion_authorized": False,
        "hardware_connected": False,
        "quality_validated": False,
        "summary": {
            "nms_survivor_count": 1,
            "ik_evaluated_count": 1,
            "ik_success_count": 1,
        },
        "candidates": [
            {
                "candidate_id": "graspnet_00001",
                "final_feasible": True,
                "nms_kept": True,
                "graspnet": {"width_m": 0.055},
                "score_camera": {
                    "quality": 0.75,
                    "selected_opening_m": 0.035,
                    "measured_width_m": 0.030,
                },
                "transforms": {"T_base_tcp": transform.tolist()},
                "world_clearance": {
                    "tcp_base_m": [float(value) for value in tcp],
                    "support_minimum_clearance_m": 0.022,
                    "tcp_workspace_margin_m": 0.008,
                },
                "ik": {
                    "success": True,
                    "position_error_m": 0.000001,
                    "orientation_error_rad": 0.0001,
                    "at_joint_limit": False,
                    "max_joint_delta_deg": 25.0,
                    "joints_rad": [0.1, 0.2, -0.3, 0.4, 0.5, -0.6],
                },
            }
        ],
    }


def _make_report(
    root: Path,
    *,
    center=(0.10, 0.20, 0.30),
) -> tuple[Path, dict]:
    batch = root / "batch"
    for frame, dx in ((4, 0.0), (5, 0.003), (6, 0.006), (7, 0.009)):
        frame_dir = batch / f"frame_{frame:02d}"
        frame_dir.mkdir(parents=True)
        tcp = (center[0] + dx, center[1], center[2])
        (frame_dir / "offline_candidates.json").write_text(
            json.dumps(_candidate_payload(tcp)),
            encoding="utf-8",
        )

    output = root / "consensus"
    write_consensus_report(
        batch,
        output,
        ConsensusConfig(minimum_distinct_frames=4),
    )
    report_path = output / "consensus.json"
    return report_path, json.loads(report_path.read_text(encoding="utf-8"))


def _write_report(path: Path, report: dict, *, allow_nan: bool = False) -> None:
    path.write_text(
        json.dumps(report, allow_nan=allow_nan),
        encoding="utf-8",
    )


def _gate(report_path: Path, **overrides) -> GraspConsensusGate:
    config = replace(
        GraspConsensusGateConfig(
            enabled=True,
            report_file=str(report_path),
        ),
        **overrides,
    )
    return GraspConsensusGate(
        config,
        pose_frame="tcp",
        tcp_offset_m=(0.0, 0.0, 0.0),
        task_mode=TaskMode.C,
    )


def _representative_pose(report: dict) -> Pose:
    accepted = next(
        cluster for cluster in report["clusters"] if cluster["accepted"]
    )
    return Pose(tuple(accepted["representative_tcp_base_m"]), (0.0, 0.0, 0.0))


def test_matching_offline_consensus_is_accepted_but_never_motion_authorized(
    tmp_path: Path,
):
    report_path, report = _make_report(tmp_path)
    decision = _gate(report_path).evaluate(_representative_pose(report))

    assert decision.accepted is True
    assert decision.offline_only is True
    assert decision.motion_authorized is False
    assert decision.report_path == str(report_path)
    assert decision.report_sha256 == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert decision.cluster_id == next(
        cluster["cluster_id"]
        for cluster in report["clusters"]
        if cluster["accepted"]
    )
    assert decision.distinct_frame_count == 4
    assert decision.pose_translation_error_m == pytest.approx(0.0, abs=1e-9)
    assert decision.pose_jaw_rotation_error_deg == pytest.approx(0.0, abs=1e-9)
    assert decision.report_quality_validated is False


def test_missing_consensus_report_fails_closed(tmp_path: Path):
    decision = _gate(tmp_path / "missing.json").evaluate(
        Pose((0.10, 0.20, 0.30), (0.0, 0.0, 0.0))
    )

    assert decision.accepted is False
    assert "cannot read consensus report" in decision.reason
    assert decision.motion_authorized is False


def test_corrupt_consensus_report_fails_closed(tmp_path: Path):
    report_path = tmp_path / "corrupt.json"
    report_path.write_text("{not valid json", encoding="utf-8")

    decision = _gate(report_path).evaluate(
        Pose((0.10, 0.20, 0.30), (0.0, 0.0, 0.0))
    )

    assert decision.accepted is False
    assert "not valid UTF-8 JSON" in decision.reason
    assert decision.motion_authorized is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("offline_only", False, "offline_only"),
        ("motion_attempted", True, "motion_attempted"),
        ("hardware_connected", True, "hardware_connected"),
        ("motion_authorized", True, "motion_authorized"),
        ("quality_validated", "false", "quality_validated"),
    ),
)
def test_unsafe_report_safety_markers_fail_closed(
    tmp_path: Path,
    field,
    value,
    message,
):
    report_path, report = _make_report(tmp_path)
    report[field] = value
    _write_report(report_path, report)

    decision = _gate(report_path).evaluate(_representative_pose(report))

    assert decision.accepted is False
    assert message in decision.reason
    assert decision.motion_authorized is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("minimum_distinct_frames", 1, "minimum_distinct_frames is weaker"),
        ("translation_tolerance_m", 0.03, "translation tolerance is weaker"),
        ("jaw_rotation_tolerance_deg", 20.0, "jaw rotation tolerance is weaker"),
        ("translation_tolerance_m", float("nan"), "positive and finite"),
    ),
)
def test_weakened_or_non_finite_report_thresholds_fail_closed(
    tmp_path: Path,
    field,
    value,
    message,
):
    report_path, report = _make_report(tmp_path)
    report["config"][field] = value
    _write_report(
        report_path,
        report,
        allow_nan=isinstance(value, float) and math.isnan(value),
    )

    decision = _gate(report_path).evaluate(_representative_pose(report))

    assert decision.accepted is False
    assert message in decision.reason


def test_tampered_cluster_member_fails_closed(tmp_path: Path):
    report_path, report = _make_report(tmp_path)
    accepted = next(
        cluster for cluster in report["clusters"] if cluster["accepted"]
    )
    accepted["members"][0]["tcp_base_m"][0] += 0.10
    _write_report(report_path, report)

    decision = _gate(report_path).evaluate(_representative_pose(report))

    assert decision.accepted is False
    assert "cluster exceeds runtime translation tolerance" in decision.reason


def test_tampered_representative_pose_fails_closed(tmp_path: Path):
    report_path, report = _make_report(tmp_path)
    accepted = next(
        cluster for cluster in report["clusters"] if cluster["accepted"]
    )
    accepted["representative_tcp_base_m"][0] += 0.001
    _write_report(report_path, report)

    decision = _gate(report_path).evaluate(_representative_pose(report))

    assert decision.accepted is False
    assert "representative TCP position disagrees with member" in decision.reason


def test_online_tcp_translation_mismatch_fails_closed(tmp_path: Path):
    report_path, report = _make_report(tmp_path)
    matched = _representative_pose(report)
    planned = Pose(
        (matched.xyz[0] + 0.03, matched.xyz[1], matched.xyz[2]),
        matched.rpy,
    )

    decision = _gate(report_path).evaluate(planned)

    assert decision.accepted is False
    assert "planned TCP pose does not match" in decision.reason
    assert decision.pose_translation_error_m == pytest.approx(0.03)


def test_online_jaw_rotation_mismatch_fails_closed(tmp_path: Path):
    report_path, report = _make_report(tmp_path)
    matched = _representative_pose(report)
    planned = Pose(matched.xyz, (0.0, 0.0, math.radians(30.0)))

    decision = _gate(report_path).evaluate(planned)

    assert decision.accepted is False
    assert "grasp jaw orientation does not match" in decision.reason
    assert decision.pose_jaw_rotation_error_deg == pytest.approx(30.0)


def test_adaptive_task_c_pipeline_checks_consensus_before_any_command(
    tmp_path: Path,
):
    config = apply_mode(load_config(ARBITRARY_MOCK_CONFIG), "C")
    config = replace(
        config,
        task=replace(config.task, max_picks=1),
        grasp_consensus=replace(
            config.grasp_consensus,
            enabled=True,
            report_file=str(tmp_path / "does_not_exist.json"),
        ),
    )
    pipeline = build_pick_pipeline(config)

    result = pipeline.run_once()

    assert result.status is PickStatus.REJECTED
    assert "offline consensus gate rejected" in result.message
    assert any(
        event.phase is TaskPhase.PLAN
        and "offline consensus gate rejected" in event.message
        for event in result.events
    )
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []


def test_factory_wires_accepted_gate_into_offline_mock_pipeline(tmp_path: Path):
    base = load_config(MOCK_CONFIG)
    reference_pipeline = build_pipeline(base)
    reference_result = reference_pipeline.run_once()
    assert reference_result.success is True
    grasp_pose = next(
        pose for name, pose in reference_pipeline.motion.moves if name == "grasp"
    )

    report_path, _report = _make_report(tmp_path, center=grasp_pose.xyz)
    config = replace(
        base,
        pose=replace(base.pose, robot_pose_frame="tcp"),
        grasp_consensus=replace(
            base.grasp_consensus,
            enabled=True,
            report_file=str(report_path),
        ),
    )

    gate = build_grasp_consensus_gate(config)
    assert isinstance(gate, GraspConsensusGate)

    pipeline = build_pipeline(config)
    result = pipeline.run_once()

    assert result.success is True
    assert result.status is PickStatus.SUCCESS
    assert len(pipeline.motion.moves) == 6
    assert pipeline.gripper.events == ["open", "close", "open"]
    passed = [
        event.message
        for event in result.events
        if event.phase is TaskPhase.PLAN
        and "offline consensus gate passed" in event.message
    ]
    assert len(passed) == 1
    assert "motion_authorized=false" in passed[0]
