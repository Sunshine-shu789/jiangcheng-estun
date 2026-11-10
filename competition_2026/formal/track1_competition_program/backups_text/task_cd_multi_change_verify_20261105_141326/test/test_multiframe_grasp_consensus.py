import json
from pathlib import Path

import numpy as np
import pytest

from track1_grasp.tools.summarize_multiframe_grasp_consensus import (
    ConsensusConfig,
    load_consensus_config,
    summarize_consensus,
    write_consensus_report,
)


def make_payload(
    *,
    tcp=(0.0, 0.0, 0.1),
    rotation=None,
    ik_success=True,
    nms_kept=True,
    final_feasible=True,
    safety_overrides=None,
    summary_overrides=None,
):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = tcp
    candidate = {
        "candidate_id": "graspnet_00001",
        "final_feasible": final_feasible,
        "nms_kept": nms_kept,
        "graspnet": {"width_m": 0.055},
        "score_camera": {
            "quality": 0.75,
            "selected_opening_m": 0.035,
            "measured_width_m": 0.030,
        },
        "transforms": {"T_base_tcp": transform.tolist()},
        "world_clearance": {
            "tcp_base_m": list(tcp),
            "support_minimum_clearance_m": 0.022,
            "tcp_workspace_margin_m": 0.008,
        },
        "ik": (
            {
                "success": True,
                "position_error_m": 0.000001,
                "orientation_error_rad": 0.0001,
                "at_joint_limit": False,
                "max_joint_delta_deg": 25.0,
                "joints_rad": [0.1, 0.2, -0.3, 0.4, 0.5, -0.6],
            }
            if ik_success
            else {
                "success": False,
                "position_error_m": 0.02,
                "orientation_error_rad": 0.3,
                "at_joint_limit": True,
                "max_joint_delta_deg": 90.0,
            }
        ),
    }
    payload = {
        "offline_only": True,
        "motion_attempted": False,
        "motion_authorized": False,
        "hardware_connected": False,
        "quality_validated": False,
        "summary": {
            "nms_survivor_count": 1 if nms_kept else 0,
            "ik_evaluated_count": 1 if nms_kept else 0,
            "ik_success_count": 1 if nms_kept and ik_success else 0,
        },
        "candidates": [candidate],
    }
    payload.update(safety_overrides or {})
    payload["summary"].update(summary_overrides or {})
    return payload


def write_frames(root: Path, frames: dict[int, dict]) -> Path:
    for frame, payload in frames.items():
        frame_dir = root / f"frame_{frame:02d}"
        frame_dir.mkdir(parents=True)
        (frame_dir / "offline_candidates.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return root


def test_four_independent_frames_form_fail_closed_consensus(tmp_path: Path):
    write_frames(
        tmp_path,
        {
            4: make_payload(tcp=(0.000, 0.000, 0.100)),
            5: make_payload(tcp=(0.003, 0.001, 0.100)),
            6: make_payload(tcp=(0.006, -0.002, 0.101)),
            7: make_payload(tcp=(0.009, 0.002, 0.099)),
            8: make_payload(tcp=(0.080, 0.080, 0.100)),
        },
    )

    summary = summarize_consensus(tmp_path, ConsensusConfig(minimum_distinct_frames=4))

    assert summary["input_valid"] is True
    assert summary["consensus_reached"] is True
    assert summary["counts"]["accepted_clusters"] == 1
    assert summary["counts"]["maximum_distinct_frames_in_cluster"] == 4
    best = summary["clusters"][0]
    assert best["accepted"] is True
    assert best["frames"] == [4, 5, 6, 7]
    assert best["representative_candidate"].startswith("frame_")
    assert summary["offline_only"] is True
    assert summary["motion_attempted"] is False
    assert summary["motion_authorized"] is False


def test_three_frames_do_not_meet_default_gate(tmp_path: Path):
    write_frames(
        tmp_path,
        {
            4: make_payload(tcp=(0.000, 0.000, 0.100)),
            5: make_payload(tcp=(0.003, 0.000, 0.100)),
            6: make_payload(tcp=(0.006, 0.000, 0.100)),
        },
    )

    summary = summarize_consensus(tmp_path, ConsensusConfig())

    assert summary["input_valid"] is True
    assert summary["consensus_reached"] is False
    assert summary["status"] == "no_consensus_fail_closed"
    assert summary["clusters"][0]["accepted"] is False
    assert "insufficient_distinct_frames" in summary["clusters"][0]["rejection_reasons"]


def test_parallel_jaw_180_degree_flip_is_same_pose(tmp_path: Path):
    jaw_flip = np.diag([1.0, -1.0, -1.0])
    write_frames(
        tmp_path,
        {
            4: make_payload(tcp=(0.000, 0.000, 0.100)),
            5: make_payload(tcp=(0.002, 0.000, 0.100), rotation=jaw_flip),
            6: make_payload(tcp=(0.004, 0.000, 0.100)),
            7: make_payload(tcp=(0.006, 0.000, 0.100), rotation=jaw_flip),
        },
    )

    summary = summarize_consensus(tmp_path)

    assert summary["clusters"][0]["accepted"] is True
    assert summary["clusters"][0]["maximum_pairwise_jaw_rotation_deg"] == pytest.approx(
        0.0, abs=1e-7
    )


def test_multiple_candidates_in_one_frame_do_not_inflate_frame_count(tmp_path: Path):
    payload = make_payload()
    extra = json.loads(json.dumps(payload["candidates"][0]))
    extra["candidate_id"] = "graspnet_00002"
    extra["transforms"]["T_base_tcp"][0][3] += 0.005
    extra["world_clearance"]["tcp_base_m"][0] += 0.005
    payload["candidates"].append(extra)
    payload["summary"].update(
        nms_survivor_count=2,
        ik_evaluated_count=2,
        ik_success_count=2,
    )
    write_frames(tmp_path, {4: payload})

    summary = summarize_consensus(tmp_path, ConsensusConfig(minimum_distinct_frames=2))

    assert summary["clusters"][0]["candidate_count"] == 2
    assert summary["clusters"][0]["distinct_frame_count"] == 1
    assert summary["clusters"][0]["accepted"] is False


def test_unsafe_or_incomplete_input_fails_closed(tmp_path: Path):
    write_frames(
        tmp_path,
        {
            4: make_payload(
                safety_overrides={"motion_attempted": True},
            ),
            5: make_payload(
                summary_overrides={"ik_evaluated_count": 0},
            ),
        },
    )

    summary = summarize_consensus(tmp_path)

    assert summary["input_valid"] is False
    assert summary["consensus_reached"] is False
    assert summary["status"] == "input_invalid_fail_closed"
    assert any("motion_attempted" in error for error in summary["errors"])
    assert any("ik_evaluated_count" in error for error in summary["errors"])


def test_ik_failed_and_nms_suppressed_candidates_are_not_clustered(tmp_path: Path):
    write_frames(
        tmp_path,
        {
            4: make_payload(ik_success=False),
            5: make_payload(nms_kept=False),
            6: make_payload(final_feasible=False),
        },
    )

    summary = summarize_consensus(tmp_path)

    assert summary["counts"]["eligible_nms_final_ik_success_candidates"] == 0
    assert summary["clusters"] == []
    assert summary["consensus_reached"] is False


def test_write_report_outputs_json_markdown_and_csv(tmp_path: Path):
    batch = tmp_path / "batch"
    batch.mkdir()
    write_frames(
        batch,
        {
            4: make_payload(tcp=(0.000, 0.000, 0.100)),
            5: make_payload(tcp=(0.002, 0.000, 0.100)),
            6: make_payload(tcp=(0.004, 0.000, 0.100)),
            7: make_payload(tcp=(0.006, 0.000, 0.100)),
        },
    )

    output = tmp_path / "output"
    summary = write_consensus_report(batch, output, ConsensusConfig())

    assert summary["consensus_reached"] is True
    assert (output / "consensus.json").is_file()
    assert "离线多帧抓取共识报告" in (output / "consensus.md").read_text(
        encoding="utf-8"
    )
    assert (output / "consensus_clusters.csv").is_file()
    assert (output / "consensus_members.csv").is_file()


def test_default_config_is_explicitly_offline_only():
    config = load_consensus_config()
    assert config.name == "offline_multiframe_grasp_consensus_v1"
    assert config.validated is False
    assert config.minimum_distinct_frames == 4
    assert config.translation_tolerance_m == pytest.approx(0.025)
    assert config.jaw_rotation_tolerance_deg == pytest.approx(15.0)


def test_ik_success_at_joint_limit_invalidates_input(tmp_path: Path):
    payload = make_payload()
    payload["candidates"][0]["ik"]["at_joint_limit"] = True
    write_frames(tmp_path, {4: payload})

    summary = summarize_consensus(tmp_path)

    assert summary["input_valid"] is False
    assert summary["consensus_reached"] is False
    assert summary["counts"]["integrity_clean_frames"] == 0
    assert any("at_joint_limit" in error for error in summary["errors"])
