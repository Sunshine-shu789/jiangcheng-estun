import json
from pathlib import Path

from track1_grasp.tools.summarize_offline_graspnet_batch import write_summary


def make_payload(*, final: bool) -> dict:
    final_count = 1 if final else 0
    candidate = {
        "candidate_id": "graspnet_test",
        "final_feasible": final,
        "nms_kept": final,
        "graspnet": {"width_m": 0.055},
        "score_camera": {
            "quality": 0.71,
            "selected_opening_m": 0.035,
            "measured_width_m": 0.026,
        },
        "world_clearance": {
            "tcp_base_m": [0.1, 0.2, 0.03],
            "support_minimum_clearance_m": 0.022,
            "tcp_workspace_margin_m": 0.010,
        },
        "ik": {
            "success": True,
            "position_error_m": 0.000004,
            "orientation_error_rad": 0.00014,
            "at_joint_limit": False,
            "max_joint_delta_deg": 37.0,
        },
    }
    return {
        "offline_only": True,
        "motion_attempted": False,
        "quality_validated": False,
        "motion_authorized": False,
        "hardware_connected": False,
        "support_plane": {
            "valid": True,
            "inlier_ratio": 0.67,
            "tilt_deg": 0.72,
        },
        "summary": {
            "candidate_count": 1,
            "scoring_error_count": 0,
            "support_plane_valid": True,
            "graspnet_reference_score_feasible_count": final_count,
            "aperture_search_trial_count": 10 if final else 0,
            "score_feasible_count": final_count,
            "world_clearance_feasible_count": final_count,
            "final_feasible_count": final_count,
            "nms_survivor_count": final_count,
            "ik_evaluated_count": final_count,
            "ik_success_count": final_count,
        },
        "candidates": [candidate],
    }


def test_write_summary_reports_frame_coverage_and_candidate(tmp_path: Path):
    for frame, final in ((4, False), (5, True)):
        frame_dir = tmp_path / f"frame_{frame:02d}"
        frame_dir.mkdir()
        (frame_dir / "offline_candidates.json").write_text(
            json.dumps(make_payload(final=final)), encoding="utf-8"
        )

    summary = write_summary(tmp_path)

    assert summary["frames_discovered"] == 2
    assert summary["frames_completed"] == 2
    assert summary["frames_with_final_candidate"] == 1
    assert summary["frames_with_ik_success"] == 1
    assert summary["frames_safety_ok"] == 2
    assert summary["final_feasible_count"] == 1
    assert summary["ik_success_count"] == 1
    assert summary["final_candidates"][0]["frame"] == 5
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["frames_completed"] == 2


def test_ik_completeness_tracks_nms_survivors_not_pre_nms_final_count(tmp_path: Path):
    payload = make_payload(final=True)
    payload["summary"].update(
        final_feasible_count=4,
        nms_survivor_count=1,
        ik_evaluated_count=1,
        ik_success_count=1,
    )
    frame_dir = tmp_path / "frame_04"
    frame_dir.mkdir()
    (frame_dir / "offline_candidates.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    summary = write_summary(tmp_path)

    frame = summary["frames"][0]
    assert frame["checks"]["ik_complete"] is True
    assert frame["checks"]["has_ik_success"] is True
    assert summary["nms_survivor_count"] == 1
    assert "| 4 | 1 | 1 | 1 | 4 | 1 | 1 | 1 | yes | ok |" in (
        tmp_path / "summary.md"
    ).read_text(encoding="utf-8")
