from pathlib import Path

import numpy as np
import pytest

from track1_grasp.jaw_grasp_scoring import (
    JawAwareGraspScorer,
    JawGraspCandidate,
    load_jaw_scoring_config,
)
from track1_grasp.jaw_profile import JawProfile
from track1_grasp.jaw_world_clearance import (
    GripperClearanceConfig,
    apply_gripper_world_clearance,
    evaluate_gripper_world_clearance,
    load_gripper_clearance_config,
    sample_gripper_envelope_local,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT.parents[1]
PROFILE_PATH = (
    DATA_ROOT
    / "robot_description"
    / "estun_s3_60_g2"
    / "config"
    / "fae2mr86m_ms42dc_jaw_profile.json"
)
DIMENSIONS_PATH = (
    DATA_ROOT
    / "robot_description"
    / "estun_s3_60_g2"
    / "config"
    / "fae2mr86m_ms42dc_dimensions.json"
)
CONFIG_PATH = PACKAGE_ROOT / "config" / "jaw_scoring_fae2mr86m_ms42dc_cad_offline.yaml"


@pytest.fixture()
def profile() -> JawProfile:
    return JawProfile.from_files(PROFILE_PATH, DIMENSIONS_PATH)


def contact_transform(z_m: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = (0.12, -0.03, float(z_m))
    return transform


def test_envelope_sampling_includes_tip_bottom_and_body_corners(profile: JawProfile):
    fingers, body = sample_gripper_envelope_local(
        profile,
        state_fraction=0.5,
        finger_y_samples=3,
    )

    assert len(fingers) > 0
    assert np.max(fingers[:, 0]) == pytest.approx(profile.finger_tip_x_m)
    assert np.min(fingers[:, 2]) == pytest.approx(-0.5 * profile.finger_height_m)
    assert np.max(fingers[:, 2]) == pytest.approx(0.5 * profile.finger_height_m)
    assert body.shape == (8, 3)
    assert np.min(body[:, 0]) == pytest.approx(profile.body_envelope.rear_x_m)
    assert np.max(body[:, 0]) == pytest.approx(profile.body_envelope.root_x_m)


def test_high_upright_pose_passes_support_and_workspace_checks(profile: JawProfile):
    config = GripperClearanceConfig()
    result = evaluate_gripper_world_clearance(
        profile,
        contact_transform(0.20),
        width_m=0.040,
        depth_m=0.040,
        config=config,
    )

    assert result.feasible
    assert result.reason_codes == ()
    assert result.support_minimum_clearance_m >= config.minimum_clearance_m
    assert result.tcp_workspace_margin_m >= 0.0
    assert result.support_minimum_region in {"fingers", "body"}


def test_whole_gripper_is_rejected_even_when_tcp_is_above_workspace(profile: JawProfile):
    result = evaluate_gripper_world_clearance(
        profile,
        contact_transform(0.04),
        width_m=0.040,
        depth_m=0.040,
    )

    assert result.tcp_workspace_margin_m > 0.0
    assert not result.feasible
    assert result.reason_codes == ("table_clearance",)
    assert result.support_minimum_clearance_m < 0.020


def test_low_tcp_reports_below_workspace_and_table_clearance(profile: JawProfile):
    result = evaluate_gripper_world_clearance(
        profile,
        contact_transform(0.015),
        width_m=0.040,
        depth_m=0.040,
    )

    assert not result.feasible
    assert "below_workspace" in result.reason_codes
    assert "table_clearance" in result.reason_codes
    assert result.tcp_base_m[2] == pytest.approx(0.015, abs=1e-12)
    assert result.tcp_workspace_margin_m < 0.0


def test_clearance_config_loads_from_scoring_yaml():
    clearance = load_gripper_clearance_config(CONFIG_PATH)
    scoring = load_jaw_scoring_config(CONFIG_PATH)

    assert clearance.minimum_clearance_m == pytest.approx(0.020)
    assert clearance.workspace_min_z_m == pytest.approx(0.020)
    assert clearance.support_normal_base == pytest.approx((0.0, 0.0, 1.0))
    assert scoring.validated is False


def _candidate(z_m: float) -> JawGraspCandidate:
    return JawGraspCandidate(
        candidate_id="world_clearance_test",
        translation_m=(0.12, -0.03, float(z_m)),
        rotation=tuple(np.eye(3).reshape(-1)),
        width_m=0.040,
        depth_m=0.040,
        height_m=0.020,
        tolerance_m=0.020,
        network_score=0.90,
    )


def _valid_target(profile: JawProfile, candidate: JawGraspCandidate) -> np.ndarray:
    local_points: list[list[float]] = []
    contact_x = profile.contact_plane_x_m(candidate.depth_m)
    section = profile.section(contact_x, candidate.width_m)
    depths, inner_half, _ = profile.profile_arrays(
        state_fraction=section.state_fraction
    )
    for x in (-0.006, -0.003, 0.0, 0.003, 0.006):
        inner = float(np.interp(contact_x + x, depths, inner_half))
        for z in (-0.004, 0.0, 0.004):
            local_points.extend(
                (
                    [x, -inner, z],
                    [x, inner, z],
                    [x, -0.6 * inner, z],
                    [x, 0.0, z],
                    [x, 0.6 * inner, z],
                )
            )
    return (
        np.asarray(local_points, dtype=float) @ candidate.rotation_matrix.T
        + candidate.translation
    )


def test_apply_clearance_changes_final_feasibility_fail_closed(profile: JawProfile):
    scorer = JawAwareGraspScorer(profile)
    high = _candidate(0.20)
    high_score = scorer.score(high, _valid_target(profile, high))
    assert high_score.feasible
    accepted = apply_gripper_world_clearance(high_score, profile)
    assert accepted.feasible
    assert accepted.world_clearance is not None
    assert accepted.to_dict()["world_clearance"]["feasible"] is True

    low = _candidate(0.04)
    low_score = scorer.score(low, _valid_target(profile, low))
    assert low_score.feasible
    rejected = apply_gripper_world_clearance(low_score, profile)
    assert not rejected.feasible
    assert "table_clearance" in rejected.reason_codes
    assert rejected.to_dict()["world_clearance"]["feasible"] is False
