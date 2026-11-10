from pathlib import Path

import numpy as np
import pytest

from track1_grasp.jaw_grasp_scoring import (
    JawAwareGraspScorer,
    JawGraspCandidate,
    JawScoringConfig,
    jaw_symmetric_rotation_distance,
    load_jaw_scoring_config,
    nms_jaw_grasp_scores,
)
from track1_grasp.jaw_profile import JawProfile


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
IDENTITY = np.eye(3)
FLIP_X_180 = np.diag([1.0, -1.0, -1.0])


@pytest.fixture()
def profile() -> JawProfile:
    return JawProfile.from_files(PROFILE_PATH, DIMENSIONS_PATH)


def candidate(
    *,
    candidate_id: str = "cad_center",
    rotation: np.ndarray = IDENTITY,
    width_m: float = 0.040,
    tolerance_m: float = 0.020,
) -> JawGraspCandidate:
    return JawGraspCandidate(
        candidate_id=candidate_id,
        translation_m=(0.12, -0.03, 0.28),
        rotation=tuple(float(value) for value in rotation.reshape(-1)),
        width_m=width_m,
        depth_m=0.040,
        height_m=0.020,
        tolerance_m=tolerance_m,
        network_score=0.90,
    )


def to_source(local_points, grasp: JawGraspCandidate) -> np.ndarray:
    local = np.asarray(local_points, dtype=float)
    return local @ grasp.rotation_matrix.T + grasp.translation


def contact_target_local(
    profile: JawProfile,
    grasp: JawGraspCandidate,
) -> np.ndarray:
    points: list[list[float]] = []
    x_values = (-0.006, -0.003, 0.0, 0.003, 0.006)
    z_values = (-0.004, 0.0, 0.004)
    contact_x = profile.contact_plane_x_m(grasp.depth_m)
    section = profile.section(contact_x, grasp.width_m)
    depths, inner_half, _ = profile.profile_arrays(
        state_fraction=section.state_fraction
    )
    # Follow the V-shaped CAD inner surfaces at the selected opening state.
    for x in x_values:
        inner = float(np.interp(contact_x + x, depths, inner_half))
        for z in z_values:
            points.append([x, -inner, z])
            points.append([x, inner, z])
        # Cavity occupancy proves the object is actually between the fingers.
        for fraction in (-0.6, -0.3, 0.0, 0.3, 0.6):
            for z in z_values:
                points.append([x, fraction * inner, z])
    return np.asarray(points, dtype=float)


def test_cad_contact_grasp_is_feasible_and_maps_all_frames(profile: JawProfile):
    grasp = candidate()
    target = to_source(contact_target_local(profile, grasp), grasp)
    scorer = JawAwareGraspScorer(profile)

    result = scorer.score(grasp, target)

    assert result.feasible
    assert not result.reason_codes
    assert result.quality > 0.85
    assert result.inner_target_points >= scorer.config.min_inner_points
    assert result.contact_left_points >= scorer.config.min_contact_points_per_side
    assert result.contact_right_points >= scorer.config.min_contact_points_per_side
    assert result.target_penetration_points == 0
    # The V-shaped profile varies by about 2 mm over this reference band, so
    # the cloud-derived percentile need not equal the nominal centre width.
    assert result.measured_width_m == pytest.approx(0.040, abs=0.002)
    np.testing.assert_allclose(result.contact_transform[:3, 3], grasp.translation)
    np.testing.assert_allclose(
        result.tcp_transform,
        profile.graspnet_contact_to_tcp_transform(
            grasp.translation,
            grasp.rotation_matrix,
            grasp.depth_m,
        ),
    )


def test_empty_grasp_is_rejected(profile: JawProfile):
    grasp = candidate()
    shifted = to_source(
        contact_target_local(profile, grasp) + np.asarray([0.0, 0.20, 0.0]),
        grasp,
    )

    result = JawAwareGraspScorer(profile).score(grasp, shifted)

    assert not result.feasible
    assert result.inner_target_points == 0
    assert "empty_grasp" in result.reason_codes
    assert "missing_left_contact" in result.reason_codes
    assert "missing_right_contact" in result.reason_codes


def test_target_penetration_is_distinct_from_obstacle_collision(profile: JawProfile):
    grasp = candidate(width_m=0.030)
    penetration_local = []
    for x in (-0.004, 0.0, 0.004):
        for y in (-0.020, 0.020):
            for z in (-0.004, 0.0, 0.004):
                penetration_local.append([x, y, z])
    target = to_source(penetration_local, grasp)

    result = JawAwareGraspScorer(profile).score(grasp, target)

    assert not result.feasible
    assert result.target_penetration_points > 3
    assert "target_penetration" in result.reason_codes
    assert result.obstacle_finger_points == 0


def test_network_measurement_width_conflict_is_fail_closed(profile: JawProfile):
    # The synthetic target follows a 40 mm CAD opening.  A 56 mm network width
    # differs by 16 mm, beyond the configured 10 mm tolerance, and must not be
    # accepted merely because all contact and collision checks happen to pass.
    reference = candidate(width_m=0.040, tolerance_m=0.020)
    target = to_source(contact_target_local(profile, reference), reference)
    grasp = candidate(width_m=0.056, tolerance_m=0.020)

    result = JawAwareGraspScorer(profile).score(grasp, target)

    assert result.measured_width_m == pytest.approx(0.040, abs=0.002)
    assert "measured_width_conflict" in result.reason_codes
    assert result.quality_components["width"] == pytest.approx(0.0)
    assert not result.feasible


def test_unmeasured_width_is_fail_closed(profile: JawProfile):
    grasp = candidate()
    scorer = JawAwareGraspScorer(profile)

    result = scorer.score(grasp, np.empty((0, 3), dtype=float))

    assert "no_target_points" in result.reason_codes
    assert "width_unmeasured" in result.reason_codes
    assert not result.feasible


@pytest.mark.parametrize(
    ("obstacle_local", "expected_reason", "counter_name"),
    [
        ([[0.0, 0.030, -0.002], [0.002, 0.030, 0.0], [0.004, 0.030, 0.002]],
         "finger_collision", "obstacle_finger_points"),
        ([[0.0, 0.0, -0.002], [0.002, 0.0, 0.0], [0.004, 0.0, 0.002]],
         "cavity_obstruction", "obstacle_cavity_points"),
        ([[-0.056, 0.0, -0.002], [-0.054, 0.0, 0.0], [-0.052, 0.0, 0.002]],
         "body_collision", "obstacle_body_points"),
        ([[-0.155, 0.0, -0.002], [-0.154, 0.0, 0.0], [-0.153, 0.0, 0.002]],
         "approach_collision", "obstacle_approach_points"),
    ],
)
def test_obstacle_regions_are_scored_separately(
    profile: JawProfile,
    obstacle_local,
    expected_reason: str,
    counter_name: str,
):
    grasp = candidate()
    target = to_source(contact_target_local(profile, grasp), grasp)
    obstacle = to_source(obstacle_local, grasp)

    result = JawAwareGraspScorer(profile).score(grasp, target, obstacle)

    assert not result.feasible
    assert expected_reason in result.reason_codes
    assert getattr(result, counter_name) >= 3
    assert result.obstacle_clearance_m == pytest.approx(0.0)


def test_parallel_jaw_180_degree_symmetry_and_nms(profile: JawProfile):
    scorer = JawAwareGraspScorer(profile)
    first = candidate(candidate_id="first", tolerance_m=0.020)
    second = candidate(
        candidate_id="second",
        rotation=first.rotation_matrix @ FLIP_X_180,
        tolerance_m=0.010,
    )
    assert jaw_symmetric_rotation_distance(
        first.rotation_matrix, second.rotation_matrix
    ) == pytest.approx(0.0, abs=1e-7)
    first_score = scorer.score(
        first,
        to_source(contact_target_local(profile, first), first),
    )
    second_score = scorer.score(
        second,
        to_source(contact_target_local(profile, second), second),
    )
    assert first_score.feasible
    assert second_score.feasible

    kept, suppressed = nms_jaw_grasp_scores(
        [second_score, first_score], scorer.config
    )

    assert len(kept) == 1
    assert kept[0].candidate.candidate_id == "first"
    assert len(suppressed) == 1
    assert suppressed[0].candidate_id == "second"
    assert suppressed[0].kept_candidate_id == "first"


def test_scoring_config_is_explicitly_nonvalidated_and_loadable():
    config = load_jaw_scoring_config(CONFIG_PATH)
    assert isinstance(config, JawScoringConfig)
    assert config.name == "fae2mr86m_ms42dc_cad_offline_v1"
    assert config.validated is False
    assert config.contact_band_m == pytest.approx(0.007)
    assert config.max_collision_points == 2
    assert config.require_measured_width is True


def test_no_candidate_nms_is_safe():
    kept, suppressed = nms_jaw_grasp_scores([], JawScoringConfig())
    assert kept == []
    assert suppressed == []
