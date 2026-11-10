from pathlib import Path

import numpy as np
import pytest

from track1_grasp.jaw_grasp_scoring import JawAwareGraspScorer, JawGraspCandidate
from track1_grasp.jaw_opening_search import (
    JawOpeningSearchConfig,
    search_jaw_opening,
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


@pytest.fixture()
def profile() -> JawProfile:
    return JawProfile.from_files(PROFILE_PATH, DIMENSIONS_PATH)


def candidate(*, reference_width_m: float = 0.056) -> JawGraspCandidate:
    return JawGraspCandidate(
        candidate_id="opening_search_test",
        translation_m=(0.12, -0.03, 0.28),
        rotation=tuple(np.eye(3, dtype=float).reshape(-1)),
        width_m=float(reference_width_m),
        depth_m=0.040,
        height_m=0.020,
        tolerance_m=0.020,
        network_score=0.90,
    )


def target_at_opening(
    profile: JawProfile,
    grasp: JawGraspCandidate,
    opening_m: float,
) -> np.ndarray:
    points: list[list[float]] = []
    contact_x = profile.contact_plane_x_m(grasp.depth_m)
    section = profile.section(contact_x, float(opening_m))
    depths, inner_half, _ = profile.profile_arrays(
        state_fraction=section.state_fraction
    )
    for x in (-0.006, -0.003, 0.0, 0.003, 0.006):
        inner = float(np.interp(contact_x + x, depths, inner_half))
        for z in (-0.004, 0.0, 0.004):
            points.extend(
                (
                    [x, -inner, z],
                    [x, inner, z],
                    [x, -0.6 * inner, z],
                    [x, 0.0, z],
                    [x, 0.6 * inner, z],
                )
            )
    return (
        np.asarray(points, dtype=float) @ grasp.rotation_matrix.T
        + grasp.translation
    )


def test_search_selects_opening_below_graspnet_reference_width(profile: JawProfile):
    grasp = candidate(reference_width_m=0.056)
    target = target_at_opening(profile, grasp, 0.040)
    scorer = JawAwareGraspScorer(profile)

    reference = scorer.score(grasp, target)
    result = search_jaw_opening(
        profile,
        scorer,
        grasp,
        target,
        config=JawOpeningSearchConfig(step_m=0.0005, max_trials=96),
    )

    # Legacy behavior remains fail-closed when the reference width is treated
    # as the contact width.
    assert not reference.feasible
    assert "measured_width_conflict" in reference.reason_codes

    assert result.status == "selected"
    assert result.best_feasible is not None
    selected = result.best_feasible
    assert 0.040 <= selected.opening_m <= 0.056
    assert selected.opening_m <= float(result.reference_score.measured_width_m) + 0.010
    assert selected.candidate.jaw_opening_m == pytest.approx(selected.opening_m)
    assert selected.candidate.width_m == pytest.approx(0.056)
    assert selected.score.feasible
    assert "measured_width_conflict" not in selected.score.reason_codes


def test_search_fails_closed_when_reference_width_is_below_measured_width(
    profile: JawProfile,
):
    reference = candidate(reference_width_m=0.056)
    target = target_at_opening(profile, reference, 0.040)
    narrow_reference = candidate(reference_width_m=0.030)

    result = search_jaw_opening(
        profile,
        JawAwareGraspScorer(profile),
        narrow_reference,
        target,
        config=JawOpeningSearchConfig(step_m=0.0005, max_trials=96),
    )

    assert result.status == "no_opening_bounds"
    assert result.feasible_trials == ()
    assert result.reference_candidate.width_m == pytest.approx(0.030)


def test_selected_opening_cannot_exceed_graspnet_reference_width():
    with pytest.raises(ValueError, match="must not exceed"):
        JawGraspCandidate(
            candidate_id="invalid",
            translation_m=(0.0, 0.0, 0.2),
            rotation=tuple(np.eye(3, dtype=float).reshape(-1)),
            width_m=0.030,
            selected_opening_m=0.031,
            depth_m=0.040,
        )
