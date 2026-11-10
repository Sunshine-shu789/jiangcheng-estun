from pathlib import Path

import numpy as np
import pytest

from track1_grasp.jaw_profile import JawProfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    PACKAGE_ROOT.parents[1]
    / "robot_description"
    / "estun_s3_60_g2"
    / "config"
    / "fae2mr86m_ms42dc_jaw_profile.json"
)
DIMENSIONS_PATH = (
    PACKAGE_ROOT.parents[1]
    / "robot_description"
    / "estun_s3_60_g2"
    / "config"
    / "fae2mr86m_ms42dc_dimensions.json"
)


@pytest.fixture()
def profile() -> JawProfile:
    return JawProfile.from_files(PROFILE_PATH, DIMENSIONS_PATH)


def test_profile_loads_cad_tip_tcp_and_body_geometry(profile: JawProfile):
    assert profile.finger_tip_x_m == pytest.approx(0.086391)
    assert profile.tcp_x_m == pytest.approx(0.050)
    assert profile.finger_height_m == pytest.approx(0.020714, abs=1e-6)
    assert profile.body_envelope.rear_x_m < 0.0
    assert profile.body_envelope.root_x_m == pytest.approx(0.0)
    assert profile.body_envelope.half_width_m > 0.0
    assert profile.body_envelope.half_height_m > 0.0


@pytest.mark.parametrize(
    ("depth_m", "contact_x_m", "tcp_offset_x_m"),
    [
        (0.01, 0.076391, -0.026391),
        (0.04, 0.046391, 0.003609),
    ],
)
def test_contact_plane_and_tcp_offsets_cover_graspnet_depth_classes(
    profile: JawProfile,
    depth_m: float,
    contact_x_m: float,
    tcp_offset_x_m: float,
):
    assert profile.contact_plane_x_m(depth_m) == pytest.approx(contact_x_m)
    np.testing.assert_allclose(
        profile.tcp_offset_from_contact_m(depth_m),
        [tcp_offset_x_m, 0.0, 0.0],
        atol=1e-12,
    )


def test_tcp_forward_inverse_transform_round_trip(profile: JawProfile):
    angle = np.radians(31.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.asarray([0.11, -0.04, 0.37])

    tcp = profile.graspnet_contact_to_tcp_transform(
        translation,
        rotation,
        0.04,
    )
    recovered = profile.tcp_to_graspnet_contact_transform(tcp, 0.04)

    np.testing.assert_allclose(
        recovered,
        np.block(
            [
                [rotation, translation[:, None]],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tcp[:3, 3],
        translation + rotation @ np.asarray([0.003609, 0.0, 0.0]),
        atol=1e-12,
    )


def test_profile_rejects_non_positive_or_too_deep_graspnet_depth(profile: JawProfile):
    with pytest.raises(ValueError, match="positive"):
        profile.contact_plane_x_m(0.0)
    with pytest.raises(ValueError, match="exceeds"):
        profile.contact_plane_x_m(profile.finger_tip_x_m + 0.001)


def test_width_feasibility_uses_variable_cad_profile(profile: JawProfile):
    contact_x = profile.contact_plane_x_m(0.04)
    assert profile.width_feasibility(contact_x, 0.040)[0]
    width_ok, reasons = profile.width_feasibility(contact_x, 0.100)
    assert not width_ok
    assert "exceeds_profile_opening" in reasons
    width_ok, reasons = profile.width_feasibility(contact_x, 0.000)
    assert not width_ok
    assert "below_profile_closed_opening" in reasons


def test_section_reports_linear_approximation(profile: JawProfile):
    depth = profile.contact_plane_x_m(0.04)
    section = profile.section(depth, 0.040)
    assert section.approximate
    assert 0.0 < section.state_fraction < 1.0
    assert section.opening_m == pytest.approx(0.040)
    assert section.inner_half_width_m == pytest.approx(0.020)
    assert section.outer_half_width_m > section.inner_half_width_m
    assert section.finger_height_m == pytest.approx(profile.finger_height_m)
