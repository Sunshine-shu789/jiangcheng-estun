from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from track1_grasp.config import (
    GraspGeometryPolicyConfig,
    apply_mode,
    load_config,
)
from track1_grasp.grasp_geometry import (
    analyze_grasp_geometry,
    rotation_from_tool_axis_and_closure,
    rotation_matrix_from_tool_axes,
    select_grasp_geometry,
)
from track1_grasp.kinematics import matrix_to_rpy, rpy_to_matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_CONFIG = (
    ROOT.parents[1] / "config" / "track1_task_cd_40class_v11_official.yaml"
)
SUPPORT_CENTER = np.asarray((0.0, 0.0, 0.4), dtype=float)
SUPPORT_NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=float)


def _policy(**overrides):
    values = {
        "enabled": True,
        "allow_rim": True,
        "allow_horizontal_sidewall": True,
    }
    values.update(overrides)
    return GraspGeometryPolicyConfig(**values)


def _evidence(points: np.ndarray, center=(0.0, 0.0, 0.36)):
    return analyze_grasp_geometry(
        points,
        object_center_camera_xyz=center,
        support_plane_center_camera_xyz=SUPPORT_CENTER,
        support_plane_normal_camera=SUPPORT_NORMAL,
        support_plane_rms_m=0.0005,
        support_plane_inlier_ratio=0.98,
    )


def _box_cloud() -> np.ndarray:
    rng = np.random.default_rng(1729)
    half_x, half_y, height = 0.07, 0.025, 0.04
    top = np.column_stack(
        (
            rng.uniform(-half_x, half_x, 3000),
            rng.uniform(-half_y, half_y, 3000),
            np.full(3000, SUPPORT_CENTER[2] - height),
        )
    )
    z = rng.uniform(SUPPORT_CENTER[2] - height, SUPPORT_CENTER[2], 1500)
    y = rng.uniform(-half_y, half_y, len(z))
    x = rng.uniform(-half_x, half_x, len(z))
    walls = (
        np.column_stack((np.full_like(y, half_x), y, z)),
        np.column_stack((np.full_like(y, -half_x), y, z)),
        np.column_stack((x, np.full_like(x, half_y), z)),
        np.column_stack((x, np.full_like(x, -half_y), z)),
    )
    return np.vstack((top, *walls))


def _cylinder_cloud(*, open_top: bool) -> np.ndarray:
    radius, height = 0.03, 0.08
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    z_values = np.linspace(
        SUPPORT_CENTER[2] - height + 0.001,
        SUPPORT_CENTER[2] - 0.001,
        101,
    )
    angle_grid, z_grid = np.meshgrid(angles, z_values)
    walls = np.column_stack(
        (
            radius * np.cos(angle_grid.ravel()),
            radius * np.sin(angle_grid.ravel()),
            z_grid.ravel(),
        )
    )
    rng = np.random.default_rng(2741)
    if open_top:
        rim = np.column_stack(
            (
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full_like(angles, SUPPORT_CENTER[2] - height),
            )
        )
        inner_radius = radius * np.sqrt(rng.random(1200)) * 0.4
        inner_angle = rng.uniform(0.0, 2.0 * np.pi, 1200)
        interior = np.column_stack(
            (
                inner_radius * np.cos(inner_angle),
                inner_radius * np.sin(inner_angle),
                np.full(1200, SUPPORT_CENTER[2] - height + 0.020),
            )
        )
        return np.vstack((walls, rim, interior))

    top_radius = radius * np.sqrt(rng.random(3000))
    top_angle = rng.uniform(0.0, 2.0 * np.pi, 3000)
    top = np.column_stack(
        (
            top_radius * np.cos(top_angle),
            top_radius * np.sin(top_angle),
            np.full(3000, SUPPORT_CENTER[2] - height),
        )
    )
    return np.vstack((walls, top))


def test_elongated_box_selects_only_vertical_short_edge():
    evidence = _evidence(_box_cloud(), center=(0.0, 0.0, 0.38))
    decision = select_grasp_geometry(evidence, _policy(), label="9-Mug")

    assert evidence.footprint_axis_ratio == pytest.approx(2.8, rel=0.12)
    assert evidence.short_edge_confidence >= 0.85
    assert [candidate.mode for candidate in decision.candidates] == [
        "vertical_short_edge"
    ]
    candidate = decision.candidates[0]
    assert candidate.tool_axis_camera[2] > 0.99
    assert candidate.closure_axis_camera == pytest.approx(
        evidence.footprint_minor_axis_camera
    )
    assert abs(
        float(
            np.dot(
                np.asarray(candidate.closure_axis_camera),
                np.asarray(evidence.footprint_major_axis_camera),
            )
        )
    ) < 1e-12


def test_short_edge_requires_a_valid_minor_axis():
    evidence = replace(
        _evidence(_box_cloud(), center=(0.0, 0.0, 0.38)),
        footprint_minor_axis_camera=None,
    )

    decision = select_grasp_geometry(evidence, _policy(), label="4-Frenchs_Mustard")

    assert "vertical_short_edge" not in {
        candidate.mode for candidate in decision.candidates
    }


def test_open_cup_selects_measured_rim_before_optional_sidewall():
    evidence = _evidence(_cylinder_cloud(open_top=True))
    decision = select_grasp_geometry(
        evidence,
        _policy(),
        label="not-a-known-container-label",
    )

    assert evidence.rim is not None
    assert evidence.rim.radius_m == pytest.approx(0.03, abs=0.001)
    assert evidence.rim.circularity >= 0.95
    assert evidence.rim.confidence >= 0.75
    assert decision.candidates[0].mode == "vertical_rim"
    assert {candidate.mode for candidate in decision.candidates} == {
        "vertical_rim",
        "horizontal_sidewall",
    }


def test_closed_cylinder_has_no_fake_rim_and_requires_sidewall_opt_in():
    evidence = _evidence(_cylinder_cloud(open_top=False))

    assert evidence.rim is None
    assert evidence.sidewall is not None
    assert evidence.sidewall.radius_m == pytest.approx(0.03, abs=0.001)
    assert evidence.sidewall.height_m == pytest.approx(0.08, abs=0.002)

    labels_do_not_create_modes = select_grasp_geometry(
        evidence,
        _policy(allow_horizontal_sidewall=False),
        label="38-Jumbo_Pencil_Cup",
    )
    assert labels_do_not_create_modes.candidates == ()
    assert "no measured open rim" in labels_do_not_create_modes.reason

    enabled = select_grasp_geometry(
        evidence,
        _policy(allow_horizontal_sidewall=True),
        label="8-Plastic_Banana",
    )
    assert [candidate.mode for candidate in enabled.candidates] == [
        "horizontal_sidewall"
    ]


def test_support_plane_is_required_and_nonfinite_points_are_filtered():
    points = _box_cloud()
    with_nan = np.vstack((points, np.full((10, 3), np.nan)))
    missing = analyze_grasp_geometry(
        with_nan,
        object_center_camera_xyz=(0.0, 0.0, 0.38),
    )

    assert missing.support_plane_valid is False
    assert select_grasp_geometry(missing, _policy()).candidates == ()


def test_degenerate_cloud_is_rejected_fail_closed():
    with pytest.raises(ValueError, match="degenerate|finite"):
        analyze_grasp_geometry(
            np.zeros((30, 3), dtype=float),
            object_center_camera_xyz=(0.0, 0.0, 0.36),
            support_plane_center_camera_xyz=SUPPORT_CENTER,
            support_plane_normal_camera=SUPPORT_NORMAL,
            support_plane_rms_m=0.0005,
            support_plane_inlier_ratio=0.98,
        )


def test_grasp_axes_form_right_handed_rotation_and_reject_parallel_axes():
    rotation = rotation_matrix_from_tool_axes((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="parallel"):
        rotation_matrix_from_tool_axes((0.0, 0.0, 1.0), (0.0, 0.0, -2.0))


def test_rotation_from_tool_axis_maps_configured_closure_axis():
    tilt_rad = np.radians(15.0)
    requested_tool_z = np.asarray((np.sin(tilt_rad), 0.0, -np.cos(tilt_rad)))
    closure_world = np.asarray((0.0, 1.0, 0.0))
    closure_axis_tool = np.asarray((3.0, 4.0))

    rotation = rotation_from_tool_axis_and_closure(
        requested_tool_z,
        closure_world,
        closure_axis_tool,
    )

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rotation[:, 2], requested_tool_z, atol=1e-12)
    np.testing.assert_allclose(
        rotation @ np.asarray((0.6, 0.8, 0.0)),
        closure_world,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rpy_to_matrix(matrix_to_rpy(rotation)),
        rotation,
        atol=1e-12,
    )


def test_geometry_config_requires_deferred_axis_rejection():
    base = load_config(FORMAL_CONFIG)
    short_edge = replace(
        base.task_cd.short_edge_grasp,
        geometry_policy=_policy(),
        pca=replace(
            base.task_cd.short_edge_grasp.pca,
            defer_axis_ratio_rejection=False,
        ),
    )
    config = replace(base, task_cd=replace(base.task_cd, short_edge_grasp=short_edge))

    with pytest.raises(ValueError, match="defer_axis_ratio_rejection=true"):
        config.validate()


@pytest.mark.parametrize(
    "overrides, message",
    (
        (
            {"horizontal_execution_enabled": True, "allow_horizontal_sidewall": False},
            "requires geometry_policy.enabled=true and allow_horizontal_sidewall=true",
        ),
        (
            {
                "horizontal_execution_enabled": True,
                "allow_horizontal_sidewall": True,
                "horizontal_route_verified": False,
            },
            "requires horizontal_route_verified=true",
        ),
        (
            {
                "horizontal_execution_enabled": True,
                "allow_horizontal_sidewall": True,
                "horizontal_route_verified": True,
                "horizontal_calibration_id": "",
            },
            "requires a non-empty horizontal_calibration_id",
        ),
    ),
)
def test_horizontal_execution_is_fail_closed(overrides, message):
    base = load_config(FORMAL_CONFIG)
    short_edge = replace(
        base.task_cd.short_edge_grasp,
        geometry_policy=_policy(**overrides),
        pca=replace(
            base.task_cd.short_edge_grasp.pca,
            defer_axis_ratio_rejection=True,
        ),
    )
    config = replace(base, task_cd=replace(base.task_cd, short_edge_grasp=short_edge))

    with pytest.raises(ValueError, match=message):
        config.validate()
