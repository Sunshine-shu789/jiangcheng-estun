import numpy as np
import pytest

from track1_grasp.support_plane import (
    SupportPlaneFitConfig,
    fit_support_plane_base,
    select_local_support_points,
)


def _plane_cloud(
    rng: np.random.Generator,
    count: int,
    *,
    a: float,
    b: float,
    c: float,
    noise_m: float = 0.00025,
) -> np.ndarray:
    xy = rng.uniform((-0.20, -0.20), (0.20, 0.20), size=(count, 2))
    z = c + a * xy[:, 0] + b * xy[:, 1]
    z += rng.normal(0.0, noise_m, size=count)
    return np.column_stack((xy, z))


def _test_config(**overrides) -> SupportPlaneFitConfig:
    values = {
        "ransac_threshold_m": 0.002,
        "ransac_iterations": 256,
        "max_fit_points": 5000,
        "min_inliers": 300,
        "min_inlier_ratio": 0.35,
        "max_rms_m": 0.0035,
        "min_normal_z": 0.80,
        "seed": 20261030,
    }
    values.update(overrides)
    return SupportPlaneFitConfig(**values)


def test_horizontal_plane_estimates_base_relative_height() -> None:
    rng = np.random.default_rng(11)
    points = _plane_cloud(rng, 3000, a=0.0, b=0.0, c=-0.040)

    estimate = fit_support_plane_base(points, _test_config())

    assert estimate.valid
    assert estimate.reason_codes == ()
    assert estimate.origin_base_m is not None
    assert estimate.normal_base_m is not None
    assert estimate.origin_base_m[2] == pytest.approx(-0.040, abs=1e-4)
    assert estimate.normal_base_m[2] > 0.999
    assert estimate.inlier_ratio > 0.95


def test_tilted_plane_normal_and_offset_are_reported() -> None:
    rng = np.random.default_rng(12)
    a, b, c = 0.030, -0.020, 0.050
    points = _plane_cloud(rng, 4000, a=a, b=b, c=c)
    expected_normal = np.asarray([-a, -b, 1.0], dtype=float)
    expected_normal /= np.linalg.norm(expected_normal)

    estimate = fit_support_plane_base(points, _test_config())

    assert estimate.valid
    assert estimate.normal_base_m is not None
    assert estimate.origin_base_m is not None
    np.testing.assert_allclose(estimate.normal_base_m, expected_normal, atol=2e-4)
    assert estimate.origin_base_m[2] == pytest.approx(c, abs=5e-4)
    assert estimate.tilt_deg is not None
    assert estimate.tilt_deg == pytest.approx(2.06, abs=0.03)


def test_vertical_box_wall_and_outliers_do_not_override_table() -> None:
    rng = np.random.default_rng(13)
    table = _plane_cloud(rng, 8000, a=0.0, b=0.0, c=-0.040)
    wall_yz = rng.uniform((-0.20, -0.08), (0.20, 0.20), size=(12000, 2))
    wall = np.column_stack(
        (np.full(len(wall_yz), 0.30), wall_yz[:, 0], wall_yz[:, 1])
    )
    outliers = rng.uniform((-0.20, -0.20, -0.02), (0.20, 0.20, 0.18), size=(400, 3))
    points = np.vstack((table, wall, outliers))
    config = _test_config(
        max_fit_points=20000,
        min_inlier_ratio=0.30,
        ransac_iterations=512,
    )

    estimate = fit_support_plane_base(points, config)

    assert estimate.valid
    assert estimate.reason_codes == ()
    assert estimate.normal_base_m is not None
    assert estimate.origin_base_m is not None
    assert estimate.normal_base_m[2] > 0.999
    assert estimate.origin_base_m[2] == pytest.approx(-0.040, abs=1e-4)


def test_insufficient_inliers_fail_closed() -> None:
    rng = np.random.default_rng(14)
    points = _plane_cloud(rng, 100, a=0.0, b=0.0, c=-0.040)
    config = _test_config(
        max_fit_points=1000,
        min_inliers=300,
        min_inlier_ratio=0.90,
        ransac_iterations=64,
    )

    estimate = fit_support_plane_base(points, config)

    assert not estimate.valid
    assert "support_plane_too_few_inliers" in estimate.reason_codes


def test_vertical_only_cloud_fails_closed_without_z_zero_fallback() -> None:
    rng = np.random.default_rng(15)
    yz = rng.uniform((-0.20, -0.10), (0.20, 0.20), size=(2000, 2))
    points = np.column_stack((np.full(len(yz), 0.30), yz[:, 0], yz[:, 1]))

    estimate = fit_support_plane_base(points, _test_config())

    assert not estimate.valid
    assert "support_plane_no_near_horizontal_plane" in estimate.reason_codes
    assert estimate.origin_base_m is None
    assert estimate.normal_base_m is None


def test_local_support_selection_uses_target_xy_center_radius() -> None:
    obstacles = np.asarray(
        [
            [0.01, 0.01, -0.04],
            [0.14, 0.00, -0.04],
            [0.30, 0.00, -0.04],
            [np.nan, 0.00, 0.00],
        ],
        dtype=float,
    )
    targets = np.asarray([[0.0, 0.0, 0.02], [0.02, 0.0, 0.03]])

    selected, center = select_local_support_points(
        obstacles,
        targets,
        local_xy_radius_m=0.16,
    )

    assert center == pytest.approx((0.01, 0.0))
    assert selected.shape == (2, 3)
    assert np.all(selected[:, 0] < 0.16)
