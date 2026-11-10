from math import acos, degrees
from pathlib import Path

import cv2
import numpy as np
import pytest

from track1_grasp import adaptive as adaptive_module
from track1_grasp.adaptive import SupportPlaneEstimate, estimate_object_pose
from track1_grasp.config import AdaptivePcaConfig
from track1_grasp.domain import CameraIntrinsics


IMAGE_HEIGHT = 600
IMAGE_WIDTH = 700
FIXTURES = Path(__file__).with_name("fixtures")


def _waist_config(
    *,
    min_mask_points: int = 400,
    min_path_px: float = 60.0,
) -> AdaptivePcaConfig:
    return AdaptivePcaConfig(
        min_valid_points=200,
        trim_percentile=0.0,
        waist_midline_enabled=True,
        waist_midline_min_mask_points=min_mask_points,
        waist_midline_min_path_px=min_path_px,
        support_plane_required=True,
    )


def _curved_banana_mask(
    *,
    flip_horizontal: bool = False,
    with_small_defects: bool = False,
) -> np.ndarray:
    """Return an asymmetric, tapered banana silhouette in pixel coordinates."""

    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    parameter = np.linspace(0.0, 1.0, 600)
    angle = np.deg2rad(180.0 + 180.0 * parameter)
    radius = 170.0 + 18.0 * np.sin(np.pi * parameter)
    center_u = 330.0 + radius * np.cos(angle)
    center_v = 300.0 + radius * np.sin(angle)
    centers = np.rint(np.column_stack((center_u, center_v))).astype(np.int32)
    widths = 48.0 * (1.0 + 0.7 * (parameter - 0.5))
    for point, width in zip(centers[::4], widths[::4], strict=True):
        cv2.circle(
            image,
            tuple(point),
            int(round(float(width) * 0.5)),
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
    mask = image > 127
    if with_small_defects:
        # The side notch does not reach the centreline.  The isolated speck must
        # be discarded by the largest-component cleanup.
        mask[89:100, 270:285] = False
        mask[540:550, 620:630] = True
    if flip_horizontal:
        mask = mask[:, ::-1]
    return mask


def _global_pca_minor_axis(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(float)
    centered = points - np.mean(points, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order[1]]


def _acute_axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = abs(float(np.dot(first, second)))
    return degrees(acos(float(np.clip(cosine, 0.0, 1.0))))


def _support_plane() -> SupportPlaneEstimate:
    return SupportPlaneEstimate(
        center_camera_xyz=(0.0, 0.0, 0.45),
        normal_camera=(0.0, 0.0, 1.0),
        rms_m=0.001,
        valid_points=1000,
        candidate_points=1000,
        inlier_ratio=0.9,
        median_residual_m=0.001,
    )


def test_builtin_zhang_suen_thinning_does_not_require_opencv_contrib():
    mask = np.zeros((100, 180), dtype=np.uint8)
    cv2.rectangle(mask, (20, 30), (159, 69), 255, -1)

    skeleton = adaptive_module._zhang_suen_thin(mask > 0)

    assert skeleton.dtype == bool
    assert skeleton.any()
    assert int(np.count_nonzero(skeleton)) < int(np.count_nonzero(mask))
    assert np.any(skeleton[48:52, 30:150])


def test_curved_banana_waist_uses_local_short_axis_not_global_pca_minor_axis():
    mask = _curved_banana_mask()
    estimate = adaptive_module._estimate_waist_midline_pixel(
        mask,
        _waist_config(),
    )
    ys, xs = np.nonzero(mask)
    global_centroid = np.asarray((xs.mean(), ys.mean()), dtype=float)
    local_short_axis = np.asarray(
        (-estimate.tangent_uv[1], estimate.tangent_uv[0]),
        dtype=float,
    )

    assert np.linalg.norm(np.asarray(estimate.center_uv) - global_centroid) > 60.0
    assert estimate.center_uv[1] < global_centroid[1] - 50.0
    assert abs(float(local_short_axis[1])) > 0.98
    assert (
        _acute_axis_angle_deg(
            local_short_axis,
            _global_pca_minor_axis(mask),
        )
        > 7.0
    )


def test_waist_geometry_is_stable_under_left_right_image_flip():
    mask = _curved_banana_mask()
    flipped = _curved_banana_mask(flip_horizontal=True)
    config = _waist_config()
    original = adaptive_module._estimate_waist_midline_pixel(mask, config)
    mirrored = adaptive_module._estimate_waist_midline_pixel(flipped, config)

    assert mirrored.center_uv[0] == pytest.approx(
        IMAGE_WIDTH - 1 - original.center_uv[0],
        abs=1.0,
    )
    assert mirrored.center_uv[1] == pytest.approx(original.center_uv[1], abs=1.0)
    reflected_tangent = np.asarray(
        (-original.tangent_uv[0], original.tangent_uv[1]),
        dtype=float,
    )
    assert abs(float(np.dot(reflected_tangent, mirrored.tangent_uv))) > 0.98
    assert mirrored.path_length_px == pytest.approx(
        original.path_length_px,
        rel=0.01,
    )
    assert mirrored.half_width_px == pytest.approx(
        original.half_width_px,
        abs=1.0,
    )


def test_straight_banana_remains_a_centreline_regression_case():
    mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.line(mask, (150, 200), (450, 200), 255, 50, lineType=cv2.LINE_AA)

    estimate = adaptive_module._estimate_waist_midline_pixel(
        mask > 127,
        _waist_config(),
    )

    assert estimate.center_uv == pytest.approx((300.0, 200.0), abs=6.0)
    assert abs(float(estimate.tangent_uv[0])) > 0.98
    assert abs(float(estimate.tangent_uv[1])) < 0.08
    assert estimate.path_length_px > 275.0
    assert estimate.half_width_px == pytest.approx(25.0, abs=3.0)


def test_small_notch_and_isolated_noise_do_not_move_the_waist():
    config = _waist_config()
    baseline = adaptive_module._estimate_waist_midline_pixel(
        _curved_banana_mask(),
        config,
    )
    noisy = adaptive_module._estimate_waist_midline_pixel(
        _curved_banana_mask(with_small_defects=True),
        config,
    )

    assert np.linalg.norm(
        np.asarray(noisy.center_uv) - np.asarray(baseline.center_uv)
    ) < 1.0
    assert abs(
        float(np.dot(noisy.tangent_uv, baseline.tangent_uv))
    ) > 0.99
    assert noisy.path_length_px == pytest.approx(
        baseline.path_length_px,
        rel=0.01,
    )
    assert noisy.half_width_px == pytest.approx(
        baseline.half_width_px,
        abs=1.0,
    )


def test_real_camera_mask_uses_waist_midpoint_and_measured_local_width():
    mask = cv2.imread(
        str(FIXTURES / "banana_waist_real_mask_000001.png"),
        cv2.IMREAD_GRAYSCALE,
    ) > 0
    estimate = adaptive_module._estimate_waist_midline_pixel(
        mask,
        _waist_config(),
    )
    ys, xs = np.nonzero(mask)
    global_centroid = np.asarray((xs.mean(), ys.mean()), dtype=float)
    local_short_axis = np.asarray(
        (-estimate.tangent_uv[1], estimate.tangent_uv[0]),
        dtype=float,
    )

    assert estimate.center_uv == pytest.approx((88.27, 72.10), abs=1.5)
    assert np.linalg.norm(np.asarray(estimate.center_uv) - global_centroid) > 8.0
    assert float(np.dot(local_short_axis, (0.328, 0.945))) > 0.98
    assert estimate.path_length_px == pytest.approx(154.3, abs=3.0)
    assert estimate.half_width_px == pytest.approx(21.0, abs=2.0)


def test_broken_annotated_mask_fails_closed_instead_of_using_thin_bridge():
    mask = cv2.imread(
        str(FIXTURES / "banana_waist_annotated_mask_000003.png"),
        cv2.IMREAD_GRAYSCALE,
    ) > 0

    with pytest.raises(ValueError, match="width is inconsistent"):
        adaptive_module._estimate_waist_midline_pixel(mask, _waist_config())


def test_waist_pose_integration_returns_finite_local_axes_on_support_plane():
    mask = _curved_banana_mask()
    ys, xs = np.nonzero(mask)
    bbox = (
        float(xs.min() - 2),
        float(ys.min() - 2),
        float(xs.max() + 3),
        float(ys.max() + 3),
    )
    depth = np.full(mask.shape, 0.40, dtype=float)
    intrinsics = CameraIntrinsics(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        fx=390.0,
        fy=390.0,
        cx=330.0,
        cy=300.0,
    )
    pixel_geometry = adaptive_module._estimate_waist_midline_pixel(
        mask,
        _waist_config(),
    )

    pose = estimate_object_pose(
        depth,
        bbox,
        intrinsics,
        _waist_config(),
        object_mask=mask,
        scene_support_plane=_support_plane(),
    )

    expected_minor_camera = np.asarray(
        (
            -pixel_geometry.tangent_uv[1] / intrinsics.fx,
            pixel_geometry.tangent_uv[0] / intrinsics.fy,
            0.0,
        ),
        dtype=float,
    )
    expected_minor_camera /= np.linalg.norm(expected_minor_camera)
    assert pose.axis_source == "waist_midline"
    assert pose.center_camera_xyz[2] == pytest.approx(0.45)
    assert pose.waist_midpoint_uv_px == pytest.approx(
        pixel_geometry.center_uv
    )
    assert pose.waist_path_length_px == pytest.approx(
        pixel_geometry.path_length_px
    )
    assert pose.waist_half_width_px == pytest.approx(
        pixel_geometry.half_width_px
    )
    assert abs(
        float(
            np.dot(
                np.asarray(pose.minor_axis_camera),
                expected_minor_camera,
            )
        )
    ) == pytest.approx(1.0, abs=1e-9)
    assert abs(float(np.asarray(pose.minor_axis_camera)[2])) < 1e-9


def test_waist_midline_fails_closed_without_mask_or_support_plane():
    depth = np.full((120, 160), 0.40, dtype=float)
    intrinsics = CameraIntrinsics(
        width=160,
        height=120,
        fx=200.0,
        fy=200.0,
        cx=80.0,
        cy=60.0,
    )
    with pytest.raises(
        ValueError,
        match="requires the target-colour object mask",
    ):
        estimate_object_pose(
            depth,
            (20.0, 20.0, 140.0, 100.0),
            intrinsics,
            _waist_config(),
        )

    mask = np.zeros(depth.shape, dtype=bool)
    mask[30:90, 40:120] = True
    config_without_required_plane = AdaptivePcaConfig(
        waist_midline_enabled=True,
        support_plane_required=False,
    )
    with pytest.raises(
        ValueError,
        match="requires a required support plane",
    ):
        estimate_object_pose(
            depth,
            (20.0, 20.0, 140.0, 100.0),
            intrinsics,
            config_without_required_plane,
            object_mask=mask,
        )


def test_waist_midline_rejects_empty_or_too_short_masks():
    with pytest.raises(ValueError, match="too few pixels"):
        adaptive_module._estimate_waist_midline_pixel(
            np.zeros((200, 240), dtype=bool),
            _waist_config(),
        )

    short_mask = np.zeros((200, 240), dtype=bool)
    short_mask[95:115, 100:140] = True
    with pytest.raises(ValueError, match="path is too short"):
        adaptive_module._estimate_waist_midline_pixel(
            short_mask,
            _waist_config(min_mask_points=300, min_path_px=60.0),
        )
