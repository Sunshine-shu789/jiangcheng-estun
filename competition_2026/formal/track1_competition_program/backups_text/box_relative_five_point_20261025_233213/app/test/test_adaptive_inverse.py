from dataclasses import replace
from math import cos, pi, radians, sin
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from track1_grasp import factory as factory_module
from track1_grasp import vision as vision_module
from track1_grasp import adaptive as adaptive_module
from track1_grasp.adaptive import (
    SupportPlaneEstimate,
    _fit_support_plane_robust,
    _otsu_depth_split,
    _support_plane_rejection_reason,
    estimate_object_pose,
)
from track1_grasp.config import (
    AdaptiveCandidatesConfig,
    AdaptiveInverseConfig,
    AdaptiveSelectionConfig,
    TaskMode,
    VisionConfig,
    apply_mode,
    load_config,
)
from track1_grasp.detection_stabilizer import DetectionStabilizer
from track1_grasp.domain import (
    Detection,
    Frame,
    GripperState,
    MotionPlan,
    PickStatus,
    Pose,
    RobotState,
)
from track1_grasp.factory import build_pick_pipeline
from track1_grasp.planner import (
    AdaptiveGraspCandidate,
    PosePlanner,
    euler_to_matrix,
)


ROOT = Path(__file__).resolve().parents[3]
MOCK_CONFIG = ROOT / "app" / "track1_grasp" / "config" / "track1_mock.yaml"
FORMAL_CONFIG = (
    ROOT / "config" / "track1_task_b_final_20261020_live.yaml"
)
ADAPTIVE_MOCK_CONFIG = (
    ROOT / "config" / "track1_task_b_adaptive_mock_20261022.yaml"
)
ARBITRARY_MOCK_CONFIG = (
    ROOT / "config" / "track1_task_b_arbitrary_mock.yaml"
)
EXTRINSICS = ROOT / "config" / "camera_to_flange.d405_assumed.yaml"
GRASP_RPY = (pi, 0.0, 0.0)


def adaptive_config() -> object:
    base = apply_mode(load_config(MOCK_CONFIG), "B")
    candidates = replace(
        base.adaptive_inverse.candidates,
        max_count=3,
        use_visual_grasp_z=True,
        approach_height_m=0.08,
    )
    adaptive = replace(
        base.adaptive_inverse,
        enabled=True,
        require_validated_extrinsics=False,
        candidates=candidates,
    )
    return replace(
        base,
        task=replace(base.task, max_picks=1),
        vision=replace(
            base.vision,
            backend="mock",
            require_depth=True,
            scenario_file=base.vision.scenario_file,
        ),
        task_b=replace(
            base.task_b,
            place_poses=(),
            place_poses_file="",
            initial_pose_file="",
            rounds=1,
            return_to_pick=True,
            target_class="8-Plastic_Banana",
            place_names=("pick_1",),
            point_grasp_z_m=(),
            point_grasp_xy_from_observation=(),
            point_grasp_xy_m=(),
            point_grasp_xy_offset_m=(),
            point_grasp_rpy=(),
            point_vertical_grasp_descent=(),
            point_vertical_grasp_approach_height_m=(),
            point_observation_joints_deg=(),
            return_to_initial_after_each_pick=False,
            return_to_initial_after_task=False,
            operator_pause_after_each_pick=False,
            lift_hold_s=0.0,
        ),
        pose=replace(
            base.pose,
            camera_frame="flange",
            extrinsics_file=str(EXTRINSICS),
            use_detection_base_xyz=False,
            fallback_camera_to_base=False,
            robot_pose_frame="flange",
            gripper_tip_offset_flange_m=(0.0, 0.0, 0.165),
        ),
        adaptive_inverse=adaptive,
        motion=replace(
            base.motion,
            backend="mock",
            dry_run=True,
            workspace_min=(-0.65, -0.65, 0.02),
            workspace_max=(0.65, 0.65, 0.80),
            workspace_margin=0.01,
            grasp_z_m=0.166,
            grasp_rpy=GRASP_RPY,
            speed_mps=0.12,
        ),
        search=replace(
            base.search,
            enabled=False,
            observe_frames=1,
            confirm_hits=1,
        ),
        gripper=replace(base.gripper, backend="mock", open_before_approach=True),
    )


def adaptive_detection(
    *,
    major_axis_camera: tuple[float, float, float] = (1.0, 0.0, 0.0),
    held: bool = False,
) -> Detection:
    return Detection(
        label="8-Plastic_Banana",
        confidence=0.95,
        bbox=(220.0, 180.0, 360.0, 300.0),
        depth_m=0.08,
        camera_xyz=(0.02, 0.01, 0.08),
        object_center_camera_xyz=(0.02, 0.01, 0.08),
        major_axis_camera=major_axis_camera,
        minor_axis_camera=(0.0, 1.0, 0.0),
        major_axis_length_m=0.12,
        minor_axis_length_m=0.04,
        pose_confidence=0.95,
        attributes={"held": held},
    )


def robot_state(rpy: tuple[float, float, float] = GRASP_RPY) -> RobotState:
    return RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), rpy),
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        message="mock-ready",
    )


def vertical_plan(detection: Detection, *, x: float) -> MotionPlan:
    grasp = Pose((x, 0.49, 0.166), GRASP_RPY)
    high = Pose((x, 0.49, 0.246), GRASP_RPY)
    return MotionPlan(
        target=detection,
        grasp_pose=grasp,
        pregrasp_pose=high,
        lift_pose=high,
        transfer_pose=high,
        release_pose=grasp,
        retreat_pose=high,
        place_name="pick_1",
    )


def candidate(
    detection: Detection,
    *,
    x: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
):
    return AdaptiveGraspCandidate(
        plan=vertical_plan(detection, x=x),
        yaw_offset_deg=yaw_deg,
        pitch_offset_deg=pitch_deg,
        orientation_change_rad=radians(abs(yaw_deg)),
        object_center_base_xyz=(x, 0.49, 0.166),
        major_axis_base=(1.0, 0.0, 0.0),
    )


def synthetic_adaptive_pipeline(config=None):
    pipeline = build_pick_pipeline(config or adaptive_config())
    detection = adaptive_detection()
    state = pipeline.motion.get_state()
    pipeline._acquire_task_b_target = lambda *_args, **_kwargs: (
        detection,
        Frame("synthetic-adaptive", (detection,)),
        state,
        "synthetic adaptive detection",
    )
    return pipeline, detection


def make_object_depth(
    *,
    add_invalid_points: bool = False,
) -> tuple[np.ndarray, SimpleNamespace, tuple[int, int, int, int]]:
    depth = np.full((480, 640), np.nan, dtype=float)
    x0, y0, x1, y1 = 260, 210, 380, 270
    v, u = np.mgrid[y0:y1, x0:x1]
    intrinsics = SimpleNamespace(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    nominal_z = 0.40
    x_approx = (u - intrinsics.cx) / intrinsics.fx * nominal_z
    patch = nominal_z + 0.08 * x_approx
    if add_invalid_points:
        patch[::17, ::19] = np.nan
        patch[3::31, 5::37] = 1.50
        patch[0, 0] = -1.0
    depth[y0:y1, x0:x1] = patch
    return depth, intrinsics, (x0, y0, x1, y1)


def make_object_on_tilted_support_depth() -> tuple[
    np.ndarray, SimpleNamespace, tuple[int, int, int, int]
]:
    depth = np.full((480, 640), np.nan, dtype=float)
    x0, y0, x1, y1 = 200, 160, 420, 300
    v, u = np.mgrid[y0:y1, x0:x1]
    intrinsics = SimpleNamespace(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    ray_u = (u - intrinsics.cx) / intrinsics.fx
    ray_v = (v - intrinsics.cy) / intrinsics.fy
    support_z = 0.40 / (1.0 - 0.03 * ray_u + 0.02 * ray_v)
    depth[y0:y1, x0:x1] = support_z

    object_mask = (np.abs(u - 310) <= 50) & (np.abs(v - 230) <= 15)
    depth[y0:y1, x0:x1][object_mask] = support_z[object_mask] - 0.050
    return depth, intrinsics, (x0, y0, x1, y1)


def make_full_support_scene() -> tuple[
    np.ndarray,
    SimpleNamespace,
    tuple[int, int, int, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    height, width = 240, 320
    v, u = np.mgrid[0:height, 0:width]
    intrinsics = SimpleNamespace(fx=600.0, fy=600.0, cx=160.0, cy=120.0)
    ray_u = (u - intrinsics.cx) / intrinsics.fx
    ray_v = (v - intrinsics.cy) / intrinsics.fy
    support_z = 1.0 / (2.5 - 0.03 * ray_u + 0.02 * ray_v)
    depth = support_z.copy()

    bbox = (105, 96, 215, 144)
    object_mask = (
        ((u - 160.0) / 52.0) ** 2 + ((v - 120.0) / 14.0) ** 2
    ) <= 1.0
    depth[object_mask] = support_z[object_mask] - 0.050

    # A near transparent-wall return just outside the bbox must be rejected by
    # the support-depth window.  The yellow guard patch at the same depth as the
    # support must instead be rejected by the target-colour guard.
    wall_mask = np.zeros_like(object_mask)
    wall_mask[68:78, 110:210] = True
    depth[wall_mask] = 0.300
    color_guard_mask = object_mask.copy()
    color_guard_mask[110:124, 230:250] = True
    return (
        depth,
        intrinsics,
        bbox,
        object_mask,
        wall_mask,
        color_guard_mask,
    )


def make_low_profile_support_scene() -> tuple[
    np.ndarray,
    SimpleNamespace,
    tuple[int, int, int, int],
    np.ndarray,
]:
    height, width = 240, 320
    v, u = np.mgrid[0:height, 0:width]
    intrinsics = SimpleNamespace(fx=500.0, fy=500.0, cx=160.0, cy=120.0)
    ray_u = (u - intrinsics.cx) / intrinsics.fx
    ray_v = (v - intrinsics.cy) / intrinsics.fy
    support_z = 1.0 / (2.5 - 0.0003 * ray_u + 0.0002 * ray_v)
    depth = support_z.copy()

    bbox = (105, 90, 215, 150)
    object_mask = (
        ((u - 160.0) / 52.0) ** 2 + ((v - 120.0) / 14.0) ** 2
    ) <= 1.0
    # Only 8 mm of relief: deliberately below the 20 mm depth-mode gate.
    depth[object_mask] = support_z[object_mask] - 0.008

    # Isolated near returns must not affect the colour-guided object cloud.
    depth[95:100, 110:120] = 0.10

    # Colour guard pixels outside the bbox must stay out of support samples.
    colour_object_mask = object_mask.copy()
    colour_object_mask[80:90, 230:245] = True
    return depth, intrinsics, bbox, colour_object_mask


def test_depth_split_requires_two_modes() -> None:
    unimodal = np.linspace(0.20, 0.26, 20000)
    assert _otsu_depth_split(unimodal, min_cluster_points=100) is None

    bimodal = np.concatenate(
        (
            np.linspace(0.220, 0.235, 3000),
            np.linspace(0.265, 0.280, 7000),
        )
    )
    split = _otsu_depth_split(bimodal, min_cluster_points=100)

    assert split is not None
    threshold, separation = split
    assert 0.234 < threshold < 0.265
    assert separation > 0.040


def test_low_profile_object_uses_colour_guided_support_plane_fallback() -> None:
    depth, intrinsics, bbox, colour_object_mask = make_low_profile_support_scene()
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        trim_percentile=5.0,
        min_axis_ratio=1.1,
        support_plane_required=True,
        support_plane_min_points=100,
        support_plane_min_inlier_ratio=0.35,
    )
    x0, y0, x1, y1 = bbox
    patch = depth[y0:y1, x0:x1]
    assert _otsu_depth_split(
        patch[np.isfinite(patch)],
        max(
            int(config.min_valid_points),
            int(np.ceil(0.05 * np.count_nonzero(np.isfinite(patch)))),
        ),
    ) is None

    estimate = estimate_object_pose(
        depth,
        bbox,
        intrinsics,
        config,
        object_mask=colour_object_mask,
    )

    np.testing.assert_allclose(
        estimate.center_camera_xyz,
        (0.0, 0.0, 0.4),
        atol=2e-3,
    )
    assert abs(estimate.major_axis_camera[0]) > 0.99
    assert estimate.major_axis_length_m > estimate.minor_axis_length_m * 2.0
    assert estimate.support_plane_rms_m == pytest.approx(0.0, abs=5e-5)
    assert estimate.support_plane_normal_camera is not None
    assert abs(estimate.support_plane_normal_camera[0]) < 0.01
    assert abs(estimate.support_plane_normal_camera[1]) < 0.01
    assert estimate.support_plane_valid_points >= 100
    assert estimate.support_plane_inlier_ratio > 0.95


def test_low_profile_fallback_still_rejects_without_a_real_support_plane() -> None:
    depth, intrinsics, bbox, colour_object_mask = make_low_profile_support_scene()
    object_only = np.full_like(depth, np.nan)
    object_only[np.isfinite(depth) & (depth < 0.395)] = depth[
        np.isfinite(depth) & (depth < 0.395)
    ]
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        support_plane_required=True,
        support_plane_min_points=100,
    )

    with pytest.raises(ValueError, match="no reliable support plane"):
        estimate_object_pose(
            object_only,
            bbox,
            intrinsics,
            config,
            object_mask=colour_object_mask,
        )


def test_support_plane_diagnostics_and_full_frame_guard_masks(monkeypatch) -> None:
    (
        depth,
        intrinsics,
        bbox,
        _object_mask,
        wall_mask,
        color_guard_mask,
    ) = make_full_support_scene()
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        trim_percentile=2.0,
        support_plane_required=True,
        support_plane_min_points=100,
        support_plane_min_inlier_ratio=0.35,
    )
    captured_masks: list[np.ndarray] = []
    original = adaptive_module._masked_support_plane

    def recording_fit(depth_arg, mask, intrinsics_arg, config_arg):
        captured_masks.append(np.asarray(mask, dtype=bool).copy())
        return original(depth_arg, mask, intrinsics_arg, config_arg)

    monkeypatch.setattr(
        adaptive_module,
        "_masked_support_plane",
        recording_fit,
    )

    estimate = estimate_object_pose(
        depth,
        bbox,
        intrinsics,
        config,
        object_mask=color_guard_mask,
    )

    assert captured_masks
    support_ring = captured_masks[0]
    assert support_ring.shape == depth.shape
    assert not np.any(support_ring[wall_mask])
    assert not np.any(support_ring[color_guard_mask & ~_object_mask])
    assert np.any(support_ring[145:155, 70:90])
    assert estimate.support_plane_rms_m == pytest.approx(0.0, abs=5e-5)
    assert estimate.support_plane_normal_camera is not None
    assert 0.0 < estimate.support_plane_normal_camera[2] <= 1.0
    assert abs(estimate.support_plane_normal_camera[0]) < 0.05
    assert abs(estimate.support_plane_normal_camera[1]) < 0.05
    assert estimate.support_plane_valid_points >= 100
    assert estimate.support_plane_candidate_points >= (
        estimate.support_plane_valid_points
    )
    assert estimate.support_plane_inlier_ratio > 0.95
    np.testing.assert_allclose(
        estimate.center_camera_xyz,
        (0.0, 0.0, 0.4),
        atol=2e-3,
    )


def test_support_plane_rejection_covers_quality_and_orientation_gates() -> None:
    config = load_config(MOCK_CONFIG).adaptive_inverse.pca
    good = SupportPlaneEstimate(
        center_camera_xyz=(0.0, 0.0, 0.4),
        normal_camera=(0.0, 0.0, 1.0),
        rms_m=0.001,
        valid_points=1000,
        candidate_points=2000,
        inlier_ratio=0.5,
        median_residual_m=0.0005,
    )

    assert _support_plane_rejection_reason(good, config) == ""
    assert "inliers" in _support_plane_rejection_reason(
        replace(good, valid_points=10),
        config,
    )
    assert "inlier ratio" in _support_plane_rejection_reason(
        replace(good, inlier_ratio=0.1),
        config,
    )
    assert "RMS" in _support_plane_rejection_reason(
        replace(good, rms_m=0.01),
        config,
    )
    assert "normal" in _support_plane_rejection_reason(
        replace(good, normal_camera=(1.0, 0.0, 0.0)),
        config,
    )


def test_robust_support_fit_rejects_dominant_vertical_wall() -> None:
    rng = np.random.default_rng(7)
    table_xy = rng.uniform((-0.1, -0.1), (0.1, 0.1), size=(800, 2))
    table = np.column_stack((table_xy, np.full(800, 0.4)))
    wall_yz = rng.uniform((-0.1, 0.38), (0.1, 0.42), size=(1600, 2))
    wall = np.column_stack((np.full(1600, 0.2), wall_yz))
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        support_plane_ransac_iterations=96,
        support_plane_max_samples=3000,
    )

    estimate = _fit_support_plane_robust(np.vstack((table, wall)), config)

    assert estimate is not None
    assert abs(estimate.normal_camera[0]) > 0.95
    assert "normal" in _support_plane_rejection_reason(estimate, config)


def test_required_support_plane_rejects_a_flat_object_only_patch() -> None:
    depth = np.full((120, 160), np.nan, dtype=float)
    v, u = np.mgrid[40:80, 40:120]
    depth[40:80, 40:120] = 0.35 + 0.02 * (u - 80.0) / 80.0
    intrinsics = SimpleNamespace(fx=300.0, fy=300.0, cx=80.0, cy=60.0)
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        support_plane_required=True,
        support_plane_min_points=50,
    )

    with pytest.raises(ValueError, match="no reliable support plane"):
        estimate_object_pose(depth, (40, 40, 120, 80), intrinsics, config)


def test_pca_ignores_larger_background_and_projects_to_support_plane() -> None:
    depth, intrinsics, bbox = make_object_on_tilted_support_depth()
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        trim_percentile=5.0,
        min_axis_ratio=1.10,
    )

    estimate = estimate_object_pose(depth, bbox, intrinsics, config)

    ray_u = (310.0 - intrinsics.cx) / intrinsics.fx
    ray_v = (230.0 - intrinsics.cy) / intrinsics.fy
    expected_support_z = 0.40 / (1.0 - 0.03 * ray_u + 0.02 * ray_v)
    np.testing.assert_allclose(
        estimate.center_camera_xyz,
        (ray_u * expected_support_z, ray_v * expected_support_z, expected_support_z),
        atol=1e-3,
    )
    assert abs(estimate.major_axis_camera[0]) > 0.99
    assert estimate.major_axis_length_m > 0.050
    assert estimate.minor_axis_length_m < 0.030


def test_pca_estimates_tilted_object_center_axes_and_lengths():
    depth, intrinsics, bbox = make_object_depth()
    config = load_config(MOCK_CONFIG).adaptive_inverse.pca
    config = replace(
        config,
        min_valid_points=100,
        min_axis_ratio=1.10,
        trim_percentile=0.0,
    )

    estimate = estimate_object_pose(depth, bbox, intrinsics, config)

    np.testing.assert_allclose(estimate.center_camera_xyz, (0.0, 0.0, 0.4), atol=2e-3)
    assert abs(estimate.major_axis_camera[0]) > 0.995
    assert abs(estimate.major_axis_camera[2]) > 0.03
    assert estimate.major_axis_length_m == pytest.approx(0.080, abs=0.003)
    assert estimate.minor_axis_length_m == pytest.approx(0.040, abs=0.003)
    assert estimate.major_axis_length_m > estimate.minor_axis_length_m * 1.7
    assert estimate.pose_confidence > 0.50


def test_pca_rejects_invalid_depth_and_filters_background_outliers():
    depth, intrinsics, bbox = make_object_depth(add_invalid_points=True)
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        trim_percentile=2.0,
    )

    estimate = estimate_object_pose(depth, bbox, intrinsics, config)

    np.testing.assert_allclose(estimate.center_camera_xyz, (0.0, 0.0, 0.4), atol=2e-3)
    assert estimate.valid_points < (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    assert estimate.depth_inlier_ratio < 1.0

    with pytest.raises(ValueError, match="valid points"):
        estimate_object_pose(
            np.full((50, 50), np.nan),
            (0, 0, 50, 50),
            intrinsics,
            replace(config, min_valid_points=3),
        )


def test_stabilizer_resolves_axis_sign_flip_and_smooths_pose():
    config = VisionConfig(
        backend="mock",
        show=False,
        continuous=False,
        min_confidence=0.35,
        stabilizer_inference_min_confidence=0.10,
        stabilizer_confirm_hits=1,
        stabilizer_depth_alpha=0.25,
    )
    tracker = DetectionStabilizer(config)
    first = adaptive_detection()
    angle = radians(10.0)
    flipped = replace(
        adaptive_detection(
            major_axis_camera=(-cos(angle), -sin(angle), 0.0)
        ),
        object_center_camera_xyz=(0.03, 0.01, 0.10),
        major_axis_length_m=0.16,
    )

    stable_first = tracker.update((first,), 640, 480)[0]
    stable_second = tracker.update((flipped,), 640, 480)[0]

    first_axis = np.asarray(stable_first.major_axis_camera)
    second_axis = np.asarray(stable_second.major_axis_camera)
    assert float(np.dot(first_axis, second_axis)) > 0.99
    assert second_axis[0] > 0.99
    assert stable_second.object_center_camera_xyz == pytest.approx(
        (0.0225, 0.01, 0.085)
    )
    assert stable_second.major_axis_length_m == pytest.approx(0.13)


def test_adaptive_candidates_follow_object_axis_and_respect_max_count():
    config = adaptive_config()
    planner = PosePlanner(config)
    angle = radians(10.0)
    detection = adaptive_detection(
        major_axis_camera=(cos(angle), sin(angle), 0.0)
    )

    candidates = planner.plan_adaptive_candidates(
        detection,
        robot_state((0.0, 0.0, 0.0)),
        grasp_rpy=(0.0, 0.0, 0.0),
    )
    flipped = planner.plan_adaptive_candidates(
        adaptive_detection(
            major_axis_camera=(-cos(angle), -sin(angle), 0.0)
        ),
        robot_state((0.0, 0.0, 0.0)),
        grasp_rpy=(0.0, 0.0, 0.0),
    )

    assert len(candidates) == 3
    assert [item.yaw_offset_deg for item in candidates] == pytest.approx(
        [10.0, 55.0, -35.0]
    )
    assert len(flipped) == 3
    assert flipped[0].yaw_offset_deg == pytest.approx(10.0)
    for item in candidates:
        plan = item.plan
        assert plan.pregrasp_pose.xyz[:2] == pytest.approx(
            plan.grasp_pose.xyz[:2]
        )
        assert plan.pregrasp_pose.rpy == pytest.approx(plan.grasp_pose.rpy)
        assert (
            plan.pregrasp_pose.xyz[2] - plan.grasp_pose.xyz[2]
        ) == pytest.approx(0.08)


def test_adaptive_candidate_converts_observed_jaw_tip_point_to_flange_target():
    config = adaptive_config()
    planner = PosePlanner(config)
    # With flange RPY=(pi, 0, 0) and flange Z=0.300 m, this camera point maps
    # to base XYZ=(0.100, 0.010, 0.00224) m, the static-calibration tabletop
    # object-centre height.
    detection = replace(
        adaptive_detection(major_axis_camera=(1.0, 0.0, 0.0)),
        object_center_camera_xyz=(0.100, 0.055, 0.29776),
        camera_xyz=(0.100, 0.055, 0.29776),
    )

    candidates = planner.plan_adaptive_candidates(
        detection,
        robot_state(),
        grasp_rpy=GRASP_RPY,
    )

    assert candidates
    first = candidates[0]
    assert first.object_center_base_xyz == pytest.approx(
        (0.100, 0.010, 0.00224), abs=1e-9
    )
    # The +Z flange offset becomes -Z in base for a top grasp.  RobotMove must
    # receive the flange pose, not the tabletop point returned by vision.
    assert first.plan.grasp_pose.xyz == pytest.approx(
        (0.100, 0.010, 0.16724), abs=1e-9
    )
    assert first.plan.pregrasp_pose.xyz == pytest.approx(
        (0.100, 0.010, 0.24724), abs=1e-9
    )
    assert first.plan.pregrasp_pose.xyz[:2] == pytest.approx(
        first.plan.grasp_pose.xyz[:2]
    )
    assert first.plan.pregrasp_pose.rpy == pytest.approx(
        first.plan.grasp_pose.rpy
    )


def test_pitch_candidates_follow_safety_order_and_preserve_the_observed_tip():
    base = adaptive_config()
    clearance_m = 0.003
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            candidates=replace(
                base.adaptive_inverse.candidates,
                max_count=4,
                yaw_offsets_deg=(0.0, 45.0, -45.0, 90.0, -90.0),
                pitch_offsets_deg=(0.0, -20.0, -16.0),
                jaw_tip_clearance_m=clearance_m,
            ),
        ),
    )
    planner = PosePlanner(config)
    angle = radians(10.0)
    detection = adaptive_detection(
        major_axis_camera=(cos(angle), sin(angle), 0.0)
    )

    candidates = planner.plan_adaptive_candidates(
        detection,
        robot_state(),
        grasp_rpy=(0.0, 0.0, 0.0),
    )

    # The primary object-axis yaw receives all configured wrist fallbacks
    # before the next heading is considered.
    assert [
        (round(item.yaw_offset_deg, 6), round(item.pitch_offset_deg, 6))
        for item in candidates
    ] == [
        (-10.0, 0.0),
        (-10.0, -20.0),
        (-10.0, -16.0),
        (35.0, 0.0),
    ]
    tip_offset = np.asarray(config.pose.gripper_tip_offset_flange_m)
    for item in candidates:
        rotation = euler_to_matrix(item.plan.grasp_pose.rpy)
        commanded_tip = np.asarray(item.plan.grasp_pose.xyz) + rotation @ tip_offset
        expected_tip = np.asarray(item.object_center_base_xyz) + np.array(
            (0.0, 0.0, clearance_m)
        )
        assert commanded_tip == pytest.approx(expected_tip, abs=1e-9)


def test_adaptive_candidate_applies_configured_jaw_tip_clearance():
    base = adaptive_config()
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            candidates=replace(
                base.adaptive_inverse.candidates,
                jaw_tip_clearance_m=0.002,
            ),
        ),
    )
    planner = PosePlanner(config)
    detection = replace(
        adaptive_detection(major_axis_camera=(1.0, 0.0, 0.0)),
        object_center_camera_xyz=(0.100, 0.055, 0.29776),
        camera_xyz=(0.100, 0.055, 0.29776),
    )

    candidates = planner.plan_adaptive_candidates(
        detection,
        robot_state(),
        grasp_rpy=GRASP_RPY,
    )

    assert candidates
    first = candidates[0]
    assert first.plan.grasp_pose.xyz == pytest.approx(
        (0.100, 0.010, 0.16924), abs=1e-9
    )
    assert first.plan.pregrasp_pose.xyz == pytest.approx(
        (0.100, 0.010, 0.24924), abs=1e-9
    )


def test_jaw_tip_clearance_is_parsed_and_rejects_invalid_values():
    parsed = AdaptiveCandidatesConfig.from_dict({"jaw_tip_clearance_m": 0.002})
    assert parsed.jaw_tip_clearance_m == pytest.approx(0.002)

    for invalid_value in (-1e-4, float("nan")):
        base = adaptive_config()
        invalid = replace(
            base,
            adaptive_inverse=replace(
                base.adaptive_inverse,
                candidates=replace(
                    base.adaptive_inverse.candidates,
                    jaw_tip_clearance_m=invalid_value,
                ),
            ),
        )
        with pytest.raises(ValueError, match="jaw_tip_clearance_m"):
            invalid.validate()


def test_pitch_offsets_and_first_valid_selection_are_parsed_and_validated():
    candidates = AdaptiveCandidatesConfig.from_dict(
        {"pitch_offsets_deg": [0.0, -20.0, -16.0]}
    )
    selection = AdaptiveSelectionConfig.from_dict(
        {"stop_after_first_valid": True}
    )
    assert candidates.pitch_offsets_deg == pytest.approx((0.0, -20.0, -16.0))
    assert selection.stop_after_first_valid is True

    for invalid_offsets in ((), (float("nan"),), (-91.0,), (91.0,)):
        base = adaptive_config()
        invalid = replace(
            base,
            adaptive_inverse=replace(
                base.adaptive_inverse,
                candidates=replace(
                    base.adaptive_inverse.candidates,
                    pitch_offsets_deg=invalid_offsets,
                ),
            ),
        )
        with pytest.raises(ValueError, match="pitch_offsets_deg"):
            invalid.validate()

    with pytest.raises(
        ValueError,
        match="pitch_offsets_deg must be a sequence",
    ):
        AdaptiveCandidatesConfig.from_dict({"pitch_offsets_deg": "0,-20"})


def test_visual_grasp_without_gripper_tip_offset_is_rejected():
    config = replace(
        adaptive_config(),
        pose=replace(
            adaptive_config().pose,
            gripper_tip_offset_flange_m=None,
        ),
    )

    with pytest.raises(
        ValueError,
        match="gripper_tip_offset_flange_m",
    ):
        config.validate()


def test_visual_grasp_z_is_not_overridden_by_point_or_global_height():
    config = adaptive_config()
    planner = PosePlanner(config)
    detection = replace(adaptive_detection(), base_xyz=(0.10, 0.20, 0.123))
    anchor = Pose((0.10, 0.20, 0.50), GRASP_RPY)

    plan = planner.plan(
        detection,
        robot_state(),
        place_name="pick_1",
        return_to_pick=True,
        grasp_z_m=0.40,
        grasp_rpy=GRASP_RPY,
        task_b_anchor_pose=anchor,
        grasp_xyz_override_m=(0.10, 0.20, 0.123),
        vertical_grasp_descent=True,
        vertical_grasp_approach_height_m=0.08,
        use_visual_grasp_z=True,
    )

    assert plan.grasp_pose.xyz == pytest.approx((0.10, 0.20, 0.123))
    assert plan.pregrasp_pose.xyz == pytest.approx((0.10, 0.20, 0.203))
    assert plan.release_pose.xyz == pytest.approx(plan.grasp_pose.xyz)


def test_endpoint_condition_filter_rejects_expensive_singular_candidate():
    base = adaptive_config()
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            selection=replace(
                base.adaptive_inverse.selection,
                stop_after_first_valid=True,
            ),
        ),
    )
    pipeline, detection = synthetic_adaptive_pipeline(config)
    pipeline.planner.plan_adaptive_candidates = lambda *_args, **_kwargs: (
        candidate(detection, x=0.10),
        candidate(detection, x=0.20, pitch_deg=-20.0),
    )
    condition_sigmas = iter((0.005, 0.050))
    condition_calls: list[float | None] = []

    def condition_check(_pose, *, label):
        assert "adaptive candidate" in label
        sigma = next(condition_sigmas)
        condition_calls.append(sigma)
        return sigma

    pipeline.motion.precheck_pose_condition = condition_check
    original = pipeline.motion.precheck_linear_targets
    route_calls: list[tuple[Pose, ...]] = []

    def precheck(targets, *, step_m):
        route_calls.append(tuple(targets))
        return original(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = precheck
    result = pipeline.run_once()

    assert result.success
    assert condition_calls == [0.005, 0.050]
    assert len(route_calls) == 1
    assert pipeline.motion.moves[0][1].xyz[0] == pytest.approx(0.20)
    event_text = " | ".join(event.message for event in result.events)
    assert "pregrasp endpoint minimum Jacobian sigma" in event_text
    assert "below required" in event_text


@pytest.mark.parametrize(
    ("stop_after_first_valid", "expected_precheck_calls"),
    ((True, 1), (False, 2)),
)
def test_first_valid_selection_can_stop_after_one_full_route(
    stop_after_first_valid,
    expected_precheck_calls,
):
    base = adaptive_config()
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            selection=replace(
                base.adaptive_inverse.selection,
                stop_after_first_valid=stop_after_first_valid,
            ),
        ),
    )
    pipeline, detection = synthetic_adaptive_pipeline(config)
    pipeline.planner.plan_adaptive_candidates = lambda *_args, **_kwargs: (
        candidate(detection, x=0.10, pitch_deg=-16.0),
        candidate(detection, x=0.20, yaw_deg=45.0),
    )
    original = pipeline.motion.precheck_linear_targets
    calls: list[tuple[Pose, ...]] = []

    def precheck(targets, *, step_m):
        calls.append(tuple(targets))
        return original(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = precheck
    result = pipeline.run_once()

    assert result.success
    assert len(calls) == expected_precheck_calls
    assert pipeline.motion.moves
    assert pipeline.motion.moves[0][1].xyz[0] == pytest.approx(0.10)
    event_text = " | ".join(event.message for event in result.events)
    assert "pitch_offset=-16.00 deg" in event_text
    if stop_after_first_valid:
        assert "selection=first_safe_candidate" in event_text
    else:
        assert "selection=ranked" in event_text


def test_first_failed_adaptive_candidate_falls_back_to_second_route():
    pipeline, detection = synthetic_adaptive_pipeline()
    first = candidate(detection, x=0.10)
    second = candidate(detection, x=0.20, yaw_deg=45.0)
    pipeline.planner.plan_adaptive_candidates = lambda *_args, **_kwargs: (
        first,
        second,
    )
    original = pipeline.motion.precheck_linear_targets
    calls: list[tuple[Pose, ...]] = []

    def precheck(targets, *, step_m):
        calls.append(tuple(targets))
        if len(calls) == 1:
            raise RuntimeError("synthetic first-candidate rejection")
        return original(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = precheck
    result = pipeline.run_once()

    assert result.success
    assert len(calls) == 2
    assert pipeline.motion.moves
    assert all(name for name, _pose in pipeline.motion.moves)
    assert pipeline.motion.moves[0][1].xyz[0] == pytest.approx(0.20)


def test_all_adaptive_candidates_rejected_before_any_arm_or_gripper_action():
    pipeline, detection = synthetic_adaptive_pipeline()
    pipeline.planner.plan_adaptive_candidates = lambda *_args, **_kwargs: (
        candidate(detection, x=0.10),
        candidate(detection, x=0.20, yaw_deg=45.0),
    )

    def reject(_targets, *, step_m):
        raise RuntimeError("synthetic full adaptive route rejection")

    pipeline.motion.precheck_linear_targets = reject
    result = pipeline.run_once()

    assert result.status is PickStatus.REJECTED
    assert "all 2 adaptive grasp candidate" in result.message
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []


def test_adaptive_detection_rejects_held_track():
    pipeline = build_pick_pipeline(adaptive_config())

    assert pipeline._is_usable(adaptive_detection())
    assert not pipeline._is_usable(adaptive_detection(held=True))


def test_adaptive_config_requires_validated_extrinsics_and_known_pose_frame():
    base = adaptive_config()
    requiring_validation = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            require_validated_extrinsics=True,
        ),
    )
    with pytest.raises(ValueError, match="validated extrinsics"):
        requiring_validation.validate()

    unknown_frame = replace(
        base,
        pose=replace(base.pose, robot_pose_frame="unknown"),
    )
    with pytest.raises(ValueError, match="robot_pose_frame"):
        unknown_frame.validate()


def test_arbitrary_placement_visual_grasp_requires_support_plane():
    base = load_config(ARBITRARY_MOCK_CONFIG)
    arbitrary = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            arbitrary_placement=True,
            pca=replace(
                base.adaptive_inverse.pca,
                support_plane_required=False,
            ),
        ),
    )

    with pytest.raises(ValueError, match="support_plane_required=true"):
        arbitrary.validate()

    required = replace(
        arbitrary,
        adaptive_inverse=replace(
            arbitrary.adaptive_inverse,
            pca=replace(
                arbitrary.adaptive_inverse.pca,
                support_plane_required=True,
            ),
        ),
    )
    required.validate()


def test_formal_five_point_config_keeps_adaptive_disabled_and_parameters_unchanged():
    config = load_config(FORMAL_CONFIG)

    assert config.task.mode is TaskMode.B
    assert config.adaptive_inverse.enabled is False
    assert config.task_b.point_grasp_xy_offset_m == (
        (0.0, 0.0),
        (0.040, 0.0),
        (0.040, -0.015),
        (0.0, -0.030),
        (-0.020, -0.030),
    )
    assert config.motion.speed_mps == pytest.approx(0.12)
    assert config.motion.joint_speed_rad_s == pytest.approx(0.6)
    assert config.motion.joint_speed_safety_limit_rad_s == pytest.approx(0.7)
    assert config.motion.workspace_min == pytest.approx((-0.65, -0.65, 0.02))
    assert config.motion.workspace_max == pytest.approx((0.65, 0.65, 0.80))
    assert config.motion.cartesian_min_jacobian_singular_value == pytest.approx(0.01)


def test_adaptive_mock_five_point_config_is_offline_only():
    config = load_config(ADAPTIVE_MOCK_CONFIG)
    config.validate()

    assert config.task.mode is TaskMode.B
    assert config.adaptive_inverse.enabled is True
    assert config.adaptive_inverse.require_validated_extrinsics is False
    assert config.adaptive_inverse.candidates.use_visual_grasp_z is True
    assert config.adaptive_inverse.candidates.jaw_tip_clearance_m == pytest.approx(
        0.002
    )
    assert config.vision.backend == "mock"
    assert config.motion.backend == "mock"
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"
    assert config.task_b.rounds == 5
    assert len(config.place_poses) == 5
    assert Path(config.vision.scenario_file).is_file()
    assert config.motion.speed_mps == pytest.approx(0.12)
    assert config.motion.joint_speed_rad_s == pytest.approx(0.6)
    assert config.motion.joint_speed_safety_limit_rad_s == pytest.approx(0.7)


def test_build_vision_and_factory_only_pass_adaptive_config_when_enabled(
    monkeypatch,
):
    constructor_calls: list[dict[str, object]] = []

    class RecordingVision:
        def __init__(self, config, *, log=None, **kwargs):
            constructor_calls.append(dict(kwargs))
            self.config = config

        def close(self):
            return None

    monkeypatch.setattr(vision_module, "RealSenseYoloVision", RecordingVision)
    vision_config = replace(
        load_config(MOCK_CONFIG).vision,
        backend="realsense_yolo",
        continuous=False,
    )
    adaptive = replace(
        vision_config,
        backend="realsense_yolo",
    )
    adaptive_inverse = AdaptiveInverseConfig(enabled=True)

    vision_module.build_vision(vision_config, adaptive_inverse=None)
    vision_module.build_vision(
        adaptive,
        adaptive_inverse=adaptive_inverse,
    )

    assert constructor_calls[0] == {}
    assert constructor_calls[1]["adaptive_inverse"] is adaptive_inverse

    factory_calls: list[AdaptiveInverseConfig | None] = []

    def fake_build_vision(_config, *, log=None, adaptive_inverse=None):
        factory_calls.append(adaptive_inverse)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(factory_module, "build_vision", fake_build_vision)
    monkeypatch.setattr(
        factory_module,
        "build_motion",
        lambda _config, node=None, log=None: SimpleNamespace(dry_run=True),
    )
    monkeypatch.setattr(
        factory_module,
        "build_gripper",
        lambda _config, node=None, log=None: SimpleNamespace(
            state=GripperState.OPEN,
            events=[],
        ),
    )
    monkeypatch.setattr(
        factory_module,
        "build_hold_verifier",
        lambda _config, _gripper, log=None: SimpleNamespace(),
    )
    base = adaptive_config()
    disabled = replace(
        base,
        adaptive_inverse=replace(base.adaptive_inverse, enabled=False),
    )
    enabled = base

    factory_module.build_pick_pipeline(disabled)
    factory_module.build_pick_pipeline(enabled)

    assert factory_calls == [None, enabled.adaptive_inverse]
