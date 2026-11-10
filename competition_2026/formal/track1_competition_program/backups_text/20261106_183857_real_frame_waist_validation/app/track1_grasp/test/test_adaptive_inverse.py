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
    _component_consensus_support_plane,
    _fit_support_plane_robust,
    _otsu_depth_split,
    _support_plane_rejection_reason,
    estimate_object_pose,
    estimate_scene_support_plane,
)
from track1_grasp.config import (
    AdaptiveCandidatesConfig,
    AdaptiveInverseConfig,
    AdaptiveSelectionConfig,
    TaskMode,
    VisionConfig,
    apply_mode,
    load_config,
    load_joint_pose_file,
)
from track1_grasp.detection_stabilizer import DetectionStabilizer
from track1_grasp.domain import (
    CameraIntrinsics,
    CaptureRobotState,
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
from track1_grasp.vision import RealSenseYoloVision


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
POINT2_SAFE_CONFIG = (
    ROOT / "config" / "track1_task_b_point2_safe_20261027.yaml"
)
FIVE_POINT_ADAPTIVE_SAFE_CONFIG = (
    ROOT
    / "config"
    / "track1_task_b_five_point_adaptive_safe_20261102.yaml"
)
PICK1_LOW_SPEED_LIVE_CONFIG = (
    ROOT
    / "config"
    / "track1_task_b_pick1_single_low_speed_live_20261106.yaml"
)
PICK1_FAST_LIVE_CONFIG = (
    ROOT
    / "config"
    / "track1_task_b_pick1_single_fast_live_20261106.yaml"
)
EXTRINSICS = ROOT / "config" / "camera_to_flange.d405_assumed.yaml"
GRASP_RPY = (pi, 0.0, 0.0)

ADAPTIVE_POINT_CENTERS = (
    (-0.029246342, 0.482700096, 0.005),
    (-0.142883909, 0.532708714, 0.006),
    (-0.092008678, 0.431708580, 0.007),
    (0.046673617, 0.402881069, 0.008),
    (0.039196350, 0.498712736, 0.009),
)
ADAPTIVE_PCA_YAW_DEG = (5.0, 38.0, -17.0, 61.0, -42.0)


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
    minor_axis_camera: tuple[float, float, float] | None = None,
    held: bool = False,
) -> Detection:
    major = np.asarray(major_axis_camera, dtype=float)
    major_xy_norm = float(np.linalg.norm(major[:2]))
    if minor_axis_camera is None:
        if not np.isfinite(major_xy_norm) or major_xy_norm <= 1e-12:
            derived_minor = (0.0, 1.0, 0.0)
        else:
            derived_minor = (
                -major[1] / major_xy_norm,
                major[0] / major_xy_norm,
                0.0,
            )
    else:
        derived_minor = minor_axis_camera
    return Detection(
        label="8-Plastic_Banana",
        confidence=0.95,
        bbox=(220.0, 180.0, 360.0, 300.0),
        depth_m=0.08,
        camera_xyz=(0.02, 0.01, 0.08),
        object_center_camera_xyz=(0.02, 0.01, 0.08),
        major_axis_camera=major_axis_camera,
        minor_axis_camera=derived_minor,
        major_axis_length_m=0.12,
        minor_axis_length_m=0.04,
        pose_confidence=0.95,
        attributes={
            "held": held,
            "adaptive_support_plane_valid": True,
            "adaptive_axis_source": "waist_midline",
            "adaptive_waist_midpoint_uv_px": (290.0, 240.0),
            "adaptive_waist_path_length_px": 120.0,
            "adaptive_waist_half_width_px": 20.0,
        },
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
    orientation_source: str = "object_relative",
):
    return AdaptiveGraspCandidate(
        plan=vertical_plan(detection, x=x),
        yaw_offset_deg=yaw_deg,
        pitch_offset_deg=pitch_deg,
        orientation_change_rad=radians(abs(yaw_deg)),
        object_center_base_xyz=(x, 0.49, 0.166),
        major_axis_base=(1.0, 0.0, 0.0),
        orientation_source=orientation_source,
    )


def synthetic_adaptive_pipeline(config=None):
    pipeline = build_pick_pipeline(config or adaptive_config())
    detection = adaptive_detection()
    state = pipeline.motion.get_state()
    frame = Frame(
        "synthetic-adaptive",
        (detection,),
        capture_robot_state=CaptureRobotState(before=state, after=state),
    )
    pipeline._acquire_task_b_target = lambda *_args, **_kwargs: (
        detection,
        frame,
        state,
        "synthetic adaptive detection",
    )
    return pipeline, detection


def rescue_adaptive_config():
    base = adaptive_config()
    candidates = replace(
        base.adaptive_inverse.candidates,
        max_count=64,
        lock_taught_orientation=False,
        taught_orientation_rescue_enabled=True,
        yaw_offsets_deg=(0.0, 45.0, -45.0),
        pitch_offsets_deg=(0.0, -5.0),
        taught_yaw_offsets_deg=(0.0, -15.0, 15.0),
        taught_pitch_offsets_deg=(0.0, -5.0),
        use_visual_grasp_z=False,
    )
    return replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            candidates=candidates,
            selection=replace(
                base.adaptive_inverse.selection,
                stop_after_first_valid=True,
            ),
        ),
    )


def fixed_geometry_planner(config, *, yaw_rad: float):
    class FixedGeometryPlanner(PosePlanner):
        def _adaptive_object_geometry(self, _detection, _robot_state):
            return (
                np.array((0.10, 0.20, 0.166), dtype=float),
                np.array((cos(yaw_rad), sin(yaw_rad), 0.0), dtype=float),
                np.eye(4),
            )

    return FixedGeometryPlanner(config)


def test_adaptive_planner_uses_capture_feedback_not_virtual_planning_state():
    pipeline = build_pick_pipeline(adaptive_config())
    detection = adaptive_detection()
    capture_state = RobotState(
        ready=True,
        pose=Pose((0.11, 0.22, 0.33), GRASP_RPY),
        joints=(0.11, -0.22, 0.33, -0.44, 0.55, -0.66),
        message="camera-time feedback",
    )
    virtual_state = pipeline.motion.get_state()
    assert virtual_state.joints != capture_state.joints
    frame = Frame(
        "feedback-pose-regression",
        (detection,),
        capture_robot_state=CaptureRobotState(
            before=capture_state,
            after=capture_state,
        ),
    )
    pipeline._acquire_task_b_target = lambda *_args, **_kwargs: (
        detection,
        frame,
        virtual_state,
        "synthetic dry-run observation",
    )
    planner_states: list[RobotState] = []

    def record_planner_state(_target, state, **_kwargs):
        planner_states.append(state)
        return ()

    pipeline.planner.plan_adaptive_candidates = record_planner_state

    result = pipeline.run_once()

    assert planner_states == [capture_state]
    assert planner_states[0] is capture_state
    assert planner_states[0] is not virtual_state
    assert result.status is PickStatus.REJECTED
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []


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


def make_otsu_near_mode_with_coloured_clutter() -> tuple[
    np.ndarray,
    SimpleNamespace,
    tuple[int, int, int, int],
    np.ndarray,
    np.ndarray,
]:
    height, width = 240, 320
    v, u = np.mgrid[0:height, 0:width]
    intrinsics = SimpleNamespace(fx=500.0, fy=500.0, cx=160.0, cy=120.0)
    depth = np.full((height, width), 0.400, dtype=float)

    # Both regions share the Otsu near-depth mode.  Only ``object_mask`` is
    # target-coloured; the second region represents same-depth coloured clutter
    # inside the detector box.
    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[112:128, 75:145] = True
    clutter_mask = np.zeros((height, width), dtype=bool)
    clutter_mask[102:138, 152:172] = True
    near_mode = object_mask | clutter_mask
    depth[near_mode] = 0.350

    bbox = (60, 90, 180, 150)
    return depth, intrinsics, bbox, object_mask, clutter_mask


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


def test_otsu_near_mode_colour_mask_excludes_same_depth_clutter_from_pca() -> None:
    depth, intrinsics, bbox, object_mask, clutter_mask = (
        make_otsu_near_mode_with_coloured_clutter()
    )
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        trim_percentile=0.0,
        min_axis_ratio=1.1,
        support_plane_required=True,
        support_plane_min_points=100,
        support_plane_min_inlier_ratio=0.35,
    )
    x0, y0, x1, y1 = bbox
    patch = depth[y0:y1, x0:x1]
    split = _otsu_depth_split(
        patch[np.isfinite(patch)],
        max(
            int(config.min_valid_points),
            int(np.ceil(0.05 * np.count_nonzero(np.isfinite(patch)))),
        ),
    )
    assert split is not None
    threshold, _ = split
    assert np.all(patch[object_mask[y0:y1, x0:x1]] <= threshold)
    assert np.all(patch[clutter_mask[y0:y1, x0:x1]] <= threshold)

    estimate = estimate_object_pose(
        depth,
        bbox,
        intrinsics,
        config,
        object_mask=object_mask,
    )
    unmasked = estimate_object_pose(depth, bbox, intrinsics, config)

    assert estimate.valid_points == int(np.count_nonzero(object_mask))
    assert unmasked.valid_points == int(
        np.count_nonzero(object_mask | clutter_mask)
    )
    assert estimate.valid_points < unmasked.valid_points
    np.testing.assert_allclose(
        estimate.center_camera_xyz,
        (-0.04, 0.0, 0.4),
        atol=1e-3,
    )
    assert estimate.major_axis_length_m == pytest.approx(0.0483, abs=2e-3)
    assert estimate.minor_axis_length_m == pytest.approx(0.0105, abs=2e-3)
    assert estimate.major_axis_length_m < unmasked.major_axis_length_m


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


def test_scene_support_plane_ignores_objects_wall_and_gripper_outliers() -> None:
    height, width = 180, 240
    v, u = np.mgrid[0:height, 0:width]
    intrinsics = SimpleNamespace(fx=300.0, fy=300.0, cx=120.0, cy=90.0)
    ray_u = (u - intrinsics.cx) / intrinsics.fx
    ray_v = (v - intrinsics.cy) / intrinsics.fy
    support_z = 0.40 / (1.0 - 0.03 * ray_u + 0.02 * ray_v)
    depth = support_z.copy()
    boxes = (
        (45.0, 50.0, 95.0, 100.0),
        (145.0, 50.0, 195.0, 100.0),
        (85.0, 115.0, 155.0, 155.0),
    )
    for x0, y0, x1, y1 in boxes:
        ix0, iy0, ix1, iy1 = (int(value) for value in (x0, y0, x1, y1))
        depth[iy0:iy1, ix0:ix1] = (
            support_z[iy0:iy1, ix0:ix1] - 0.050
        )
    # A transparent wall and isolated gripper/depth spikes must not survive
    # the local-normal and local-depth-span guards.
    depth[:, 115:123] = 0.24 + 0.10 * ray_u[:, 115:123]
    rng = np.random.default_rng(9)
    depth[rng.integers(0, height, 80), rng.integers(0, width, 80)] = 0.12
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        support_plane_min_points=100,
        support_plane_ransac_iterations=128,
    )

    estimate = estimate_scene_support_plane(depth, boxes, intrinsics, config)

    assert estimate is not None
    assert estimate.normal_camera[2] > 0.99
    assert estimate.rms_m < 0.001
    assert estimate.valid_points >= 100
    assert estimate.inlier_ratio > 0.95


def make_component_consensus_scene() -> tuple[
    np.ndarray,
    np.ndarray,
    SimpleNamespace,
    tuple[tuple[float, float, float, float], ...],
    tuple[tuple[int, int, int, int], ...],
]:
    height, width = 200, 300
    depth = np.full((height, width), 0.40, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    components = (
        (20, 18, 65, 45),
        (105, 15, 150, 43),
        (235, 22, 282, 50),
        (30, 118, 78, 148),
        (122, 132, 172, 165),
        (222, 115, 270, 145),
    )
    for x0, y0, x1, y1 in components:
        mask[y0:y1, x0:x1] = True
    intrinsics = SimpleNamespace(fx=300.0, fy=300.0, cx=150.0, cy=100.0)
    bboxes = ((116.0, 72.0, 174.0, 112.0),)
    depth[72:112, 116:174] = 0.25
    return depth, mask, intrinsics, bboxes, components


def make_transparent_roi_scene(
    *,
    far_depth_m: float = 0.40,
    far_patch_count: int = 6,
    include_near_object: bool = True,
) -> tuple[np.ndarray, SimpleNamespace, tuple[tuple[float, float, float, float], ...]]:
    """Build a box-interior scene with separated far surface patches."""

    height, width = 200, 300
    depth = np.full((height, width), np.nan, dtype=float)
    if include_near_object:
        depth[70:130, 125:175] = 0.25
    patches = (
        (98, 55, 118, 75),
        (182, 55, 202, 75),
        (98, 125, 118, 145),
        (182, 125, 202, 145),
        (130, 48, 150, 62),
        (160, 138, 180, 152),
    )
    for x0, y0, x1, y1 in patches[:far_patch_count]:
        depth[y0:y1, x0:x1] = far_depth_m
    intrinsics = SimpleNamespace(fx=300.0, fy=300.0, cx=150.0, cy=100.0)
    bboxes = ((90.0, 45.0, 210.0, 155.0),)
    return depth, intrinsics, bboxes


def test_transparent_roi_far_fallback_recovers_box_floor() -> None:
    depth, intrinsics, bboxes = make_transparent_roi_scene()
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        support_plane_min_points=100,
        support_plane_min_support_components=5,
    )

    strict_mask = adaptive_module._scene_support_candidate_mask(
        depth,
        bboxes,
        intrinsics,
        config,
    )
    fallback_mask = (
        adaptive_module._scene_support_roi_far_candidate_mask(
            depth,
            bboxes,
            intrinsics,
            config,
        )
    )
    estimate = estimate_scene_support_plane(
        depth,
        bboxes,
        intrinsics,
        config,
    )

    assert not np.any(strict_mask)
    assert np.count_nonzero(fallback_mask) >= config.support_plane_min_points
    assert estimate is not None
    assert estimate.center_camera_xyz[2] == pytest.approx(0.40, abs=1e-5)
    assert estimate.normal_camera[2] > 0.999
    assert estimate.rms_m < 1e-6


def test_transparent_roi_fallback_keeps_safety_gates() -> None:
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        support_plane_min_points=100,
        support_plane_min_support_components=5,
    )

    # Six separated patches of an object top are not a support plane: the
    # predicted plane has no measured clearance from the detected target.
    object_top, intrinsics, bboxes = make_transparent_roi_scene(
        far_depth_m=0.35,
        include_near_object=False,
    )
    assert (
        estimate_scene_support_plane(
            object_top,
            bboxes,
            intrinsics,
            config,
        )
        is None
    )

    # Four valid but separated support patches remain insufficient.
    sparse_support, sparse_intrinsics, sparse_bboxes = make_transparent_roi_scene(
        far_patch_count=4,
    )
    assert (
        estimate_scene_support_plane(
            sparse_support,
            sparse_bboxes,
            sparse_intrinsics,
            config,
        )
        is None
    )

    # A strongly slanted transparent sidewall and unstructured depth noise are
    # rejected by the local normal/span filters before component consensus.
    sidewall = object_top.copy()
    rows, cols = np.indices(sidewall.shape, dtype=float)
    ray_u = (cols - intrinsics.cx) / intrinsics.fx
    for x0, y0, x1, y1 in (
        (98, 55, 118, 75),
        (182, 55, 202, 75),
        (98, 125, 118, 145),
        (182, 125, 202, 145),
        (130, 48, 150, 62),
        (160, 138, 180, 152),
    ):
        sidewall[y0:y1, x0:x1] = 0.45 / (
            1.0 - 1.8 * ray_u[y0:y1, x0:x1]
        )
    sidewall_mask = adaptive_module._scene_support_roi_far_candidate_mask(
        sidewall,
        bboxes,
        intrinsics,
        config,
    )
    assert not np.any(sidewall_mask)

    noise = np.full_like(object_top, np.nan)
    noise[45:155, 90:210] = np.random.default_rng(22).uniform(
        0.20,
        0.70,
        size=(110, 120),
    )
    assert (
        estimate_scene_support_plane(
            noise,
            bboxes,
            intrinsics,
            config,
        )
        is None
    )


def test_component_consensus_aggregates_dispersed_table_patches() -> None:
    depth, mask, intrinsics, bboxes, components = make_component_consensus_scene()
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        support_plane_min_points=100,
        support_plane_min_support_components=5,
    )

    estimate = _component_consensus_support_plane(
        depth,
        mask,
        bboxes,
        intrinsics,
        config,
    )

    assert estimate is not None
    assert estimate.center_camera_xyz[2] == pytest.approx(0.40, abs=1e-5)
    assert estimate.normal_camera[2] > 0.999
    assert estimate.valid_points == sum(
        (x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in components
    )


def test_component_consensus_rejects_large_object_top_and_near_fragments() -> None:
    depth, mask, intrinsics, bboxes, _components = make_component_consensus_scene()
    # A large, perfectly planar object top is only one connected component and
    # therefore cannot impersonate a tabletop supported by separated patches.
    mask[70:115, 80:220] = True
    depth[70:115, 80:220] = 0.30
    # A collection of small near-plane glints has enough points to form
    # components but not enough independent support to win consensus.
    for x0 in (40, 95, 185, 240):
        for y0 in (55, 90, 150):
            mask[y0 : y0 + 5, x0 : x0 + 5] = True
            depth[y0 : y0 + 5, x0 : x0 + 5] = 0.26
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        support_plane_min_points=100,
        support_plane_min_support_components=5,
    )

    estimate = _component_consensus_support_plane(
        depth,
        mask,
        bboxes,
        intrinsics,
        config,
    )

    assert estimate is not None
    assert estimate.center_camera_xyz[2] == pytest.approx(0.40, abs=1e-5)
    assert estimate.inlier_ratio > 0.99


def test_component_consensus_fails_closed_without_independent_patches_or_clearance() -> None:
    depth, _mask, intrinsics, bboxes, _components = make_component_consensus_scene()
    one_patch = np.zeros_like(depth, dtype=bool)
    one_patch[40:160, 40:260] = True
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        support_plane_min_points=100,
        support_plane_min_support_components=5,
    )
    assert (
        _component_consensus_support_plane(
            depth,
            one_patch,
            bboxes,
            intrinsics,
            config,
        )
        is None
    )

    _depth, mask, _intrinsics, _bboxes, _components = make_component_consensus_scene()
    target_behind_plane = depth.copy()
    target_behind_plane[72:112, 116:174] = 0.45
    assert (
        _component_consensus_support_plane(
            target_behind_plane,
            mask,
            bboxes,
            intrinsics,
            config,
        )
        is None
    )


def test_scene_support_plane_fails_closed_without_visible_tabletop() -> None:
    height, width = 180, 240
    intrinsics = SimpleNamespace(fx=300.0, fy=300.0, cx=120.0, cy=90.0)
    v, u = np.mgrid[0:height, 0:width]
    ray_u = (u - intrinsics.cx) / intrinsics.fx
    boxes = ((45.0, 50.0, 95.0, 100.0),)
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        support_plane_min_points=100,
    )
    rng = np.random.default_rng(11)

    noisy = 0.25 + rng.random((height, width)) * 0.20
    assert estimate_scene_support_plane(noisy, boxes, intrinsics, config) is None

    vertical_wall = np.full((height, width), np.nan)
    vertical_wall[:, 115:123] = 0.24 + 0.10 * ray_u[:, 115:123]
    assert (
        estimate_scene_support_plane(
            vertical_wall,
            boxes,
            intrinsics,
            config,
        )
        is None
    )

    # The object top extends only around the detector box.  It is intentionally
    # guarded out rather than being accepted as the support plane.
    object_only = np.full((height, width), np.nan)
    object_only[40:110, 35:105] = 0.35
    assert (
        estimate_scene_support_plane(
            object_only,
            boxes,
            intrinsics,
            config,
        )
        is None
    )


def test_validated_scene_support_plane_skips_target_local_refit(monkeypatch) -> None:
    depth, intrinsics, bbox = make_object_depth()
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        support_plane_required=True,
        support_plane_min_points=100,
    )
    scene_plane = SupportPlaneEstimate(
        center_camera_xyz=(0.0, 0.0, 0.4),
        normal_camera=(0.0, 0.0, 1.0),
        rms_m=0.0005,
        valid_points=500,
        candidate_points=600,
        inlier_ratio=500.0 / 600.0,
        median_residual_m=0.0004,
    )

    def unexpected_refit(*_args, **_kwargs):
        raise AssertionError("target-local support-plane refit must be skipped")

    monkeypatch.setattr(adaptive_module, "_masked_support_plane", unexpected_refit)

    estimate = estimate_object_pose(
        depth,
        bbox,
        intrinsics,
        config,
        scene_support_plane=scene_plane,
    )

    assert estimate.support_plane_center_camera_xyz == scene_plane.center_camera_xyz
    assert estimate.support_plane_normal_camera == scene_plane.normal_camera
    assert estimate.support_plane_rms_m == scene_plane.rms_m
    assert estimate.support_plane_valid_points == scene_plane.valid_points
    assert estimate.support_plane_candidate_points == scene_plane.candidate_points
    assert estimate.support_plane_inlier_ratio == scene_plane.inlier_ratio
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


def test_deferred_pca_retains_same_frame_geometry_evidence():
    depth, intrinsics, bbox, _object_mask, _wall, _colour_guard = (
        make_full_support_scene()
    )
    config = replace(
        load_config(MOCK_CONFIG).adaptive_inverse.pca,
        min_valid_points=100,
        support_plane_min_points=100,
        defer_axis_ratio_rejection=True,
    )

    estimate = estimate_object_pose(depth, bbox, intrinsics, config)

    evidence = estimate.geometry_evidence
    assert evidence is not None
    assert evidence.support_plane_valid is True
    assert evidence.support_plane_quality > 0.90
    assert evidence.footprint_axis_ratio == pytest.approx(3.36, rel=0.10)
    assert evidence.short_edge_confidence > 0.90
    assert estimate.axis_ratio == pytest.approx(3.43, rel=0.10)
    # The legacy return contract still projects the command point down to the
    # support plane, while geometry evidence retains the measured object centre.
    assert evidence.object_center_camera_xyz[2] < estimate.center_camera_xyz[2]


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


def test_rescue_candidates_follow_pca_then_taught_yaw_outer_pitch_inner_order():
    config = rescue_adaptive_config()
    planner = fixed_geometry_planner(config, yaw_rad=radians(20.0))
    taught_rpy = (pi, 0.0, 0.0)

    candidates = planner.plan_adaptive_candidates(
        adaptive_detection(),
        robot_state(taught_rpy),
        place_name="pick_1",
        grasp_z_m=0.166,
        grasp_rpy=taught_rpy,
        task_b_anchor_pose=Pose((0.10, 0.20, 0.40), taught_rpy),
    )

    assert [item.orientation_source for item in candidates] == [
        "object_relative",
        "object_relative",
        "object_relative",
        "object_relative",
        "object_relative",
        "object_relative",
        "taught_rescue",
        "taught_rescue",
        "taught_rescue",
        "taught_rescue",
        "taught_rescue",
        "taught_rescue",
    ]
    assert [item.yaw_offset_deg for item in candidates] == pytest.approx(
        [
            20.0,
            20.0,
            65.0,
            65.0,
            -25.0,
            -25.0,
            0.0,
            0.0,
            -15.0,
            -15.0,
            15.0,
            15.0,
        ]
    )
    assert [item.pitch_offset_deg for item in candidates] == pytest.approx(
        [0.0, -5.0] * 6
    )


def test_taught_rescue_rotation_matches_the_documented_composition():
    config = rescue_adaptive_config()
    planner = fixed_geometry_planner(config, yaw_rad=radians(20.0))
    taught_rpy = (3.05, 0.04, -0.35)
    candidates = planner.plan_adaptive_candidates(
        adaptive_detection(),
        robot_state(taught_rpy),
        place_name="pick_1",
        grasp_z_m=0.166,
        grasp_rpy=taught_rpy,
        task_b_anchor_pose=Pose((0.10, 0.20, 0.40), taught_rpy),
    )
    rescue = next(
        item
        for item in candidates
        if item.orientation_source == "taught_rescue"
        and item.yaw_offset_deg == pytest.approx(-15.0)
        and item.pitch_offset_deg == pytest.approx(-5.0)
    )
    yaw_delta = radians(-15.0)
    rotation_z = np.array(
        (
            (cos(yaw_delta), -sin(yaw_delta), 0.0),
            (sin(yaw_delta), cos(yaw_delta), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    expected_rotation = rotation_z @ euler_to_matrix(
        (
            taught_rpy[0],
            taught_rpy[1] + radians(-5.0),
            taught_rpy[2],
        )
    )

    assert euler_to_matrix(rescue.plan.grasp_pose.rpy) == pytest.approx(
        expected_rotation,
        abs=1e-10,
    )


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


def test_learned_jaw_tip_object_offset_is_invariant_to_pca_sign_flip():
    base = adaptive_config()
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            candidates=replace(
                base.adaptive_inverse.candidates,
                max_count=1,
            ),
        ),
    )
    planner = PosePlanner(config)
    learned_offset = (0.004288649, -0.023589442)
    detection = replace(
        adaptive_detection(major_axis_camera=(1.0, 0.0, 0.0)),
        object_center_camera_xyz=(0.100, 0.055, 0.29776),
        camera_xyz=(0.100, 0.055, 0.29776),
    )
    flipped_detection = replace(
        detection,
        major_axis_camera=(-1.0, 0.0, 0.0),
    )

    original = planner.plan_adaptive_candidates(
        detection,
        robot_state(),
        grasp_rpy=GRASP_RPY,
        jaw_tip_object_offset_m=learned_offset,
    )
    flipped = planner.plan_adaptive_candidates(
        flipped_detection,
        robot_state(),
        grasp_rpy=GRASP_RPY,
        jaw_tip_object_offset_m=learned_offset,
    )

    assert len(original) == 1
    assert len(flipped) == 1
    assert flipped[0].major_axis_horizontal_base == pytest.approx(
        original[0].major_axis_horizontal_base
    )
    assert flipped[0].jaw_tip_object_offset_m == pytest.approx(learned_offset)
    assert flipped[0].commanded_jaw_tip_base_xyz == pytest.approx(
        original[0].commanded_jaw_tip_base_xyz,
        abs=1e-12,
    )


def test_learned_jaw_tip_offset_cannot_bypass_grasp_depth_floor():
    base = adaptive_config()
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            candidates=replace(
                base.adaptive_inverse.candidates,
                max_count=1,
                use_visual_grasp_z=False,
                minimum_grasp_flange_z_m=0.166534092,
            ),
        ),
    )
    planner = PosePlanner(config)

    with pytest.raises(
        ValueError,
        match=r"planned grasp flange Z 0\.100000 m is below the hard minimum",
    ):
        planner.plan_adaptive_candidates(
            adaptive_detection(),
            robot_state(),
            grasp_z_m=0.100,
            grasp_rpy=GRASP_RPY,
            jaw_tip_object_offset_m=(0.004288649, -0.023589442),
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


def test_safe_point2_config_locks_taught_rpy_and_depth():
    config = load_config(POINT2_SAFE_CONFIG)
    taught_z = config.task_b.point_grasp_z_m[0]
    taught_rpy = config.task_b.point_grasp_rpy[0]
    taught_pitch_offsets = config.adaptive_inverse.candidates.taught_pitch_offsets_deg
    learned_offset = config.task_b.point_grasp_jaw_tip_object_offset_m[0]
    assert config.adaptive_inverse.candidates.lock_taught_orientation is True
    assert config.adaptive_inverse.candidates.yaw_offsets_deg == (0.0,)
    assert taught_pitch_offsets == pytest.approx((0.0, -5.0, -10.0))
    assert learned_offset is not None

    class DangerousVisualGeometryPlanner(PosePlanner):
        def _adaptive_object_geometry(self, _detection, _robot_state):
            return (
                np.array((0.10, 0.20, -0.034179), dtype=float),
                np.array((np.cos(0.7), np.sin(0.7), 0.0), dtype=float),
                np.eye(4),
            )

    planner = DangerousVisualGeometryPlanner(config)
    candidates = planner.plan_adaptive_candidates(
        adaptive_detection(),
        robot_state(),
        place_name="pick_2",
        grasp_z_m=taught_z,
        grasp_rpy=taught_rpy,
        task_b_anchor_pose=Pose((0.10, 0.20, 0.50), taught_rpy),
        vertical_grasp_approach_height_m=0.075,
        jaw_tip_object_offset_m=learned_offset,
    )

    assert len(candidates) == len(taught_pitch_offsets)
    assert [item.yaw_offset_deg for item in candidates] == pytest.approx(
        [0.0] * len(taught_pitch_offsets)
    )
    assert [item.pitch_offset_deg for item in candidates] == pytest.approx(
        taught_pitch_offsets
    )
    major_xy = np.array((np.cos(0.7), np.sin(0.7)))
    minor_xy = np.array((-major_xy[1], major_xy[0]))
    expected_jaw_tip_xy = (
        np.array((0.10, 0.20))
        + learned_offset[0] * major_xy
        + learned_offset[1] * minor_xy
    )
    for candidate, pitch_offset_deg in zip(
        candidates,
        taught_pitch_offsets,
    ):
        expected_rpy = (
            taught_rpy[0],
            taught_rpy[1] + radians(pitch_offset_deg),
            taught_rpy[2],
        )
        assert candidate.plan.grasp_pose.rpy == pytest.approx(expected_rpy)
        assert candidate.plan.pregrasp_pose.rpy == pytest.approx(expected_rpy)
        assert candidate.commanded_jaw_tip_base_xyz is not None
        assert candidate.commanded_jaw_tip_base_xyz[:2] == pytest.approx(
            expected_jaw_tip_xy,
            abs=1e-9,
        )
    assert all(
        candidate.plan.grasp_pose.xyz[2] == pytest.approx(taught_z)
        for candidate in candidates
    )
    assert all(
        candidate.plan.pregrasp_pose.xyz[2]
        == pytest.approx(taught_z + 0.075)
        for candidate in candidates
    )


def test_safe_point2_config_fails_closed_without_learned_jaw_tip_offset():
    config = load_config(POINT2_SAFE_CONFIG)
    taught_z = config.task_b.point_grasp_z_m[0]
    taught_rpy = config.task_b.point_grasp_rpy[0]

    class SyntheticGeometryPlanner(PosePlanner):
        def _adaptive_object_geometry(self, _detection, _robot_state):
            return (
                np.array((0.10, 0.20, -0.034179), dtype=float),
                np.array((1.0, 0.0, 0.0), dtype=float),
                np.eye(4),
            )

    planner = SyntheticGeometryPlanner(config)
    with pytest.raises(
        RuntimeError,
        match="requires a learned jaw-tip object offset",
    ):
        planner.plan_adaptive_candidates(
            adaptive_detection(),
            robot_state(),
            place_name="pick_2",
            grasp_z_m=taught_z,
            grasp_rpy=taught_rpy,
            task_b_anchor_pose=Pose((0.10, 0.20, 0.50), taught_rpy),
            vertical_grasp_approach_height_m=0.075,
        )


def test_accident_grasp_depth_is_rejected_before_any_precheck_or_command():
    base = adaptive_config()
    candidates = replace(
        base.adaptive_inverse.candidates,
        max_count=1,
        yaw_offsets_deg=(0.0,),
        pitch_offsets_deg=(20.0,),
        use_visual_grasp_z=False,
        minimum_grasp_flange_z_m=0.166534092,
    )
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            candidates=candidates,
        ),
    )
    rejection_logs: list[str] = []

    class AccidentDepthPlanner(PosePlanner):
        def __init__(self):
            super().__init__(config, log=rejection_logs.append)
            self.plan_calls = 0

        def _adaptive_object_geometry(self, _detection, _robot_state):
            return (
                np.array((0.10, 0.20, -0.034179), dtype=float),
                np.array((1.0, 0.0, 0.0), dtype=float),
                np.eye(4),
            )

        def plan(self, detection, _robot_state, **_kwargs):
            self.plan_calls += 1
            accident_z = 0.117953
            high = Pose((0.10, 0.20, accident_z + 0.075), GRASP_RPY)
            return MotionPlan(
                target=detection,
                grasp_pose=Pose((0.10, 0.20, accident_z), GRASP_RPY),
                pregrasp_pose=high,
                lift_pose=high,
                transfer_pose=high,
                release_pose=Pose((0.10, 0.20, accident_z), GRASP_RPY),
                retreat_pose=high,
                place_name="pick_2",
            )

    pipeline, _detection = synthetic_adaptive_pipeline(config)
    planner = AccidentDepthPlanner()
    pipeline.planner = planner
    endpoint_calls: list[tuple[Pose, str]] = []
    route_calls: list[tuple[Pose, ...]] = []
    motion_calls: list[str] = []
    gripper_calls: list[str] = []

    pipeline.motion.precheck_pose_condition = (
        lambda pose, *, label: endpoint_calls.append((pose, label))
    )
    pipeline.motion.precheck_linear_targets = (
        lambda targets, *, step_m: route_calls.append(tuple(targets))
    )
    pipeline.motion.execute_prechecked_segment = (
        lambda *_args, **_kwargs: motion_calls.append("segment")
    )
    pipeline.gripper.open = lambda: gripper_calls.append("open")
    pipeline.gripper.close = lambda: gripper_calls.append("close")

    result = pipeline.run_once()

    assert result.status is PickStatus.REJECTED
    assert planner.plan_calls == 1
    assert "planned grasp flange Z 0.117953 m" in result.message
    assert "minimum 0.166534 m" in result.message
    assert any(
        "candidate rejected by grasp-depth floor" in message
        for message in rejection_logs
    )
    assert endpoint_calls == []
    assert route_calls == []
    assert motion_calls == []
    assert gripper_calls == []
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []


def test_strict_pca_short_edge_plans_one_vertical_axis_aligned_candidate():
    config = load_config(FIVE_POINT_ADAPTIVE_SAFE_CONFIG)
    planner = PosePlanner(config)
    pca_yaw_rad = radians(20.0)
    detection = adaptive_detection(
        major_axis_camera=(
            cos(pca_yaw_rad),
            sin(pca_yaw_rad),
            0.0,
        )
    )
    center = np.asarray(ADAPTIVE_POINT_CENTERS[0], dtype=float)
    major = np.asarray(
        (cos(pca_yaw_rad), sin(pca_yaw_rad), 0.0),
        dtype=float,
    )
    planner._adaptive_object_geometry = lambda _detection, _state: (
        center,
        major,
        np.eye(4),
    )
    taught_rpy = config.task_b.point_grasp_rpy[0]
    robot = robot_state(taught_rpy)

    candidates = planner.plan_adaptive_candidates(
        detection,
        robot,
        place_name="pick_1",
        grasp_rpy=taught_rpy,
        task_b_anchor_pose=Pose(
            (float(center[0]), float(center[1]), 0.40),
            taught_rpy,
        ),
        vertical_grasp_approach_height_m=0.075,
        jaw_tip_object_offset_m=(
            config.task_b.point_grasp_jaw_tip_object_offset_m[0]
        ),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.orientation_source == "task_b_banana_waist_short_axis"
    assert candidate.plan.grasp_geometry_mode == "task_b_banana_waist_short_axis"
    assert candidate.object_center_base_xyz == pytest.approx(center)
    assert candidate.commanded_jaw_tip_base_xyz[:2] == pytest.approx(
        center[:2], abs=1e-12
    )
    assert candidate.jaw_tip_object_offset_m is None
    assert candidate.plan.grasp_close_width_m == pytest.approx(0.04)
    assert candidate.plan.grasp_close_width_m == pytest.approx(
        detection.minor_axis_length_m
    )
    assert candidate.plan.release_pose.xyz == pytest.approx(
        candidate.plan.grasp_pose.xyz
    )
    assert candidate.plan.release_pose.rpy == pytest.approx(
        candidate.plan.grasp_pose.rpy
    )

    rotation = euler_to_matrix(candidate.plan.grasp_pose.rpy)
    assert rotation[:, 2] == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
    closure_axis = rotation @ np.array((1.0, 0.0, 0.0))
    assert abs(float(closure_axis[2])) <= 1e-9

    base_camera = (
        planner._base_flange_transform(robot) @ planner._camera_transform()
    )
    measured_minor = base_camera[:3, :3] @ np.asarray(
        detection.minor_axis_camera,
        dtype=float,
    )
    measured_minor_xy = measured_minor[:2]
    measured_minor_xy /= np.linalg.norm(measured_minor_xy)
    closure_xy = closure_axis[:2]
    closure_xy /= np.linalg.norm(closure_xy)
    assert abs(float(closure_xy @ measured_minor_xy)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("detection", "message"),
    (
        pytest.param(
            replace(
                adaptive_detection(),
                attributes={
                    **adaptive_detection().attributes,
                    "adaptive_support_plane_valid": False,
                },
            ),
            "validated support plane",
            id="missing-support-plane",
        ),
        pytest.param(
            replace(adaptive_detection(), minor_axis_camera=None),
            "minor_axis_camera",
            id="missing-minor-axis",
        ),
        pytest.param(
            replace(
                adaptive_detection(),
                minor_axis_camera=(float("nan"), 1.0, 0.0),
            ),
            "minor axis must be finite and non-zero",
            id="invalid-minor-axis",
        ),
        pytest.param(
            replace(
                adaptive_detection(),
                major_axis_length_m=None,
                minor_axis_length_m=None,
            ),
            "measured major and minor axis lengths",
            id="missing-axis-lengths",
        ),
        pytest.param(
            replace(adaptive_detection(), major_axis_length_m=0.041),
            "axis ratio",
            id="insufficient-axis-ratio",
        ),
    ),
)
def test_strict_pca_short_edge_rejects_incomplete_geometry_before_motion(
    detection,
    message,
):
    config = load_config(FIVE_POINT_ADAPTIVE_SAFE_CONFIG)
    plan_calls: list[str] = []

    class NoMotionPlanner(PosePlanner):
        def _adaptive_object_geometry(self, _detection, _state):
            return (
                np.asarray(ADAPTIVE_POINT_CENTERS[0], dtype=float),
                np.array((1.0, 0.0, 0.0), dtype=float),
                np.eye(4),
            )

        def plan(self, *_args, **_kwargs):
            plan_calls.append("plan")
            raise AssertionError("invalid PCA geometry reached motion planning")

    planner = NoMotionPlanner(config)
    taught_rpy = config.task_b.point_grasp_rpy[0]
    with pytest.raises(RuntimeError, match=message):
        planner.plan_adaptive_candidates(
            detection,
            robot_state(taught_rpy),
            place_name="pick_1",
            grasp_rpy=taught_rpy,
            task_b_anchor_pose=Pose(
                (ADAPTIVE_POINT_CENTERS[0][0], ADAPTIVE_POINT_CENTERS[0][1], 0.40),
                taught_rpy,
            ),
            vertical_grasp_approach_height_m=0.075,
            jaw_tip_object_offset_m=(
                config.task_b.point_grasp_jaw_tip_object_offset_m[0]
            ),
        )
    assert plan_calls == []


def test_strict_pca_short_edge_replanner_rejects_nonzero_learned_offset():
    config = load_config(FIVE_POINT_ADAPTIVE_SAFE_CONFIG)
    planner = PosePlanner(config)
    detection = adaptive_detection()
    taught_rpy = config.task_b.point_grasp_rpy[0]
    planner._adaptive_object_geometry = lambda _detection, _state: (
        np.asarray(ADAPTIVE_POINT_CENTERS[0], dtype=float),
        np.array((1.0, 0.0, 0.0), dtype=float),
        np.eye(4),
    )

    with pytest.raises(
        RuntimeError,
        match="严格 PCA 中心抓取不允许学习 XY 偏移",
    ):
        planner.plan_adaptive_candidates(
            detection,
            robot_state(taught_rpy),
            place_name="pick_1",
            grasp_rpy=taught_rpy,
            task_b_anchor_pose=Pose(
                (ADAPTIVE_POINT_CENTERS[0][0], ADAPTIVE_POINT_CENTERS[0][1], 0.40),
                taught_rpy,
            ),
            vertical_grasp_approach_height_m=0.075,
            jaw_tip_object_offset_m=(0.004, -0.023),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "point_grasp_jaw_tip_object_offset_m",
            ((0.004, -0.023), None, None, None, None),
            "严格 PCA 中心抓取不允许学习 XY 偏移",
        ),
        (
            "point_grasp_xy_m",
            ((0.10, 0.20), None, None, None, None),
            "严格 PCA 中心抓取不允许 point_grasp_xy_m 固定抓取点",
        ),
        (
            "point_grasp_xy_offset_m",
            ((0.004, -0.023), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            "严格 PCA 中心抓取不允许 point_grasp_xy_offset_m XY 修正",
        ),
        (
            "point_grasp_xy_from_observation",
            (True, False, False, False, False),
            "严格 PCA 中心抓取不允许使用 observation-anchor XY",
        ),
    ),
)
def test_strict_pca_config_fails_closed_for_any_xy_correction(
    field,
    value,
    message,
):
    config = load_config(FIVE_POINT_ADAPTIVE_SAFE_CONFIG)
    invalid = replace(config, task_b=replace(config.task_b, **{field: value}))

    with pytest.raises(ValueError, match=message):
        invalid.validate()


def test_task_b_center_pca_profiles_keep_zero_xy_corrections_and_expected_speeds():
    low_speed = load_config(PICK1_LOW_SPEED_LIVE_CONFIG)
    five_point = load_config(FIVE_POINT_ADAPTIVE_SAFE_CONFIG)
    fast = load_config(PICK1_FAST_LIVE_CONFIG)

    for config in (low_speed, five_point, fast):
        assert config.adaptive_inverse.candidates.require_closure_axis_alignment is True
        assert config.adaptive_inverse.candidates.max_count == 1
        assert config.adaptive_inverse.selection.stop_after_first_valid is True
        assert config.task_b.point_grasp_xy_offset_m == ()
        assert all(
            value is None
            for value in config.task_b.point_grasp_jaw_tip_object_offset_m
        )

    assert low_speed.motion.speed_mps == pytest.approx(0.01)
    assert five_point.motion.speed_mps == pytest.approx(0.12)
    assert five_point.motion.dry_run is True
    assert five_point.motion.backend == "robotmove"
    assert five_point.gripper.backend == "mock"
    assert fast.motion.speed_mps == pytest.approx(0.06)
    assert fast.motion.dry_run is True
    assert fast.gripper.backend == "mock"


def test_annotated_frame_marks_pca_center_long_axis_and_short_axis(tmp_path):
    cv2 = pytest.importorskip("cv2")
    config = load_config(FIVE_POINT_ADAPTIVE_SAFE_CONFIG).vision
    vision = RealSenseYoloVision(config)
    vision._output_dir = tmp_path
    vision._last_save_s = -1e9
    vision._frame_index = 7
    detection = replace(
        adaptive_detection(),
        bbox=(110.0, 80.0, 210.0, 160.0),
        camera_xyz=(0.0, 0.0, 0.50),
        object_center_camera_xyz=(0.0, 0.0, 0.50),
        major_axis_camera=(1.0, 0.0, 0.0),
        minor_axis_camera=(0.0, 1.0, 0.0),
        major_axis_length_m=0.20,
        minor_axis_length_m=0.08,
    )
    color = np.full((240, 320, 3), 32, dtype=np.uint8)
    intrinsics = CameraIntrinsics(
        width=320,
        height=240,
        fx=500.0,
        fy=500.0,
        cx=160.0,
        cy=120.0,
    )

    path = vision._save_frame(color, (detection,), intrinsics)

    assert path == tmp_path / "frame_000007.jpg"
    assert path.is_file()
    annotated = cv2.imread(str(path))
    assert annotated is not None
    bgr = annotated.astype(np.int16)
    yellow = (
        (bgr[:, :, 0] < 100)
        & (bgr[:, :, 1] > 160)
        & (bgr[:, :, 2] > 160)
    )
    cyan = (
        (bgr[:, :, 0] > 160)
        & (bgr[:, :, 1] > 160)
        & (bgr[:, :, 2] < 100)
    )
    assert int(np.count_nonzero(yellow)) > 0
    assert int(np.count_nonzero(cyan)) > 0


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


def test_first_pca_route_failure_falls_back_to_taught_rescue_route():
    pipeline, detection = synthetic_adaptive_pipeline(rescue_adaptive_config())
    pipeline.planner.plan_adaptive_candidates = lambda *_args, **_kwargs: (
        candidate(detection, x=0.10, yaw_deg=20.0),
        candidate(
            detection,
            x=0.20,
            yaw_deg=-15.0,
            pitch_deg=-5.0,
            orientation_source="taught_rescue",
        ),
    )
    original = pipeline.motion.precheck_linear_targets
    calls: list[tuple[Pose, ...]] = []

    def precheck(targets, *, step_m):
        calls.append(tuple(targets))
        if len(calls) == 1:
            raise RuntimeError("synthetic PCA route rejection")
        return original(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = precheck
    result = pipeline.run_once()

    assert result.success
    assert len(calls) == 2
    assert pipeline.motion.moves[0][1].xyz[0] == pytest.approx(0.20)
    event_text = " | ".join(event.message for event in result.events)
    assert "orientation_source=taught_rescue" in event_text


def test_all_adaptive_candidates_rejected_before_any_arm_or_gripper_action():
    pipeline, detection = synthetic_adaptive_pipeline(rescue_adaptive_config())
    pipeline.planner.plan_adaptive_candidates = lambda *_args, **_kwargs: (
        candidate(detection, x=0.10, yaw_deg=20.0),
        candidate(
            detection,
            x=0.20,
            yaw_deg=-15.0,
            pitch_deg=-5.0,
            orientation_source="taught_rescue",
        ),
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


def _five_point_adaptive_mock_config():
    config = load_config(FIVE_POINT_ADAPTIVE_SAFE_CONFIG)
    return replace(
        config,
        task_b=replace(config.task_b, lift_hold_s=0.0),
        vision=replace(
            config.vision,
            backend="mock",
            scenario_file=str(
                ROOT
                / "app"
                / "track1_grasp"
                / "scenarios"
                / "mock_task_b_scene.yaml"
            ),
            show=False,
            continuous=False,
        ),
        motion=replace(config.motion, backend="mock", dry_run=True),
        gripper=replace(config.gripper, backend="mock"),
    )


def _install_five_point_adaptive_mock(pipeline, *, fail_place=""):
    """Exercise the real adaptive planner with deterministic mock geometry.

    ``fail_place`` rejects every adaptive route for that point.  The separate
    success test passes ``fail_sources`` through the returned dictionary by
    updating ``scenario["fail_sources"]`` before ``pipeline.run()``.
    """

    initial_q = load_joint_pose_file(
        pipeline.config.task_b.initial_pose_file,
        expected_role="task_b_initial_pose",
    )
    acquisitions: list[dict[str, object]] = []
    geometry_calls: list[tuple[tuple[float, float, float], float]] = []
    candidate_plans: list[tuple[str, tuple[AdaptiveGraspCandidate, ...]]] = []
    route_attempts: list[tuple[str, str, float, float]] = []
    active_candidates: tuple[AdaptiveGraspCandidate, ...] = ()
    active_place = ""
    scenario = {
        "fail_place": fail_place,
        "fail_sources": (),
        "acquisitions": acquisitions,
        "geometry_calls": geometry_calls,
        "candidate_plans": candidate_plans,
        "route_attempts": route_attempts,
        "initial_q": initial_q,
    }
    geometry_source = iter(
        zip(ADAPTIVE_POINT_CENTERS, ADAPTIVE_PCA_YAW_DEG, strict=True)
    )

    def geometry(_detection, _robot_state):
        center, yaw_deg = next(geometry_source)
        geometry_calls.append((center, yaw_deg))
        yaw_rad = radians(yaw_deg)
        return (
            np.asarray(center, dtype=float),
            np.asarray((cos(yaw_rad), sin(yaw_rad), 0.0), dtype=float),
            np.eye(4),
        )

    point_yaws = {
        f"pick_{index}": yaw_deg
        for index, yaw_deg in enumerate(ADAPTIVE_PCA_YAW_DEG, start=1)
    }

    def acquire(
        initial_state,
        expected_point,
        expected_name,
        expected_label,
        round_number,
        total_rounds,
        _emit,
        *,
        completed_points=(),
    ):
        state = pipeline.motion.get_state()
        yaw_rad = radians(point_yaws.get(expected_name, 0.0))
        base_camera = (
            pipeline.planner._base_flange_transform(state)
            @ pipeline.planner._camera_transform()
        )
        desired_major_base = np.asarray(
            (cos(yaw_rad), sin(yaw_rad), 0.0),
            dtype=float,
        )
        major_camera = base_camera[:3, :3].T @ desired_major_base
        major_camera /= np.linalg.norm(major_camera)
        detection = adaptive_detection(
            major_axis_camera=tuple(float(value) for value in major_camera)
        )
        capture = CaptureRobotState(before=state, after=state)
        frame = Frame(
            f"formal-adaptive-{expected_name}",
            (detection,),
            capture_robot_state=capture,
        )
        acquisitions.append(
            {
                "name": expected_name,
                "point": tuple(expected_point),
                "label": expected_label,
                "round": round_number,
                "total_rounds": total_rounds,
                "completed_points": tuple(completed_points),
                "initial_state": initial_state,
            }
        )
        return (
            detection,
            frame,
            state,
            f"mock confirmed at {expected_name}",
        )

    original_plan = pipeline.planner.plan_adaptive_candidates

    def plan(*args, **kwargs):
        nonlocal active_candidates, active_place
        candidates = original_plan(*args, **kwargs)
        active_place = str(kwargs.get("place_name", ""))
        active_candidates = tuple(candidates)
        candidate_plans.append((active_place, active_candidates))
        return candidates

    original_precheck = pipeline.motion.precheck_linear_targets

    def precheck(targets, *, step_m):
        target_tuple = tuple(targets)
        selected = next(
            (
                candidate
                for candidate in active_candidates
                if tuple(
                    pose for _name, pose in candidate.plan.ordered_waypoints()
                )
                == target_tuple
            ),
            None,
        )
        source = (
            "unknown"
            if selected is None
            else str(selected.orientation_source)
        )
        yaw_offset = 0.0 if selected is None else float(selected.yaw_offset_deg)
        pitch_offset = (
            0.0 if selected is None else float(selected.pitch_offset_deg)
        )
        route_attempts.append(
            (active_place, source, yaw_offset, pitch_offset)
        )
        if active_place == scenario["fail_place"] and (
            not scenario["fail_sources"]
            or source in scenario["fail_sources"]
        ):
            raise RuntimeError(
                f"synthetic {source} route rejection at {active_place}"
            )
        return original_precheck(targets, step_m=step_m)

    pipeline.planner._adaptive_object_geometry = geometry
    pipeline.planner.plan_adaptive_candidates = plan
    pipeline._acquire_task_b_target = acquire
    pipeline.motion.precheck_linear_targets = precheck
    return scenario


def test_five_point_adaptive_safe_config_runs_continuously_with_strict_pca():
    config = _five_point_adaptive_mock_config()
    pipeline = build_pick_pipeline(config)
    scenario = _install_five_point_adaptive_mock(pipeline)

    try:
        report = pipeline.run()
    finally:
        pipeline.close()

    assert report.requested_picks == 5
    assert report.attempts == 5
    assert report.successes == 5
    assert [result.place_name for result in report.results] == [
        "pick_1",
        "pick_2",
        "pick_3",
        "pick_4",
        "pick_5",
    ]
    assert [item["name"] for item in scenario["acquisitions"]] == [
        "pick_1",
        "pick_2",
        "pick_3",
        "pick_4",
        "pick_5",
    ]
    assert scenario["geometry_calls"] == list(
        zip(ADAPTIVE_POINT_CENTERS, ADAPTIVE_PCA_YAW_DEG, strict=True)
    )
    assert [len(candidates) for _name, candidates in scenario["candidate_plans"]] == [
        1,
        1,
        1,
        1,
        1,
    ]
    assert [
        (place, source)
        for place, source, _yaw, _pitch in scenario["route_attempts"]
    ] == [
        ("pick_1", "task_b_banana_waist_short_axis"),
        ("pick_2", "task_b_banana_waist_short_axis"),
        ("pick_3", "task_b_banana_waist_short_axis"),
        ("pick_4", "task_b_banana_waist_short_axis"),
        ("pick_5", "task_b_banana_waist_short_axis"),
    ]
    for _name, candidates in scenario["candidate_plans"]:
        assert (
            candidates[0].plan.grasp_geometry_mode
            == "task_b_banana_waist_short_axis"
        )
        assert candidates[0].plan.grasp_close_width_m == pytest.approx(0.04)
        assert candidates[0].plan.grasp_close_width_m == pytest.approx(
            candidates[0].plan.target.minor_axis_length_m
        )

    assert pipeline.motion.dry_run is True
    assert len(pipeline.motion.moves) == 30
    for index in range(5):
        assert [
            name
            for name, _pose in pipeline.motion.moves[index * 6 : (index + 1) * 6]
        ] == [
            "pregrasp",
            "grasp",
            "lift",
            "transfer",
            "release",
            "retreat",
        ]
    expected_joint_names = ["task_b_initial_pose"]
    for point_number in range(1, 6):
        expected_joint_names.extend(
            [
                f"task_b_observation_pick_{point_number}",
                f"task_b_return_initial_pick_{point_number}",
            ]
        )
    assert [
        name for name, _target in pipeline.motion.joint_moves
    ] == expected_joint_names
    initial_q = scenario["initial_q"]
    assert pipeline.motion.joint_moves[-1][1] == pytest.approx(initial_q)
    assert pipeline.motion.get_state().joints == pytest.approx(initial_q)
    assert pipeline.gripper.events == [
        "open",
        "close",
        "open",
        "close",
        "open",
        "close",
        "open",
        "close",
        "open",
        "close",
        "open",
    ]
    assert not any(
        "taught_rescue" in event.message
        for result in report.results
        for event in result.events
    )

def test_five_point_adaptive_safe_config_stops_after_a_rejected_third_point():
    config = _five_point_adaptive_mock_config()
    pipeline = build_pick_pipeline(config)
    scenario = _install_five_point_adaptive_mock(pipeline, fail_place="pick_3")

    try:
        report = pipeline.run()
    finally:
        pipeline.close()

    assert report.requested_picks == 5
    assert report.attempts == 3
    assert report.successes == 2
    assert [result.place_name for result in report.results] == [
        "pick_1",
        "pick_2",
        "pick_3",
    ]
    assert report.results[-1].status is PickStatus.REJECTED
    assert [item["name"] for item in scenario["acquisitions"]] == [
        "pick_1",
        "pick_2",
        "pick_3",
    ]
    assert [len(candidates) for _name, candidates in scenario["candidate_plans"]] == [
        1,
        1,
        1,
    ]
    assert [
        (place, source)
        for place, source, _yaw, _pitch in scenario["route_attempts"]
    ] == [
        ("pick_1", "task_b_banana_waist_short_axis"),
        ("pick_2", "task_b_banana_waist_short_axis"),
        ("pick_3", "task_b_banana_waist_short_axis"),
    ]
    assert len(pipeline.motion.moves) == 12
    assert [name for name, _pose in pipeline.motion.moves] == [
        name
        for _ in range(2)
        for name in (
            "pregrasp",
            "grasp",
            "lift",
            "transfer",
            "release",
            "retreat",
        )
    ]
    assert [name for name, _target in pipeline.motion.joint_moves] == [
        "task_b_initial_pose",
        "task_b_observation_pick_1",
        "task_b_return_initial_pick_1",
        "task_b_observation_pick_2",
        "task_b_return_initial_pick_2",
        "task_b_observation_pick_3",
    ]
    assert pipeline.gripper.events == [
        "open",
        "close",
        "open",
        "close",
        "open",
    ]
    assert any(
        "candidate(s) were rejected before any motion or gripper command"
        in event.message
        for event in report.results[-1].events
    )


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

    factory_calls: list[tuple[AdaptiveInverseConfig | None, object]] = []

    def fake_build_vision(
        _config,
        *,
        log=None,
        adaptive_inverse=None,
        object_pose_config=None,
    ):
        factory_calls.append((adaptive_inverse, object_pose_config))
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(factory_module, "build_vision", fake_build_vision)
    monkeypatch.setattr(
        factory_module,
        "build_motion",
        lambda _config, node=None, log=None, profile=None: SimpleNamespace(dry_run=True),
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

    assert factory_calls == [
        (None, None),
        (enabled.adaptive_inverse, None),
    ]
