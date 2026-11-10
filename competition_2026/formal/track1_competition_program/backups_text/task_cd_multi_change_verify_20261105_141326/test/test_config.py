from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from track1_grasp.config import (
    AdaptiveCandidatesConfig,
    CornerSmallObjectFilterConfig,
    GraspConsensusGateConfig,
    MotionConfig,
    SearchConfig,
    TaskBConfig,
    Track1Config,
    VisionConfig,
    apply_mode,
    load_config,
    load_config_for_read_only_vision,
)
from track1_grasp.collision import GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC
from track1_grasp.domain import TaskMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"
LIVE_CONFIG = ROOT / "config" / "track1_live.yaml"
FORMAL_ROOT = Path(__file__).resolve().parents[3]
FORMAL_TASK_A_CONFIGS = (
    ROOT / "config" / "track1_task_a_legacy.yaml",
    FORMAL_ROOT / "config" / "track1_task_a_legacy.yaml",
)
ARBITRARY_MOCK_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_b_arbitrary_mock.yaml"
)
ARBITRARY_HARDWARE_CONFIGS = (
    FORMAL_ROOT / "config" / "track1_task_b_arbitrary_single_trial.yaml",
    FORMAL_ROOT / "config" / "track1_task_b_arbitrary_single_preflight.yaml",
    FORMAL_ROOT / "config" / "track1_task_b_arbitrary_single_live.yaml",
)
POINT2_SAFE_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_b_point2_safe_20261027.yaml"
)
ADAPTIVE_SAFE_CONFIG = (
    FORMAL_ROOT
    / "config"
    / "track1_task_b_five_point_adaptive_safe_20261102.yaml"
)
FORMAL_CD_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml"
)
FORMAL_CD_CONFIGS = (
    FORMAL_ROOT / "config" / "track1_task_c_v11_official_20261102.yaml",
    FORMAL_ROOT / "config" / "track1_task_d_v11_official_20261102.yaml",
    FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml",
    FORMAL_ROOT
    / "config"
    / "track1_task_c_v11_official_mustard_single_20261102.yaml",
)
FORMAL_CORNER_SMALL_FILTER_CONFIGS = FORMAL_CD_CONFIGS[:3]


def _synthetic_box_clearance_payload():
    # Test-only geometry.  It must never enter an official YAML file.
    return {
        "required": True,
        "geometry_validated": True,
        "geometry_source": GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC,
        "center_xy_m": [0.0, 0.0],
        "yaw_rad": 0.0,
        "interior_size_m": [2.0, 2.0],
        "wall_clearance_m": 0.0,
        "obstacle_clearance_m": 0.0,
        "gripper_footprint_flange_xy_m": [
            [-0.06, -0.02],
            [0.06, -0.02],
            [0.06, 0.02],
            [-0.06, 0.02],
        ],
        "obstacles_m": [],
    }


@pytest.mark.raw_formal_cd_config
@pytest.mark.parametrize("config_path", FORMAL_CORNER_SMALL_FILTER_CONFIGS)
def test_official_cd_yaml_does_not_require_unmeasured_box_clearance(config_path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    clearance = payload["task_cd"].get("box_clearance")

    assert clearance is None or clearance.get("required", False) is False


@pytest.mark.raw_formal_cd_config
@pytest.mark.parametrize("config_path", FORMAL_CORNER_SMALL_FILTER_CONFIGS)
def test_official_cd_yaml_enables_adaptive_corner_small_filter(config_path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["vision"]["corner_small_object_filter"] == {
        "enabled": True,
        "mode": "adaptive",
    }


@pytest.mark.raw_formal_cd_config
def test_synthetic_task_cd_box_clearance_is_rejected_outside_fully_mock():
    config_path = FORMAL_CD_CONFIGS[0]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["task_cd"]["box_clearance"] = _synthetic_box_clearance_payload()

    with pytest.raises(
        ValueError,
        match=(
            "task_cd\\.box_clearance synthetic geometry is restricted to the "
            "fully offline Mock pipeline"
        ),
    ):
        Track1Config.from_dict(payload, base_dir=config_path.parent)


@pytest.mark.raw_formal_cd_config
@pytest.mark.parametrize(
    ("config_path", "task_mode"),
    (
        (FORMAL_CD_CONFIGS[0], TaskMode.C),
        (FORMAL_CD_CONFIGS[1], TaskMode.D),
        (FORMAL_CD_CONFIGS[2], TaskMode.D),
    ),
)
def test_read_only_vision_loader_forces_non_actuating_config(config_path, task_mode):
    config = load_config_for_read_only_vision(config_path, task_mode=task_mode)

    assert config.task.mode is task_mode
    assert config.motion.backend == "mock"
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"
    assert config.task_cd.box_clearance.required is False

    # The loader is an in-memory diagnostic view; the shipped mission YAML must
    # remain non-actuating when loaded normally.
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["motion"]["backend"] != "mock"
    assert payload["motion"]["dry_run"] is False
    assert payload["gripper"]["backend"] != "mock"
    clearance = payload["task_cd"].get("box_clearance", {})
    assert clearance.get("required", False) is False


@pytest.mark.raw_formal_cd_config
def test_read_only_vision_loader_rejects_non_cd_modes():
    with pytest.raises(ValueError, match="restricted to Task C/D"):
        load_config_for_read_only_vision(FORMAL_CD_CONFIG, task_mode=TaskMode.A)


@pytest.mark.raw_formal_cd_config
@pytest.mark.parametrize("config_path", FORMAL_CD_CONFIGS)
def test_official_cd_yaml_enables_only_measured_vertical_geometry_modes(config_path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    short_edge = payload["task_cd"]["short_edge_grasp"]

    assert short_edge["pca"]["defer_axis_ratio_rejection"] is True
    assert short_edge["geometry_policy"] == {
        "enabled": True,
        "allow_rim": True,
        "allow_vertical_center": True,
        "allow_horizontal_sidewall": False,
    }


def test_load_mock_config_resolves_relative_paths():
    config = load_config(CONFIG)
    assert config.task.mode is TaskMode.C
    assert Path(config.vision.scenario_file).is_file()
    assert config.motion.backend == "mock"


def test_each_task_selects_the_expected_target_policy():
    base = load_config(CONFIG)
    assert apply_mode(base, "A").task.mode is TaskMode.A
    assert apply_mode(base, "A").target.policy == "any"
    assert apply_mode(base, "B").target.policy == "any"
    assert apply_mode(base, "C").target.policy == "known"
    assert apply_mode(base, "D").target.policy == "known_and_unseen"
    ordered = apply_mode(base, "E")
    assert ordered.target.policy == "ordered"
    assert ordered.order == ("mug", "object", "bowl", "book")


def test_mock_config_exposes_separate_task_scenarios():
    base = load_config(CONFIG)
    expected = {
        "A": "mock_task_a_scene.yaml",
        "B": "mock_task_b_scene.yaml",
        "C": "mock_task_c_scene.yaml",
        "D": "mock_task_d_scene.yaml",
    }
    for mode, expected_name in expected.items():
        config = apply_mode(base, mode)
        scenario = config.vision.scenario_for(config.task.mode)
        assert Path(scenario).name == expected_name
        assert Path(scenario).is_file()


def test_task_b_reference_points_come_from_the_example_file():
    config = apply_mode(load_config(CONFIG), "B")
    poses = config.place_poses
    assert len(poses) == 5
    assert [pose[:2] for pose in poses] == [
        (0.15, -0.15),
        (-0.15, -0.10),
        (0.12, 0.12),
        (-0.12, 0.12),
        (0.00, 0.00),
    ]
    assert config.task_b.return_to_pick is True
    assert config.task_b.target_class == "8-Plastic_Banana"
    assert len(config.task_b.place_names) == 5


def test_task_a_iou_threshold_matches_the_competition():
    config = load_config(CONFIG)
    assert config.task_a.iou_threshold == 0.75


@pytest.mark.parametrize("config_path", FORMAL_TASK_A_CONFIGS)
def test_formal_task_a_uses_j1_j4_dual_arc_and_locks_j5_j6(config_path):
    raw_config = load_config(config_path)
    assert raw_config.task.mode is TaskMode.A
    config = apply_mode(raw_config, "A")
    patrol = config.task_a.patrol

    assert patrol.enabled is True
    assert patrol.mode == "joint_arc"
    assert patrol.position_count == 10
    assert patrol.movable_joints == (1, 2, 3, 4)
    assert patrol.joint1_relative_targets_deg == (
        135.0,
        67.5,
        0.0,
        -67.5,
        -135.0,
        -135.0,
        -67.5,
        0.0,
        67.5,
        135.0,
    )
    assert len(patrol.joint_offsets_deg) == 10
    assert all(len(row) == 3 for row in patrol.joint_offsets_deg)
    assert patrol.arc_indices == (1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
    assert patrol.min_move_distance_m == 0.15
    assert patrol.max_initial_move_deg == 120.0
    assert patrol.max_target_move_deg == 120.0
    assert patrol.post_detection_dwell_s == 3.0
    assert patrol.return_to_start is True
    assert patrol.manual_step is False


def test_historical_patrol_without_a_mode_remains_cartesian_compatible():
    config = load_config(CONFIG)

    assert config.task_a.patrol.mode == "cartesian_box"


def test_patrol_rejects_a_stride_shorter_than_the_required_move():
    config = load_config(CONFIG)
    invalid = replace(
        config,
        task_a=replace(
            config.task_a,
            patrol=replace(
                config.task_a.patrol,
                enabled=True,
                mode="cartesian_box",
                min_move_distance_m=0.15,
                stride_m=0.10,
            ),
        ),
    )
    with pytest.raises(ValueError, match="stride_m"):
        invalid.validate()


def test_live_stabilizer_settings_are_loaded_from_nested_yaml():
    config = VisionConfig.from_dict(
        {
            "backend": "realsense_yolo",
            "stabilizer": {
                "enabled": True,
                "inference_min_confidence": 0.12,
                "confirm_hits": 4,
                "max_misses": 2,
                "class_switch_hits": 5,
            },
        },
        ROOT,
    )
    assert config.stabilize is True
    assert config.stabilizer_inference_min_confidence == 0.12
    assert config.stabilizer_confirm_hits == 4
    assert config.stabilizer_max_misses == 2
    assert config.stabilizer_class_switch_hits == 5


def test_corner_small_object_filter_is_loaded_from_nested_yaml():
    config = VisionConfig.from_dict(
        {
            "corner_small_object_filter": {
                "enabled": True,
                "mode": "adaptive",
                "corner_zone_ratio": 0.28,
                "max_box_width_ratio": 0.10,
                "max_box_height_ratio": 0.13,
            }
        },
        ROOT,
    )

    assert config.corner_small_object_filter.enabled is True
    assert config.corner_small_object_filter.mode == "adaptive"
    assert config.corner_small_object_filter.corner_zone_ratio == 0.28
    assert config.corner_small_object_filter.max_box_width_ratio == 0.10
    assert config.corner_small_object_filter.max_box_height_ratio == 0.13


def test_corner_small_object_filter_rejects_string_boolean():
    with pytest.raises(ValueError, match="must be a boolean"):
        VisionConfig.from_dict(
            {
                "corner_small_object_filter": {
                    "enabled": "true",
                }
            },
            ROOT,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"corner_zone_ratio": 0.0},
        {"corner_zone_ratio": 0.5},
        {"max_box_width_ratio": 0.0},
        {"max_box_height_ratio": 1.01},
        {"corner_zone_ratio": float("nan")},
        {"mode": "unknown"},
    ),
)
def test_corner_small_object_filter_rejects_invalid_ratios(overrides):
    config = CornerSmallObjectFilterConfig(enabled=True, **overrides)

    with pytest.raises(ValueError, match="corner_small_object_filter"):
        config.validate()


def test_motion_parses_optional_verified_task_b_grasp_baseline():
    config = MotionConfig.from_dict(
        {
            "grasp_z_m": 0.161534092,
            "grasp_rpy": [3.082206718, 0.098641098, -0.072880619],
        }
    )

    assert config.grasp_z_m == pytest.approx(0.161534092)
    assert config.grasp_rpy == pytest.approx(
        (3.082206718, 0.098641098, -0.072880619)
    )
    assert MotionConfig.from_dict({}).grasp_z_m is None
    assert MotionConfig.from_dict({}).grasp_rpy is None


@pytest.mark.parametrize(
    "field_name",
    ("ik_position_tolerance_m", "ik_orientation_tolerance_rad"),
)
@pytest.mark.parametrize(
    "value",
    (0.0, -1e-6, float("nan"), float("inf"), 0.005000001),
)
def test_motion_ik_acceptance_tolerances_fail_closed(field_name, value):
    config = load_config(CONFIG)
    invalid = replace(
        config,
        motion=replace(config.motion, **{field_name: value}),
    )

    with pytest.raises(ValueError, match=rf"motion\.{field_name}"):
        invalid.validate()


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("ik_position_tolerance_m", 0.002),
        ("ik_orientation_tolerance_rad", 0.005),
    ),
)
def test_motion_ik_acceptance_tolerances_accept_upper_boundary(field_name, value):
    config = load_config(CONFIG)
    boundary = replace(
        config,
        motion=replace(config.motion, **{field_name: value}),
    )

    boundary.validate()


def test_task_b_per_point_grasp_overrides_must_match_the_point_count():
    base = apply_mode(load_config(CONFIG), "B")
    point_count = len(base.place_poses)
    valid = replace(
        base,
        task_b=replace(
            base.task_b,
            point_grasp_z_m=tuple(0.16 + index * 0.001 for index in range(point_count)),
            point_grasp_xy_from_observation=tuple(
                index == 1 for index in range(point_count)
            ),
            point_grasp_xy_m=tuple(
                (0.10 + index * 0.001, 0.20)
                if index == 2
                else None
                for index in range(point_count)
            ),
            point_grasp_xy_offset_m=tuple(
                (index * 0.001, 0.0) for index in range(point_count)
            ),
            point_grasp_rpy=tuple(
                (3.1, 0.05 + index * 0.001, -0.1) for index in range(point_count)
            ),
            point_vertical_grasp_approach_height_m=tuple(
                0.075 if index == 1 else None for index in range(point_count)
            ),
        ),
    )
    valid.validate()

    invalid = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_grasp_rpy=valid.task_b.point_grasp_rpy[:-1],
        ),
    )
    with pytest.raises(ValueError, match="exactly 5 entries"):
        invalid.validate()

    invalid_xy_count = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_grasp_xy_from_observation=(
                valid.task_b.point_grasp_xy_from_observation[:-1]
            ),
        ),
    )
    with pytest.raises(ValueError, match="exactly 5 entries"):
        invalid_xy_count.validate()

    invalid_xy_value = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_grasp_xy_from_observation=(
                "false",
                True,
                False,
                False,
                False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="entry 1 must be boolean"):
        invalid_xy_value.validate()

    invalid_explicit_xy_count = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_grasp_xy_m=valid.task_b.point_grasp_xy_m[:-1],
        ),
    )
    with pytest.raises(ValueError, match="exactly 5 entries"):
        invalid_explicit_xy_count.validate()

    invalid_offset_count = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_grasp_xy_offset_m=valid.task_b.point_grasp_xy_offset_m[:-1],
        ),
    )
    with pytest.raises(ValueError, match="exactly 5 entries"):
        invalid_offset_count.validate()

    invalid_offset_value = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_grasp_xy_offset_m=(
                (0.0, 0.0),
                (float("nan"), 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="entry 2 must contain finite values"):
        invalid_offset_value.validate()

    invalid_approach_count = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_vertical_grasp_approach_height_m=(
                valid.task_b.point_vertical_grasp_approach_height_m[:-1]
            ),
        ),
    )
    with pytest.raises(ValueError, match="exactly 5 entries"):
        invalid_approach_count.validate()

    invalid_approach_value = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_vertical_grasp_approach_height_m=(
                None,
                0.0,
                None,
                None,
                None,
            ),
        ),
    )
    with pytest.raises(ValueError, match="entry 2 must be positive and finite"):
        invalid_approach_value.validate()

    conflicting_xy = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_grasp_xy_m=(
                None,
                (0.10, 0.20),
                None,
                None,
                None,
            ),
        ),
    )
    with pytest.raises(ValueError, match="conflicts with"):
        conflicting_xy.validate()


def test_task_b_parses_point_observation_xy_switches():
    parsed = TaskBConfig.from_dict(
        {
            "point_grasp_xy_from_observation": [False, True, False],
            "point_grasp_xy_m": [None, None, [0.0658, 0.4439]],
            "point_grasp_xy_offset_m": [
                [0.0, 0.0],
                [0.015, 0.0],
                [0.0, -0.005],
            ],
        },
        ROOT,
    )

    assert parsed.point_grasp_xy_from_observation == (False, True, False)
    assert parsed.point_grasp_xy_m == (None, None, (0.0658, 0.4439))
    assert parsed.point_grasp_xy_offset_m == (
        (0.0, 0.0),
        (0.015, 0.0),
        (0.0, -0.005),
    )
    with pytest.raises(ValueError, match="boolean entries"):
        TaskBConfig.from_dict(
            {"point_grasp_xy_from_observation": [False, "true"]},
            ROOT,
        )
    with pytest.raises(ValueError, match="2-value vector"):
        TaskBConfig.from_dict(
            {"point_grasp_xy_m": [(0.1, 0.2, 0.3)]},
            ROOT,
        )
    with pytest.raises(ValueError, match="2-value vector"):
        TaskBConfig.from_dict(
            {"point_grasp_xy_offset_m": [(0.015, 0.0, 0.0)]},
            ROOT,
        )


def test_task_b_parses_per_point_vertical_approach_heights():
    parsed = TaskBConfig.from_dict(
        {
            "point_vertical_grasp_approach_height_m": [
                None,
                0.075,
                None,
                0.06,
            ],
        },
        ROOT,
    )

    assert parsed.point_vertical_grasp_approach_height_m == (
        None,
        0.075,
        None,
        0.06,
    )


def test_task_b_rejects_non_finite_per_point_grasp_values():
    base = apply_mode(load_config(CONFIG), "B")
    invalid_z = replace(
        base,
        task_b=replace(
            base.task_b,
            point_grasp_z_m=(0.16, float("nan"), None, None, None),
        ),
    )
    with pytest.raises(ValueError, match="entry 2 must be finite"):
        invalid_z.validate()

    invalid_rpy = replace(
        base,
        task_b=replace(
            base.task_b,
            point_grasp_rpy=(
                (float("inf"), 0.0, 0.0),
                None,
                None,
                None,
                None,
            ),
        ),
    )
    with pytest.raises(ValueError, match="entry 1 must contain finite values"):
        invalid_rpy.validate()


@pytest.mark.parametrize(
    "field_name",
    (
        "return_to_initial_after_task",
        "return_to_initial_after_each_pick",
    ),
)
def test_task_b_return_to_initial_requires_a_pose_file(field_name):
    base = apply_mode(load_config(CONFIG), "B")
    invalid = replace(
        base,
        task_b=replace(
            base.task_b,
            initial_pose_file="",
            **{field_name: True},
        ),
    )
    with pytest.raises(ValueError, match=field_name):
        invalid.validate()


def test_task_b_parses_and_validates_observation_joint_poses():
    parsed = TaskBConfig.from_dict(
        {
            "point_observation_joints_deg": [
                [10.0, -5.0, 4.0, 3.0, -2.0, 6.0],
                None,
            ],
            "observation_pose_tolerance_deg": 0.25,
        },
        ROOT,
    )
    assert parsed.point_observation_joints_deg == (
        (10.0, -5.0, 4.0, 3.0, -2.0, 6.0),
        None,
    )
    assert parsed.observation_pose_tolerance_deg == pytest.approx(0.25)

    parsed_with_observation_return = TaskBConfig.from_dict(
        {
            "point_observation_joints_deg": [
                [10.0, -5.0, 4.0, 3.0, -2.0, 6.0],
            ],
            "return_to_observation_after_each_pick": True,
        },
        ROOT,
    )
    assert parsed_with_observation_return.return_to_observation_after_each_pick

    base = apply_mode(load_config(CONFIG), "B")
    point_count = len(base.place_poses)
    valid = replace(
        base,
        task_b=replace(
            base.task_b,
            point_observation_joints_deg=(
                (10.0, -5.0, 4.0, 3.0, -2.0, 6.0),
                None,
                (9.0, -4.0, 3.0, 2.0, -1.0, 5.0),
                None,
                (8.0, -3.0, 2.0, 1.0, 0.0, 4.0),
            ),
        ),
    )
    assert len(valid.task_b.point_observation_joints_deg) == point_count
    valid.validate()

    invalid_count = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_observation_joints_deg=valid.task_b.point_observation_joints_deg[:-1],
        ),
    )
    with pytest.raises(ValueError, match="exactly 5 entries"):
        invalid_count.validate()

    invalid_value = replace(
        valid,
        task_b=replace(
            valid.task_b,
            point_observation_joints_deg=(
                (float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0),
                None,
                None,
                None,
                None,
            ),
        ),
    )
    with pytest.raises(ValueError, match="entry 1 must contain finite values"):
        invalid_value.validate()

    invalid_tolerance = replace(
        valid,
        task_b=replace(valid.task_b, observation_pose_tolerance_deg=0.0),
    )
    with pytest.raises(ValueError, match="observation_pose_tolerance_deg"):
        invalid_tolerance.validate()

    missing_observation = replace(
        base,
        task_b=replace(
            base.task_b,
            point_observation_joints_deg=(),
            return_to_initial_after_each_pick=False,
            return_to_observation_after_each_pick=True,
        ),
    )
    with pytest.raises(ValueError, match="needs task_b.point_observation_joints_deg"):
        missing_observation.validate()

    incomplete_observation = replace(
        valid,
        task_b=replace(
            valid.task_b,
            return_to_initial_after_each_pick=False,
            return_to_observation_after_each_pick=True,
        ),
    )
    with pytest.raises(ValueError, match="concrete observation posture"):
        incomplete_observation.validate()

    conflicting_returns = replace(
        valid,
        task_b=replace(
            valid.task_b,
            initial_pose_file="/tmp/task_b_initial_pose.yaml",
            point_observation_joints_deg=tuple(
                (10.0, -5.0, 4.0, 3.0, -2.0, 6.0)
                for _ in range(point_count)
            ),
            return_to_initial_after_each_pick=True,
            return_to_observation_after_each_pick=True,
        ),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        conflicting_returns.validate()


@pytest.mark.parametrize(
    "config_name",
    [
        "track1_task_b_20261020.yaml",
        "track1_task_b_final_20261020_live.yaml",
        "track1_task_b_resume2_20261020_live.yaml",
        "track1_task_b_resume3_5_20261020_live.yaml",
    ],
)
def test_formal_task_b_configs_keep_verified_safe_search_views(config_name):
    formal_root = Path(__file__).resolve().parents[3]
    config = load_config(formal_root / "config" / config_name)

    assert config.search.observation_offsets_xy_m == (
        (0.00, 0.00),
        (0.08, 0.00),
        (0.00, -0.08),
    )
    assert config.motion.cartesian_min_jacobian_singular_value == pytest.approx(0.01)


def test_live_fixed_point_adaptive_config_requires_a_grasp_depth_floor():
    config = load_config(POINT2_SAFE_CONFIG)
    invalid = replace(
        config,
        adaptive_inverse=replace(
            config.adaptive_inverse,
            candidates=replace(
                config.adaptive_inverse.candidates,
                minimum_grasp_flange_z_m=None,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="live fixed-point adaptive grasping requires",
    ):
        invalid.validate()


def test_live_fixed_point_adaptive_config_rejects_visual_z_with_taught_depth():
    config = load_config(POINT2_SAFE_CONFIG)
    invalid = replace(
        config,
        adaptive_inverse=replace(
            config.adaptive_inverse,
            candidates=replace(
                config.adaptive_inverse.candidates,
                use_visual_grasp_z=True,
            ),
        ),
    )

    with pytest.raises(ValueError, match="cannot combine"):
        invalid.validate()


def test_live_fixed_point_adaptive_config_rejects_depth_below_floor():
    config = load_config(POINT2_SAFE_CONFIG)
    invalid = replace(
        config,
        task_b=replace(config.task_b, point_grasp_z_m=(0.117953,)),
    )

    with pytest.raises(ValueError, match="configured fixed grasp flange Z is below"):
        invalid.validate()


def test_locked_taught_orientation_rejects_extra_candidate_rotation():
    config = load_config(POINT2_SAFE_CONFIG)
    invalid = replace(
        config,
        adaptive_inverse=replace(
            config.adaptive_inverse,
            candidates=replace(
                config.adaptive_inverse.candidates,
                yaw_offsets_deg=(0.0, 45.0),
            ),
        ),
    )

    with pytest.raises(ValueError, match="lock_taught_orientation"):
        invalid.validate()


def test_taught_orientation_rescue_defaults_yaml_and_validation():
    defaults = AdaptiveCandidatesConfig.from_dict({})
    assert defaults.lock_taught_orientation is False
    assert defaults.taught_orientation_rescue_enabled is False
    assert defaults.taught_yaw_offsets_deg == ()
    assert defaults.taught_pitch_offsets_deg == ()

    parsed = AdaptiveCandidatesConfig.from_dict(
        {
            "taught_orientation_rescue_enabled": True,
            "taught_yaw_offsets_deg": [0.0, -15.0, 15.0],
            "taught_pitch_offsets_deg": [0.0, -5.0],
        }
    )
    assert parsed.taught_orientation_rescue_enabled is True
    assert parsed.taught_yaw_offsets_deg == pytest.approx((0.0, -15.0, 15.0))
    assert parsed.taught_pitch_offsets_deg == pytest.approx((0.0, -5.0))

    config = load_config(ADAPTIVE_SAFE_CONFIG)
    config.validate()
    candidates = config.adaptive_inverse.candidates
    assert candidates.lock_taught_orientation is False
    assert candidates.taught_orientation_rescue_enabled is True
    assert candidates.taught_yaw_offsets_deg == pytest.approx(
        (0.0, -15.0, 15.0, -30.0, 30.0)
    )
    assert candidates.taught_pitch_offsets_deg == pytest.approx(
        (0.0, -5.0, -10.0)
    )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        (
            "lock_taught_orientation",
            "false",
            "lock_taught_orientation must be a boolean",
        ),
        (
            "taught_orientation_rescue_enabled",
            "false",
            "taught_orientation_rescue_enabled must be a boolean",
        ),
        (
            "use_visual_grasp_z",
            "false",
            "use_visual_grasp_z must be a boolean",
        ),
        (
            "require_jaw_tip_object_offset",
            "false",
            "require_jaw_tip_object_offset must be a boolean",
        ),
        (
            "taught_yaw_offsets_deg",
            "0,-15",
            "taught_yaw_offsets_deg must be a sequence",
        ),
        (
            "taught_pitch_offsets_deg",
            "0,-5",
            "taught_pitch_offsets_deg must be a sequence",
        ),
        (
            "taught_yaw_offsets_deg",
            [],
            "taught_yaw_offsets_deg must not be empty",
        ),
        (
            "taught_pitch_offsets_deg",
            [],
            "taught_pitch_offsets_deg must not be empty",
        ),
    ),
)
def test_adaptive_rescue_candidate_parsing_rejects_unsafe_types(
    field_name, value, message
):
    with pytest.raises(ValueError, match=message):
        AdaptiveCandidatesConfig.from_dict({field_name: value})


@pytest.mark.parametrize(
    ("field_name", "values"),
    (
        ("taught_yaw_offsets_deg", (float("nan"),)),
        ("taught_yaw_offsets_deg", (float("inf"),)),
        ("taught_pitch_offsets_deg", (0.0, float("nan"))),
        ("taught_pitch_offsets_deg", (0.0, float("inf"))),
    ),
)
def test_adaptive_rescue_candidate_validation_rejects_non_finite_offsets(
    field_name, values
):
    config = load_config(POINT2_SAFE_CONFIG)
    candidates = replace(
        config.adaptive_inverse.candidates,
        lock_taught_orientation=False,
        taught_orientation_rescue_enabled=True,
        taught_yaw_offsets_deg=(0.0, -15.0, 15.0),
        taught_pitch_offsets_deg=(0.0, -5.0, -10.0),
    )
    candidates = replace(candidates, **{field_name: values})
    invalid = replace(
        config,
        adaptive_inverse=replace(config.adaptive_inverse, candidates=candidates),
    )

    with pytest.raises(ValueError, match=field_name):
        invalid.validate()


def test_adaptive_rescue_candidate_validation_rejects_unsafe_combinations():
    config = load_config(POINT2_SAFE_CONFIG)
    valid_candidates = replace(
        config.adaptive_inverse.candidates,
        lock_taught_orientation=False,
        taught_orientation_rescue_enabled=True,
        taught_yaw_offsets_deg=(0.0, -15.0, 15.0),
        taught_pitch_offsets_deg=(0.0, -5.0, -10.0),
    )

    both_modes = replace(
        valid_candidates,
        lock_taught_orientation=True,
    )
    missing_yaw = replace(valid_candidates, taught_yaw_offsets_deg=())
    missing_pitch = replace(valid_candidates, taught_pitch_offsets_deg=())
    nonzero_first_pitch = replace(
        valid_candidates,
        taught_pitch_offsets_deg=(-5.0, 0.0),
    )

    cases = (
        (both_modes, "mutually exclusive"),
        (missing_yaw, "requires taught_yaw_offsets_deg"),
        (missing_pitch, "requires taught_pitch_offsets_deg"),
        (nonzero_first_pitch, "must start with 0.0"),
    )
    for candidates, message in cases:
        invalid = replace(
            config,
            adaptive_inverse=replace(
                config.adaptive_inverse,
                candidates=candidates,
            ),
        )
        with pytest.raises(ValueError, match=message):
            invalid.validate()

    arbitrary_base = load_config(ARBITRARY_MOCK_CONFIG)
    arbitrary_candidates = replace(
        arbitrary_base.adaptive_inverse.candidates,
        lock_taught_orientation=False,
        taught_orientation_rescue_enabled=True,
        taught_yaw_offsets_deg=(0.0, -15.0, 15.0),
        taught_pitch_offsets_deg=(0.0, -5.0, -10.0),
    )
    arbitrary = replace(
        arbitrary_base,
        adaptive_inverse=replace(
            arbitrary_base.adaptive_inverse,
            candidates=arbitrary_candidates,
        ),
    )
    with pytest.raises(ValueError, match="only supported for fixed-point"):
        arbitrary.validate()


def test_point2_safe_config_loads_confirmed_absolute_ik_seed():
    config = load_config(POINT2_SAFE_CONFIG)

    expected_seed = (
        95.162161976,
        87.247202447,
        -21.787902852,
        -20.862341925,
        87.957880257,
        -82.543682644,
    )
    assert len(config.motion.ik_fallback_seeds_deg) == 1
    assert config.motion.ik_fallback_seeds_deg[0] == pytest.approx(
        expected_seed
    )
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"


@pytest.mark.parametrize(
    ("learned_seed", "message"),
    (
        ((0.0,) * 5, r"ik_fallback_seeds_deg\[0\].*exactly six"),
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, float("nan")),
            r"ik_fallback_seeds_deg\[0\].*non-finite",
        ),
        (
            (200.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            r"ik_fallback_seeds_deg\[0\].*joint limits",
        ),
    ),
)
def test_motion_rejects_invalid_learned_ik_seed(learned_seed, message):
    config = load_config(POINT2_SAFE_CONFIG)
    invalid = replace(
        config,
        motion=replace(
            config.motion,
            ik_fallback_seeds_deg=(learned_seed,),
        ),
    )

    with pytest.raises(ValueError, match=message):
        invalid.validate()


def test_search_config_parses_relative_observation_adjustments():
    config = SearchConfig.from_dict(
        {
            "enabled": True,
            "arbitrary_observation_offsets_xyz_m": [
                [0.0, 0.0, 0.020],
                [-0.010, 0.005, 0.015],
            ],
        }
    )

    assert config.arbitrary_observation_offsets_xyz_m == (
        (0.0, 0.0, 0.020),
        (-0.010, 0.005, 0.015),
    )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ([[0.0, 0.020]], r"entries must be \[x, y, z\]"),
        ([["x", 0.0, 0.020]], "entries must be numeric"),
        ([[float("nan"), 0.0, 0.020]], "entry 1 must be finite"),
        ([[0.0, 0.0, 0.0]], "entry 1 must not be"),
        ([[0.0, 0.0, 0.101]], "exceeds the 0.10 m"),
    ),
)
def test_search_config_rejects_invalid_relative_observation_adjustments(
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        SearchConfig.from_dict(
            {"arbitrary_observation_offsets_xyz_m": value}
        )


def test_arbitrary_mock_config_enables_relative_observation_adjustment():
    config = load_config(ARBITRARY_MOCK_CONFIG)

    assert config.search.enabled is True
    assert config.search.arbitrary_observation_offsets_xyz_m == (
        (0.0, 0.0, 0.020),
    )
    assert config.search.return_to_initial_on_miss is True
    assert config.motion.cartesian_min_jacobian_singular_value == pytest.approx(
        0.01
    )


@pytest.mark.parametrize("config_path", ARBITRARY_HARDWARE_CONFIGS)
def test_arbitrary_hardware_configs_fail_closed_without_box_geometry(
    config_path,
):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    clearance = payload["adaptive_inverse"]["box_clearance"]
    assert clearance["required"] is True
    assert clearance["geometry_validated"] is False
    assert clearance["geometry_source"] == "unconfigured"

    with pytest.raises(
        ValueError,
        match=(
            "box_clearance is required but complete box geometry and a "
            "validated gripper footprint are missing"
        ),
    ):
        load_config(config_path)


def test_task_b_launcher_preserves_stdin_for_operator_pause():
    formal_root = Path(__file__).resolve().parents[3]
    launcher = formal_root / "bin" / "track1_task_runner"
    source = launcher.read_text(encoding="utf-8")

    # Bash redirects asynchronous command stdin to /dev/null unless it is
    # explicitly preserved.  Task B's operator gate must remain interactive.
    assert '<&0 > >(tee -a "$LOG_FILE")' in source


def _enabled_consensus_config():
    base = apply_mode(load_config(CONFIG), "C")
    gate = replace(
        base.grasp_consensus,
        enabled=True,
        report_file=str(ROOT / "reports" / "scene_consensus.json"),
    )
    return replace(
        base,
        pose=replace(base.pose, robot_pose_frame="tcp"),
        grasp_consensus=gate,
    )


def test_grasp_consensus_gate_is_disabled_by_default():
    config = load_config(CONFIG)

    assert config.grasp_consensus == GraspConsensusGateConfig()
    assert config.grasp_consensus.enabled is False
    assert config.grasp_consensus.minimum_distinct_frames == 4
    assert config.grasp_consensus.maximum_translation_tolerance_m == pytest.approx(
        0.025
    )
    assert config.grasp_consensus.maximum_jaw_rotation_tolerance_deg == pytest.approx(
        15.0
    )


def test_enabled_grasp_consensus_accepts_safe_offline_task_c_config():
    config = _enabled_consensus_config()

    config.validate()


def test_enabled_grasp_consensus_rejects_non_cd_task():
    config = _enabled_consensus_config()
    invalid = replace(
        config,
        task=replace(config.task, mode=TaskMode.E),
    )

    with pytest.raises(ValueError, match="restricted to Task C/D"):
        invalid.validate()


def test_enabled_grasp_consensus_rejects_real_motion():
    config = _enabled_consensus_config()
    invalid = replace(
        config,
        motion=replace(config.motion, dry_run=False),
    )

    with pytest.raises(ValueError, match="motion.dry_run=true"):
        invalid.validate()


def test_enabled_grasp_consensus_rejects_non_mock_gripper():
    config = _enabled_consensus_config()
    invalid = replace(
        config,
        gripper=replace(
            config.gripper,
            backend="taska",
            taska_config=str(ROOT / "config" / "unused_taska.yaml"),
        ),
    )

    with pytest.raises(ValueError, match="gripper.backend=mock"):
        invalid.validate()


def test_enabled_grasp_consensus_rejects_search_motion():
    config = _enabled_consensus_config()
    invalid = replace(
        config,
        search=replace(config.search, enabled=True),
    )

    with pytest.raises(ValueError, match="search.enabled=false"):
        invalid.validate()


def test_enabled_grasp_consensus_requires_report_path():
    config = _enabled_consensus_config()
    invalid = replace(
        config,
        grasp_consensus=replace(config.grasp_consensus, report_file=""),
    )

    with pytest.raises(ValueError, match="report_file is required"):
        invalid.validate()


def test_enabled_grasp_consensus_requires_explicit_pose_frame():
    config = _enabled_consensus_config()
    invalid = replace(
        config,
        pose=replace(config.pose, robot_pose_frame="unknown"),
    )

    with pytest.raises(ValueError, match="robot_pose_frame=flange or tcp"):
        invalid.validate()


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        (
            "minimum_distinct_frames",
            3,
            "minimum_distinct_frames must be an integer >= 4",
        ),
        (
            "maximum_translation_tolerance_m",
            0.0251,
            "must not exceed the safe maximum",
        ),
        (
            "maximum_jaw_rotation_tolerance_deg",
            15.1,
            "must not exceed the safe maximum",
        ),
        (
            "maximum_pose_translation_error_m",
            0.0251,
            "must not exceed the safe maximum",
        ),
        (
            "maximum_pose_jaw_rotation_error_deg",
            15.1,
            "must not exceed the safe maximum",
        ),
        (
            "maximum_translation_tolerance_m",
            float("nan"),
            "must be positive and finite",
        ),
    ),
)
def test_grasp_consensus_thresholds_fail_closed(field_name, value, message):
    config = _enabled_consensus_config()
    invalid = replace(
        config,
        grasp_consensus=replace(
            config.grasp_consensus,
            **{field_name: value},
        ),
    )

    with pytest.raises(ValueError, match=message):
        invalid.validate()



@pytest.mark.parametrize(
    ("tilt_angles_deg", "message"),
    (
        ((0.0,), "must be finite and non-zero"),
        ((float("nan"),), "must be finite and non-zero"),
        ((15.0, 15.0), "must not contain duplicates"),
        ((30.001,), "must not exceed 30 degrees"),
    ),
)
def test_formal_cd_short_edge_tilt_angles_fail_closed(
    tilt_angles_deg,
    message,
):
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    invalid = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                tilt_angles_deg=tilt_angles_deg,
            ),
        ),
    )

    with pytest.raises(ValueError, match=message):
        invalid.validate()


def test_formal_cd_short_edge_tilt_must_respect_global_safety_limit():
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    safety = replace(
        config.motion.grasp_safety,
        maximum_tool_tilt_deg=20.0,
    )
    invalid = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                tilt_angles_deg=(25.0,),
                tool_axis_approach_height_m=0.08,
            ),
        ),
        motion=replace(config.motion, grasp_safety=safety),
    )

    with pytest.raises(ValueError, match="maximum_tool_tilt_deg"):
        invalid.validate()


def test_formal_cd_short_edge_tilt_requires_approach_height():
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    invalid = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                tilt_angles_deg=(15.0,),
                tool_axis_approach_height_m=None,
            ),
        ),
    )

    with pytest.raises(ValueError, match="tool_axis_approach_height_m is required"):
        invalid.validate()


@pytest.mark.parametrize(
    ("approach_height_m", "message"),
    (
        (0.049, "below motion.grasp_safety.minimum_approach_height_m"),
        (0.121, "above motion.grasp_safety.maximum_approach_height_m"),
    ),
)
def test_formal_cd_short_edge_approach_height_respects_safety_bounds(
    approach_height_m,
    message,
):
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    invalid = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                tilt_angles_deg=(15.0,),
                tool_axis_approach_height_m=approach_height_m,
            ),
        ),
    )

    with pytest.raises(ValueError, match=message):
        invalid.validate()
