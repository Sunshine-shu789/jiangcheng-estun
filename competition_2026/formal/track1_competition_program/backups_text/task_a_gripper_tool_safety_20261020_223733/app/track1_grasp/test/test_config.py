from dataclasses import replace
from pathlib import Path

import pytest

from track1_grasp.config import (
    MotionConfig,
    TaskBConfig,
    VisionConfig,
    apply_mode,
    load_config,
)
from track1_grasp.domain import TaskMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"
LIVE_CONFIG = ROOT / "config" / "track1_live.yaml"


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
    assert apply_mode(base, "C").target.policy == "any"
    assert apply_mode(base, "D").target.policy == "known_and_unseen"
    ordered = apply_mode(base, "E")
    assert ordered.target.policy == "ordered"
    assert ordered.order == ("mug", "object", "bowl", "book")


def test_mock_config_exposes_a_per_task_scenario_for_task_a():
    config = apply_mode(load_config(CONFIG), "A")
    scenario = config.vision.scenario_for(config.task.mode)
    assert Path(scenario).name == "mock_task_a_scene.yaml"
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


def test_live_task_a_enables_a_ten_position_patrol_with_15cm_minimum():
    config = apply_mode(load_config(LIVE_CONFIG), "A")
    patrol = config.task_a.patrol
    assert patrol.enabled is True
    assert patrol.position_count == 10
    assert patrol.min_move_distance_m == 0.15
    assert patrol.stride_m >= patrol.min_move_distance_m
    assert patrol.linear_step_m == 0.01


def test_patrol_rejects_a_stride_shorter_than_the_required_move():
    config = load_config(CONFIG)
    invalid = replace(
        config,
        task_a=replace(
            config.task_a,
            patrol=replace(
                config.task_a.patrol,
                enabled=True,
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


def test_task_b_launcher_preserves_stdin_for_operator_pause():
    formal_root = Path(__file__).resolve().parents[3]
    launcher = formal_root / "bin" / "track1_task_runner"
    source = launcher.read_text(encoding="utf-8")

    # Bash redirects asynchronous command stdin to /dev/null unless it is
    # explicitly preserved.  Task B's operator gate must remain interactive.
    assert '<&0 > >(tee -a "$LOG_FILE")' in source
