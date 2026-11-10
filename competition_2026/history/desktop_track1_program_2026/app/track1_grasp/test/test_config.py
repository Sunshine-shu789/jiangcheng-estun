from dataclasses import replace
from pathlib import Path

import pytest

from track1_grasp.config import VisionConfig, apply_mode, load_config
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
    assert config.task_b.target_class == "6-Foam_Brick"
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
