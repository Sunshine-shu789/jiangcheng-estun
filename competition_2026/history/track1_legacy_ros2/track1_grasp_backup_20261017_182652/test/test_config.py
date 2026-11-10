from pathlib import Path

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import TaskMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"


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


def test_task_b_place_poses_come_from_the_example_file():
    config = apply_mode(load_config(CONFIG), "B")
    poses = config.place_poses
    assert len(poses) == 5
    # four bin corners plus the centre
    assert poses[4][:2] == (0.0, 0.0)
    assert len(config.task_b.place_names) == 5


def test_task_a_iou_threshold_matches_the_competition():
    config = load_config(CONFIG)
    assert config.task_a.iou_threshold == 0.75
