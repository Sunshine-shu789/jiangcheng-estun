from pathlib import Path

from task_cde_grasp.config import apply_mode, load_config
from task_cde_grasp.domain import TaskMode


ROOT = Path(__file__).resolve().parents[1]


def test_load_mock_config_resolves_relative_paths():
    config = load_config(ROOT / "config" / "cde_mock.yaml")
    assert config.task.mode is TaskMode.C
    assert Path(config.vision.scenario_file).is_file()
    assert config.motion.backend == "mock"


def test_apply_mode_changes_policy_and_order_remains_available():
    base = load_config(ROOT / "config" / "cde_mock.yaml")
    assert apply_mode(base, "C").target.policy == "any"
    assert apply_mode(base, "D").target.policy == "known_and_unseen"
    ordered = apply_mode(base, "E")
    assert ordered.target.policy == "ordered"
    assert ordered.order == ("mug", "object", "bowl", "book")
