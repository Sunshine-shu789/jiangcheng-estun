import py_compile
from pathlib import Path

import pytest

from track1_grasp.config import load_config
from track1_grasp.config_paths import package_config_path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = ROOT / "launch"


def test_package_config_resolver_works_from_source_tree():
    assert package_config_path("track1_mock.yaml") == ROOT / "config" / "track1_mock.yaml"


def test_live_config_is_safe_test_configuration():
    config = load_config(ROOT / "config" / "track1_live.yaml")

    assert config.vision.backend == "realsense_yolo"
    weights = Path(config.vision.weights)
    assert weights.is_file()
    assert weights.parent.name == "weights"
    assert config.vision.target_class == "6-Foam_Brick"
    assert config.vision.rotate_180 is True
    assert config.vision.continuous is True
    assert config.vision.show is True
    assert config.vision.stabilize is True
    assert config.vision.stabilizer_confirm_hits == 3
    assert config.vision.save_annotated is False
    assert config.motion.backend == "robotmove"
    assert config.motion.motion_mode == "joint_ik"
    assert config.motion.dry_run is True
    assert config.servo.max_step_m == pytest.approx(0.005)
    assert config.gripper.backend == "mock"


def test_search_approach_watch_mode_keeps_preview_without_enabling_motion():
    pytest.importorskip("rclpy")
    from track1_grasp.nodes.search_approach_node import build_parser

    args = build_parser().parse_args(["--watch"])

    assert args.watch is True
    assert args.execute is False


@pytest.mark.parametrize("task", ("a", "b", "c", "d", "e"))
def test_task_launch_entry_points_compile(task, tmp_path):
    path = LAUNCH_DIR / f"task_{task}.launch.py"
    py_compile.compile(str(path), cfile=tmp_path / f"task_{task}.pyc", doraise=True)
