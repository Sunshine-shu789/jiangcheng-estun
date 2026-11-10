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


def test_task_b_live_config_waits_for_each_next_point_and_returns_in_place():
    config = load_config(ROOT.parents[1] / "config" / "track1_task_b_live.yaml")

    assert config.task.mode.value == "B"
    assert config.task.max_picks == 5
    assert config.task.stop_on_error is True
    assert config.task_b.rounds == 5
    assert config.task_b.return_to_pick is True
    assert config.task_b.target_class == "8-Plastic_Banana"
    assert config.task_b.next_point_timeout_s == 0.0
    assert config.task_b.next_point_poll_s > 0.0
    assert len(config.place_poses) == 5
    assert config.search.enabled is True
    assert config.search.observation_offsets_xy_m == (
        (0.0, 0.0),
        (-0.10, 0.0),
        (0.10, 0.0),
        (0.0, -0.10),
        (0.0, 0.10),
    )
    assert config.search.observation_clearance_m == pytest.approx(0.20)
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"


def test_search_approach_watch_mode_keeps_preview_without_enabling_motion():
    pytest.importorskip("rclpy")
    from track1_grasp.nodes.search_approach_node import build_parser

    args = build_parser().parse_args(["--watch"])

    assert args.watch is True
    assert args.execute is False


def test_operator_acceptance_cli_and_clearing_exit_code_are_strict():
    pytest.importorskip("rclpy")
    from track1_grasp.domain import RunReport
    from track1_grasp.nodes.track1_node import (
        build_parser,
        clearing_run_exit_code,
    )

    args = build_parser().parse_args(
        ["--task", "C", "--acceptance-max-successes", "1"]
    )
    assert args.acceptance_max_successes == 1

    completed_once = RunReport(0, 1, 1, ())
    rejected_once = RunReport(0, 1, 0, ())
    confirmed_empty = RunReport(0, 1, 0, (), empty_bin_confirmed=True)

    assert clearing_run_exit_code(
        completed_once,
        acceptance_max_successes=1,
    ) == 0
    assert clearing_run_exit_code(
        rejected_once,
        acceptance_max_successes=1,
    ) == 1
    assert clearing_run_exit_code(
        completed_once,
        acceptance_max_successes=None,
    ) == 1
    assert clearing_run_exit_code(
        confirmed_empty,
        acceptance_max_successes=None,
    ) == 0


@pytest.mark.parametrize("task", ("a", "b", "c", "d", "e"))
def test_task_launch_entry_points_compile(task, tmp_path):
    path = LAUNCH_DIR / f"task_{task}.launch.py"
    py_compile.compile(str(path), cfile=tmp_path / f"task_{task}.pyc", doraise=True)
