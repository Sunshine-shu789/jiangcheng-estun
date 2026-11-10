from dataclasses import replace
from pathlib import Path

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import TaskMode
from track1_grasp.factory import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"


def test_task_b_places_the_object_at_five_slots():
    config = apply_mode(load_config(CONFIG), "B")
    assert config.task.mode is TaskMode.B
    pipeline = build_pipeline(config)
    report = pipeline.run()

    assert report.successes == 5
    assert report.requested_picks == 5
    places = [result.place_name for result in report.results]
    assert places == [
        "corner_front_left",
        "corner_front_right",
        "corner_back_right",
        "corner_back_left",
        "centre",
    ]

    # every cycle walks pregrasp -> grasp -> lift -> transfer -> release -> retreat
    names = [name for name, _pose in pipeline.motion.moves[:6]]
    assert names == ["pregrasp", "grasp", "lift", "transfer", "release", "retreat"]
    for move_name, pose in pipeline.motion.moves:
        if move_name == "release":
            assert pose.xyz[2] == 0.08


def test_task_b_uses_the_configured_place_poses():
    config = apply_mode(load_config(CONFIG), "B")
    poses = config.place_poses
    assert len(poses) == 5
    pipeline = build_pipeline(config)
    report = pipeline.run()
    released = [pose for name, pose in pipeline.motion.moves if name == "release"]
    assert [tuple(p.xyz) for p in released] == [tuple(p[:3]) for p in poses]
    assert report.successes == 5


def test_task_b_requires_place_poses():
    config = apply_mode(load_config(CONFIG), "B")
    broken = replace(config, task_b=replace(config.task_b, place_poses_file=""))
    try:
        broken.validate()
    except ValueError as exc:
        assert "task B needs" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("task B should require place poses")
