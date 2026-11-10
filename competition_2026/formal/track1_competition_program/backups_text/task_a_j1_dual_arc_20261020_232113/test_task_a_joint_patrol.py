from dataclasses import replace
import math
from pathlib import Path

import pytest

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import Detection, Frame
from track1_grasp.motion import MockMotionBackend
from track1_grasp.vision_pipeline import VisionDemoPipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_task_a_legacy.yaml"

TEN_CLASSES = (
    "1-Cheez-it",
    "2-Starkist_Tuna",
    "3-Scissors",
    "4-Frenchs_Mustard",
    "5-Tomato_Soup",
    "6-Foam_Brick",
    "7-Clamp",
    "8-Plastic_Banana",
    "9-Mug",
    "10-meat_can",
)
HISTORICAL_STARTUP_DEG = (82.36, 54.32, -32.29, -53.57, 84.38, -2.34)
HISTORICAL_STARTUP_RAD = tuple(
    math.radians(value) for value in HISTORICAL_STARTUP_DEG
)


def joint_patrol_config():
    config = apply_mode(load_config(CONFIG), "A")
    visibility = replace(config.task_a.patrol.object_visibility, enabled=False)
    return replace(
        config,
        task_a=replace(
            config.task_a,
            patrol=replace(
                config.task_a.patrol,
                enabled=True,
                settle_s=0.0,
                object_visibility=visibility,
            ),
        ),
    )


class StartupMotion(MockMotionBackend):
    def __init__(self, config):
        super().__init__(config)
        self._joints = HISTORICAL_STARTUP_RAD


class ScriptedVision:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.closed = False

    def capture(self):
        if self.index >= len(self.frames):
            raise AssertionError("scripted vision exhausted")
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def close(self):
        self.closed = True


def class_frames(labels=TEN_CLASSES):
    return [
        Frame(
            frame_id=f"joint-patrol-{index:02d}",
            detections=(
                Detection(
                    label,
                    0.95,
                    (20.0, 20.0, 120.0, 120.0),
                ),
            ),
        )
        for index, label in enumerate(labels, start=1)
    ]


def make_pipeline(config, *, frames=None):
    motion = StartupMotion(config.motion)
    vision = ScriptedVision(class_frames() if frames is None else frames)
    pipeline = VisionDemoPipeline(
        config,
        vision=vision,
        motion=motion,
        log=lambda _message: None,
        sleep=lambda _seconds: None,
    )
    return pipeline, motion, vision


def joint_route_kwargs(config):
    patrol = config.task_a.patrol
    return {
        "allowed_joints": tuple(joint - 1 for joint in patrol.movable_joints),
        "max_initial_move_deg": patrol.max_initial_move_deg,
        "max_target_move_deg": patrol.max_target_move_deg,
        "max_step_deg": patrol.joint_max_step_deg,
        "min_endpoint_distance_m": patrol.min_move_distance_m,
        "name_prefix": "task-a-test",
    }


def test_every_joint_target_exactly_preserves_j1_j5_j6():
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)

    targets = pipeline._joint_patrol_targets(motion.get_state())

    assert len(targets) == 10
    for target in targets:
        for locked_index in (0, 4, 5):
            assert target[locked_index] == HISTORICAL_STARTUP_RAD[locked_index]
        changed = {
            index
            for index, value in enumerate(target)
            if value != HISTORICAL_STARTUP_RAD[index]
        }
        assert changed.issubset({1, 2, 3})


@pytest.mark.parametrize("locked_joint", (1, 5, 6))
def test_config_rejects_locked_joints_in_the_movable_set(locked_joint):
    config = joint_patrol_config()
    invalid = replace(
        config,
        task_a=replace(
            config.task_a,
            patrol=replace(
                config.task_a.patrol,
                movable_joints=(locked_joint,),
                joint_offsets_deg=tuple((0.0,) for _ in range(10)),
            ),
        ),
    )

    with pytest.raises(ValueError, match="Joint1, Joint5 and Joint6 are locked"):
        invalid.validate()


@pytest.mark.parametrize("invalid_joint_index", (0, 4, 5))
def test_joint_route_precheck_rejects_j1_j5_j6_immediately(invalid_joint_index):
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)
    targets = pipeline._joint_patrol_targets(motion.get_state())
    kwargs = joint_route_kwargs(config)
    kwargs["allowed_joints"] = (*kwargs["allowed_joints"], invalid_joint_index)

    with pytest.raises(
        ValueError,
        match=rf"Joint2, Joint3 and Joint4.*invalid indexes=\[{invalid_joint_index}\]",
    ):
        motion.precheck_joint_route(targets, **kwargs)

    assert motion.joint_moves == []


def test_historical_startup_pose_keeps_all_ten_segments_above_15_cm():
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)
    targets = pipeline._joint_patrol_targets(motion.get_state())

    route = motion.precheck_joint_route(targets, **joint_route_kwargs(config))

    assert len(route.targets) == 10
    assert len(route.tcp_targets) == 10
    assert len(route.distances_m) == 10
    assert min(route.distances_m) >= config.task_a.patrol.min_move_distance_m
    assert motion.joint_moves == []


def test_all_ten_joint_targets_are_prechecked_before_any_step_executes():
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)
    events = []

    original_precheck = motion.precheck_joint_route
    original_execute = motion.execute_joint_route_step

    def precheck(targets, **kwargs):
        events.append(("precheck", len(targets)))
        return original_precheck(targets, **kwargs)

    def execute(route, *, index, name):
        events.append(("execute", index + 1))
        return original_execute(route, index=index, name=name)

    motion.precheck_joint_route = precheck
    motion.execute_joint_route_step = execute

    report = pipeline.run()

    assert events == [("precheck", 10), *(
        ("execute", index) for index in range(1, 11)
    )]
    assert report.frames == 10
    assert len(report.patrol_joint_targets_deg) == 10
    assert len(motion.joint_moves) == 10


def test_patrol_report_covers_all_ten_configured_classes():
    config = joint_patrol_config()
    pipeline, _motion, vision = make_pipeline(config)

    report = pipeline.run()

    assert report.frames == 10
    assert report.patrol_detected_classes == tuple(sorted(TEN_CLASSES))
    assert report.patrol_missing_classes == ()
    assert vision.index == 10


def test_prepare_config_revalidates_command_line_overrides():
    pytest.importorskip("rclpy")
    from track1_grasp.nodes.track1_node import prepare_config

    config = joint_patrol_config()
    prepared = prepare_config(
        config,
        max_picks=1,
        frames=5,
        execute=True,
    )

    assert prepared.task.max_picks == 1
    assert prepared.task_a.max_frames == 5
    assert prepared.motion.dry_run is False

    with pytest.raises(ValueError, match="max_picks must be >= 1"):
        prepare_config(config, max_picks=0)
    with pytest.raises(ValueError, match="max_frames must be >= 0"):
        prepare_config(config, frames=-1)
