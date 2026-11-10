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


def test_j1_uses_absolute_dual_arc_targets_and_j5_j6_stay_exact():
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)

    targets = pipeline._joint_patrol_targets(motion.get_state())

    assert len(targets) == 10
    for target, expected_j1_deg in zip(
        targets, config.task_a.patrol.joint1_targets_deg
    ):
        assert target[0] == pytest.approx(math.radians(expected_j1_deg))
        for locked_index in (4, 5):
            assert target[locked_index] == HISTORICAL_STARTUP_RAD[locked_index]
        changed = {
            index
            for index, value in enumerate(target)
            if value != HISTORICAL_STARTUP_RAD[index]
        }
        assert changed.issubset({0, 1, 2, 3})


def test_j1_targets_cover_two_full_270_degree_arcs():
    config = joint_patrol_config()
    targets = config.task_a.patrol.joint1_targets_deg

    assert targets[:5] == pytest.approx((135.0, 67.5, 0.0, -67.5, -135.0))
    assert targets[5:] == pytest.approx((-135.0, -67.5, 0.0, 67.5, 135.0))
    assert targets[4] == targets[5]
    assert min(targets) == -135.0
    assert max(targets) == 135.0


@pytest.mark.parametrize(
    "movable_joints",
    (
        (2, 4),
        (1, 2, 3),
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 6),
    ),
)
def test_config_requires_exactly_j1_j2_j3_j4_movable(movable_joints):
    config = joint_patrol_config()
    invalid = replace(
        config,
        task_a=replace(
            config.task_a,
            patrol=replace(
                config.task_a.patrol,
                movable_joints=movable_joints,
            ),
        ),
    )

    with pytest.raises(ValueError, match="must be exactly.*Joint5 and Joint6"):
        invalid.validate()


@pytest.mark.parametrize(
    ("j1_targets", "message"),
    (
        (
            (135.0, 67.5, 0.0, -67.5, -136.0,
             -135.0, -67.5, 0.0, 67.5, 135.0),
            "safe 270-degree range",
        ),
        (
            (135.0, 0.0, 67.5, -67.5, -135.0,
             -135.0, -67.5, 0.0, 67.5, 135.0),
            "first J1 arc must decrease monotonically",
        ),
        (
            (135.0, 67.5, 0.0, -67.5, -135.0,
             -135.0, 0.0, -67.5, 67.5, 135.0),
            "second J1 arc must increase monotonically",
        ),
    ),
)
def test_config_rejects_invalid_j1_dual_arc(j1_targets, message):
    config = joint_patrol_config()
    invalid = replace(
        config,
        task_a=replace(
            config.task_a,
            patrol=replace(
                config.task_a.patrol,
                joint1_targets_deg=j1_targets,
            ),
        ),
    )

    with pytest.raises(ValueError, match=message):
        invalid.validate()


@pytest.mark.parametrize("invalid_joint_index", (4, 5))
def test_joint_route_precheck_rejects_j5_j6_immediately(invalid_joint_index):
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)
    targets = pipeline._joint_patrol_targets(motion.get_state())
    kwargs = joint_route_kwargs(config)
    kwargs["allowed_joints"] = (*kwargs["allowed_joints"], invalid_joint_index)

    with pytest.raises(
        ValueError,
        match=rf"Joint1, Joint2, Joint3 and Joint4.*invalid indexes=\[{invalid_joint_index}\]",
    ):
        motion.precheck_joint_route(targets, **kwargs)

    assert motion.joint_moves == []


def test_historical_startup_pose_keeps_all_ten_segments_above_15_cm():
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)
    targets = pipeline._joint_patrol_targets(motion.get_state())
    route = pipeline._precheck_joint_patrol_route(motion.get_state())

    assert len(targets) == 10
    assert len(route.targets) == 11
    assert len(route.tcp_targets) == 11
    assert len(route.distances_m) == 11
    assert min(route.distances_m[:10]) >= config.task_a.patrol.min_move_distance_m
    assert motion.joint_moves == []


def test_all_ten_positions_and_return_to_start_are_prechecked_first():
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)

    route = pipeline._precheck_joint_patrol_route(motion.get_state())

    expected = tuple(
        tuple(math.degrees(value) for value in target)
        for target in pipeline._joint_patrol_targets(motion.get_state())
    )
    assert route.targets_deg[:10] == expected
    assert route.targets_deg[-1] == pytest.approx(HISTORICAL_STARTUP_DEG)


def test_arc_switch_between_points_5_and_6_keeps_j1_at_minus_135():
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)
    targets = pipeline._joint_patrol_targets(motion.get_state())

    assert math.degrees(targets[4][0]) == pytest.approx(-135.0)
    assert targets[5][0] == targets[4][0]
    assert math.degrees(targets[4][1] - HISTORICAL_STARTUP_RAD[1]) == pytest.approx(15.0)
    assert math.degrees(targets[5][1] - HISTORICAL_STARTUP_RAD[1]) == pytest.approx(-9.0)
    assert targets[4][2:] == targets[5][2:] == HISTORICAL_STARTUP_RAD[2:]


def test_automatic_patrol_dwells_three_seconds_then_returns_to_start():
    config = joint_patrol_config()
    visibility = replace(
        config.task_a.patrol.object_visibility,
        enabled=True,
    )
    config = replace(
        config,
        task_a=replace(
            config.task_a,
            patrol=replace(config.task_a.patrol, object_visibility=visibility),
        ),
    )
    pipeline, motion, _vision = make_pipeline(config)
    frames = class_frames()
    events = []

    original_precheck = motion.precheck_joint_route
    original_execute = motion.execute_joint_route_step

    def precheck(targets, **kwargs):
        events.append(("precheck", len(targets)))
        return original_precheck(targets, **kwargs)

    def execute(route, *, index, name):
        events.append(("execute", index + 1))
        return original_execute(route, index=index, name=name)

    def observe(position):
        events.append(("observe", position))
        return frames[position - 1], True

    def sleep(seconds):
        events.append(("sleep", seconds))

    motion.precheck_joint_route = precheck
    motion.execute_joint_route_step = execute
    pipeline._observe_target_at_position = observe
    pipeline._sleep = sleep
    pipeline._patrol_operator_gate = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("automatic patrol must not wait for operator input")
    )

    report = pipeline.run()

    expected = [("precheck", 11)]
    for position in range(1, 11):
        expected.append(("execute", position))
        expected.append(("observe", position))
        expected.append(("sleep", 3.0))
    expected.append(("execute", 11))
    assert events == expected
    assert report.frames == 10
    assert motion.get_state().joints == pytest.approx(HISTORICAL_STARTUP_RAD)


def test_route_locks_j5_j6_and_allows_j1_j4():
    config = joint_patrol_config()
    pipeline, motion, _vision = make_pipeline(config)

    route = pipeline._precheck_joint_patrol_route(motion.get_state())

    assert route.fixed_indices == (4, 5)
    for target in route.targets:
        assert target[4] == HISTORICAL_STARTUP_RAD[4]
        assert target[5] == HISTORICAL_STARTUP_RAD[5]


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

    assert events == [("precheck", 11), *(
        ("execute", index) for index in range(1, 12)
    )]
    assert report.frames == 10
    assert len(report.patrol_joint_targets_deg) == 10
    assert len(motion.joint_moves) == 11
    assert motion.get_state().joints == pytest.approx(HISTORICAL_STARTUP_RAD)


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
