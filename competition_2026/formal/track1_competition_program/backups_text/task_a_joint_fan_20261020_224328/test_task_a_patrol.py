from dataclasses import replace
import math
from pathlib import Path

import pytest

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import Detection, Frame, Pose, RobotState
from track1_grasp.factory import build_pipeline
from track1_grasp.motion import MockMotionBackend
from track1_grasp.vision import MockVisionSource
from track1_grasp.vision_pipeline import VisionDemoPipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"


def patrol_config():
    config = apply_mode(load_config(CONFIG), "A")
    return replace(
        config,
        task_a=replace(
            config.task_a,
            patrol=replace(config.task_a.patrol, enabled=True, settle_s=0.0),
        ),
    )


def direct_pipeline(config, motion):
    scenario = config.vision.scenario_files["A"]
    vision = MockVisionSource(replace(config.vision, scenario_file=scenario))
    return VisionDemoPipeline(config, vision=vision, motion=motion)


def segment_length(waypoints):
    return sum(
        math.dist(left.xyz, right.xyz)
        for left, right in zip(waypoints, waypoints[1:])
    )


def test_task_a_patrol_moves_ten_straight_segments_of_at_least_15_cm():
    config = patrol_config()
    pipeline = build_pipeline(config)

    report = pipeline.run()

    assert report.frames == 10
    assert len(report.patrol_targets) == 10
    assert len(report.patrol_distances_m) == 10
    assert all(distance >= 0.15 for distance in report.patrol_distances_m)
    assert len(pipeline.motion.linear_moves) == 10
    assert pipeline.motion.moves == []
    assert pipeline.vision._closed is False

    for (_name, waypoints), expected_distance in zip(
        pipeline.motion.linear_moves, report.patrol_distances_m
    ):
        assert abs(segment_length(waypoints) - expected_distance) < 1e-9
        assert all(
            math.dist(left.xyz, right.xyz) <= config.task_a.patrol.linear_step_m + 1e-9
            for left, right in zip(waypoints, waypoints[1:])
        )

    pipeline.close()
    assert pipeline.vision._closed is True


def test_all_patrol_targets_are_prechecked_before_linear_execution():
    config = patrol_config()
    pipeline = build_pipeline(config)
    events = []

    original_precheck = pipeline.motion.precheck_linear_targets
    original_execute = pipeline.motion.execute_linear_route

    def precheck(targets, *, step_m):
        events.append(("precheck", len(targets)))
        return original_precheck(targets, step_m=step_m)

    def execute(route, *, names, speed_mps):
        events.append(("execute", len(route.segments)))
        return original_execute(route, names=names, speed_mps=speed_mps)

    pipeline.motion.precheck_linear_targets = precheck
    pipeline.motion.execute_linear_route = execute
    pipeline.run()

    assert events[0] == ("precheck", 10)
    assert all(event[0] == "execute" for event in events[1:])
    assert len(events) == 11
    pipeline.close()


class FailingPrecheckMotion:
    def __init__(self):
        self.config = load_config(CONFIG).motion
        self.executions = 0
        self.closed = False

    @property
    def dry_run(self):
        return True

    def get_state(self):
        return RobotState(
            ready=True,
            pose=Pose((0.0, 0.0, 0.30), self.config.fixed_rpy),
            message="test-ready",
        )

    def precheck_linear_targets(self, targets, *, step_m):
        raise RuntimeError("planned route leaves the reachable workspace")

    def execute_linear_route(self, route, *, names, speed_mps):
        self.executions += 1

    def close(self):
        self.closed = True


def test_patrol_precheck_failure_happens_before_any_execution():
    config = patrol_config()
    motion = FailingPrecheckMotion()
    pipeline = direct_pipeline(config, motion)

    with pytest.raises(RuntimeError, match="failed pre-check"):
        pipeline.run()

    assert motion.executions == 0
    assert pipeline.vision._closed is False
    pipeline.close()
    assert motion.closed is True


def test_patrol_box_exercises_both_directions_on_all_three_axes():
    config = patrol_config()
    motion = MockMotionBackend(config.motion)
    pipeline = direct_pipeline(config, motion)

    report = pipeline.run()

    assert report.frames == 10
    waypoints = (motion.get_state().pose, *report.patrol_targets)
    directions = set()
    for previous, target in zip(waypoints, waypoints[1:]):
        delta = tuple(
            target.xyz[index] - previous.xyz[index] for index in range(3)
        )
        changed_axes = [index for index, value in enumerate(delta) if abs(value) > 1e-9]
        assert len(changed_axes) == 1
        axis = changed_axes[0]
        assert abs(delta[axis]) == pytest.approx(config.task_a.patrol.stride_m)
        directions.add((axis, 1 if delta[axis] > 0.0 else -1))

    assert directions == {
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (2, 1),
        (2, -1),
    }
    pipeline.close()


class WaitingMotion(MockMotionBackend):
    def __init__(self, config):
        super().__init__(config)
        self.wait_calls = []

    def wait_for_state(self, timeout_s):
        self.wait_calls.append(timeout_s)
        return self.get_state()


def test_patrol_waits_for_robot_info_before_route_planning():
    config = patrol_config()
    motion = WaitingMotion(config.motion)
    pipeline = direct_pipeline(config, motion)

    report = pipeline.run()

    assert motion.wait_calls == [10.0]
    assert report.frames == 10
    pipeline.close()


class MissingRobotInfoMotion(FailingPrecheckMotion):
    def __init__(self):
        super().__init__()
        self.wait_calls = []

    def wait_for_state(self, timeout_s):
        self.wait_calls.append(timeout_s)
        return RobotState(
            ready=False,
            pose=None,
            fault=False,
            message="no fresh RobotInfo",
        )


def test_patrol_reports_missing_robot_info_clearly_without_execution():
    config = patrol_config()
    motion = MissingRobotInfoMotion()
    pipeline = direct_pipeline(config, motion)

    with pytest.raises(RuntimeError, match="/RobotInfo"):
        pipeline.run()

    assert motion.wait_calls == [10.0]
    assert motion.executions == 0
    pipeline.close()


def visibility_config(*, observation_window=10.0):
    config = patrol_config()
    visibility = replace(
        config.task_a.patrol.object_visibility,
        enabled=True,
        visible_confirm_frames=3,
        observation_window_s=observation_window,
        poll_s=0.0,
    )
    return replace(
        config,
        vision=replace(config.vision, target_class="mug"),
        task_a=replace(
            config.task_a,
            patrol=replace(config.task_a.patrol, object_visibility=visibility),
        ),
    )


def visibility_frame(index, visible):
    detections = (
        (Detection("mug", 0.90, (50.0, 60.0, 170.0, 220.0)),)
        if visible
        else ()
    )
    return Frame(frame_id=f"visibility-{index:03d}", detections=detections)


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


class AdvancingScriptedVision(ScriptedVision):
    """Scripted camera whose capture calls advance a deterministic clock."""

    def __init__(self, frames, *, step_s=0.1):
        super().__init__(frames)
        self.step_s = step_s
        self.now = 0.0

    def capture(self):
        frame = super().capture()
        self.now += self.step_s
        return frame

    def clock(self):
        return self.now


class RepeatingVisibilityVision:
    def __init__(self, visible):
        self.visible = visible
        self.index = 0
        self.closed = False

    def capture(self):
        self.index += 1
        return visibility_frame(self.index, self.visible)

    def close(self):
        self.closed = True


class StepClock:
    def __init__(self, step=0.1):
        self.value = 0.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


def no_sleep(_seconds):
    return None


def test_visibility_gate_requires_three_frames_and_resets_after_reappearance():
    states = [False, True, True, True]
    vision = ScriptedVision([visibility_frame(i, state) for i, state in enumerate(states)])
    pipeline = VisionDemoPipeline(
        visibility_config(),
        vision=vision,
        motion=None,
        log=lambda _message: None,
        sleep=no_sleep,
    )

    frame, confirmed = pipeline._observe_target_at_position(1)

    assert confirmed is True
    assert frame.frame_id == "visibility-003"
    assert vision.index == 4


def test_empty_target_class_uses_the_configured_class_allow_list():
    config = visibility_config()
    config = replace(
        config,
        vision=replace(
            config.vision,
            target_class="",
            classes=("mug", "bowl"),
        ),
    )
    pipeline = VisionDemoPipeline(
        config,
        vision=ScriptedVision([]),
        motion=None,
        log=lambda _message: None,
    )

    assert pipeline._target_is_visible(
        Frame(
            frame_id="mug",
            detections=(
                Detection("mug", 0.90, (50.0, 60.0, 170.0, 220.0)),
            ),
        )
    ) is True
    assert pipeline._target_is_visible(
        Frame(
            frame_id="unrelated",
            detections=(
                Detection("book", 0.99, (50.0, 60.0, 170.0, 220.0)),
            ),
        )
    ) is False


def test_observation_window_advances_instead_of_raising_when_target_is_absent():
    vision = RepeatingVisibilityVision(visible=False)
    pipeline = VisionDemoPipeline(
        visibility_config(observation_window=0.3),
        vision=vision,
        motion=None,
        log=lambda _message: None,
        clock=StepClock(),
        sleep=no_sleep,
    )

    _frame, confirmed = pipeline._observe_target_at_position(1)

    assert confirmed is False
    assert vision.index == 3


def test_patrol_continues_after_an_unseen_position_and_completes_all_ten_moves():
    frames = [visibility_frame(i, True) for i in range(3)]
    frames += [visibility_frame(3 + i, False) for i in range(10)]
    frames += [visibility_frame(13 + i, True) for i in range(24)]

    config = visibility_config(observation_window=1.0)
    config = replace(
        config,
        vision=replace(
            config.vision,
            target_class="",
            classes=("mug", "bowl"),
        ),
    )
    vision = AdvancingScriptedVision(frames)
    motion = MockMotionBackend(config.motion)
    execute_calls = 0
    original_execute = motion.execute_linear_route

    def execute(route, *, names, speed_mps):
        nonlocal execute_calls
        execute_calls += 1
        return original_execute(route, names=names, speed_mps=speed_mps)

    motion.execute_linear_route = execute
    pipeline = VisionDemoPipeline(
        config,
        vision=vision,
        motion=motion,
        log=lambda _message: None,
        clock=vision.clock,
        sleep=no_sleep,
    )

    report = pipeline.run()

    assert report.frames == 10
    assert execute_calls == 10
    assert report.patrol_visible_positions == (1, 3, 4, 5, 6, 7, 8, 9, 10)
    assert report.patrol_detected_classes == ("mug",)
    assert report.patrol_missing_classes == ("bowl",)
    assert vision.index == len(frames)
    pipeline.close()
