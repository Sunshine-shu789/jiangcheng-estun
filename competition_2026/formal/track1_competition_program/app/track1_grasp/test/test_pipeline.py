from dataclasses import replace
from pathlib import Path

import pytest

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import PickPhase, PickStatus, RobotState
from track1_grasp.factory import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"


def test_mock_c_run_completes_and_opens_before_approach():
    config = load_config(CONFIG)
    pipeline = build_pipeline(config)
    report = pipeline.run()
    assert report.successes == 4
    assert report.attempts == 4

    first = report.results[0]
    phases = [event.phase for event in first.events]
    assert phases.index(PickPhase.OPEN_GRIPPER) < phases.index(PickPhase.APPROACH)
    assert phases.index(PickPhase.GRASP) < phases.index(PickPhase.LIFT)

    names = [name for name, _pose in pipeline.motion.moves[:6]]
    assert names == ["pregrasp", "grasp", "lift", "transfer", "release", "retreat"]

    event_count = len(pipeline.gripper.events)
    pipeline.close()
    assert len(pipeline.gripper.events) == event_count


def test_pipeline_prechecks_complete_route_before_open_and_first_move():
    pipeline = build_pipeline(load_config(CONFIG))
    calls = []
    original_precheck = pipeline.motion.precheck_linear_targets

    def recording_precheck(targets, *, step_m):
        calls.append((tuple(targets), step_m))
        assert pipeline.gripper.events == []
        assert pipeline.motion.moves == []
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = recording_precheck
    pipeline.motion.move_to = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("pipeline re-planned a waypoint during execution")
    )

    result = pipeline.run_once()

    assert result.success
    assert len(calls) == 1
    prechecked_targets, step_m = calls[0]
    assert step_m == pytest.approx(pipeline.config.motion.cartesian_step_m)
    assert prechecked_targets == tuple(pose for _name, pose in pipeline.motion.moves)
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]


def test_pipeline_route_precheck_failure_keeps_gripper_and_arm_idle():
    pipeline = build_pipeline(load_config(CONFIG))

    def rejecting_precheck(_targets, *, step_m):
        raise RuntimeError("synthetic full-route rejection")

    pipeline.motion.precheck_linear_targets = rejecting_precheck

    result = pipeline.run_once()

    assert result.status == PickStatus.REJECTED
    assert "synthetic full-route rejection" in result.message
    assert pipeline.gripper.events == []
    assert pipeline.motion.moves == []


def test_live_pipeline_waits_for_fresh_robot_info_before_readiness_gate():
    base = load_config(CONFIG)
    config = replace(base, motion=replace(base.motion, dry_run=False))
    pipeline = build_pipeline(config)
    fresh = pipeline.motion.get_state()
    waited = []
    stale = RobotState(
        ready=False,
        pose=None,
        message="no fresh RobotInfo",
        joints=None,
    )
    pipeline.motion.get_state = lambda: stale
    pipeline.motion.wait_for_state = lambda timeout_s: (
        waited.append(timeout_s) or fresh
    )

    result = pipeline.run_once()

    assert result.success
    assert waited
    assert waited[0] == pytest.approx(
        max(1.0, config.motion.robot_info_timeout_s)
    )


def test_e_mode_follows_scripted_order():
    config = apply_mode(load_config(CONFIG), "E")
    pipeline = build_pipeline(config)
    report = pipeline.run()
    assert [result.target.label for result in report.results] == [
        "mug",
        "object",
        "bowl",
        "book",
    ]
    assert report.successes == 4


def test_workspace_rejection_never_moves():
    base = load_config(CONFIG)
    config = replace(
        base,
        motion=replace(
            base.motion,
            workspace_min=(-0.05, -0.05, 0.0),
            workspace_max=(0.05, 0.05, 0.60),
        ),
    )
    pipeline = build_pipeline(config)
    report = pipeline.run(max_picks=1)
    assert report.successes == 0
    assert report.results[0].status == PickStatus.REJECTED
    assert pipeline.motion.moves == []


class FailingVerifier:
    def verify(self, detection, plan):
        return False, "test hold verifier rejected"


def test_failed_hold_check_opens_and_retreats():
    pipeline = build_pipeline(load_config(CONFIG))
    pipeline.hold_verifier = FailingVerifier()
    result = pipeline.run_once()
    assert result.status == PickStatus.ABORTED
    assert pipeline.gripper.state.value == "open"
    assert pipeline.motion.moves[-1][0] == "recovery-retreat"
