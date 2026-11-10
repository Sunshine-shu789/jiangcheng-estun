from dataclasses import replace
from pathlib import Path

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import PickPhase, PickStatus
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
