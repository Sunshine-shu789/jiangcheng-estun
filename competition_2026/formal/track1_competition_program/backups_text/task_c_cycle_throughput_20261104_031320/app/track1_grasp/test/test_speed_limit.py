"""Safety-bound tests for operator-selectable arm speed profiles."""

from __future__ import annotations

import json
import math
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from track1_grasp import factory as factory_module
from track1_grasp.config import MotionConfig, apply_mode, load_config
from track1_grasp.domain import GripperState, Pose, RobotState
from track1_grasp.motion import (
    MAX_TRAJECTORY_TIMEOUT_S,
    LinearSegmentPlan,
    RobotMoveMotionBackend,
)
from track1_grasp.speed_limit import (
    DEFAULT_SPEED_SCALE,
    MIN_SPEED_SCALE,
    PROFILE_ORDER,
    PROFILE_SPECS,
    SpeedLimitStore,
    bounded_cartesian_speed_mps,
    bounded_joint_speed_rad_s,
    load_speed_capabilities,
    normalize_backend_speed_scale,
    normalize_speed_scale,
)


def test_speed_scale_accepts_only_the_operator_range():
    for value in (MIN_SPEED_SCALE, 0.25, 0.5, 1.0):
        assert normalize_speed_scale(value) == pytest.approx(value)

    for value in (
        0.0,
        -0.01,
        -1.0,
        0.099,
        1.0001,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        None,
        "0.5",
    ):
        with pytest.raises(ValueError):
            normalize_speed_scale(value)


def test_backend_scale_caps_above_100_percent_but_never_turns_into_zero():
    assert normalize_backend_speed_scale(2.0) == 1.0
    assert normalize_backend_speed_scale(100.0) == 1.0
    for value in (0.0, -1.0, 0.05, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            normalize_backend_speed_scale(value)


def test_missing_state_defaults_to_full_safe_configuration_speed(tmp_path):
    store = SpeedLimitStore(tmp_path / "missing-speed-state.json")

    assert store.scale_for("A") == DEFAULT_SPEED_SCALE
    assert store.scale_for("E") == DEFAULT_SPEED_SCALE
    assert not (tmp_path / "missing-speed-state.json").exists()


def test_missing_profile_uses_full_speed_while_preserving_other_profiles(tmp_path):
    path = tmp_path / "arm_speed_overrides.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {"A": {"scale": 0.25}},
            }
        ),
        encoding="utf-8",
    )
    store = SpeedLimitStore(path)

    assert store.scale_for("A") == pytest.approx(0.25)
    assert store.scale_for("B") == DEFAULT_SPEED_SCALE


def test_corrupt_state_keeps_last_valid_value_instead_of_becoming_zero(tmp_path):
    path = tmp_path / "arm_speed_overrides.json"
    store = SpeedLimitStore(path)
    store.save_scale("A", 0.4)
    path.write_text("{this is not valid json", encoding="utf-8")

    assert store.scale_for("A") == pytest.approx(0.4)


def test_zero_or_invalid_state_value_falls_back_to_full_speed(tmp_path):
    path = tmp_path / "arm_speed_overrides.json"
    for value in (0.0, -1.0, 0.05, float("nan"), float("inf")):
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {"A": {"scale": value}},
                },
                allow_nan=True,
            ),
            encoding="utf-8",
        )
        store = SpeedLimitStore(path)
        assert store.scale_for("A") == DEFAULT_SPEED_SCALE


def test_state_above_100_percent_is_clamped_to_configuration_cap(tmp_path):
    path = tmp_path / "arm_speed_overrides.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {"A": {"scale": 99.0}},
            }
        ),
        encoding="utf-8",
    )

    assert SpeedLimitStore(path).scale_for("A") == 1.0


def test_save_is_atomic_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "arm_speed_overrides.json"
    store = SpeedLimitStore(path)

    assert store.save_scale("A", MIN_SPEED_SCALE) == pytest.approx(
        MIN_SPEED_SCALE
    )
    assert store.save_scale("B", 0.5) == pytest.approx(0.5)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["profiles"]["A"]["scale"] == pytest.approx(0.1)
    assert payload["profiles"]["B"]["scale"] == pytest.approx(0.5)
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_invalid_save_does_not_replace_the_existing_safe_state(tmp_path):
    path = tmp_path / "arm_speed_overrides.json"
    store = SpeedLimitStore(path)
    store.save_scale("A", 0.4)
    before = path.read_bytes()

    with pytest.raises(ValueError):
        store.save_scale("A", 0.0)
    with pytest.raises(ValueError):
        store.save_scale("A", float("nan"))

    assert path.read_bytes() == before
    assert store.scale_for("A") == pytest.approx(0.4)


def test_formal_profiles_have_the_approved_speed_caps():
    # These values are the 100% limits.  They must not be raised by the UI or
    # by an operator-edited state file. Tasks B/C/D intentionally use the
    # faster main-motion profiles requested for the competition.
    expected = {
        "A": 0.02,
        "B": 0.12,
        "C": 0.18,
        "D": 0.09,
        "E": 0.01,
    }
    expected_joint = {
        "A": (0.6, 0.6),
        "B": (0.6, 0.7),
        "C": (2.4, 2.4),
        "D": (1.8, 1.8),
        "E": (0.6, 0.6),
    }

    assert set(PROFILE_ORDER) == set(expected)
    assert set(PROFILE_ORDER) == set(expected_joint)
    for key in PROFILE_ORDER:
        capabilities = load_speed_capabilities(key)
        assert capabilities.config_path.samefile(PROFILE_SPECS[key].config_path)
        assert capabilities.config_path.exists()
        assert capabilities.speed_mps == pytest.approx(expected[key])
        joint_speed, joint_limit = expected_joint[key]
        assert capabilities.joint_speed_rad_s == pytest.approx(joint_speed)
        assert capabilities.joint_speed_safety_limit_rad_s == pytest.approx(
            joint_limit
        )
        assert capabilities.safe_joint_speed_rad_s == pytest.approx(
            min(joint_speed, joint_limit)
        )
        assert capabilities.safe_joint_speed_rad_s <= (
            capabilities.joint_speed_safety_limit_rad_s
        )

    # Task B must remain non-actuating until its field checks are complete.
    task_b = load_speed_capabilities("B")
    assert task_b.dry_run is True
    assert task_b.gripper_backend == "mock"


def test_factory_passes_each_runtime_task_profile_to_motion(monkeypatch):
    seen_profiles: list[str | None] = []

    monkeypatch.setattr(
        factory_module,
        "build_vision",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )

    def fake_build_motion(_config, *, node=None, log=None, profile=None):
        del node, log
        seen_profiles.append(profile)
        return SimpleNamespace(dry_run=True)

    monkeypatch.setattr(factory_module, "build_motion", fake_build_motion)
    monkeypatch.setattr(
        factory_module,
        "build_gripper",
        lambda *_args, **_kwargs: SimpleNamespace(
            state=GripperState.OPEN,
            events=[],
        ),
    )
    monkeypatch.setattr(
        factory_module,
        "build_hold_verifier",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    for key in PROFILE_ORDER:
        config = apply_mode(load_config(PROFILE_SPECS[key].config_path), key)
        factory_module.build_pipeline(config)

    assert seen_profiles == list(PROFILE_ORDER)


def test_cartesian_speed_is_capped_by_configuration_and_scaled_once():
    assert bounded_cartesian_speed_mps(0.03, 0.02, 0.5) == pytest.approx(0.01)
    assert bounded_cartesian_speed_mps(0.03, 0.02, 1.0) == pytest.approx(0.02)
    assert bounded_cartesian_speed_mps(0.01, 0.03, 0.5) == pytest.approx(0.005)
    # A bad backend value is clamped, not allowed to exceed 100%.
    assert bounded_cartesian_speed_mps(0.03, 0.02, 50.0) == pytest.approx(0.02)


def test_joint_speed_is_capped_by_both_configuration_and_safety_limit():
    assert bounded_joint_speed_rad_s(0.6, 0.5, 0.5) == pytest.approx(0.25)
    assert bounded_joint_speed_rad_s(0.6, 0.6, 1.0) == pytest.approx(0.6)
    assert bounded_joint_speed_rad_s(0.6, 0.6, 50.0) == pytest.approx(0.6)


class _SequenceScaleStore:
    def __init__(self, values):
        self._values = iter(values)

    def scale_for(self, profile: str) -> float:
        assert profile == "B"
        return next(self._values)


def test_each_prechecked_segment_reads_a_fresh_scale_and_does_not_double_apply_it(
    monkeypatch,
):
    start = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    first_target = Pose((0.12, 0.0, 0.0), (0.0, 0.0, 0.0))
    second_target = Pose((0.24, 0.0, 0.0), (0.0, 0.0, 0.0))
    first = LinearSegmentPlan(
        target=first_target,
        waypoints=(start, first_target),
        joint_waypoints=((0.0,) * 6, (0.24, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    second = LinearSegmentPlan(
        target=second_target,
        waypoints=(first_target, second_target),
        joint_waypoints=(
            (0.24, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.48, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    config = MotionConfig(
        backend="robotmove",
        dry_run=False,
        speed_mps=0.30,
        joint_speed_rad_s=0.6,
        joint_speed_safety_limit_rad_s=0.6,
        # This synthetic segment intentionally has a larger joint increment
        # than a production 2 mm refinement step, so keep its execution guard
        # consistent with the test's 0.24 rad waypoint.
        cartesian_joint_step_deg=30.0,
    )
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config
    backend.profile = "B"
    backend._speed_store = _SequenceScaleStore((0.5, 1.0))
    backend._log = lambda _message: None
    current = {
        "state": RobotState(
            ready=True,
            pose=start,
            joints=(0.0,) * 6,
            message="AutoReady",
        )
    }
    sent_velocities = []
    sent_poses = iter((first_target, second_target))

    monkeypatch.setattr(backend, "wait_for_state", lambda _timeout: current["state"])

    def send(targets, *, name, velocity_rad_s, timeout_s, reject_timeout_s):
        del name, timeout_s, reject_timeout_s
        sent_velocities.append(float(velocity_rad_s))
        endpoint = np.asarray(targets, dtype=float)[-1]
        current["state"] = RobotState(
            ready=True,
            pose=next(sent_poses),
            joints=tuple(float(value) for value in endpoint),
            message="AutoReady",
        )
        return endpoint

    monkeypatch.setattr(backend, "_send_joint_trajectory", send)

    backend.execute_prechecked_segment(first, name="first", speed_mps=0.12)
    backend.execute_prechecked_segment(second, name="second", speed_mps=0.12)

    # At 50%, the same 0.12 m segment takes twice as long and therefore uses
    # half the derived joint rate.  The second segment reads 100% afresh.
    assert sent_velocities == pytest.approx([0.12, 0.24])


def test_backend_rejects_a_broken_store_that_tries_to_zero_speed():
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = MotionConfig(backend="robotmove")
    backend.profile = "B"
    backend._speed_store = _SequenceScaleStore((0.0,))
    backend._log = lambda _message: None

    # The consumer boundary keeps the historical 100% cap when a malformed
    # store returns zero.  It never publishes a zero-velocity trajectory.
    assert backend._speed_scale() == 1.0


def test_backend_without_speed_members_retains_legacy_full_speed_compatibility():
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = MotionConfig(backend="robotmove")

    assert backend._speed_scale() == 1.0


def test_low_speed_trajectory_gets_more_time_but_has_a_hard_upper_bound():
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = MotionConfig(
        backend="robotmove",
        joint_move_timeout_s=8.0,
        move_timeout_s=30.0,
    )

    full_speed_timeout = backend._trajectory_timeout_s(
        10,
        joint_travel_rad=1.0,
        velocity_rad_s=0.6,
    )
    ten_percent_timeout = backend._trajectory_timeout_s(
        10,
        joint_travel_rad=1.0,
        velocity_rad_s=0.06,
    )
    extreme_timeout = backend._trajectory_timeout_s(
        10,
        joint_travel_rad=1.0e9,
        velocity_rad_s=0.01,
    )

    assert ten_percent_timeout > full_speed_timeout
    assert extreme_timeout == MAX_TRAJECTORY_TIMEOUT_S
    assert MAX_TRAJECTORY_TIMEOUT_S > 0.0
