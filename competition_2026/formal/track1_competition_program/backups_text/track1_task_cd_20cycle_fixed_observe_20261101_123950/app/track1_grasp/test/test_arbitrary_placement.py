from dataclasses import replace
from math import pi
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from track1_grasp.collision import GEOMETRY_SOURCE_FIELD_MEASURED
from track1_grasp.config import load_config
from track1_grasp.domain import Frame, PickStatus, TaskMode
from track1_grasp.factory import build_pick_pipeline
from track1_grasp.kinematics import rpy_to_matrix, solve_pose_ik_multistart
from track1_grasp.planner import AdaptiveGraspCandidate, PosePlanner, euler_to_matrix
from track1_grasp.tools.check_task_b_arbitrary_matrix import (
    DEFAULT_INITIAL_JOINTS_DEG,
    MatrixSample,
    _backend_for_state,
    _initial_state,
    _sample_detection,
)
from track1_grasp.pipeline import PickPipeline


ROOT = Path(__file__).resolve().parents[3]
ARBITRARY_CONFIG = ROOT / "config" / "track1_task_b_arbitrary_mock.yaml"
ARBITRARY_SINGLE_CONFIGS = (
    ROOT / "config" / "track1_task_b_arbitrary_single_trial.yaml",
    ROOT / "config" / "track1_task_b_arbitrary_single_preflight.yaml",
    ROOT / "config" / "track1_task_b_arbitrary_single_live.yaml",
)
ARBITRARY_LAUNCHER = ROOT / "bin" / "run_task_b_arbitrary"
ARBITRARY_DESKTOP = ROOT / "任务B_任意摆放抓取.desktop"
TARGET_LABEL = "8-Plastic_Banana"
TILTED_PITCH_OFFSETS_DEG = (0.0, -20.0, -16.0)


class ScriptedVision:
    def __init__(self, frames):
        self.frames = tuple(frames)
        self.index = 0

    def capture(self):
        if self.index >= len(self.frames):
            raise StopIteration("scripted vision exhausted")
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def close(self):
        return None


def arbitrary_config():
    config = load_config(ARBITRARY_CONFIG)
    config.validate()
    return config


def tilted_arbitrary_config():
    """Return a synthetic-random tilted visual configuration for offline tests."""

    config = arbitrary_config()
    candidates = replace(
        config.adaptive_inverse.candidates,
        pitch_offsets_deg=TILTED_PITCH_OFFSETS_DEG,
        max_count=(
            len(config.adaptive_inverse.candidates.yaw_offsets_deg)
            * len(TILTED_PITCH_OFFSETS_DEG)
        ),
    )
    config = replace(
        config,
        adaptive_inverse=replace(
            config.adaptive_inverse,
            tilted_visual_grasp=True,
            candidates=candidates,
        ),
    )
    config.validate()
    return config


def test_arbitrary_mock_config_contains_no_legacy_point_state():
    config = arbitrary_config()

    assert config.task.mode is TaskMode.B
    assert config.adaptive_inverse.enabled is True
    assert config.adaptive_inverse.arbitrary_placement is True
    assert config.adaptive_inverse.require_validated_extrinsics is True
    assert Path(config.pose.extrinsics_file).name == (
        "camera_to_flange.d405_calibrated_20261025.yaml"
    )
    extrinsics = yaml.safe_load(
        Path(config.pose.extrinsics_file).read_text(encoding="utf-8")
    )
    assert extrinsics["validated"] is True
    assert extrinsics["calibration_id"].strip()
    assert config.adaptive_inverse.candidates.use_visual_grasp_z is True
    assert config.adaptive_inverse.candidates.jaw_tip_clearance_m == pytest.approx(
        0.002
    )
    assert config.pose.gripper_tip_offset_flange_m == pytest.approx(
        (0.0, 0.0, 0.165)
    )
    assert config.task_b.place_poses == ()
    assert config.task_b.place_poses_file == ""
    assert config.task_b.place_names == ()
    assert config.task_b.point_grasp_z_m == ()
    assert config.task_b.point_grasp_xy_from_observation == ()
    assert config.task_b.point_grasp_xy_m == ()
    assert config.task_b.point_grasp_xy_offset_m == ()
    assert config.task_b.point_grasp_rpy == ()
    assert config.task_b.point_vertical_grasp_descent == ()
    assert config.task_b.point_vertical_grasp_approach_height_m == ()
    assert config.task_b.point_observation_joints_deg == ()
    assert Path(config.vision.scenario_file).is_file()


def test_arbitrary_candidate_coverage_matches_across_all_runtime_chains():
    for path in (ARBITRARY_CONFIG, *ARBITRARY_SINGLE_CONFIGS):
        # The three hardware-chain YAMLs intentionally fail closed until their
        # box geometry is field measured.  Check their candidate contract from
        # the raw documents rather than bypassing that safety gate in memory.
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        adaptive = payload["adaptive_inverse"]
        candidates = adaptive["candidates"]
        assert adaptive["arbitrary_placement"] is True
        assert candidates["max_count"] == 5, path.name
        assert candidates["pitch_offsets_deg"] == pytest.approx(
            (0.0,)
        ), path.name
        expected_approach_m = 0.08 if path == ARBITRARY_CONFIG else 0.05
        expected_tip_clearance_m = (
            0.002 if path == ARBITRARY_CONFIG else 0.025
        )
        assert candidates["approach_height_m"] == pytest.approx(
            expected_approach_m
        ), path.name
        assert candidates["jaw_tip_clearance_m"] == pytest.approx(
            expected_tip_clearance_m
        ), path.name
        assert adaptive["selection"]["stop_after_first_valid"] is True


def test_arbitrary_candidates_keep_tool_axis_vertical_down():
    config = arbitrary_config()
    q_initial, state = _initial_state(DEFAULT_INITIAL_JOINTS_DEG)
    planner = PosePlanner(config)
    sample = MatrixSample("vertical", 0.0, 0.49, -0.038478, 37.0, 0.0)
    candidates = planner.plan_adaptive_candidates(
        _sample_detection(planner, state, sample),
        state,
    )

    assert len(candidates) == 5
    for candidate in candidates:
        rotation = rpy_to_matrix(candidate.plan.grasp_pose.rpy)
        assert rotation @ np.array((0.0, 0.0, 1.0)) == pytest.approx(
            (0.0, 0.0, -1.0), abs=1e-7
        )
        assert candidate.pitch_offset_deg == pytest.approx(0.0)
        assert (
            candidate.plan.pregrasp_pose.xyz[2]
            - candidate.plan.grasp_pose.xyz[2]
        ) == pytest.approx(0.08)


def test_tilted_visual_grasp_is_an_explicit_opt_in():
    config = arbitrary_config()
    rejected = replace(
        config,
        adaptive_inverse=replace(
            config.adaptive_inverse,
            candidates=replace(
                config.adaptive_inverse.candidates,
                pitch_offsets_deg=TILTED_PITCH_OFFSETS_DEG,
            ),
        ),
    )

    with pytest.raises(ValueError, match="tilted_visual_grasp=true"):
        rejected.validate()

    tilted_arbitrary_config()


def test_tilted_candidates_preserve_the_visual_tip_and_tool_axis_geometry():
    config = tilted_arbitrary_config()
    _q_initial, state = _initial_state(DEFAULT_INITIAL_JOINTS_DEG)
    planner = PosePlanner(config)
    sample = MatrixSample("tilted", 0.01, 0.49, -0.038478, 37.0, 0.0)
    detection = _sample_detection(planner, state, sample)
    candidates = planner.plan_adaptive_candidates(
        detection,
        state,
        place_name="synthetic_random_target",
    )
    approach_height_m = float(
        config.adaptive_inverse.candidates.approach_height_m
    )
    clearance_m = float(
        config.adaptive_inverse.candidates.jaw_tip_clearance_m
    )
    tip_offset = np.asarray(
        config.pose.gripper_tip_offset_flange_m,
        dtype=float,
    )
    expected_tip = np.array(
        [
            sample.x_m,
            sample.y_m,
            sample.z_m + clearance_m,
        ],
        dtype=float,
    )

    assert any(
        abs(candidate.pitch_offset_deg) > 1e-9 for candidate in candidates
    )
    for candidate in candidates:
        plan = candidate.plan
        rotation = euler_to_matrix(plan.grasp_pose.rpy)
        tool_z = rotation[:, 2]
        commanded_tip = np.asarray(plan.grasp_pose.xyz) + rotation @ tip_offset
        expected_pregrasp = np.asarray(plan.grasp_pose.xyz) - (
            approach_height_m * tool_z
        )

        assert tool_z[2] < 0.0
        np.testing.assert_allclose(commanded_tip, expected_tip, atol=1e-9)
        np.testing.assert_allclose(
            plan.pregrasp_pose.xyz,
            expected_pregrasp,
            atol=1e-9,
        )
        assert plan.pregrasp_pose.rpy == pytest.approx(plan.grasp_pose.rpy)
        assert plan.lift_pose == plan.pregrasp_pose
        assert plan.transfer_pose == plan.pregrasp_pose
        assert plan.retreat_pose == plan.pregrasp_pose
        assert plan.release_pose == plan.grasp_pose


def test_tilted_plan_validator_rejects_tampered_high_or_release_pose():
    config = tilted_arbitrary_config()
    _q_initial, state = _initial_state(DEFAULT_INITIAL_JOINTS_DEG)
    planner = PosePlanner(config)
    sample = MatrixSample("validator", 0.01, 0.49, -0.038478, 37.0, 0.0)
    candidates = planner.plan_adaptive_candidates(
        _sample_detection(planner, state, sample),
        state,
        place_name="synthetic_random_target",
    )
    candidate = next(
        candidate for candidate in candidates if candidate.pitch_offset_deg < 0.0
    )
    pipeline = build_pick_pipeline(config)
    pipeline._validate_adaptive_candidate_plan(candidate)

    shifted_high = replace(
        candidate.plan.pregrasp_pose,
        xyz=(
            candidate.plan.pregrasp_pose.xyz[0] + 0.001,
            *candidate.plan.pregrasp_pose.xyz[1:],
        ),
    )
    bad_high = replace(
        candidate,
        plan=replace(candidate.plan, pregrasp_pose=shifted_high),
    )
    with pytest.raises(RuntimeError, match="pregrasp is not the configured"):
        pipeline._validate_adaptive_candidate_plan(bad_high)

    shifted_release = replace(
        candidate.plan.release_pose,
        xyz=(
            candidate.plan.release_pose.xyz[0] + 0.001,
            *candidate.plan.release_pose.xyz[1:],
        ),
    )
    bad_release = replace(
        candidate,
        plan=replace(candidate.plan, release_pose=shifted_release),
    )
    with pytest.raises(RuntimeError, match="release pose is not identical"):
        pipeline._validate_adaptive_candidate_plan(bad_release)


def test_tilted_candidate_prechecks_all_six_route_segments():
    config = tilted_arbitrary_config()
    _q_initial, state = _initial_state(DEFAULT_INITIAL_JOINTS_DEG)
    planner = PosePlanner(config)
    sample = MatrixSample("six_segments", 0.01, 0.49, -0.038478, 37.0, 0.0)
    candidates = planner.plan_adaptive_candidates(
        _sample_detection(planner, state, sample),
        state,
        place_name="synthetic_random_target",
    )
    candidate: AdaptiveGraspCandidate = next(
        candidate for candidate in candidates if candidate.pitch_offset_deg < 0.0
    )
    logs: list[str] = []
    backend = _backend_for_state(config, robot_state=state, logs=logs)
    pipeline = object.__new__(PickPipeline)
    pipeline.config = config
    pipeline.motion = backend
    pipeline._log = logs.append
    pipeline._tilted_visual_grasp = True
    pipeline._arbitrary_placement = True
    pipeline._box_relative_five_point = False

    selected_plan, route, failures = pipeline._precheck_adaptive_candidates(
        (candidate,),
        lambda _phase, message: logs.append(str(message)),
        place_name=sample.sample_id,
    )

    assert selected_plan is candidate.plan
    assert route is not None
    assert failures == []
    assert len(route.segments) == 6
    assert [segment.target for segment in route.segments] == [
        pose for _name, pose in candidate.plan.ordered_waypoints()
    ]


def test_arbitrary_object_heading_changes_the_commanded_j6_posture():
    config = arbitrary_config()
    config = replace(
        config,
        motion=replace(
            config.motion,
            dry_run=True,
            robot_info_timeout_s=3600.0,
        ),
        adaptive_inverse=replace(
            config.adaptive_inverse,
            candidates=replace(
                config.adaptive_inverse.candidates,
                axis_alignment_offset_deg=90.0,
            ),
        ),
    )
    q_initial, state = _initial_state(DEFAULT_INITIAL_JOINTS_DEG)
    planner = PosePlanner(config)
    solutions = []
    for yaw_deg in (0.0, 90.0):
        sample = MatrixSample(
            f"j6_{yaw_deg:g}", 0.0, 0.49, -0.038478, yaw_deg, 0.0
        )
        candidate = planner.plan_adaptive_candidates(
            _sample_detection(planner, state, sample),
            state,
        )[0]
        q_solution = solve_pose_ik_multistart(
            q_initial,
            candidate.plan.grasp_pose.xyz,
            rpy_to_matrix(candidate.plan.grasp_pose.rpy),
        )
        solutions.append(q_solution)

    j6_delta_deg = np.degrees(solutions[1][5] - solutions[0][5])
    assert abs(j6_delta_deg) > 60.0
    assert abs(np.degrees(solutions[0][5] - q_initial[5])) > 60.0


def test_dynamic_launcher_defaults_to_box_mode_and_keeps_arbitrary_override():
    launcher = ARBITRARY_LAUNCHER.read_text(encoding="utf-8")
    desktop = ARBITRARY_DESKTOP.read_text(encoding="utf-8")

    assert os.access(ARBITRARY_LAUNCHER, os.X_OK)
    assert "track1_task_b_box_relative_mock.yaml" in launcher
    assert "TRACK1_CONFIG:-" in launcher
    assert 'track1_task_runner" "B"' in launcher
    assert "run_task_b_arbitrary" in desktop
    assert "箱体动态五点" in desktop
    assert "离线试运行" in desktop

    five_point_launcher = (ROOT / "bin" / "run_task_b").read_text(
        encoding="utf-8"
    )
    assert "TRACK1_CONFIG:-" in five_point_launcher
    assert "track1_task_b_five_point_fixed_safe_20261027.yaml" in five_point_launcher
    assert "track1_task_b_five_point_fixed_live_20261027.yaml" not in five_point_launcher
    assert "track1_task_b_final_20261020_live.yaml" not in five_point_launcher
    assert "track1_task_b_arbitrary_mock.yaml" not in five_point_launcher


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("place_poses", ((0.12, 0.43, 0.03, pi, 0.0, 0.0),)),
        ("place_poses_file", "legacy_pick_points.yaml"),
        ("place_names", ("legacy_pick_1",)),
        ("point_grasp_z_m", (0.17,)),
        ("point_grasp_xy_from_observation", (True,)),
        ("point_grasp_xy_m", ((0.12, 0.43),)),
        ("point_grasp_xy_offset_m", ((0.0, 0.0),)),
        ("point_grasp_rpy", ((pi, 0.0, 0.0),)),
        ("point_vertical_grasp_descent", (False,)),
        ("point_vertical_grasp_approach_height_m", (0.08,)),
        ("point_observation_joints_deg", ((0.0,) * 6,)),
    ),
)
def test_arbitrary_mode_rejects_every_legacy_point_field(field, value):
    config = arbitrary_config()
    invalid = replace(
        config,
        task_b=replace(config.task_b, rounds=1, **{field: value}),
    )

    with pytest.raises(ValueError, match=field):
        invalid.validate()


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"enabled": False, "require_validated_extrinsics": False},
            "requires adaptive_inverse.enabled=true",
        ),
        (
            {"candidates": None},
            "use_visual_grasp_z=true",
        ),
    ),
)
def test_arbitrary_mode_rejects_unsafe_adaptive_switches(changes, message):
    config = arbitrary_config()
    adaptive_changes = dict(changes)
    if adaptive_changes.get("candidates") is None:
        adaptive_changes["candidates"] = replace(
            config.adaptive_inverse.candidates,
            use_visual_grasp_z=False,
        )
    invalid = replace(
        config,
        adaptive_inverse=replace(config.adaptive_inverse, **adaptive_changes),
    )

    with pytest.raises(ValueError, match=message):
        invalid.validate()


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"pitch_offsets_deg": (20.0,)},
            "pitch_offsets_deg to contain only 0.0",
        ),
        (
            {"grasp_rpy": (pi, 0.1, 0.0)},
            "tool Z axis vertical down",
        ),
    ),
)
def test_arbitrary_mode_rejects_non_vertical_grasp_requests(changes, message):
    config = arbitrary_config()
    if "pitch_offsets_deg" in changes:
        invalid = replace(
            config,
            adaptive_inverse=replace(
                config.adaptive_inverse,
                candidates=replace(
                    config.adaptive_inverse.candidates,
                    **changes,
                ),
            ),
        )
    else:
        invalid = replace(
            config,
            motion=replace(config.motion, **changes),
        )

    with pytest.raises(ValueError, match=message):
        invalid.validate()


def test_pipeline_rejects_non_vertical_arbitrary_grasp_plan():
    config = arbitrary_config()
    q_initial, state = _initial_state(DEFAULT_INITIAL_JOINTS_DEG)
    planner = PosePlanner(config)
    sample = MatrixSample("vertical", 0.0, 0.49, -0.038478, 37.0, 0.0)
    candidate = planner.plan_adaptive_candidates(
        _sample_detection(planner, state, sample),
        state,
    )[0]
    tilted_plan = replace(
        candidate.plan,
        grasp_pose=replace(
            candidate.plan.grasp_pose,
            rpy=(pi, 0.1, 0.0),
        ),
    )
    pipeline = build_pick_pipeline(config)

    with pytest.raises(RuntimeError, match="tool Z axis is not vertical down"):
        pipeline._validate_vertical_grasp_plan(
            tilted_plan,
            lambda *_args: None,
            "tilted_grasp",
            require_tool_down=True,
        )


def test_arbitrary_mode_is_restricted_to_task_b_and_return_to_pick():
    config = arbitrary_config()
    wrong_task = replace(config, task=replace(config.task, mode=TaskMode.C))
    with pytest.raises(ValueError, match="restricted to Task B"):
        wrong_task.validate()

    no_return = replace(
        config,
        task_b=replace(config.task_b, return_to_pick=False),
    )
    with pytest.raises(ValueError, match="return_to_pick=true"):
        no_return.validate()


def test_synthetic_box_geometry_is_restricted_to_the_offline_mock_chain():
    config = arbitrary_config()
    config = replace(
        config,
        motion=replace(config.motion, dry_run=False),
    )

    with pytest.raises(
        ValueError,
        match="synthetic geometry is restricted to the fully offline Mock pipeline",
    ):
        config.validate()


def test_arbitrary_mode_requires_validated_extrinsics_outside_offline_mock():
    config = arbitrary_config()
    field_geometry = replace(
        config.adaptive_inverse.box_clearance,
        geometry_source=GEOMETRY_SOURCE_FIELD_MEASURED,
    )
    not_dry_run = replace(
        config,
        motion=replace(config.motion, dry_run=False),
        adaptive_inverse=replace(
            config.adaptive_inverse,
            box_clearance=field_geometry,
            require_validated_extrinsics=False,
        ),
    )

    with pytest.raises(ValueError, match="require_validated_extrinsics=true"):
        not_dry_run.validate()


def test_five_arbitrary_rounds_follow_current_visual_pose_and_store_no_xy():
    pipeline = build_pick_pipeline(arbitrary_config())

    report = pipeline.run()

    assert report.requested_picks == 5
    assert report.attempts == 5
    assert report.successes == 5
    assert [result.place_name for result in report.results] == [
        f"arbitrary_round_{index}" for index in range(1, 6)
    ]
    assert all(result.success for result in report.results)
    assert all(result.target is not None for result in report.results)
    assert all(result.target.label == TARGET_LABEL for result in report.results)

    # The wrong high-confidence class in round 1 was filtered before policy
    # selection.  The next four instances are otherwise independent frames.
    assert [result.target.instance_id for result in report.results] == [
        f"arbitrary_banana_round_{index}" for index in range(1, 6)
    ]

    assert pipeline._task_b_pick_index == 5
    assert pipeline._task_b_completed_xy == []
    assert pipeline._task_b_target_label == TARGET_LABEL
    assert len(pipeline.motion.moves) == 30
    assert len(pipeline.gripper.events) == 11

    grasps = [pose for name, pose in pipeline.motion.moves if name == "grasp"]
    pregrasps = [
        pose for name, pose in pipeline.motion.moves if name == "pregrasp"
    ]
    assert len(grasps) == 5
    assert len({tuple(round(v, 6) for v in pose.xyz[:2]) for pose in grasps}) == 5
    assert len({tuple(round(v, 6) for v in pose.rpy) for pose in grasps}) == 5
    for pregrasp, grasp in zip(pregrasps, grasps):
        assert pregrasp.xyz[:2] == pytest.approx(grasp.xyz[:2])
        assert pregrasp.rpy == pytest.approx(grasp.rpy)
        assert pregrasp.xyz[2] - grasp.xyz[2] == pytest.approx(0.08)


def test_arbitrary_round_never_calls_legacy_point_acquisition_or_anchor(monkeypatch):
    pipeline = build_pick_pipeline(arbitrary_config())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy five-point helper was called")

    for name in (
        "_next_place",
        "_ensure_task_b_observation_pose",
        "_task_b_observation_anchor",
        "_acquire_task_b_target",
    ):
        monkeypatch.setattr(pipeline, name, forbidden)

    original = pipeline.planner.plan_adaptive_candidates
    planner_calls = []

    def record_planner_call(*args, **kwargs):
        planner_calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pipeline.planner,
        "plan_adaptive_candidates",
        record_planner_call,
    )

    result = pipeline.run_once()

    assert result.success
    assert result.place_name == "arbitrary_round_1"
    assert result.target is not None
    assert result.target.label == TARGET_LABEL
    assert planner_calls == [
        {
            "place_name": "arbitrary_round_1",
            "return_to_pick": True,
        }
    ]
    assert pipeline._task_b_completed_xy == []


def test_arbitrary_miss_with_search_disabled_issues_no_pose_or_gripper_command():
    config = arbitrary_config()
    config = replace(config, search=replace(config.search, enabled=False))
    pipeline = build_pick_pipeline(config)
    pipeline.vision = ScriptedVision([Frame("empty-current-frame", ())])

    result = pipeline.run_once()

    assert result.status is PickStatus.NO_TARGET
    assert pipeline.motion.moves == []
    assert pipeline.motion.joint_moves == []
    assert pipeline.gripper.events == []
    assert pipeline._task_b_completed_xy == []


def test_arbitrary_miss_adjusts_observation_then_returns_before_stopping():
    config = arbitrary_config()
    assert config.search.arbitrary_observation_offsets_xyz_m == (
        (0.0, 0.0, 0.020),
    )
    pipeline = build_pick_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            Frame("empty-initial-frame", ()),
            Frame("empty-adjusted-frame", ()),
        ]
    )
    initial_pose = pipeline.motion.get_state().pose
    assert initial_pose is not None

    result = pipeline.run_once()

    assert result.status is PickStatus.NO_TARGET
    assert [name for name, _pose in pipeline.motion.moves] == [
        "task_observation_adjust_01",
        "task_observation_return",
    ]
    adjusted_pose = pipeline.motion.moves[0][1]
    returned_pose = pipeline.motion.moves[1][1]
    assert adjusted_pose.xyz == pytest.approx(
        np.asarray(initial_pose.xyz) + np.asarray((0.0, 0.0, 0.020))
    )
    assert adjusted_pose.rpy == pytest.approx(initial_pose.rpy)
    assert returned_pose.xyz == pytest.approx(initial_pose.xyz)
    assert returned_pose.rpy == pytest.approx(initial_pose.rpy)
    assert pipeline.motion.joint_moves == []
    assert pipeline.gripper.events == []
    assert pipeline._task_b_completed_xy == []


def test_all_arbitrary_candidates_rejected_before_any_motion_or_gripper(monkeypatch):
    pipeline = build_pick_pipeline(arbitrary_config())

    def reject_route(_targets, *, step_m):
        raise RuntimeError("synthetic arbitrary candidate rejection")

    monkeypatch.setattr(
        pipeline.motion,
        "precheck_linear_targets",
        reject_route,
    )

    result = pipeline.run_once()

    assert result.status is PickStatus.REJECTED
    assert "all 5 adaptive grasp candidate" in result.message
    assert pipeline.motion.moves == []
    assert pipeline.motion.joint_moves == []
    assert pipeline.gripper.events == []
    assert pipeline._task_b_completed_xy == []
