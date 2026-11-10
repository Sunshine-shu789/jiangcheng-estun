from dataclasses import replace
from pathlib import Path

import pytest

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import (
    Detection,
    Frame,
    PickStatus,
    Pose,
    RobotState,
    TaskMode,
)
from track1_grasp.factory import build_pipeline
from track1_grasp.planner import PosePlanner


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = ROOT.parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_TASK_B_CONFIG = PROGRAM_ROOT / "config" / "track1_task_b_20261020.yaml"
TASK_B_LAUNCHER = PROGRAM_ROOT / "bin" / "run_task_b"


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


def target_detection(
    *,
    x=0.15,
    y=-0.15,
    label="8-Plastic_Banana",
    confidence=0.95,
    bbox=(200, 150, 320, 280),
):
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        depth_m=0.38,
        camera_xyz=(x, y, 0.38),
        base_xyz=(x, y, 0.035),
        graspability=1.0,
        instance_id="plastic_banana_1",
    )


def frame(frame_id, detections=()):
    return Frame(frame_id=frame_id, detections=tuple(detections))


def task_b_config(rounds=5, *, timeout_s=0.01):
    base = apply_mode(load_config(CONFIG), "B")
    return replace(
        base,
        task_b=replace(
            base.task_b,
            rounds=rounds,
            next_point_timeout_s=timeout_s,
            next_point_poll_s=0.001,
        ),
        search=replace(
            base.search,
            enabled=False,
            observe_frames=1,
            confirm_hits=1,
        ),
    )


def task_b_config_with_initial_pose(path, *, tolerance_deg=0.5):
    config = task_b_config()
    return replace(
        config,
        task_b=replace(
            config.task_b,
            initial_pose_file=str(path),
            initial_pose_tolerance_deg=tolerance_deg,
        ),
    )


def write_initial_pose(path, joints, *, role="task_b_initial_pose"):
    values = ", ".join(str(value) for value in joints)
    path.write_text(
        f"role: {role!r}\njoint_positions_rad: [{values}]\n",
        encoding="utf-8",
    )
    return path


def task_b_config_with_active_search():
    config = task_b_config(rounds=1)
    return replace(
        config,
        search=replace(
            config.search,
            enabled=True,
            observation_offsets_xy_m=((0.0, 0.0), (-0.10, 0.0)),
            observation_clearance_m=0.20,
            settle_s=0.0,
            observe_frames=1,
            confirm_hits=1,
            return_to_initial_on_miss=True,
            pitch_down_offsets_deg=(0.0,),
            yaw_offsets_deg=(0.0,),
        ),
    )


def test_task_b_uses_visual_xy_with_verified_grasp_z_and_rpy():
    base = task_b_config(rounds=1)
    config = replace(
        base,
        motion=replace(
            base.motion,
            orientation="current",
            grasp_z_m=0.161534092,
            grasp_rpy=(3.082206718, 0.098641098, -0.072880619),
        ),
    )
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.0, 0.2, -0.1)),
        message="ready",
    )
    detection = target_detection(x=0.12, y=-0.11)

    plan = planner.plan(detection, state, return_to_pick=True)

    assert plan.grasp_pose.xyz[:2] == pytest.approx((0.12, -0.11))
    assert plan.grasp_pose.xyz[2] == pytest.approx(0.161534092)
    assert plan.grasp_pose.rpy == pytest.approx(
        (3.082206718, 0.098641098, -0.072880619)
    )
    assert plan.pregrasp_pose.xyz[2] == pytest.approx(
        0.161534092 + config.motion.approach_height
    )
    assert plan.lift_pose.xyz[2] == pytest.approx(
        0.161534092 + config.motion.lift_height
    )
    assert plan.release_pose.xyz == plan.grasp_pose.xyz


def test_task_b_planner_accepts_per_point_grasp_override():
    base = task_b_config(rounds=1)
    config = replace(
        base,
        motion=replace(
            base.motion,
            orientation="current",
            grasp_z_m=0.15,
            grasp_rpy=(3.0, 0.0, 0.0),
        ),
    )
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.1, 0.2, -0.1)),
        message="ready",
    )

    plan = planner.plan(
        target_detection(x=0.12, y=-0.11),
        state,
        return_to_pick=True,
        grasp_z_m=0.168,
        grasp_rpy=(3.05, 0.08, -0.4),
    )

    assert plan.grasp_pose.xyz[2] == pytest.approx(0.168)
    assert plan.grasp_pose.rpy == pytest.approx((3.05, 0.08, -0.4))
    assert plan.release_pose.xyz == plan.grasp_pose.xyz
    assert plan.release_pose.rpy == plan.grasp_pose.rpy


def test_verified_grasp_baseline_does_not_change_non_return_plans():
    base = task_b_config(rounds=1)
    config = replace(
        base,
        motion=replace(
            base.motion,
            grasp_z_offset=0.01,
            orientation="current",
            grasp_z_m=0.161534092,
            grasp_rpy=(3.082206718, 0.098641098, -0.072880619),
        ),
    )
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.0, 0.2, -0.1)),
        message="ready",
    )
    detection = target_detection(x=0.12, y=-0.11)

    plan = planner.plan(
        detection,
        state,
        place_pose=(0.25, 0.25, 0.08, 3.14159265, 0.0, 0.0),
        return_to_pick=False,
    )

    assert plan.grasp_pose.xyz[2] == pytest.approx(0.035 + 0.01)
    assert plan.grasp_pose.rpy == pytest.approx(state.pose.rpy)


def test_task_b_moves_to_active_search_view_before_picking_when_initial_view_misses():
    config = task_b_config_with_active_search()
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("initial_miss"),
            frame("view_1_miss"),
            frame("view_2_hit", (target_detection(),)),
        ]
    )
    initial_state = pipeline.motion.get_state()
    expected_point = config.place_poses[0]

    target, _frame, _state, reason = pipeline._acquire_target(
        initial_state,
        lambda *_args, **_kwargs: None,
        expected_point=expected_point,
        expected_label=config.task_b.target_class,
        expected_name=config.task_b.place_names[0],
    )

    assert target is not None
    assert "confirmed" in reason
    assert pipeline.vision.index == 3
    assert [name for name, _pose in pipeline.motion.moves] == [
        "task_search_position_01",
        "task_search_position_02",
    ]
    assert not any(name == "pregrasp" for name, _pose in pipeline.motion.moves)
    for _name, pose in pipeline.motion.moves:
        assert pose.xyz[2] == expected_point[2] + config.search.observation_clearance_m


def test_task_b_active_search_uses_recorded_observation_orientation():
    config = task_b_config_with_active_search()
    config = replace(
        config,
        motion=replace(config.motion, orientation="current"),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("initial_miss"),
            frame("view_1_hit", (target_detection(),)),
        ]
    )
    initial_state = pipeline.motion.get_state()
    expected_point = (0.15, -0.15, 0.035, 3.05, 0.08, -0.12)

    target, _frame, _state, reason = pipeline._acquire_target(
        initial_state,
        lambda *_args, **_kwargs: None,
        expected_point=expected_point,
        expected_label=config.task_b.target_class,
        expected_name=config.task_b.place_names[0],
    )

    assert target is not None
    assert "confirmed" in reason
    assert len(pipeline.motion.moves) == 1
    name, pose = pipeline.motion.moves[0]
    assert name == "task_search_position_01"
    assert pose.rpy == pytest.approx(expected_point[3:6])


def test_task_b_active_search_returns_to_initial_pose_and_never_picks_on_a_miss():
    config = task_b_config_with_active_search()
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("initial_miss"),
            frame("view_1_miss"),
            frame("view_2_miss"),
        ]
    )
    initial_state = pipeline.motion.get_state()
    assert initial_state.pose is not None

    target, _frame, final_state, reason = pipeline._acquire_target(
        initial_state,
        lambda *_args, **_kwargs: None,
        expected_point=config.place_poses[0],
        expected_label=config.task_b.target_class,
        expected_name=config.task_b.place_names[0],
    )

    assert target is None
    assert "target not confirmed" in reason
    assert pipeline.vision.index == 3
    assert [name for name, _pose in pipeline.motion.moves] == [
        "task_search_position_01",
        "task_search_position_02",
        "task_search_return",
    ]
    assert pipeline.motion.moves[-1][1] == initial_state.pose
    assert final_state.pose == initial_state.pose
    assert not any(
        name in {"pregrasp", "grasp", "lift", "transfer", "release", "retreat"}
        for name, _pose in pipeline.motion.moves
    )


def test_task_b_completes_five_rounds_and_returns_each_object_in_place():
    config = task_b_config()
    assert config.task.mode is TaskMode.B
    pipeline = build_pipeline(config)
    report = pipeline.run()

    assert report.successes == 5
    assert report.requested_picks == 5
    assert [result.place_name for result in report.results] == [
        "pick_1",
        "pick_2",
        "pick_3",
        "pick_4",
        "pick_5",
    ]

    assert len(pipeline.motion.moves) == 30
    for index in range(5):
        moves = pipeline.motion.moves[index * 6 : (index + 1) * 6]
        assert [name for name, _pose in moves] == [
            "pregrasp",
            "grasp",
            "lift",
            "transfer",
            "release",
            "retreat",
        ]
        grasp = moves[1][1]
        release = moves[4][1]
        tool_clearance = moves[5][1]
        assert release.xyz == grasp.xyz
        assert tool_clearance.xyz[:2] == grasp.xyz[:2]
        assert tool_clearance.xyz[2] > grasp.xyz[2]


def test_task_b_uses_a_different_grasp_pose_for_each_expected_point():
    base = task_b_config(rounds=2)
    overrides_z = (0.151, 0.162, None, None, None)
    overrides_rpy = (
        (3.00, 0.01, -0.10),
        (3.10, 0.09, -0.45),
        None,
        None,
        None,
    )
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            point_grasp_z_m=overrides_z,
            point_grasp_rpy=overrides_rpy,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.15, y=-0.15),)),
            frame("pick_2", (target_detection(x=-0.15, y=-0.10),)),
        ]
    )

    first = pipeline.run_once()
    second = pipeline.run_once()

    assert first.success and second.success
    grasps = [pose for name, pose in pipeline.motion.moves if name == "grasp"]
    assert len(grasps) == 2
    assert grasps[0].xyz[2] == pytest.approx(overrides_z[0])
    assert grasps[0].rpy == pytest.approx(overrides_rpy[0])
    assert grasps[1].xyz[2] == pytest.approx(overrides_z[1])
    assert grasps[1].rpy == pytest.approx(overrides_rpy[1])


def test_task_b_returns_to_initial_pose_only_after_all_points_finish(tmp_path):
    target_q = (0.10, -0.04, 0.03, 0.02, -0.01, 0.05)
    pose_file = write_initial_pose(tmp_path / "initial_pose.yaml", target_q)
    base = task_b_config(rounds=5)
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            initial_pose_file=str(pose_file),
            return_to_initial_after_task=True,
        ),
    )
    pipeline = build_pipeline(config)
    sequence = []

    def record_joint_move(target, *, name, **kwargs):
        sequence.append(f"joint:{name}")

    original_execute = pipeline.motion.execute_prechecked_segment

    def record_segment(segment, *, name, speed_mps):
        original_execute(segment, name=name, speed_mps=speed_mps)
        sequence.append(f"segment:{name}")

    pipeline.motion.move_joints = record_joint_move
    pipeline.motion.execute_prechecked_segment = record_segment

    report = pipeline.run()

    assert report.successes == 5
    assert [event for event in sequence if event.startswith("joint:")] == [
        "joint:task_b_initial_pose",
        "joint:task_b_return_initial",
    ]
    assert sequence[-2:] == ["segment:retreat", "joint:task_b_return_initial"]


def test_task_b_failure_does_not_continue_or_return_to_initial_pose(tmp_path):
    target_q = (0.10, -0.04, 0.03, 0.02, -0.01, 0.05)
    pose_file = write_initial_pose(tmp_path / "initial_pose.yaml", target_q)
    base = task_b_config(rounds=2)
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            initial_pose_file=str(pose_file),
            return_to_initial_after_task=True,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [frame("pick_1", (target_detection(x=0.15, y=-0.15),))]
    )
    joint_moves = []

    def record_joint_move(target, *, name, **kwargs):
        joint_moves.append(name)

    pipeline.motion.move_joints = record_joint_move

    report = pipeline.run()

    assert report.successes == 1
    assert pipeline._task_b_pick_index == 1
    assert joint_moves == ["task_b_initial_pose"]
    assert not any(
        name == "task_b_return_initial" for name, _pose in pipeline.motion.moves
    )


def test_task_b_reference_points_are_not_fixed_drop_slots():
    config = task_b_config()
    pipeline = build_pipeline(config)
    report = pipeline.run()

    released_xy = [
        pose.xyz[:2] for name, pose in pipeline.motion.moves if name == "release"
    ]
    grasps_xy = [pose.xyz[:2] for name, pose in pipeline.motion.moves if name == "grasp"]
    expected_xy = [pose[:2] for pose in config.place_poses]
    assert released_xy == grasps_xy
    assert released_xy == expected_xy
    assert report.successes == 5


def test_task_b_lift_hold_happens_after_lift_and_before_return(monkeypatch):
    base = task_b_config(rounds=1)
    config = replace(
        base,
        task_b=replace(base.task_b, lift_hold_s=2.0),
    )
    pipeline = build_pipeline(config)
    sequence = []
    original_execute = pipeline.motion.execute_prechecked_segment

    def recording_execute(segment, *, name, speed_mps):
        original_execute(segment, name=name, speed_mps=speed_mps)
        sequence.append(name)

    def recording_sleep(seconds):
        sequence.append(f"hold:{seconds:.1f}s")

    pipeline.motion.execute_prechecked_segment = recording_execute
    monkeypatch.setattr("track1_grasp.pipeline.time.sleep", recording_sleep)

    result = pipeline.run_once()

    assert result.success
    assert sequence == [
        "pregrasp",
        "grasp",
        "lift",
        "hold:2.0s",
        "transfer",
        "release",
        "retreat",
    ]


def test_task_b_requires_reference_points():
    config = apply_mode(load_config(CONFIG), "B")
    broken = replace(config, task_b=replace(config.task_b, place_poses_file=""))
    try:
        broken.validate()
    except ValueError as exc:
        assert "task B needs" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("task B should require reference points")


def test_task_b_rejects_a_single_frame_false_positive_before_motion():
    config = task_b_config()
    config = replace(
        config,
        search=replace(
            config.search,
            enabled=False,
            observe_frames=3,
            confirm_hits=2,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("one_hit", (target_detection(),)),
            frame("miss_1"),
            frame("miss_2"),
        ]
    )

    result = pipeline.run_once()

    assert result.status == PickStatus.NO_TARGET
    assert pipeline.vision.index == 3
    assert pipeline.motion.moves == []
    assert pipeline._task_b_pick_index == 0


def test_task_b_accepts_only_after_consecutive_confirmation():
    config = task_b_config()
    config = replace(
        config,
        search=replace(
            config.search,
            enabled=False,
            observe_frames=4,
            confirm_hits=2,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("hit_1", (target_detection(),)),
            frame("hit_2", (target_detection(),)),
            frame("unused", (target_detection(),)),
        ]
    )

    result = pipeline.run_once()

    assert result.status == PickStatus.SUCCESS
    assert pipeline.vision.index == 2
    assert pipeline.motion.moves[0][0] == "pregrasp"


def test_old_point_remaining_visible_does_not_advance_to_the_next_round():
    config = task_b_config(rounds=2, timeout_s=0.02)
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.15, y=-0.15),)),
            *(
                frame(
                    f"old_point_{index}",
                    (target_detection(x=0.15, y=-0.15, confidence=0.99),),
                )
                for index in range(8)
            ),
        ]
    )

    first = pipeline.run_once()
    moves_after_first = len(pipeline.motion.moves)
    second = pipeline.run_once()

    assert first.status == PickStatus.SUCCESS
    assert pipeline._task_b_pick_index == 1
    assert second.status == PickStatus.NO_TARGET
    assert pipeline._task_b_pick_index == 1
    assert len(pipeline.motion.moves) == moves_after_first == 6


def test_previous_point_cannot_trigger_next_round_when_tolerances_overlap():
    base = apply_mode(load_config(CONFIG), "B")
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            place_poses=(
                (0.10, -0.15, 0.035, 3.14159265, 0.0, 0.0),
                (0.15, -0.15, 0.035, 3.14159265, 0.0, 0.0),
            ),
            place_poses_file="",
            rounds=2,
            place_names=("pick_1", "pick_2"),
            next_point_timeout_s=1.0,
            next_point_poll_s=0.001,
        ),
        search=replace(
            base.search,
            enabled=False,
            observe_frames=1,
            confirm_hits=1,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.10, y=-0.15),)),
            frame("old_point", (target_detection(x=0.10, y=-0.15, confidence=0.99),)),
            frame("pick_2", (target_detection(x=0.15, y=-0.15),)),
        ]
    )

    first = pipeline.run_once()
    second = pipeline.run_once()

    assert first.status == PickStatus.SUCCESS
    assert second.status == PickStatus.SUCCESS
    assert pipeline._task_b_pick_index == 2
    assert pipeline.motion.moves[6][0] == "pregrasp"
    assert pipeline.motion.moves[7][1].xyz[:2] == (0.15, -0.15)


def test_previous_point_inside_current_tolerance_never_starts_motion():
    base = apply_mode(load_config(CONFIG), "B")
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            place_poses=(
                (0.10, -0.15, 0.035, 3.14159265, 0.0, 0.0),
                (0.15, -0.15, 0.035, 3.14159265, 0.0, 0.0),
            ),
            place_poses_file="",
            rounds=2,
            place_names=("pick_1", "pick_2"),
            next_point_timeout_s=0.0001,
            next_point_poll_s=0.001,
        ),
        search=replace(
            base.search,
            enabled=False,
            observe_frames=1,
            confirm_hits=1,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.10, y=-0.15),)),
            frame("old_point_a", (target_detection(x=0.10, y=-0.15, confidence=0.99),)),
            frame("old_point_b", (target_detection(x=0.10, y=-0.15, confidence=0.99),)),
        ]
    )

    first = pipeline.run_once()
    moves_after_first = len(pipeline.motion.moves)
    second = pipeline.run_once()

    assert first.status == PickStatus.SUCCESS
    assert second.status in {PickStatus.NO_TARGET, PickStatus.ABORTED}
    assert pipeline._task_b_pick_index == 1
    assert len(pipeline.motion.moves) == moves_after_first == 6
    assert (
        "completed point" in second.message
        or "another reference point" in second.message
    )


def test_future_reference_inside_current_tolerance_never_starts_motion():
    base = apply_mode(load_config(CONFIG), "B")
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            place_poses=(
                (0.10, -0.15, 0.035, 3.14159265, 0.0, 0.0),
                (0.15, -0.15, 0.035, 3.14159265, 0.0, 0.0),
                (0.30, -0.15, 0.035, 3.14159265, 0.0, 0.0),
            ),
            place_poses_file="",
            rounds=3,
            place_names=("pick_1", "pick_2", "pick_3"),
            pick_pose_tolerance_m=0.08,
            point_assignment_margin_m=0.01,
            next_point_timeout_s=0.0001,
            next_point_poll_s=0.001,
        ),
        search=replace(
            base.search,
            enabled=False,
            observe_frames=1,
            confirm_hits=1,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame(
                f"future_point_visible_{index}",
                (target_detection(x=0.15, y=-0.15),),
            )
            for index in range(50)
        ]
    )

    result = pipeline.run_once()

    assert result.status == PickStatus.NO_TARGET
    assert pipeline._task_b_pick_index == 0
    assert pipeline.motion.moves == []
    assert "another reference point" in result.message


def test_actual_release_position_is_excluded_near_next_point():
    base = apply_mode(load_config(CONFIG), "B")
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            place_poses=(
                (0.10, -0.15, 0.035, 3.14159265, 0.0, 0.0),
                (0.20, -0.15, 0.035, 3.14159265, 0.0, 0.0),
            ),
            place_poses_file="",
            rounds=2,
            place_names=("pick_1", "pick_2"),
            pick_pose_tolerance_m=0.08,
            point_assignment_margin_m=0.01,
            next_point_timeout_s=0.01,
            next_point_poll_s=0.001,
        ),
        search=replace(
            base.search,
            enabled=False,
            observe_frames=1,
            confirm_hits=1,
        ),
    )
    pipeline = build_pipeline(config)
    # The first object is released at x=0.14.  In round two, the detection
    # at x=0.17 is clearly closer to pick_2 than to pick_1, but it is closer
    # to the actual released position than the assignment margin allows.
    # The completed-release check must therefore reject it.
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.14, y=-0.15),)),
            *(
                frame(
                    f"returned_pick_1_{index}",
                    (target_detection(x=0.17, y=-0.15, confidence=0.99),),
                )
                for index in range(50)
            ),
        ]
    )

    first = pipeline.run_once()
    moves_after_first = len(pipeline.motion.moves)
    second = pipeline.run_once()

    assert first.status == PickStatus.SUCCESS
    assert pipeline._task_b_pick_index == 1
    assert pipeline._task_b_completed_xy == [(0.14, -0.15)]
    assert second.status == PickStatus.NO_TARGET
    assert pipeline._task_b_pick_index == 1
    assert len(pipeline.motion.moves) == moves_after_first == 6
    assert "completed point" in second.message


def test_reference_points_must_be_distinguishable():
    base = apply_mode(load_config(CONFIG), "B")
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            place_poses=(
                (0.10, -0.15, 0.035, 3.14159265, 0.0, 0.0),
                (0.11, -0.15, 0.035, 3.14159265, 0.0, 0.0),
            ),
            place_poses_file="",
            rounds=2,
        ),
    )
    try:
        config.validate()
    except ValueError as exc:
        assert "point assignment needs" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("too-close reference points should be rejected")


def test_next_round_starts_only_after_second_point_is_visually_confirmed():
    config = task_b_config(rounds=2, timeout_s=1.0)
    config = replace(
        config,
        search=replace(
            config.search,
            enabled=False,
            observe_frames=2,
            confirm_hits=2,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1_a", (target_detection(x=0.15, y=-0.15),)),
            frame("pick_1_b", (target_detection(x=0.15, y=-0.15),)),
            frame("old_point_a", (target_detection(x=0.15, y=-0.15),)),
            frame("old_point_b", (target_detection(x=0.15, y=-0.15),)),
            frame("pick_2_a", (target_detection(x=-0.15, y=-0.10),)),
            frame("pick_2_b", (target_detection(x=-0.15, y=-0.10),)),
        ]
    )

    first = pipeline.run_once()
    first_moves = len(pipeline.motion.moves)
    second = pipeline.run_once()

    assert first.status == PickStatus.SUCCESS
    assert second.status == PickStatus.SUCCESS
    assert pipeline._task_b_pick_index == 2
    assert first_moves == 6
    assert len(pipeline.motion.moves) == 12
    assert pipeline.motion.moves[6][0] == "pregrasp"


def test_wrong_class_and_out_of_tolerance_target_are_filtered_before_motion():
    config = task_b_config(rounds=1, timeout_s=0.02)
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        frame(
            f"rejected_{index}",
            (
                (
                    target_detection(label="10-meat_can", confidence=0.99)
                    if index % 2 == 0
                    else target_detection(x=0.30, y=-0.15, confidence=0.98)
                ),
            ),
        )
        for index in range(50)
    )

    result = pipeline.run_once()

    assert result.status == PickStatus.NO_TARGET
    assert pipeline._task_b_pick_index == 0
    assert pipeline.motion.moves == []
    assert "expected point 'pick_1'" in result.message
    assert "8-Plastic_Banana" in result.message


class FailingVerifier:
    def verify(self, detection, plan):
        return False, "test hold verifier rejected"


def test_failed_grasp_does_not_advance_to_the_next_point():
    config = task_b_config(rounds=2)
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision([frame("pick_1", (target_detection(),))])
    pipeline.hold_verifier = FailingVerifier()

    result = pipeline.run_once()

    assert result.status == PickStatus.ABORTED
    assert pipeline._task_b_pick_index == 0
    assert pipeline.motion.moves[-1][0] == "recovery-retreat"
    assert not any(name == "release" for name, _pose in pipeline.motion.moves)


def test_release_failure_does_not_advance_to_the_next_point():
    config = task_b_config(rounds=2)
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision([frame("pick_1", (target_detection(),))])

    original_open = pipeline.gripper.open
    calls = 0

    def fail_on_release():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("test release command failed")
        original_open()

    pipeline.gripper.open = fail_on_release
    result = pipeline.run_once()

    assert result.status == PickStatus.ABORTED
    assert pipeline._task_b_pick_index == 0
    assert calls == 2


def test_retreat_failure_does_not_advance_to_the_next_point():
    config = task_b_config(rounds=2)
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision([frame("pick_1", (target_detection(),))])

    original_execute = pipeline.motion.execute_prechecked_segment

    def fail_on_retreat(segment, *, name, speed_mps):
        if name == "retreat":
            raise RuntimeError("test retreat failed")
        original_execute(segment, name=name, speed_mps=speed_mps)

    pipeline.motion.execute_prechecked_segment = fail_on_retreat
    result = pipeline.run_once()

    assert result.status == PickStatus.ABORTED
    assert pipeline._task_b_pick_index == 0
    assert pipeline.gripper.events[-1] == "open"


def test_task_b_run_request_is_capped_by_round_count():
    config = task_b_config(rounds=2)
    pipeline = build_pipeline(config)

    report = pipeline.run(max_picks=5)

    assert report.requested_picks == 2
    assert report.successes == 2
    assert pipeline._task_b_pick_index == 2


def test_task_b_skips_initial_pose_motion_when_already_inside_tolerance(tmp_path):
    pose_file = write_initial_pose(
        tmp_path / "initial_pose.yaml",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    pipeline = build_pipeline(task_b_config_with_initial_pose(pose_file))

    report = pipeline.run()

    assert report.successes == 5
    assert pipeline.motion.joint_moves == []


def test_task_b_moves_six_axes_once_before_first_visual_capture(tmp_path):
    target_q = (0.10, -0.04, 0.03, 0.02, -0.01, 0.05)
    pose_file = write_initial_pose(tmp_path / "initial_pose.yaml", target_q)
    pipeline = build_pipeline(task_b_config_with_initial_pose(pose_file))

    capture_joint_move_counts = []
    original_capture = pipeline.vision.capture

    def checked_capture():
        capture_joint_move_counts.append(len(pipeline.motion.joint_moves))
        return original_capture()

    pipeline.vision.capture = checked_capture
    report = pipeline.run()

    assert report.successes == 5
    assert capture_joint_move_counts
    assert all(count == 1 for count in capture_joint_move_counts)
    assert [name for name, _target in pipeline.motion.joint_moves] == [
        "task_b_initial_pose"
    ]
    assert pipeline.motion.joint_moves[0][1] == pytest.approx(target_q)


def test_task_b_missing_initial_pose_file_aborts_before_vision(tmp_path):
    pipeline = build_pipeline(
        task_b_config_with_initial_pose(tmp_path / "missing.yaml")
    )

    with pytest.raises(RuntimeError, match="initial pose is invalid"):
        pipeline.run()

    assert pipeline.vision._index == 0
    assert pipeline.motion.joint_moves == []


def test_task_b_invalid_initial_pose_role_aborts_before_vision(tmp_path):
    pose_file = write_initial_pose(
        tmp_path / "wrong_role.yaml",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        role="search_pose",
    )
    pipeline = build_pipeline(task_b_config_with_initial_pose(pose_file))

    with pytest.raises(RuntimeError, match="initial pose is invalid"):
        pipeline.run()

    assert pipeline.vision._index == 0
    assert pipeline.motion.joint_moves == []


def test_task_b_non_finite_initial_pose_aborts_before_vision(tmp_path):
    pose_file = write_initial_pose(
        tmp_path / "non_finite.yaml",
        (0.0, 0.0, float("nan"), 0.0, 0.0, 0.0),
    )
    pipeline = build_pipeline(task_b_config_with_initial_pose(pose_file))

    with pytest.raises(RuntimeError, match="initial pose is invalid"):
        pipeline.run()

    assert pipeline.vision._index == 0
    assert pipeline.motion.joint_moves == []


def test_task_b_non_ready_controller_aborts_before_vision(tmp_path):
    pose_file = write_initial_pose(
        tmp_path / "initial_pose.yaml",
        (0.10, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    pipeline = build_pipeline(task_b_config_with_initial_pose(pose_file))
    pipeline.motion.get_state = lambda: RobotState(
        ready=False,
        message="controller state=AutoRunning",
    )

    with pytest.raises(RuntimeError, match="requires AutoReady"):
        pipeline.run()

    assert pipeline.vision._index == 0
    assert pipeline.motion.joint_moves == []


def test_formal_20261020_config_encodes_complete_five_point_workflow():
    config = load_config(FORMAL_TASK_B_CONFIG)

    assert config.task.mode is TaskMode.B
    assert config.task_b.rounds == 5
    assert config.task_b.return_to_pick is True
    assert config.task_b.return_to_initial_after_task is True
    assert config.task_b.keep_same_target is True
    assert config.task_b.target_class == "8-Plastic_Banana"
    assert config.task_b.next_point_timeout_s == 0.0
    assert config.task_b.lift_hold_s == pytest.approx(2.0)
    assert config.task_b.place_poses_file.endswith("task_b_place_poses_20261020.yaml")
    assert config.task_b.initial_pose_file.endswith(
        "task_b_initial_pose_20261019_233355.yaml"
    )
    assert config.motion.backend == "robotmove"
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"
    assert config.motion.speed_mps == pytest.approx(0.02)
    assert config.motion.cartesian_step_m == pytest.approx(0.002)
    assert config.motion.lift_height == pytest.approx(0.08)
    assert config.pose.use_detection_base_xyz is False

    expected_xy = (
        (-0.041246342, 0.490700096),
        (-0.124883909, 0.521708714),
        (-0.129313605, 0.425258157),
        (0.053972672, 0.406601410),
        (0.034196350, 0.486712736),
    )
    for pose, expected in zip(config.place_poses, expected_xy):
        assert pose[:2] == pytest.approx(expected)
    assert config.task_b.point_grasp_z_m == pytest.approx(
        (0.161534092, 0.161534092, 0.167258106, 0.164424007, 0.161454634)
    )
    expected_rpy = (
        (3.029562510, 0.069559687, -0.068061620),
        (3.093678813, 0.128615562, 0.224744345),
        (3.021631068, -0.004533933, -0.792201360),
        (3.087441818, 0.120447929, -0.056104146),
        (3.026471987, 0.054535204, -0.656420178),
    )
    for actual, expected in zip(config.task_b.point_grasp_rpy, expected_rpy):
        assert actual == pytest.approx(expected)

    launcher_text = TASK_B_LAUNCHER.read_text(encoding="utf-8")
    assert "config/track1_task_b_20261020.yaml" in launcher_text
    assert "config/track1_task_b_live.yaml" not in launcher_text


def test_formal_20261020_config_completes_all_five_points_in_mock(monkeypatch):
    base = load_config(FORMAL_TASK_B_CONFIG)
    config = replace(
        base,
        vision=replace(
            base.vision,
            backend="mock",
            scenario_file=str(ROOT / "scenarios" / "mock_task_b_scene.yaml"),
            show=False,
            continuous=False,
        ),
        pose=replace(base.pose, use_detection_base_xyz=True),
        motion=replace(base.motion, backend="mock", dry_run=True),
        gripper=replace(base.gripper, backend="mock"),
    )
    detections = tuple(
        target_detection(x=float(pose[0]), y=float(pose[1]))
        for pose in config.place_poses
    )
    frame_sequence = tuple(
        frame(f"formal_five_point_{index:02d}", detections)
        for index in range(30)
    )

    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(frame_sequence)
    execution_speeds = []
    original_execute = pipeline.motion.execute_prechecked_segment

    def execute_and_simulate_cartesian_joint_change(segment, *, name, speed_mps):
        execution_speeds.append(float(speed_mps))
        original_execute(segment, name=name, speed_mps=speed_mps)
        # The mock backend records Cartesian poses but does not derive joint
        # feedback from them.  Mark the arm away from its startup joint posture
        # after every segment so the final six-axis return is exercised too.
        pipeline.motion._joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pipeline.motion.execute_prechecked_segment = (
        execute_and_simulate_cartesian_joint_change
    )
    monkeypatch.setattr("track1_grasp.pipeline.time.sleep", lambda _seconds: None)

    try:
        report = pipeline.run()
    finally:
        pipeline.close()

    assert report.requested_picks == 5
    assert report.successes == 5
    assert [result.place_name for result in report.results] == [
        "pick_1",
        "pick_2",
        "pick_3",
        "pick_4",
        "pick_5",
    ]
    assert [name for name, _target in pipeline.motion.joint_moves] == [
        "task_b_initial_pose",
        "task_b_return_initial",
    ]
    assert len(execution_speeds) == 30
    assert execution_speeds == pytest.approx([0.02] * 30)

    moves = pipeline.motion.moves
    assert len(moves) == 30
    for index, reference in enumerate(config.place_poses):
        cycle = moves[index * 6 : (index + 1) * 6]
        assert [name for name, _pose in cycle] == [
            "pregrasp",
            "grasp",
            "lift",
            "transfer",
            "release",
            "retreat",
        ]
        grasp_pose = cycle[1][1]
        release_pose = cycle[4][1]
        retreat_pose = cycle[5][1]
        assert grasp_pose.xyz[:2] == pytest.approx(reference[:2])
        assert grasp_pose.xyz[2] == pytest.approx(
            config.task_b.point_grasp_z_m[index]
        )
        assert grasp_pose.rpy == pytest.approx(config.task_b.point_grasp_rpy[index])
        assert release_pose.xyz == grasp_pose.xyz
        assert retreat_pose.xyz[:2] == grasp_pose.xyz[:2]
        assert retreat_pose.xyz[2] > grasp_pose.xyz[2]

