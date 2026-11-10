from dataclasses import replace
from pathlib import Path

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import Detection, Frame, PickStatus, TaskMode
from track1_grasp.factory import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"


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
    label="6-Foam_Brick",
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
        instance_id="foam_brick_1",
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
    assert "completed point" in second.message


def test_actual_release_position_is_excluded_when_it_is_closer_to_next_point():
    base = apply_mode(load_config(CONFIG), "B")
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            place_poses=(
                (0.15, -0.15, 0.035, 3.14159265, 0.0, 0.0),
                (0.29, -0.15, 0.035, 3.14159265, 0.0, 0.0),
            ),
            place_poses_file="",
            rounds=2,
            place_names=("pick_1", "pick_2"),
            pick_pose_tolerance_m=0.08,
            point_assignment_margin_m=0.0,
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
    # The first object is observed 7.5 cm from its reference point.  The
    # returned object is closer to pick_2 (6.5 cm) than to the configured
    # pick_1 reference (7.5 cm), so only the actual released XY can reject it.
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.225, y=-0.15),)),
            *(
                frame(
                    f"returned_pick_1_{index}",
                    (target_detection(x=0.225, y=-0.15, confidence=0.99),),
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
    assert pipeline._task_b_completed_xy == [(0.225, -0.15)]
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
    assert "6-Foam_Brick" in result.message


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

    original_move_to = pipeline.motion.move_to

    def fail_on_retreat(pose, *, name, speed_mps):
        if name == "retreat":
            raise RuntimeError("test retreat failed")
        original_move_to(pose, name=name, speed_mps=speed_mps)

    pipeline.motion.move_to = fail_on_retreat
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
