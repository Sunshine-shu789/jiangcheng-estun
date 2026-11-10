from dataclasses import replace
from math import inf, nan, pi
from pathlib import Path

import pytest

from track1_grasp.config import (
    TaskCDClearConfig,
    apply_mode,
    load_config,
)
from track1_grasp.domain import Detection, Frame, PickStatus
from track1_grasp.factory import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_V11_CONFIG = FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml"
GRASP_RPY = (pi, 0.0, 0.0)
TIP_OFFSET_M = (0.0, 0.0, 0.165)
PLACE_POSE = (0.25, 0.25, 0.08, *GRASP_RPY)
OBSERVATION_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
OBSERVATION_RADIANS = (
    1.55319086466,
    0.08273905488,
    1.179547496455,
    -0.01242763614,
    1.55199244266,
    -0.00872451216,
)


class ScriptedVision:
    def __init__(self, frames, *, joint_moves=None):
        self.frames = tuple(frames)
        self.index = 0
        self._joint_moves = joint_moves
        self.observation_joint_move_counts = []

    def capture(self):
        if self._joint_moves is not None:
            self.observation_joint_move_counts.append(len(self._joint_moves))
        if self.index >= len(self.frames):
            raise StopIteration("scripted vision exhausted")
        frame = self.frames[self.index]
        self.index += 1
        return frame


def write_observation_pose(
    path: Path,
    *,
    role: str = "task_cd_observation",
    joints=OBSERVATION_JOINTS,
) -> Path:
    values = ", ".join(str(value) for value in joints)
    path.write_text(
        f"role: {role!r}\njoint_positions_rad: [{values}]\n",
        encoding="utf-8",
    )
    return path


def cd_config(
    observation_pose: Path,
    *,
    mode: str = "C",
    place_pose=PLACE_POSE,
    place_pose_frame: str | None = "jaw_tip",
    place_pose_verified: bool = True,
    observation_tolerance_deg: float = 0.5,
    search_enabled: bool = False,
    validate: bool = True,
):
    base = apply_mode(load_config(CONFIG), mode)
    safety = replace(
        base.motion.grasp_safety,
        enabled=True,
        require_flange_tip_offset=True,
        require_tool_down=True,
    )
    config = replace(
        base,
        task=replace(base.task, max_picks=1),
        task_cd=TaskCDClearConfig(
            enabled=True,
            observation_pose_file=str(observation_pose),
            observation_pose_tolerance_deg=observation_tolerance_deg,
            return_to_observation_after_each_pick=True,
            place_pose_verified=place_pose_verified,
            exclude_completed_tracks=True,
        ),
        vision=replace(
            base.vision,
            require_depth=True,
            stabilize=True,
            stabilizer_max_per_class=0,
        ),
        pose=replace(
            base.pose,
            robot_pose_frame="flange",
            gripper_tip_offset_flange_m=TIP_OFFSET_M,
        ),
        motion=replace(
            base.motion,
            orientation="fixed",
            fixed_rpy=GRASP_RPY,
            grasp_rpy=GRASP_RPY,
            place_pose=place_pose,
            place_pose_frame=place_pose_frame,
            grasp_safety=safety,
            workspace_min=(-0.65, -0.65, 0.0),
            workspace_max=(0.65, 0.65, 0.80),
            workspace_margin=0.0,
        ),
        search=replace(
            base.search,
            enabled=search_enabled,
            observe_frames=1,
            confirm_hits=1,
        ),
    )
    if validate:
        config.validate()
    return config


def detection(
    label: str,
    instance_id: str,
    *,
    x: float = 0.10,
    y: float = 0.0,
    graspability: float = 0.95,
    confidence: float = 0.95,
    bbox=(100.0, 100.0, 200.0, 200.0),
) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        depth_m=0.40,
        base_xyz=(x, y, 0.02),
        graspability=graspability,
        instance_id=instance_id,
    )


def frame(frame_id: str, *detections: Detection) -> Frame:
    return Frame(frame_id=frame_id, detections=tuple(detections))


def pipeline_with_frames(config, frames):
    pipeline = build_pipeline(config)
    vision = ScriptedVision(frames, joint_moves=pipeline.motion.joint_moves)
    pipeline.vision = vision
    return pipeline, vision


def test_task_cd_clear_config_parses_only_real_yaml_booleans(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    parsed = TaskCDClearConfig.from_dict(
        {
            "enabled": True,
            "observation_pose_file": observation_pose.name,
            "observation_pose_tolerance_deg": 0.4,
            "return_to_observation_after_each_pick": True,
            "place_pose_verified": True,
            "exclude_completed_tracks": True,
        },
        tmp_path,
    )

    assert parsed.enabled is True
    assert parsed.return_to_observation_after_each_pick is True
    assert parsed.place_pose_verified is True
    assert parsed.exclude_completed_tracks is True
    assert Path(parsed.observation_pose_file) == observation_pose


@pytest.mark.parametrize(
    "field_name",
    (
        "enabled",
        "return_to_observation_after_each_pick",
        "place_pose_verified",
        "exclude_completed_tracks",
    ),
)
@pytest.mark.parametrize("invalid_value", ("true", "false", 1, 0))
def test_task_cd_safety_flags_reject_non_boolean_values(
    tmp_path, field_name, invalid_value
):
    with pytest.raises(
        ValueError,
        match=rf"task_cd\.{field_name} must be a boolean",
    ):
        TaskCDClearConfig.from_dict({field_name: invalid_value}, tmp_path)


@pytest.mark.parametrize(
    "tolerance_deg",
    (0.0, -0.1, nan, inf),
)
def test_task_cd_observation_tolerance_must_be_positive_and_finite(
    tmp_path, tolerance_deg
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")

    with pytest.raises(
        ValueError,
        match="task_cd.observation_pose_tolerance_deg must be positive and finite",
    ):
        cd_config(
            observation_pose,
            observation_tolerance_deg=tolerance_deg,
        )


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        (
            "role: task_b_observation\njoint_positions_rad: [0, 0, 0, 0, 0, 0]\n",
            "joint pose role must",
        ),
        (
            "role: task_cd_observation\njoint_positions_rad: [0, 0, 0, 0, 0]\n",
            "exactly six values",
        ),
        (
            "role: task_cd_observation\n"
            "joint_positions_rad: [0, 0, .nan, 0, 0, 0]\n",
            "non-finite value",
        ),
        (
            "role: task_cd_observation\njoint_positions_rad: [10, 0, 0, 0, 0, 0]\n",
            "violates joint limits",
        ),
    ),
)
def test_task_cd_observation_pose_rejects_unsafe_files(
    tmp_path, contents, message
):
    observation_pose = tmp_path / "observation.yaml"
    observation_pose.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        cd_config(observation_pose)


@pytest.mark.parametrize(
    ("place_pose", "place_pose_frame", "verified", "message"),
    (
        (None, "jaw_tip", True, "requires an explicit motion.place_pose"),
        (PLACE_POSE, "flange", True, "requires motion.place_pose_frame=jaw_tip"),
        (
            PLACE_POSE,
            "jaw_tip",
            False,
            "requires task_cd.place_pose_verified=true",
        ),
        (
            (0.25, 0.25, nan, *GRASP_RPY),
            "jaw_tip",
            True,
            "place_pose with six finite values",
        ),
    ),
)
def test_task_cd_place_pose_fails_closed(
    tmp_path,
    place_pose,
    place_pose_frame,
    verified,
    message,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")

    with pytest.raises(ValueError, match=message):
        cd_config(
            observation_pose,
            place_pose=place_pose,
            place_pose_frame=place_pose_frame,
            place_pose_verified=verified,
        )


def test_task_cd_observation_joint_radians_are_not_converted_twice(tmp_path):
    observation_pose = write_observation_pose(
        tmp_path / "observation.yaml",
        joints=OBSERVATION_RADIANS,
    )
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("empty_bin")],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.NO_TARGET]
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]
    assert pipeline.motion.joint_moves[0][1] == pytest.approx(OBSERVATION_RADIANS)


def test_task_cd_clear_requires_fixed_observation_search_to_be_disabled(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")

    with pytest.raises(
        ValueError,
        match="task_cd.enabled requires search.enabled=false",
    ):
        cd_config(observation_pose, search_enabled=True)


def test_all_rejected_candidates_issue_no_cartesian_or_gripper_command(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame(
                "two_candidates",
                detection("mug", "mug_1", graspability=1.0),
                detection("bowl", "bowl_1", graspability=0.90),
            )
        ],
    )

    def reject_route(_targets, *, step_m):
        raise RuntimeError(f"synthetic route rejection at step {step_m}")

    pipeline.motion.precheck_linear_targets = reject_route
    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert report.successes == 0
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]
    rejection_messages = [
        event.message
        for result in report.results
        for event in result.events
        if "rejected before motion" in event.message
    ]
    for instance_id in ("mug_1", "bowl_1"):
        assert any(
            f"track={instance_id!r}" in message
            for message in rejection_messages
        )


def test_clear_loop_stops_after_one_successful_grasp_when_max_picks_is_one(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    mug = detection("mug", "mug_1", graspability=1.0)
    bowl = detection("bowl", "bowl_1", graspability=0.90)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("observation_1", mug, bowl),
            frame("observation_2", mug, bowl),
        ],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert report.successes == 1
    assert report.results[0].target.instance_id == "mug_1"
    assert pipeline._task_cd_completed_instance_ids == {"mug_1"}
    assert pipeline._task_cd_completed_pick_count == 1
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry",
        "task_cd_return_observation_after_track_1",
    ]
    assert vision.observation_joint_move_counts == [1]
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_close_travel_failure_blocks_lift_transfer_and_success(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        gripper=replace(
            config.gripper,
            require_close_travel_evidence=True,
        ),
    )
    mustard = replace(
        detection("mustard", "mustard_1", graspability=1.0),
        major_axis_length_m=0.100,
        minor_axis_length_m=0.05164,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("observation_1", mustard)],
    )

    def reject_short_close(width_m):
        raise RuntimeError(
            "MS42DC effective close travel 205.7 deg is below the "
            "CAD-derived minimum 349.3 deg for object width "
            f"{width_m:.6f} m; refusing to treat the jaws as closed"
        )

    pipeline.gripper.close_for_width = reject_short_close

    report = pipeline.run(max_picks=1)

    assert report.successes == 0
    assert [result.status for result in report.results] == [PickStatus.ABORTED]
    assert report.results[0].recoverable_failure is False
    assert "effective close travel 205.7 deg" in report.results[0].message
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
    ]
    assert pipeline.gripper.events == ["open"]
    assert pipeline._task_cd_completed_instance_ids == set()
    assert pipeline._task_cd_completed_pick_count == 0


def test_clear_loop_returns_to_observation_before_second_allowed_pick(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    mug = detection("mug", "mug_1", graspability=1.0)
    bowl = detection("bowl", "bowl_1", graspability=0.90)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("observation_1", mug, bowl),
            frame("observation_2", mug, bowl),
        ],
    )

    report = pipeline.run(max_picks=2)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.SUCCESS,
    ]
    assert report.successes == 2
    assert [
        result.target.instance_id
        for result in report.results
        if result.success
    ] == ["mug_1", "bowl_1"]
    assert pipeline._task_cd_completed_instance_ids == {"mug_1", "bowl_1"}
    assert pipeline._task_cd_completed_pick_count == 2
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry",
        "task_cd_return_observation_after_track_1",
        "task_cd_return_observation_after_track_2",
    ]
    assert vision.observation_joint_move_counts == [1, 2]
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    assert pipeline.gripper.events == [
        "open",
        "close",
        "open",
        "close",
        "open",
    ]


def test_safely_recovered_hold_failure_returns_and_continues_without_retry(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    failed = detection("mug", "failed_1", graspability=1.0)
    # The detector loses and re-creates the same physical item after the arm
    # returns to the observation pose.  Spatial deduplication must still stop it
    # from consuming the second allowed grasp attempt.
    failed_reidentified = detection("mug", "failed_2", graspability=1.0)
    next_item = detection("bowl", "next_1", graspability=0.90)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("first_observation", failed),
            frame(
                "observation_after_recovery",
                failed_reidentified,
                next_item,
            ),
        ],
    )

    class FailOnlyFirstTrack:
        rejected = False

        def verify(self, target, _plan):
            if target.instance_id == "failed_1":
                self.rejected = True
                return False, "synthetic empty close on failed_1"
            return True, "synthetic hold accepted"

    pipeline.hold_verifier = FailOnlyFirstTrack()

    report = pipeline.run(max_picks=2)

    assert [result.status for result in report.results] == [
        PickStatus.ABORTED,
        PickStatus.SUCCESS,
    ]
    assert report.results[0].recoverable_failure is True
    assert report.results[0].target.instance_id == "failed_1"
    assert report.results[1].target.instance_id == "next_1"
    assert pipeline._task_cd_failed_instance_ids == {"failed_1"}
    assert pipeline._task_cd_completed_instance_ids == {"next_1"}
    assert pipeline._task_cd_completed_pick_count == 1
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry",
        "task_cd_return_observation_after_failed_track_1",
        "task_cd_return_observation_after_track_1",
    ]
    assert vision.observation_joint_move_counts == [1, 2]
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "task_cd_recovery_retreat",
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    assert pipeline.gripper.events == [
        "open",
        "close",
        "open",
        "close",
        "open",
    ]
    assert any(
        "failed_tracks_excluded=1" in event.message
        for result in report.results
        for event in result.events
    ) or any(
        "failed track 'failed_1'" in event.message
        for result in report.results
        for event in result.events
    )
    assert any(
        "failed_spatial_excluded=1" in event.message
        for result in report.results
        for event in result.events
    )
    assert len(pipeline._task_cd_failed_spatial_fingerprints) == 1


def test_max_picks_one_stops_after_a_recovered_empty_close(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, vision = pipeline_with_frames(
        config,
        [frame("one_candidate", detection("mug", "mug_1"))],
    )

    class AlwaysFailHold:
        def verify(self, _target, _plan):
            return False, "synthetic empty close"

    pipeline.hold_verifier = AlwaysFailHold()

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.ABORTED]
    assert report.results[0].recoverable_failure is True
    assert pipeline._task_cd_failed_instance_ids == {"mug_1"}
    assert len(pipeline._task_cd_failed_spatial_fingerprints) == 1
    assert pipeline._task_cd_completed_instance_ids == set()
    assert vision.observation_joint_move_counts == [1]
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry",
        "task_cd_return_observation_after_failed_track_1",
    ]
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "task_cd_recovery_retreat",
    ]
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_completed_target_reidentified_with_new_track_id_is_excluded(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    first_observation = detection("mug", "mug_1", graspability=1.0)
    reidentified = detection("mug", "mug_99", graspability=1.0)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("first_observation", first_observation),
            frame("observation_after_return", reidentified),
        ],
    )

    report = pipeline.run(max_picks=2)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert report.successes == 1
    assert report.results[0].target.instance_id == "mug_1"
    assert pipeline._task_cd_completed_instance_ids == {"mug_1"}
    assert len(pipeline._task_cd_completed_spatial_fingerprints) == 1
    assert vision.observation_joint_move_counts == [1, 2]
    assert any(
        "completed_spatial_excluded=1" in event.message
        for result in report.results
        for event in result.events
    )
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_nearby_same_class_object_is_not_spatially_excluded(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    first_object = detection(
        "mug",
        "mug_1",
        x=0.10,
        graspability=1.0,
        bbox=(100.0, 100.0, 200.0, 200.0),
    )
    # Centre separation is 40 mm, beyond the 25 mm re-ID radius, and the boxes
    # do not overlap.  This is intentionally a different nearby instance.
    second_object = detection(
        "mug",
        "mug_2",
        x=0.14,
        graspability=0.90,
        bbox=(220.0, 100.0, 320.0, 200.0),
    )
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("first_observation", first_object),
            frame("second_observation", second_object),
        ],
    )

    report = pipeline.run(max_picks=2)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.SUCCESS,
    ]
    assert [result.target.instance_id for result in report.results] == [
        "mug_1",
        "mug_2",
    ]
    assert pipeline._task_cd_completed_instance_ids == {"mug_1", "mug_2"}
    assert len(pipeline._task_cd_completed_spatial_fingerprints) == 2
    assert vision.observation_joint_move_counts == [1, 2]


def test_hold_failure_stops_when_safe_recovery_precheck_fails(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("one_candidate", detection("mug", "mug_1"))],
    )

    class AlwaysFailHold:
        rejected = False

        def verify(self, _target, _plan):
            self.rejected = True
            return False, "synthetic empty close"

    pipeline.hold_verifier = AlwaysFailHold()
    original_precheck = pipeline.motion.precheck_linear_targets

    def reject_recovery_route(targets, *, step_m):
        if pipeline.hold_verifier.rejected and len(targets) == 1:
            raise RuntimeError("synthetic recovery route rejection")
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = reject_recovery_route

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.ABORTED]
    assert report.results[0].recoverable_failure is False
    assert "recovery retreat precheck failed" in report.results[0].message
    assert "synthetic recovery route rejection" in report.results[0].message
    assert pipeline._task_cd_failed_instance_ids == set()
    assert pipeline._task_cd_completed_instance_ids == set()
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
    ]
    assert pipeline.gripper.events == ["open", "close"]
    assert pipeline.gripper.state.value == "closed"
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]


def test_first_candidate_rejection_falls_through_to_second_candidate(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    unreachable = detection(
        "mug",
        "unreachable_1",
        x=0.90,
        graspability=1.0,
    )
    reachable = detection(
        "bowl",
        "reachable_1",
        x=0.10,
        graspability=0.90,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("first_choice_rejected", unreachable, reachable),
            frame("empty_after_second_choice"),
        ],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert report.results[0].target.instance_id == "reachable_1"
    assert pipeline._task_cd_completed_instance_ids == {"reachable_1"}
    assert any(
        "unreachable_1" in event.message and "rejected before motion" in event.message
        for event in report.results[0].events
    )
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]


def test_retreat_failure_never_marks_track_complete_or_commits_success(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("one_candidate", detection("mug", "mug_1"))],
    )
    original_execute = pipeline.motion.execute_prechecked_segment
    success_commits = []
    failure_commits = []

    def fail_on_retreat(segment, *, name, speed_mps):
        if name == "retreat":
            raise RuntimeError("synthetic retreat failure")
        return original_execute(segment, name=name, speed_mps=speed_mps)

    pipeline.motion.execute_prechecked_segment = fail_on_retreat
    pipeline.policy.commit_success = success_commits.append
    pipeline.policy.commit_failure = lambda: failure_commits.append(True)

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.ABORTED]
    assert "synthetic retreat failure" in report.results[0].message
    assert pipeline._task_cd_completed_instance_ids == set()
    assert pipeline._task_cd_completed_pick_count == 0
    assert success_commits == []
    assert failure_commits == [True]
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
    ]
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]


def test_empty_bin_finishes_normally_without_cartesian_or_gripper_command(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("empty_bin")],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.NO_TARGET]
    assert report.successes == 0
    assert "no unfinished graspable target" in report.results[0].message
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]


def test_pinned_target_excludes_off_target_before_candidate_or_precheck(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        vision=replace(config.vision, target_class="mug"),
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("off_target_only", detection("bowl", "bowl_1"))],
    )

    precheck_called = False

    def unexpected_precheck(*_args, **_kwargs):
        nonlocal precheck_called
        precheck_called = True
        raise AssertionError("off-target must not reach route precheck")

    def unexpected_policy_select(_detections):
        raise AssertionError("off-target must not reach target policy")

    pipeline.motion.precheck_linear_targets = unexpected_precheck
    pipeline.policy.select = unexpected_policy_select
    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.NO_TARGET]
    assert precheck_called is False
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]
    assert any(
        "off_target_excluded=1" in event.message
        for event in report.results[0].events
    )


def test_formal_v11_config_enables_authorized_live_clear_loop_without_max_grasp_z():
    config = load_config(FORMAL_V11_CONFIG)

    assert config.task.max_picks == 40
    assert config.motion.backend == "robotmove"
    assert config.motion.dry_run is False
    assert config.gripper.backend == "ms42dc"
    assert config.gripper.require_feedback is True
    assert config.gripper.hold_min_angle_deg == pytest.approx(1978.0)
    assert config.task_cd.enabled is True
    assert config.task_cd.return_to_observation_after_each_pick is True
    assert config.task_cd.place_pose_verified is True
    assert config.task_cd.exclude_completed_tracks is True
    assert config.search.enabled is False
    assert config.vision.stabilizer_max_per_class == 0
    assert config.motion.grasp_safety.maximum_tool_tilt_deg == 2.0
    assert config.motion.grasp_safety.maximum_place_tool_tilt_deg == 18.0
    assert config.motion.grasp_safety.maximum_grasp_flange_z_m is None
    assert config.motion.grasp_safety.maximum_grasp_tip_z_m is None
