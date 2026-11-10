from dataclasses import replace
from math import inf, nan, pi
from pathlib import Path

import pytest

from track1_grasp.config import (
    GripperSelfFilterConfig,
    TaskCDClearConfig,
    TaskCDEmptyConfirmationConfig,
    TaskCDRelocationScanConfig,
    apply_mode,
    load_config,
)
from track1_grasp.domain import Detection, Frame, PickResult, PickStatus, TaskPhase
from track1_grasp.factory import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_V11_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_c_v11_official_20261102.yaml"
)
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


SCAN_MOVE_PREFIXES = (
    "task_cd_relocation_scan_",
    "task_cd_relocation_return_",
)


def scan_moves(pipeline):
    """Return only relocation-scan outbound/return Cartesian moves."""

    return [
        (name, pose)
        for name, pose in pipeline.motion.moves
        if name.startswith(SCAN_MOVE_PREFIXES)
    ]


def non_scan_moves(pipeline):
    """Return commanded Cartesian moves excluding relocation-scan travel."""

    return [
        (name, pose)
        for name, pose in pipeline.motion.moves
        if not name.startswith(SCAN_MOVE_PREFIXES)
    ]


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
    relocation_offsets=((0.02, 0.0, 0.0),),
    complete_box_before_planning: bool = False,
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
            gripper_self_filter=GripperSelfFilterConfig(enabled=True),
            relocation_scan=TaskCDRelocationScanConfig(
                enabled=True,
                offsets_xyz_m=relocation_offsets,
                settle_s=0.0,
                complete_box_before_planning=complete_box_before_planning,
            ),
            empty_confirmation=TaskCDEmptyConfirmationConfig(
                enabled=True,
                require_complete_full_scan=True,
                confirmation_rounds=2,
                require_all_views_successful=True,
                reject_if_any_item_present=True,
            ),
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


def empty_full_scan_frames(prefix: str, rounds: int = 2):
    """Build one fixed and one relocation empty frame per complete round."""

    frames = []
    for round_index in range(1, rounds + 1):
        frames.extend(
            (
                frame(f"{prefix}_round_{round_index}_fixed"),
                frame(f"{prefix}_round_{round_index}_relocation"),
            )
        )
    return frames


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


def test_complete_box_before_planning_defaults_true_and_rejects_non_boolean():
    parsed = TaskCDRelocationScanConfig.from_dict(
        {"enabled": True, "offsets_xyz_m": [[0.02, 0.0, 0.0]]}
    )
    assert parsed.complete_box_before_planning is True

    with pytest.raises(
        ValueError,
        match=(
            "task_cd.relocation_scan.complete_box_before_planning must be "
            "a boolean"
        ),
    ):
        TaskCDRelocationScanConfig.from_dict(
            {
                "enabled": True,
                "offsets_xyz_m": [[0.02, 0.0, 0.0]],
                "complete_box_before_planning": "true",
            }
        )


def test_formal_task_c_enables_complete_box_before_planning():
    config = load_config(FORMAL_V11_CONFIG)
    assert config.task_cd.relocation_scan.complete_box_before_planning is True


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
        empty_full_scan_frames("empty"),
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


def test_rejected_candidates_never_count_as_empty_and_issue_no_commands(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame(
                "two_candidates",
                detection("mug", "mug_1", graspability=1.0),
                detection("bowl", "bowl_1", graspability=0.90),
            ),
            frame("fixed_view_after_rejection"),
            frame("relocation_view_after_rejection"),
            *empty_full_scan_frames("empty_after_rejection"),
        ],
    )

    original_precheck = pipeline.motion.precheck_linear_targets
    rejected_candidates = 0

    def reject_route(targets, *, step_m):
        nonlocal rejected_candidates
        if len(targets) == 6 and rejected_candidates < 2:
            rejected_candidates += 1
            raise RuntimeError(f"synthetic route rejection at step {step_m}")
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = reject_route
    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.REJECTED,
        PickStatus.NO_TARGET,
    ]
    assert report.requested_picks == 0
    assert report.successes == 0
    assert non_scan_moves(pipeline) == []
    assert [name for name, _pose in scan_moves(pipeline)] == [
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
    ]
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]
    assert vision.observation_joint_move_counts == [1] * 5
    rejection_messages = [
        event.message
        for event in report.results[0].events
        if "rejected before motion" in event.message
    ]
    for instance_id in ("mug_1", "bowl_1"):
        assert any(
            f"track={instance_id!r}" in message
            for message in rejection_messages
        )


def test_complete_box_scan_collects_every_view_before_candidate_planning(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    offsets = (
        (0.02, 0.0, 0.0),
        (0.0, 0.02, 0.0),
        (0.0, 0.0, 0.02),
    )
    config = cd_config(
        observation_pose,
        relocation_offsets=offsets,
        complete_box_before_planning=True,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame(
                "fixed_candidate",
                detection("mug", "fixed_1", graspability=1.0),
            ),
            frame(
                "relocation_candidate_1",
                detection("mug", "view_1", graspability=0.95),
            ),
            frame(
                "relocation_candidate_2",
                detection("mug", "view_2", graspability=0.90),
            ),
            frame(
                "relocation_candidate_3",
                detection("mug", "view_3", graspability=0.85),
            ),
        ],
    )

    original_precheck = pipeline._precheck_task_cd_candidate
    scan_moves_at_first_plan = []

    def record_precheck(target, capture_state, route_state, emit):
        scan_moves_at_first_plan.append(len(scan_moves(pipeline)))
        return original_precheck(target, capture_state, route_state, emit)

    pipeline._precheck_task_cd_candidate = record_precheck
    report = pipeline.run(acceptance_max_successes=1)

    assert report.successes == 1
    assert scan_moves_at_first_plan == [2 * len(offsets)]
    assert [name for name, _pose in scan_moves(pipeline)] == [
        name
        for index in range(1, len(offsets) + 1)
        for name in (
            f"task_cd_relocation_scan_{index:02d}",
            f"task_cd_relocation_return_{index:02d}",
        )
    ]
    select_messages = [
        event.message
        for event in report.results[0].events
        if event.phase is TaskPhase.SELECT
    ]
    assert any(
        "4 confirmed unfinished candidate(s)" in message
        for message in select_messages
    )
    assert any(
        "complete box observation finished" in event.message
        and "4 unique confirmed candidate(s)" in event.message
        for event in report.results[0].events
    )


def test_temporarily_rejected_candidate_is_retried_and_can_be_picked(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    target = detection("mug", "mug_1", graspability=1.0)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("candidate_before_rejection", target),
            frame("fixed_view_after_rejection"),
            frame("relocation_view_after_rejection"),
            frame("candidate_after_cooling", target),
            *empty_full_scan_frames("empty_after_pick"),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    candidate_precheck_count = 0

    def reject_first_candidate_route(targets, *, step_m):
        nonlocal candidate_precheck_count
        if len(targets) != 6:
            return original_precheck(targets, step_m=step_m)
        candidate_precheck_count += 1
        if candidate_precheck_count == 1:
            raise RuntimeError(f"synthetic route rejection at step {step_m}")
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = reject_first_candidate_route

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.REJECTED,
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert report.successes == 1
    assert report.results[0].target.instance_id == "mug_1"
    assert report.results[1].target.instance_id == "mug_1"
    assert pipeline._task_cd_completed_instance_ids == {"mug_1"}
    assert pipeline._task_cd_completed_pick_count == 1
    assert pipeline._task_cd_rejected_instance_ids == set()
    assert pipeline._task_cd_rejected_spatial_fingerprints == []
    assert vision.observation_joint_move_counts == [1, 1, 1, 1, 2, 2, 2, 2]


def test_vision_exhaustion_is_aborted_and_never_reported_as_empty(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(config, [])

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.ABORTED]
    assert report.successes == 0
    assert "emptiness is unproven" in report.results[0].message
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []


def test_max_picks_is_ignored_until_the_bin_is_empty(
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
            *empty_full_scan_frames("empty_after_pick_2"),
        ],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert report.requested_picks == 0
    assert report.successes == 2
    assert report.results[0].target.instance_id == "mug_1"
    assert report.results[1].target.instance_id == "bowl_1"
    assert pipeline._task_cd_completed_instance_ids == {"mug_1", "bowl_1"}
    assert pipeline._task_cd_completed_pick_count == 2
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry",
        "task_cd_return_observation_after_track_1",
        "task_cd_return_observation_after_track_2",
    ]
    assert vision.observation_joint_move_counts == [1, 2, 3, 3, 3, 3]
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
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


def test_acceptance_success_limit_stops_after_one_completed_pick(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    mug = detection("mug", "mug_1", graspability=1.0)
    bowl = detection("bowl", "bowl_1", graspability=0.90)
    pipeline, vision = pipeline_with_frames(
        config,
        [frame("observation_1", mug, bowl)],
    )

    report = pipeline.run(acceptance_max_successes=1)

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
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    assert pipeline.gripper.events == ["open", "close", "open"]


@pytest.mark.parametrize("status", (PickStatus.REJECTED, PickStatus.NO_TARGET))
def test_acceptance_non_success_stops_after_one_scan_without_releasing_exclusions(
    tmp_path,
    status,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    pipeline = build_pipeline(cd_config(observation_pose))
    target = detection("mug", "mug_rejected")
    calls = []
    released = []

    def reject_once(_state):
        calls.append(1)
        assert len(calls) == 1, "acceptance mode started a second observation cycle"
        pipeline._task_cd_rejected_instance_ids.add(target.instance_id)
        return PickResult(status, target, "synthetic non-success")

    pipeline._run_task_cd_clear_once = reject_once
    pipeline._clear_task_cd_temporary_exclusions = lambda: released.append(1)

    report = pipeline.run(acceptance_max_successes=1)

    assert calls == [1]
    assert released == []
    assert report.successes == 0
    assert [result.status for result in report.results] == [status]
    assert pipeline._task_cd_rejected_instance_ids == {target.instance_id}


def test_acceptance_recoverable_failure_returns_then_stops_without_retry(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    pipeline = build_pipeline(cd_config(observation_pose))
    target = detection("mug", "mug_failed")
    calls = []

    def fail_once(_state):
        calls.append(1)
        assert len(calls) == 1, "acceptance mode retried a recovered failed grasp"
        return PickResult(
            PickStatus.ABORTED,
            target,
            "synthetic safely recovered failed grasp",
            recoverable_failure=True,
        )

    pipeline._run_task_cd_clear_once = fail_once

    report = pipeline.run(acceptance_max_successes=1)

    assert calls == [1]
    assert report.successes == 0
    assert [result.status for result in report.results] == [PickStatus.ABORTED]
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry",
        "task_cd_return_observation_after_failed_track_1",
    ]


@pytest.mark.parametrize("limit", (0, -1, True, 1.0, 1.5, "1", "invalid"))
def test_acceptance_success_limit_rejects_invalid_values(tmp_path, limit):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    pipeline, _vision = pipeline_with_frames(cd_config(observation_pose), [])

    with pytest.raises(
        ValueError,
        match="acceptance_max_successes must be an integer >= 1",
    ):
        pipeline.run(acceptance_max_successes=limit)


def test_empty_confirmation_requires_fixed_and_relocation_views(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("target_visible", detection("mug", "mug_1")),
            *empty_full_scan_frames("empty_after_pick"),
        ],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert report.requested_picks == 0
    assert "complete full-bin scan confirmed empty" in report.results[1].message
    assert any(
        "2 consecutive complete box scan round(s)" in event.message
        and "4 successful view observation(s)" in event.message
        for event in report.results[1].events
    )
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    assert [name for name, _pose in scan_moves(pipeline)] == [
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
    ]
    assert vision.observation_joint_move_counts == [1, 2, 2, 2, 2]


def test_one_clean_round_is_not_enough_to_declare_the_bin_empty(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("target_visible", detection("mug", "mug_1")),
            frame("round_1_fixed_empty"),
            frame("round_1_relocation_empty"),
        ],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.ABORTED,
    ]
    assert "vision source exhausted" in report.results[1].message
    assert "emptiness is unproven" in report.results[1].message
    assert all(result.status is not PickStatus.NO_TARGET for result in report.results)


def test_fixed_view_empty_but_relocation_view_item_is_not_empty(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    # The item has metric evidence but no stable track id, so it cannot become
    # an executable candidate.  It still proves that the physical bin is not
    # empty and must prevent a terminal NO_TARGET result.
    unexecutable_item = detection("mug", "")
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("fixed_view_empty"),
            frame("relocation_view_has_item", unexecutable_item),
            *empty_full_scan_frames("eventual_empty"),
        ],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.REJECTED,
        PickStatus.NO_TARGET,
    ]
    assert "physical item observation" in report.results[0].message
    assert "not empty" in report.results[0].message


def test_oversized_candidate_is_rejected_by_width_gate_before_any_command(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        task=replace(config.task, sample_period_s=0.0),
        gripper=replace(
            config.gripper,
            require_close_travel_evidence=True,
            close_travel_width_margin_m=0.005,
        ),
    )
    target = replace(
        detection("mug", "oversized_width_target", graspability=1.0),
        major_axis_length_m=0.120,
        minor_axis_length_m=0.07096,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("observation_with_oversized_target", target)],
    )

    report = pipeline.run(max_picks=1)

    assert report.results[0].status is PickStatus.REJECTED
    assert report.results[0].target.instance_id == "oversized_width_target"
    rejection_messages = [
        event.message
        for event in report.results[0].events
        if "rejected before motion" in event.message
    ]
    assert len(rejection_messages) == 1
    assert "measured width 0.070960 m" in rejection_messages[0]
    assert "exceeds the CAD opening 0.072549 m" in rejection_messages[0]
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []


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
    target = replace(
        detection("mug", "close_travel_target_1", graspability=1.0),
        major_axis_length_m=0.100,
        minor_axis_length_m=0.05164,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("observation_1", target)],
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
            *empty_full_scan_frames("empty_after_pick_2"),
        ],
    )

    report = pipeline.run(max_picks=2)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
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
    assert vision.observation_joint_move_counts == [1, 2, 3, 3, 3, 3]
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
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


def test_recovered_empty_close_is_not_success_and_failed_location_is_retried(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    failed = detection("mug", "failed_1", graspability=1.0)
    # The detector loses and re-creates the same physical item after the arm
    # returns to the observation pose.  Spatial deduplication must first defer
    # that location, try another object, then release the temporary exclusion
    # after a complete multi-view scan and retry the failed location.
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
            *empty_full_scan_frames("empty_after_other_pick"),
            frame("failed_location_retried", failed_reidentified),
            *empty_full_scan_frames("empty_after_retry"),
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

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.ABORTED,
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert report.requested_picks == 0
    assert report.successes == 2
    assert report.results[0].recoverable_failure is True
    assert report.results[0].target.instance_id == "failed_1"
    assert report.results[1].target.instance_id == "next_1"
    assert report.results[3].target.instance_id == "failed_2"
    assert pipeline._task_cd_failed_instance_ids == set()
    assert pipeline._task_cd_failed_spatial_fingerprints == []
    assert pipeline._task_cd_completed_instance_ids == {"next_1", "failed_2"}
    assert pipeline._task_cd_completed_pick_count == 2
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry",
        "task_cd_return_observation_after_failed_track_1",
        "task_cd_return_observation_after_track_1",
        "task_cd_return_observation_after_track_2",
    ]
    assert vision.observation_joint_move_counts == [
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
    ]
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
        "pregrasp",
        "grasp",
        "task_cd_recovery_retreat",
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


def test_empty_close_retries_after_cooldown_and_is_never_counted_as_success(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("one_candidate", detection("mug", "mug_1")),
            *empty_full_scan_frames("empty_after_recovery"),
            frame("candidate_retried_after_cooldown", detection("mug", "mug_1")),
            *empty_full_scan_frames("empty_after_retry"),
        ],
    )

    class FailOnlyOnce:
        failed = False

        def verify(self, _target, _plan):
            if not self.failed:
                self.failed = True
                return False, "synthetic empty close"
            return True, "synthetic hold accepted after retry"

    pipeline.hold_verifier = FailOnlyOnce()

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.ABORTED,
        PickStatus.NO_TARGET,
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert report.requested_picks == 0
    assert report.successes == 1
    assert report.results[0].recoverable_failure is True
    assert report.results[0].success is False
    assert report.results[1].success is False
    assert report.results[2].target.instance_id == "mug_1"
    assert pipeline._task_cd_failed_instance_ids == set()
    assert pipeline._task_cd_failed_spatial_fingerprints == []
    assert pipeline._task_cd_completed_instance_ids == {"mug_1"}
    assert pipeline._task_cd_completed_pick_count == 1
    assert vision.observation_joint_move_counts == [
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
    ]
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry",
        "task_cd_return_observation_after_failed_track_1",
        "task_cd_return_observation_after_track_1",
    ]
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
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
            frame("relocation_view_empty"),
            *empty_full_scan_frames("eventual_empty"),
        ],
    )

    report = pipeline.run(max_picks=2)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.REJECTED,
        PickStatus.NO_TARGET,
    ]
    assert report.successes == 1
    assert report.results[0].target.instance_id == "mug_1"
    assert pipeline._task_cd_completed_instance_ids == {"mug_1"}
    assert len(pipeline._task_cd_completed_spatial_fingerprints) == 1
    assert vision.observation_joint_move_counts == [1, 2, 2, 2, 2, 2, 2]
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
            *empty_full_scan_frames("empty_after_pick_2"),
        ],
    )

    report = pipeline.run(max_picks=2)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert [
        result.target.instance_id
        for result in report.results
        if result.target is not None
    ] == [
        "mug_1",
        "mug_2",
    ]
    assert pipeline._task_cd_completed_instance_ids == {"mug_1", "mug_2"}
    assert len(pipeline._task_cd_completed_spatial_fingerprints) == 2
    assert vision.observation_joint_move_counts == [1, 2, 3, 3, 3, 3]


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
            *empty_full_scan_frames("empty_after_second_choice"),
        ],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert report.results[0].target.instance_id == "reachable_1"
    assert pipeline._task_cd_completed_instance_ids == {"reachable_1"}
    assert any(
        "unreachable_1" in event.message and "rejected before motion" in event.message
        for event in report.results[0].events
    )
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
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


def test_empty_bin_scans_then_finishes_without_grasp_or_gripper_command(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        empty_full_scan_frames("empty"),
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.NO_TARGET]
    assert report.successes == 0
    assert "complete full-bin scan confirmed empty" in report.results[0].message
    assert non_scan_moves(pipeline) == []
    assert [name for name, _pose in scan_moves(pipeline)] == [
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
    ]
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
        [
            frame("off_target_only", detection("bowl", "bowl_1")),
            frame("relocation_view_empty"),
            *empty_full_scan_frames("eventual_empty"),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    precheck_routes = []

    def recording_precheck(targets, *, step_m):
        precheck_routes.append(tuple(targets))
        return original_precheck(targets, step_m=step_m)

    def unexpected_policy_select(_detections):
        raise AssertionError("off-target must not reach target policy")

    pipeline.motion.precheck_linear_targets = recording_precheck
    pipeline.policy.select = unexpected_policy_select
    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.REJECTED,
        PickStatus.NO_TARGET,
    ]
    assert len(precheck_routes) == 2
    assert len(precheck_routes[0]) == 2
    assert non_scan_moves(pipeline) == []
    assert [name for name, _pose in scan_moves(pipeline)] == [
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
    ]
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

    # max_picks remains only as a compatible configuration field.  Task C with
    # task_cd.enabled ignores it and stops solely on a verified empty bin.
    assert config.task.max_picks == 40
    assert config.motion.backend == "robotmove"
    assert config.motion.dry_run is False
    assert config.motion.workspace_min == pytest.approx((-0.80, -0.80, 0.02))
    assert config.motion.workspace_max == pytest.approx((0.80, 0.80, 0.80))
    assert config.motion.speed_mps == pytest.approx(0.09)
    assert config.motion.joint_speed_rad_s == pytest.approx(1.8)
    assert config.motion.joint_speed_safety_limit_rad_s == pytest.approx(1.8)
    assert config.gripper.backend == "ms42dc"
    assert config.gripper.require_feedback is True
    assert config.gripper.hold_min_angle_deg == pytest.approx(1978.0)
    assert config.task_cd.enabled is True
    assert config.task_cd.return_to_observation_after_each_pick is True
    assert config.task_cd.place_pose_verified is True
    assert config.task_cd.exclude_completed_tracks is True
    assert config.search.enabled is False
    assert config.vision.stabilizer_max_per_class == 0
    assert config.motion.grasp_safety.maximum_tool_tilt_deg == 30.0
    assert config.motion.grasp_safety.maximum_place_tool_tilt_deg == 18.0
    assert config.motion.grasp_safety.maximum_horizontal_reach_m == pytest.approx(
        0.82
    )
    assert config.motion.grasp_safety.maximum_grasp_flange_z_m is None
    assert config.motion.grasp_safety.maximum_grasp_tip_z_m is None
