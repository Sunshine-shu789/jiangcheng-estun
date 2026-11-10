from dataclasses import replace
from math import inf, nan, pi
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

from track1_grasp.config import (
    CornerSmallObjectFilterConfig,
    GripperSelfFilterConfig,
    TaskCDClearConfig,
    TaskCDEmptyConfirmationConfig,
    TaskCDRelocationScanConfig,
    apply_mode,
    load_config,
)
from track1_grasp.domain import Detection, Frame, PickResult, PickStatus, TaskPhase
from track1_grasp.factory import build_pipeline
from track1_grasp.kinematics import controller_fk, matrix_to_rpy
from track1_grasp.motion import RobotMoveMotionBackend


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_V11_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_c_v11_official_20261102.yaml"
)
FORMAL_TASK_D_V11_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_d_v11_official_20261102.yaml"
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
    startup_direct_grasp: bool = False,
    empty_check_interval_successes: int = 20,
    fixed_view_miss_retry_frames: int = 3,
    fixed_view_no_progress_cycles: int = 3,
    blocked_retry_period_s: float = 0.001,
    place_pose=PLACE_POSE,
    place_pose_frame: str | None = "jaw_tip",
    place_pose_verified: bool = True,
    observation_tolerance_deg: float = 0.5,
    search_enabled: bool = False,
    relocation_offsets=((0.02, 0.0, 0.0),),
    complete_box_before_planning: bool = False,
    fast_forward_enabled: bool = False,
    fast_forward_offset_xyz_m=(0.0, 0.11, 0.0),
    fast_forward_observe_frames: int = 2,
    fast_forward_confirm_hits: int = 2,
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
            startup_direct_grasp=startup_direct_grasp,
            empty_check_interval_successes=empty_check_interval_successes,
            fixed_view_miss_retry_frames=fixed_view_miss_retry_frames,
            fixed_view_no_progress_cycles=fixed_view_no_progress_cycles,
            blocked_retry_period_s=blocked_retry_period_s,
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
                fast_forward_enabled=fast_forward_enabled,
                fast_forward_offset_xyz_m=fast_forward_offset_xyz_m,
                fast_forward_settle_s=0.0,
                fast_forward_observe_frames=fast_forward_observe_frames,
                fast_forward_confirm_hits=fast_forward_confirm_hits,
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


def robot_feedback_backend(config, joints):
    """Build a minimal physical-feedback backend for capture regression tests."""

    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config.motion
    backend._log = lambda _message: None
    backend._lock = threading.Lock()
    backend._state = ""
    backend._end_positions = []
    backend._joint_positions = []
    backend._dry_run_joint_positions = None
    backend._fault_flag = 0
    backend._stamp = 0.0
    backend._feedback_sequence = 0
    backend._on_info(robot_info_message(joints))
    return backend


def robot_info_message(joints):
    values = tuple(float(value) for value in joints)
    transform = controller_fk(values)
    return SimpleNamespace(
        state="AutoReady",
        end_positions=[
            *[float(value) for value in transform[:3, 3]],
            *[float(value) for value in matrix_to_rpy(transform[:3, :3])],
        ],
        joint_positions=list(values),
        fault_flag=0,
    )


def test_task_cd_capture_waits_for_robot_info_after_slow_first_inference(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        motion=replace(config.motion, robot_info_timeout_s=0.03),
    )
    pipeline, _vision = pipeline_with_frames(config, ())
    actual = OBSERVATION_RADIANS
    updated = tuple(
        value + (0.00001 if index == 0 else 0.0)
        for index, value in enumerate(actual)
    )
    backend = robot_feedback_backend(config, actual)
    pipeline.motion = backend

    class SlowInferenceVision:
        def capture(self):
            # Simulate the observed first-YOLO capture stall, then deliver a
            # physical RobotInfo callback only after the stall.
            time.sleep(0.05)
            backend._on_info(robot_info_message(updated))
            return frame("slow_first_inference")

    pipeline.vision = SlowInferenceVision()
    waited = []
    original_wait = backend.wait_for_feedback_state

    def record_wait(timeout_s, *, newer_than):
        waited.append((timeout_s, newer_than))
        return original_wait(timeout_s, newer_than=newer_than)

    backend.wait_for_feedback_state = record_wait
    result = pipeline._task_cd_capture_frame(
        lambda *_args: None,
        context="slow first inference regression",
    )

    assert result.frame_id == "slow_first_inference"
    assert result.capture_robot_state is not None
    assert result.capture_robot_state.before.joints == pytest.approx(actual)
    assert result.capture_robot_state.after.joints == pytest.approx(updated)
    assert waited == [(pytest.approx(0.03), 1)]


def test_task_cd_capture_remains_fail_closed_without_post_capture_feedback(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        motion=replace(config.motion, robot_info_timeout_s=0.02),
    )
    pipeline, _vision = pipeline_with_frames(config, ())
    actual = OBSERVATION_RADIANS
    backend = robot_feedback_backend(config, actual)
    pipeline.motion = backend

    class StalledFeedbackVision:
        def capture(self):
            time.sleep(0.05)
            return frame("no_post_capture_feedback")

    pipeline.vision = StalledFeedbackVision()
    started = time.monotonic()
    with pytest.raises(
        RuntimeError,
        match="after state is not ready: no fresh RobotInfo",
    ):
        pipeline._task_cd_capture_frame(
            lambda *_args: None,
            context="stalled feedback regression",
        )

    assert time.monotonic() - started >= 0.02


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


def test_formal_task_c_starts_fixed_and_plans_immediately_after_detection():
    config = load_config(FORMAL_V11_CONFIG)
    scan = config.task_cd.relocation_scan
    assert config.task_cd.startup_direct_grasp is True
    assert config.task_cd.empty_check_interval_successes == 20
    assert config.task_cd.fixed_view_no_progress_cycles == 3
    assert config.task_cd.blocked_retry_period_s == pytest.approx(5.0)
    assert scan.fast_forward_enabled is False
    assert scan.fast_forward_observe_frames == 1
    assert scan.fast_forward_confirm_hits == 1
    assert scan.complete_box_before_planning is False
    assert config.search.observe_frames == 1
    assert config.search.confirm_hits == 1
    assert config.vision.stabilizer_confirm_hits == 1
    assert config.motion.trajectory_compression_enabled is True
    assert config.motion.trajectory_compression_max_joint_step_deg == pytest.approx(
        0.95
    )
    assert config.task_cd.relocation_scan.settle_s == pytest.approx(0.15)


def test_formal_task_d_starts_fixed_and_plans_immediately_after_detection():
    config = load_config(FORMAL_TASK_D_V11_CONFIG)
    scan = config.task_cd.relocation_scan
    assert config.task_cd.startup_direct_grasp is True
    assert config.task_cd.empty_check_interval_successes == 20
    assert config.task_cd.fixed_view_no_progress_cycles == 3
    assert config.task_cd.blocked_retry_period_s == pytest.approx(5.0)
    assert scan.fast_forward_enabled is False
    assert scan.fast_forward_observe_frames == 1
    assert scan.fast_forward_confirm_hits == 1
    assert scan.complete_box_before_planning is False
    assert config.search.observe_frames == 1
    assert config.search.confirm_hits == 1
    assert config.vision.stabilizer_confirm_hits == 1
    assert config.motion.trajectory_compression_enabled is True
    assert config.motion.trajectory_compression_max_joint_step_deg == pytest.approx(
        0.95
    )
    assert config.task_cd.relocation_scan.settle_s == pytest.approx(0.15)


def test_ordinary_single_frame_candidate_executes_without_scan(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    target = detection("mug", "single_frame_target", graspability=1.0)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("single_frame_target_frame", target),
            frame("must_not_be_observed", target),
        ],
    )

    def unexpected_scan_helper(*_args, **_kwargs):
        raise AssertionError("single-frame direct grasp entered a scan/acquire helper")

    pipeline._acquire_task_cd_fast_forward_candidates = unexpected_scan_helper
    pipeline._acquire_task_cd_candidates = unexpected_scan_helper

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert report.results[0].target.instance_id == "single_frame_target"
    assert vision.index == 1
    assert scan_moves(pipeline) == []
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_fixed_view_miss_retries_never_relocate_or_claim_empty(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("ordinary_single_empty_frame"),
            frame("ordinary_retry_empty_frame_2"),
            frame("ordinary_retry_empty_frame_3"),
        ],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert "fixed-view miss retries exhausted" in " ".join(
        event.message for event in report.results[0].events
    )
    assert "emptiness is unproven" in report.results[0].message
    assert "after the configured success interval" in report.results[0].message
    assert vision.index == 3
    assert scan_moves(pipeline) == []
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []


def test_no_progress_blocks_without_ending_then_resumes_when_target_appears(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(
        observation_pose,
        fixed_view_no_progress_cycles=2,
        blocked_retry_period_s=0.001,
    )
    target = detection("mug", "target_after_operator_change", graspability=1.0)
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("blocked_empty_1"),
            frame("blocked_empty_2"),
            frame("blocked_empty_3"),
            frame("blocked_empty_4"),
            frame("blocked_empty_5"),
            frame("blocked_empty_6"),
            frame("new_executable_target", target),
        ],
    )
    logs = []
    waits = []
    pipeline._log = logs.append

    def record_wait(reason, *, delay_s=None):
        waits.append((reason, delay_s))

    pipeline._wait_task_cd_retry = record_wait

    report = pipeline.run()

    assert [result.status for result in report.results] == [
        PickStatus.REJECTED,
        PickStatus.REJECTED,
        PickStatus.SUCCESS,
        PickStatus.ABORTED,
    ]
    assert report.successes == 1
    assert report.results[2].target.instance_id == "target_after_operator_change"
    assert any("[run] C/D BLOCKED" in message for message in logs)
    assert any(
        reason.startswith("safe blocked wait")
        and delay_s == pytest.approx(0.001)
        for reason, delay_s in waits
    )
    assert scan_moves(pipeline) == []


def test_single_frame_physical_evidence_is_passed_to_full_scan(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    physical_item_without_track = detection("mug", "")
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("single_physical_item", physical_item_without_track),
            frame("fixed_retry_after_physical_evidence_2"),
            frame("fixed_retry_after_physical_evidence_3"),
        ],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert report.results[0].target is None
    assert "bin is not empty" in report.results[0].message
    assert "none produced an executable grasp candidate" in report.results[0].message
    assert "no relocation scan is allowed" in report.results[0].message
    assert vision.index == 3
    assert scan_moves(pipeline) == []
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []


def test_single_frame_precheck_rejection_is_deferred_without_motion(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    target = detection("mug", "precheck_rejected", graspability=1.0)
    pipeline, vision = pipeline_with_frames(
        config,
        [frame("precheck_rejected_frame", target)],
    )

    def reject_precheck(_target, _capture_state, _route_state, emit):
        emit(TaskPhase.PLAN, "synthetic precheck rejection")
        return None

    pipeline._precheck_task_cd_candidate = reject_precheck
    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert report.results[0].target.instance_id == "precheck_rejected"
    assert "deferred temporarily" in report.results[0].message
    assert "another visible target" in report.results[0].message
    assert vision.index == 1
    assert scan_moves(pipeline) == []
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    assert pipeline._task_cd_rejected_instance_ids == {"precheck_rejected"}


def test_next_cycle_after_success_uses_single_frame_not_full_scan(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    first_target = detection("mug", "first_target", graspability=1.0)
    second_target = detection("bowl", "second_target", graspability=0.9)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("first_single_frame", first_target),
            frame("second_single_frame", second_target),
        ],
    )

    report = pipeline.run(acceptance_max_successes=2)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.SUCCESS,
    ]
    assert report.successes == 2
    # The second pick starts from one fixed-view frame and does not perform the
    # periodic full-bin scan.  Operator-acceptance mode stops only after the
    # requested success count and explicitly does not claim that the bin is
    # empty.
    assert vision.index == 2
    assert scan_moves(pipeline) == []
    assert pipeline.gripper.events == [
        "open",
        "close",
        "open",
        "close",
        "open",
    ]


def test_periodic_full_empty_check_runs_only_after_twenty_successes(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    pipeline = build_pipeline(
        cd_config(observation_pose, empty_check_interval_successes=20)
    )
    calls = []

    def scripted_cycle(_state, *, force_full_scan=False):
        calls.append(force_full_scan)
        if len(calls) <= 20:
            return PickResult(
                PickStatus.SUCCESS,
                detection("mug", f"periodic_{len(calls)}"),
                "synthetic completed transfer, release and retreat",
            )
        return PickResult(
            PickStatus.NO_TARGET,
            None,
            "synthetic complete empty check",
        )

    pipeline._run_task_cd_clear_once = scripted_cycle
    report = pipeline.run()

    assert report.successes == 20
    assert calls == [False] * 20 + [True]


def test_task_cd_plans_from_first_visible_frame_without_extra_observation(
    tmp_path,
):
    """A stabilized first frame must lead directly to planning, not a scan."""

    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        search=replace(config.search, observe_frames=4, confirm_hits=1),
    )
    config.validate()
    target = detection("mug", "mug_first_visible_frame", graspability=1.0)
    pipeline, vision = pipeline_with_frames(
        config,
        [frame("first_visible_frame", target)],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert report.results[0].target.instance_id == "mug_first_visible_frame"
    assert vision.index == 1
    assert scan_moves(pipeline) == []
    assert any(
        "first usable candidate enters" in event.message
        and "without relocation-scan motion" in event.message
        for event in report.results[0].events
    )


def test_task_cd_confirmation_returns_as_soon_as_latest_frame_is_visible(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        search=replace(config.search, observe_frames=4, confirm_hits=2),
    )
    config.validate()
    target = detection("mug", "mug_fast_confirmation", graspability=1.0)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("fast_confirmation_1", target),
            frame("fast_confirmation_2", target),
            frame("must_not_be_waited_for"),
        ],
    )
    events = []

    observation = pipeline._observe_task_cd_candidates(
        lambda phase, message: events.append((phase, message)),
        observation_name="confirmation regression",
        observe_frames_override=4,
        confirm_hits_override=2,
    )

    assert [candidate.detection.instance_id for candidate in observation.candidates] == [
        "mug_fast_confirmation"
    ]
    assert vision.index == 2
    assert scan_moves(pipeline) == []
    assert any(
        "planning starts immediately" in message
        for _phase, message in events
    )


def test_task_cd_confirmation_tolerates_one_missed_detection_frame(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        search=replace(config.search, observe_frames=4, confirm_hits=2),
    )
    config.validate()
    target = detection("mug", "mug_dropout", graspability=1.0)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("dropout_hit_1", target),
            frame("dropout_miss_1"),
            frame("dropout_hit_2", target),
            frame("must_not_be_waited_for"),
        ],
    )
    events = []

    observation = pipeline._observe_task_cd_candidates(
        lambda phase, message: events.append((phase, message)),
        observation_name="dropout regression",
        observe_frames_override=4,
        confirm_hits_override=2,
    )

    assert [candidate.detection.instance_id for candidate in observation.candidates] == [
        "mug_dropout"
    ]
    assert vision.index == 3
    assert scan_moves(pipeline) == []


def test_task_cd_confirmation_reassociates_a_replaced_track_id(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        search=replace(config.search, observe_frames=4, confirm_hits=2),
    )
    config.validate()
    first_track = detection("mug", "mug_track_before_reid", graspability=1.0)
    replacement_track = detection(
        "mug",
        "mug_track_after_reid",
        graspability=1.0,
    )
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("reid_hit_1", first_track),
            frame("reid_hit_2", replacement_track),
            frame("must_not_be_waited_for"),
        ],
    )
    events = []

    observation = pipeline._observe_task_cd_candidates(
        lambda phase, message: events.append((phase, message)),
        observation_name="track replacement regression",
        observe_frames_override=4,
        confirm_hits_override=2,
    )

    assert [candidate.detection.instance_id for candidate in observation.candidates] == [
        "mug_track_after_reid"
    ]
    assert vision.index == 2
    assert scan_moves(pipeline) == []


def test_fixed_view_candidate_skips_relocation_route_precheck(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    offsets = (
        (0.02, 0.0, 0.0),
        (-0.02, 0.0, 0.0),
        (0.0, 0.02, 0.0),
        (0.0, -0.02, 0.0),
    )
    config = cd_config(
        observation_pose,
        relocation_offsets=offsets,
        complete_box_before_planning=False,
    )
    target = detection("mug", "fixed_fast_path", graspability=1.0)
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("fixed_fast_path_candidate", target)],
    )

    route_prechecks = 0
    original_precheck = pipeline._precheck_task_cd_relocation_route

    def record_route_precheck(anchor, poses, emit, **kwargs):
        nonlocal route_prechecks
        route_prechecks += 1
        return original_precheck(anchor, poses, emit, **kwargs)

    pipeline._precheck_task_cd_relocation_route = record_route_precheck
    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert route_prechecks == 0
    assert scan_moves(pipeline) == []
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]


def test_corner_small_object_keeps_normal_target_first_without_losing_evidence(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        vision=replace(
            config.vision,
            corner_small_object_filter=CornerSmallObjectFilterConfig(
                enabled=True,
                mode="adaptive",
            ),
        ),
    )
    config.validate()
    normal = detection(
        "mug",
        "normal_target",
        bbox=(250.0, 150.0, 390.0, 290.0),
    )
    corner_small = detection(
        "10-meat_can",
        "corner_small_target",
        x=0.35,
        bbox=(5.0, 405.0, 45.0, 445.0),
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("normal_and_corner_small", normal, corner_small)],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert report.results[0].target.instance_id == "normal_target"
    assert any(
        "corner_small_detected=1" in event.message
        and "physical_detection_evidence=2" in event.message
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


def test_middle_targets_are_picked_before_large_corner_targets(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        vision=replace(
            config.vision,
            corner_small_object_filter=CornerSmallObjectFilterConfig(
                enabled=True,
                mode="adaptive",
            ),
        ),
    )
    config.validate()
    middle = detection(
        "mug",
        "middle_target",
        x=0.08,
        graspability=0.80,
        bbox=(250.0, 150.0, 390.0, 290.0),
    )
    large_corner = detection(
        "bowl",
        "large_corner_target",
        x=0.05,
        y=0.20,
        graspability=1.0,
        bbox=(5.0, 405.0, 145.0, 445.0),
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("middle_and_large_corner", middle, large_corner),
            frame("middle_done_large_corner_remains", middle, large_corner),
        ],
    )

    report = pipeline.run(acceptance_max_successes=2)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.SUCCESS,
    ]
    assert [result.target.instance_id for result in report.results] == [
        "middle_target",
        "large_corner_target",
    ]
    assert report.successes == 2


def test_only_corner_small_object_is_picked_instead_of_claimed_empty(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        vision=replace(
            config.vision,
            corner_small_object_filter=CornerSmallObjectFilterConfig(
                enabled=True,
                mode="adaptive",
            ),
        ),
    )
    config.validate()
    corner_small = detection(
        "mug",
        "corner_small_only",
        x=0.35,
        bbox=(5.0, 405.0, 45.0, 445.0),
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("corner_small_only_fixed", corner_small)],
    )

    result = pipeline._run_task_cd_clear_once(
        pipeline.motion.get_state(),
        force_full_scan=True,
    )

    assert result.status is PickStatus.SUCCESS
    assert result.target is not None
    assert result.target.instance_id == "corner_small_only"
    assert result.empty_bin_confirmed is False
    assert any(
        "corner_small_detected=1" in event.message
        and "physical_detection_evidence=1" in event.message
        for event in result.events
    )
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    assert scan_moves(pipeline) == []
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_unexecutable_corner_small_item_never_confirms_empty(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        vision=replace(
            config.vision,
            corner_small_object_filter=CornerSmallObjectFilterConfig(
                enabled=True,
                mode="adaptive",
            ),
        ),
    )
    config.validate()
    # A real detection without a stable track id cannot be executed, but it is
    # still positive physical evidence and must prevent a terminal NO_TARGET.
    corner_small = detection(
        "mug",
        "",
        x=0.35,
        bbox=(5.0, 405.0, 45.0, 445.0),
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("corner_small_unexecutable_fixed", corner_small),
            frame("corner_small_unexecutable_relocation", corner_small),
        ],
    )

    result = pipeline._run_task_cd_clear_once(
        pipeline.motion.get_state(),
        force_full_scan=True,
    )

    assert result.status is PickStatus.REJECTED
    assert result.empty_bin_confirmed is False
    assert "physical item observation" in result.message
    assert "not empty" in result.message
    assert any(
        "corner_small_detected=1" in event.message
        and "physical_detection_evidence=1" in event.message
        for event in result.events
    )
    assert non_scan_moves(pipeline) == []
    assert [name for name, _pose in scan_moves(pipeline)] == [
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
    ]
    assert pipeline.gripper.events == []


def test_periodic_empty_check_prechecks_and_runs_relocation_rounds(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    offsets = ((0.02, 0.0, 0.0),)
    config = cd_config(
        observation_pose,
        relocation_offsets=offsets,
        complete_box_before_planning=False,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        empty_full_scan_frames("empty_fallback"),
    )

    route_prechecks = 0
    original_precheck = pipeline._precheck_task_cd_relocation_route

    def record_route_precheck(anchor, poses, emit, **kwargs):
        nonlocal route_prechecks
        route_prechecks += 1
        return original_precheck(anchor, poses, emit, **kwargs)

    pipeline._precheck_task_cd_relocation_route = record_route_precheck
    result = pipeline._run_task_cd_clear_once(
        pipeline.motion.get_state(),
        force_full_scan=True,
    )

    assert result.status is PickStatus.NO_TARGET
    assert route_prechecks == 1
    assert [name for name, _pose in scan_moves(pipeline)] == [
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
    ]
    assert pipeline.gripper.events == []


def test_formal_task_c_preserves_the_verified_place_pose():
    config = load_config(FORMAL_V11_CONFIG)

    assert config.motion.place_pose == pytest.approx(
        (
            0.270377682201,
            0.335654065085,
            0.246477463808,
            -2.84530955462,
            0.015450189026,
            -0.92808733871,
        )
    )
    assert config.motion.place_pose_frame == "jaw_tip"


def test_legacy_fast_forward_flag_does_not_delay_first_frame_grasp(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    forward_offset = (0.0, 0.11, 0.0)
    config = cd_config(
        observation_pose,
        relocation_offsets=(forward_offset,),
        fast_forward_enabled=True,
        fast_forward_offset_xyz_m=forward_offset,
    )
    target = detection("mug", "mug_direct", graspability=1.0)
    pipeline, vision = pipeline_with_frames(
        config,
        [frame("ordinary_first_frame", target), frame("must_not_be_observed", target)],
    )

    original_precheck = pipeline._precheck_task_cd_candidate
    observed_planning_states = []

    def record_planning_state(target, capture_state, route_state, emit):
        observed_planning_states.append(
            (tuple(capture_state.pose.xyz), tuple(route_state.pose.xyz))
        )
        return original_precheck(target, capture_state, route_state, emit)

    pipeline._precheck_task_cd_candidate = record_planning_state
    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert report.successes == 1
    names = [name for name, _pose in pipeline.motion.moves]
    assert "task_cd_fast_forward" not in names
    assert "task_cd_fast_forward_return" not in names
    assert scan_moves(pipeline) == []
    assert observed_planning_states == [
        ((0.0, 0.0, 0.30), (0.0, 0.0, 0.30))
    ]
    assert vision.observation_joint_move_counts == [1]


def test_legacy_fast_forward_flag_keeps_fixed_view_miss_retries(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    forward_offset = (0.0, 0.11, 0.0)
    config = cd_config(
        observation_pose,
        relocation_offsets=(forward_offset,),
        fast_forward_enabled=True,
        fast_forward_offset_xyz_m=forward_offset,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("fast_forward_empty_1"),
            frame("fast_forward_empty_2"),
            frame("fast_forward_empty_3"),
        ],
    )

    state = pipeline._move_to_task_cd_observation_pose(
        move_name="task_cd_observation_entry"
    )
    result = pipeline._run_task_cd_clear_once(state)

    assert result.status is PickStatus.REJECTED
    assert scan_moves(pipeline) == []
    assert "task_cd_fast_forward" not in {
        name for name, _pose in pipeline.motion.moves
    }
    assert pipeline.gripper.events == []


def test_periodic_scan_route_rejection_issues_no_arm_or_gripper_motion(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(
        observation_pose,
        relocation_offsets=((0.02, 0.0, 0.0),),
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("periodic_fixed_empty"), frame("unused_relocation")],
    )

    def reject_relocation_route(*_args, **_kwargs):
        raise RuntimeError("synthetic periodic scan route rejection")

    pipeline._precheck_task_cd_relocation_route = reject_relocation_route
    result = pipeline._run_task_cd_clear_once(
        pipeline.motion.get_state(),
        force_full_scan=True,
    )

    assert result.status is PickStatus.ABORTED
    assert "synthetic periodic scan route rejection" in result.message
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []


def test_fixed_view_vision_exhaustion_aborts_without_motion(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(
        observation_pose,
        relocation_offsets=((0.02, 0.0, 0.0),),
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("only_one_fast_forward_frame")],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.ABORTED]
    assert scan_moves(pipeline) == []
    assert "vision source exhausted" in report.results[0].message
    assert pipeline.gripper.events == []


def test_rejected_candidate_defers_and_stops_before_unsafe_retry(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    forward_offset = (0.0, 0.11, 0.0)
    config = cd_config(
        observation_pose,
        relocation_offsets=(forward_offset,),
        fast_forward_enabled=True,
        fast_forward_offset_xyz_m=forward_offset,
        fast_forward_observe_frames=1,
        fast_forward_confirm_hits=1,
    )
    target = detection("mug", "mug_rejected", graspability=1.0)
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("fast_forward_rejection", target),
            frame("fast_forward_rejected_target_excluded", target),
            frame("fixed_view_rejected_target_excluded", target),
            frame("relocation_view_rejected_target_excluded", target),
        ],
    )

    def reject_candidate(target, capture_state, route_state, emit):
        emit(TaskPhase.PLAN, "synthetic read-only candidate rejection")
        return None

    pipeline._precheck_task_cd_candidate = reject_candidate
    state = pipeline._move_to_task_cd_observation_pose(
        move_name="task_cd_observation_entry"
    )
    first = pipeline._run_task_cd_clear_once(state)
    second = pipeline._run_task_cd_clear_once(state)

    assert first.status is PickStatus.REJECTED
    assert second.status is PickStatus.REJECTED
    assert first.target.instance_id == "mug_rejected"
    assert second.target is None
    assert "deferred temporarily" in first.message
    assert "not empty" in second.message
    assert "no relocation scan is allowed" in second.message
    assert pipeline._task_cd_fixed_view_exclusion_release_used is False
    assert scan_moves(pipeline) == []
    assert pipeline.gripper.events == []


@pytest.mark.parametrize(
    "field_name",
    (
        "enabled",
        "startup_direct_grasp",
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
    "invalid_value",
    (True, False, 0, -1, 1.0, 20.0, "20", None),
)
def test_empty_check_interval_rejects_non_positive_non_integers(
    tmp_path, invalid_value
):
    with pytest.raises(
        ValueError,
        match=(
            "task_cd.empty_check_interval_successes must be an integer >= 1"
        ),
    ):
        TaskCDClearConfig.from_dict(
            {"empty_check_interval_successes": invalid_value}, tmp_path
        )


@pytest.mark.parametrize(
    "invalid_value",
    (True, False, 0, -1, 1.0, 3.0, "3", None),
)
def test_fixed_view_no_progress_cycles_rejects_non_positive_non_integers(
    tmp_path, invalid_value
):
    with pytest.raises(
        ValueError,
        match="task_cd.fixed_view_no_progress_cycles must be an integer >= 1",
    ):
        TaskCDClearConfig.from_dict(
            {"fixed_view_no_progress_cycles": invalid_value}, tmp_path
        )


@pytest.mark.parametrize(
    "invalid_value",
    (True, False, 0.0, -1.0, nan, inf, "5.0"),
)
def test_blocked_retry_period_rejects_non_positive_non_finite_values(
    tmp_path, invalid_value
):
    with pytest.raises(
        ValueError,
        match="task_cd.blocked_retry_period_s must be positive and finite",
    ):
        TaskCDClearConfig.from_dict(
            {"blocked_retry_period_s": invalid_value}, tmp_path
        )


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

    state = pipeline._move_to_task_cd_observation_pose(
        move_name="task_cd_observation_entry"
    )
    result = pipeline._run_task_cd_clear_once(state)

    assert result.status is PickStatus.REJECTED
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
    mug = detection("mug", "mug_1", graspability=1.0)
    bowl = detection("bowl", "bowl_1", graspability=0.90)
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("two_candidates", mug, bowl),
            frame("fixed_view_after_rejection", mug, bowl),
            frame("relocation_view_after_rejection", mug, bowl),
            frame("fixed_view_after_rejection_retry_3", mug, bowl),
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
    state = pipeline._move_to_task_cd_observation_pose(
        move_name="task_cd_observation_entry"
    )
    first = pipeline._run_task_cd_clear_once(state)
    second = pipeline._run_task_cd_clear_once(state)

    assert first.status is PickStatus.REJECTED
    assert second.status is PickStatus.REJECTED
    assert first.target is not None
    assert second.target is None
    assert non_scan_moves(pipeline) == []
    # Rejected locations are excluded on the next observation.  Once physical
    # evidence remains but no executable candidate exists, the cycle returns a
    # non-empty rejection; the outer clearing loop remains active and blocked
    # instead of treating this as completion.
    assert scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]
    assert vision.index == 4
    rejection_messages = [
        event.message
        for event in first.events
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
    events = []
    state = pipeline._move_to_task_cd_observation_pose(
        move_name="task_cd_observation_entry",
        emit=lambda phase, message: events.append((phase, message)),
    )
    result = pipeline._run_task_cd_clear_once(state, force_full_scan=True)

    assert result.success
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
        for event in result.events
        if event.phase is TaskPhase.SELECT
    ]
    assert any(
        "1 confirmed unfinished candidate(s)" in message
        for message in select_messages
    )
    assert any(
        "complete box observation finished" in event.message
        and "1 unique confirmed candidate(s)" in event.message
        for event in result.events
    )


def test_read_only_candidate_rejection_skips_it_and_grasps_another(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose, empty_check_interval_successes=1)
    rejected = detection(
        "mug",
        "mug_1",
        x=0.10,
        graspability=1.0,
        bbox=(100.0, 100.0, 200.0, 200.0),
    )
    safe = detection(
        "bowl",
        "bowl_1",
        x=0.14,
        graspability=0.90,
        bbox=(220.0, 100.0, 320.0, 200.0),
    )
    pipeline, vision = pipeline_with_frames(
        config,
        [
            frame("candidate_before_rejection", rejected),
            frame("safe_candidate_after_rejection", safe),
            *empty_full_scan_frames("empty_after_pick"),
            # NO_TARGET with deferred rejections triggers one conservative
            # exclusion-release check before emptiness is final.
            *empty_full_scan_frames("empty_after_exclusion_release"),
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
    assert report.results[1].target.instance_id == "bowl_1"
    assert pipeline._task_cd_completed_instance_ids == {"bowl_1"}
    assert pipeline._task_cd_completed_pick_count == 1
    # The completed transfer releases the old rejection so the changed scene can
    # be reconsidered.  This test lowers the empty-check milestone to one only
    # so the finite scripted scene can verify that normal completion still
    # requires the complete fixed-plus-relocation empty scan.
    assert pipeline._task_cd_rejected_instance_ids == set()
    assert pipeline._task_cd_rejected_spatial_fingerprints == []
    assert vision.index == 6
    assert [name for name, _pose in scan_moves(pipeline)] == [
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
    ]
    assert pipeline.gripper.events == ["open", "close", "open"]


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
    config = cd_config(observation_pose, empty_check_interval_successes=2)
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
    assert report.empty_bin_confirmed is True
    assert report.results[-1].empty_bin_confirmed is True
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

    def reject_once(_state, *, force_full_scan=False):
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

    def fail_once(_state, *, force_full_scan=False):
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


def test_periodic_empty_confirmation_requires_fixed_and_relocation_views(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        empty_full_scan_frames("empty_after_pick"),
    )

    result = pipeline._run_task_cd_clear_once(
        pipeline.motion.get_state(),
        force_full_scan=True,
    )

    assert result.status is PickStatus.NO_TARGET
    assert result.empty_bin_confirmed is True
    assert "complete full-bin scan confirmed empty" in result.message
    assert any(
        "2 consecutive complete box scan round(s)" in event.message
        and "4 successful view observation(s)" in event.message
        for event in result.events
    )
    assert non_scan_moves(pipeline) == []
    assert [name for name, _pose in scan_moves(pipeline)] == [
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
        "task_cd_relocation_scan_01",
        "task_cd_relocation_return_01",
    ]


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

    state = pipeline._move_to_task_cd_observation_pose(
        move_name="task_cd_observation_entry"
    )
    result = pipeline._run_task_cd_clear_once(state)

    assert result.status is PickStatus.REJECTED
    assert result.empty_bin_confirmed is False
    assert "physical item observation" in result.message
    assert "not empty" in result.message


def test_oversized_candidate_is_skipped_then_stops_without_grasp_or_gripper_command(
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
        minor_axis_length_m=0.110,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [
            frame("observation_with_oversized_target", target),
            frame("fixed_view_oversized_target_excluded_1", target),
            frame("fixed_view_oversized_target_excluded_2", target),
            frame("fixed_view_oversized_target_excluded_3", target),
        ],
    )

    first = pipeline._run_task_cd_clear_once(pipeline.motion.get_state())
    second = pipeline._run_task_cd_clear_once(pipeline.motion.get_state())

    assert first.status is PickStatus.REJECTED
    assert second.status is PickStatus.REJECTED
    assert first.target.instance_id == "oversized_width_target"
    assert second.target is None
    rejection_messages = [
        event.message
        for event in first.events
        if "rejected before motion" in event.message
    ]
    assert len(rejection_messages) == 1
    assert "measured width 0.110000 m" in rejection_messages[0]
    assert "maximum CAD opening 0.105497 m" in rejection_messages[0]
    assert "fixed-view miss retries" in second.message
    assert "not empty" in second.message
    assert "no relocation scan is allowed" in second.message
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


def test_width_aware_close_at_old_empty_angle_continues_to_release(tmp_path):
    from track1_grasp.gripper import MS42DCHoldVerifier, MS42DCStatus

    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    config = replace(
        config,
        gripper=replace(
            config.gripper,
            require_close_travel_evidence=True,
            hold_min_angle_deg=0.0,
        ),
    )
    target = replace(
        detection("mug", "old_angle_target_1", graspability=1.0),
        major_axis_length_m=0.090,
        minor_axis_length_m=0.033233,
    )
    pipeline, _vision = pipeline_with_frames(
        config,
        [frame("observation_1", target)],
    )

    # The real width-aware close already accepted this object's travel.  Keep
    # the mock pipeline at the same state and replay the exact final angle from
    # the failed hardware run (1978.0 deg).
    pipeline.gripper.close_for_width = lambda _width_m: pipeline.gripper.close()

    class StableAtOldEmptyAngle:
        def read_status(self, **_kwargs):
            return MS42DCStatus(1, True, 0, 19780)

    pipeline.hold_verifier = MS42DCHoldVerifier(
        StableAtOldEmptyAngle(), config.gripper
    )

    report = pipeline.run(max_picks=1)

    assert report.results[0].status is PickStatus.SUCCESS
    assert report.results[0].success is True
    assert any(
        "no fixed empty-close angle" in event.message
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
    # One pre-approach open, one grasp close, and one release open.  There is
    # no recovery/reopen between close and the release waypoint.
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_clear_loop_returns_to_observation_before_second_allowed_pick(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose, empty_check_interval_successes=2)
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
    assert "complete full-bin scan confirmed empty" in report.results[2].message
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
    assert len(scan_moves(pipeline)) == 4
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
    config = cd_config(observation_pose, empty_check_interval_successes=2)
    failed = detection("mug", "failed_1", graspability=1.0)
    # The detector loses and re-creates the same physical item after the arm
    # returns to the observation pose.  Spatial deduplication must first defer
    # that location and picks another object.  The completed transfer then
    # releases the temporary exclusion so the re-identified item is retried
    # from the next fixed-view frame without any relocation scan.
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
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert report.requested_picks == 0
    assert report.successes == 2
    assert report.results[0].recoverable_failure is True
    assert report.results[0].target.instance_id == "failed_1"
    assert report.results[1].target.instance_id == "next_1"
    assert report.results[2].target.instance_id == "failed_2"
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
    assert vision.observation_joint_move_counts == [1, 2, 3, 4, 4, 4, 4]
    assert len(scan_moves(pipeline)) == 4
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
    config = cd_config(observation_pose, empty_check_interval_successes=1)
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
        PickStatus.REJECTED,
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
    assert len(scan_moves(pipeline)) == 4
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

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert report.successes == 1
    assert report.results[0].target.instance_id == "mug_1"
    assert pipeline._task_cd_completed_instance_ids == {"mug_1"}
    assert len(pipeline._task_cd_completed_spatial_fingerprints) == 1
    # The completed physical location may be falsely re-identified with a new
    # track id.  A conservative full scan must exclude it from grasping, but it
    # must also preserve the physical evidence and refuse to claim the bin is
    # empty.
    full_scan = pipeline._run_task_cd_clear_once(
        pipeline.motion.get_state(), force_full_scan=True
    )
    assert full_scan.status is PickStatus.REJECTED
    assert "physical item observation" in full_scan.message
    assert "not empty" in full_scan.message
    assert any(
        "completed_spatial_excluded=1" in event.message
        for result in (report.results[0], full_scan)
        for event in result.events
    )
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_nearby_same_class_object_is_not_spatially_excluded(tmp_path):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose, empty_check_interval_successes=2)
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
    assert "complete full-bin scan confirmed empty" in report.results[2].message
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
    assert len(scan_moves(pipeline)) == 4


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

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert report.results[0].target.instance_id == "reachable_1"
    assert scan_moves(pipeline) == []
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


def test_ordinary_empty_bin_does_not_scan_or_claim_empty_before_check_interval(
    tmp_path,
):
    observation_pose = write_observation_pose(tmp_path / "observation.yaml")
    config = cd_config(observation_pose)
    pipeline, _vision = pipeline_with_frames(
        config,
        empty_full_scan_frames("empty"),
    )

    state = pipeline._move_to_task_cd_observation_pose(
        move_name="task_cd_observation_entry"
    )
    result = pipeline._run_task_cd_clear_once(state)

    assert result.status is PickStatus.REJECTED
    assert "fixed-view miss retries exhausted" in result.message
    assert "emptiness is unproven" in result.message
    assert non_scan_moves(pipeline) == []
    assert scan_moves(pipeline) == []
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
    state = pipeline._move_to_task_cd_observation_pose(
        move_name="task_cd_observation_entry"
    )
    result = pipeline._run_task_cd_clear_once(state)

    assert result.status is PickStatus.REJECTED
    assert precheck_routes == []
    assert non_scan_moves(pipeline) == []
    assert scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]
    assert any(
        "off_target_excluded=1" in event.message
        for event in result.events
    )


@pytest.mark.raw_formal_cd_config
def test_formal_v11_config_enables_authorized_live_clear_loop_without_max_grasp_z():
    payload = yaml.safe_load(FORMAL_V11_CONFIG.read_text(encoding="utf-8"))

    # max_picks remains only as a compatible configuration field.  Task C with
    # task_cd.enabled ignores it and stops solely on a verified empty bin.
    assert payload["task"]["max_picks"] == 40
    assert payload["motion"]["backend"] == "robotmove"
    assert payload["motion"]["dry_run"] is False
    assert payload["motion"]["workspace_min"] == pytest.approx(
        [-0.80, -0.80, 0.02]
    )
    assert payload["motion"]["workspace_max"] == pytest.approx(
        [0.80, 0.80, 0.80]
    )
    assert payload["motion"]["speed_mps"] == pytest.approx(0.18)
    assert payload["motion"]["joint_speed_rad_s"] == pytest.approx(2.4)
    assert payload["motion"][
        "joint_speed_safety_limit_rad_s"
    ] == pytest.approx(2.4)
    assert payload["gripper"]["backend"] == "ms42dc"
    assert payload["gripper"]["require_feedback"] is True
    assert payload["gripper"]["hold_min_angle_deg"] == pytest.approx(0.0)
    assert payload["task_cd"]["enabled"] is True
    assert payload["task_cd"]["empty_check_interval_successes"] == 20
    assert payload["task_cd"]["return_to_observation_after_each_pick"] is True
    assert payload["task_cd"]["place_pose_verified"] is True
    assert payload["task_cd"]["exclude_completed_tracks"] is True
    assert payload["search"]["enabled"] is False
    assert payload["vision"]["stabilizer"]["max_per_class"] == 0
    safety = payload["motion"]["grasp_safety"]
    assert safety["maximum_tool_tilt_deg"] == 30.0
    assert safety["maximum_place_tool_tilt_deg"] == 18.0
    assert safety["maximum_horizontal_reach_m"] == pytest.approx(0.82)
    assert safety["maximum_grasp_flange_z_m"] is None
    assert safety["maximum_grasp_tip_z_m"] is None
