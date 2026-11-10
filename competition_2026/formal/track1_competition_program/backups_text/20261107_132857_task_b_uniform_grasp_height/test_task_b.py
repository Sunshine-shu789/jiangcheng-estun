import math
from dataclasses import replace
from pathlib import Path

import numpy as np
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
from track1_grasp.kinematics import controller_fk, matrix_to_rpy
from track1_grasp.planner import PosePlanner, euler_to_matrix, load_extrinsics


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = ROOT.parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_TASK_B_CONFIG = PROGRAM_ROOT / "config" / "track1_task_b_20261020.yaml"
FORMAL_TASK_B_FINAL_LIVE_CONFIG = (
    PROGRAM_ROOT / "config" / "track1_task_b_final_20261020_live.yaml"
)
FORMAL_TASK_B_RESUME2_FINAL_LIVE_CONFIG = (
    PROGRAM_ROOT / "config" / "track1_task_b_resume2_final_20261020_live.yaml"
)
FORMAL_TASK_B_POINT2_SAFE_CONFIG = (
    PROGRAM_ROOT / "config" / "track1_task_b_point2_safe_20261027.yaml"
)
FORMAL_TASK_B_FIXED_SAFE_CONFIG = (
    PROGRAM_ROOT
    / "config"
    / "track1_task_b_five_point_fixed_safe_20261027.yaml"
)
FORMAL_TASK_B_FIXED_LIVE_CONFIG = (
    PROGRAM_ROOT
    / "config"
    / "track1_task_b_five_point_fixed_live_20261027.yaml"
)
FORMAL_TASK_B_POINT2_UNSAFE_CONFIGS = (
    PROGRAM_ROOT / "config" / "track1_task_b_point2_random_preflight_20261026.yaml",
    PROGRAM_ROOT / "config" / "track1_task_b_point2_random_live_20261027.yaml",
)
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


def task_b_config_with_relative_observation_search(
    *,
    observe_frames=1,
    confirm_hits=1,
):
    config = task_b_config(rounds=1)
    return replace(
        config,
        search=replace(
            config.search,
            enabled=True,
            arbitrary_observation_offsets_xyz_m=((0.0, 0.0, 0.020),),
            settle_s=0.0,
            observe_frames=observe_frames,
            confirm_hits=confirm_hits,
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


def test_task_b_observation_anchor_supplies_all_high_waypoints():
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
    anchor = Pose((0.18, 0.05, 0.30), (2.95, 0.12, -0.20))

    plan = planner.plan(
        target_detection(x=0.12, y=-0.11),
        state,
        return_to_pick=True,
        grasp_z_m=0.168,
        grasp_rpy=(3.05, 0.08, -0.4),
        task_b_anchor_pose=anchor,
    )

    assert plan.pregrasp_pose == anchor
    assert plan.lift_pose == anchor
    assert plan.transfer_pose == anchor
    assert plan.retreat_pose == anchor
    assert plan.grasp_pose.xyz == pytest.approx((0.12, -0.11, 0.168))
    assert plan.grasp_pose.rpy == pytest.approx((3.05, 0.08, -0.4))
    assert plan.release_pose == plan.grasp_pose


def test_task_b_observation_anchor_can_define_the_vertical_grasp_column():
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
    anchor = Pose((0.18, 0.05, 0.30), (2.95, 0.12, -0.20))

    plan = planner.plan(
        target_detection(x=0.12, y=-0.11),
        state,
        return_to_pick=True,
        grasp_z_m=0.173,
        grasp_rpy=(3.05, 0.08, -0.4),
        task_b_anchor_pose=anchor,
        grasp_xy_from_observation=True,
    )

    assert plan.pregrasp_pose == anchor
    assert plan.grasp_pose.xyz == pytest.approx((0.18, 0.05, 0.173))
    assert plan.grasp_pose.rpy == pytest.approx((3.05, 0.08, -0.4))
    assert plan.release_pose == plan.grasp_pose
    assert plan.lift_pose == anchor
    assert plan.transfer_pose == anchor
    assert plan.retreat_pose == anchor


def test_task_b_vertical_grasp_descent_locks_xy_and_rpy_from_high_pose():
    base = task_b_config(rounds=1)
    config = replace(
        base,
        motion=replace(
            base.motion,
            orientation="current",
            approach_height=0.08,
            lift_height=0.08,
        ),
    )
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.1, 0.2, -0.1)),
        message="ready",
    )
    anchor = Pose((0.18, 0.05, 0.30), (2.95, 0.12, -0.20))
    grasp_rpy = (3.05, 0.08, -0.4)

    plan = planner.plan(
        target_detection(x=0.12, y=-0.11),
        state,
        return_to_pick=True,
        grasp_z_m=0.173,
        grasp_rpy=grasp_rpy,
        task_b_anchor_pose=anchor,
        vertical_grasp_descent=True,
    )

    expected_high_pose = Pose((0.12, -0.11, 0.253), grasp_rpy)
    assert plan.pregrasp_pose == expected_high_pose
    assert plan.lift_pose == expected_high_pose
    assert plan.transfer_pose == expected_high_pose
    assert plan.retreat_pose == expected_high_pose
    assert plan.grasp_pose.xyz == pytest.approx((0.12, -0.11, 0.173))
    assert plan.grasp_pose.rpy == pytest.approx(grasp_rpy)
    assert plan.pregrasp_pose.xyz[:2] == plan.grasp_pose.xyz[:2]
    assert plan.pregrasp_pose.rpy == plan.grasp_pose.rpy
    assert plan.pregrasp_pose.xyz[2] - plan.grasp_pose.xyz[2] == pytest.approx(0.08)
    assert plan.release_pose == plan.grasp_pose


@pytest.mark.parametrize(
    ("grasp_rpy", "flange_offset", "expected_xy"),
    [
        pytest.param(
            (0.0, 0.0, 0.0),
            (0.015, 0.0),
            (0.135, -0.11),
            id="flange-x",
        ),
        pytest.param(
            (math.pi, 0.0, 0.0),
            (0.0, 0.015),
            (0.12, -0.125),
            id="flange-y-rotated-by-roll-pi",
        ),
    ],
)
def test_task_b_flange_xy_offset_moves_the_complete_vertical_column(
    grasp_rpy, flange_offset, expected_xy
):
    base = task_b_config(rounds=1)
    config = replace(
        base,
        motion=replace(
            base.motion,
            orientation="current",
            approach_height=0.08,
            lift_height=0.08,
        ),
    )
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.1, 0.2, -0.1)),
        message="ready",
    )
    anchor = Pose((0.18, 0.05, 0.30), (2.95, 0.12, -0.20))

    plan = planner.plan(
        target_detection(x=0.12, y=-0.11),
        state,
        return_to_pick=True,
        grasp_z_m=0.173,
        grasp_rpy=grasp_rpy,
        task_b_anchor_pose=anchor,
        grasp_xy_offset_m=flange_offset,
        vertical_grasp_descent=True,
    )

    expected_high_pose = Pose((*expected_xy, 0.253), grasp_rpy)
    assert plan.pregrasp_pose == expected_high_pose
    assert plan.lift_pose == expected_high_pose
    assert plan.transfer_pose == expected_high_pose
    assert plan.retreat_pose == expected_high_pose
    assert plan.grasp_pose.xyz == pytest.approx((*expected_xy, 0.173))
    assert plan.grasp_pose.rpy == pytest.approx(grasp_rpy)
    assert plan.release_pose == plan.grasp_pose


def test_task_b_vertical_grasp_uses_fixed_eight_centimetre_approach():
    base = task_b_config(rounds=1)
    config = replace(
        base,
        motion=replace(
            base.motion,
            orientation="current",
            approach_height=0.08,
            lift_height=0.08,
        ),
    )
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.1, 0.2, -0.1)),
        message="ready",
    )
    # The observation anchor is only 47 mm above the grasp.  The vertical grasp
    # must still use the fixed 80 mm approach height once and then descend.
    anchor = Pose((0.18, 0.05, 0.22), (2.95, 0.12, -0.20))
    grasp_rpy = (3.05, 0.08, -0.4)

    plan = planner.plan(
        target_detection(x=0.12, y=-0.11),
        state,
        return_to_pick=True,
        grasp_z_m=0.173,
        grasp_rpy=grasp_rpy,
        task_b_anchor_pose=anchor,
        vertical_grasp_descent=True,
    )

    expected_high_pose = Pose((0.12, -0.11, 0.253), grasp_rpy)
    assert plan.pregrasp_pose == expected_high_pose
    assert plan.lift_pose == expected_high_pose
    assert plan.transfer_pose == expected_high_pose
    assert plan.retreat_pose == expected_high_pose
    assert plan.pregrasp_pose.xyz[2] == pytest.approx(0.253)
    assert plan.pregrasp_pose.xyz[2] - plan.grasp_pose.xyz[2] == pytest.approx(0.08)
    assert plan.grasp_pose.xyz[2] == pytest.approx(0.173)


def test_task_b_explicit_grasp_xy_uses_a_vertical_column_under_the_anchor_height():
    base = task_b_config(rounds=1)
    config = replace(
        base,
        motion=replace(
            base.motion,
            orientation="current",
            approach_height=0.08,
            lift_height=0.08,
        ),
    )
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.1, 0.2, -0.1)),
        message="ready",
    )
    anchor = Pose((0.10, 0.02, 0.30), (2.95, 0.12, -0.20))
    grasp_xy = (0.128, 0.046)
    grasp_rpy = (3.05, 0.08, -0.4)

    plan = planner.plan(
        target_detection(x=0.12, y=-0.11),
        state,
        return_to_pick=True,
        grasp_z_m=0.173,
        grasp_rpy=grasp_rpy,
        task_b_anchor_pose=anchor,
        grasp_xy_m=grasp_xy,
    )

    high_pose = Pose((*grasp_xy, 0.30), grasp_rpy)
    assert plan.pregrasp_pose == high_pose
    assert plan.lift_pose == high_pose
    assert plan.transfer_pose == high_pose
    assert plan.retreat_pose == high_pose
    assert plan.grasp_pose.xyz == pytest.approx((*grasp_xy, 0.173))
    assert plan.grasp_pose.rpy == pytest.approx(grasp_rpy)
    assert plan.pregrasp_pose.xyz[:2] == plan.grasp_pose.xyz[:2]
    assert plan.release_pose == plan.grasp_pose


def test_task_b_explicit_and_observation_grasp_xy_are_mutually_exclusive():
    planner = PosePlanner(task_b_config(rounds=1))
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.1, 0.2, -0.1)),
        message="ready",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        planner.plan(
            target_detection(),
            state,
            return_to_pick=True,
            task_b_anchor_pose=Pose((0.18, 0.05, 0.30), (3.0, 0.0, 0.0)),
            grasp_xy_from_observation=True,
            grasp_xy_m=(0.18, 0.05),
        )


def test_task_b_observation_grasp_xy_requires_an_anchor():
    config = task_b_config(rounds=1)
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.1, 0.2, -0.1)),
        message="ready",
    )

    with pytest.raises(ValueError, match="needs a task_b_anchor_pose"):
        planner.plan(
            target_detection(),
            state,
            return_to_pick=True,
            grasp_xy_from_observation=True,
        )


def test_task_b_observation_anchor_is_rejected_for_non_return_plan():
    config = task_b_config(rounds=1)
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.1, 0.2, -0.1)),
        message="ready",
    )

    with pytest.raises(ValueError, match="only valid with return_to_pick"):
        planner.plan(
            target_detection(),
            state,
            return_to_pick=False,
            task_b_anchor_pose=Pose((0.18, 0.05, 0.30), (3.0, 0.0, 0.0)),
        )


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

    messages = []
    target, _frame, final_state, reason = pipeline._acquire_target(
        initial_state,
        lambda _phase, message, **_kwargs: messages.append(message),
        expected_point=config.place_poses[0],
        expected_label=config.task_b.target_class,
        expected_name=config.task_b.place_names[0],
    )

    assert target is None
    assert "target not confirmed" in reason
    assert any(
        message.startswith("active search view 1/2: no confirmed target:")
        for message in messages
    )
    assert any(
        message.startswith("active search view 2/2: no confirmed target:")
        for message in messages
    )
    assert pipeline.vision.index == 3
    assert [name for name, _pose in pipeline.motion.moves] == [
        "task_search_position_01",
        "task_search_position_02",
    ]
    assert len(pipeline.motion.joint_moves) == 1
    return_name, return_joints = pipeline.motion.joint_moves[0]
    assert return_name == "task_search_return"
    assert return_joints == pytest.approx(initial_state.joints)
    assert final_state.joints == pytest.approx(initial_state.joints)
    assert not any(
        name in {"pregrasp", "grasp", "lift", "transfer", "release", "retreat"}
        for name, _pose in pipeline.motion.moves
    )


def test_task_b_prechecks_complete_cartesian_search_route_before_any_move():
    config = task_b_config_with_active_search()
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("initial_miss"),
            frame("view_1_hit", (target_detection(),)),
        ]
    )
    initial_state = pipeline.motion.get_state()
    original_precheck = pipeline.motion.precheck_linear_targets
    observed = []

    def recording_precheck(targets, *, step_m):
        observed.append(
            (
                "precheck",
                len(targets),
                len(pipeline.motion.moves),
            )
        )
        return original_precheck(targets, step_m=step_m)

    def recording_move(pose, *, name, speed_mps):
        observed.append(("move", name, len(pipeline.motion.moves)))
        pipeline.motion.moves.append((name, pose))

    pipeline.motion.precheck_linear_targets = recording_precheck
    pipeline.motion.move_to = recording_move

    pipeline._acquire_target(
        initial_state,
        lambda *_args, **_kwargs: None,
        expected_point=config.place_poses[0],
        expected_label=config.task_b.target_class,
        expected_name=config.task_b.place_names[0],
    )

    assert observed[0] == (
        "precheck",
        len(config.search.observation_offsets_xy_m),
        0,
    )
    assert observed[1] == ("move", "task_search_position_01", 0)


def test_task_b_active_search_fails_closed_before_motion_on_precheck_error():
    config = task_b_config_with_active_search()
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision([frame("initial_miss")])
    initial_state = pipeline.motion.get_state()

    def reject_search_route(_targets, *, step_m):
        raise RuntimeError(
            "near-singular posture at target 2 path waypoint: minimum weighted "
            "Jacobian singular value="
            f"0.009000 < {config.motion.cartesian_min_jacobian_singular_value:.6f}"
        )

    pipeline.motion.precheck_linear_targets = reject_search_route

    with pytest.raises(
        RuntimeError,
        match=(
            "active search route precheck rejected.*pick_1.*"
            "failed_search_view=2/2.*No arm or gripper command was issued"
        ),
    ):
        pipeline._acquire_target(
            initial_state,
            lambda *_args, **_kwargs: None,
            expected_point=config.place_poses[0],
            expected_label=config.task_b.target_class,
            expected_name=config.task_b.place_names[0],
        )

    assert pipeline.vision.index == 1


def test_task_b_active_search_moves_fail_closed_when_controller_ignores_move():
    config = task_b_config_with_active_search()
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision([frame("initial_miss")])
    initial_state = pipeline.motion.get_state()
    attempted = []

    def reject_move(pose, *, name, speed_mps):
        attempted.append((name, pose, speed_mps))
        raise RuntimeError(
            f"controller ignored joint move {name!r}; state stayed AutoReady "
            "and joints did not change"
        )

    pipeline.motion.move_to = reject_move

    with pytest.raises(
        RuntimeError,
        match=(
            "active search view 1/2 move failed; stopping without a pick: "
            "controller ignored joint move"
        ),
    ):
        pipeline._acquire_target(
            initial_state,
            lambda *_args, **_kwargs: None,
            expected_point=config.place_poses[0],
            expected_label=config.task_b.target_class,
            expected_name=config.task_b.place_names[0],
        )

    assert [name for name, _pose, _speed in attempted] == [
        "task_search_position_01"
    ]
    assert pipeline.motion.joint_moves == []
    assert pipeline.vision.index == 1
    assert pipeline.motion.moves == []
    assert pipeline.motion.joint_moves == []


def test_relative_observation_adjusts_only_after_full_initial_miss():
    config = task_b_config_with_relative_observation_search(
        observe_frames=2,
        confirm_hits=1,
    )
    pipeline = build_pipeline(config)
    initial_pose = Pose(
        (-0.0627, 0.2026, 0.2995),
        (-3.03964, -0.02044, 0.10680),
    )
    pipeline.motion._pose = initial_pose
    pipeline.vision = ScriptedVision(
        [
            frame("initial_miss_1"),
            frame("initial_miss_2"),
            frame("adjusted_hit", (target_detection(),)),
        ]
    )
    initial_state = pipeline.motion.get_state()
    original_precheck = pipeline.motion.precheck_linear_targets
    precheck_calls = []

    def recording_precheck(targets, *, step_m):
        precheck_calls.append((pipeline.vision.index, tuple(targets), step_m))
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = recording_precheck

    target, _frame, _state, reason = pipeline._acquire_target(
        initial_state,
        lambda *_args, **_kwargs: None,
        expected_label=config.task_b.target_class,
        expected_name="arbitrary_round_1",
    )

    assert target is not None
    assert "confirmed" in reason
    assert len(precheck_calls) == 1
    vision_index, targets, step_m = precheck_calls[0]
    assert vision_index == 2
    assert step_m == pytest.approx(config.motion.cartesian_step_m)
    assert len(targets) == 2
    assert targets[0].xyz == pytest.approx(
        np.asarray(initial_pose.xyz) + np.asarray((0.0, 0.0, 0.020))
    )
    assert targets[0].rpy == pytest.approx(initial_pose.rpy)
    assert targets[1].xyz == pytest.approx(initial_pose.xyz)
    assert targets[1].rpy == pytest.approx(initial_pose.rpy)
    assert [name for name, _pose in pipeline.motion.moves] == [
        "task_observation_adjust_01"
    ]
    assert pipeline.gripper.events == []


def test_relative_observation_all_miss_returns_to_initial_without_gripper():
    config = task_b_config_with_relative_observation_search()
    pipeline = build_pipeline(config)
    initial_pose = Pose(
        (0.031, -0.142, 0.287),
        (-3.02, 0.04, -0.08),
    )
    pipeline.motion._pose = initial_pose
    pipeline.vision = ScriptedVision(
        [frame("initial_miss"), frame("adjusted_miss")]
    )
    initial_state = pipeline.motion.get_state()

    target, _frame, final_state, reason = pipeline._acquire_target(
        initial_state,
        lambda *_args, **_kwargs: None,
        expected_label=config.task_b.target_class,
        expected_name="arbitrary_round_1",
    )

    assert target is None
    assert "returned to the initial observation pose" in reason
    assert [name for name, _pose in pipeline.motion.moves] == [
        "task_observation_adjust_01",
        "task_observation_return",
    ]
    assert pipeline.motion.moves[0][1].xyz == pytest.approx(
        np.asarray(initial_pose.xyz) + np.asarray((0.0, 0.0, 0.020))
    )
    assert pipeline.motion.moves[1][1].xyz == pytest.approx(initial_pose.xyz)
    assert pipeline.motion.moves[1][1].rpy == pytest.approx(initial_pose.rpy)
    assert final_state.pose.xyz == pytest.approx(initial_pose.xyz)
    assert pipeline.gripper.events == []


def test_relative_observation_first_view_success_sends_no_motion_or_precheck():
    config = task_b_config_with_relative_observation_search()
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [frame("initial_hit", (target_detection(),))]
    )
    initial_state = pipeline.motion.get_state()
    precheck_calls = []

    def recording_precheck(targets, *, step_m):
        precheck_calls.append((tuple(targets), step_m))
        raise AssertionError("initial recognition success must not move")

    pipeline.motion.precheck_linear_targets = recording_precheck

    target, _frame, _state, reason = pipeline._acquire_target(
        initial_state,
        lambda *_args, **_kwargs: None,
        expected_label=config.task_b.target_class,
        expected_name="arbitrary_round_1",
    )

    assert target is not None
    assert "confirmed" in reason
    assert precheck_calls == []
    assert pipeline.motion.moves == []
    assert pipeline.motion.joint_moves == []
    assert pipeline.gripper.events == []


def test_relative_observation_precheck_failure_issues_no_motion():
    config = task_b_config_with_relative_observation_search()
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision([frame("initial_miss")])
    initial_state = pipeline.motion.get_state()
    messages = []

    def reject_observation_route(_targets, *, step_m):
        raise RuntimeError(
            "near-singular posture at target 2 path waypoint: minimum weighted "
            f"Jacobian singular value=0.009000 < "
            f"{config.motion.cartesian_min_jacobian_singular_value:.6f}"
        )

    pipeline.motion.precheck_linear_targets = reject_observation_route

    target, _frame, _state, reason = pipeline._acquire_target(
        initial_state,
        lambda _phase, message, **_kwargs: messages.append(message),
        expected_label=config.task_b.target_class,
        expected_name="arbitrary_round_1",
    )

    assert target is None
    assert "rejected by the all-or-nothing safety precheck" in reason
    assert pipeline.vision.index == 1
    assert pipeline.motion.moves == []
    assert pipeline.motion.joint_moves == []
    assert pipeline.gripper.events == []
    assert any(
        "observation adjustment route precheck rejected before any motion" in message
        and "No arm or gripper command was issued" in message
        for message in messages
    )


def test_task_b_route_rejection_identifies_point_visual_xy_and_waypoint():
    config = task_b_config(rounds=1)
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [frame("pick_1_hit", (target_detection(x=0.15, y=-0.15),))]
    )

    def reject_full_route(_targets, *, step_m):
        raise RuntimeError(
            "near-singular posture at target 2 path waypoint: minimum weighted "
            "Jacobian singular value=0.009000 < 0.010000"
        )

    pipeline.motion.precheck_linear_targets = reject_full_route

    result = pipeline.run_once()

    assert result.status is PickStatus.REJECTED
    assert "Task B point 'pick_1'" in result.message
    assert "visual_xy=(0.150000, -0.150000)" in result.message
    assert "failed_waypoint='grasp'" in result.message
    assert "Try another bounded wrist orientation" in result.message
    assert "Do not move the object" in result.message
    assert "do not lower the configured singularity threshold" in result.message
    assert "No arm or gripper command was issued" in result.message
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []


def test_task_b_unreachable_pregrasp_is_rejected_before_full_route_or_gripper():
    config = task_b_config(rounds=2)
    pipeline = build_pipeline(config)
    pipeline._task_b_pick_index = 1
    pipeline.vision = ScriptedVision(
        [frame("pick_2_hit", (target_detection(x=-0.15, y=-0.10),))]
    )
    endpoint_calls = []
    full_route_calls = []
    motion_calls = []
    gripper_calls = []

    def reject_endpoint(pose, *, label):
        endpoint_calls.append((pose, label))
        assert pipeline.motion.moves == []
        assert pipeline.gripper.events == []
        raise RuntimeError(
            "pose IK did not converge for any damping: "
            "position=0.041152 m, rotation=0.444991 rad"
        )

    def record_full_route(targets, *, step_m):
        full_route_calls.append((tuple(targets), step_m))
        raise AssertionError("unreachable endpoint must not start the full route")

    def reject_motion(*_args, **_kwargs):
        motion_calls.append((_args, _kwargs))
        raise AssertionError("no motion command may be issued")

    def reject_gripper(*_args, **_kwargs):
        gripper_calls.append((_args, _kwargs))
        raise AssertionError("no gripper command may be issued")

    pipeline.motion.precheck_pose_condition = reject_endpoint
    pipeline.motion.precheck_linear_targets = record_full_route
    pipeline.motion.execute_prechecked_segment = reject_motion
    pipeline.gripper.open = reject_gripper
    pipeline.gripper.close = reject_gripper

    result = pipeline.run_once()

    assert result.status is PickStatus.REJECTED
    assert "机械臂正在搜索可达抓取姿态" in result.message
    assert "物体保持不动" in result.message
    assert "未发送任何运动或夹爪命令" in result.message
    assert "position=0.041152 m" in result.message
    assert len(endpoint_calls) == 1
    assert endpoint_calls[0][0].xyz == pytest.approx((-0.15, -0.10, 0.115))
    assert endpoint_calls[0][1] == "pick_2 pregrasp"
    assert full_route_calls == []
    assert motion_calls == []
    assert gripper_calls == []
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []


def test_task_b_pregrasp_condition_passes_before_full_route_precheck():
    config = task_b_config(rounds=1)
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [frame("pick_1_hit", (target_detection(x=0.15, y=-0.15),))]
    )
    observed = []
    original_route_precheck = pipeline.motion.precheck_linear_targets

    def accept_endpoint(pose, *, label):
        assert pipeline.motion.moves == []
        assert pipeline.gripper.events == []
        observed.append(("endpoint", pose, label))
        return 0.02

    def recording_route_precheck(targets, *, step_m):
        assert pipeline.motion.moves == []
        assert pipeline.gripper.events == []
        observed.append(("route", tuple(targets), step_m))
        return original_route_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_pose_condition = accept_endpoint
    pipeline.motion.precheck_linear_targets = recording_route_precheck

    result = pipeline.run_once()

    assert result.success
    assert [entry[0] for entry in observed] == ["endpoint", "route"]
    assert observed[0][2] == "pick_1 pregrasp"
    assert observed[1][1][0] == observed[0][1]
    assert len(observed[1][1]) == 6


def test_task_b_point2_safe_config_uses_taught_depth_and_read_only_backends():
    config = load_config(FORMAL_TASK_B_POINT2_SAFE_CONFIG)

    assert config.vision.backend == "realsense_yolo"
    assert config.motion.backend == "robotmove"
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"
    assert config.adaptive_inverse.enabled is True
    assert config.adaptive_inverse.candidates.use_visual_grasp_z is False
    assert config.adaptive_inverse.candidates.lock_taught_orientation is True
    assert config.adaptive_inverse.candidates.yaw_offsets_deg == (0.0,)
    assert config.adaptive_inverse.candidates.pitch_offsets_deg == (0.0,)
    assert config.adaptive_inverse.candidates.minimum_grasp_flange_z_m == pytest.approx(
        0.166534092
    )
    assert config.task_b.point_grasp_z_m == pytest.approx((0.166534092,))
    assert config.adaptive_inverse.box_clearance.required is False
    assert config.task_b.vertical_grasp_approach_height_m == pytest.approx(0.075)
    assert config.motion.ik_position_tolerance_m == pytest.approx(0.0002)
    assert config.motion.ik_orientation_tolerance_rad == pytest.approx(0.0002)
    assert config.motion.cartesian_joint_step_deg == pytest.approx(0.5)
    assert config.motion.cartesian_joint_acceleration_limit_deg == pytest.approx(
        0.25
    )
    assert config.motion.cartesian_min_jacobian_singular_value == pytest.approx(
        0.01
    )
    assert config.task_b.return_to_initial_after_each_pick is False
    assert config.task_b.return_to_observation_after_each_pick is True
    assert len(config.task_b.point_observation_joints_deg) == 1
    assert config.task_b.point_observation_joints_deg[0] is not None


def test_task_b_restored_fixed_five_point_configs_disable_adaptive_search():
    safe = load_config(FORMAL_TASK_B_FIXED_SAFE_CONFIG)
    live = load_config(FORMAL_TASK_B_FIXED_LIVE_CONFIG)

    expected_offsets = (
        (0.000, 0.000),
        (0.040, 0.000),
        (0.040, -0.015),
        (0.000, -0.030),
        (-0.020, -0.030),
    )
    expected_rpy = (
        (3.029562510, 0.069559687, -0.068061620),
        (3.128682244906, 0.048279522802, 0.365742992703),
        (3.045178411, 0.059528589, -0.397915592),
        (3.083158245, 0.050086616, -0.347881105),
        (3.110701, 0.030298, -0.514942),
    )
    expected_z = (
        0.166534092,
        0.166534092,
        0.166534092,
        0.169903173540,
        0.167454634,
    )

    for config in (safe, live):
        assert config.task.mode is TaskMode.B
        assert config.adaptive_inverse.enabled is False
        assert config.task_b.rounds == 5
        assert config.task_b.return_to_pick is True
        assert config.task_b.return_to_initial_after_each_pick is True
        assert config.task_b.return_to_observation_after_each_pick is False
        assert config.task_b.operator_pause_after_each_pick is True
        assert config.task_b.pick_pose_tolerance_m == pytest.approx(0.15)
        assert config.task_b.require_nearest_reference_point is False
        assert config.task_b.point_assignment_margin_m == pytest.approx(0.0)
        assert config.task_b.point_grasp_z_m == pytest.approx(expected_z)
        assert config.task_b.point_grasp_xy_from_observation == (
            False,
            False,
            False,
            False,
            False,
        )
        assert config.task_b.point_grasp_xy_m == (None,) * 5
        assert config.task_b.point_grasp_xy_offset_m == expected_offsets
        assert config.task_b.point_grasp_jaw_tip_object_offset_m == ()
        for actual, expected in zip(config.task_b.point_grasp_rpy, expected_rpy):
            assert actual == pytest.approx(expected)
        assert config.task_b.point_vertical_grasp_descent == (True,) * 5
        assert config.task_b.point_vertical_grasp_approach_height_m == pytest.approx(
            (None, 0.060, None, None, None)
        )
        assert config.task_b.vertical_grasp_approach_height_m == pytest.approx(0.08)
        assert len(config.task_b.point_observation_joints_deg) == 5
        assert config.motion.backend == "robotmove"
        assert config.motion.ik_position_tolerance_m == pytest.approx(0.0002)
        assert config.motion.ik_orientation_tolerance_rad == pytest.approx(0.0002)
        assert config.motion.cartesian_joint_step_deg == pytest.approx(0.5)
        assert config.motion.cartesian_joint_acceleration_limit_deg == pytest.approx(
            0.25
        )
        assert config.motion.cartesian_min_jacobian_singular_value == pytest.approx(
            0.01
        )
        assert config.motion.cartesian_max_waypoints == 6000

    assert safe.motion.dry_run is True
    assert safe.gripper.backend == "mock"
    assert live.motion.dry_run is False
    assert live.gripper.backend == "ms42dc"

    launcher_text = TASK_B_LAUNCHER.read_text(encoding="utf-8")
    assert "track1_task_b_five_point_adaptive_safe_20261102.yaml" in launcher_text
    assert "track1_task_b_five_point_fixed_live_20261027.yaml" not in launcher_text
    assert "track1_task_b_final_20261020_live.yaml" not in launcher_text


@pytest.mark.parametrize("config_path", FORMAL_TASK_B_POINT2_UNSAFE_CONFIGS)
def test_task_b_point2_visual_depth_accident_configs_fail_closed(config_path):
    with pytest.raises(
        ValueError,
        match=r"support_plane_required=true|minimum_grasp_flange_z_m",
    ):
        load_config(config_path)


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


def test_task_b_returns_to_initial_pose_after_each_point(tmp_path):
    target_q = (0.10, -0.04, 0.03, 0.02, -0.01, 0.05)
    pose_file = write_initial_pose(tmp_path / "initial_pose.yaml", target_q)
    base = task_b_config(rounds=5)
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            initial_pose_file=str(pose_file),
            return_to_initial_after_each_pick=True,
            return_to_initial_after_task=False,
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
        "joint:task_b_return_initial_pick_1",
        "joint:task_b_return_initial_pick_2",
        "joint:task_b_return_initial_pick_3",
        "joint:task_b_return_initial_pick_4",
        "joint:task_b_return_initial_pick_5",
    ]
    for point_number in range(1, 6):
        retreat_index = sequence.index(
            "segment:retreat",
            (point_number - 1) * 7,
        )
        return_index = sequence.index(
            f"joint:task_b_return_initial_pick_{point_number}",
            retreat_index + 1,
        )
        assert return_index == retreat_index + 1


def test_task_b_failure_at_next_point_stops_after_returning_previous_point(tmp_path):
    target_q = (0.10, -0.04, 0.03, 0.02, -0.01, 0.05)
    pose_file = write_initial_pose(tmp_path / "initial_pose.yaml", target_q)
    base = task_b_config(rounds=2)
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            initial_pose_file=str(pose_file),
            return_to_initial_after_each_pick=True,
            return_to_initial_after_task=False,
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
    assert joint_moves == [
        "task_b_initial_pose",
        "task_b_return_initial_pick_1",
    ]
    assert len(pipeline.motion.moves) == 6
    assert pipeline.motion.moves[-1][0] == "retreat"


def test_task_b_initial_return_failure_stops_before_operator_gate(tmp_path):
    pose_file = write_initial_pose(
        tmp_path / "initial_pose.yaml",
        (0.10, -0.04, 0.03, 0.02, -0.01, 0.05),
    )
    base = task_b_config(rounds=2)
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            initial_pose_file=str(pose_file),
            operator_pause_after_each_pick=True,
            return_to_initial_after_each_pick=True,
            return_to_initial_after_task=False,
        ),
    )
    gates = []
    pipeline = build_pipeline(
        config,
        operator_gate=lambda name, completed, total: gates.append(
            (name, completed, total)
        ),
    )
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.15, y=-0.15),)),
            frame("pick_2", (target_detection(x=-0.15, y=-0.10),)),
        ]
    )
    joint_moves = []

    def reject_initial_return(target, *, name, **kwargs):
        joint_moves.append(name)
        if name == "task_b_return_initial_pick_1":
            raise RuntimeError("synthetic initial-return rejection")

    pipeline.motion.move_joints = reject_initial_return

    with pytest.raises(RuntimeError, match="initial pose motion failed"):
        pipeline.run()

    assert joint_moves == [
        "task_b_initial_pose",
        "task_b_return_initial_pick_1",
    ]
    assert gates == []
    assert pipeline.vision.index == 1
    assert len(pipeline.motion.moves) == 6
    assert pipeline.motion.moves[-1][0] == "retreat"


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


def test_task_b_moves_to_each_observation_pose_before_visual_search():
    base = task_b_config(rounds=2)
    observations = (
        (10.0, -5.0, 4.0, 3.0, -2.0, 6.0),
        (8.0, -4.0, 3.0, 2.0, -1.0, 5.0),
    )
    config = replace(
        base,
        motion=replace(
            base.motion,
            workspace_min=(-0.65, -0.65, 0.02),
            workspace_max=(0.65, 0.65, 0.80),
            workspace_margin=0.01,
        ),
        task_b=replace(
            base.task_b,
            point_observation_joints_deg=observations,
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
    assert [name for name, _target in pipeline.motion.joint_moves] == [
        "task_b_observation_pick_1",
        "task_b_observation_pick_2",
    ]
    assert pipeline.motion.joint_moves[0][1] == pytest.approx(
        tuple(math.radians(value) for value in observations[0])
    )
    assert pipeline.motion.joint_moves[1][1] == pytest.approx(
        tuple(math.radians(value) for value in observations[1])
    )


def test_task_b_returns_to_observation_pose_after_each_successful_pick():
    base = task_b_config(rounds=2)
    observations = (
        (10.0, -5.0, 4.0, 3.0, -2.0, 6.0),
        (8.0, -4.0, 3.0, 2.0, -1.0, 5.0),
    )
    config = replace(
        base,
        motion=replace(
            base.motion,
            workspace_min=(-0.65, -0.65, 0.02),
            workspace_max=(0.65, 0.65, 0.80),
            workspace_margin=0.01,
        ),
        task_b=replace(
            base.task_b,
            point_observation_joints_deg=observations,
            return_to_initial_after_each_pick=False,
            return_to_observation_after_each_pick=True,
            return_to_initial_after_task=False,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.15, y=-0.15),)),
            frame("pick_2", (target_detection(x=-0.15, y=-0.10),)),
        ]
    )
    sequence = []
    joint_moves = []
    observation_checks = []
    original_move_joints = pipeline.motion.move_joints
    original_execute = pipeline.motion.execute_prechecked_segment
    original_ensure_observation = pipeline._ensure_task_b_observation_pose

    def record_joint_move(target, *, name, **kwargs):
        original_move_joints(target, name=name, **kwargs)
        sequence.append(f"joint:{name}")
        joint_moves.append((name, np.asarray(target, dtype=float)))

    def record_segment(segment, *, name, speed_mps):
        original_execute(segment, name=name, speed_mps=speed_mps)
        sequence.append(f"segment:{name}")

    def record_observation_check(
        point_index,
        place_name,
        state,
        *,
        move_name=None,
    ):
        check_name = move_name or f"task_b_observation_{place_name}"
        observation_checks.append(
            (point_index, place_name, check_name)
        )
        sequence.append(f"observe:{check_name}")
        return original_ensure_observation(
            point_index,
            place_name,
            state,
            move_name=move_name,
        )

    pipeline.motion.move_joints = record_joint_move
    pipeline.motion.execute_prechecked_segment = record_segment
    pipeline._ensure_task_b_observation_pose = record_observation_check

    report = pipeline.run()

    segment_sequence = [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    expected = []
    for point_number in (1, 2):
        expected.append(f"observe:task_b_observation_pick_{point_number}")
        expected.append(f"joint:task_b_observation_pick_{point_number}")
        expected.extend(f"segment:{name}" for name in segment_sequence)
        expected.append(f"observe:task_b_return_observation_pick_{point_number}")

    assert report.successes == 2
    assert sequence == expected
    assert [name for name, _target in joint_moves] == [
        "task_b_observation_pick_1",
        "task_b_observation_pick_2",
    ]
    assert joint_moves[0][1] == pytest.approx(
        tuple(math.radians(value) for value in observations[0])
    )
    assert joint_moves[1][1] == pytest.approx(
        tuple(math.radians(value) for value in observations[1])
    )
    assert observation_checks == [
        (0, "pick_1", "task_b_observation_pick_1"),
        (0, "pick_1", "task_b_return_observation_pick_1"),
        (1, "pick_2", "task_b_observation_pick_2"),
        (1, "pick_2", "task_b_return_observation_pick_2"),
    ]


def test_task_b_observation_return_failure_prevents_next_acquisition():
    base = task_b_config(rounds=2)
    observation = (10.0, -5.0, 4.0, 3.0, -2.0, 6.0)
    config = replace(
        base,
        motion=replace(
            base.motion,
            workspace_min=(-0.65, -0.65, 0.02),
            workspace_max=(0.65, 0.65, 0.80),
            workspace_margin=0.01,
        ),
        task_b=replace(
            base.task_b,
            point_observation_joints_deg=(observation, observation),
            return_to_initial_after_each_pick=False,
            return_to_observation_after_each_pick=True,
            return_to_initial_after_task=False,
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.15, y=-0.15),)),
            frame("pick_2", (target_detection(x=-0.15, y=-0.10),)),
        ]
    )
    original_ensure_observation = pipeline._ensure_task_b_observation_pose

    def fail_observation_return(
        point_index,
        place_name,
        state,
        *,
        move_name=None,
    ):
        if move_name == "task_b_return_observation_pick_1":
            raise RuntimeError("synthetic observation return rejection")
        return original_ensure_observation(
            point_index,
            place_name,
            state,
            move_name=move_name,
        )

    pipeline._ensure_task_b_observation_pose = fail_observation_return

    with pytest.raises(
        RuntimeError,
        match="return to observation pose failed after 'pick_1'",
    ):
        pipeline.run()

    assert pipeline._task_b_pick_index == 1
    assert pipeline.vision.index == 1


def test_task_b_pipeline_can_use_observation_xy_for_a_vertical_grasp():
    observation_deg = (
        91.405846375,
        74.366790849,
        -18.643502588,
        -35.900559036,
        87.641336807,
        -19.600974643,
    )
    visual_xy = (-0.15564433, 0.52594438)
    grasp_rpy = (3.093678813, 0.128615562, 0.224744345)
    base = task_b_config(rounds=1)
    config = replace(
        base,
        motion=replace(
            base.motion,
            orientation="current",
            workspace_min=(-0.65, -0.65, 0.02),
            workspace_max=(0.65, 0.65, 0.80),
            workspace_margin=0.01,
        ),
        task_b=replace(
            base.task_b,
            place_poses=(
                (*visual_xy, 0.041215741, *grasp_rpy),
            ),
            place_poses_file="",
            place_names=("pick_2",),
            point_grasp_z_m=(0.161534092,),
            point_grasp_xy_from_observation=(True,),
            point_grasp_rpy=(grasp_rpy,),
            point_observation_joints_deg=(observation_deg,),
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [frame("pick_2", (target_detection(x=visual_xy[0], y=visual_xy[1]),))]
    )
    transform = controller_fk(
        tuple(math.radians(value) for value in observation_deg)
    )
    anchor = Pose(
        tuple(float(value) for value in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )

    result = pipeline.run_once()

    assert result.success
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    assert pipeline.motion.moves[0][1] == anchor
    assert pipeline.motion.moves[2][1] == anchor
    assert pipeline.motion.moves[3][1] == anchor
    assert pipeline.motion.moves[5][1] == anchor
    assert pipeline.motion.moves[1][1].xyz[:2] == pytest.approx(anchor.xyz[:2])
    assert pipeline.motion.moves[1][1].xyz[:2] != pytest.approx(visual_xy)
    assert pipeline.motion.moves[1][1].xyz[2] == pytest.approx(0.161534092)
    assert pipeline.motion.moves[1][1].rpy == pytest.approx(grasp_rpy)
    assert pipeline.motion.moves[4][1] == pipeline.motion.moves[1][1]
    assert pipeline.vision.index == 1


def test_observation_pose_failure_stops_before_vision_or_gripper():
    base = task_b_config(rounds=1)
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            point_observation_joints_deg=(
                (10.0, -5.0, 4.0, 3.0, -2.0, 6.0),
            ),
        ),
    )
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(
        [frame("unused", (target_detection(),))]
    )

    def reject_observation_move(_target, *, name, **_kwargs):
        raise RuntimeError(f"synthetic rejection for {name}")

    pipeline.motion.move_joints = reject_observation_move
    result = pipeline.run_once()

    assert result.status == PickStatus.ABORTED
    assert "observation pose" in result.message
    assert pipeline.vision.index == 0
    assert pipeline.gripper.events == []
    assert pipeline.motion.moves == []


def test_task_b_requires_reference_points():
    config = apply_mode(load_config(CONFIG), "B")
    broken = replace(config, task_b=replace(config.task_b, place_poses_file=""))
    try:
        broken.validate()
    except ValueError as exc:
        assert "task B needs" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("task B should require reference points")


def test_task_b_operator_pause_follows_return_to_initial_before_next_point(tmp_path):
    base = apply_mode(load_config(CONFIG), "B")
    pose_file = write_initial_pose(
        tmp_path / "initial_pose.yaml",
        (0.10, -0.04, 0.03, 0.02, -0.01, 0.05),
    )
    config = replace(
        base,
        task_b=replace(
            base.task_b,
            initial_pose_file=str(pose_file),
            place_poses=(
                (0.10, -0.15, 0.035, 3.14159265, 0.0, 0.0),
                (0.15, -0.15, 0.035, 3.14159265, 0.0, 0.0),
            ),
            place_poses_file="",
            rounds=2,
            place_names=("pick_1", "pick_2"),
            next_point_timeout_s=1.0,
            next_point_poll_s=0.001,
            operator_pause_after_each_pick=True,
            return_to_initial_after_each_pick=True,
            return_to_initial_after_task=False,
        ),
        search=replace(
            base.search,
            enabled=False,
            observe_frames=1,
            confirm_hits=1,
        ),
    )
    pauses = []
    sequence = []
    pipeline = build_pipeline(
        config,
        operator_gate=lambda name, completed, total: (
            pauses.append((name, completed, total)),
            sequence.append("operator_gate"),
        ),
    )
    pipeline.vision = ScriptedVision(
        [
            frame("pick_1", (target_detection(x=0.10, y=-0.15),)),
            frame("pick_2", (target_detection(x=0.15, y=-0.15),)),
        ]
    )

    def record_joint_move(target, *, name, **kwargs):
        sequence.append(f"joint:{name}")

    original_execute = pipeline.motion.execute_prechecked_segment

    def record_segment(segment, *, name, speed_mps):
        original_execute(segment, name=name, speed_mps=speed_mps)
        sequence.append(f"segment:{name}")

    pipeline.motion.move_joints = record_joint_move
    pipeline.motion.execute_prechecked_segment = record_segment

    report = pipeline.run()

    assert report.successes == 2
    assert pauses == [("pick_1", 1, 2)]
    assert sequence == [
        "joint:task_b_initial_pose",
        "segment:pregrasp",
        "segment:grasp",
        "segment:lift",
        "segment:transfer",
        "segment:release",
        "segment:retreat",
        "joint:task_b_return_initial_pick_1",
        "operator_gate",
        "segment:pregrasp",
        "segment:grasp",
        "segment:lift",
        "segment:transfer",
        "segment:release",
        "segment:retreat",
        "joint:task_b_return_initial_pick_2",
    ]


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


def test_adaptive_task_b_assigns_banana_by_waist_center_not_bbox_anchor():
    base = apply_mode(load_config(CONFIG), "B")
    config = replace(
        base,
        pose=replace(
            base.pose,
            use_detection_base_xyz=False,
            fallback_camera_to_base=True,
            fixed_camera_to_base_xyz=(0.0, 0.0, 0.0),
        ),
        adaptive_inverse=replace(
            base.adaptive_inverse,
            enabled=True,
            pca=replace(
                base.adaptive_inverse.pca,
                waist_midline_enabled=True,
            ),
        ),
        task_b=replace(
            base.task_b,
            place_poses=(
                (0.056674, 0.417881, 0.035, 3.14159265, 0.0, 0.0),
                (0.034196, 0.486713, 0.035, 3.14159265, 0.0, 0.0),
            ),
            place_poses_file="",
            rounds=2,
            place_names=("pick_4", "pick_5"),
            pick_pose_tolerance_m=0.08,
            point_assignment_margin_m=0.0,
        ),
    )
    pipeline = build_pipeline(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (3.14159265, 0.0, 0.0)),
        message="ready",
    )
    # The detector anchor is exactly at pick_5, while the adaptive Banana
    # waist centre is exactly at pick_4.  Assignment must follow the waist
    # centre because that is the point the grasp planner commands.
    detection = replace(
        target_detection(x=0.034196, y=0.486713),
        object_center_camera_xyz=(0.056674, 0.417881, 0.38),
    )

    accepted, reason = pipeline._filter_task_b_candidates(
        [detection],
        state,
        config.task_b.place_poses[0],
        config.task_b.target_class,
        "pick_4",
    )

    assert accepted == [detection]
    assert "pick_4" in reason

    # The real 2026-11-06 waist-centre observation remains closer to pick_5.
    # The strict nearest-reference safety gate must still reject it at pick_4.
    observed_too_far = replace(
        detection,
        object_center_camera_xyz=(0.058615, 0.459211, 0.38),
    )
    rejected, rejection_reason = pipeline._filter_task_b_candidates(
        [observed_too_far],
        state,
        config.task_b.place_poses[0],
        config.task_b.target_class,
        "pick_4",
    )

    assert rejected == []
    assert "closest to reference point 2" in rejection_reason


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


def test_disabled_nearest_reference_accepts_stable_detection_in_current_window():
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
            require_nearest_reference_point=False,
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
                "banana_inside_current_window",
                (target_detection(x=0.15, y=-0.15),),
            )
        ]
    )

    result = pipeline.run_once()

    assert result.status == PickStatus.SUCCESS
    assert pipeline._task_b_pick_index == 1
    assert pipeline.motion.moves
    assert pipeline.motion.moves[0][1].xyz[:2] == pytest.approx((0.15, -0.15))


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
    assert config.task_b.return_to_initial_after_each_pick is True
    assert config.task_b.return_to_initial_after_task is False
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
        (-0.108008678, 0.417708580),
        (0.056673616879, 0.417881068652),
        (0.034196350, 0.486712736),
    )
    for pose, expected in zip(config.place_poses, expected_xy):
        assert pose[:2] == pytest.approx(expected)
    assert config.task_b.point_grasp_z_m == pytest.approx(
        (0.161534092, 0.161534092, 0.16218029524630417, 0.169903173540, 0.161454634)
    )
    expected_rpy = (
        (3.029562510, 0.069559687, -0.068061620),
        (3.093678813, 0.128615562, 0.224744345),
        (3.045178411, 0.059528589, -0.397915592),
        (3.083158245009, 0.050086616429, -0.347881104760),
        (3.026471987, 0.054535204, -0.656420178),
    )
    for actual, expected in zip(config.task_b.point_grasp_rpy, expected_rpy):
        assert actual == pytest.approx(expected)

    assert config.task_b.point_grasp_xy_from_observation == (
        False,
        False,
        True,
        False,
        False,
    )
    assert config.task_b.point_grasp_xy_m == (
        None,
        None,
        None,
        (0.065803249086, 0.443883453325),
        None,
    )
    assert len(config.task_b.point_observation_joints_deg) == 5
    assert config.task_b.point_observation_joints_deg[1] == pytest.approx(
        (91.405846375, 74.366790849, -18.643502588,
         -35.900559036, 87.641336807, -19.600974643)
    )
    assert config.task_b.point_observation_joints_deg[2] == pytest.approx(
        (88.606279345, 94.897322912, -79.332457850,
         -73.643387215, 84.809611892, 21.237936864)
    )
    assert config.task_b.point_observation_joints_deg[3] == pytest.approx(
        (66.769885390, 93.322318979, -80.589018615,
         -75.774917104, 87.317606471, -3.217390651)
    )

    launcher_text = TASK_B_LAUNCHER.read_text(encoding="utf-8")
    assert "track1_task_b_five_point_adaptive_safe_20261102.yaml" in launcher_text
    assert "track1_task_b_final_20261020_live.yaml" not in launcher_text
    assert "track1_task_b_five_point_fixed_live_20261027.yaml" not in launcher_text
    assert "track1_task_b_point2_safe_20261027.yaml" not in launcher_text
    assert "track1_task_b_20261020.yaml" not in launcher_text

    desktop_text = (
        PROGRAM_ROOT / "任务B_五点位拾放.desktop"
    ).read_text(encoding="utf-8")
    assert "Name=任务B 五点位拾放" in desktop_text
    assert (
        "Exec=/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/"
        "bin/run_task_b_full_live"
    ) in desktop_text
    assert "dry-run + mock" not in desktop_text
    assert "新加卷1" not in desktop_text
    assert "/home/sunshine" not in desktop_text


def test_final_live_launcher_config_is_the_validated_complete_workflow():
    config = load_config(FORMAL_TASK_B_FINAL_LIVE_CONFIG)

    assert config.task.mode is TaskMode.B
    assert config.task_b.rounds == 5
    assert config.task_b.place_names == (
        "pick_1",
        "pick_2",
        "pick_3",
        "pick_4",
        "pick_5",
    )
    assert config.task_b.return_to_pick is True
    assert config.task_b.return_to_initial_after_each_pick is True
    assert config.task_b.return_to_initial_after_task is False
    assert config.task_b.operator_pause_after_each_pick is True
    assert config.task_b.keep_same_target is True
    assert config.task_b.target_class == "8-Plastic_Banana"
    assert config.task_b.next_point_timeout_s == 0.0
    assert config.task_b.place_poses_file.endswith("task_b_place_poses_20261020.yaml")
    assert config.motion.backend == "robotmove"
    assert config.motion.dry_run is False
    assert config.motion.speed_mps == pytest.approx(0.12)
    assert config.motion.joint_speed_rad_s == pytest.approx(0.6)
    assert config.motion.joint_speed_safety_limit_rad_s == pytest.approx(0.7)
    assert config.gripper.backend == "ms42dc"
    assert config.gripper.serial_port == "/dev/motor_serial"
    assert config.gripper.require_feedback is True
    assert config.task_b.point_grasp_z_m == pytest.approx(
        (0.166534092, 0.166534092, 0.16218029524630417, 0.169903173540, 0.167454634)
    )
    # Point 2 deliberately uses the verified observation-posture RPY and a
    # point-only 60 mm approach height to keep the transition from the shared
    # initial observation posture away from a high-pose singularity.
    assert config.task_b.point_grasp_rpy[1] == pytest.approx(
        (3.128682244906, 0.048279522802, 0.365742992703)
    )
    assert config.task_b.point_grasp_xy_from_observation == (
        False,
        False,
        False,
        False,
        False,
    )
    assert config.task_b.point_grasp_xy_m == (None, None, None, None, None)
    assert config.task_b.point_grasp_xy_offset_m == (
        (0.0, 0.0),
        (0.040, 0.0),
        (0.040, -0.015),
        (0.0, -0.030),
        (-0.020, -0.030),
    )
    assert config.task_b.point_grasp_rpy[4] == pytest.approx(
        (3.110701, 0.030298, -0.514942)
    )
    assert config.task_b.point_vertical_grasp_descent == (
        True,
        True,
        True,
        True,
        True,
    )
    assert config.task_b.point_vertical_grasp_approach_height_m == pytest.approx(
        (None, 0.060, None, None, None)
    )
    assert config.task_b.vertical_grasp_approach_height_m == pytest.approx(0.08)
    assert len(config.task_b.point_observation_joints_deg) == 5
    initial_observation = (
        82.656979106,
        54.310204210,
        -32.193324618,
        -53.452584322,
        84.445689922,
        -2.057875745,
    )
    for observation in config.task_b.point_observation_joints_deg:
        assert observation == pytest.approx(initial_observation)
    assert config.search.enabled is False
    assert config.search.observation_offsets_xy_m == (
        (0.00, 0.00),
        (0.08, 0.00),
        (0.00, -0.08),
    )


def test_resume2_final_live_config_uses_vertical_pick2_and_raised_heights():
    config = load_config(FORMAL_TASK_B_RESUME2_FINAL_LIVE_CONFIG)

    assert config.task_b.place_names == (
        "pick_2",
        "pick_3",
        "pick_4",
        "pick_5",
    )
    assert config.task_b.point_grasp_z_m == pytest.approx(
        (0.166534092, 0.172258106, 0.169424007, 0.167454634)
    )
    assert config.task_b.point_grasp_rpy[0] == pytest.approx(
        (3.128682244906, 0.048279522802, 0.365742992703)
    )
    assert config.task_b.point_grasp_xy_from_observation == (
        True,
        False,
        False,
        True,
    )
    assert config.task_b.point_grasp_xy_m == (
        None,
        None,
        None,
        None,
    )
    assert config.task_b.point_grasp_rpy[3] == pytest.approx(
        (3.110701, 0.030298, -0.514942)
    )
    assert config.task_b.point_vertical_grasp_descent == (
        True,
        True,
        True,
        True,
    )


def test_fixed_live_config_completes_all_five_points_in_mock(monkeypatch):
    base = load_config(FORMAL_TASK_B_FIXED_LIVE_CONFIG)
    config = replace(
        base,
        task_b=replace(base.task_b, operator_pause_after_each_pick=False),
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
        "task_b_observation_pick_1",
        "task_b_return_initial_pick_1",
        "task_b_observation_pick_2",
        "task_b_return_initial_pick_2",
        "task_b_observation_pick_3",
        "task_b_return_initial_pick_3",
        "task_b_observation_pick_4",
        "task_b_return_initial_pick_4",
        "task_b_observation_pick_5",
        "task_b_return_initial_pick_5",
    ]
    assert len(execution_speeds) == 30
    assert execution_speeds == pytest.approx([0.01] * 30)

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
        observation_q = tuple(
            math.radians(value)
            for value in config.task_b.point_observation_joints_deg[index]
        )
        transform = controller_fk(observation_q)
        anchor = Pose(
            tuple(float(value) for value in transform[:3, 3]),
            matrix_to_rpy(transform[:3, :3]),
        )
        grasp_pose = cycle[1][1]
        release_pose = cycle[4][1]
        explicit_grasp_xy = (
            index < len(config.task_b.point_grasp_xy_m)
            and config.task_b.point_grasp_xy_m[index] is not None
        )
        if explicit_grasp_xy:
            expected_xy = config.task_b.point_grasp_xy_m[index]
        elif config.task_b.point_grasp_xy_from_observation[index]:
            expected_xy = anchor.xyz[:2]
        else:
            expected_xy = reference[:2]
        if index < len(config.task_b.point_grasp_xy_offset_m):
            offset = config.task_b.point_grasp_xy_offset_m[index]
            grasp_rpy = config.task_b.point_grasp_rpy[index]
            assert grasp_rpy is not None
            base_xy_delta = euler_to_matrix(grasp_rpy) @ np.asarray(
                (float(offset[0]), float(offset[1]), 0.0)
            )
            expected_xy = (
                float(expected_xy[0]) + float(base_xy_delta[0]),
                float(expected_xy[1]) + float(base_xy_delta[1]),
            )

        if config.task_b.point_vertical_grasp_descent[index]:
            # The vertical route begins at the point-specific approach height
            # above the grasp and moves directly downward with the same XY/RPY.
            approach_height_m = config.task_b.vertical_grasp_approach_height_m
            if index < len(
                config.task_b.point_vertical_grasp_approach_height_m
            ):
                point_height_m = (
                    config.task_b.point_vertical_grasp_approach_height_m[index]
                )
                if point_height_m is not None:
                    approach_height_m = point_height_m
            expected_high_pose = Pose(
                (
                    *expected_xy,
                    config.task_b.point_grasp_z_m[index]
                    + approach_height_m,
                ),
                config.task_b.point_grasp_rpy[index],
            )
        elif explicit_grasp_xy:
            high_z = max(
                anchor.xyz[2],
                config.task_b.point_grasp_z_m[index]
                + config.motion.approach_height,
                config.task_b.point_grasp_z_m[index]
                + config.motion.lift_height,
            )
            expected_high_pose = Pose(
                (*expected_xy, high_z),
                config.task_b.point_grasp_rpy[index],
            )
        else:
            expected_high_pose = anchor
        assert cycle[0][1] == expected_high_pose
        assert cycle[2][1] == expected_high_pose
        assert cycle[3][1] == expected_high_pose
        assert cycle[5][1] == expected_high_pose
        assert grasp_pose.xyz[:2] == pytest.approx(expected_xy)
        assert grasp_pose.xyz[2] == pytest.approx(
            config.task_b.point_grasp_z_m[index]
        )
        assert grasp_pose.rpy == pytest.approx(config.task_b.point_grasp_rpy[index])
        if config.task_b.point_vertical_grasp_descent[index]:
            assert cycle[0][1].xyz[:2] == pytest.approx(grasp_pose.xyz[:2])
            assert cycle[0][1].rpy == pytest.approx(grasp_pose.rpy)
        assert release_pose.xyz == grasp_pose.xyz


def test_d405_optical_center_is_65_mm_on_flange_negative_y():
    transform = load_extrinsics(
        ROOT / "config" / "camera_to_flange.d405_assumed.yaml"
    )
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        transform[:3, 3], np.asarray((0.0, -0.065, 0.0)), atol=1e-12
    )

    installed = load_extrinsics(
        PROGRAM_ROOT / "config" / "camera_to_flange.d405_assumed.yaml"
    )
    np.testing.assert_allclose(installed, transform, atol=1e-12)
