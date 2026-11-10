from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np
import pytest

from track1_grasp.collision import (
    BoxClearanceConfig,
    BoxObstacleConfig,
    GEOMETRY_SOURCE_FIELD_MEASURED,
    check_box_clearance,
)
from track1_grasp.config import load_config
from track1_grasp.domain import PickStatus
from track1_grasp.factory import build_pick_pipeline
from track1_grasp.kinematics import rpy_to_matrix
from track1_grasp.planner import PosePlanner
from track1_grasp.tools.check_task_b_arbitrary_matrix import (
    DEFAULT_INITIAL_JOINTS_DEG,
    MatrixSample,
    _initial_state,
    _sample_detection,
)


ROOT = Path(__file__).resolve().parents[3]
ARBITRARY_MOCK_CONFIG = ROOT / "config" / "track1_task_b_arbitrary_mock.yaml"
IDENTITY = np.eye(3)


def arbitrary_geometry() -> BoxClearanceConfig:
    return load_config(ARBITRARY_MOCK_CONFIG).adaptive_inverse.box_clearance


def simple_clearance_config(
    *,
    footprint: tuple[tuple[float, float], ...] = (
        (-0.06, -0.02),
        (0.06, -0.02),
        (0.06, 0.02),
        (-0.06, 0.02),
    ),
    obstacles: tuple[BoxObstacleConfig, ...] = (),
) -> BoxClearanceConfig:
    return BoxClearanceConfig(
        required=True,
        geometry_validated=True,
        geometry_source=GEOMETRY_SOURCE_FIELD_MEASURED,
        center_xy_m=(0.0, 0.0),
        interior_size_m=(2.0, 2.0),
        gripper_footprint_flange_xy_m=footprint,
        obstacles_m=obstacles,
    )


def test_center_grasp_clears_synthetic_box_geometry():
    result = check_box_clearance(
        arbitrary_geometry(),
        flange_xy_m=(0.0, 0.10),
        rotation=IDENTITY,
    )

    assert result.allowed
    assert result.min_wall_clearance_m is not None
    assert result.min_wall_clearance_m > 0.08
    assert result.min_obstacle_clearance_m is not None
    assert result.min_obstacle_clearance_m > 0.02


def test_grasp_near_box_wall_is_rejected_before_obstacle_checks():
    result = check_box_clearance(
        arbitrary_geometry(),
        flange_xy_m=(0.65, 0.10),
        rotation=IDENTITY,
    )

    assert not result.allowed
    assert "wall clearance" in result.reason
    assert result.min_wall_clearance_m is not None


@pytest.mark.parametrize(
    ("xy", "corner"),
    (
        ((0.64, 0.71), "northwest"),
        ((-0.64, 0.71), "northeast"),
        ((0.64, -0.51), "southwest"),
        ((-0.64, -0.51), "southeast"),
    ),
)
def test_all_four_corner_l_blocks_reject_an_overlapping_grasp(xy, corner):
    geometry = replace(arbitrary_geometry(), wall_clearance_m=0.0)

    result = check_box_clearance(
        geometry,
        flange_xy_m=xy,
        rotation=IDENTITY,
    )

    assert not result.allowed
    assert "obstacle clearance" in result.reason
    assert corner in result.reason


@pytest.mark.parametrize(
    ("xy", "boss"),
    (
        ((0.0, 0.755), "north_edge_boss"),
        ((0.0, -0.555), "south_edge_boss"),
    ),
)
def test_north_and_south_edge_bosses_reject_an_overlapping_grasp(xy, boss):
    geometry = replace(arbitrary_geometry(), wall_clearance_m=0.0)

    result = check_box_clearance(
        geometry,
        flange_xy_m=xy,
        rotation=IDENTITY,
    )

    assert not result.allowed
    assert "obstacle clearance" in result.reason
    assert boss in result.reason


def test_obstacle_vertex_near_footprint_edge_is_detected():
    geometry = simple_clearance_config(
        footprint=((-0.60, -0.10), (0.60, -0.10), (0.00, -0.20)),
        obstacles=(
            BoxObstacleConfig(
                name="center_blocker",
                vertices_m=((-0.01, 0.10), (0.01, 0.10), (0.00, 0.22)),
            ),
        ),
    )
    geometry = replace(geometry, obstacle_clearance_m=0.50)

    result = check_box_clearance(
        geometry,
        flange_xy_m=(0.0, 0.0),
        rotation=IDENTITY,
    )

    assert not result.allowed
    assert "center_blocker" in result.reason
    assert result.min_obstacle_clearance_m == pytest.approx(0.20)


def test_crossing_convex_obstacle_edges_are_reported_as_zero_distance():
    geometry = simple_clearance_config(
        footprint=((0.0, 0.30), (0.26, -0.15), (-0.26, -0.15)),
        obstacles=(
            BoxObstacleConfig(
                name="crossing_blocker",
                vertices_m=((0.0, -0.30), (0.26, 0.15), (-0.26, 0.15)),
            ),
        ),
    )

    result = check_box_clearance(
        geometry,
        flange_xy_m=(0.0, 0.0),
        rotation=IDENTITY,
    )

    assert not result.allowed
    assert "crossing_blocker" in result.reason
    assert result.min_obstacle_clearance_m == pytest.approx(0.0)


def test_full_3d_rotation_projects_the_non_symmetric_footprint():
    geometry = simple_clearance_config()

    identity = check_box_clearance(
        geometry,
        flange_xy_m=(0.0, 0.0),
        rotation=IDENTITY,
    )
    quarter_turn = check_box_clearance(
        geometry,
        flange_xy_m=(0.0, 0.0),
        rotation=rpy_to_matrix((0.0, 0.0, pi * 0.5)),
    )

    identity_points = np.asarray(identity.footprint_base_xy_m)
    quarter_turn_points = np.asarray(quarter_turn.footprint_base_xy_m)
    assert identity.allowed and quarter_turn.allowed
    assert np.ptp(identity_points[:, 0]) == pytest.approx(0.12)
    assert np.ptp(identity_points[:, 1]) == pytest.approx(0.04)
    assert np.ptp(quarter_turn_points[:, 0]) == pytest.approx(0.04)
    assert np.ptp(quarter_turn_points[:, 1]) == pytest.approx(0.12)


def test_non_centered_flange_footprint_is_projected_without_recentering():
    footprint = (
        (0.010, 0.020),
        (0.040, 0.020),
        (0.040, 0.030),
        (0.010, 0.030),
    )
    geometry = simple_clearance_config(footprint=footprint)

    result = check_box_clearance(
        geometry,
        flange_xy_m=(0.10, -0.20),
        rotation=IDENTITY,
    )

    assert result.allowed
    assert np.asarray(result.footprint_base_xy_m) == pytest.approx(
        np.asarray(footprint) + np.asarray((0.10, -0.20))
    )


def test_planner_skips_first_yaw_collision_and_keeps_next_safe_yaw():
    base = load_config(ARBITRARY_MOCK_CONFIG)
    obstacle = BoxObstacleConfig(
        name="east_candidate_blocker",
        vertices_m=(
            (0.115, -0.10),
            (0.130, -0.10),
            (0.130, 0.30),
            (0.115, 0.30),
        ),
    )
    geometry = BoxClearanceConfig(
        required=True,
        geometry_validated=True,
        geometry_source=base.adaptive_inverse.box_clearance.geometry_source,
        center_xy_m=(0.0, 0.0),
        interior_size_m=(1.40, 1.40),
        wall_clearance_m=0.08,
        obstacle_clearance_m=0.01,
        gripper_footprint_flange_xy_m=(
            base.adaptive_inverse.box_clearance.gripper_footprint_flange_xy_m
        ),
        obstacles_m=(obstacle,),
    )
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            box_clearance=geometry,
            candidates=replace(
                base.adaptive_inverse.candidates,
                max_count=2,
                yaw_offsets_deg=(0.0, 90.0),
            ),
        ),
    )
    config.validate()
    _q_initial, state = _initial_state(DEFAULT_INITIAL_JOINTS_DEG)
    planner = PosePlanner(config)
    sample = MatrixSample("box_yaw_fallback", 0.05, 0.10, 0.0, 0.0, 0.0)
    detection = _sample_detection(planner, state, sample)

    first = check_box_clearance(
        geometry,
        flange_xy_m=(0.05, 0.10),
        rotation=rpy_to_matrix((pi, 0.0, 0.0)),
    )
    second = check_box_clearance(
        geometry,
        flange_xy_m=(0.05, 0.10),
        rotation=rpy_to_matrix((pi, 0.0, pi * 0.5)),
    )
    candidates = planner.plan_adaptive_candidates(detection, state)

    assert not first.allowed
    assert second.allowed
    assert len(candidates) == 1
    assert candidates[0].yaw_offset_deg == pytest.approx(90.0)
    assert candidates[0].plan.grasp_pose.xyz[:2] == pytest.approx((0.05, 0.10))


def test_all_wall_collisions_issue_no_motion_or_gripper_command():
    base = load_config(ARBITRARY_MOCK_CONFIG)
    geometry = replace(
        base.adaptive_inverse.box_clearance,
        wall_clearance_m=0.65,
    )
    config = replace(
        base,
        adaptive_inverse=replace(
            base.adaptive_inverse,
            box_clearance=geometry,
        ),
    )
    config.validate()
    pipeline = build_pick_pipeline(config)

    result = pipeline.run_once()

    assert result.status is PickStatus.REJECTED
    event_text = " | ".join(event.message for event in result.events)
    assert "wall clearance" in event_text
    assert pipeline.motion.moves == []
    assert pipeline.motion.joint_moves == []
    assert pipeline.gripper.events == []
    assert pipeline._task_b_completed_xy == []


@pytest.mark.parametrize(
    ("geometry", "message"),
    (
        (
            BoxClearanceConfig(required=True),
            "required but complete box geometry",
        ),
        (
            BoxClearanceConfig(
                required=True,
                geometry_validated=False,
                geometry_source=GEOMETRY_SOURCE_FIELD_MEASURED,
                center_xy_m=(0.0, 0.0),
                interior_size_m=(1.0, 1.0),
                gripper_footprint_flange_xy_m=(
                    (-0.01, -0.01),
                    (0.01, -0.01),
                    (0.01, 0.01),
                    (-0.01, 0.01),
                ),
            ),
            "geometry_validated must be true",
        ),
    ),
)
def test_missing_or_unvalidated_required_geometry_fails_closed(
    geometry,
    message,
):
    with pytest.raises(ValueError, match=message):
        check_box_clearance(
            geometry,
            flange_xy_m=(0.0, 0.0),
            rotation=IDENTITY,
        )
