from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from track1_grasp.grasp_quality import (
    GraspCandidate,
    GraspScoringConfig,
    ParallelJawGraspScorer,
    load_scoring_config,
    nms_grasps,
)
from track1_grasp.tools.score_grasp_quality import (
    build_report,
    synthetic_demo_scene,
)


IDENTITY_3X3 = np.eye(3, dtype=float)
IDENTITY = tuple(float(value) for value in IDENTITY_3X3.reshape(-1))
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "grasp_quality_ms42dc_estimated.yaml"
)


def cube_surface(
    center: tuple[float, float, float],
    half_size: tuple[float, float, float],
    step_m: float = 0.0025,
) -> np.ndarray:
    axes = [
        np.arange(center[index] - half_size[index], center[index] + half_size[index] + 1e-9, step_m)
        for index in range(3)
    ]
    points: list[np.ndarray] = []
    for face_axis in range(3):
        first_axis, second_axis = [axis for axis in range(3) if axis != face_axis]
        for first in axes[first_axis]:
            for second in axes[second_axis]:
                for sign in (-1.0, 1.0):
                    point = np.zeros(3, dtype=float)
                    point[face_axis] = (
                        center[face_axis] + sign * half_size[face_axis]
                    )
                    point[first_axis] = first
                    point[second_axis] = second
                    points.append(point)
    return np.asarray(points)


def center_cube_points() -> np.ndarray:
    # The object is entirely between the finger base (local X=0) and tips
    # (local X=depth), which is the valid cavity for this score convention.
    return cube_surface((0.025, 0.0, 0.0), (0.020, 0.030, 0.030))


def candidate(
    *,
    candidate_id: str = "center",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, ...] = IDENTITY,
    width_m: float = 0.064,
) -> GraspCandidate:
    return GraspCandidate(
        candidate_id=candidate_id,
        translation_m=translation,
        rotation=rotation,
        width_m=width_m,
        height_m=0.060,
        depth_m=0.050,
        tolerance_m=0.010,
        network_score=0.9,
    )


def test_normal_antipodal_cube_grasp_is_feasible_without_false_collision():
    result = ParallelJawGraspScorer().score(
        candidate(),
        center_cube_points(),
        obstacle_points=None,
    )

    assert result.feasible
    assert result.quality > 0.8
    assert result.contact_left_points > 100
    assert result.contact_right_points > 100
    assert result.antipodal_score == pytest.approx(1.0)
    assert result.measured_width_m == pytest.approx(0.060)
    assert result.collision_voxels == 0
    assert result.left_finger_iou == 0.0
    assert result.right_finger_iou == 0.0
    assert result.target_penetration_voxels == 0
    assert not result.reason_codes


def test_target_penetration_is_not_reported_as_obstacle_collision():
    result = ParallelJawGraspScorer().score(
        candidate(width_m=0.055),
        center_cube_points(),
        obstacle_points=None,
    )

    assert not result.feasible
    assert result.target_penetration_voxels > 0
    assert "target_penetration" in result.reason_codes
    assert result.collision_voxels == 0
    assert result.left_finger_iou == 0.0
    assert result.right_finger_iou == 0.0


def test_empty_grasp_is_rejected():
    shifted_target = cube_surface((0.025, 0.120, 0.0), (0.020, 0.030, 0.030))

    result = ParallelJawGraspScorer().score(candidate(), shifted_target)

    assert not result.feasible
    assert result.inner_points == 0
    assert "empty_grasp" in result.reason_codes
    assert "missing_left_contact" in result.reason_codes
    assert "missing_right_contact" in result.reason_codes


def test_obstacle_inside_left_finger_rejects_candidate():
    obstacle = np.asarray([[0.025, -0.037, 0.0]], dtype=float)

    result = ParallelJawGraspScorer().score(
        candidate(),
        center_cube_points(),
        obstacle_points=obstacle,
    )

    assert not result.feasible
    assert result.collision_voxels == 1
    assert result.left_finger_iou > 0.0
    assert result.obstacle_clearance_m == pytest.approx(0.0)
    assert "left_finger_collision" in result.reason_codes


def test_candidate_above_usable_opening_is_rejected():
    result = ParallelJawGraspScorer().score(
        candidate(width_m=0.151),
        center_cube_points(),
    )

    assert not result.feasible
    assert result.opening_margin_m < 0.0
    assert "exceeds_usable_opening" in result.reason_codes


def test_graspnet_17_float_array_round_trip():
    grasp = np.zeros(17, dtype=float)
    grasp[:4] = (0.875, 0.064, 0.060, 0.050)
    grasp[4:13] = IDENTITY_3X3.reshape(-1)
    grasp[13:16] = (0.10, -0.02, 0.30)
    grasp[16] = 7

    parsed = GraspCandidate.from_graspnet_array(grasp, tolerance_m=0.012)

    assert parsed.network_score == pytest.approx(0.875)
    assert parsed.width_m == pytest.approx(0.064)
    assert parsed.height_m == pytest.approx(0.060)
    assert parsed.depth_m == pytest.approx(0.050)
    assert parsed.translation_m == pytest.approx((0.10, -0.02, 0.30))
    assert np.asarray(parsed.rotation).reshape(3, 3) == pytest.approx(IDENTITY_3X3)
    assert parsed.tolerance_m == pytest.approx(0.012)
    assert parsed.object_id == 7


def test_3d_nms_suppresses_near_duplicate_and_keeps_distinct_grasp():
    base = ParallelJawGraspScorer().score(candidate(), center_cube_points())
    assert base.feasible

    first = replace(base, candidate=candidate(candidate_id="first"), quality=0.90)
    duplicate = replace(
        base,
        candidate=candidate(candidate_id="duplicate", translation=(0.010, 0.0, 0.0)),
        quality=0.80,
    )
    distinct = replace(
        base,
        candidate=candidate(candidate_id="distinct", translation=(0.100, 0.0, 0.0)),
        quality=0.70,
    )

    kept, suppressed = nms_grasps([duplicate, distinct, first])

    assert [item.candidate.candidate_id for item in kept] == ["first", "distinct"]
    assert len(suppressed) == 1
    assert suppressed[0].candidate_id == "duplicate"
    assert suppressed[0].kept_candidate_id == "first"


def test_rotated_candidate_uses_local_gripper_frame():
    angle = np.deg2rad(90.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    local_points = center_cube_points()
    world_points = local_points @ rotation.T

    result = ParallelJawGraspScorer().score(
        candidate(rotation=tuple(float(value) for value in rotation.reshape(-1))),
        world_points,
    )

    assert result.feasible
    assert result.contact_left_points > 100
    assert result.contact_right_points > 100


def test_scoring_config_loads_geometry_and_thresholds(tmp_path):
    path = tmp_path / "quality.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "test_geometry",
                "validated": False,
                "gripper": {
                    "max_opening_m": 0.120,
                    "finger_length_m": 0.070,
                    "finger_thickness_m": 0.012,
                },
                "scoring": {
                    "voxel_size_m": 0.003,
                    "target_penetration_voxel_hits": 5,
                    "nms_translation_m": 0.040,
                    "nms_rotation_deg": 25.0,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_scoring_config(path)

    assert isinstance(config, GraspScoringConfig)
    assert config.name == "test_geometry"
    assert not config.validated
    assert config.geometry.max_opening_m == pytest.approx(0.120)
    assert config.geometry.finger_length_m == pytest.approx(0.070)
    assert config.geometry.finger_thickness_m == pytest.approx(0.012)
    assert config.voxel_size_m == pytest.approx(0.003)
    assert config.target_penetration_voxel_hits == 5
    assert config.nms_translation_m == pytest.approx(0.040)
    assert config.nms_rotation_deg == pytest.approx(25.0)


def test_bundled_ms42dc_config_is_explicitly_unvalidated():
    config = load_scoring_config(CONFIG_PATH)

    assert config.name == "ms42dc_estimated_offline_v1"
    assert config.validated is False
    assert config.geometry.max_opening_m == pytest.approx(0.140)

def test_offline_demo_report_scores_and_applies_nms():
    report = build_report(
        synthetic_demo_scene(),
        load_scoring_config(CONFIG_PATH),
    )

    assert report["counts"] == {
        "candidates": 5,
        "feasible": 2,
        "infeasible": 3,
        "selected_after_nms": 1,
        "suppressed_by_nms": 1,
    }
    assert report["selected"][0]["candidate"]["candidate_id"] == "demo_good"
    reasons = {
        item["candidate"]["candidate_id"]: item["reason_codes"]
        for item in report["results"]
    }
    assert reasons["demo_collision"] == ["left_finger_collision"]
    assert reasons["demo_narrow"] == ["target_penetration"]
