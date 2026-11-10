from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from track1_grasp.collision import GEOMETRY_SOURCE_FIELD_MEASURED
from track1_grasp.config import load_config
from track1_grasp.domain import Pose, RobotState, TaskMode
from track1_grasp.factory import build_pick_pipeline
from track1_grasp.planner import PosePlanner, transform_point
from track1_grasp.tools.check_task_b_arbitrary_matrix import build_random_samples
from track1_grasp.vision import MockVisionSource


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "track1_task_b_arbitrary_tilted_mock_20261026.yaml"
LEGACY_VERTICAL_CONFIG = ROOT / "config" / "track1_task_b_arbitrary_mock.yaml"


def tilted_config():
    config = load_config(CONFIG)
    config.validate()
    return config


def test_tilted_mock_config_is_offline_and_has_no_remembered_points():
    config = tilted_config()

    assert config.task.mode is TaskMode.B
    assert config.motion.backend == "mock"
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"
    assert config.adaptive_inverse.enabled is True
    assert config.adaptive_inverse.arbitrary_placement is True
    assert config.adaptive_inverse.tilted_visual_grasp is True
    assert config.adaptive_inverse.candidates.pitch_offsets_deg == pytest.approx(
        (-20.0, 20.0, -30.0, 30.0, -16.0, 16.0, 0.0)
    )
    assert config.adaptive_inverse.candidates.max_count == 35
    assert config.adaptive_inverse.box_clearance.geometry_source != (
        GEOMETRY_SOURCE_FIELD_MEASURED
    )
    assert config.adaptive_inverse.box_clearance.geometry_source == (
        "offline_mock_synthetic_test_only"
    )

    assert config.task_b.place_poses == ()
    assert config.task_b.place_poses_file == ""
    assert config.task_b.point_grasp_xy_m == ()
    assert config.task_b.point_grasp_rpy == ()
    assert config.task_b.point_observation_joints_deg == ()

    scenario = Path(config.vision.scenario_file)
    payload = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    detections = [
        detection
        for frame in payload["frames"]
        for detection in frame.get("detections", ())
    ]
    assert len(detections) == 6
    assert all("world_xyz" in detection for detection in detections)
    assert all("camera_xyz" not in detection for detection in detections)


def test_legacy_vertical_config_did_not_enable_tilted_fallback():
    config = load_config(LEGACY_VERTICAL_CONFIG)
    assert config.adaptive_inverse.tilted_visual_grasp is False
    assert config.adaptive_inverse.candidates.pitch_offsets_deg == pytest.approx(
        (0.0,)
    )


def test_dynamic_mock_reprojects_world_target_through_current_flange_pose(tmp_path):
    config = tilted_config()
    scenario = tmp_path / "dynamic_projection.yaml"
    scenario.write_text(
        yaml.safe_dump(
            {
                "frames": [
                    {
                        "frame_id": "dynamic",
                        "detections": [
                            {
                                "label": "8-Plastic_Banana",
                                "instance_id": "dynamic_1",
                                "confidence": 0.95,
                                "bbox": [220, 180, 360, 300],
                                "world_xyz": [0.081, 0.493, -0.038478],
                                "world_major_axis": [
                                    0.816513,
                                    0.571729,
                                    0.080199,
                                ],
                                "world_minor_axis": [
                                    -0.573576,
                                    0.819152,
                                    0.0,
                                ],
                                "major_axis_length_m": 0.12,
                                "minor_axis_length_m": 0.04,
                                "pose_confidence": 0.95,
                                "graspability": 1.0,
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source = MockVisionSource(
        replace(
            config.vision,
            scenario_file=str(scenario),
            loop_scenario=True,
        )
    )
    current = [
        RobotState(
            ready=True,
            pose=Pose((0.0, 0.0, 0.30), (np.pi, 0.0, 0.0)),
            message="mock-ready",
            joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            controller_state="MockReady",
            fault_flag=0,
        )
    ]
    source.bind_robot_state_provider(
        lambda: current[0],
        extrinsics_file=config.pose.extrinsics_file,
    )
    planner = PosePlanner(config)

    first = source.capture().detections[0]
    second_state = RobotState(
        ready=True,
        pose=Pose((0.035, -0.022, 0.355), (2.91, 0.12, -0.18)),
        message="mock-ready",
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        controller_state="MockReady",
        fault_flag=0,
    )
    current[0] = second_state
    second = source.capture().detections[0]

    world = np.array((0.081, 0.493, -0.038478), dtype=float)
    major_world = np.array((0.816513, 0.571729, 0.080199), dtype=float)
    # Restore the first state only for the arithmetic assertion; both captured
    # frames already hold their own projected values.
    first_state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), (np.pi, 0.0, 0.0)),
        message="mock-ready",
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        controller_state="MockReady",
        fault_flag=0,
    )
    first_transform = planner.base_camera_transform(first_state)
    second_transform = planner.base_camera_transform(second_state)

    np.testing.assert_allclose(
        transform_point(first_transform, first.camera_xyz), world, atol=1e-10
    )
    np.testing.assert_allclose(
        transform_point(second_transform, second.camera_xyz), world, atol=1e-10
    )
    np.testing.assert_allclose(
        first_transform[:3, :3] @ np.asarray(first.major_axis_camera),
        major_world,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        second_transform[:3, :3] @ np.asarray(second.major_axis_camera),
        major_world,
        atol=1e-10,
    )
    assert first.camera_xyz != pytest.approx(second.camera_xyz)
    assert first.depth_m == pytest.approx(first.camera_xyz[2])
    assert second.depth_m == pytest.approx(second.camera_xyz[2])
    assert first.attributes["mock_world_projection"] is True
    assert second.attributes["mock_world_projection"] is True


def test_dynamic_mock_fails_closed_without_current_flange_pose(tmp_path):
    config = tilted_config()
    scenario = tmp_path / "unbound_dynamic.yaml"
    scenario.write_text(
        yaml.safe_dump(
            {
                "frames": [
                    {
                        "frame_id": "unbound",
                        "detections": [
                            {
                                "label": "8-Plastic_Banana",
                                "confidence": 0.95,
                                "bbox": [0, 0, 10, 10],
                                "world_xyz": [0.0, 0.5, -0.038],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source = MockVisionSource(
        replace(config.vision, scenario_file=str(scenario))
    )

    with pytest.raises(RuntimeError, match="no robot pose provider"):
        source.capture()


def test_complete_tilted_mock_runs_five_of_five_with_dynamic_observations():
    config = tilted_config()
    pipeline = build_pick_pipeline(config)

    report = pipeline.run()

    assert report.requested_picks == 5
    assert report.attempts == 5
    assert report.successes == 5
    assert all(result.success for result in report.results)
    assert [result.place_name for result in report.results] == [
        f"arbitrary_round_{index}" for index in range(1, 6)
    ]
    assert len(pipeline.motion.moves) == 30
    assert len(pipeline.gripper.events) == 11
    assert pipeline.motion.get_state().fault is False
    assert pipeline.motion.dry_run is True
    assert pipeline._task_b_completed_xy == []


def test_seeded_random_matrix_is_reproducible_and_stays_inside_bounds():
    kwargs = dict(
        count=12,
        seed=20261026,
        support_z_m=-0.038478,
        x_range_m=(-0.14, 0.06),
        y_range_m=(0.40, 0.637),
        z_offset_range_mm=(-3.0, 3.0),
        yaw_range_deg=(-180.0, 180.0),
        axis_tilt_range_deg=(-6.0, 6.0),
    )
    first = build_random_samples(**kwargs)
    second = build_random_samples(**kwargs)

    assert first == second
    assert len(first) == 12
    assert len({sample.sample_id for sample in first}) == 12
    for sample in first:
        assert -0.14 <= sample.x_m <= 0.06
        assert 0.40 <= sample.y_m <= 0.637
        assert -0.041478 <= sample.z_m <= -0.035478
        assert -180.0 <= sample.yaw_deg <= 180.0
        assert -6.0 <= sample.axis_tilt_deg <= 6.0
