"""Regression tests for the controller-matched S3-60-G2 description."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PACKAGE_ROOT / "scripts" / "validate_kinematics.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("estun_description_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def test_generated_artifacts_exist_and_are_well_formed():
    urdf = PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf"
    xacro = PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf.xacro"
    assert urdf.is_file()
    assert xacro.is_file()
    assert ET.parse(urdf).getroot().tag == "robot"
    assert ET.parse(xacro).getroot().tag == "robot"


def test_urdf_matches_controller_dh_for_random_configurations():
    report = validator.validate(
        urdf_path=PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf",
        kinematics_path=validator.DEFAULT_KINEMATICS,
        camera_path=validator.DEFAULT_CAMERA,
        samples=256,
    )
    assert report["status"] == "pass", report["failures"]
    assert report["kdl_joint_count"] == 6
    assert report["kdl_chain_joints"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]


def test_known_on_site_configuration_matches_controller():
    kinematics = validator.import_kinematics(validator.DEFAULT_KINEMATICS)
    q = np.radians([90.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    actual = validator.urdf_fk(PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf", q)
    expected = np.asarray(kinematics.controller_fk(q), dtype=float)
    assert np.linalg.norm(actual[:3, 3] - expected[:3, 3]) < 1e-8
    assert validator.rotation_error(actual, expected) < 1e-8
    assert np.linalg.norm(actual[:3, 3] - [-0.106964, 0.123500, 0.751726]) < 1e-8


def test_flange_is_a_zero_offset_child_of_link6():
    root = ET.parse(PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf").getroot()
    joint = next(j for j in root.findall("joint") if j.get("name") == "flange_fixed")
    assert joint.get("type") == "fixed"
    assert joint.find("parent").get("link") == "link6"
    assert joint.find("child").get("link") == "flange_link"
    origin = joint.find("origin")
    assert np.allclose(
        [float(value) for value in origin.get("xyz").split()], [0.0, 0.0, 0.0]
    )
    assert np.allclose(
        [float(value) for value in origin.get("rpy").split()], [0.0, 0.0, 0.0]
    )


def test_joint_limits_match_controller_exactly():
    kinematics = validator.import_kinematics(validator.DEFAULT_KINEMATICS)
    errors = validator.validate_joint_limits(
        PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf", kinematics
    )
    assert errors["max_lower_error_rad"] < 1e-12
    assert errors["max_upper_error_rad"] < 1e-12


def test_camera_extrinsics_match_validated_calibration():
    errors = validator.validate_camera(
        PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf",
        validator.DEFAULT_CAMERA,
    )
    assert errors["position_error_m"] < 1e-10
    assert errors["rotation_error_rad"] < 1e-10


def test_d405_geometry_uses_official_case_rebased_to_color_optical_frame():
    import yaml

    geometry = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "geometry_estimates.yaml").read_text(
            encoding="utf-8"
        )
    )["d405_camera_link"]
    assert geometry["frame_semantics"] == "camera_color_optical_frame"
    assert geometry["status"] == "official_d405_case_geometry_rebased_to_optical_frame"
    assert geometry["collision"]["shape"] == "box"
    assert geometry["collision"]["box_size_m"] == [0.023, 0.042, 0.042]

    for key in ("packaged_mesh", "packaged_license", "provenance_readme"):
        assert (PACKAGE_ROOT / geometry["source"][key]).is_file()
    for path_key, sha_key in (
        ("packaged_mesh", "packaged_mesh_sha256"),
        ("packaged_license", "packaged_license_sha256"),
    ):
        path = PACKAGE_ROOT / geometry["source"][path_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == geometry["source"][
            sha_key
        ]

    root = ET.parse(PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf").getroot()
    camera = next(
        link for link in root.findall("link") if link.get("name") == "d405_camera_link"
    )
    visual_origin = camera.find("visual/origin")
    collision_origin = camera.find("collision/origin")
    np.testing.assert_allclose(
        [float(value) for value in visual_origin.get("xyz").split()],
        geometry["visual"]["xyz_m"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [float(value) for value in visual_origin.get("rpy").split()],
        geometry["visual"]["rpy_rad"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [float(value) for value in collision_origin.get("xyz").split()],
        geometry["collision"]["xyz_m"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [float(value) for value in collision_origin.get("rpy").split()],
        geometry["collision"]["rpy_rad"],
        atol=1e-12,
    )
    collision_box = camera.find("collision/geometry/box")
    np.testing.assert_allclose(
        [float(value) for value in collision_box.get("size").split()],
        geometry["collision"]["box_size_m"],
        atol=1e-12,
    )


def test_step_component_meshes_are_used_in_link_local_geometry():
    root = ET.parse(PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf").getroot()
    mesh_references = [
        element.get("filename", "")
        for element in root.findall(".//mesh")
    ]
    arm_mesh_references = [
        path for path in mesh_references if "/mesh/link_local/" in path
    ]
    assert len(arm_mesh_references) == 14
    assert sum("/visual/" in path for path in arm_mesh_references) == 7
    assert sum("/collision/" in path for path in arm_mesh_references) == 7
    assert sum(path.endswith("/mesh/camera/d405.stl") for path in mesh_references) == 1
    for link_name in ("base_link", *(f"link{index}" for index in range(1, 7))):
        for variant in ("visual", "collision"):
            relative = f"mesh/link_local/{variant}/{link_name}.stl"
            assert (PACKAGE_ROOT / relative).is_file()
            assert (
                f"package://estun_s3_60_g2_description/{relative}"
                in mesh_references
            )

    generation = validator.json.loads(
        (PACKAGE_ROOT / "mesh" / "link_local" / "generation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert generation["status"] == "step_component_meshes_link_local"
    assert {
        link: entry["step_component"]
        for link, entry in generation["links"].items()
    } == {
        "base_link": "NAUO7",
        "link1": "NAUO3",
        "link2": "NAUO1",
        "link3": "NAUO2",
        "link4": "NAUO4",
        "link5": "NAUO5",
        "link6": "NAUO6",
    }

    for name in (
        "NAUO1.stl",
        "NAUO2.stl",
        "NAUO3.stl",
        "NAUO4.stl",
        "NAUO5.stl",
        "NAUO6.stl",
        "NAUO7.stl",
    ):
        assert (PACKAGE_ROOT / "mesh" / name).is_file()


def test_gripper_links_meshes_joints_and_tcp_match_assembly_authority():
    import yaml

    assembly = yaml.safe_load(
        (
            PACKAGE_ROOT
            / "config"
            / "gripper_assembly.fae2mr86m_ms42dc.yaml"
        ).read_text(encoding="utf-8")
    )
    root = ET.parse(PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf").getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}

    body_link = assembly["mounting"]["body_link"]
    for name in (
        body_link,
        "gripper_left_finger_link",
        "gripper_right_finger_link",
        "grasp_tcp",
    ):
        assert name in links

    mount = joints["gripper_body_fixed"]
    assert mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == "flange_link"
    assert mount.find("child").get("link") == body_link
    origin = mount.find("origin")
    np.testing.assert_allclose(
        [float(value) for value in origin.get("xyz").split()],
        [0.0, 0.0, 0.091751924],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [float(value) for value in origin.get("rpy").split()],
        [0.0, -math.pi / 2.0, 0.0],
        atol=1e-12,
    )

    for side, expected_axis in (("left", [0.0, 1.0, 0.0]), ("right", [0.0, -1.0, 0.0])):
        spec = assembly["finger_joints"][side]
        joint = joints[spec["name"]]
        assert joint.get("type") == "prismatic"
        assert joint.find("parent").get("link") == body_link
        assert joint.find("child").get("link") == spec["child"]
        np.testing.assert_allclose(
            [float(value) for value in joint.find("axis").get("xyz").split()],
            expected_axis,
            atol=1e-12,
        )
        limit = joint.find("limit")
        assert float(limit.get("lower")) == -0.0362745
        assert float(limit.get("upper")) == 0.0
        assert float(limit.get("effort")) == 30.0
        assert float(limit.get("velocity")) == 0.02

    tcp = joints["grasp_tcp_fixed"]
    assert tcp.get("type") == "fixed"
    assert tcp.find("parent").get("link") == body_link
    assert tcp.find("child").get("link") == "grasp_tcp"
    np.testing.assert_allclose(
        [float(value) for value in tcp.find("origin").get("xyz").split()],
        [0.05, 0.0, 0.0],
        atol=1e-12,
    )

    for relative, expected_sha in (
        ("mesh/gripper/visual/gripper_body_link.stl", None),
        ("mesh/gripper/collision/gripper_body_link.stl", None),
        ("mesh/gripper/visual/gripper_left_finger_link.stl", None),
        ("mesh/gripper/collision/gripper_left_finger_link.stl", None),
        ("mesh/gripper/visual/gripper_right_finger_link.stl", None),
        ("mesh/gripper/collision/gripper_right_finger_link.stl", None),
    ):
        path = PACKAGE_ROOT / relative
        assert path.is_file()
        if expected_sha is not None:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha

    generation = json.loads(
        (
            PACKAGE_ROOT / "mesh" / "gripper" / "generation_report.json"
        ).read_text(encoding="utf-8")
    )
    assert generation["status"] == "provisional_offline"
    assert generation["source_step_sha256"] == assembly["source"]["step_sha256"]
    assert set(generation["links"]) == {
        "gripper_body_link",
        "gripper_left_finger_link",
        "gripper_right_finger_link",
    }


def test_model_manifest_matches_current_authorities():
    manifest = json.loads(
        (PACKAGE_ROOT / "validation" / "model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "offline_arm_and_gripper_geometry_validated"
    assert manifest["safety_scope"] == "offline_only_no_robot_motion_no_gripper_command"
    assert manifest["step_component_to_link"] == {
        "base_link": "NAUO7",
        "link1": "NAUO3",
        "link2": "NAUO1",
        "link3": "NAUO2",
        "link4": "NAUO4",
        "link5": "NAUO5",
        "link6": "NAUO6",
    }

    project_root = PACKAGE_ROOT.parents[1]
    records = [
        manifest["source_step"],
        manifest["gripper_source_step"],
        *manifest["authorities"].values(),
        *manifest["validation_artifacts"].values(),
    ]
    for record in records:
        path = project_root / record["path"]
        assert path.stat().st_size == record["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    assert "installed gripper clocking and shim thickness" in manifest["not_represented"]
    assert "GraspNet real-robot grasp labels and success labels" in manifest["not_represented"]
    assert manifest["gripper_assembly"]["status"] == "provisional_offline"
    assert manifest["gripper_assembly"]["installed_clocking_status"] == "unmeasured"
    assert any("prismatic finger joints" in caveat for caveat in manifest["caveats"])
    smoke = json.loads(
        (
            project_root
            / manifest["validation_artifacts"]["moveit_offline_smoke_report"]["path"]
        ).read_text(encoding="utf-8")
    )
    assert smoke["status"] == "pass"
    assert smoke["safety"]["trajectory_execution"] == "disabled"
    assert smoke["safety"]["robot_motion_commands_sent"] is False
    assert smoke["safety"]["gripper_commands_sent"] is False
    assert smoke["model_scope"]["tcp_link"] == "grasp_tcp"
    assert smoke["ik_round_trip"]["group_name"] == "arm_with_tool"
    assert smoke["collision_check"]["states"]["start_open"]["valid"] is True
    assert (
        smoke["collision_check"]["states"]["start_rigid_close_envelope"]["valid"]
        is True
    )
