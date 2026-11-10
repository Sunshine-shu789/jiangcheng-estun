"""Static consistency checks for the offline MoveIt configuration."""

from __future__ import annotations

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import xacro
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_ROOT = PACKAGE_ROOT.parent / "estun_s3_60_g2"
URDF_PATH = DESCRIPTION_ROOT / "urdf" / "estun_s3_60_g2.urdf"
XACRO_PATH = DESCRIPTION_ROOT / "urdf" / "estun_s3_60_g2.urdf.xacro"
SRDF_PATH = PACKAGE_ROOT / "config" / "estun_s3_60_g2.srdf"
LIMITS_PATH = PACKAGE_ROOT / "config" / "joint_limits.yaml"
KINEMATICS_PATH = PACKAGE_ROOT / "config" / "kinematics.yaml"
OMPL_PATH = PACKAGE_ROOT / "config" / "ompl_planning.yaml"
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "move_group.launch.py"

ARM_JOINTS = [f"joint{index}" for index in range(1, 7)]
FINGER_JOINTS = [
    "gripper_left_finger_joint",
    "gripper_right_finger_joint",
]


def _robot() -> ET.Element:
    document = xacro.process_file(str(XACRO_PATH))
    return ET.fromstring(document.toxml())


def test_srdf_groups_cover_arm_gripper_and_tool_tcp():
    root = ET.parse(SRDF_PATH).getroot()
    assert root.attrib["name"] == "estun_s3_60_g2"

    groups = {group.attrib["name"]: group for group in root.findall("group")}
    assert groups.keys() >= {"arm", "gripper", "arm_with_tool"}
    assert groups["arm"].find("chain").attrib == {
        "base_link": "base_link",
        "tip_link": "flange_link",
    }
    assert groups["arm_with_tool"].find("chain").attrib == {
        "base_link": "base_link",
        "tip_link": "grasp_tcp",
    }
    assert [joint.attrib["name"] for joint in groups["gripper"].findall("joint")] == (
        FINGER_JOINTS
    )

    states = {state.attrib["name"]: state for state in root.findall("group_state")}
    assert states.keys() >= {"zero", "offline_demo", "open", "closed"}
    for name in ("zero", "offline_demo"):
        state = states[name]
        assert state.attrib["group"] == "arm"
        assert [joint.attrib["name"] for joint in state.findall("joint")] == ARM_JOINTS
    for name in ("open", "closed"):
        state = states[name]
        assert state.attrib["group"] == "gripper"
        assert [joint.attrib["name"] for joint in state.findall("joint")] == FINGER_JOINTS


def test_srdf_keeps_real_tool_and_camera_collision_pairs_active():
    root = ET.parse(SRDF_PATH).getroot()
    disabled = {
        tuple(sorted((entry.attrib["link1"], entry.attrib["link2"])))
        for entry in root.findall("disable_collisions")
    }
    required_disabled = {
        ("flange_link", "gripper_body_link"),
        ("gripper_body_link", "gripper_left_finger_link"),
        ("gripper_body_link", "gripper_right_finger_link"),
        ("gripper_body_link", "grasp_tcp"),
        ("flange_link", "d405_camera_link"),
    }
    for pair in required_disabled:
        assert tuple(sorted(pair)) in disabled

    must_remain_checked = {
        ("gripper_left_finger_link", "gripper_right_finger_link"),
        ("gripper_body_link", "d405_camera_link"),
        ("gripper_left_finger_link", "d405_camera_link"),
        ("gripper_right_finger_link", "d405_camera_link"),
        ("link6", "gripper_body_link"),
        ("link6", "gripper_left_finger_link"),
        ("link6", "gripper_right_finger_link"),
    }
    for pair in must_remain_checked:
        assert tuple(sorted(pair)) not in disabled


def test_moveit_joint_limits_match_controller_and_gripper_urdf():
    robot = _robot()
    urdf_limits = {}
    for joint in robot.findall("joint"):
        if joint.attrib.get("type") not in ("revolute", "prismatic"):
            continue
        limit = joint.find("limit")
        assert limit is not None
        urdf_limits[joint.attrib["name"]] = {
            "lower": float(limit.attrib["lower"]),
            "upper": float(limit.attrib["upper"]),
            "velocity": float(limit.attrib["velocity"]),
            "effort": float(limit.attrib["effort"]),
        }

    moveit_limits = yaml.safe_load(LIMITS_PATH.read_text(encoding="utf-8"))[
        "joint_limits"
    ]
    assert moveit_limits.keys() == urdf_limits.keys()
    for name, expected in urdf_limits.items():
        configured = moveit_limits[name]
        assert math.isclose(
            configured["min_position"], expected["lower"], abs_tol=1e-12
        )
        assert math.isclose(
            configured["max_position"], expected["upper"], abs_tol=1e-12
        )
        assert math.isclose(
            configured["max_velocity"], expected["velocity"], abs_tol=1e-12
        )
        assert math.isclose(
            configured["max_effort"], expected["effort"], abs_tol=1e-12
        )
        assert configured["has_position_limits"] is True
        assert configured["has_velocity_limits"] is True
        assert configured["has_acceleration_limits"] is True
        assert configured["has_effort_limits"] is True


def test_kdl_and_ompl_are_the_offline_baseline_for_tool_point_ik():
    kinematics = yaml.safe_load(KINEMATICS_PATH.read_text(encoding="utf-8"))
    for group in ("arm", "arm_with_tool"):
        configured = kinematics[group]
        assert (
            configured["kinematics_solver"]
            == "kdl_kinematics_plugin/KDLKinematicsPlugin"
        )
        assert configured["kinematics_solver_timeout"] == 0.1
        assert configured["kinematics_solver_attempts"] == 20

    ompl = yaml.safe_load(OMPL_PATH.read_text(encoding="utf-8"))
    assert ompl["planning_plugin"] == "ompl_interface/OMPLPlanner"
    assert "default_planner_request_adapters/AddTimeOptimalParameterization" in (
        ompl["request_adapters"]
    )
    for group in ("arm", "arm_with_tool"):
        assert ompl[group]["default_planner_config"] == "RRTConnect"
        assert ompl["planner_configs"]["RRTConnect"]["type"] == "geometric::RRTConnect"


def test_launch_hard_disables_trajectory_execution():
    launch = LAUNCH_PATH.read_text(encoding="utf-8")
    assert '{"allow_trajectory_execution": False}' in launch
    assert "moveit_ros_move_group" in launch
    assert "trajectory_execution" not in launch.replace(
        '{"allow_trajectory_execution": False}', ""
    )


def test_urdf_uses_arm_and_gripper_step_meshes_for_visual_and_collision_geometry():
    text = URDF_PATH.read_text(encoding="utf-8")
    robot = _robot()
    mesh_paths = [
        geometry[0].attrib["filename"]
        for geometry in robot.findall(".//geometry")
        if len(geometry) == 1 and geometry[0].tag == "mesh"
    ]
    # Seven arm links plus three gripper links, each visual/collision, and the
    # official D405 visual mesh whose collision is the upstream xacro box.
    assert len(mesh_paths) == 21
    expected_meshes = []
    for link_name in ("base_link", *(f"link{index}" for index in range(1, 7))):
        for variant in ("visual", "collision"):
            expected_meshes.append(
                f"package://estun_s3_60_g2_description/"
                f"mesh/link_local/{variant}/{link_name}.stl"
            )
    for link_name in (
        "gripper_body_link",
        "gripper_left_finger_link",
        "gripper_right_finger_link",
    ):
        for variant in ("visual", "collision"):
            expected_meshes.append(
                f"package://estun_s3_60_g2_description/"
                f"mesh/gripper/{variant}/{link_name}.stl"
            )
    expected_meshes.append(
        "package://estun_s3_60_g2_description/mesh/camera/d405.stl"
    )
    assert set(mesh_paths) == set(expected_meshes)
    for path in expected_meshes:
        relative = path.split("/estun_s3_60_g2_description/", 1)[1]
        assert (DESCRIPTION_ROOT / relative).is_file()
    assert "package://estun_s3_60_g2_description/mesh/NAUO" not in text
