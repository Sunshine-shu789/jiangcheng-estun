"""Spawn the S3-60 Eco 2024 in Gazebo Classic with ros2_control controllers."""

import os
import re
import subprocess

import numpy as np
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

XML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
PACKAGE_URI = re.compile(r"package://([A-Za-z0-9_]+)/")
LINK = re.compile(r"<link\s+name=\"[^\"]+\".*?</link>", re.DOTALL)
INERTIA = re.compile(r"<inertia\s+[^>]*/>")
MASS = re.compile(r"<mass\s+value=\"([^\"]+)\"")

# The SolidWorks export gives every link a singular inertia matrix: one
# principal moment is 0 and for link2 it is even slightly negative, which
# Gazebo's ODE solver cannot handle (the arm then ignores its commands).
# Principal moments are therefore floored to a small fraction of the link
# mass and the tensor is rebuilt, which keeps the exported values otherwise.
MIN_INERTIA_PER_KG = 1e-3  # kg m^2 per kg => radius of gyration ~3 cm


def resolve_package_uris(xml):
    """Rewrite package:// mesh URIs to file:// paths Gazebo can open.

    Gazebo Classic does not resolve `package://` on its own here, and a mesh
    it cannot open makes model loading give up before the model plugin (i.e.
    gazebo_ros2_control) is ever loaded.
    """

    def replace(match):
        try:
            share = get_package_share_directory(match.group(1))
        except PackageNotFoundError:
            return match.group(0)
        return f"file://{share}/"

    return PACKAGE_URI.sub(replace, xml)


def repair_inertias(xml):
    """Make every inertia tensor positive definite for Gazebo's solver."""

    def fix_link(match):
        block = match.group(0)
        mass = MASS.search(block)
        inertia = INERTIA.search(block)
        if not mass or not inertia:
            return block
        values = {
            key: float(value)
            for key, value in re.findall(r"(\w+)=\"([^\"]*)\"", inertia.group(0))
        }
        tensor = np.array(
            [
                [values["ixx"], values.get("ixy", 0.0), values.get("ixz", 0.0)],
                [values.get("ixy", 0.0), values["iyy"], values.get("iyz", 0.0)],
                [values.get("ixz", 0.0), values.get("iyz", 0.0), values["izz"]],
            ]
        )
        moments, axes = np.linalg.eigh(tensor)
        floor = MIN_INERTIA_PER_KG * float(mass.group(1))
        repaired = axes @ np.diag(np.maximum(moments, floor)) @ axes.T
        attrs = {
            "ixx": repaired[0, 0],
            "ixy": repaired[0, 1],
            "ixz": repaired[0, 2],
            "iyy": repaired[1, 1],
            "iyz": repaired[1, 2],
            "izz": repaired[2, 2],
        }
        ordered = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        replacement = "<inertia " + " ".join(
            f'{key}="{attrs[key]:.6g}"' for key in ordered
        ) + "/>"
        return block.replace(inertia.group(0), replacement)

    return LINK.sub(fix_link, xml)


def robot_state_publisher_node(context, *args, **kwargs):
    """Publish the xacro description with XML comments stripped.

    gazebo_ros2_control copies ``robot_description`` into the controller
    manager node options as an rcl parameter override rule. rcl parses that
    rule as YAML, which fails on the `: ` sequences and xacro banner inside
    XML comments, so the comments have to go before the description reaches
    robot_state_publisher.
    """
    pkg_share = get_package_share_directory("s3_60_eco_2024")
    xacro_file = os.path.join(pkg_share, "urdf", "s3_60_eco_2024.gazebo.urdf.xacro")
    gravity = LaunchConfiguration("gravity").perform(context)
    xml = subprocess.run(
        ["xacro", xacro_file, f"gravity:={gravity}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    xml = repair_inertias(resolve_package_uris(XML_COMMENT.sub("", xml))).strip()

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[
                {
                    "robot_description": xml,
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }
            ],
        )
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory("s3_60_eco_2024")
    rviz_config = os.path.join(pkg_share, "rviz", "display.rviz")
    gazebo_launch = os.path.join(
        get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py"
    )

    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={"gui": gui}.items(),
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic",
            "robot_description",
            "-entity",
            "s3_60_eco_2024",
            "-z",
            LaunchConfiguration("z"),
        ],
        output="screen",
    )

    def spawner(controller):
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                controller,
                "--controller-manager",
                "/controller_manager",
                "--controller-manager-timeout",
                "60",
            ],
            parameters=[{"use_sim_time": use_sim_time}],
            output="screen",
        )

    # The controllers can only be loaded once the model (and with it the
    # gazebo_ros2_control plugin) is in the world.
    load_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_entity,
            on_exit=[
                spawner("joint_state_broadcaster"),
                spawner("joint_trajectory_controller"),
            ],
        )
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(rviz),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Start the Gazebo client; set false for headless",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Also start RViz2 alongside Gazebo",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use Gazebo's /clock as the time source",
            ),
            DeclareLaunchArgument(
                "gravity",
                default_value="false",
                description=(
                    "Let the arm links fall under gravity (true) or hold their "
                    "commanded pose like a servo brake does (false)"
                ),
            ),
            DeclareLaunchArgument(
                "z",
                default_value="0.016",
                description=(
                    "Spawn height of the robot base in metres. The base mesh "
                    "reaches 15 mm below its origin, and a base buried in the "
                    "ground plane makes the contact solver throw the whole arm "
                    "around, so the default leaves a millimetre of clearance"
                ),
            ),
            gazebo,
            OpaqueFunction(function=robot_state_publisher_node),
            spawn_entity,
            load_controllers,
            rviz2,
        ]
    )
