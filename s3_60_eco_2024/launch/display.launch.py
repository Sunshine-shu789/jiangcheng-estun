"""Display the S3-60 Eco 2024 robot model in RViz2 with a joint slider GUI."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("s3_60_eco_2024")
    urdf_path = os.path.join(pkg_share, "urdf", "s3_60_eco_2024.urdf")
    rviz_config = os.path.join(pkg_share, "rviz", "display.rviz")

    with open(urdf_path, encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Start joint_state_publisher_gui (joint sliders)",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz2",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
                condition=IfCondition(LaunchConfiguration("gui")),
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ]
    )
