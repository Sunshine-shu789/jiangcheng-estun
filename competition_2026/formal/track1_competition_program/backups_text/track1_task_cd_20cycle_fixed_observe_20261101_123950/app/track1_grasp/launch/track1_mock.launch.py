"""Hardware-free launch for any Track 1 task.

    ros2 launch track1_grasp track1_mock.launch.py task:=A
    ros2 launch track1_grasp track1_mock.launch.py task:=B max_picks:=5
    ros2 launch track1_grasp track1_mock.launch.py task:=E max_picks:=4
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("track1_grasp"), "config", "track1_mock.yaml"
    )
    config = LaunchConfiguration("config")
    task = LaunchConfiguration("task")
    max_picks = LaunchConfiguration("max_picks")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("task", default_value="C"),
            DeclareLaunchArgument("max_picks", default_value="4"),
            Node(
                package="track1_grasp",
                executable="mock_demo",
                name="track1_mock_demo",
                output="screen",
                arguments=[
                    "--config", config,
                    "--task", task,
                    "--max-picks", max_picks,
                ],
            ),
        ]
    )
