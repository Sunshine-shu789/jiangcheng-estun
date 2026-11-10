"""Real-hardware launch for tasks B/C/D/E (dry-run unless execute:=true).

    ros2 launch track1_grasp track1_real.launch.py task:=B
    ros2 launch track1_grasp track1_real.launch.py task:=C execute:=true

Task A is vision-only; use ``annotate_node`` (or ``track1_node --task A``)
instead of this launch file.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("track1_grasp"),
        "config",
        "track1_real_template.yaml",
    )
    config = LaunchConfiguration("config")
    task = LaunchConfiguration("task")
    execute = LaunchConfiguration("execute")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("task", default_value="C"),
            DeclareLaunchArgument("execute", default_value="false"),
            Node(
                package="track1_grasp",
                executable="track1_node",
                name="track1_grasp_dry_run",
                output="screen",
                arguments=["--config", config, "--task", task],
                condition=UnlessCondition(execute),
            ),
            Node(
                package="track1_grasp",
                executable="track1_node",
                name="track1_grasp_execute",
                output="screen",
                arguments=["--config", config, "--task", task, "--execute"],
                condition=IfCondition(execute),
            ),
        ]
    )
