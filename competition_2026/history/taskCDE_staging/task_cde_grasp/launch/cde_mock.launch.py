import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("task_cde_grasp"), "config", "cde_mock.yaml"
    )
    config = LaunchConfiguration("config")
    mode = LaunchConfiguration("mode")
    max_picks = LaunchConfiguration("max_picks")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("mode", default_value="C"),
            DeclareLaunchArgument("max_picks", default_value="4"),
            Node(
                package="task_cde_grasp",
                executable="mock_demo",
                name="task_cde_mock_demo",
                output="screen",
                arguments=[
                    "--config", config,
                    "--mode", mode,
                    "--max-picks", max_picks,
                ],
            ),
        ]
    )
