import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("task_cde_grasp"),
        "config",
        "cde_real_template.yaml",
    )
    config = LaunchConfiguration("config")
    mode = LaunchConfiguration("mode")
    execute = LaunchConfiguration("execute")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("mode", default_value="C"),
            DeclareLaunchArgument("execute", default_value="false"),
            Node(
                package="task_cde_grasp",
                executable="pick_node",
                name="task_cde_grasp_dry_run",
                output="screen",
                arguments=["--config", config, "--mode", mode],
                condition=UnlessCondition(execute),
            ),
            Node(
                package="task_cde_grasp",
                executable="pick_node",
                name="task_cde_grasp_execute",
                output="screen",
                arguments=["--config", config, "--mode", mode, "--execute"],
                condition=IfCondition(execute),
            ),
        ]
    )
