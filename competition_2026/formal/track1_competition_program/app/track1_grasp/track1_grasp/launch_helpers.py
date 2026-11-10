"""Shared launch construction for the five Track 1 task entry points."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def build_task_launch(
    *, mode: str, default_max_picks: int | None
) -> LaunchDescription:
    """Create the thin A/B/C/D/E wrapper around the shared Track 1 runner."""

    selected = str(mode).upper()
    share = get_package_share_directory("track1_grasp")
    default_config = os.path.join(share, "config", "track1_live.yaml")
    config = LaunchConfiguration("config")
    execute = LaunchConfiguration("execute")
    max_picks = LaunchConfiguration("max_picks")

    common_arguments = ["--config", config, "--task", selected]
    actions = [
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("execute", default_value="false"),
    ]
    if selected == "A":
        frames = LaunchConfiguration("frames")
        actions.append(DeclareLaunchArgument("frames", default_value="0"))
        actions.append(
            Node(
                package="track1_grasp",
                executable="track1_node",
                name="track1_task_a",
                output="screen",
                arguments=[*common_arguments, "--frames", frames],
            )
        )
        return LaunchDescription(actions)

    arguments = list(common_arguments)
    if default_max_picks is not None:
        actions.append(
            DeclareLaunchArgument(
                "max_picks",
                default_value=str(default_max_picks),
            )
        )
        arguments.extend(["--max-picks", max_picks])
    actions.extend(
        [
            Node(
                package="track1_grasp",
                executable="track1_node",
                name=f"track1_task_{selected.lower()}_dry_run",
                output="screen",
                arguments=arguments,
                condition=UnlessCondition(execute),
            ),
            Node(
                package="track1_grasp",
                executable="track1_node",
                name=f"track1_task_{selected.lower()}_execute",
                output="screen",
                arguments=[*arguments, "--execute"],
                condition=IfCondition(execute),
            ),
        ]
    )
    return LaunchDescription(actions)
