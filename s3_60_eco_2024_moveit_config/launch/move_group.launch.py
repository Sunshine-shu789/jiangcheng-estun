"""Start move_group for the S3-60 Eco 2024.

配合仿真是这样用的（两个终端）：

    ros2 launch s3_60_eco_2024 gazebo.launch.py           # 仿真 + 控制器
    ros2 launch s3_60_eco_2024_moveit_config move_group.launch.py

move_group 从 Gazebo 的 joint_states / TF 里拿当前状态，规划结果通过
joint_trajectory_controller 的 action 执行（配置见 config/moveit_controllers.yaml）。
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory("s3_60_eco_2024_moveit_config"), "config"
    )
    robot_dir = get_package_share_directory("s3_60_eco_2024")

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            {"robot_description": read_text(os.path.join(robot_dir, "urdf", "s3_60_eco_2024.urdf"))},
            {
                "robot_description_semantic": read_text(
                    os.path.join(config_dir, "s3_60_eco_2024.srdf")
                )
            },
            {"robot_description_kinematics": load_yaml(os.path.join(config_dir, "kinematics.yaml"))},
            # joint_limits.yaml 在 MoveIt 里是通过 robot_description_planning 传的
            {"robot_description_planning": load_yaml(os.path.join(config_dir, "joint_limits.yaml"))},
            load_yaml(os.path.join(config_dir, "ompl_planning.yaml")),
            {
                "planning_pipelines": ["ompl"],
                "default_planning_pipeline": "ompl",
            },
            load_yaml(os.path.join(config_dir, "moveit_controllers.yaml")),
            {"publish_planning_scene": True},
            {"publish_geometry_updates": True},
            {"publish_state_updates": True},
            {"publish_transforms_updates": True},
            {"publish_robot_description": True},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="和 Gazebo 一起跑时用仿真时钟；接真机时设成 false",
            ),
            move_group,
        ]
    )
