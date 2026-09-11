"""Start move_group for the S3-60 Eco 2024.

配合仿真是这样用的（两个终端）：

    ros2 launch s3_60_eco_2024 gazebo.launch.py           # 仿真 + 控制器
    ros2 launch s3_60_eco_2024_moveit_config move_group.launch.py rviz:=true

move_group 从 Gazebo 的 joint_states / TF 里拿当前状态，规划结果通过
joint_trajectory_controller 的 action 执行（配置见 config/moveit_controllers.yaml）。

rviz:=true 额外打开带 MotionPlanning 显示的 RViz（rviz/moveit.rviz）：
规划场景里的碰撞体——包括 load_track1_items.py 加载的 40 件物品——都在
MotionPlanning -> Scene Geometry 里显示。
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def generate_launch_description():
    share_dir = get_package_share_directory("s3_60_eco_2024_moveit_config")
    config_dir = os.path.join(share_dir, "config")
    robot_dir = get_package_share_directory("s3_60_eco_2024")
    rviz_config = os.path.join(share_dir, "rviz", "moveit.rviz")

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
            # 规划流水线的参数必须挂在流水线名下面（ompl.planning_plugin 等），
            # 平铺传进去 move_group 是读不到的（会退回 CHOMP、且不做时间参数化）。
            {"ompl": load_yaml(os.path.join(config_dir, "ompl_planning.yaml"))},
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

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        # 和 Gazebo 一起用时必须跟仿真时钟走，否则 TF/规划场景的时间戳对不上。
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="和 Gazebo 一起跑时用仿真时钟；接真机时设成 false",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="同时打开带 MotionPlanning 显示的 RViz2（看 40 件物品）",
            ),
            move_group,
            rviz,
        ]
    )
