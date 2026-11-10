"""Start a purely offline MoveIt 2 planning session for the Estun S3-60-G2.

No robot driver, ROS 2 controller, or trajectory execution interface is
started.  move_group is explicitly configured with trajectory execution
disabled, so planning and IK can be validated without commanding hardware.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("estun_s3_60_g2_description"))
    moveit_share = Path(get_package_share_directory("estun_s3_60_g2_moveit_config"))
    config_dir = moveit_share / "config"
    rviz_config = moveit_share / "rviz" / "moveit.rviz"

    robot_description = xacro.process_file(
        str(description_share / "urdf" / "estun_s3_60_g2.urdf.xacro")
    ).toxml()
    semantic_description = _read(config_dir / "estun_s3_60_g2.srdf")
    kinematics = _load_yaml(config_dir / "kinematics.yaml")
    joint_limits = _load_yaml(config_dir / "joint_limits.yaml")
    ompl = _load_yaml(config_dir / "ompl_planning.yaml")

    use_sim_time = LaunchConfiguration("use_sim_time")
    start_joint_state_publisher = LaunchConfiguration("joint_state_publisher")
    start_rviz = LaunchConfiguration("rviz")

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
            {"publish_frequency": 30.0},
        ],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
            {"rate": 30},
        ],
        condition=IfCondition(start_joint_state_publisher),
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"robot_description_semantic": semantic_description},
            {"robot_description_kinematics": kinematics},
            {"robot_description_planning": joint_limits},
            {"planning_pipelines": ["ompl"]},
            {"default_planning_pipeline": "ompl"},
            {"ompl": ompl},
            {"allow_trajectory_execution": False},
            {"publish_planning_scene": True},
            {"publish_geometry_updates": True},
            {"publish_state_updates": True},
            {"publish_transforms_updates": True},
            {"publish_robot_description": True},
            {"use_sim_time": use_sim_time},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(rviz_config)],
        parameters=[
            {"robot_description": robot_description},
            {"robot_description_semantic": semantic_description},
            {"robot_description_kinematics": kinematics},
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use an external simulation clock; false for pure offline mode.",
            ),
            DeclareLaunchArgument(
                "joint_state_publisher",
                default_value="true",
                description="Publish zero joint states when no hardware state source exists.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Start RViz2 with the offline MotionPlanning display.",
            ),
            robot_state_publisher,
            joint_state_publisher,
            move_group,
            rviz,
        ]
    )
