#!/usr/bin/env python3
"""Plan with MoveIt and actually move the arm in Gazebo.

This is the closed-loop counterpart of ``check_reachability.py``: instead of
only asking "does an IK solution exist?", it sends a goal to move_group's
``/move_action``, lets OMPL plan, lets the simple controller manager call the
Gazebo ``joint_trajectory_controller`` and then measures whether the joints
really got there.

Usage::

    # named target from the SRDF
    ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --named demo
    ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --named home

    # explicit joint angles (rad)
    ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py \
        --joints 0.4 -0.5 0.7 0 0.5 0

    # a Track 1 object: IK above the item, then plan + execute
    ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --grasp-item 3

    # plan only, do not move
    ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --grasp-item 3 --plan-only

The script exits non-zero when IK, planning or execution fails, or when the
measured joints do not reach the target, so it can be used as a self-check in
scripts.
"""

import argparse
import math
import os
import sys
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINTS = [f"joint{i}" for i in range(1, 7)]
MOVE_ACTION = "move_action"
IK_SERVICE = "compute_ik"


def quaternion_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def tool_down_quaternion(tilt_deg, yaw_deg):
    """Tool z axis pointing down, tilted by ``tilt`` towards the robot, then yawed."""
    roll = math.pi - math.radians(tilt_deg)
    return quaternion_from_rpy(roll, 0.0, math.radians(yaw_deg))


def candidate_orientations():
    """和 check_reachability.py 一致的候选姿态：4 个倾角 x 8 个 yaw。"""
    for tilt in (0.0, 20.0, 40.0, 60.0):
        for yaw in range(0, 360, 45):
            yield f"tilt{int(tilt)}_yaw{yaw}", tool_down_quaternion(tilt, yaw)


def item_height(item):
    size = item["size"]
    if item["shape"] == "box":
        return float(size[2])
    if item["shape"] == "cylinder":
        return float(size[1])
    return float(size[0])


def parse_args(argv):
    default_yaml = os.path.join(
        get_package_share_directory("s3_60_eco_2024_moveit_config"),
        "config",
        "track1_items.yaml",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--named", choices=("home", "demo"), help="SRDF 里预设的关节目标")
    target.add_argument("--joints", nargs=6, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6"))
    target.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="法兰目标位置（base_link 坐标系），姿态用 --tilt-deg/--yaw-deg")
    target.add_argument("--grasp-item", type=int, metavar="ID",
                        help="赛道一物品编号（1-40）：目标取在物品顶面 + --approach-height")
    parser.add_argument("--yaml", default=default_yaml, help="物品尺寸表（--grasp-item 用）")
    parser.add_argument("--approach-height", type=float, default=0.05,
                        help="法兰目标比物品顶面高多少米（默认 0.05）")
    parser.add_argument("--tilt-deg", type=float, default=0.0, help="工具轴从竖直方向倾斜的角度")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="绕工具轴的偏航角")
    parser.add_argument("--plan-only", action="store_true", help="只规划，不执行")
    parser.add_argument("--planning-time", type=float, default=10.0, help="单次规划时间上限（秒）")
    parser.add_argument("--attempts", type=int, default=3, help="规划尝试次数")
    parser.add_argument("--velocity-scaling", type=float, default=0.3, help="速度缩放（0-1）")
    parser.add_argument("--accel-scaling", type=float, default=0.3, help="加速度缩放（0-1）")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="执行后允许的关节跟踪误差（弧度，默认 0.02 ≈ 1.15°）")
    parser.add_argument("--settle-timeout", type=float, default=15.0,
                        help="等机械臂到位的最长时间（秒）")
    parser.add_argument("--group", default="arm")
    return parser.parse_args(argv)


class PlanExecuteDemo(Node):
    def __init__(self):
        super().__init__("s3_60_eco_2024_plan_execute_demo")
        self.joint_positions = None
        self.joint_names = []
        self.create_subscription(JointState, "joint_states", self.on_joint_state, 10)
        self.move_client = ActionClient(self, MoveGroup, MOVE_ACTION)
        self.ik_client = self.create_client(GetPositionIK, IK_SERVICE)

    def on_joint_state(self, message):
        positions = dict(zip(message.name, message.position))
        if all(joint in positions for joint in JOINTS):
            self.joint_names = message.name
            self.joint_positions = positions

    def wait_for_joint_states(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        while self.joint_positions is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        if self.joint_positions is None:
            raise RuntimeError("等不到 /joint_states；Gazebo（或真机）起来了吗？")
        return [self.joint_positions[joint] for joint in JOINTS]

    def wait_for_services(self, timeout=30.0):
        if not self.move_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"等不到 /{MOVE_ACTION}；move_group 起来了吗？")
        if not self.ik_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"等不到 /{IK_SERVICE}；move_group 起来了吗？")

    def solve_ik(self, position, quaternion, timeout=0.2):
        request = GetPositionIK.Request()
        request.ik_request.group_name = "arm"
        request.ik_request.ik_link_name = "link6"
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = int(timeout)
        request.ik_request.timeout.nanosec = int((timeout % 1) * 1e9)
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = quaternion
        request.ik_request.pose_stamped = pose

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout + 2.0)
        response = future.result()
        if response is None or response.error_code.val != 1:
            return None
        joints = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        return [joints[joint] for joint in JOINTS]

    def move_goal(self, joint_positions, args):
        goal = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = args.group
        request.pipeline_id = "ompl"
        request.planner_id = "RRTConnect"
        request.num_planning_attempts = args.attempts
        request.allowed_planning_time = args.planning_time
        request.max_velocity_scaling_factor = args.velocity_scaling
        request.max_acceleration_scaling_factor = args.accel_scaling

        constraints = Constraints()
        constraints.name = "joint_goal"
        for joint, position in zip(JOINTS, joint_positions):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.01
            joint_constraint.tolerance_below = 0.01
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        request.goal_constraints = [constraints]

        options = PlanningOptions()
        options.plan_only = args.plan_only
        options.look_around = False
        options.planning_scene_diff.is_diff = True
        goal.request = request
        goal.planning_options = options
        return goal

    def send_goal(self, goal):
        send_future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("move_group 没有接受这个目标（goal rejected）")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("move_group 没有返回结果（action server 掉线？）")
        return wrapped.result

    def wait_until_settled(self, target, timeout, initial):
        """Wait until the joints stop moving.

        Returns (max_error, moved, final): error against the goal, how far the
        joints travelled compared to ``initial`` (the state before the goal was
        sent) and the final measured positions.
        """
        deadline = time.monotonic() + timeout
        previous = list(initial)
        stable_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            current = [self.joint_positions[joint] for joint in JOINTS]
            if max(abs(a - b) for a, b in zip(current, previous)) < 0.001:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since > 1.0:
                    break
            else:
                stable_since = None
            previous = current
        final = [self.joint_positions[joint] for joint in JOINTS]
        max_error = max(abs(a - b) for a, b in zip(final, target))
        moved = max(abs(a - b) for a, b in zip(final, initial))
        return max_error, moved, final


def describe_trajectory(trajectory):
    points = trajectory.joint_trajectory.points
    if not points:
        return "0 点"
    last = points[-1].time_from_start
    duration = last.sec + last.nanosec / 1e9
    return f"{len(points)} 点 / {duration:.2f} s"


def resolve_target(node, args):
    """Return (joint_goal, description) or raise RuntimeError."""
    if args.named == "home":
        return [0.0] * 6, "SRDF 预设 home（全零）"
    if args.named == "demo":
        return [0.4, -0.5, 0.7, 0.0, 0.5, 0.0], "SRDF 预设 demo"
    if args.joints is not None:
        return [float(v) for v in args.joints], "命令行给定关节角"

    if args.grasp_item is not None:
        with open(args.yaml, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        items = [item for item in config["items"] if item["id"] == args.grasp_item]
        if not items:
            raise RuntimeError(f"物品表里没有编号 {args.grasp_item}")
        item = items[0]
        subset_items = [i for i in config["items"] if i["subset"] == item["subset"]]
        index = subset_items.index(item)
        layout = config["layout"]
        position = (
            float(layout["x_start"]) + (index % int(layout["columns"])) * float(layout["x_step"]),
            float(layout["y_start"]) + (index // int(layout["columns"])) * float(layout["y_step"]),
            float(config["table"]["top_z"]) + item_height(item) + args.approach_height,
        )
        description = (
            f"物品 #{item['id']} {item['name_cn']}（Subset {item['subset']}，"
            f"{item['shape']} {item['size']}）顶面 + {args.approach_height:.2f} m"
        )
    else:
        position = tuple(float(v) for v in args.pose)
        description = f"笛卡尔目标 {position}"

    quaternion = tool_down_quaternion(args.tilt_deg, args.yaw_deg)
    node.get_logger().info(
        f"IK：目标 ({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})，"
        f"tilt={args.tilt_deg:g}° yaw={args.yaw_deg:g}°"
    )
    joints, label = node.solve_ik(position, quaternion), f"tilt{args.tilt_deg:g}_yaw{args.yaw_deg:g}"
    if joints is None and args.grasp_item is not None:
        # 默认姿态解不出来时（俯视抓取的工作空间很窄），自动试完整候选表。
        node.get_logger().info("默认姿态解不出来，自动尝试 32 个候选姿态……")
        for candidate_label, candidate_quaternion in candidate_orientations():
            joints = node.solve_ik(position, candidate_quaternion)
            if joints is not None:
                label = candidate_label
                break
    if joints is None:
        raise RuntimeError(
            "IK 解不出来（可能够不着或会碰撞）。先用 check_reachability.py 看这张表里"
            "哪些姿态可行，再用 --tilt-deg/--yaw-deg 指定。"
        )
    return joints, description + f"，IK 姿态 {label}"


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    node = PlanExecuteDemo()
    exit_code = 0
    try:
        node.wait_for_services()
        begin = node.wait_for_joint_states()
        node.get_logger().info("起始关节角：" + " ".join(f"{v:+.3f}" for v in begin))

        joint_goal, description = resolve_target(node, args)
        goal = node.move_goal(joint_goal, args)
        node.get_logger().info(f"目标：{description}")
        node.get_logger().info(
            "目标关节角：" + " ".join(f"{v:+.3f}" for v in joint_goal)
            + ("（只规划）" if args.plan_only else "")
        )

        result = node.send_goal(goal)
        error_code = result.error_code.val
        node.get_logger().info(
            f"规划耗时 {result.planning_time:.2f} s，轨迹 {describe_trajectory(result.planned_trajectory)}，"
            f"MoveGroup 返回 {error_code}"
        )
        if error_code != 1:
            raise RuntimeError(
                "规划/执行失败：错误码 %d（1=成功，-1=规划失败，-4=控制失败，-12=目标碰撞）" % error_code
            )
        if len(result.planned_trajectory.joint_trajectory.points) < 2:
            raise RuntimeError("规划出的轨迹不足 2 个点，不像一条可用轨迹")

        if args.plan_only:
            node.get_logger().info("--plan-only：不执行，到此为止。")
            return exit_code

        max_error, moved, final = node.wait_until_settled(joint_goal, args.settle_timeout, begin)
        node.get_logger().info("最终关节角：" + " ".join(f"{v:+.3f}" for v in final))
        node.get_logger().info(
            f"实际移动量 {math.degrees(moved):.1f}°，跟踪误差 {math.degrees(max_error):.2f}°"
        )
        if moved < 0.01:
            raise RuntimeError("闭环失败：机械臂收到轨迹但几乎没动")
        if max_error > args.tolerance:
            raise RuntimeError(
                f"闭环失败：跟踪误差 {math.degrees(max_error):.2f}° 超过 {math.degrees(args.tolerance):.2f}°"
            )
        node.get_logger().info("闭环通过：规划 → 控制 → Gazebo 执行，机械臂到位。")
        return exit_code
    except RuntimeError as error:
        node.get_logger().error(str(error))
        exit_code = 1
        return exit_code
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
