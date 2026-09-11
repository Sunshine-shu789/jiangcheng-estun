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

    # 逐件跑完一个 Subset / 全部 4 个 Subset 的 IK → 规划 → 执行
    ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --sweep-subset 1
    ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --sweep-all --out report.md

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
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningScene,
    PlanningOptions,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_track1_items import build_scene  # noqa: E402  (path 只在上面那行之后才对)

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


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def grasp_position(config, item, approach_height):
    """物品顶面中心 + approach_height，返回法兰目标位置 (x, y, z)。"""
    layout = config["layout"]
    subset_items = [i for i in config["items"] if i["subset"] == item["subset"]]
    index = subset_items.index(item)
    x = float(layout["x_start"]) + (index % int(layout["columns"])) * float(layout["x_step"])
    y = float(layout["y_start"]) + (index // int(layout["columns"])) * float(layout["y_step"])
    z = float(config["table"]["top_z"]) + item_height(item) + approach_height
    return (x, y, z)


def parse_args(argv):
    default_yaml = os.path.join(
        get_package_share_directory("s3_60_eco_2024_moveit_config"),
        "config",
        "track1_items.yaml",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=False)
    target.add_argument("--named", choices=("home", "demo"), help="SRDF 里预设的关节目标")
    target.add_argument("--joints", nargs=6, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6"))
    target.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="法兰目标位置（base_link 坐标系），姿态用 --tilt-deg/--yaw-deg")
    target.add_argument("--grasp-item", type=int, metavar="ID",
                        help="赛道一物品编号（1-40）：目标取在物品顶面 + --approach-height")
    target.add_argument("--sweep-subset", type=int, metavar="N",
                        help="逐件跑完一个 Subset 的规划+执行（不需要再给目标）")
    target.add_argument("--sweep-all", action="store_true",
                        help="逐件跑完全部 4 个 Subset 的规划+执行（不需要再给目标）")
    parser.add_argument("--yaml", default=default_yaml, help="物品尺寸表（--grasp-item 用）")
    parser.add_argument("--approach-height", type=float, default=0.05,
                        help="法兰目标比物品顶面高多少米（默认 0.05）")
    parser.add_argument("--tilt-deg", type=float, default=0.0, help="工具轴从竖直方向倾斜的角度")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="绕工具轴的偏航角")
    parser.add_argument("--plan-only", action="store_true", help="只规划，不执行")
    parser.add_argument("--planning-time", type=float, default=10.0, help="单次规划时间上限（秒）")
    parser.add_argument("--attempts", type=int, default=10, help="规划尝试次数")
    parser.add_argument("--velocity-scaling", type=float, default=0.3, help="速度缩放（0-1）")
    parser.add_argument("--accel-scaling", type=float, default=0.3, help="加速度缩放（0-1）")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="执行后允许的关节跟踪误差（弧度，默认 0.02 ≈ 1.15°）")
    parser.add_argument("--settle-timeout", type=float, default=15.0,
                        help="等机械臂到位的最长时间（秒）")
    parser.add_argument("--ik-repeat", type=int, default=2,
                        help="逐件扫描时每件把 32 个候选姿态重复试几轮（KDL 带随机重启）")
    parser.add_argument("--out", help="逐件扫描时把结果写成 markdown 报告")
    parser.add_argument("--group", default="arm")
    args = parser.parse_args(argv)
    if not (args.named or args.joints or args.pose or args.grasp_item
            or args.sweep_subset or args.sweep_all):
        parser.error("要么给一个目标（--named/--joints/--pose/--grasp-item），"
                     "要么用 --sweep-subset/--sweep-all 逐件跑")
    return args


class PlanExecuteDemo(Node):
    def __init__(self):
        super().__init__("s3_60_eco_2024_plan_execute_demo")
        self.joint_positions = None
        self.joint_names = []
        self.create_subscription(JointState, "joint_states", self.on_joint_state, 10)
        self.move_client = ActionClient(self, MoveGroup, MOVE_ACTION)
        self.ik_client = self.create_client(GetPositionIK, IK_SERVICE)
        self.scene_publisher = self.create_publisher(PlanningScene, "planning_scene", 10)

    def set_scene(self, config, items, with_table=True):
        """把规划场景换成 items 这一组（和 check_reachability.py 同一套逻辑）。"""
        scene = PlanningScene()
        scene.is_diff = True
        all_objects, _ = build_scene(config, config["items"], True)
        for collision_object in all_objects:
            removal = CollisionObject()
            removal.header.frame_id = config["frame_id"]
            removal.id = collision_object.id
            removal.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(removal)
        objects, _ = build_scene(config, items, with_table)
        scene.world.collision_objects.extend(objects)
        for _ in range(5):
            self.scene_publisher.publish(scene)
            rclpy.spin_once(self, timeout_sec=0.2)
        time.sleep(0.5)

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


def trajectory_stats(trajectory):
    """返回 (点数, 总时长秒)；轨迹为空时 (0, 0.0)。"""
    points = trajectory.joint_trajectory.points
    if not points:
        return 0, 0.0
    last = points[-1].time_from_start
    return len(points), last.sec + last.nanosec / 1e9


def describe_trajectory(trajectory):
    count, duration = trajectory_stats(trajectory)
    return f"{count} 点 / {duration:.2f} s" if count else "0 点"


def resolve_target(node, args):
    """Return (joint_goal, description) or raise RuntimeError."""
    if args.named == "home":
        return [0.0] * 6, "SRDF 预设 home（全零）"
    if args.named == "demo":
        return [0.4, -0.5, 0.7, 0.0, 0.5, 0.0], "SRDF 预设 demo"
    if args.joints is not None:
        return [float(v) for v in args.joints], "命令行给定关节角"

    if args.grasp_item is not None:
        config = load_config(args.yaml)
        items = [item for item in config["items"] if item["id"] == args.grasp_item]
        if not items:
            raise RuntimeError(f"物品表里没有编号 {args.grasp_item}")
        item = items[0]
        position = grasp_position(config, item, args.approach_height)
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


def find_ik(node, position, repeats):
    """按 32 个候选姿态（重复 repeats 轮）找一个 IK 解，返回 (标签, 关节角)。"""
    for _ in range(max(repeats, 1)):
        for label, quaternion in candidate_orientations():
            joints = node.solve_ik(position, quaternion)
            if joints is not None:
                return label, joints
    return None, None


def status_bucket(status):
    """把结果字符串归到汇总表的四类里。"""
    if status.startswith("通过") or status.startswith("规划通过"):
        return "ok"
    if status == "IK 失败":
        return "ik"
    if status.startswith("规划失败") or status.startswith("发送失败"):
        return "plan"
    return "exec"


def sweep(node, config, subsets, args):
    """逐件跑 IK → 规划 → 执行，返回每个物品一行结果。"""
    def go_home():
        """规划回全零位，失败只告警（起始位形可能已经贴着桌面）。"""
        home = [0.0] * len(JOINTS)
        begin = node.wait_for_joint_states()
        if max(abs(a - b) for a, b in zip(begin, home)) <= 0.05:
            return
        node.get_logger().info("先回零位，把起始位形归零……")
        try:
            result = node.send_goal(node.move_goal(home, args))
            if result.error_code.val == 1 and not args.plan_only:
                node.wait_until_settled(home, args.settle_timeout, begin)
        except RuntimeError as error:
            node.get_logger().warn(f"回零位失败（{error}），继续用当前位形扫描")

    rows = []
    # 先清空场景再归零：上一轮扫描的终点可能正好停在桌面/物品里，带着碰撞状态
    # 规划每一步都会失败（实测 Subset 1 曾整组 0/10）。空场景里归零一定可行。
    node.set_scene(config, [], with_table=False)
    go_home()
    for subset in subsets:
        items = [item for item in config["items"] if item["subset"] == subset]
        node.set_scene(config, items, with_table=True)
        go_home()
        node.get_logger().info(
            f"Subset {subset}：场景已就位，开始逐件 IK → 规划 → 执行（{len(items)} 件）"
        )
        for item in items:
            position = grasp_position(config, item, args.approach_height)
            label, joint_goal = find_ik(node, position, args.ik_repeat)
            row = {
                "id": item["id"],
                "subset": subset,
                "name": item["name_cn"],
                "position": position,
                "ik_pose": label,
                "points": 0,
                "duration": 0.0,
                "error_deg": None,
                "status": "IK 失败",
            }
            if joint_goal is not None:
                begin = node.wait_for_joint_states()
                try:
                    result = node.send_goal(node.move_goal(joint_goal, args))
                except RuntimeError as error:
                    row["status"] = f"发送失败（{error}）"
                else:
                    code = result.error_code.val
                    row["points"], row["duration"] = trajectory_stats(result.planned_trajectory)
                    if code != 1:
                        row["status"] = f"规划失败（{code}）"
                    elif args.plan_only:
                        row["status"] = "规划通过（未执行）"
                    else:
                        error, moved, _ = node.wait_until_settled(
                            joint_goal, args.settle_timeout, begin
                        )
                        row["error_deg"] = math.degrees(error)
                        if error > args.tolerance and moved < 0.01:
                            row["status"] = "没动"
                        elif error > args.tolerance:
                            row["status"] = "未到位"
                        else:
                            row["status"] = "通过"
            rows.append(row)
            detail = f"  {row['status']:<14s} #{item['id']:2d} {item['name_cn']:<12s}"
            if label:
                detail += f" IK {label}"
            if row["points"]:
                detail += f" | {row['points']} 点/{row['duration']:.2f} s"
            if row["error_deg"] is not None:
                detail += f" | 误差 {row['error_deg']:.2f}°"
            node.get_logger().info(detail)
    return rows


def write_sweep_report(path, args, rows):
    buckets = {row["id"]: status_bucket(row["status"]) for row in rows}
    subsets = sorted({row["subset"] for row in rows})
    lines = [
        "# 赛道一 规划 + 执行逐件验证",
        "",
        f"- 目标点：物品顶面中心 + {args.approach_height:.2f} m（法兰位置，未计爪长）",
        f"- 每件 IK：32 个候选姿态 x {max(args.ik_repeat, 1)} 轮",
        f"- 规划：RRTConnect，每件最多 {args.attempts} 次尝试、单次 {args.planning_time:g} s",
        f"- 执行：速度/加速度缩放 {args.velocity_scaling:g}/{args.accel_scaling:g}，"
        f"到位判据 = 实际移动 ≥ 0.01 rad 且跟踪误差 ≤ {math.degrees(args.tolerance):.2f}°",
        "- 每个 Subset 开始前先回全零位（home）",
        "- 相邻物品之间不回避让位，直接从上一件的终点继续规划（模拟连续作业）",
        "",
        "## 汇总",
        "",
        "| Subset | 通过 | 总数 | IK 失败 | 规划失败 | 执行未到位 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for subset in subsets:
        subset_rows = [row for row in rows if row["subset"] == subset]
        count = lambda kind: sum(1 for row in subset_rows if buckets[row["id"]] == kind)  # noqa: E731
        lines.append(
            f"| {subset} | {count('ok')} | {len(subset_rows)} | {count('ik')} | "
            f"{count('plan')} | {count('exec')} |"
        )
    total = lambda kind: sum(1 for kind_value in buckets.values() if kind_value == kind)  # noqa: E731
    lines.append(
        f"| **合计** | **{total('ok')}** | **{len(rows)}** | **{total('ik')}** | "
        f"**{total('plan')}** | **{total('exec')}** |"
    )
    lines += [
        "",
        "> 通过 = IK 成功、OMPL 规划成功、Gazebo 里到位；IK 失败/规划失败是 40 件"
        "物品逐件实测出来的，不是估计。执行列失败时先看这一件是不是刚好在机械臂"
        "的奇异位形附近（日志里看轨迹点数和误差）。",
    ]
    for subset in subsets:
        subset_rows = [row for row in rows if row["subset"] == subset]
        ok = sum(1 for row in subset_rows if buckets[row["id"]] == "ok")
        lines += [
            "",
            f"## Subset {subset}（{ok}/{len(subset_rows)}）",
            "",
            "| # | 物品 | 目标 (x,y,z) | IK 姿态 | 轨迹 | 跟踪误差 | 结果 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in subset_rows:
            trajectory = f"{row['points']} 点 / {row['duration']:.2f} s" if row["points"] else "-"
            error = f"{row['error_deg']:.2f}°" if row["error_deg"] is not None else "-"
            lines.append(
                f"| {row['id']} | {row['name']} | "
                f"({row['position'][0]:+.2f}, {row['position'][1]:+.2f}, {row['position'][2]:.2f}) | "
                f"{row['ik_pose'] or '-'} | {trajectory} | {error} | {row['status']} |"
            )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    node = PlanExecuteDemo()
    exit_code = 0
    try:
        node.wait_for_services()
        if args.sweep_all or args.sweep_subset:
            config = load_config(args.yaml)
            subsets = (
                sorted({item["subset"] for item in config["items"]})
                if args.sweep_all
                else [args.sweep_subset]
            )
            rows = sweep(node, config, subsets, args)
            if args.out:
                write_sweep_report(args.out, args, rows)
                node.get_logger().info(f"报告已写入 {args.out}")
            ok = sum(1 for row in rows if status_bucket(row["status"]) == "ok")
            node.get_logger().info(f"逐件结果：{ok}/{len(rows)} 件走完 IK → 规划 → 执行")
            return 0 if ok == len(rows) else 1

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
        if max_error > args.tolerance:
            if moved < 0.01:
                raise RuntimeError("闭环失败：机械臂收到轨迹但几乎没动")
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
