#!/usr/bin/env python3
"""Check whether the arm can reach a grasp pose above every Track 1 object.

For each object the script asks MoveIt's ``/compute_ik`` service for an inverse
kinematics solution with the flange (link6) placed above the object and the
tool axis pointing down (plus a few tilted/yawed variants, because a straight
top-down approach is not always the only way in).  ``avoid_collisions`` is on,
so a solution that would hit the table or another object is rejected too.

Usage:

    ros2 run s3_60_eco_2024_moveit_config check_reachability.py --subset 1
    ros2 run s3_60_eco_2024_moveit_config check_reachability.py --all --out report.md
"""

import argparse
import math
import os
import sys

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from rclpy.node import Node

# 工具轴向下、绕 z 转 yaw、再往机器人这一侧倾 tilt 的候选姿态。KDL 是局部求解器，
# 候选给太少会把"其实够得着"的目标误判成不可达，所以宁可多试几个。
YAW_DEGREES = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
TILT_DEGREES = (0.0, 20.0, 40.0, 60.0)
YAW_DEGREES_QUICK = (0.0, 90.0, 180.0, 270.0)
TILT_DEGREES_QUICK = (0.0, 40.0)


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


def candidate_orientations(quick=False):
    """(标签, 四元数) 列表：工具 z 轴朝下，绕 z 转 yaw，其中一部分向 -y 倾斜。"""
    tilts = TILT_DEGREES_QUICK if quick else TILT_DEGREES
    yaws = YAW_DEGREES_QUICK if quick else YAW_DEGREES
    for tilt in tilts:
        for yaw in yaws:
            # 先把 +z 翻到 -z（Rx(pi)），再绕 x 回调 tilt 角度，让工具朝机器人一侧
            # 倾斜，最后绕 z 转 yaw。
            roll = math.pi - math.radians(tilt)
            quaternion = quaternion_from_rpy(roll, 0.0, math.radians(yaw))
            yield f"tilt{int(tilt)}_yaw{int(yaw)}", quaternion


def parse_args(argv):
    default_yaml = os.path.join(
        get_package_share_directory("s3_60_eco_2024_moveit_config"),
        "config",
        "track1_items.yaml",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", default=default_yaml)
    parser.add_argument("--subset", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--approach-height",
        type=float,
        default=0.05,
        help="法兰目标点取在物品顶面再往上多少米（默认 5 cm，换真爪后应加上爪长）",
    )
    parser.add_argument("--timeout", type=float, default=0.2, help="单次 IK 超时（秒）")
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="候选姿态重复尝试轮数（KDL 带随机重启，试两轮能少一些误判，默认 2）",
    )
    parser.add_argument("--out", help="把结果写成 markdown 报告")
    parser.add_argument("--group", default="arm")
    parser.add_argument("--tip-link", default="link6")
    return parser.parse_args(argv)


class ReachabilityChecker(Node):
    def __init__(self, timeout):
        super().__init__("track1_reachability")
        self.timeout = timeout
        self.client = self.create_client(GetPositionIK, "compute_ik")
        if not self.client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError("等不到 /compute_ik 服务；move_group 起来了吗？")
        self.validity_client = self.create_client(GetStateValidity, "check_state_validity")
        self.validity_client.wait_for_service(timeout_sec=30.0)

    def current_state_contacts(self):
        """当前关节状态下机器人与场景的碰撞接触（空列表 = 无碰撞）。"""
        request = GetStateValidity.Request()
        request.group_name = "arm"
        future = self.validity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None:
            return None
        return [
            f"{c.contact_body_1} <-> {c.contact_body_2}" for c in response.contacts
        ]

    def solve(self, frame, position, quaternion):
        request = GetPositionIK.Request()
        request.ik_request.group_name = "arm"
        request.ik_request.ik_link_name = "link6"
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = int(self.timeout)
        request.ik_request.timeout.nanosec = int((self.timeout % 1) * 1e9)
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = quaternion
        request.ik_request.pose_stamped = pose

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout + 2.0)
        if future.result() is None:
            return None
        response = future.result()
        if response.error_code.val != 1:  # 1 == SUCCESS
            return None
        joints = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        return joints


def item_target(item, config, approach_height):
    """物品顶面中心再抬高 approach_height，作为法兰目标点。"""
    size = item["size"]
    if item["shape"] == "box":
        height = float(size[2])
    elif item["shape"] == "cylinder":
        height = float(size[1])
    else:
        height = float(size[0])
    # 位置由 layout 决定；调用方把 x/y 传进来更直接，这里只算高度。
    return height, float(config["table"]["top_z"]) + height + approach_height


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    with open(args.yaml, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    items = [i for i in config["items"] if args.all or i["subset"] == args.subset]
    layout = config["layout"]

    rclpy.init()
    node = ReachabilityChecker(args.timeout)

    contacts = node.current_state_contacts()
    if contacts:
        node.get_logger().warn(
            f"当前状态就和场景碰撞了（{len(contacts)} 处），IK 一定会大量失败，"
            "先修场景摆位："
        )
        for contact in contacts[:10]:
            node.get_logger().warn(f"  碰撞: {contact}")
    else:
        node.get_logger().info("当前状态无碰撞（场景摆放没问题）")

    node.get_logger().info(
        f"检查 Subset {args.subset}：{len(items)} 件，候选姿态 {len(list(candidate_orientations()))} 个/件"
    )

    rows = []
    for index, item in enumerate(items):
        column = index % int(layout["columns"])
        row = index // int(layout["columns"])
        x = float(layout["x_start"]) + column * float(layout["x_step"])
        y = float(layout["y_start"]) + row * float(layout["y_step"])
        height, z = item_target(item, config, args.approach_height)

        success_label, joints = None, None
        for _ in range(max(args.repeat, 1)):
            for label, quaternion in candidate_orientations():
                joints = node.solve(config["frame_id"], (x, y, z), quaternion)
                if joints is not None:
                    success_label = label
                    break
            if success_label:
                break
        distance = math.hypot(x, y)
        rows.append(
            {
                "id": item["id"],
                "name": item["name_cn"],
                "shape": item["shape"],
                "size": item["size"],
                "source": item["source"],
                "x": x,
                "y": y,
                "z": z,
                "distance": distance,
                "pose": success_label,
                "joints": joints,
            }
        )
        flag = "✓" if success_label else "✗"
        node.get_logger().info(
            f"  {flag} {item['id']:2d} {item['name_cn']:<12s} "
            f"目标({x:+.2f},{y:+.2f},{z:.2f}) 水平距离 {distance:.2f} m "
            f"{'姿态 ' + success_label if success_label else ''}"
        )

    ok = [r for r in rows if r["pose"]]
    node.get_logger().info(f"结果：{len(ok)}/{len(rows)} 件解出了抓取位姿")

    if args.out:
        lines = [
            f"# 赛道一 Subset {args.subset} 可达性检查",
            "",
            f"- 目标点：物品顶面中心 + {args.approach_height:.2f} m（法兰位置，未计爪长）",
            f"- 姿态候选：{len(list(candidate_orientations()))} 个（工具轴向下，"
            f"倾斜 {TILT_DEGREES} 度、yaw {YAW_DEGREES} 度），每件重复 {args.repeat} 轮",
            "- 判定：MoveIt `/compute_ik` 返回成功且不与规划场景（桌面、其它物品）碰撞",
            "",
            "| # | 物品 | 形状 | 尺寸(m) | 目标 (x,y,z) | 水平距离 | 可达 | 成功的姿态 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            lines.append(
                f"| {r['id']} | {r['name']} | {r['shape']} | {r['size']} | "
                f"({r['x']:+.2f}, {r['y']:+.2f}, {r['z']:.2f}) | {r['distance']:.2f} m | "
                f"{'是' if r['pose'] else '否'} | {r['pose'] or '-'} |"
            )
        lines += [
            "",
            f"**合计：{len(ok)}/{len(rows)} 件可达。**",
            "",
            "> 说明：尺寸数据来源见 `track1_items.yaml` 的 `source` 字段（ycb近似/估计/待实测），",
            "> 未核对实物前，本表只作工作空间参考。",
        ]
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        node.get_logger().info(f"报告已写入 {args.out}")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
