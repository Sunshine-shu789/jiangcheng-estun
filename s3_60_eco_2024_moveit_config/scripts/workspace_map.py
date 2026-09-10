#!/usr/bin/env python3
"""Map the reachable workspace of the arm for top-down grasp poses.

在给定的高度 z 上扫一个 x-y 网格，对每个格子问 /compute_ik（工具轴向下，
含倾斜与多个 yaw 的候选姿态），打印一张 ASCII 图，用来决定桌子该摆多远、
物品该放在哪一区域。

    ros2 run s3_60_eco_2024_moveit_config workspace_map.py                 # z=0.35
    ros2 run s3_60_eco_2024_moveit_config workspace_map.py --z 0.40 --x-step 0.05
    ros2 run s3_60_eco_2024_moveit_config workspace_map.py --out map.md
"""

import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node

from check_reachability import candidate_orientations


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z", type=float, default=0.35, help="扫描平面高度（米）")
    parser.add_argument("--x-min", type=float, default=-0.50)
    parser.add_argument("--x-max", type=float, default=0.50)
    parser.add_argument("--x-step", type=float, default=0.10)
    parser.add_argument("--y-min", type=float, default=0.15)
    parser.add_argument("--y-max", type=float, default=0.85)
    parser.add_argument("--y-step", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=0.15)
    parser.add_argument("--out", help="把 ASCII 图存成 markdown")
    return parser.parse_args(argv)


def frange(start, stop, step):
    values = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 6))
        value += step
    return values


class WorkspaceMapper(Node):
    def __init__(self, timeout):
        super().__init__("track1_workspace_map")
        self.timeout = timeout
        self.client = self.create_client(GetPositionIK, "compute_ik")
        if not self.client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError("等不到 /compute_ik；move_group 起来了吗？")

    def reachable(self, position, quaternions):
        for quaternion in quaternions:
            request = GetPositionIK.Request()
            request.ik_request.group_name = "arm"
            request.ik_request.ik_link_name = "link6"
            request.ik_request.avoid_collisions = True
            request.ik_request.timeout.nanosec = int(self.timeout * 1e9)
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
            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout + 2.0)
            response = future.result()
            if response is not None and response.error_code.val == 1:
                return True
        return False


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    xs = frange(args.x_min, args.x_max, args.x_step)
    ys = frange(args.y_min, args.y_max, args.y_step)
    quaternions = [q for _, q in candidate_orientations(quick=True)]

    rclpy.init()
    node = WorkspaceMapper(args.timeout)
    node.get_logger().info(
        f"扫描 z={args.z:.2f} m：x {xs[0]:+.2f}..{xs[-1]:+.2f}（{len(xs)} 列）、"
        f"y {ys[0]:.2f}..{ys[-1]:.2f}（{len(ys)} 行）"
    )

    grid = {}
    for y in sorted(ys, reverse=True):
        row = []
        for x in xs:
            ok = node.reachable((x, y, args.z), quaternions)
            grid[(x, y)] = ok
            row.append("■" if ok else "·")
        node.get_logger().info(f"y={y:.2f}  " + " ".join(row))

    reachable_cells = [cell for cell, ok in grid.items() if ok]
    node.get_logger().info(f"可达格子 {len(reachable_cells)}/{len(grid)}")
    if reachable_cells:
        ys_ok = sorted({cell[1] for cell in reachable_cells})
        xs_ok = sorted({cell[0] for cell in reachable_cells})
        node.get_logger().info(
            f"可达范围：y {ys_ok[0]:.2f}..{ys_ok[-1]:.2f} m，x {xs_ok[0]:+.2f}..{xs_ok[-1]:+.2f} m"
        )

    if args.out:
        lines = [
            f"# 工作空间扫描（z = {args.z:.2f} m，工具轴向下）",
            "",
            f"网格：x {args.x_min:+.2f}..{args.x_max:+.2f} 步长 {args.x_step:.2f}，"
            f"y {args.y_min:.2f}..{args.y_max:.2f} 步长 {args.y_step:.2f}",
            "",
            "```",
            "        " + " ".join(f"{x:+.2f}" for x in xs),
        ]
        for y in sorted(ys, reverse=True):
            row = " ".join("  ■ " if grid[(x, y)] else "  · " for x in xs)
            lines.append(f"y={y:.2f} {row}")
        lines += ["```", "", "■ = 有 IK 解（且不与场景碰撞），· = 解不出来"]
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        node.get_logger().info(f"结果已写入 {args.out}")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
