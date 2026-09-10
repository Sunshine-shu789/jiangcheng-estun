#!/usr/bin/env python3
"""Drive the S3-60 Eco 2024 through the joint_trajectory_controller.

Reads the measured joint positions from /joint_states, sends a JointTrajectory
goal to the controller and reports how far the arm lagged behind the command,
so a failed goal or an arm that ignores its commands is visible right away.

With Gazebo running:

    ros2 run s3_60_eco_2024 demo_trajectory.py            # demo pose
    ros2 run s3_60_eco_2024 demo_trajectory.py --home     # all joints to 0
    ros2 run s3_60_eco_2024 demo_trajectory.py 0 -0.5 0.7 0 0.5 0
"""

import argparse
import math
import sys
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = [f"joint{i}" for i in range(1, 7)]
CONTROLLER = "joint_trajectory_controller"
# A pose the arm can reach easily, chosen to make the motion obvious in RViz
# and Gazebo without going anywhere near the joint limits.
DEMO_POSE = [0.4, -0.5, 0.7, 0.0, 0.5, 0.0]


class TrajectoryDemo(Node):
    def __init__(self):
        super().__init__("s3_60_eco_2024_demo")
        self.positions = [0.0] * len(JOINTS)
        self.received = False
        self.create_subscription(JointState, "joint_states", self.on_joint_state, 10)
        self.client = ActionClient(
            self, FollowJointTrajectory, f"{CONTROLLER}/follow_joint_trajectory"
        )

    def on_joint_state(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name in JOINTS:
                self.positions[JOINTS.index(name)] = position
                self.received = True

    def wait_for_joint_states(self, timeout):
        """Return the measured joint positions once /joint_states is alive."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.received:
                return list(self.positions)
        raise TimeoutError(f"no /joint_states received within {timeout:.1f} s")

    def send(self, start, target, duration):
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINTS
        first = JointTrajectoryPoint()
        first.positions = start
        first.time_from_start.sec = 0
        first.time_from_start.nanosec = int(0.5e9)
        last = JointTrajectoryPoint()
        last.positions = target
        last.time_from_start.sec = int(duration)
        last.time_from_start.nanosec = int((duration % 1) * 1e9)
        trajectory.points = [first, last]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        if not self.client.wait_for_server(timeout_sec=30.0):
            raise TimeoutError(
                f"action server {CONTROLLER}/follow_joint_trajectory not available; "
                "is the controller active?"
            )
        self.get_logger().info(
            "sending goal: " + " ".join(f"{value:+.3f}" for value in target)
        )
        handle = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, handle)
        goal_handle = handle.result()
        if not goal_handle.accepted:
            raise RuntimeError("goal rejected by the controller")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result().result

    def settle(self, seconds):
        end = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.1)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Send a demo trajectory to the S3-60 Eco 2024 controller."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        type=float,
        help="six joint angles in radians; defaults to a built-in demo pose",
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="move all joints back to zero instead of the demo pose",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="seconds the trajectory should take (default: 4)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="accepted final tracking error in radians (default: 0.05)",
    )
    args = parser.parse_args(argv)
    if args.targets and args.home:
        parser.error("give either explicit targets or --home, not both")
    if args.targets and len(args.targets) != len(JOINTS):
        parser.error(f"expected {len(JOINTS)} joint angles, got {len(args.targets)}")
    if not args.targets:
        args.targets = [0.0] * len(JOINTS) if args.home else list(DEMO_POSE)
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    node = TrajectoryDemo()
    try:
        start = node.wait_for_joint_states(timeout=30.0)
        node.get_logger().info(
            "measured start: " + " ".join(f"{value:+.3f}" for value in start)
        )
        result = node.send(start, args.targets, args.duration)
        node.settle(2.0)
        measured = list(node.positions)
        error = max(abs(a - b) for a, b in zip(measured, args.targets))
        node.get_logger().info(
            "measured end:   " + " ".join(f"{value:+.3f}" for value in measured)
        )
        node.get_logger().info(
            f"controller error_code={result.error_code}, "
            f"max tracking error={math.degrees(error):.2f} deg"
        )
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            node.get_logger().error(f"trajectory failed: {result.error_string}")
            return 1
        if error > args.tolerance:
            node.get_logger().error(
                f"the arm did not reach the target (tolerance "
                f"{math.degrees(args.tolerance):.2f} deg)"
            )
            return 1
        node.get_logger().info("trajectory executed, the arm followed the command")
        return 0
    except (TimeoutError, RuntimeError) as error:
        node.get_logger().error(str(error))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
