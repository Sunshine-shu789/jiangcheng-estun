#!/usr/bin/env python3
"""Validate arm/tool FK, IK, collision semantics and OMPL without execution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import (
    GetMotionPlan,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)
from rclpy.node import Node


ARM_JOINTS = [f"joint{index}" for index in range(1, 7)]
FINGER_JOINTS = [
    "gripper_left_finger_joint",
    "gripper_right_finger_joint",
]
ALL_JOINTS = ARM_JOINTS + FINGER_JOINTS
FINGER_TRAVEL_M = 0.0362745
FINGERS_OPEN = [0.0, 0.0]
FINGERS_CLOSED = [-FINGER_TRAVEL_M, -FINGER_TRAVEL_M]
# Full prismatic closure is retained for the jaw-profile/TCP calculation, but
# it is not a valid rigid-finger CAD collision state: the real four-bar fingers
# rotate instead of translating through one another at their roots.  This
# smaller travel is the maximum closure that is meaningful for the generated
# rigid collision meshes and is used only as an internal collision envelope.
FINGERS_RIGID_ENVELOPE = [-0.0160, -0.0160]
TCP_FROM_FLANGE_M = 0.091751924 + 0.050

Q_KNOWN = [math.pi / 2.0, 0.0, 0.0, 0.0, math.pi / 2.0, 0.0]
EXPECTED_KNOWN_XYZ = (-0.106964, 0.123500, 0.751726)
# The controller zero pose folds the STEP links into one another.  It is a
# useful geometric reference, but it must not be used as a planning seed.
Q_START = [0.0, 0.5, -0.5, 0.0, 0.5, 0.0]
# Keep the baseline goal on a collision-free segment.  More demanding
# self-collision-adjacent paths belong in separate stress cases, not in the
# install smoke test.
Q_GOAL = [0.3, 0.6, -0.6, 0.3, 0.7, 0.5]


def _position_error(actual, expected) -> float:
    expected_xyz = (
        (expected.x, expected.y, expected.z)
        if all(hasattr(expected, axis) for axis in ("x", "y", "z"))
        else expected
    )
    return math.sqrt(
        (actual.x - expected_xyz[0]) ** 2
        + (actual.y - expected_xyz[1]) ** 2
        + (actual.z - expected_xyz[2]) ** 2
    )


def _quaternion_angle(first, second) -> float:
    dot = abs(
        first.x * second.x
        + first.y * second.y
        + first.z * second.z
        + first.w * second.w
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _quaternion_rotate(quaternion, vector) -> tuple[float, float, float]:
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    vx, vy, vz = vector
    # Unit-quaternion rotation: v + 2 * cross(q.xyz, cross(q.xyz, v) + w*v).
    cx = y * vz - z * vy + w * vx
    cy = z * vx - x * vz + w * vy
    cz = x * vy - y * vx + w * vz
    return (
        vx + 2.0 * (y * cz - z * cy),
        vy + 2.0 * (z * cx - x * cz),
        vz + 2.0 * (x * cy - y * cx),
    )


def _fill_joint_state(robot_state, arm_q, finger_q=FINGERS_OPEN) -> None:
    if len(arm_q) != len(ARM_JOINTS) or len(finger_q) != len(FINGER_JOINTS):
        raise ValueError("joint state must contain six arm and two finger values")
    robot_state.joint_state.name = ALL_JOINTS
    robot_state.joint_state.position = [
        float(value) for value in (*arm_q, *finger_q)
    ]
    robot_state.is_diff = False


class OfflineSmoke(Node):
    def __init__(self) -> None:
        super().__init__("estun_moveit_offline_smoke")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.validity_client = self.create_client(
            GetStateValidity, "/check_state_validity"
        )

    def wait_for_services(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        clients = (
            self.fk_client,
            self.ik_client,
            self.plan_client,
            self.validity_client,
        )
        for client in clients:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not client.wait_for_service(timeout_sec=remaining):
                raise RuntimeError(f"MoveIt service unavailable: {client.srv_name}")

    def _call(self, client, request, label: str, timeout_s: float):
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() > deadline:
                future.cancel()
                raise RuntimeError(f"{label} timed out after {timeout_s:.1f}s")
        if future.exception() is not None:
            raise RuntimeError(f"{label} failed: {future.exception()}")
        response = future.result()
        if response is None:
            raise RuntimeError(f"{label} returned no response")
        return response

    def compute_fk(self, arm_q, finger_q=FINGERS_OPEN):
        request = GetPositionFK.Request()
        request.header.frame_id = "base_link"
        request.fk_link_names = ["flange_link", "grasp_tcp"]
        _fill_joint_state(request.robot_state, arm_q, finger_q)
        response = self._call(self.fk_client, request, "compute_fk", 10.0)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"compute_fk error_code={response.error_code.val}")
        if len(response.pose_stamped) != 2:
            raise RuntimeError("compute_fk did not return flange_link and grasp_tcp")
        by_link = dict(
            zip(request.fk_link_names, response.pose_stamped, strict=True)
        )
        return by_link

    def compute_ik(self, seed_arm, seed_fingers, target: PoseStamped):
        request = GetPositionIK.Request()
        request.ik_request.group_name = "arm_with_tool"
        request.ik_request.ik_link_name = "grasp_tcp"
        request.ik_request.avoid_collisions = False
        request.ik_request.pose_stamped = target
        request.ik_request.timeout.sec = 2
        request.ik_request.timeout.nanosec = 0
        _fill_joint_state(request.ik_request.robot_state, seed_arm, seed_fingers)
        response = self._call(self.ik_client, request, "compute_ik", 10.0)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"compute_ik error_code={response.error_code.val}")
        names = list(response.solution.joint_state.name)
        positions = list(response.solution.joint_state.position)
        by_name = dict(zip(names, positions, strict=False))
        missing = [name for name in ARM_JOINTS if name not in by_name]
        if missing:
            raise RuntimeError(f"compute_ik omitted arm joints: {missing}")
        return {
            "solution_rad": [float(by_name[name]) for name in ARM_JOINTS],
            "returned_joint_names": names,
        }

    def check_state_validity(self, arm_q, finger_q):
        request = GetStateValidity.Request()
        # Empty group means whole-robot validation, including the finger
        # branches that are not part of the arm_with_tool chain.
        request.group_name = ""
        _fill_joint_state(request.robot_state, arm_q, finger_q)
        response = self._call(self.validity_client, request, "check_state_validity", 10.0)
        contacts = [
            {
                "body_1": contact.contact_body_1,
                "body_2": contact.contact_body_2,
                "depth_m": float(contact.depth),
                "position_m": [
                    contact.position.x,
                    contact.position.y,
                    contact.position.z,
                ],
            }
            for contact in response.contacts
        ]
        return {"valid": bool(response.valid), "contacts": contacts}

    def plan_to_joint_goal(self, start_arm, start_fingers, goal_arm):
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "arm_with_tool"
        motion.pipeline_id = "ompl"
        motion.planner_id = "RRTConnect"
        motion.num_planning_attempts = 5
        motion.allowed_planning_time = 8.0
        motion.max_velocity_scaling_factor = 0.1
        motion.max_acceleration_scaling_factor = 0.1
        motion.workspace_parameters.header.frame_id = "base_link"
        motion.workspace_parameters.min_corner.x = -0.9
        motion.workspace_parameters.min_corner.y = -0.9
        motion.workspace_parameters.min_corner.z = -0.2
        motion.workspace_parameters.max_corner.x = 0.9
        motion.workspace_parameters.max_corner.y = 0.9
        motion.workspace_parameters.max_corner.z = 1.2
        _fill_joint_state(motion.start_state, start_arm, start_fingers)

        goal_constraints = Constraints()
        for name, position in zip(ARM_JOINTS, goal_arm, strict=True):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(position)
            constraint.tolerance_above = 1e-4
            constraint.tolerance_below = 1e-4
            constraint.weight = 1.0
            goal_constraints.joint_constraints.append(constraint)
        motion.goal_constraints = [goal_constraints]

        response = self._call(self.plan_client, request, "plan_kinematic_path", 20.0)
        result = response.motion_plan_response
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"planning error_code={result.error_code.val}")
        trajectory = result.trajectory.joint_trajectory
        if not trajectory.points:
            raise RuntimeError("planning returned an empty trajectory")
        times = [
            point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            for point in trajectory.points
        ]
        if any(second <= first for first, second in zip(times, times[1:])):
            raise RuntimeError(f"trajectory timestamps are not strictly increasing: {times}")
        return {
            "planning_time_s": float(result.planning_time),
            "trajectory_points": len(trajectory.points),
            "duration_s": times[-1],
            "joint_names": list(trajectory.joint_names),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rclpy.init()
    node = OfflineSmoke()
    report = {
        "status": "fail",
        "safety": {
            "trajectory_execution": "disabled",
            "robot_motion_commands_sent": False,
            "gripper_commands_sent": False,
        },
    }
    try:
        node.wait_for_services(args.service_timeout)

        known_poses = node.compute_fk(Q_KNOWN, FINGERS_OPEN)
        flange_pose = known_poses["flange_link"].pose
        tcp_pose = known_poses["grasp_tcp"].pose
        known_position_error = _position_error(flange_pose.position, EXPECTED_KNOWN_XYZ)
        if known_position_error > 1e-6:
            raise RuntimeError(
                f"known-pose FK position error {known_position_error:.9g} m exceeds 1e-6 m"
            )

        expected_tcp_offset = _quaternion_rotate(
            flange_pose.orientation, (0.0, 0.0, TCP_FROM_FLANGE_M)
        )
        expected_tcp_position = (
            flange_pose.position.x + expected_tcp_offset[0],
            flange_pose.position.y + expected_tcp_offset[1],
            flange_pose.position.z + expected_tcp_offset[2],
        )
        tcp_mount_error = _position_error(tcp_pose.position, expected_tcp_position)
        if tcp_mount_error > 1e-6:
            raise RuntimeError(
                f"grasp_tcp mount error {tcp_mount_error:.9g} m exceeds 1e-6 m"
            )

        target = PoseStamped()
        target.header.frame_id = "base_link"
        target.pose = tcp_pose
        ik = node.compute_ik(Q_KNOWN, FINGERS_OPEN, target)
        ik_fk = node.compute_fk(ik["solution_rad"], FINGERS_OPEN)["grasp_tcp"]
        ik_position_error = _position_error(
            ik_fk.pose.position, target.pose.position
        )
        ik_orientation_error = _quaternion_angle(
            ik_fk.pose.orientation, target.pose.orientation
        )
        if ik_position_error > 1e-4:
            raise RuntimeError(
                f"IK round-trip position error {ik_position_error:.9g} m exceeds 1e-4 m"
            )
        if ik_orientation_error > 1e-3:
            raise RuntimeError(
                "IK round-trip orientation error "
                f"{ik_orientation_error:.9g} rad exceeds 1e-3 rad"
            )

        states = {
            "start_open": (Q_START, FINGERS_OPEN),
            "start_rigid_close_envelope": (Q_START, FINGERS_RIGID_ENVELOPE),
            "goal_open": (Q_GOAL, FINGERS_OPEN),
            "goal_rigid_close_envelope": (Q_GOAL, FINGERS_RIGID_ENVELOPE),
        }
        validity = {
            name: node.check_state_validity(arm_q, finger_q)
            for name, (arm_q, finger_q) in states.items()
        }
        for name, result in validity.items():
            if not result["valid"]:
                raise RuntimeError(
                    f"{name} is in collision: "
                    + json.dumps(result["contacts"], ensure_ascii=False)
                )

        # Diagnostic only: the rigid mesh translation at full jaw-profile
        # closure is known to be non-physical near the finger roots.
        full_closed_diagnostic = node.check_state_validity(Q_START, FINGERS_CLOSED)

        planning = node.plan_to_joint_goal(Q_START, FINGERS_OPEN, Q_GOAL)
        report.update(
            {
                "status": "pass",
                "model_scope": {
                    "planning_group": "arm_with_tool",
                    "tcp_link": "grasp_tcp",
                    "finger_joints": FINGER_JOINTS,
                    "finger_travel_m": FINGER_TRAVEL_M,
                    "finger_rigid_envelope_m": FINGERS_RIGID_ENVELOPE,
                },
                "fk_known_pose": {
                    "joints_rad": Q_KNOWN,
                    "fingers_m": FINGERS_OPEN,
                    "flange_position_m": [
                        flange_pose.position.x,
                        flange_pose.position.y,
                        flange_pose.position.z,
                    ],
                    "flange_position_error_m": known_position_error,
                    "tcp_position_m": [
                        tcp_pose.position.x,
                        tcp_pose.position.y,
                        tcp_pose.position.z,
                    ],
                    "tcp_mount_error_m": tcp_mount_error,
                },
                "ik_round_trip": {
                    "group_name": "arm_with_tool",
                    "ik_link_name": "grasp_tcp",
                    "solution_rad": ik["solution_rad"],
                    "returned_joint_names": ik["returned_joint_names"],
                    "position_error_m": ik_position_error,
                    "orientation_error_rad": ik_orientation_error,
                },
                "collision_check": {
                    "states": validity,
                    "full_closed_rigid_cad_diagnostic": full_closed_diagnostic,
                },
                "planning": planning,
                "modeling_caveat": (
                    "The full closed prismatic finger state is a jaw-profile/TCP "
                    "reference, not a valid rigid-CAD collision state. Candidate "
                    "filtering must use the jaw profile and flexible-link calibration."
                ),
            }
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return_code = 0
    except Exception as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return_code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
