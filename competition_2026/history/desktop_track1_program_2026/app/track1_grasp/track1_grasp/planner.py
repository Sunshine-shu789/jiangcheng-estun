"""Camera-to-base geometry and Cartesian waypoint planning."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from .config import Track1Config
from .domain import Detection, MotionPlan, Pose, RobotState


def euler_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def transform_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = euler_to_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def transform_point(transform: np.ndarray, point: Sequence[float]) -> np.ndarray:
    source = np.asarray(point, dtype=float)
    if source.shape != (3,):
        raise ValueError("point must be a 3-vector")
    return transform[:3, :3] @ source + transform[:3, 3]


def load_extrinsics(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "matrix" in payload:
        matrix = np.asarray(payload["matrix"], dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("extrinsics matrix must be 4x4")
        return matrix
    if "xyz" not in payload:
        raise ValueError("extrinsics file needs matrix or xyz")
    return transform_from_xyz_rpy(
        payload["xyz"], payload.get("rpy", (0.0, 0.0, 0.0))
    )


class PosePlanner:
    """Convert detections into bounded, collision-unaware Cartesian waypoints.

    The first version intentionally leaves collision-aware planning to MoveIt
    or the controller's Cartesian planner.  It provides the same plan object
    to either backend, so the orchestration remains stable.
    """

    def __init__(self, config: Track1Config, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self._extrinsics: np.ndarray | None = None

    def _camera_transform(self) -> np.ndarray:
        if self._extrinsics is not None:
            return self._extrinsics
        path = self.config.pose.extrinsics_file
        if not path:
            raise RuntimeError(
                "camera-to-base calibration is missing; install/calibrate the "
                "camera or enable pose.fallback_camera_to_base for mock mode"
            )
        self._extrinsics = load_extrinsics(path)
        self._log(f"[pose] loaded camera transform from {path}")
        return self._extrinsics

    def detection_to_base(self, detection: Detection, robot_state: RobotState) -> np.ndarray:
        if self.config.pose.use_detection_base_xyz and detection.base_xyz is not None:
            return np.asarray(detection.base_xyz, dtype=float)
        if detection.camera_xyz is None:
            raise ValueError("detection has neither base_xyz nor camera_xyz")

        camera_xyz = np.asarray(detection.camera_xyz, dtype=float)
        if self.config.pose.fallback_camera_to_base:
            offset = np.asarray(
                self.config.pose.fixed_camera_to_base_xyz, dtype=float
            )
            return camera_xyz + offset

        transform = self._camera_transform()
        if self.config.pose.camera_frame == "base":
            return transform_point(transform, camera_xyz)
        if robot_state.pose is None:
            raise RuntimeError("flange camera mode needs a current robot pose")
        base_flange = transform_from_xyz_rpy(robot_state.pose.xyz, robot_state.pose.rpy)
        return transform_point(base_flange @ transform, camera_xyz)

    def _check_workspace(self, pose: Pose) -> None:
        low = np.asarray(self.config.motion.workspace_min, dtype=float)
        high = np.asarray(self.config.motion.workspace_max, dtype=float)
        margin = float(self.config.motion.workspace_margin)
        point = np.asarray(pose.xyz, dtype=float)
        if np.any(point < low + margin) or np.any(point > high - margin):
            raise ValueError(
                f"pose {tuple(round(float(v), 4) for v in point)} "
                "is outside the configured workspace"
            )

    def _orientation(self, robot_state: RobotState) -> tuple[float, float, float]:
        if self.config.motion.orientation == "fixed":
            return self.config.motion.fixed_rpy
        if robot_state.pose is None:
            raise RuntimeError("orientation=current needs a current robot pose")
        return robot_state.pose.rpy

    def plan(
        self,
        detection: Detection,
        robot_state: RobotState,
        *,
        place_pose: Sequence[float] | None = None,
        place_name: str = "",
        return_to_pick: bool = False,
    ) -> MotionPlan:
        """Build the Cartesian waypoints for one pick cycle.

        ``return_to_pick`` creates an in-place pick-and-return plan: transfer,
        release and retreat all stay above the observed grasp XY coordinates,
        and the object is released at the same grasp pose from which it was
        lifted.  This is the required Task B behaviour.

        Without ``return_to_pick``, ``place_pose`` overrides
        ``motion.place_pose`` for this cycle and the object is moved to that
        configured drop pose.  Tasks C/D/E use that legacy mode.
        """
        base_xyz = self.detection_to_base(detection, robot_state)
        rpy = self._orientation(robot_state)
        grasp_xyz = base_xyz.copy()
        grasp_xyz[2] += float(self.config.motion.grasp_z_offset)

        grasp_pose = Pose(tuple(float(v) for v in grasp_xyz), rpy)
        pregrasp_pose = Pose(
            (
                float(grasp_xyz[0]),
                float(grasp_xyz[1]),
                float(grasp_xyz[2] + self.config.motion.approach_height),
            ),
            rpy,
        )
        lift_pose = Pose(
            (
                float(grasp_xyz[0]),
                float(grasp_xyz[1]),
                float(grasp_xyz[2] + self.config.motion.lift_height),
            ),
            rpy,
        )

        transfer_pose: Pose | None = None
        release_pose: Pose | None = None
        retreat_pose: Pose | None = None
        if return_to_pick:
            # Do not use the legacy fixed drop pose here.  The object must be
            # returned to the exact XY location where it was observed and
            # grasped.  The transfer and retreat waypoints stay on the same
            # vertical line so the return motion cannot drag the object to a
            # different reference point.
            release_xyz = tuple(float(v) for v in grasp_xyz)
            transfer_pose = Pose(
                (
                    release_xyz[0],
                    release_xyz[1],
                    release_xyz[2] + self.config.motion.place_approach_height,
                ),
                rpy,
            )
            release_pose = Pose(release_xyz, rpy)
            retreat_pose = Pose(
                (
                    release_xyz[0],
                    release_xyz[1],
                    release_xyz[2] + self.config.motion.retreat_height,
                ),
                rpy,
            )
        else:
            effective_place = (
                place_pose if place_pose is not None else self.config.motion.place_pose
            )
            if effective_place is not None:
                place = effective_place
                place_xyz = tuple(float(v) for v in place[:3])
                place_rpy = tuple(float(v) for v in place[3:6])
                transfer_pose = Pose(
                    (
                        place_xyz[0],
                        place_xyz[1],
                        place_xyz[2] + self.config.motion.place_approach_height,
                    ),
                    place_rpy,
                )
                release_pose = Pose(place_xyz, place_rpy)
                retreat_pose = Pose(
                    (
                        place_xyz[0],
                        place_xyz[1],
                        place_xyz[2] + self.config.motion.retreat_height,
                    ),
                    place_rpy,
                )

        plan = MotionPlan(
            target=detection,
            grasp_pose=grasp_pose,
            pregrasp_pose=pregrasp_pose,
            lift_pose=lift_pose,
            transfer_pose=transfer_pose,
            release_pose=release_pose,
            retreat_pose=retreat_pose,
            place_name=place_name,
        )
        for name, pose in plan.ordered_waypoints():
            try:
                self._check_workspace(pose)
            except ValueError as exc:
                raise ValueError(f"planned waypoint {name!r} rejected: {exc}") from exc
        return plan
