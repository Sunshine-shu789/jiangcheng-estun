"""Camera-to-base geometry and Cartesian waypoint planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from .config import Track1Config
from .domain import Detection, MotionPlan, Pose, RobotState
from .kinematics import matrix_to_rpy, rotation_angle_error, rotation_matrix_to_vector


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


@dataclass(frozen=True)
class AdaptiveGraspCandidate:
    """One object-relative grasp orientation and its complete motion plan."""

    plan: MotionPlan
    yaw_offset_deg: float
    orientation_change_rad: float
    object_center_base_xyz: tuple[float, float, float]
    major_axis_base: tuple[float, float, float]
    pitch_offset_deg: float = 0.0


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


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
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

    def _base_flange_transform(self, robot_state: RobotState) -> np.ndarray:
        """Return T_base_flange from the controller's declared pose semantics."""

        if robot_state.pose is None:
            raise RuntimeError("adaptive grasp needs a current robot pose")
        base_controller = transform_from_xyz_rpy(
            robot_state.pose.xyz,
            robot_state.pose.rpy,
        )
        frame = self.config.pose.robot_pose_frame
        if frame == "flange":
            return base_controller
        if frame == "tcp":
            flange_tcp = np.eye(4)
            flange_tcp[:3, 3] = np.asarray(
                self.config.pose.tcp_offset_m,
                dtype=float,
            )
            return base_controller @ np.linalg.inv(flange_tcp)
        raise RuntimeError(
            "adaptive grasp requires pose.robot_pose_frame=flange or tcp"
        )

    def _adaptive_object_geometry(
        self,
        detection: Detection,
        robot_state: RobotState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if detection.object_center_camera_xyz is None:
            raise ValueError("adaptive grasp needs object_center_camera_xyz")
        if detection.major_axis_camera is None:
            raise ValueError("adaptive grasp needs major_axis_camera")
        adaptive = self.config.adaptive_inverse
        if detection.pose_confidence is None:
            raise ValueError("adaptive grasp needs pose_confidence")
        if float(detection.pose_confidence) < float(adaptive.min_pose_confidence):
            raise ValueError(
                "object pose confidence "
                f"{float(detection.pose_confidence):.3f} is below "
                f"{float(adaptive.min_pose_confidence):.3f}"
            )

        base_flange = self._base_flange_transform(robot_state)
        flange_camera = self._camera_transform()
        base_camera = base_flange @ flange_camera
        center_base = transform_point(
            base_camera,
            detection.object_center_camera_xyz,
        )
        major_axis_base = (
            base_camera[:3, :3]
            @ np.asarray(detection.major_axis_camera, dtype=float)
        )
        major_norm = float(np.linalg.norm(major_axis_base))
        if not np.isfinite(major_norm) or major_norm <= 1e-12:
            raise ValueError("object major axis is not finite after transform")
        major_axis_base = major_axis_base / major_norm
        horizontal_norm = float(np.linalg.norm(major_axis_base[:2]))
        if horizontal_norm <= 1e-6:
            raise ValueError(
                "object major axis has no usable horizontal projection"
            )
        return center_base, major_axis_base, base_flange

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    @staticmethod
    def _wrap_axis_delta(angle: float) -> float:
        """Return the shortest equivalent delta between undirected axes."""

        return float(
            (float(angle) + np.pi * 0.5) % np.pi - np.pi * 0.5
        )

    def plan_adaptive_candidates(
        self,
        detection: Detection,
        robot_state: RobotState,
        *,
        place_name: str = "",
        return_to_pick: bool = True,
        grasp_z_m: float | None = None,
        grasp_rpy: Sequence[float] | None = None,
        task_b_anchor_pose: Pose | None = None,
        vertical_grasp_approach_height_m: float | None = None,
    ) -> tuple[AdaptiveGraspCandidate, ...]:
        """Plan object-relative grasp orientation candidates without motion.

        Candidate headings rotate the operator-taught orientation about base Z
        so the taught tool X axis aligns with the observed object major axis.
        Each heading can then be paired with configurable signed ZYX pitch
        offsets.  This keeps upright grasping as the first preference while
        giving the route checker a wrist posture that can escape a near-
        singular boundary configuration.  Every candidate retains the same
        observed jaw-tip point and uses a purely vertical pregrasp-to-grasp
        segment.  The pipeline checks the complete route before selection or
        any command.
        """

        adaptive = self.config.adaptive_inverse
        if not adaptive.enabled:
            raise ValueError("adaptive inverse planning is not enabled")
        if not return_to_pick:
            raise ValueError(
                "adaptive Task B planning requires return_to_pick=True"
            )

        center_base, major_axis_base, _base_flange = (
            self._adaptive_object_geometry(detection, robot_state)
        )
        taught_rpy = tuple(
            float(value)
            for value in (
                grasp_rpy
                if grasp_rpy is not None
                else (
                    task_b_anchor_pose.rpy
                    if task_b_anchor_pose is not None
                    else (
                        self.config.motion.grasp_rpy
                        if self.config.motion.grasp_rpy is not None
                        else self._orientation(robot_state)
                    )
                )
            )
        )
        taught_rotation = euler_to_matrix(taught_rpy)
        taught_x_yaw = float(
            np.arctan2(taught_rotation[1, 0], taught_rotation[0, 0])
        )
        observed_yaw = float(
            np.arctan2(major_axis_base[1], major_axis_base[0])
        )
        principal_delta = self._wrap_axis_delta(
            self._wrap_angle(
                observed_yaw
                - taught_x_yaw
                + np.radians(
                    float(adaptive.candidates.axis_alignment_offset_deg)
                )
            )
        )

        effective_approach_height = float(
            adaptive.candidates.approach_height_m
            if adaptive.candidates.approach_height_m is not None
            else (
                self.config.task_b.vertical_grasp_approach_height_m
                if vertical_grasp_approach_height_m is None
                else float(vertical_grasp_approach_height_m)
            )
        )
        effective_grasp_z = (
            None
            if adaptive.candidates.use_visual_grasp_z
            else grasp_z_m
        )
        anchor = task_b_anchor_pose
        if anchor is None:
            # The vertical candidate planner only needs a marker pose.  Use a
            # conservative high point above the current/observed location; the
            # actual pregrasp Z remains grasp Z plus the configured clearance.
            base_z = (
                float(robot_state.pose.xyz[2])
                if robot_state.pose is not None
                else float(center_base[2]) + effective_approach_height
            )
            anchor = Pose(
                (
                    float(center_base[0]),
                    float(center_base[1]),
                    max(base_z, float(center_base[2]) + effective_approach_height),
                ),
                taught_rpy,
            )

        candidates: list[AdaptiveGraspCandidate] = []
        seen_rotations: list[np.ndarray] = []
        candidate_failures: list[str] = []
        max_orientation_change_rad = np.radians(
            float(adaptive.candidates.max_orientation_change_deg)
        )
        tip_offset_flange = self.config.pose.gripper_tip_offset_flange_m
        if adaptive.candidates.use_visual_grasp_z and tip_offset_flange is None:
            raise RuntimeError(
                "visual grasp requires pose.gripper_tip_offset_flange_m"
            )
        commanded_tip_base = center_base.copy()
        commanded_tip_base[2] += float(
            adaptive.candidates.jaw_tip_clearance_m
        )

        # Yaw is the outer loop so the primary object-axis alignment gets the
        # full set of safe pitch alternatives before a less-favourable heading
        # is attempted.  The first configured pitch is conventionally zero.
        for configured_yaw_deg in adaptive.candidates.yaw_offsets_deg:
            delta = self._wrap_angle(
                principal_delta + np.radians(float(configured_yaw_deg))
            )
            heading_rotation = _rotation_z(delta) @ taught_rotation
            heading_roll, heading_pitch, heading_yaw = matrix_to_rpy(
                heading_rotation
            )
            for configured_pitch_deg in adaptive.candidates.pitch_offsets_deg:
                pitch_delta = np.radians(float(configured_pitch_deg))
                candidate_rpy = (
                    float(heading_roll),
                    float(heading_pitch) + float(pitch_delta),
                    float(heading_yaw),
                )
                if abs(candidate_rpy[1]) > np.pi * 0.5 + 1e-9:
                    candidate_failures.append(
                        f"yaw {float(configured_yaw_deg):.2f} deg, "
                        f"pitch {float(configured_pitch_deg):.2f} deg exceeds "
                        "the usable ZYX pitch range"
                    )
                    continue
                candidate_rotation = euler_to_matrix(candidate_rpy)
                orientation_change_rad = rotation_angle_error(
                    candidate_rotation, taught_rotation
                )
                if (
                    orientation_change_rad
                    > max_orientation_change_rad + 1e-12
                ):
                    candidate_failures.append(
                        f"yaw {float(configured_yaw_deg):.2f} deg, "
                        f"pitch {float(configured_pitch_deg):.2f} deg gives "
                        f"{np.degrees(orientation_change_rad):.2f} deg total "
                        "orientation change, above the configured limit"
                    )
                    continue
                rotation_vector = rotation_matrix_to_vector(candidate_rotation)
                if any(
                    np.linalg.norm(rotation_vector - seen) < 1e-7
                    for seen in seen_rotations
                ):
                    continue

                grasp_flange_xyz = center_base.copy()
                if adaptive.candidates.use_visual_grasp_z:
                    assert tip_offset_flange is not None
                    tip_offset_base = (
                        candidate_rotation
                        @ np.asarray(tip_offset_flange, dtype=float)
                    )
                    # The camera observes the intended jaw-tip point.  Convert
                    # it into the flange target commanded by RobotMove, using
                    # the same tilted orientation selected for this candidate.
                    grasp_flange_xyz = commanded_tip_base - tip_offset_base
                    self._log(
                        "[adaptive-plan] yaw_offset="
                        f"{np.degrees(delta):.3f} deg pitch_offset="
                        f"{float(configured_pitch_deg):.3f} deg "
                        "support_tip_base="
                        f"({center_base[0]:.6f}, {center_base[1]:.6f}, "
                        f"{center_base[2]:.6f}) m; commanded_tip_base="
                        f"({commanded_tip_base[0]:.6f}, "
                        f"{commanded_tip_base[1]:.6f}, "
                        f"{commanded_tip_base[2]:.6f}) m; flange_target="
                        f"({grasp_flange_xyz[0]:.6f}, "
                        f"{grasp_flange_xyz[1]:.6f}, "
                        f"{grasp_flange_xyz[2]:.6f}) m; "
                        f"tip_offset_base=({tip_offset_base[0]:.6f}, "
                        f"{tip_offset_base[1]:.6f}, "
                        f"{tip_offset_base[2]:.6f}) m"
                    )
                try:
                    plan = self.plan(
                        detection,
                        robot_state,
                        place_name=place_name,
                        return_to_pick=True,
                        grasp_z_m=effective_grasp_z,
                        grasp_rpy=candidate_rpy,
                        task_b_anchor_pose=anchor,
                        grasp_xyz_override_m=grasp_flange_xyz,
                        vertical_grasp_descent=True,
                        vertical_grasp_approach_height_m=(
                            effective_approach_height
                        ),
                        use_visual_grasp_z=(
                            adaptive.candidates.use_visual_grasp_z
                        ),
                    )
                except (ValueError, RuntimeError) as exc:
                    candidate_failures.append(
                        f"yaw {float(configured_yaw_deg):.2f} deg, "
                        f"pitch {float(configured_pitch_deg):.2f} deg: {exc}"
                    )
                    self._log(
                        "[adaptive-plan] candidate rejected before route "
                        "precheck: "
                        f"yaw={np.degrees(delta):.3f} deg "
                        f"pitch={float(configured_pitch_deg):.3f} deg "
                        f"error={exc}"
                    )
                    continue
                candidates.append(
                    AdaptiveGraspCandidate(
                        plan=plan,
                        yaw_offset_deg=float(np.degrees(delta)),
                        pitch_offset_deg=float(configured_pitch_deg),
                        orientation_change_rad=float(orientation_change_rad),
                        object_center_base_xyz=tuple(
                            float(value) for value in center_base
                        ),
                        major_axis_base=tuple(
                            float(value) for value in major_axis_base
                        ),
                    )
                )
                seen_rotations.append(rotation_vector)
                if len(candidates) >= int(adaptive.candidates.max_count):
                    break
            if len(candidates) >= int(adaptive.candidates.max_count):
                break

        if not candidates:
            detail = "; ".join(candidate_failures[:5])
            raise ValueError(
                "no adaptive grasp candidate remained after geometry checks"
                + (f": {detail}" if detail else "")
            )
        return tuple(candidates)

    def plan(
        self,
        detection: Detection,
        robot_state: RobotState,
        *,
        place_pose: Sequence[float] | None = None,
        place_name: str = "",
        return_to_pick: bool = False,
        grasp_z_m: float | None = None,
        grasp_rpy: Sequence[float] | None = None,
        task_b_anchor_pose: Pose | None = None,
        grasp_xy_from_observation: bool = False,
        grasp_xy_m: Sequence[float] | None = None,
        grasp_xy_offset_m: Sequence[float] | None = None,
        grasp_xyz_override_m: Sequence[float] | None = None,
        vertical_grasp_descent: bool = False,
        vertical_grasp_approach_height_m: float | None = None,
        use_visual_grasp_z: bool = False,
    ) -> MotionPlan:
        """Build the Cartesian waypoints for one pick cycle.

        ``return_to_pick`` creates an in-place pick-and-return plan: transfer,
        release and retreat all stay above the observed grasp XY coordinates,
        and the object is released at the same grasp pose from which it was
        lifted.  This is the required Task B behaviour.

        ``task_b_anchor_pose`` optionally supplies the operator-taught
        observation pose for one Task B point.  When present, every high
        waypoint (pregrasp, lift, transfer and retreat) uses that complete
        Cartesian pose.  By default grasp and release still use the visual XY
        and the point-specific grasp height/orientation.  With
        ``grasp_xy_from_observation`` enabled they instead retain the taught
        observation anchor XY, producing a straight vertical descent from that
        validated column.  Returning to a known
        well-conditioned anchor avoids asking the IK solver to carry a large
        combined translation/orientation change through a near-singular
        wrist configuration.  ``grasp_xy_m`` supplies an explicit taught grasp
        column instead; in that case the high waypoints stay at the observation
        anchor height but move directly above that column, with the final
        descent performed at the grasp orientation.  ``grasp_xy_offset_m``
        applies a final flange/tool-frame XY correction after selecting the
        visual, observation, or explicit XY.  The correction is rotated into
        the base frame by the final grasp RPY; only its base X/Y components are
        applied so the verified grasp Z stays fixed.  All high, grasp and
        release waypoints then use the corrected column.
        ``vertical_grasp_descent``
        applies the same column lock even when XY comes from vision or the
        observation anchor, and also uses the final grasp orientation at the
        high pose.  Consequently the pregrasp-to-grasp segment has zero lateral
        and zero rotational change.

        Without ``return_to_pick``, ``place_pose`` overrides
        ``motion.place_pose`` for this cycle and the object is moved to that
        configured drop pose.  Tasks C/D/E use that legacy mode.
        """
        if task_b_anchor_pose is not None and not return_to_pick:
            raise ValueError(
                "task_b_anchor_pose is only valid with return_to_pick=True"
            )
        if grasp_xy_from_observation and not return_to_pick:
            raise ValueError(
                "grasp_xy_from_observation is only valid with "
                "return_to_pick=True"
            )
        if grasp_xy_from_observation and task_b_anchor_pose is None:
            raise ValueError(
                "grasp_xy_from_observation needs a task_b_anchor_pose"
            )
        if grasp_xy_m is not None:
            if not return_to_pick:
                raise ValueError(
                    "grasp_xy_m is only valid with return_to_pick=True"
                )
            if grasp_xy_from_observation:
                raise ValueError(
                    "grasp_xy_m and grasp_xy_from_observation are mutually "
                    "exclusive"
                )
            if len(grasp_xy_m) != 2 or not all(
                np.isfinite(float(value)) for value in grasp_xy_m
            ):
                raise ValueError("grasp_xy_m must contain two finite values")
        if grasp_xy_offset_m is not None:
            if not return_to_pick:
                raise ValueError(
                    "grasp_xy_offset_m is only valid with return_to_pick=True"
                )
            if len(grasp_xy_offset_m) != 2 or not all(
                np.isfinite(float(value)) for value in grasp_xy_offset_m
            ):
                raise ValueError(
                    "grasp_xy_offset_m must contain two finite values"
                )
        if grasp_xyz_override_m is not None:
            if len(grasp_xyz_override_m) != 3 or not all(
                np.isfinite(float(value)) for value in grasp_xyz_override_m
            ):
                raise ValueError(
                    "grasp_xyz_override_m must contain three finite values"
                )
        if vertical_grasp_descent:
            if not return_to_pick:
                raise ValueError(
                    "vertical_grasp_descent is only valid with "
                    "return_to_pick=True"
                )
            if task_b_anchor_pose is None:
                raise ValueError(
                    "vertical_grasp_descent needs a task_b_anchor_pose"
                )
        if use_visual_grasp_z and not return_to_pick:
            raise ValueError(
                "use_visual_grasp_z is only valid with return_to_pick=True"
            )

        base_xyz = (
            np.asarray(grasp_xyz_override_m, dtype=float)
            if grasp_xyz_override_m is not None
            else self.detection_to_base(detection, robot_state)
        )
        rpy = self._orientation(robot_state)
        grasp_xyz = base_xyz.copy()
        if grasp_xy_from_observation:
            assert task_b_anchor_pose is not None
            grasp_xyz[0] = float(task_b_anchor_pose.xyz[0])
            grasp_xyz[1] = float(task_b_anchor_pose.xyz[1])
        elif grasp_xy_m is not None:
            grasp_xyz[0] = float(grasp_xy_m[0])
            grasp_xyz[1] = float(grasp_xy_m[1])
        effective_grasp_rpy = (
            grasp_rpy
            if grasp_rpy is not None
            else self.config.motion.grasp_rpy
        )
        if return_to_pick and effective_grasp_rpy is not None:
            rpy = tuple(float(v) for v in effective_grasp_rpy)
        if grasp_xy_offset_m is not None:
            flange_xy_offset = np.asarray(grasp_xy_offset_m, dtype=float)
            tool_to_base = euler_to_matrix(rpy)
            base_xy_delta = tool_to_base @ np.array(
                [flange_xy_offset[0], flange_xy_offset[1], 0.0],
                dtype=float,
            )
            # The correction was requested in flange XY.  Preserve the
            # operator-verified grasp height by applying only the rotated base
            # X/Y projection rather than allowing tool tilt to shift base Z.
            grasp_xyz[0] += base_xy_delta[0]
            grasp_xyz[1] += base_xy_delta[1]
            self._log(
                "[plan] flange_xy_offset="
                f"({flange_xy_offset[0]:.6f}, {flange_xy_offset[1]:.6f}) m; "
                "base_xy_delta="
                f"({base_xy_delta[0]:.6f}, {base_xy_delta[1]:.6f}) m; "
                f"base_z_delta_ignored={base_xy_delta[2]:.6f} m"
            )
        effective_grasp_z_m = (
            grasp_z_m
            if grasp_z_m is not None
            else self.config.motion.grasp_z_m
        )
        if use_visual_grasp_z:
            # Adaptive inverse grasping supplies a flange target that has
            # already been derived from the observed jaw-tip point with the
            # selected orientation and ``gripper_tip_offset_flange_m``.  Do not
            # replace it with either the per-point taught height or the global
            # motion default, and do not stack a second fixed Z offset.
            pass
        elif return_to_pick and effective_grasp_z_m is not None:
            # A depth anchor describes the visible object surface, not the TCP
            # pose needed to seat the jaws around it.  Task B may therefore use
            # an operator-verified grasp height while retaining visual XY (or
            # an explicitly selected observation-anchor XY for vertical
            # descent).
            grasp_xyz[2] = float(effective_grasp_z_m)
        else:
            grasp_xyz[2] += float(self.config.motion.grasp_z_offset)
        grasp_pose = Pose(tuple(float(v) for v in grasp_xyz), rpy)
        task_b_high_pose = task_b_anchor_pose
        if task_b_anchor_pose is not None and (
            vertical_grasp_descent or grasp_xy_m is not None
        ):
            # Establish the high pose directly over the final grasp column.
            # For vertical_grasp_descent, use a fixed operator-configured height
            # above the verified grasp Z instead of inheriting the observation
            # height.  Never add a preparatory upward move before the straight
            # descent: after visual confirmation the next motion is directly
            # downward along Z with fixed XY and RPY.
            # The explicit-XY mode retains its historic high-pose calculation.
            if vertical_grasp_descent:
                effective_approach_height_m = (
                    self.config.task_b.vertical_grasp_approach_height_m
                    if vertical_grasp_approach_height_m is None
                    else vertical_grasp_approach_height_m
                )
                if (
                    not np.isfinite(float(effective_approach_height_m))
                    or float(effective_approach_height_m) <= 0.0
                ):
                    raise ValueError(
                        "vertical_grasp_approach_height_m must be positive "
                        "and finite"
                    )
                high_z = float(grasp_xyz[2]) + float(
                    effective_approach_height_m
                )
            else:
                high_z = max(
                    float(task_b_anchor_pose.xyz[2]),
                    float(grasp_xyz[2])
                    + float(self.config.motion.approach_height),
                    float(grasp_xyz[2])
                    + float(self.config.motion.lift_height),
                )
            task_b_high_pose = Pose(
                (float(grasp_xyz[0]), float(grasp_xyz[1]), high_z),
                rpy,
            )

        if return_to_pick and task_b_high_pose is not None:
            pregrasp_pose = task_b_high_pose
            lift_pose = task_b_high_pose
        else:
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
            # grasped.  With a taught observation anchor, lift/transfer/retreat
            # all return to that exact high pose; otherwise retain the historic
            # vertical in-place path for callers without a captured anchor.
            release_xyz = tuple(float(v) for v in grasp_xyz)
            release_pose = Pose(release_xyz, rpy)
            if task_b_high_pose is not None:
                transfer_pose = task_b_high_pose
                retreat_pose = task_b_high_pose
            else:
                transfer_pose = Pose(
                    (
                        release_xyz[0],
                        release_xyz[1],
                        release_xyz[2] + self.config.motion.place_approach_height,
                    ),
                    rpy,
                )
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
