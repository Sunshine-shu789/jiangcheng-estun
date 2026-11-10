"""The hardware-independent Track 1 pick state machine.

Tasks B/C/D/E all use the same observable sequence::

    scan -> detect -> select -> plan -> open_gripper
         -> approach -> grasp -> close -> verify_hold
         -> lift -> transfer -> release -> retreat -> done

Task B additionally walks an ordered list of five expected pick points.  A
cycle is complete only after the object is released back at the same observed
point and the arm has retreated.  The next cycle is gated on a fresh,
consecutive visual confirmation of the target at the next expected point;
continued visibility of the already-returned object at the old point is not
enough to advance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from .box_pose import (
    BoxPoseError,
    build_box_point_layout,
    detection_center_base_xyz,
    fit_box_pose_from_frame,
    match_item_to_box_point,
)
from .collision import check_box_clearance
from .config import Track1Config, load_joint_pose_file
from .detection_stabilizer import box_iou
from .domain import (
    CaptureRobotState,
    Detection,
    Frame,
    GripperState,
    PickResult,
    PickStatus,
    PipelineEvent,
    Pose,
    RobotState,
    Selection,
    RunReport,
    TaskMode,
    TaskPhase,
)
from .kinematics import (
    controller_fk,
    joint_scan_target,
    joints_within_limits,
    matrix_to_rpy,
    rpy_to_matrix,
    rotation_angle_error,
)
from .policy import TargetPolicy


_PHASE_FOR_WAYPOINT = {
    "pregrasp": TaskPhase.APPROACH,
    "grasp": TaskPhase.GRASP,
    "lift": TaskPhase.LIFT,
    "transfer": TaskPhase.TRANSFER,
    "release": TaskPhase.RELEASE,
    "retreat": TaskPhase.RETREAT,
}


@dataclass(frozen=True)
class _TaskCDObservedCandidate:
    """One independently tracked C/D candidate and its consecutive hits."""

    detection: Detection
    consecutive_hits: int


class PickPipeline:
    """One object at a time: scan -> select -> open -> move -> close -> lift.

    The jaws are opened before entering the narrow approach corridor and are
    closed only after the tool reaches the planned grasp pose.  A failed hold
    check triggers an open-and-retreat recovery instead of lifting blindly.
    """

    def __init__(
        self,
        config: Track1Config,
        *,
        vision,
        policy: TargetPolicy,
        planner,
        motion,
        gripper,
        hold_verifier,
        grasp_consensus_gate=None,
        operator_gate: Callable[[str, int, int], None] | None = None,
        log=None,
    ) -> None:
        self.config = config
        self.vision = vision
        self.policy = policy
        self.planner = planner
        self.motion = motion
        self.gripper = gripper
        self.hold_verifier = hold_verifier
        self.grasp_consensus_gate = grasp_consensus_gate
        self._operator_gate = operator_gate
        self._log = log or (lambda message: None)
        self._closed = False
        self._task_b_pick_index = 0
        #: Actual base-frame XY locations where completed objects were released.
        #: The configured reference points are only expected windows; the
        #: observed release positions are what must be excluded in later rounds.
        self._task_b_completed_xy: list[tuple[float, float]] = []
        #: Stable detector track ids that have completed a C/D pick, transfer,
        #: release and final retreat.  A track is never retried in this run.
        self._task_cd_completed_instance_ids: set[str] = set()
        self._task_cd_completed_pick_count = 0
        self._task_b_target_label: str | None = (
            config.task_b.target_class.strip() or None
        )
        self._arbitrary_placement = bool(
            config.adaptive_inverse.enabled
            and config.adaptive_inverse.arbitrary_placement
        )
        self._box_relative_five_point = bool(
            config.adaptive_inverse.enabled
            and config.adaptive_inverse.box_relative_five_point.enabled
        )
        self._tilted_visual_grasp = bool(
            config.adaptive_inverse.enabled
            and config.adaptive_inverse.tilted_visual_grasp
        )
        if self._arbitrary_placement and self._box_relative_five_point:
            raise ValueError(
                "arbitrary-placement and box-relative five-point modes are "
                "mutually exclusive"
            )

    def run(self, max_picks: int | None = None) -> RunReport:
        if (
            self.config.task.mode in {TaskMode.C, TaskMode.D}
            and self.config.task_cd.enabled
        ):
            # Fixed-observation clearing has no pick-count target.  ``max_picks``
            # is deliberately ignored so an environment default can never stop
            # the run while graspable objects remain.
            return self._run_task_cd_clear()

        default_picks = (
            self.config.task_b.rounds
            if self.config.task.mode is TaskMode.B
            else self.config.task.max_picks
        )
        requested = int(max_picks or default_picks)
        if requested < 1:
            raise ValueError("max_picks must be >= 1")
        if self.config.task.mode is TaskMode.B:
            if self._arbitrary_placement or self._box_relative_five_point:
                # These dynamic Task B modes have no reference-point list.  The
                # round count is the only persisted state; every target pose is
                # acquired again from the current camera frame.  In particular,
                # box-relative mode must never fall back to place_poses.
                requested = min(requested, self.config.task_b.rounds)
            else:
                requested = min(
                    requested,
                    self.config.task_b.rounds,
                    len(self.config.place_poses),
                )
            self._ensure_task_b_initial_pose()

        results: list[PickResult] = []
        successes = 0
        attempts = 0
        while successes < requested and not self.policy.complete:
            try:
                result = self.run_once()
            except StopIteration:
                result = PickResult(
                    PickStatus.NO_TARGET,
                    None,
                    "vision source exhausted",
                    (PipelineEvent(TaskPhase.SCAN, "no more frames"),),
                )
            results.append(result)
            attempts += 1
            if result.success:
                successes += 1
                if (
                    self.config.task.mode is TaskMode.B
                    and self.config.task_b.return_to_initial_after_each_pick
                ):
                    # The supervised sequence is: finish the in-place release and
                    # retreat, return all six axes to the shared starting pose,
                    # then ask the operator to place the next banana.  Keeping
                    # this before the operator gate guarantees that no next-point
                    # motion can start while the arm is still near the box.
                    self._return_to_task_b_initial_pose(
                        point_number=self._task_b_pick_index,
                        completed_name=result.place_name,
                    )
                elif (
                    self.config.task.mode is TaskMode.B
                    and self.config.task_b.return_to_observation_after_each_pick
                ):
                    # The in-place release and final retreat have completed.
                    # Move directly to this point's taught observation posture
                    # so the next acquisition starts from the same well-
                    # conditioned, repeatable view without a detour through the
                    # shared startup pose.  The helper verifies six-axis
                    # feedback on live runs and raises on any failure.
                    self._return_to_task_b_observation_pose(
                        point_number=self._task_b_pick_index,
                        completed_name=result.place_name,
                    )
                if (
                    self.config.task.mode is TaskMode.B
                    and self.config.task_b.operator_pause_after_each_pick
                    and self._task_b_pick_index < requested
                ):
                    if self._operator_gate is None:
                        raise RuntimeError(
                            "operator pause is enabled but no operator gate is configured"
                        )
                    if self.config.task_b.return_to_observation_after_each_pick:
                        return_description = "observation pose"
                    elif self.config.task_b.return_to_initial_after_each_pick:
                        return_description = "initial pose"
                    else:
                        return_description = "retreated pick pose"
                    self._log(
                        f"[operator_pause] {result.place_name} completed and the arm "
                        f"is at the {return_description}; no motion will be issued "
                        "until the operator continues"
                    )
                    self._set_vision_status(
                        f"PAUSED: {result.place_name} complete at "
                        f"{return_description}; "
                        "waiting for operator"
                    )
                    self._operator_gate(
                        result.place_name,
                        self._task_b_pick_index,
                        requested,
                    )
                continue
            # Task B is a supervised five-point sequence.  Any rejected,
            # aborted or missed point stops the task so a later point can never
            # be attempted after an incomplete pick-and-return.
            if result.status == PickStatus.NO_TARGET or self.config.task.mode is TaskMode.B:
                break
            if self.config.task.stop_on_error:
                break
        if (
            self.config.task.mode is TaskMode.B
            and self.config.task_b.return_to_initial_after_task
            and not self.config.task_b.return_to_initial_after_each_pick
            and requested == self.config.task_b.rounds
            and successes == requested
            and self._task_b_pick_index >= self.config.task_b.rounds
        ):
            self._return_to_task_b_initial_pose()
        report = RunReport(requested, attempts, successes, tuple(results))
        self._log(
            f"[run] requested={requested} attempts={attempts} "
            f"successes={successes} rate={report.success_rate:.2%}"
        )
        return report

    def _ensure_task_b_initial_pose(self) -> None:
        """Move once to the captured Task B startup posture, then verify it.

        This gate runs before the first visual acquisition.  A malformed pose
        file, incomplete controller feedback, joint-limit violation, rejected
        motion or final position error aborts the whole Task B run rather than
        allowing a later grasp to start from an unexpected posture.
        """

        self._move_to_task_b_initial_pose(
            move_name="task_b_initial_pose",
            log_tag="task_b_initial_pose",
            status_text="MOVING: Task B initial pose",
            success_suffix="starting visual search",
        )

    def _return_to_task_b_initial_pose(
        self,
        *,
        point_number: int | None = None,
        completed_name: str = "",
    ) -> None:
        """Return all six axes to the configured startup pose.

        Per-point callers pass the completed point number and name.  The
        argument-free form remains available for the legacy final-only return
        mode.
        """

        if point_number is None:
            move_name = "task_b_return_initial"
            log_tag = move_name
            status_text = "MOVING: Task B returning to initial pose"
            success_suffix = "five-point task complete at the initial pose"
        else:
            point_label = completed_name or f"pick_{point_number}"
            move_name = f"task_b_return_initial_pick_{point_number}"
            log_tag = move_name
            status_text = (
                f"MOVING: Task B returning to initial pose after {point_label}"
            )
            success_suffix = f"{point_label} complete at the initial pose"

        self._move_to_task_b_initial_pose(
            move_name=move_name,
            log_tag=log_tag,
            status_text=status_text,
            success_suffix=success_suffix,
        )

    def _ensure_task_b_observation_pose(
        self,
        point_index: int,
        place_name: str,
        state: RobotState,
        *,
        move_name: str | None = None,
    ) -> RobotState:
        """Move to the operator-captured observation posture for one point.

        The five reference points were taught at table-facing observation
        postures.  Starting a long Cartesian approach from the high startup
        posture can pass through an ill-conditioned configuration even when
        both endpoints are safe.  Moving all six axes to the taught posture
        first keeps the subsequent visual search and Cartesian approach local
        and repeatable.  An empty entry keeps the historic behaviour.
        """

        observations = self.config.task_b.point_observation_joints_deg
        if not observations:
            return state
        if point_index < 0 or point_index >= len(observations):
            raise RuntimeError(
                f"no observation posture entry exists for point index {point_index}"
            )
        target_deg = observations[point_index]
        if target_deg is None:
            self._log(
                f"[task_b_observation_pose] {place_name}: no posture configured; "
                "keeping the current arm posture"
            )
            return state

        target_q = np.radians(np.asarray(target_deg, dtype=float))
        if not joints_within_limits(
            target_q,
            np.radians(float(self.config.motion.joint_limit_margin_deg)),
        ):
            raise RuntimeError(
                f"observation posture for {place_name!r} violates joint limits"
            )

        if not state.ready or state.joints is None:
            state = self._fresh_state(self.motion.get_state())
        if not state.ready or state.joints is None:
            raise RuntimeError(
                f"observation posture for {place_name!r} needs AutoReady with "
                f"six-joint feedback: {state.message}"
            )

        current_q = np.asarray(state.joints, dtype=float)
        deltas_deg = np.degrees(np.abs(target_q - current_q))
        max_delta_deg = float(np.max(deltas_deg))
        tolerance_deg = float(self.config.task_b.observation_pose_tolerance_deg)
        self._log(
            f"[task_b_observation_pose] {place_name}: current_deg="
            f"{np.round(np.degrees(current_q), 3).tolist()} target_deg="
            f"{np.round(target_deg, 3).tolist()} "
            f"max_delta_deg={max_delta_deg:.3f} tolerance_deg={tolerance_deg:.3f}"
        )
        if max_delta_deg <= tolerance_deg:
            self._log(
                f"[task_b_observation_pose] {place_name}: already inside "
                "tolerance; no joint motion command sent"
            )
            return self._fresh_state(state)

        if not hasattr(self.motion, "move_joints"):
            raise RuntimeError(
                "configured motion backend cannot execute a Task B observation pose"
            )
        max_step_deg = float(self.config.motion.joint_max_step_deg)
        estimated_steps = max(1, int(np.ceil(max_delta_deg / max_step_deg)))
        self._log(
            f"[task_b_observation_pose] {place_name}: moving six axes together: "
            f"dry_run={self.motion.dry_run} estimated_steps={estimated_steps} "
            f"max_step_deg={max_step_deg:.3f} total_abs_deg="
            f"{np.round(deltas_deg, 3).tolist()}"
        )
        self._set_vision_status(f"MOVING: observation pose {place_name}")
        try:
            self.motion.move_joints(
                target_q,
                name=move_name or f"task_b_observation_{place_name}",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Task B observation pose motion failed for {place_name!r}: {exc}"
            ) from exc

        if self.motion.dry_run:
            self._log(
                f"[task_b_observation_pose] {place_name}: dry-run completed; "
                "no physical motion or final-position verification was performed"
            )
            return self._fresh_state(state)

        final_state = self._fresh_state(state)
        if not final_state.ready or final_state.joints is None:
            raise RuntimeError(
                f"observation posture verification for {place_name!r} lost "
                f"robot feedback: {final_state.message}"
            )
        final_q = np.asarray(final_state.joints, dtype=float)
        residual_deg = float(np.max(np.degrees(np.abs(final_q - target_q))))
        if residual_deg > tolerance_deg:
            raise RuntimeError(
                f"observation posture verification failed for {place_name!r}: "
                f"max residual={residual_deg:.3f} deg exceeds "
                f"{tolerance_deg:.3f} deg"
            )
        self._log(
            f"[task_b_observation_pose] {place_name}: verified "
            f"residual_deg={residual_deg:.3f}; visual search may start"
        )
        return final_state

    def _return_to_task_b_observation_pose(
        self,
        *,
        point_number: int,
        completed_name: str = "",
    ) -> None:
        """Return to the just-completed point's observation posture.

        ``point_number`` is the one-based completed-point count.  The pick
        index has already advanced before this helper is called.  Any motion
        or verification failure propagates to ``run`` and prevents the next
        acquisition.
        """

        if point_number < 1:
            raise RuntimeError(
                "Task B observation return needs a completed point number >= 1"
            )
        point_index = point_number - 1
        point_label = completed_name or f"pick_{point_number}"
        move_name = f"task_b_return_observation_pick_{point_number}"
        self._log(
            f"[{move_name}] {point_label}: returning to observation pose after "
            "successful release and retreat"
        )
        try:
            state = self.motion.get_state()
            self._ensure_task_b_observation_pose(
                point_index,
                point_label,
                state,
                move_name=move_name,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Task B return to observation pose failed after {point_label!r}: "
                f"{exc}"
            ) from exc
        self._log(
            f"[{move_name}] {point_label}: observation pose ready for the next "
            "visual acquisition"
        )

    def _task_b_observation_anchor(
        self,
        point_index: int,
        place_name: str,
    ) -> Pose | None:
        """Return the taught observation pose used as a point's high anchor.

        The observation joints are converted with the same controller FK used
        by the Cartesian pre-check.  This keeps the anchor deterministic even
        on the mock backend, whose joint-only moves intentionally do not update
        its stored TCP pose.  Point grasp and release poses normally remain
        driven by the live visual XY and the per-point grasp Z/RPY overrides.
        A point may explicitly opt into retaining this anchor's XY for a
        straight vertical descent.
        """

        observations = self.config.task_b.point_observation_joints_deg
        if not observations:
            return None
        if point_index < 0 or point_index >= len(observations):
            raise RuntimeError(
                f"no observation posture entry exists for point index {point_index}"
            )
        target_deg = observations[point_index]
        if target_deg is None:
            self._log(
                f"[task_b_anchor_pose] {place_name}: no observation posture "
                "configured; using the historic approach geometry"
            )
            return None

        target_q = np.radians(np.asarray(target_deg, dtype=float))
        if not joints_within_limits(
            target_q,
            np.radians(float(self.config.motion.joint_limit_margin_deg)),
        ):
            raise RuntimeError(
                f"observation anchor for {place_name!r} violates joint limits"
            )
        transform = controller_fk(target_q)
        anchor = Pose(
            tuple(float(value) for value in transform[:3, 3]),
            matrix_to_rpy(transform[:3, :3]),
        )
        self._log(
            f"[task_b_anchor_pose] {place_name}: fixed high anchor "
            f"xyz={tuple(round(value, 6) for value in anchor.xyz)} "
            f"rpy={tuple(round(value, 6) for value in anchor.rpy)}"
        )
        return anchor

    def _task_b_route_rejection_message(
        self,
        place_name: str,
        plan,
        waypoints: Sequence[tuple[str, Pose]],
        exc: Exception,
    ) -> str:
        """Add point and visual-location context to a route rejection.

        The low-level Cartesian planner reports a failed target index, which
        is useful in logs but not enough for an operator to recover safely.
        Task B failures need the expected point, the live visual XY and the
        named waypoint that was rejected.  More importantly, a singular or
        discontinuous route must remain a pre-motion rejection: alternate
        bounded wrist postures or observation geometry are the corrective
        action, not moving the object or lowering the singularity threshold.
        """

        detail = str(exc)
        if self.config.task.mode is not TaskMode.B:
            return f"full-route precheck rejected before motion: {detail}"

        visual_x, visual_y = (float(value) for value in plan.grasp_pose.xyz[:2])
        failed_waypoint = next(
            (
                name
                for index, (name, _pose) in enumerate(waypoints, start=1)
                if f"target {index}" in detail
            ),
            None,
        )
        location = (
            f"Task B point {place_name!r} visual_xy="
            f"({visual_x:.6f}, {visual_y:.6f})"
        )
        if failed_waypoint is not None:
            location += f" failed_waypoint={failed_waypoint!r}"
        message = (
            f"{location}: full-route precheck rejected before motion: {detail}"
        )
        if "near-singular" in detail.lower():
            message += (
                "; this approach branch is poorly conditioned. Try another "
                "bounded wrist orientation or re-teach this point's observation "
                "posture. Do not move the object and do not lower the configured "
                "singularity threshold. No arm or gripper command was issued."
            )
        else:
            message += "; no arm or gripper command was issued."
        return message

    def _precheck_cartesian_search_route(
        self,
        poses: Sequence[Pose],
        emit,
        *,
        expected_point: tuple[float, ...] | None,
        expected_name: str,
    ) -> tuple[Pose, ...]:
        """Validate every configured fallback view before search motion.

        A safe initial observation does not prove that the later offset views
        are safe.  Checking the complete active-search route from the current
        posture before issuing the first search command prevents a late
        near-singular or branch-jump failure after the arm has already moved.
        This applies equally to points 2-5 for both the five-point and resume
        configurations.
        """

        if not poses:
            return ()

        workspace_safe: list[tuple[int, Pose]] = []
        for index, pose in enumerate(poses, start=1):
            if self._search_pose_within_workspace(pose):
                workspace_safe.append((index, pose))
                continue
            emit(
                TaskPhase.SCAN,
                f"active search view {index}/{len(poses)} skipped: "
                f"outside workspace xyz={tuple(round(v, 3) for v in pose.xyz)}",
            )
        if not workspace_safe:
            return ()

        try:
            route = self.motion.precheck_linear_targets(
                tuple(pose for _index, pose in workspace_safe),
                step_m=float(self.config.motion.cartesian_step_m),
            )
        except (ValueError, RuntimeError) as exc:
            reference_xy = ""
            if expected_point is not None and len(expected_point) >= 2:
                reference_xy = (
                    f" reference_xy=({float(expected_point[0]):.6f}, "
                    f"{float(expected_point[1]):.6f})"
                )
            failed_view = ""
            detail = str(exc)
            for route_index, (view_index, pose) in enumerate(
                workspace_safe, start=1
            ):
                if f"target {route_index}" not in detail:
                    continue
                failed_view = (
                    f" failed_search_view={view_index}/{len(poses)} "
                    f"xyz={tuple(round(value, 6) for value in pose.xyz)}"
                )
                break
            message = (
                f"active search route precheck rejected for Task B point "
                f"{expected_name!r}{reference_xy}{failed_view} before any "
                "search movement: "
                f"{exc}. Inspect or re-teach the observation posture and "
                "search-view offsets; do not move the object and do not lower "
                "the configured singularity threshold. No arm or gripper "
                "command was issued."
            )
            emit(TaskPhase.SCAN, message)
            raise RuntimeError(message) from exc

        if len(route.segments) != len(workspace_safe):
            message = (
                f"active search route precheck returned a different segment "
                f"count for Task B point {expected_name!r}: "
                f"{len(route.segments)} != {len(workspace_safe)}"
            )
            emit(TaskPhase.SCAN, message)
            raise RuntimeError(message)
        emit(
            TaskPhase.SCAN,
            f"active search route precheck passed for Task B point "
            f"{expected_name!r}: {len(route.segments)} segment(s)",
        )
        return tuple(pose for _index, pose in workspace_safe)

    def _relative_observation_poses(
        self,
        state: RobotState,
    ) -> tuple[Pose, ...]:
        """Build observation adjustments relative to the current TCP.

        Arbitrary-placement acquisition has no remembered object XY from which
        a search grid could be derived.  These candidates therefore use only
        the live TCP as their anchor and preserve its flange orientation.  A
        positive base-frame Z offset lifts the camera away from the table and
        gripper without inventing or reusing an object location.
        """

        search = self.config.search
        offsets = search.arbitrary_observation_offsets_xyz_m
        if not search.enabled or not offsets:
            return ()
        if state.pose is None:
            raise RuntimeError(
                "relative observation adjustment requires current TCP feedback"
            )
        xyz = np.asarray(state.pose.xyz, dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise RuntimeError(
                "relative observation adjustment received invalid TCP xyz feedback"
            )
        rpy = tuple(float(value) for value in state.pose.rpy)
        return tuple(
            Pose(
                (
                    float(xyz[0]) + float(dx),
                    float(xyz[1]) + float(dy),
                    float(xyz[2]) + float(dz),
                ),
                rpy,
            )
            for dx, dy, dz in offsets
        )

    def _precheck_relative_observation_route(
        self,
        state: RobotState,
        poses: Sequence[Pose],
        emit,
        *,
        expected_name: str,
    ):
        """Pre-check adaptive observation moves and their exact return path.

        The requested sequence ends at a copy of the current TCP pose, so the
        planner validates the outbound view changes and the reverse path to the
        original observation posture as one all-or-nothing route.  No command
        is issued when any waypoint fails workspace, IK, singularity, branch or
        joint-step checks.
        """

        if not poses:
            return None
        if state.pose is None:
            emit(
                TaskPhase.SCAN,
                "observation adjustment precheck rejected: current TCP feedback "
                "is unavailable. No arm or gripper command was issued.",
            )
            return None

        anchor = Pose(tuple(float(v) for v in state.pose.xyz), tuple(state.pose.rpy))
        targets = tuple(poses) + (anchor,)
        for index, pose in enumerate(poses, start=1):
            if self._search_pose_within_workspace(pose):
                continue
            message = (
                f"observation adjustment {index}/{len(poses)} rejected before "
                f"any motion: xyz={tuple(round(v, 3) for v in pose.xyz)} is "
                "outside the configured workspace. No arm or gripper command "
                "was issued."
            )
            emit(TaskPhase.SCAN, message)
            return None

        try:
            route = self.motion.precheck_linear_targets(
                targets,
                step_m=float(self.config.motion.cartesian_step_m),
            )
        except (ValueError, RuntimeError) as exc:
            message = (
                "observation adjustment route precheck rejected before any "
                f"motion for {expected_name!r}: {exc}. No arm or gripper "
                "command was issued."
            )
            emit(TaskPhase.SCAN, message)
            return None

        if len(route.segments) != len(targets):
            message = (
                "observation adjustment route precheck returned a different "
                f"segment count for {expected_name!r}: {len(route.segments)} != "
                f"{len(targets)}. No arm or gripper command was issued."
            )
            emit(TaskPhase.SCAN, message)
            return None

        minimum_sigma = min(
            float(segment.min_jacobian_singular_value)
            for segment in route.segments
        )
        emit(
            TaskPhase.SCAN,
            f"observation adjustment route precheck passed for {expected_name!r}: "
            f"{len(poses)} view(s) plus exact return, min_sigma={minimum_sigma:.5f}",
        )
        return route

    def _search_relative_observation_views(
        self,
        initial_state: RobotState,
        poses: Sequence[Pose],
        route,
        emit,
        *,
        expected_point: tuple[float, ...] | None,
        expected_label: str | None,
        expected_name: str,
        completed_points: Sequence[tuple[float, ...]] = (),
    ) -> tuple[Detection | None, Frame | None, RobotState, str]:
        """Visit pre-checked relative views and always return after an all-miss."""

        if len(route.segments) != len(poses) + 1:
            raise RuntimeError(
                "observation adjustment route does not contain its return segment"
            )
        if not hasattr(self.motion, "execute_prechecked_segment"):
            raise RuntimeError(
                "configured motion backend cannot execute a pre-checked "
                "observation adjustment route"
            )

        state = initial_state
        last_frame: Frame | None = None
        last_reason = "target has not been observed"
        for index, (pose, segment) in enumerate(
            zip(poses, route.segments[:-1]),
            start=1,
        ):
            if not self.motion.dry_run and not state.ready:
                raise RuntimeError(
                    "robot not ready before observation adjustment "
                    f"{index}/{len(poses)}: {state.message}"
                )
            emit(
                TaskPhase.SCAN,
                f"observation adjustment {index}/{len(poses)}: "
                f"moving to relative TCP view xyz={tuple(round(v, 3) for v in pose.xyz)}",
            )
            self._set_vision_status(
                f"MOVING: observation adjustment {index}/{len(poses)}"
            )
            try:
                self.motion.execute_prechecked_segment(
                    segment,
                    name=f"task_observation_adjust_{index:02d}",
                    speed_mps=self.config.motion.speed_mps,
                )
            except Exception as exc:
                message = (
                    f"observation adjustment {index}/{len(poses)} move failed; "
                    f"stopping without a pick: {exc}"
                )
                emit(TaskPhase.SCAN, message)
                raise RuntimeError(message) from exc
            if self.config.search.settle_s > 0.0:
                time.sleep(self.config.search.settle_s)

            state = self._fresh_state(state)
            if not self.motion.dry_run and not state.ready:
                raise RuntimeError(
                    "robot not ready at observation adjustment "
                    f"{index}/{len(poses)}: {state.message}"
                )
            target, frame, reason = self._observe_target(
                emit,
                expected_point=expected_point,
                expected_label=expected_label,
                expected_name=expected_name,
                completed_points=completed_points,
                fallback_state=state,
            )
            last_frame = frame
            last_reason = reason
            if target is not None:
                return target, frame, self._fresh_state(state), reason
            emit(
                TaskPhase.SCAN,
                f"observation adjustment {index}/{len(poses)}: "
                f"no confirmed target: {reason}",
            )

        self._set_vision_status("MOVING: returning to initial observation pose")
        try:
            self.motion.execute_prechecked_segment(
                route.segments[-1],
                name="task_observation_return",
                speed_mps=self.config.motion.speed_mps,
            )
        except Exception as exc:
            message = (
                "observation adjustment return move failed; stopping without a "
                f"pick: {exc}"
            )
            emit(TaskPhase.SCAN, message)
            raise RuntimeError(message) from exc
        if self.config.search.settle_s > 0.0:
            time.sleep(self.config.search.settle_s)
        state = self._fresh_state(state)
        if not self.motion.dry_run and not state.ready:
            raise RuntimeError(
                "robot not ready after returning from observation adjustment: "
                f"{state.message}"
            )
        return (
            None,
            last_frame,
            state,
            f"target not confirmed in {len(poses)} relative observation "
            f"posture(s); returned to the initial observation pose: {last_reason}",
        )

    def _move_to_task_b_initial_pose(
        self,
        *,
        move_name: str,
        log_tag: str,
        status_text: str,
        success_suffix: str,
    ) -> None:
        """Move to the shared Task B joint posture and verify it when live."""

        pose_file = self.config.task_b.initial_pose_file.strip()
        if not pose_file:
            self._log(
                f"[{log_tag}] no initial pose file configured; "
                "keeping the current arm posture"
            )
            return

        try:
            target_q = np.asarray(
                load_joint_pose_file(
                    pose_file,
                    expected_role="task_b_initial_pose",
                ),
                dtype=float,
            )
        except (OSError, ValueError) as exc:
            message = f"Task B initial pose is invalid: {exc}"
            self._log(f"[{log_tag}] {message}")
            self._set_vision_status(f"ABORT: {message}")
            raise RuntimeError(message) from exc

        if not joints_within_limits(
            target_q,
            np.radians(float(self.config.motion.joint_limit_margin_deg)),
        ):
            message = "Task B initial pose violates the configured joint limits"
            self._log(f"[{log_tag}] {message}")
            self._set_vision_status(f"ABORT: {message}")
            raise RuntimeError(message)

        state = self.motion.get_state()
        wait_for_state = getattr(self.motion, "wait_for_state", None)
        if callable(wait_for_state):
            # The pipeline worker starts concurrently with the RobotInfo
            # subscription.  Wait for one bounded fresh sample before rejecting
            # a healthy controller merely because DDS discovery has not
            # completed yet.  A genuinely missing controller still times out.
            state = wait_for_state(
                max(1.0, float(self.config.motion.robot_info_timeout_s))
            )
        state = self._fresh_state(state)
        if not state.ready or state.joints is None:
            message = (
                "Task B initial pose check requires AutoReady, fault_flag=0, "
                f"and six-joint feedback: {state.message}"
            )
            self._log(f"[{log_tag}] {message}")
            self._set_vision_status(f"ABORT: {message}")
            raise RuntimeError(message)

        current_q = np.asarray(state.joints, dtype=float)
        deltas_deg = np.degrees(np.abs(target_q - current_q))
        max_delta_deg = float(np.max(deltas_deg))
        tolerance_deg = float(self.config.task_b.initial_pose_tolerance_deg)
        self._log(
            f"[{log_tag}] current_deg="
            f"{np.round(np.degrees(current_q), 3).tolist()} target_deg="
            f"{np.round(np.degrees(target_q), 3).tolist()} "
            f"max_delta_deg={max_delta_deg:.3f} tolerance_deg={tolerance_deg:.3f}"
        )
        if max_delta_deg <= tolerance_deg:
            self._log(
                f"[{log_tag}] already inside tolerance; "
                "no joint motion command sent"
            )
            return

        if not hasattr(self.motion, "move_joints"):
            raise RuntimeError(
                "configured motion backend cannot execute the Task B initial pose"
            )
        max_step_deg = float(self.config.motion.joint_max_step_deg)
        estimated_steps = max(
            1,
            int(np.ceil(max_delta_deg / max_step_deg)),
        )
        self._log(
            f"[{log_tag}] moving six axes together: "
            f"dry_run={self.motion.dry_run} estimated_steps={estimated_steps} "
            f"max_step_deg={max_step_deg:.3f} total_abs_deg="
            f"{np.round(deltas_deg, 3).tolist()}"
        )
        self._set_vision_status(status_text)
        try:
            self.motion.move_joints(
                target_q,
                name=move_name,
            )
        except Exception as exc:
            message = f"Task B initial pose motion failed: {exc}"
            self._log(f"[{log_tag}] {message}")
            self._set_vision_status(f"ABORT: {message}")
            raise RuntimeError(message) from exc

        if self.motion.dry_run:
            self._log(
                f"[{log_tag}] dry-run completed; no physical motion "
                "or final-position verification was performed"
            )
            return

        final_state = self._fresh_state(state)
        if not final_state.ready or final_state.joints is None:
            message = (
                "Task B initial pose verification lost robot feedback: "
                f"{final_state.message}"
            )
            self._log(f"[{log_tag}] {message}")
            self._set_vision_status(f"ABORT: {message}")
            raise RuntimeError(message)
        final_q = np.asarray(final_state.joints, dtype=float)
        residual_deg = float(np.max(np.degrees(np.abs(final_q - target_q))))
        if residual_deg > tolerance_deg:
            message = (
                "Task B initial pose verification failed: "
                f"max residual={residual_deg:.3f} deg exceeds "
                f"{tolerance_deg:.3f} deg"
            )
            self._log(f"[{log_tag}] {message}")
            self._set_vision_status(f"ABORT: {message}")
            raise RuntimeError(message)
        self._log(
            f"[{log_tag}] verified residual_deg={residual_deg:.3f}; "
            f"{success_suffix}"
        )

    def _next_place(self) -> tuple[str, tuple[float, ...] | None]:
        """Return the next expected Task B pick point and its label.

        The historical method name is retained because the rest of the
        pipeline still speaks in terms of "place" waypoints.  For Task B these
        configured poses are reference *pick* points, not five drop slots.
        """

        if self.config.task.mode is not TaskMode.B:
            return "", None
        poses = self.config.place_poses
        if not poses:
            return "", None
        if self._task_b_pick_index >= len(poses):
            raise RuntimeError(
                "task B pick sequence is complete; no reference point remains"
            )
        names = self.config.task_b.place_names
        pose = poses[self._task_b_pick_index]
        if names:
            name = names[self._task_b_pick_index % len(names)]
        else:
            name = f"pick_{self._task_b_pick_index + 1}"
        return name, pose

    def _expected_task_b_label(self) -> str | None:
        """Return the class that is legal for the current Task B round."""

        configured = self.config.task_b.target_class.strip() or None
        if self.config.task_b.keep_same_target:
            return self._task_b_target_label or configured
        return configured

    def _run_task_cd_clear(self) -> RunReport:
        """Clear the bin from one taught observation posture.

        The arm enters the observation pose once, then after every complete
        pick-transfer-release-retreat cycle returns to the same six-axis
        posture before taking the next fresh multi-frame observation.  The
        loop terminates only when the current observation has no unfinished
        graspable candidate, a read-only precheck rejects every candidate, or
        an already-started motion/actuator action aborts.
        """

        results: list[PickResult] = []
        entry_events = [PipelineEvent(TaskPhase.IDLE, "starting C/D bin clearing")]

        def entry_emit(phase: TaskPhase, message: str) -> None:
            entry_events.append(PipelineEvent(phase, message))
            self._log(f"[{phase.value}] {message}")
            self._set_vision_status(f"{phase.value.upper()}: {message}")

        try:
            state = self._move_to_task_cd_observation_pose(
                move_name="task_cd_observation_entry",
                emit=entry_emit,
            )
        except Exception as exc:
            message = f"C/D observation pose failed before clearing: {exc}"
            entry_events.append(PipelineEvent(TaskPhase.ABORT, message))
            results.append(
                PickResult(
                    PickStatus.ABORTED,
                    None,
                    message,
                    tuple(entry_events),
                )
            )
            report = RunReport(1, 1, 0, tuple(results))
            self._log(f"[run] C/D clearing aborted: {message}")
            return report

        while True:
            result = self._run_task_cd_clear_once(state)
            results.append(result)
            if not result.success:
                break

            if not self.config.task_cd.return_to_observation_after_each_pick:
                # Configuration validation rejects this setting.  Keep the
                # runtime fail-closed rather than silently skipping the return.
                message = (
                    "C/D clearing requires a return to the observation pose "
                    "after every completed pick"
                )
                results.append(
                    PickResult(
                        PickStatus.ABORTED,
                        result.target,
                        message,
                        (PipelineEvent(TaskPhase.ABORT, message),),
                    )
                )
                break

            return_events: list[PipelineEvent] = [
                PipelineEvent(
                    TaskPhase.IDLE,
                    f"track {self._task_cd_completed_pick_count} completed; "
                    "returning to the fixed observation pose",
                )
            ]

            def return_emit(phase: TaskPhase, message: str) -> None:
                return_events.append(PipelineEvent(phase, message))
                self._log(f"[{phase.value}] {message}")
                self._set_vision_status(f"{phase.value.upper()}: {message}")

            try:
                state = self._move_to_task_cd_observation_pose(
                    move_name=(
                        "task_cd_return_observation_after_track_"
                        f"{self._task_cd_completed_pick_count}"
                    ),
                    emit=return_emit,
                )
            except Exception as exc:
                message = (
                    "C/D return to observation pose failed after a completed "
                    f"pick: {exc}"
                )
                return_events.append(PipelineEvent(TaskPhase.ABORT, message))
                results.append(
                    PickResult(
                        PickStatus.ABORTED,
                        None,
                        message,
                        tuple(return_events),
                    )
                )
                break

        attempts = len(results)
        successes = sum(result.success for result in results)
        report = RunReport(
            max(1, attempts),
            attempts,
            successes,
            tuple(results),
        )
        self._log(
            f"[run] C/D clear attempts={attempts} successes={successes} "
            f"completed_tracks={len(self._task_cd_completed_instance_ids)} "
            f"final_status={results[-1].status.value}"
        )
        return report

    def _run_task_cd_clear_once(self, state: RobotState) -> PickResult:
        """Observe once, precheck candidates in order, and execute at most one."""

        events: list[PipelineEvent] = [
            PipelineEvent(TaskPhase.IDLE, "starting C/D clear observation")
        ]

        def emit(phase: TaskPhase, message: str) -> None:
            events.append(PipelineEvent(phase, message))
            self._log(f"[{phase.value}] {message}")
            self._set_vision_status(f"{phase.value.upper()}: {message}")

        try:
            state = self._fresh_state(state)
            candidates = self._observe_task_cd_candidates(emit)
        except StopIteration:
            message = "vision source exhausted"
            emit(TaskPhase.SCAN, message)
            return PickResult(
                PickStatus.NO_TARGET,
                None,
                message,
                tuple(events),
            )
        except (TimeoutError, RuntimeError) as exc:
            message = f"C/D observation failed: {exc}"
            emit(TaskPhase.ABORT, message)
            return PickResult(
                PickStatus.ABORTED,
                None,
                message,
                tuple(events),
            )

        if not candidates:
            message = "no unfinished graspable target remains in the bin"
            emit(TaskPhase.SELECT, message)
            return PickResult(
                PickStatus.NO_TARGET,
                None,
                message,
                tuple(events),
            )

        emit(
            TaskPhase.SELECT,
            f"{len(candidates)} confirmed unfinished candidate(s), ordered by "
            "graspability then detection score",
        )

        for index, candidate in enumerate(candidates, start=1):
            target = candidate.detection
            selection = self.policy.select([target])
            if not selection.accepted or selection.detection is None:
                emit(
                    TaskPhase.PLAN,
                    f"candidate {index}/{len(candidates)} "
                    f"{target.label!r}#{target.instance_id} is not admissible "
                    f"under the current target policy: {selection.reason}",
                )
                continue

            emit(
                TaskPhase.PLAN,
                f"candidate {index}/{len(candidates)}: "
                f"{target.label!r} track={target.instance_id!r} "
                f"graspability={target.graspability:.2f} "
                f"score={target.score():.2f}",
            )
            prechecked = self._precheck_task_cd_candidate(
                target,
                state,
                emit,
            )
            if prechecked is None:
                continue

            plan, route, waypoints = prechecked
            return self._execute_task_cd_plan(
                target,
                selection,
                plan,
                route,
                waypoints,
                events,
                emit,
            )

        message = (
            f"all {len(candidates)} confirmed C/D candidate(s) were rejected "
            "during read-only planning/precheck; no Cartesian motion or gripper "
            "command was issued"
        )
        emit(TaskPhase.PLAN, message)
        self.policy.commit_failure()
        return PickResult(
            PickStatus.REJECTED,
            candidates[0].detection,
            message,
            tuple(events),
        )

    def _move_to_task_cd_observation_pose(
        self,
        *,
        move_name: str,
        emit=None,
    ) -> RobotState:
        """Move all six axes to the fixed C/D observation posture."""

        # load_joint_pose_file() already returns ``joint_positions_rad``.
        target_q = np.asarray(
            load_joint_pose_file(
                self.config.task_cd.observation_pose_file,
                expected_role="task_cd_observation",
            ),
            dtype=float,
        )
        if target_q.shape != (6,) or not np.all(np.isfinite(target_q)):
            raise ValueError(
                "C/D observation posture must contain six finite joints"
            )
        if not joints_within_limits(
            target_q,
            np.radians(float(self.config.motion.joint_limit_margin_deg)),
        ):
            raise RuntimeError(
                "C/D observation posture violates configured joint limits"
            )

        state = self.motion.get_state()
        if not self.motion.dry_run:
            wait_for_state = getattr(self.motion, "wait_for_state", None)
            if callable(wait_for_state):
                state = wait_for_state(
                    max(1.0, float(self.config.motion.robot_info_timeout_s))
                )
            if (
                not state.ready
                or state.fault
                or state.fault_flag != 0
                or state.joints is None
                or len(state.joints) != 6
                or not np.all(np.isfinite(np.asarray(state.joints, dtype=float)))
            ):
                raise RuntimeError(
                    "C/D observation motion needs AutoReady, fault_flag=0 and "
                    f"six-axis feedback: {state.message}"
                )

        if not hasattr(self.motion, "move_joints"):
            raise RuntimeError(
                "configured motion backend cannot move to a C/D observation pose"
            )
        if emit is not None:
            emit(
                TaskPhase.SCAN,
                f"C/D moving to fixed observation pose: joints_deg="
                f"{np.round(np.degrees(target_q), 3).tolist()}",
            )
        self.motion.move_joints(target_q, name=move_name)

        if self.motion.dry_run:
            return self.motion.get_state()

        wait_for_state = getattr(self.motion, "wait_for_state", None)
        feedback = (
            wait_for_state(
                max(1.0, float(self.config.motion.robot_info_timeout_s))
            )
            if callable(wait_for_state)
            else self.motion.get_state()
        )
        if (
            not feedback.ready
            or feedback.fault
            or feedback.fault_flag != 0
            or feedback.joints is None
            or len(feedback.joints) != 6
            or not np.all(np.isfinite(np.asarray(feedback.joints, dtype=float)))
        ):
            raise RuntimeError(
                "C/D observation pose lost ready six-axis feedback after the "
                f"joint move: {feedback.message}"
            )
        final_q = np.asarray(feedback.joints, dtype=float)
        residual_deg = float(np.max(np.degrees(np.abs(final_q - target_q))))
        tolerance_deg = float(
            self.config.task_cd.observation_pose_tolerance_deg
        )
        if residual_deg > tolerance_deg:
            raise RuntimeError(
                "C/D observation pose verification failed: "
                f"max residual={residual_deg:.3f} deg exceeds "
                f"{tolerance_deg:.3f} deg"
            )
        self._log(
            f"[{move_name}] verified residual_deg={residual_deg:.3f}"
        )
        return feedback

    def _observe_task_cd_candidates(
        self,
        emit,
    ) -> list[_TaskCDObservedCandidate]:
        """Collect and confirm every unfinished track from fixed-view frames."""

        observe_frames = max(1, int(self.config.search.observe_frames))
        confirm_hits = min(
            max(1, int(self.config.search.confirm_hits)),
            observe_frames,
        )
        tracks: list[_TaskCDObservedCandidate] = []

        for index in range(observe_frames):
            emit(
                TaskPhase.DETECT,
                f"C/D observation {index + 1}/{observe_frames}: waiting for a "
                "fresh fixed-view frame",
            )
            try:
                frame = self.vision.capture()
            except StopIteration:
                emit(
                    TaskPhase.SCAN,
                    "C/D fixed-view vision source exhausted; treating the "
                    "observation as empty",
                )
                return []

            frame_candidates: list[Detection] = []
            excluded_completed = 0
            excluded_without_track = 0
            for detection in frame.detections:
                if not self._is_usable(detection):
                    continue
                if detection.attributes.get("held") is True:
                    continue
                instance_id = str(detection.instance_id or "").strip()
                if not instance_id:
                    excluded_without_track += 1
                    continue
                if instance_id in self._task_cd_completed_instance_ids:
                    excluded_completed += 1
                    continue
                selection = self.policy.select([detection])
                if not selection.accepted or selection.detection is None:
                    continue
                selected = selection.detection
                # Never count duplicate boxes for the same physical track as
                # multiple consecutive confirmations in one frame.
                if any(
                    self._same_target(existing, selected)
                    for existing in frame_candidates
                ):
                    continue
                frame_candidates.append(selected)

            next_tracks: list[_TaskCDObservedCandidate] = []
            matched_track_indexes: set[int] = set()
            for detection in frame_candidates:
                match_index = next(
                    (
                        track_index
                        for track_index, previous in enumerate(tracks)
                        if track_index not in matched_track_indexes
                        and self._same_target(previous.detection, detection)
                    ),
                    None,
                )
                if match_index is None:
                    next_tracks.append(
                        _TaskCDObservedCandidate(
                            detection=detection,
                            consecutive_hits=1,
                        )
                    )
                    continue

                matched_track_indexes.add(match_index)
                previous = tracks[match_index]
                best_detection = (
                    detection
                    if detection.score() > previous.detection.score()
                    else previous.detection
                )
                next_tracks.append(
                    _TaskCDObservedCandidate(
                        detection=best_detection,
                        consecutive_hits=previous.consecutive_hits + 1,
                    )
                )
            # A track absent from this frame breaks its consecutive-hit run.
            tracks = next_tracks

            emit(
                TaskPhase.DETECT,
                f"frame {frame.frame_id!r}: {len(frame.detections)} raw, "
                f"{len(frame_candidates)} policy-admissible unfinished track(s), "
                f"completed_tracks_excluded={excluded_completed}, "
                f"missing_track_id_excluded={excluded_without_track}",
            )
            if index + 1 < observe_frames:
                time.sleep(0.05)

        confirmed = [
            track for track in tracks if track.consecutive_hits >= confirm_hits
        ]
        confirmed.sort(
            key=lambda track: (
                -float(track.detection.graspability),
                -float(track.detection.score()),
                -float(track.detection.confidence),
                str(track.detection.instance_id),
                tuple(float(value) for value in track.detection.bbox),
            )
        )
        emit(
            TaskPhase.DETECT,
            f"C/D fixed-view confirmation complete: "
            f"{len(confirmed)}/{len(tracks)} active track(s) reached "
            f"{confirm_hits}/{observe_frames} consecutive frame(s)",
        )
        return confirmed

    def _task_cd_short_edge_grasp_rpy(
        self,
        target: Detection,
        state: RobotState,
    ) -> tuple[tuple[float, float, float] | None, dict[str, object]]:
        """Derive the C/D grasp RPY that aligns tool +Y with the major axis.

        The tool +Y axis spans the jaw opening.  Aligning it with the object's
        PCA major axis therefore closes the jaws across the short edge.  The
        method is deliberately fail-closed: any incomplete or non-finite PCA
        geometry raises before planning, opening the gripper or moving.
        """

        short_edge = self.config.task_cd.short_edge_grasp
        if short_edge is None or not short_edge.enabled:
            return None, {}

        required = {
            "object_center_camera_xyz": target.object_center_camera_xyz,
            "major_axis_camera": target.major_axis_camera,
            "minor_axis_camera": target.minor_axis_camera,
            "major_axis_length_m": target.major_axis_length_m,
            "minor_axis_length_m": target.minor_axis_length_m,
            "pose_confidence": target.pose_confidence,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                "C/D short-edge grasp requires complete PCA data; missing "
                + ", ".join(missing)
            )

        pose_confidence = float(target.pose_confidence)
        if not np.isfinite(pose_confidence) or pose_confidence < float(
            short_edge.min_pose_confidence
        ):
            raise RuntimeError(
                "C/D short-edge pose confidence "
                f"{pose_confidence:.6f} is below "
                f"{float(short_edge.min_pose_confidence):.6f}"
            )
        major_length = float(target.major_axis_length_m)
        minor_length = float(target.minor_axis_length_m)
        if (
            not np.isfinite(major_length)
            or not np.isfinite(minor_length)
            or major_length <= 0.0
            or minor_length <= 0.0
        ):
            raise RuntimeError("C/D short-edge PCA axis lengths must be positive")
        axis_ratio = major_length / minor_length
        if axis_ratio + 1e-12 < float(short_edge.pca.min_axis_ratio):
            raise RuntimeError(
                "C/D short-edge PCA axis ratio "
                f"{axis_ratio:.6f} is below "
                f"{float(short_edge.pca.min_axis_ratio):.6f}"
            )

        major_axis_camera = np.asarray(target.major_axis_camera, dtype=float)
        if (
            major_axis_camera.shape != (3,)
            or not np.all(np.isfinite(major_axis_camera))
            or float(np.linalg.norm(major_axis_camera)) <= 1e-12
        ):
            raise RuntimeError(
                "C/D short-edge major_axis_camera must be a finite non-zero "
                "3-vector"
            )
        base_camera = np.asarray(self.planner.base_camera_transform(state), dtype=float)
        if base_camera.shape != (4, 4) or not np.all(np.isfinite(base_camera)):
            raise RuntimeError(
                "C/D short-edge camera-to-base transform must be a finite 4x4"
            )
        major_axis_base = base_camera[:3, :3] @ major_axis_camera
        if not np.all(np.isfinite(major_axis_base)):
            raise RuntimeError(
                "C/D short-edge major axis became non-finite in base frame"
            )
        horizontal_norm = float(np.linalg.norm(major_axis_base[:2]))
        if not np.isfinite(horizontal_norm) or horizontal_norm <= 1e-6:
            raise RuntimeError(
                "C/D short-edge major axis has no usable horizontal projection "
                "in the robot base frame"
            )

        mx = float(major_axis_base[0])
        my = float(major_axis_base[1])
        yaw = float(np.arctan2(mx, -my))
        grasp_rpy = (float(np.pi), 0.0, yaw)
        tool_y_base = np.asarray(rpy_to_matrix(grasp_rpy)[:, 1], dtype=float)
        horizontal_axis = np.asarray(
            [mx / horizontal_norm, my / horizontal_norm, 0.0],
            dtype=float,
        )
        alignment = float(abs(np.dot(tool_y_base, horizontal_axis)))
        if not np.isfinite(alignment) or alignment < 0.999999:
            raise RuntimeError(
                "C/D short-edge tool +Y alignment check failed: "
                f"absolute dot={alignment:.9f}"
            )

        return grasp_rpy, {
            "major_axis_base_xy": (mx, my),
            "horizontal_axis_unit_xy": (
                float(horizontal_axis[0]),
                float(horizontal_axis[1]),
            ),
            "yaw": yaw,
            "grasp_rpy": grasp_rpy,
            "tool_y_base": tuple(float(value) for value in tool_y_base),
            "alignment": alignment,
            "pose_confidence": pose_confidence,
            "axis_ratio": axis_ratio,
        }

    def _task_cd_short_edge_grasp_rpy_variants(
        self,
        grasp_rpy: tuple[float, float, float] | None,
        diagnostics: dict[str, object],
    ) -> tuple[
        tuple[
            str,
            tuple[float, float, float] | None,
            dict[str, object],
        ],
        ...,
    ]:
        """Return wrist-equivalent RPY choices for a PCA short-edge heading.

        The jaw opening axis is undirected, so ``yaw + pi`` aligns the tool
        with the same long axis while rotating the wrist 180 degrees.  Trying
        that equivalent wrist configuration can avoid a near-singular IK
        branch without changing the grasp location or jaw closing direction.
        """

        if grasp_rpy is None:
            # Preserve the legacy fixed-orientation C/D path when the PCA
            # short-edge feature is disabled.  Passing ``None`` to the
            # planner retains motion.grasp_rpy instead of rejecting the plan.
            return (("legacy", None, {}),)

        primary = tuple(float(value) for value in grasp_rpy)
        variants = [("pca-primary", primary, diagnostics)]
        if not diagnostics:
            return tuple(variants)

        alternate_yaw = float(
            (primary[2] + float(np.pi) + float(np.pi)) % (2.0 * float(np.pi))
            - float(np.pi)
        )
        alternate = (float(np.pi), 0.0, alternate_yaw)
        tool_y = np.asarray(rpy_to_matrix(alternate)[:, 1], dtype=float)
        horizontal_axis = np.asarray(
            diagnostics["horizontal_axis_unit_xy"],
            dtype=float,
        )
        if (
            tool_y.shape != (3,)
            or horizontal_axis.shape != (2,)
            or not np.all(np.isfinite(tool_y))
            or not np.all(np.isfinite(horizontal_axis))
            or abs(float(tool_y[2])) > 1e-9
            or abs(float(np.dot(tool_y[:2], horizontal_axis))) < 0.999999
        ):
            raise RuntimeError(
                "C/D equivalent short-edge wrist variant lost PCA alignment"
            )

        alternate_diagnostics = dict(diagnostics)
        alternate_diagnostics.update(
            {
                "yaw": alternate_yaw,
                "grasp_rpy": alternate,
                "tool_y_base": tuple(float(value) for value in tool_y),
                "alignment": abs(
                    float(np.dot(tool_y[:2], horizontal_axis))
                ),
            }
        )
        variants.append(
            ("pca-equivalent-yaw-plus-pi", alternate, alternate_diagnostics)
        )
        return tuple(variants)

    def _precheck_task_cd_candidate(
        self,
        target: Detection,
        state: RobotState,
        emit,
    ):
        """Fully validate one candidate without moving either actuator."""

        try:
            grasp_rpy, short_edge_diagnostics = (
                self._task_cd_short_edge_grasp_rpy(target, state)
            )
            grasp_variants = self._task_cd_short_edge_grasp_rpy_variants(
                grasp_rpy,
                short_edge_diagnostics,
            )
            if short_edge_diagnostics:
                axis_xy = short_edge_diagnostics["major_axis_base_xy"]
                tool_y = short_edge_diagnostics["tool_y_base"]
                emit(
                    TaskPhase.PLAN,
                    f"candidate {target.label!r} track={target.instance_id!r} "
                    "short-edge PCA heading: "
                    f"major_axis_base_xy=({axis_xy[0]:.6f}, "
                    f"{axis_xy[1]:.6f}); "
                    f"yaw={short_edge_diagnostics['yaw']:.9f} rad; "
                    "grasp_rpy=("
                    + ", ".join(
                        f"{float(value):.9f}"
                        for value in short_edge_diagnostics["grasp_rpy"]
                    )
                    + "); tool_y_base=("
                    + ", ".join(
                        f"{float(value):.9f}"
                        for value in short_edge_diagnostics["tool_y_base"]
                    )
                    + "); "
                    "abs(tool_y dot major_axis)="
                    f"{short_edge_diagnostics['alignment']:.9f}; "
                    f"pose_confidence="
                    f"{short_edge_diagnostics['pose_confidence']:.6f}; "
                    f"axis_ratio={short_edge_diagnostics['axis_ratio']:.6f}",
                )
                short_edge = self.config.task_cd.short_edge_grasp
                if short_edge is not None and short_edge.jaw_tip_z_m is not None:
                    emit(
                        TaskPhase.PLAN,
                        f"candidate {target.label!r} "
                        f"track={target.instance_id!r} short-edge insertion: "
                        "fixed jaw-tip Z="
                        f"{float(short_edge.jaw_tip_z_m):.9f} m; "
                        "the detected top-surface Z is not used as the grasp "
                        "depth",
                    )
        except (ValueError, RuntimeError) as exc:
            emit(
                TaskPhase.PLAN,
                f"candidate {target.label!r} track={target.instance_id!r} "
                f"rejected before motion: {exc}",
            )
            return None

        if not grasp_variants:
            emit(
                TaskPhase.PLAN,
                f"candidate {target.label!r} track={target.instance_id!r} "
                "rejected before motion: no explicit grasp orientation",
            )
            return None

        last_error: ValueError | RuntimeError | None = None
        plan = None
        route = None
        waypoints = None
        selected_variant = ""
        for variant_name, candidate_rpy, candidate_diagnostics in grasp_variants:
            if variant_name == "pca-equivalent-yaw-plus-pi":
                emit(
                    TaskPhase.PLAN,
                    f"candidate {target.label!r} track={target.instance_id!r} "
                    "trying equivalent short-edge wrist yaw: "
                    f"yaw={candidate_diagnostics['yaw']:.9f} rad; "
                    "abs(tool_y dot major_axis)="
                    f"{candidate_diagnostics['alignment']:.9f}",
                )
            try:
                candidate_plan = self.planner.plan(
                    target,
                    state,
                    place_pose=self.config.motion.place_pose,
                    place_name="task_cd_place",
                    return_to_pick=False,
                    grasp_target_is_jaw_tip=(
                        self.config.motion.grasp_safety.enabled
                    ),
                    grasp_rpy=candidate_rpy,
                    jaw_tip_z_m=(
                        self.config.task_cd.short_edge_grasp.jaw_tip_z_m
                        if self.config.task_cd.short_edge_grasp is not None
                        else None
                    ),
                )
                self._validate_task_cd_grasp_safety_plan(
                    candidate_plan,
                    emit,
                    "task_cd_place",
                    expected_grasp_rpy=candidate_rpy,
                )
                candidate_waypoints = candidate_plan.ordered_waypoints()
                expected_names = (
                    "pregrasp",
                    "grasp",
                    "lift",
                    "transfer",
                    "release",
                    "retreat",
                )
                actual_names = tuple(
                    name for name, _pose in candidate_waypoints
                )
                if actual_names != expected_names:
                    raise RuntimeError(
                        "C/D plan must contain the exact pregrasp, grasp, lift, "
                        "transfer, release and retreat sequence; got "
                        f"{actual_names}"
                    )
                candidate_route = self.motion.precheck_linear_targets(
                    tuple(pose for _name, pose in candidate_waypoints),
                    step_m=self.config.motion.cartesian_step_m,
                )
                if len(candidate_route.segments) != len(candidate_waypoints):
                    raise RuntimeError(
                        "C/D full-route precheck returned a different segment "
                        f"count: {len(candidate_route.segments)} != "
                        f"{len(candidate_waypoints)}"
                    )
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                continue

            plan = candidate_plan
            route = candidate_route
            waypoints = candidate_waypoints
            selected_variant = variant_name
            break

        if plan is None or route is None or waypoints is None:
            emit(
                TaskPhase.PLAN,
                f"candidate {target.label!r} track={target.instance_id!r} "
                f"rejected before motion after "
                f"{len(grasp_variants)} grasp-orientation precheck(s): "
                f"{last_error}",
            )
            return None

        if self.grasp_consensus_gate is not None:
            try:
                consensus = self.grasp_consensus_gate.evaluate(plan.grasp_pose)
            except Exception as exc:
                consensus = None
                consensus_error = (
                    f"offline consensus gate raised unexpectedly: {exc}"
                )
            else:
                consensus_error = ""
            if (
                consensus is None
                or not consensus.accepted
                or consensus.offline_only is not True
                or consensus.motion_authorized is not False
            ):
                reason = (
                    consensus_error
                    or (consensus.reason if consensus is not None else "")
                    or "offline consensus gate did not return a safe acceptance"
                )
                emit(
                    TaskPhase.PLAN,
                    f"candidate {target.label!r} track={target.instance_id!r} "
                    f"rejected by offline consensus before motion: {reason}",
                )
                return None
            emit(
                TaskPhase.PLAN,
                f"candidate {target.label!r} track={target.instance_id!r} "
                f"offline consensus passed: cluster_id={consensus.cluster_id}; "
                "motion_authorized=false",
            )

        emit(
            TaskPhase.PLAN,
            f"candidate {target.label!r} track={target.instance_id!r} passed "
            f"the complete read-only precheck with {len(waypoints)} waypoint(s); "
            f"short_edge_variant={selected_variant}",
        )
        return plan, route, waypoints

    def _execute_task_cd_plan(
        self,
        target: Detection,
        selection: Selection,
        plan,
        route,
        waypoints,
        events: list[PipelineEvent],
        emit,
    ) -> PickResult:
        """Execute one already fully prechecked C/D plan."""

        try:
            if (
                self.config.gripper.open_before_approach
                and self.gripper.state != GripperState.OPEN
            ):
                emit(
                    TaskPhase.OPEN_GRIPPER,
                    "opening jaws after all candidate gates passed and before "
                    "the prechecked approach",
                )
                self.gripper.open()

            for (name, pose), segment in zip(waypoints, route.segments):
                if segment.target != pose:
                    raise RuntimeError(
                        f"prechecked C/D segment {name!r} target changed "
                        "before execution"
                    )
                phase = _PHASE_FOR_WAYPOINT.get(name, TaskPhase.APPROACH)
                emit(
                    phase,
                    f"move to {name}: xyz="
                    f"{tuple(round(float(value), 4) for value in pose.xyz)}",
                )
                self.motion.execute_prechecked_segment(
                    segment,
                    name=name,
                    speed_mps=self.config.motion.speed_mps,
                )

                if name == "grasp":
                    emit(TaskPhase.GRASP, "reached grasp pose; closing jaws")
                    self.gripper.close()
                    emit(
                        TaskPhase.VERIFY_HOLD,
                        "checking whether the object is held",
                    )
                    held, reason = self.hold_verifier.verify(target, plan)
                    if not held:
                        recovered, recovery_message = (
                            self._recover_task_cd_failed_grasp(plan, emit)
                        )
                        message = (
                            f"hold verification failed: {reason}; "
                            f"{recovery_message}"
                        )
                        emit(TaskPhase.ABORT, message)
                        self.policy.commit_failure()
                        return PickResult(
                            PickStatus.ABORTED,
                            target,
                            message,
                            tuple(events),
                        )
                    emit(TaskPhase.VERIFY_HOLD, reason)
                elif name == "release":
                    emit(
                        TaskPhase.RELEASE,
                        "opening jaws at the fixed verified place pose",
                    )
                    self.gripper.open()
        except Exception as exc:
            message = f"C/D prechecked execution failed: {exc}"
            emit(TaskPhase.ABORT, message)
            self.policy.commit_failure()
            return PickResult(
                PickStatus.ABORTED,
                target,
                message,
                tuple(events),
            )

        # Completion is committed only after the final prechecked retreat
        # segment returned successfully.  There is no earlier success state.
        instance_id = str(target.instance_id).strip()
        if not instance_id:
            raise RuntimeError("C/D successful target lost its stable track id")
        self._task_cd_completed_instance_ids.add(instance_id)
        self._task_cd_completed_pick_count += 1
        self.policy.commit_success(selection)
        message = (
            f"{target.label!r} track={instance_id!r} picked, transferred to the "
            "fixed place pose, released and retreated; track marked complete"
        )
        emit(TaskPhase.DONE, message)
        return PickResult(
            PickStatus.SUCCESS,
            target,
            message,
            tuple(events),
            place_name=plan.place_name,
        )

    def _recover_task_cd_failed_grasp(
        self,
        plan,
        emit,
    ) -> tuple[bool, str]:
        """Open and retreat from a failed hold using a fresh prechecked route.

        The historical held-object recovery uses ``move_to`` and is therefore
        intentionally not reused here.  If the reverse retreat cannot be
        prechecked, no Cartesian recovery command is issued.
        """

        try:
            recovery_route = self.motion.precheck_linear_targets(
                (plan.pregrasp_pose,),
                step_m=self.config.motion.cartesian_step_m,
            )
        except (ValueError, RuntimeError) as exc:
            return False, (
                "recovery retreat precheck failed; jaws were not reopened and "
                f"no recovery arm motion was issued: {exc}"
            )
        if (
            len(recovery_route.segments) != 1
            or recovery_route.segments[0].target != plan.pregrasp_pose
        ):
            return False, (
                "recovery retreat precheck returned an invalid segment; jaws "
                "were not reopened and no recovery arm motion was issued"
            )

        try:
            self.gripper.open()
            emit(
                TaskPhase.RETREAT,
                "failed hold recovery: retreating to the prechecked pregrasp "
                "pose with jaws open",
            )
            self.motion.execute_prechecked_segment(
                recovery_route.segments[0],
                name="task_cd_recovery_retreat",
                speed_mps=self.config.motion.speed_mps,
            )
        except Exception as exc:
            return False, f"prechecked failed-hold recovery did not complete: {exc}"
        return True, "jaws opened and the prechecked recovery retreat completed"

    def run_once(self) -> PickResult:
        events: list[PipelineEvent] = [
            PipelineEvent(TaskPhase.IDLE, "starting pick cycle")
        ]

        def emit(phase: TaskPhase, message: str) -> None:
            events.append(PipelineEvent(phase, message))
            self._log(f"[{phase.value}] {message}")
            self._set_vision_status(f"{phase.value.upper()}: {message}")

        state = self.motion.get_state()
        if not self.motion.dry_run:
            wait_for_state = getattr(self.motion, "wait_for_state", None)
            if callable(wait_for_state):
                # The ROS executor and pipeline worker start concurrently.  Do
                # not interpret the subscription-created "no feedback yet"
                # state as a controller fault; wait for one bounded fresh
                # RobotInfo sample before applying the readiness gate.
                state = wait_for_state(
                    max(1.0, float(self.config.motion.robot_info_timeout_s))
                )
        if not self.motion.dry_run and not state.ready:
            emit(TaskPhase.ABORT, f"robot not ready: {state.message}")
            return PickResult(PickStatus.ABORTED, None, state.message, tuple(events))

        if self._arbitrary_placement or self._box_relative_five_point:
            mode_prefix = (
                "box_relative_round"
                if self._box_relative_five_point
                else "arbitrary_round"
            )
            place_name = f"{mode_prefix}_{self._task_b_pick_index + 1}"
            place_pose = None
        else:
            place_name, place_pose = self._next_place()
        expected_label = (
            self._expected_task_b_label()
            if self.config.task.mode is TaskMode.B
            else None
        )
        completed_points = (
            tuple(self._task_b_completed_xy)
            if (
                self.config.task.mode is TaskMode.B
                and not self._arbitrary_placement
                and not self._box_relative_five_point
            )
            else ()
        )
        if (
            self.config.task.mode is TaskMode.B
            and not self._arbitrary_placement
            and not self._box_relative_five_point
        ):
            try:
                state = self._ensure_task_b_observation_pose(
                    self._task_b_pick_index,
                    place_name,
                    state,
                )
            except RuntimeError as exc:
                emit(TaskPhase.ABORT, f"observation pose failed: {exc}")
                return PickResult(
                    PickStatus.ABORTED,
                    None,
                    str(exc),
                    tuple(events),
                )

        try:
            if self._arbitrary_placement or self._box_relative_five_point:
                mode_text = (
                    "box-relative five-point"
                    if self._box_relative_five_point
                    else "arbitrary-placement"
                )
                emit(
                    TaskPhase.SCAN,
                    f"{place_name}: acquiring the current {mode_text} target "
                    "from one stable RGB-D frame; no remembered point, legacy "
                    "grasp override, or previous release XY is used",
                )
                target, frame, state, acquisition_reason = self._acquire_target(
                    state,
                    emit,
                    expected_label=expected_label,
                    expected_name=place_name,
                )
            elif self.config.task.mode is TaskMode.B:
                round_number = self._task_b_pick_index + 1
                total_rounds = self.config.task_b.rounds
                target, frame, state, acquisition_reason = (
                    self._acquire_task_b_target(
                        state,
                        place_pose,
                        place_name,
                        expected_label,
                        round_number,
                        total_rounds,
                        emit,
                        completed_points=completed_points,
                    )
                )
            else:
                emit(TaskPhase.SCAN, "acquiring and confirming the target")
                target, frame, state, acquisition_reason = self._acquire_target(
                    state, emit
                )
        except StopIteration as exc:
            emit(TaskPhase.SCAN, str(exc))
            return PickResult(PickStatus.NO_TARGET, None, str(exc), tuple(events))
        except (TimeoutError, RuntimeError) as exc:
            emit(TaskPhase.ABORT, f"target acquisition failed: {exc}")
            return PickResult(PickStatus.ABORTED, None, str(exc), tuple(events))

        if target is None:
            emit(TaskPhase.SELECT, acquisition_reason)
            return PickResult(
                PickStatus.NO_TARGET,
                None,
                acquisition_reason,
                tuple(events),
            )

        if expected_label is not None and target.label != expected_label:
            reason = (
                f"task B expected target class {expected_label!r}; "
                f"detected {target.label!r}"
            )
            emit(TaskPhase.SELECT, reason)
            return PickResult(PickStatus.REJECTED, target, reason, tuple(events))

        emit(
            TaskPhase.SELECT,
            f"selected {target.label!r} conf={target.confidence:.2f} "
            f"graspability={target.graspability:.2f}: {acquisition_reason}",
        )

        return_to_pick = (
            self.config.task.mode is TaskMode.B
            and self.config.task_b.return_to_pick
        )
        point_grasp_z_m: float | None = None
        point_grasp_xy_from_observation = False
        point_grasp_xy_m: tuple[float, float] | None = None
        point_grasp_xy_offset_m: tuple[float, float] | None = None
        point_grasp_jaw_tip_object_offset_m: tuple[float, float] | None = None
        point_grasp_rpy: tuple[float, float, float] | None = None
        point_vertical_grasp_descent = False
        point_vertical_grasp_approach_height_m: float | None = None
        task_b_anchor_pose: Pose | None = None
        try:
            if (
                self.config.task.mode is TaskMode.B
                and return_to_pick
                and not self._arbitrary_placement
                and not self._box_relative_five_point
            ):
                point_index = self._task_b_pick_index
                if point_index < len(self.config.task_b.point_grasp_z_m):
                    point_grasp_z_m = self.config.task_b.point_grasp_z_m[point_index]
                if point_index < len(
                    self.config.task_b.point_grasp_xy_from_observation
                ):
                    point_grasp_xy_from_observation = (
                        self.config.task_b.point_grasp_xy_from_observation[
                            point_index
                        ]
                    )
                if point_index < len(self.config.task_b.point_grasp_xy_m):
                    point_grasp_xy_m = self.config.task_b.point_grasp_xy_m[
                        point_index
                    ]
                if point_index < len(
                    self.config.task_b.point_grasp_xy_offset_m
                ):
                    point_grasp_xy_offset_m = (
                        self.config.task_b.point_grasp_xy_offset_m[point_index]
                    )
                if point_index < len(
                    self.config.task_b.point_grasp_jaw_tip_object_offset_m
                ):
                    point_grasp_jaw_tip_object_offset_m = (
                        self.config.task_b.point_grasp_jaw_tip_object_offset_m[
                            point_index
                        ]
                    )
                if point_index < len(self.config.task_b.point_grasp_rpy):
                    point_grasp_rpy = self.config.task_b.point_grasp_rpy[point_index]
                if point_index < len(
                    self.config.task_b.point_vertical_grasp_descent
                ):
                    point_vertical_grasp_descent = (
                        self.config.task_b.point_vertical_grasp_descent[point_index]
                    )
                if point_index < len(
                    self.config.task_b.point_vertical_grasp_approach_height_m
                ):
                    point_vertical_grasp_approach_height_m = (
                        self.config.task_b.point_vertical_grasp_approach_height_m[
                            point_index
                        ]
                    )
                task_b_anchor_pose = self._task_b_observation_anchor(
                    point_index,
                    place_name,
                )
            adaptive_enabled = bool(self.config.adaptive_inverse.enabled)
            visual_robot_state = state
            if adaptive_enabled:
                if frame is None or frame.capture_robot_state is None:
                    raise RuntimeError(
                        "adaptive grasp requires the live robot feedback state "
                        "captured with the exact RGB-D frame"
                    )
                # A dry-run observation move only advances the virtual planning
                # state.  The physical camera did not move, so its detections
                # must be interpreted with the live RobotInfo pose stored with
                # this frame, never with the virtual planning pose.
                visual_robot_state = frame.capture_robot_state.before
                if self.motion.dry_run:
                    visual_xyz = (
                        "unavailable"
                        if visual_robot_state.pose is None
                        else tuple(
                            round(float(value), 6)
                            for value in visual_robot_state.pose.xyz
                        )
                    )
                    planning_xyz = (
                        "unavailable"
                        if state.pose is None
                        else tuple(
                            round(float(value), 6)
                            for value in state.pose.xyz
                        )
                    )
                    emit(
                        TaskPhase.PLAN,
                        f"{place_name}: RGB-D coordinates use live RobotInfo "
                        f"flange_xyz={visual_xyz}; route planning uses the "
                        f"dry-run virtual flange_xyz={planning_xyz}",
                    )
                if self._box_relative_five_point:
                    if frame is None:
                        raise RuntimeError(
                            "box-relative mode requires the exact RGB-D frame "
                            "used for target acquisition"
                        )
                    capture = frame.capture_robot_state
                    if capture is None:
                        raise RuntimeError(
                            "box-relative mode requires capture_robot_state "
                            "before and after the RGB-D frame"
                        )
                    box_relative = self.config.adaptive_inverse.box_relative_five_point
                    box_clearance = self.config.adaptive_inverse.box_clearance
                    base_camera = self.planner.base_camera_transform(
                        capture.before
                    )
                    box_pose = fit_box_pose_from_frame(
                        frame,
                        base_camera_transform=base_camera,
                        expected_interior_size_m=box_clearance.interior_size_m,
                        config=box_relative,
                    )
                    box_layout = build_box_point_layout(
                        box_clearance.interior_size_m,
                        gripper_max_outer_m=box_relative.gripper_max_outer_m,
                        wall_clearance_m=box_clearance.wall_clearance_m,
                    )
                    adaptive_candidates = self.planner.plan_box_relative_candidates(
                        target,
                        frame,
                        capture.before,
                        box_pose,
                        box_layout,
                        place_name=place_name,
                        return_to_pick=True,
                    )
                    emit(
                        TaskPhase.PLAN,
                        f"{place_name}: fitted box centre="
                        f"({box_pose.center_base_xyz_m[0]:.4f}, "
                        f"{box_pose.center_base_xyz_m[1]:.4f}, "
                        f"{box_pose.center_base_xyz_m[2]:.4f}) "
                        f"yaw={np.degrees(box_pose.yaw_rad):.2f} deg; "
                        f"generated {len(adaptive_candidates)} fixed box-local "
                        "grasp point candidate(s)",
                    )
                elif self._arbitrary_placement:
                    # Deliberately pass no legacy point height, per-point RPY,
                    # observation anchor, or point-specific approach height.
                    # The object-relative planner must derive the grasp column
                    # entirely from this acknowledged visual observation.
                    adaptive_candidates = self.planner.plan_adaptive_candidates(
                        target,
                        visual_robot_state,
                        place_name=place_name,
                        return_to_pick=True,
                    )
                else:
                    adaptive_candidates = self.planner.plan_adaptive_candidates(
                        target,
                        visual_robot_state,
                        place_name=place_name,
                        return_to_pick=True,
                        grasp_z_m=point_grasp_z_m,
                        grasp_rpy=point_grasp_rpy,
                        task_b_anchor_pose=task_b_anchor_pose,
                        vertical_grasp_approach_height_m=(
                            point_vertical_grasp_approach_height_m
                        ),
                        jaw_tip_object_offset_m=(
                            point_grasp_jaw_tip_object_offset_m
                        ),
                    )
                emit(
                    TaskPhase.PLAN,
                    f"{place_name}: generated {len(adaptive_candidates)} "
                    "object-relative grasp candidate(s)",
                )
            else:
                plan = self.planner.plan(
                    target,
                    state,
                    place_pose=None if return_to_pick else place_pose,
                    place_name=place_name,
                    return_to_pick=return_to_pick,
                    grasp_z_m=point_grasp_z_m,
                    grasp_rpy=point_grasp_rpy,
                    task_b_anchor_pose=task_b_anchor_pose,
                    grasp_xy_from_observation=point_grasp_xy_from_observation,
                    grasp_xy_m=point_grasp_xy_m,
                    grasp_xy_offset_m=point_grasp_xy_offset_m,
                    vertical_grasp_descent=point_vertical_grasp_descent,
                    vertical_grasp_approach_height_m=(
                        point_vertical_grasp_approach_height_m
                    ),
                    grasp_target_is_jaw_tip=(
                        self.config.task.mode in {TaskMode.C, TaskMode.D}
                        and self.config.motion.grasp_safety.enabled
                    ),
                )
                # The visual target is a jaw-tip point for C/D.  Apply the
                # complete geometry envelope before the route-level IK/FK and
                # collision precheck can consume the plan.
                self._validate_task_cd_grasp_safety_plan(plan, emit, place_name)
                if point_vertical_grasp_descent:
                    self._validate_vertical_grasp_plan(plan, emit, place_name)
        except (ValueError, RuntimeError) as exc:
            emit(TaskPhase.PLAN, f"rejected: {exc}")
            self.policy.commit_failure()
            return PickResult(PickStatus.REJECTED, target, str(exc), tuple(events))

        # Adaptive candidates are checked as a safety-ordered fallback list.
        # Depending on configuration the pipeline either stops at the first
        # fully valid route or evaluates every route and ranks the valid ones.
        # No candidate route is executed during this search, and the gripper
        # remains untouched until a route has passed.
        if self._box_relative_five_point:
            successful, adaptive_failures = (
                self._precheck_all_box_relative_candidates(
                    adaptive_candidates,
                    emit,
                    place_name=place_name,
                )
            )
            if len(successful) != len(adaptive_candidates):
                message = (
                    f"{place_name}: {len(adaptive_candidates) - len(successful)} "
                    "of five box-relative routes failed the all-or-nothing "
                    "precheck; no item match was attempted and no motion or "
                    "gripper command was issued. "
                    + " | ".join(adaptive_failures[:5])
                )
                emit(TaskPhase.PLAN, message)
                self.policy.commit_failure()
                return PickResult(
                    PickStatus.REJECTED,
                    target,
                    message,
                    tuple(events),
                )

            # Item matching is deliberately delayed until every one of the
            # five possible routes has passed.  A match can therefore never
            # select an unchecked route out of the five.
            capture = frame.capture_robot_state
            assert capture is not None
            base_camera = self.planner.base_camera_transform(capture.before)
            item_center = detection_center_base_xyz(
                target,
                base_camera_transform=base_camera,
            )
            try:
                item_match = match_item_to_box_point(
                    item_center,
                    pose=box_pose,
                    layout=box_layout,
                    config=self.config.adaptive_inverse.box_relative_five_point,
                )
            except (BoxPoseError, ValueError, RuntimeError) as exc:
                message = (
                    f"{place_name}: item-to-box-point matching failed after all "
                    f"five routes passed; no motion: {exc}"
                )
                emit(TaskPhase.PLAN, message)
                self.policy.commit_failure()
                return PickResult(
                    PickStatus.REJECTED,
                    target,
                    message,
                    tuple(events),
                )
            selected_name = item_match.point.name
            selected = successful.get(selected_name)
            if selected is None:
                message = (
                    f"{place_name}: item matched {selected_name!r}, but that "
                    "route is absent from the complete precheck result; no motion"
                )
                emit(TaskPhase.PLAN, message)
                self.policy.commit_failure()
                return PickResult(
                    PickStatus.REJECTED,
                    target,
                    message,
                    tuple(events),
                )
            selected_candidate, plan, route = selected
            if selected_candidate.box_point_name != selected_name:
                raise RuntimeError(
                    "box-relative item match and selected route disagree"
                )
            adaptive_candidates = (
                replace(selected_candidate, item_match=item_match),
            )
            self._validate_vertical_grasp_plan(
                plan,
                emit,
                place_name,
                require_tool_down=True,
            )
            waypoints = plan.ordered_waypoints()
            emit(
                TaskPhase.PLAN,
                f"{place_name}: all five routes prechecked; item match "
                f"{selected_name!r} distance={item_match.distance_m:.4f} m "
                f"margin={item_match.margin_m:.4f} m; selected the matching "
                f"prechecked route with {len(waypoints)} waypoint(s)",
            )
        elif adaptive_enabled:
            plan, route, adaptive_failures = self._precheck_adaptive_candidates(
                adaptive_candidates,
                emit,
                place_name=place_name,
            )
            if plan is None:
                message = (
                    f"{place_name}: all {len(adaptive_candidates)} adaptive grasp "
                    "candidate(s) were rejected before any motion or gripper "
                    "command. "
                    + " | ".join(adaptive_failures[:5])
                )
                emit(TaskPhase.PLAN, message)
                self.policy.commit_failure()
                return PickResult(
                    PickStatus.REJECTED,
                    target,
                    message,
                    tuple(events),
                )
            if self._tilted_visual_grasp:
                self._validate_tool_axis_grasp_plan(plan, emit, place_name)
            else:
                self._validate_vertical_grasp_plan(
                    plan,
                    emit,
                    place_name,
                    require_tool_down=self._arbitrary_placement,
                )
            waypoints = plan.ordered_waypoints()
            emit(
                TaskPhase.PLAN,
                f"selected adaptive full-route candidate; "
                f"planned {len(waypoints)} waypoint(s)",
            )
        else:
            waypoints = plan.ordered_waypoints()
            emit(TaskPhase.PLAN, f"planned {len(waypoints)} waypoint(s)")

            # Screen the high approach endpoint before refining the complete
            # six-segment route.  A remote or ill-conditioned grasp point can
            # fail fixed-pose IK here in a few seconds, whereas discovering the
            # same failure through the Cartesian recursion costs tens of
            # seconds.  This check is read-only and runs before the gripper is
            # opened or the first motion command is published.
            condition_check = getattr(
                self.motion,
                "precheck_pose_condition",
                None,
            )
            if (
                self.config.task.mode is TaskMode.B
                and callable(condition_check)
                and waypoints
            ):
                try:
                    endpoint_sigma = condition_check(
                        waypoints[0][1],
                        label=f"{place_name} pregrasp",
                    )
                    if endpoint_sigma is not None:
                        endpoint_sigma = float(endpoint_sigma)
                        if not np.isfinite(endpoint_sigma):
                            raise RuntimeError(
                                "pregrasp endpoint returned a non-finite "
                                "Jacobian sigma"
                            )
                except (TypeError, ValueError, RuntimeError) as exc:
                    visual_x, visual_y = (
                        float(value) for value in plan.grasp_pose.xyz[:2]
                    )
                    message = (
                        f"Task B point {place_name!r} visual_xy="
                        f"({visual_x:.6f}, {visual_y:.6f}): "
                        "机械臂正在搜索可达抓取姿态；当前 pregrasp 端点未通过，"
                        "物体保持不动，未发送任何运动或夹爪命令。"
                        f"pregrasp 端点预检失败: {exc}"
                    )
                    emit(TaskPhase.PLAN, message)
                    self.policy.commit_failure()
                    return PickResult(
                        PickStatus.REJECTED,
                        target,
                        message,
                        tuple(events),
                    )
                if endpoint_sigma is not None:
                    emit(
                        TaskPhase.PLAN,
                        f"pregrasp endpoint condition passed: "
                        f"min_sigma={endpoint_sigma:.6f}",
                    )

            # Validate the entire motion sequence before opening the jaws or
            # publishing the first command.  Without this route-level gate, a
            # late IK/limit failure could leave the robot holding an object
            # halfway through the pick-and-return sequence.
            try:
                route = self.motion.precheck_linear_targets(
                    tuple(pose for _name, pose in waypoints),
                    step_m=self.config.motion.cartesian_step_m,
                )
            except (ValueError, RuntimeError) as exc:
                message = self._task_b_route_rejection_message(
                    place_name,
                    plan,
                    waypoints,
                    exc,
                )
                emit(TaskPhase.PLAN, message)
                self.policy.commit_failure()
                return PickResult(PickStatus.REJECTED, target, message, tuple(events))
            emit(
                TaskPhase.PLAN,
                f"full-route precheck passed: {len(route.segments)} segment(s)",
            )
            if len(route.segments) != len(waypoints):
                message = (
                    "full-route precheck returned a different segment count: "
                    f"{len(route.segments)} != {len(waypoints)}"
                )
                emit(TaskPhase.PLAN, message)
                self.policy.commit_failure()
                return PickResult(PickStatus.REJECTED, target, message, tuple(events))

        # Bind the already prechecked route to an immutable per-scene
        # offline consensus before the jaws can open or any command can be
        # published.  The gate never authorizes hardware motion; it only
        # decides whether this offline dry-run plan may continue.
        if self.grasp_consensus_gate is not None:
            try:
                consensus = self.grasp_consensus_gate.evaluate(plan.grasp_pose)
            except Exception as exc:
                consensus = None
                consensus_error = f"consensus gate raised unexpectedly: {exc}"
            else:
                consensus_error = ""
            if (
                consensus is None
                or not consensus.accepted
                or consensus.offline_only is not True
                or consensus.motion_authorized is not False
            ):
                reason = (
                    consensus_error
                    or (consensus.reason if consensus is not None else "")
                    or "offline consensus gate did not return a safe acceptance"
                )
                message = (
                    "offline consensus gate rejected before gripper open or "
                    f"motion: {reason}"
                )
                emit(TaskPhase.PLAN, message)
                self.policy.commit_failure()
                return PickResult(
                    PickStatus.REJECTED,
                    target,
                    message,
                    tuple(events),
                )
            translation_error = (
                "n/a"
                if consensus.pose_translation_error_m is None
                else f"{consensus.pose_translation_error_m:.6f} m"
            )
            rotation_error = (
                "n/a"
                if consensus.pose_jaw_rotation_error_deg is None
                else f"{consensus.pose_jaw_rotation_error_deg:.6f} deg"
            )
            emit(
                TaskPhase.PLAN,
                "offline consensus gate passed: "
                f"cluster_id={consensus.cluster_id}; "
                f"representative={consensus.representative_candidate}; "
                f"distinct_frames={consensus.distinct_frame_count}; "
                f"pose_translation_error={translation_error}; "
                f"pose_jaw_rotation_error={rotation_error}; "
                f"report_sha256={consensus.report_sha256}; "
                f"quality_validated={str(consensus.report_quality_validated).lower()}; "
                "motion_authorized=false",
            )

        if self.config.gripper.open_before_approach and self.gripper.state != GripperState.OPEN:
            emit(TaskPhase.OPEN_GRIPPER, "opening jaws before entering approach corridor")
            try:
                self.gripper.open()
            except Exception as exc:
                emit(TaskPhase.ABORT, f"gripper open failed: {exc}")
                return PickResult(PickStatus.ABORTED, target, str(exc), tuple(events))

        try:
            for (name, pose), segment in zip(waypoints, route.segments):
                if segment.target != pose:
                    raise RuntimeError(
                        f"pre-checked segment {name!r} target changed after planning"
                    )
                phase = _PHASE_FOR_WAYPOINT.get(name, TaskPhase.APPROACH)
                emit(phase, f"move to {name}: xyz={tuple(round(v, 4) for v in pose.xyz)}")
                self.motion.execute_prechecked_segment(
                    segment,
                    name=name,
                    speed_mps=self.config.motion.speed_mps,
                )
                if name == "lift" and return_to_pick:
                    hold_s = float(self.config.task_b.lift_hold_s)
                    if hold_s > 0.0:
                        emit(
                            TaskPhase.LIFT,
                            f"lift pose reached; holding for {hold_s:.1f} s before return",
                        )
                        time.sleep(hold_s)
                elif name == "grasp":
                    emit(TaskPhase.GRASP, "reached grasp pose; closing jaws")
                    self.gripper.close()
                    emit(TaskPhase.VERIFY_HOLD, "checking whether the object is held")
                    held, reason = self.hold_verifier.verify(target, plan)
                    if not held:
                        self._recover_after_failed_grasp(plan, events, reason)
                        self.policy.commit_failure()
                        return PickResult(
                            PickStatus.ABORTED,
                            target,
                            reason,
                            tuple(events),
                        )
                    emit(TaskPhase.VERIFY_HOLD, reason)
                elif name == "release":
                    emit(TaskPhase.RELEASE, "opening jaws to release the object")
                    self.gripper.open()
        except Exception as exc:
            emit(TaskPhase.ABORT, f"motion failed: {exc}")
            self.policy.commit_failure()
            return PickResult(PickStatus.ABORTED, target, str(exc), tuple(events))

        # The confirmed acquisition object is not retained as a Selection in
        # every backend, so rebuild the equivalent successful selection.
        selection = self.policy.select([target])
        self.policy.commit_success(selection)
        if self.config.task.mode is TaskMode.B:
            if self.config.task_b.keep_same_target:
                self._task_b_target_label = target.label
            completed = self._task_b_pick_index + 1
            if not self._arbitrary_placement and not self._box_relative_five_point:
                # Legacy five-point mode advances only after every waypoint,
                # including release and the final retreat, has completed
                # without raising.  Record the actual released XY so a returned
                # object cannot be mistaken for the next reference point.
                self._task_b_completed_xy.append(
                    (
                        float(plan.grasp_pose.xyz[0]),
                        float(plan.grasp_pose.xyz[1]),
                    )
                )
            self._task_b_pick_index = completed
            if self._box_relative_five_point:
                if completed < self.config.task_b.rounds:
                    message = (
                        f"{place_name}: box-relative target returned in place "
                        "and the arm retreated; no legacy reference or released "
                        "XY was stored, so the next round requires a fresh stable "
                        f"RGB-D frame ({completed}/{self.config.task_b.rounds})"
                    )
                else:
                    message = (
                        f"{place_name}: box-relative target returned in place "
                        "and the arm retreated; all five fixed box-local points "
                        "completed from current-frame box poses"
                    )
            elif self._arbitrary_placement:
                if completed < self.config.task_b.rounds:
                    message = (
                        f"{place_name}: arbitrary-placement target returned "
                        "in place and the arm retreated; this round's XY was "
                        "not stored, so the next round requires a fresh visual "
                        f"confirmation ({completed}/{self.config.task_b.rounds})"
                    )
                else:
                    message = (
                        f"{place_name}: arbitrary-placement target returned "
                        "in place and the arm retreated; all rounds completed "
                        "without retaining any object-position history"
                    )
            elif completed < self.config.task_b.rounds:
                message = (
                    f"{place_name}: object returned at its pick point and arm "
                    f"retreated; next round waits for visual confirmation at "
                    f"expected point {completed + 1}/{self.config.task_b.rounds}"
                )
            else:
                message = (
                    f"{place_name}: object returned at its pick point and arm "
                    "retreated; all expected points completed"
                )
        elif plan.release_pose is None:
            message = "object lifted and held; no place pose configured"
        else:
            message = "object picked, transferred and released"
        emit(TaskPhase.DONE, message)
        return PickResult(
            PickStatus.SUCCESS, target, message, tuple(events), place_name=plan.place_name
        )

    def _acquire_task_b_target(
        self,
        initial_state: RobotState,
        expected_point: tuple[float, ...] | None,
        expected_name: str,
        expected_label: str | None,
        round_number: int,
        total_rounds: int,
        emit,
        *,
        completed_points: Sequence[tuple[float, ...]] = (),
    ) -> tuple[Detection | None, Frame | None, RobotState, str]:
        """Wait until the current round's target is seen at its reference point.

        A returned object remains visible at the previous point, so a plain
        "any target" observation would incorrectly start the next cycle.
        The point filter is therefore applied before target selection and this
        method keeps polling until the target is confirmed at the next point or
        the configured timeout expires.
        """

        if expected_point is None:
            raise RuntimeError(
                f"task B round {round_number} has no configured reference point"
            )

        timeout_s = float(self.config.task_b.next_point_timeout_s)
        poll_s = float(self.config.task_b.next_point_poll_s)
        started = time.monotonic()
        attempt = 0
        last_reason = "target has not been observed at the expected point"
        state = initial_state
        frame: Frame | None = None
        while True:
            attempt += 1
            point_xyz = tuple(float(v) for v in expected_point[:2])
            label_text = expected_label or "any class"
            emit(
                TaskPhase.SCAN,
                f"round {round_number}/{total_rounds}: waiting for {label_text} "
                f"at {expected_name!r} xy=({point_xyz[0]:.3f}, "
                f"{point_xyz[1]:.3f}) within "
                f"{self.config.task_b.pick_pose_tolerance_m:.3f} m, "
                f"completed_points_excluded={len(completed_points)} "
                f"(wait attempt {attempt})",
            )
            target, frame, state, reason = self._acquire_target(
                state,
                emit,
                expected_point=expected_point,
                expected_label=expected_label,
                expected_name=expected_name,
                completed_points=completed_points,
            )
            if target is not None:
                emit(
                    TaskPhase.SCAN,
                    f"round {round_number}/{total_rounds}: target confirmed at "
                    f"expected point {expected_name!r}",
                )
                return target, frame, state, reason

            last_reason = reason
            elapsed = time.monotonic() - started
            if timeout_s > 0.0 and elapsed >= timeout_s:
                message = (
                    f"timed out after {timeout_s:.1f} s waiting for target at "
                    f"expected point {expected_name!r} (round "
                    f"{round_number}/{total_rounds}): {last_reason}"
                )
                emit(TaskPhase.SCAN, message)
                return None, frame, state, message
            if timeout_s > 0.0:
                remaining = max(0.0, timeout_s - elapsed)
                wait_text = f"; {remaining:.1f} s remaining"
            else:
                wait_text = "; waiting until interrupted"
            emit(
                TaskPhase.SCAN,
                f"target not yet confirmed at {expected_name!r}{wait_text}; "
                "no pick motion will be issued",
            )
            time.sleep(poll_s)

    def _acquire_target(
        self,
        initial_state: RobotState,
        emit,
        *,
        expected_point: tuple[float, ...] | None = None,
        expected_label: str | None = None,
        expected_name: str = "",
        completed_points: Sequence[tuple[float, ...]] = (),
    ) -> tuple[Detection | None, Frame | None, RobotState, str]:
        """Confirm a target while actively scanning safe observation points.

        Task B supplies an expected pick point.  When Cartesian observation
        offsets are configured, the arm first moves to safe viewpoints around
        that point and looks for the banana there.  This is deliberately more
        than a stationary crop/wrist sweep: an empty current view must produce
        real search motion.  The legacy wrist-only sweep remains as a fallback
        for tasks without reference points and for compatibility with older
        configurations.
        """

        target, frame, reason = self._observe_target(
            emit,
            expected_point=expected_point,
            expected_label=expected_label,
            expected_name=expected_name,
            completed_points=completed_points,
            fallback_state=initial_state,
        )
        if target is not None:
            return target, frame, self._fresh_state(initial_state), reason

        search = self.config.search
        if not search.enabled:
            return None, frame, self._fresh_state(initial_state), reason

        state = self._fresh_state(initial_state)
        last_frame = frame
        last_reason = reason

        relative_observation_poses = self._relative_observation_poses(state)
        if relative_observation_poses:
            relative_route = self._precheck_relative_observation_route(
                state,
                relative_observation_poses,
                emit,
                expected_name=expected_name,
            )
            if relative_route is None:
                last_reason = (
                    f"{last_reason}; relative observation adjustment was "
                    "rejected by the all-or-nothing safety precheck"
                )
            else:
                target, frame, state, reason = (
                    self._search_relative_observation_views(
                        state,
                        relative_observation_poses,
                        relative_route,
                        emit,
                        expected_point=expected_point,
                        expected_label=expected_label,
                        expected_name=expected_name,
                        completed_points=completed_points,
                    )
                )
                if target is not None:
                    return target, frame, self._fresh_state(state), reason
                if frame is not None:
                    last_frame = frame
                last_reason = reason

        observation_poses = self._precheck_cartesian_search_route(
            self._cartesian_search_poses(state, expected_point),
            emit,
            expected_point=expected_point,
            expected_name=expected_name,
        )
        if observation_poses:
            target, frame, state, reason = self._search_cartesian_views(
                state,
                observation_poses,
                emit,
                expected_point=expected_point,
                expected_label=expected_label,
                expected_name=expected_name,
                completed_points=completed_points,
            )
            if target is not None:
                return target, frame, self._fresh_state(state), reason
            if frame is not None:
                last_frame = frame
            last_reason = reason

            if search.return_to_initial_on_miss:
                try:
                    if initial_state.joints is None:
                        raise RuntimeError(
                            "initial joint feedback is unavailable for the "
                            "six-axis-safe search return"
                        )
                    if not hasattr(self.motion, "move_joints"):
                        raise RuntimeError(
                            "configured motion backend cannot execute the "
                            "joint-space search return"
                        )
                    self._set_vision_status("MOVING: returning to initial search pose")
                    # The search views can leave the wrist in an ill-conditioned
                    # Cartesian branch.  Returning through the recorded six-joint
                    # startup posture is deterministic and avoids re-solving IK
                    # at the singular boundary after a visual miss.
                    self.motion.move_joints(
                        initial_state.joints,
                        name="task_search_return",
                    )
                except Exception as exc:
                    emit(
                        TaskPhase.SCAN,
                        f"search return move failed; stopping without a pick: {exc}",
                    )
                    return (
                        None,
                        last_frame,
                        self._fresh_state(state),
                        f"{last_reason}; return-to-initial failed: {exc}",
                    )
                if search.settle_s > 0.0:
                    time.sleep(search.settle_s)
                state = self._fresh_state(state)

        views = self._search_views()
        scan_views = views[1:]
        if not scan_views:
            return (
                None,
                last_frame,
                self._fresh_state(state),
                f"{last_reason}; no additional wrist search views configured",
            )

        if self.motion.dry_run:
            self._validate_search_views(state, scan_views, emit)
            return (
                None,
                last_frame,
                self._fresh_state(state),
                f"{last_reason}; dry-run checked {len(scan_views)} wrist view(s) "
                "but did not send motion commands",
            )

        state = self._fresh_state(state)
        if not state.ready or state.joints is None:
            raise RuntimeError(
                "automatic target search needs AutoReady with joint feedback: "
                f"{state.message}"
            )
        if not hasattr(self.motion, "move_joints"):
            raise RuntimeError("configured motion backend cannot run joint-space search")

        q_initial = np.asarray(state.joints, dtype=float)
        for index, (pitch_deg, yaw_deg) in enumerate(scan_views, start=1):
            try:
                q_target = joint_scan_target(
                    q_initial,
                    np.asarray(self.config.servo.camera_to_tool, dtype=float),
                    pitch_down_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                )
            except RuntimeError as exc:
                emit(
                    TaskPhase.SCAN,
                    f"search view {index}/{len(scan_views)} skipped: IK failed: {exc}",
                )
                continue
            if not joints_within_limits(q_target):
                emit(
                    TaskPhase.SCAN,
                    f"search view {index}/{len(scan_views)} skipped: joint limits",
                )
                continue

            state = self._fresh_state(state)
            if not state.ready:
                raise RuntimeError(f"robot not ready during search: {state.message}")
            self._set_vision_status(
                f"MOVING: wrist search view {index}/{len(scan_views)} "
                f"pitch={pitch_deg:+.0f} yaw={yaw_deg:+.0f}"
            )
            self.motion.move_joints(
                q_target,
                name=(
                    f"task_search_{index:02d}_pitch_{pitch_deg:+.0f}deg_"
                    f"yaw_{yaw_deg:+.0f}deg"
                ),
            )
            if search.settle_s > 0.0:
                time.sleep(search.settle_s)

            state = self._fresh_state(state)
            target, frame, reason = self._observe_target(
                emit,
                expected_point=expected_point,
                expected_label=expected_label,
                expected_name=expected_name,
                completed_points=completed_points,
                fallback_state=state,
            )
            if target is not None:
                return target, frame, self._fresh_state(state), reason
            if frame is not None:
                last_frame = frame
            last_reason = reason

        if search.return_to_initial_on_miss:
            self._set_vision_status("MOVING: returning to initial search pose")
            self.motion.move_joints(q_initial, name="task_search_return")
        return (
            None,
            last_frame,
            self._fresh_state(state),
            f"target not confirmed in the current view or {len(scan_views)} "
            f"wrist search views: {last_reason}",
        )

    def _cartesian_search_poses(
        self,
        state: RobotState,
        expected_point: tuple[float, ...] | None,
    ) -> tuple[Pose, ...]:
        """Build safe, active observation poses around a Task B reference point."""

        if expected_point is None:
            return ()
        search = self.config.search
        if not search.observation_offsets_xy_m:
            return ()
        if len(expected_point) < 3:
            raise RuntimeError(f"invalid Task B reference point: {expected_point!r}")
        base_x, base_y, base_z = (float(v) for v in expected_point[:3])
        clearance = float(search.observation_clearance_m)
        if self.config.motion.orientation == "fixed":
            rpy = tuple(float(v) for v in self.config.motion.fixed_rpy)
        elif len(expected_point) >= 6:
            # The five point files store the operator-validated observation
            # orientation.  Reuse it for active search instead of carrying the
            # previous point's retreat orientation into the next point.
            rpy = tuple(float(v) for v in expected_point[3:6])
        elif state.pose is not None:
            rpy = tuple(float(v) for v in state.pose.rpy)
        else:
            rpy = tuple(float(v) for v in self.config.motion.fixed_rpy)
        return tuple(
            Pose((base_x + float(dx), base_y + float(dy), base_z + clearance), rpy)
            for dx, dy in search.observation_offsets_xy_m
        )

    def _search_pose_within_workspace(self, pose: Pose) -> bool:
        low = np.asarray(self.config.motion.workspace_min, dtype=float) + float(
            self.config.motion.workspace_margin
        )
        high = np.asarray(self.config.motion.workspace_max, dtype=float) - float(
            self.config.motion.workspace_margin
        )
        xyz = np.asarray(pose.xyz, dtype=float)
        return bool(np.all(xyz >= low - 1e-9) and np.all(xyz <= high + 1e-9))

    def _search_cartesian_views(
        self,
        initial_state: RobotState,
        poses: Sequence[Pose],
        emit,
        *,
        expected_point: tuple[float, ...] | None,
        expected_label: str | None,
        expected_name: str,
        completed_points: Sequence[tuple[float, ...]] = (),
    ) -> tuple[Detection | None, Frame | None, RobotState, str]:
        """Visit safe Cartesian viewpoints and observe after each move."""

        state = initial_state
        last_frame: Frame | None = None
        last_reason = "target has not been observed"
        for index, pose in enumerate(poses, start=1):
            if not self._search_pose_within_workspace(pose):
                emit(
                    TaskPhase.SCAN,
                    f"active search view {index}/{len(poses)} skipped: "
                    f"outside workspace xyz={tuple(round(v, 3) for v in pose.xyz)}",
                )
                continue
            if not self.motion.dry_run and not state.ready:
                raise RuntimeError(
                    f"robot not ready before active search view {index}: {state.message}"
                )
            emit(
                TaskPhase.SCAN,
                f"active search move {index}/{len(poses)}: "
                f"xyz={tuple(round(v, 3) for v in pose.xyz)}",
            )
            self._set_vision_status(
                f"MOVING: active search view {index}/{len(poses)}"
            )
            try:
                self.motion.move_to(
                    pose,
                    name=f"task_search_position_{index:02d}",
                    speed_mps=self.config.motion.speed_mps,
                )
            except Exception as exc:
                message = (
                    f"active search view {index}/{len(poses)} move failed; "
                    f"stopping without a pick: {exc}"
                )
                emit(TaskPhase.SCAN, message)
                raise RuntimeError(message) from exc
            if self.config.search.settle_s > 0.0:
                time.sleep(self.config.search.settle_s)

            state = self._fresh_state(state)
            target, frame, reason = self._observe_target(
                emit,
                expected_point=expected_point,
                expected_label=expected_label,
                expected_name=expected_name,
                completed_points=completed_points,
                fallback_state=state,
            )
            last_frame = frame
            last_reason = reason
            if target is not None:
                return target, frame, self._fresh_state(state), reason
            emit(
                TaskPhase.SCAN,
                f"active search view {index}/{len(poses)}: "
                f"no confirmed target: {reason}",
            )

        return None, last_frame, self._fresh_state(state), last_reason

    def _observe_target(
        self,
        emit,
        *,
        expected_point: tuple[float, ...] | None = None,
        expected_label: str | None = None,
        expected_name: str = "",
        completed_points: Sequence[tuple[float, ...]] = (),
        fallback_state: RobotState | None = None,
    ) -> tuple[Detection | None, Frame | None, str]:
        """Require the same target box/class in consecutive fresh frames.

        When ``expected_point`` is supplied, detections are filtered by the
        base-frame XY reference point before ``TargetPolicy.select`` is called.
        This ordering is important: a high-confidence object at the previous
        point must never win selection while the next point is still empty.
        """

        search = self.config.search
        observe_frames = max(1, int(search.observe_frames))
        confirm_hits = min(max(1, int(search.confirm_hits)), observe_frames)
        consecutive_hits = 0
        previous: Detection | None = None
        best: Detection | None = None
        best_frame: Frame | None = None
        best_score = (-1.0, -1.0, -1.0)
        latest_frame: Frame | None = None
        last_reason = "no usable detection"

        for index in range(observe_frames):
            emit(
                TaskPhase.DETECT,
                f"observation {index + 1}/{observe_frames}: waiting for a fresh frame",
            )
            if self.config.adaptive_inverse.enabled:
                # Keep the physical feedback state attached to the exact RGB-D
                # frame.  ``get_state`` may expose a dry-run virtual posture,
                # whereas ``get_feedback_state`` always follows RobotInfo.
                before = self._motion_feedback_state()
                frame = self.vision.capture()
                after = self._motion_feedback_state()
                capture_state = self._validate_adaptive_capture_state(
                    before,
                    after,
                    frame_id=frame.frame_id,
                )
                frame = replace(frame, capture_robot_state=capture_state)
            else:
                frame = self.vision.capture()
            latest_frame = frame
            usable = [
                detection
                for detection in frame.detections
                if self._is_usable(detection)
            ]
            label_filter_reason = "no usable detection"
            if expected_label is not None:
                usable = [
                    detection
                    for detection in usable
                    if detection.label == expected_label
                ]
                label_filter_reason = (
                    f"no usable detection with expected class "
                    f"{expected_label!r}"
                )
            if expected_point is not None:
                state = (
                    self._fresh_state(fallback_state)
                    if fallback_state is not None
                    else fallback_state
                )
                usable, filter_reason = self._filter_task_b_candidates(
                    usable,
                    state,
                    expected_point,
                    expected_label,
                    expected_name,
                    completed_points,
                )
                if usable:
                    selection = self.policy.select(usable)
                else:
                    selection = Selection(None, False, filter_reason)
            else:
                selection = (
                    self.policy.select(usable)
                    if usable
                    else Selection(None, False, label_filter_reason)
                )
            if selection.accepted and selection.detection is not None:
                candidate = selection.detection
                if previous is not None and self._same_target(previous, candidate):
                    consecutive_hits += 1
                else:
                    consecutive_hits = 1
                previous = candidate
                candidate_score = self._candidate_score(selection)
                if best is None or candidate_score > best_score:
                    best = candidate
                    best_frame = frame
                    best_score = candidate_score
                emit(
                    TaskPhase.DETECT,
                    f"frame {frame.frame_id!r}: {len(frame.detections)} detection(s), "
                    f"{candidate.label} conf={candidate.confidence:.2f}, "
                    f"confirmation {consecutive_hits}/{confirm_hits}",
                )
                if consecutive_hits >= confirm_hits:
                    return (
                        candidate,
                        frame,
                        f"{candidate.label} confirmed in {consecutive_hits} consecutive frames",
                    )
            else:
                consecutive_hits = 0
                previous = None
                last_reason = selection.reason
                emit(
                    TaskPhase.DETECT,
                    f"frame {frame.frame_id!r}: {len(frame.detections)} detection(s), "
                    f"no confirmed target: {selection.reason}",
                )

            if index + 1 < observe_frames:
                time.sleep(0.05)

        reason = (
            f"target not confirmed in {observe_frames} frame(s): {last_reason}"
        )
        if best is not None:
            reason += (
                f"; best unconfirmed detection was {best.label!r} "
                f"conf={best.confidence:.2f}"
            )
        return None, best_frame or latest_frame, reason

    def _filter_task_b_candidates(
        self,
        detections: list[Detection],
        state: RobotState | None,
        expected_point: tuple[float, ...],
        expected_label: str | None,
        expected_name: str,
        completed_points: Sequence[tuple[float, ...]] = (),
    ) -> tuple[list[Detection], str]:
        """Keep only detections that clearly belong to the current point.

        The comparison is deliberately two-dimensional.  Pick-point files are
        reference locations in the robot XY plane; table-height calibration is
        handled later by the grasp-depth/z offset, so a small Z error must not
        make an otherwise valid target disappear.

        Every configured five-point reference is considered, not only the
        current and already completed points.  By default a detection is
        accepted only when the current point is its nearest reference and is
        closer than the next-nearest reference by ``point_assignment_margin_m``.
        The operator-approved fixed-point mode may disable that nearest-point
        requirement so a stable class/depth detection inside the current
        distance window starts the taught grasp immediately.

        For rounds after the first, ``completed_points`` contains every point
        whose pick-and-return cycle already finished.  A candidate must also
        be closer to the current point than to all of those points by at
        least ``point_assignment_margin_m``.  The actual released XY is used,
        because it can differ from the configured reference.
        """

        expected_xy = np.asarray(expected_point[:2], dtype=float)
        if expected_xy.shape != (2,) or not np.all(np.isfinite(expected_xy)):
            return [], f"invalid Task B reference point {expected_point!r}"

        reference_points: list[tuple[int, np.ndarray]] = []
        for index, point in enumerate(self.config.place_poses, start=1):
            xy = np.asarray(point[:2], dtype=float)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                return [], f"invalid configured Task B reference point {index}: {point!r}"
            reference_points.append((index, xy))
        if not reference_points:
            return [], "no Task B reference points are configured"
        expected_index = next(
            (
                index
                for index, xy in reference_points
                if np.allclose(xy, expected_xy, rtol=0.0, atol=1e-9)
            ),
            None,
        )
        if expected_index is None:
            return [], (
                f"expected Task B reference point {expected_point!r} is not in "
                "the configured point list"
            )

        excluded: list[tuple[int, np.ndarray]] = []
        for index, point in enumerate(completed_points, start=1):
            xy = np.asarray(point[:2], dtype=float)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                return [], f"invalid completed Task B reference point {index}: {point!r}"
            excluded.append((index, xy))

        tolerance_m = float(self.config.task_b.pick_pose_tolerance_m)
        assignment_margin_m = float(self.config.task_b.point_assignment_margin_m)
        require_nearest_reference = bool(
            self.config.task_b.require_nearest_reference_point
        )
        accepted: list[tuple[Detection, float]] = []
        ambiguous: list[str] = []
        misassigned: list[str] = []
        reference_ambiguous: list[str] = []
        nearest_text = "no detection had a usable base-frame position"
        nearest_distance = float("inf")

        for detection in detections:
            try:
                base_xyz = np.asarray(
                    self.planner.detection_to_base(detection, state), dtype=float
                ).reshape(-1)
            except Exception as exc:
                nearest_text = (
                    f"nearest usable candidate could not be transformed: {exc}"
                )
                continue
            if base_xyz.size < 3 or not np.all(np.isfinite(base_xyz[:3])):
                nearest_text = "a candidate produced a non-finite base-frame position"
                continue

            distance_m = float(np.linalg.norm(base_xyz[:2] - expected_xy))
            if distance_m < nearest_distance:
                nearest_distance = distance_m
                actual = detection.label
                expected = expected_label or "any class"
                nearest_text = (
                    f"nearest detection {actual!r} was {distance_m:.3f} m from "
                    f"{expected_name!r} (expected class {expected!r})"
                )

            if expected_label is not None and detection.label != expected_label:
                continue
            if distance_m > tolerance_m:
                continue

            if require_nearest_reference:
                reference_distances = sorted(
                    (
                        float(np.linalg.norm(base_xyz[:2] - xy)),
                        index,
                    )
                    for index, xy in reference_points
                )
                nearest_index = reference_distances[0][1]
                if nearest_index != expected_index:
                    misassigned.append(
                        f"{detection.label!r} at ({base_xyz[0]:.3f}, "
                        f"{base_xyz[1]:.3f}) is {distance_m:.3f} m from "
                        f"{expected_name!r} but closest to reference point "
                        f"{nearest_index} at {reference_distances[0][0]:.3f} m"
                    )
                    continue
                if len(reference_distances) > 1:
                    second_distance, second_index = reference_distances[1]
                    if second_distance - distance_m < assignment_margin_m:
                        reference_ambiguous.append(
                            f"{detection.label!r} at ({base_xyz[0]:.3f}, "
                            f"{base_xyz[1]:.3f}) is only {second_distance - distance_m:.3f} m "
                            f"closer to {expected_name!r} than to reference point "
                            f"{second_index}; required margin "
                            f"{assignment_margin_m:.3f} m"
                        )
                        continue

            ambiguous_index: int | None = None
            ambiguous_distance = float("inf")
            for completed_index, completed_xy in excluded:
                completed_distance = float(
                    np.linalg.norm(base_xyz[:2] - completed_xy)
                )
                if completed_distance <= distance_m + assignment_margin_m:
                    ambiguous_index = completed_index
                    ambiguous_distance = completed_distance
                    break
            if ambiguous_index is not None:
                ambiguous.append(
                    f"{detection.label!r} at ({base_xyz[0]:.3f}, "
                    f"{base_xyz[1]:.3f}) is {distance_m:.3f} m from "
                    f"{expected_name!r} but {ambiguous_distance:.3f} m from "
                    f"completed point {ambiguous_index}; required margin "
                    f"{assignment_margin_m:.3f} m"
                )
                continue

            accepted.append((detection, distance_m))

        if accepted:
            nearest_accepted = min(distance for _, distance in accepted)
            expected = expected_label or "any class"
            reason = (
                f"accepted {len(accepted)} detection(s) at {expected_name!r}: "
                f"class={expected!r}, nearest_xy={nearest_accepted:.3f} m, "
                f"tolerance={tolerance_m:.3f} m, "
                f"nearest_reference_required={require_nearest_reference}, "
                f"assignment_margin={assignment_margin_m:.3f} m, "
                f"all_reference_points_checked={len(reference_points)}, "
                f"completed_points_excluded={len(excluded)}"
            )
            return [detection for detection, _ in accepted], reason

        expected = expected_label or "any class"
        reason = (
            f"no target at expected point {expected_name!r} within "
            f"{tolerance_m:.3f} m (expected class {expected!r}); {nearest_text}"
        )
        if ambiguous:
            reason += "; rejected ambiguous detection(s) near completed point(s): "
            reason += " | ".join(ambiguous[:3])
        if misassigned:
            reason += "; rejected detection(s) closest to another reference point: "
            reason += " | ".join(misassigned[:3])
        if reference_ambiguous:
            reason += "; rejected detection(s) without enough reference-point margin: "
            reason += " | ".join(reference_ambiguous[:3])
        return [], reason

    def _search_views(self) -> list[tuple[float, float]]:
        search = self.config.search
        if not search.enabled:
            return [(0.0, 0.0)]
        pitches = search.pitch_down_offsets_deg or (0.0,)
        yaws = search.yaw_offsets_deg or (0.0,)
        return [(float(pitch), float(yaw)) for pitch in pitches for yaw in yaws]

    def _validate_search_views(self, state: RobotState, views, emit) -> None:
        if not state.ready or state.joints is None:
            emit(
                TaskPhase.SCAN,
                "search IK validation skipped: no AutoReady joint feedback",
            )
            return
        q_initial = np.asarray(state.joints, dtype=float)
        for index, (pitch_deg, yaw_deg) in enumerate(views, start=1):
            try:
                q_target = joint_scan_target(
                    q_initial,
                    np.asarray(self.config.servo.camera_to_tool, dtype=float),
                    pitch_down_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                )
            except RuntimeError as exc:
                emit(
                    TaskPhase.SCAN,
                    f"search IK {index}/{len(views)} pitch={pitch_deg:+.0f} "
                    f"yaw={yaw_deg:+.0f} failed: {exc}",
                )
                continue
            emit(
                TaskPhase.SCAN,
                f"search IK {index}/{len(views)} pitch={pitch_deg:+.0f} "
                f"yaw={yaw_deg:+.0f} limits_ok={joints_within_limits(q_target)}",
            )

    def _motion_feedback_state(self) -> RobotState:
        """Return physical controller feedback without dry-run virtual motion."""

        get_feedback_state = getattr(self.motion, "get_feedback_state", None)
        if callable(get_feedback_state):
            return get_feedback_state()
        return self.motion.get_state()

    def _validate_adaptive_capture_state(
        self,
        before: RobotState,
        after: RobotState,
        *,
        frame_id: str,
    ) -> CaptureRobotState:
        """Fail closed unless robot feedback is complete and stable per frame."""

        config = self.config.adaptive_inverse.box_relative_five_point
        for label, state in (("before", before), ("after", after)):
            if not state.ready:
                raise RuntimeError(
                    f"adaptive RGB-D frame {frame_id!r} {label} state is "
                    f"not ready: {state.message}"
                )
            if state.fault:
                raise RuntimeError(
                    f"adaptive RGB-D frame {frame_id!r} {label} state has "
                    f"a fault: {state.message}"
                )
            if state.pose is None:
                raise RuntimeError(
                    f"adaptive RGB-D frame {frame_id!r} {label} state has "
                    "no robot pose"
                )
            pose_values = np.asarray(
                (*state.pose.xyz, *state.pose.rpy), dtype=float
            )
            if pose_values.shape != (6,) or not np.all(
                np.isfinite(pose_values)
            ):
                raise RuntimeError(
                    f"adaptive RGB-D frame {frame_id!r} {label} pose is "
                    "malformed"
                )
            if state.joints is None:
                raise RuntimeError(
                    f"adaptive RGB-D frame {frame_id!r} {label} state has "
                    "no six-joint feedback"
                )
            joints = np.asarray(state.joints, dtype=float)
            if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                raise RuntimeError(
                    f"adaptive RGB-D frame {frame_id!r} {label} joints "
                    "must contain six finite values"
                )

        before_joints = np.asarray(before.joints, dtype=float)
        after_joints = np.asarray(after.joints, dtype=float)
        joint_delta_deg = float(
            np.degrees(np.max(np.abs(after_joints - before_joints)))
        )
        if joint_delta_deg > float(config.max_capture_joint_delta_deg):
            raise RuntimeError(
                f"adaptive RGB-D frame {frame_id!r} joint drift "
                f"{joint_delta_deg:.6f} deg exceeds "
                f"{config.max_capture_joint_delta_deg:.6f} deg"
            )

        before_xyz = np.asarray(before.pose.xyz, dtype=float)
        after_xyz = np.asarray(after.pose.xyz, dtype=float)
        translation_delta_m = float(np.linalg.norm(after_xyz - before_xyz))
        if translation_delta_m > float(config.max_capture_pose_translation_m):
            raise RuntimeError(
                f"adaptive RGB-D frame {frame_id!r} translation drift "
                f"{translation_delta_m:.6f} m exceeds "
                f"{config.max_capture_pose_translation_m:.6f} m"
            )
        rotation_delta_rad = rotation_angle_error(
            rpy_to_matrix(before.pose.rpy),
            rpy_to_matrix(after.pose.rpy),
        )
        rotation_limit_rad = float(
            np.radians(config.max_capture_pose_rotation_deg)
        )
        if rotation_delta_rad > rotation_limit_rad:
            raise RuntimeError(
                f"adaptive RGB-D frame {frame_id!r} rotation drift "
                f"{np.degrees(rotation_delta_rad):.6f} deg exceeds "
                f"{config.max_capture_pose_rotation_deg:.6f} deg"
            )
        return CaptureRobotState(before=before, after=after)

    def _precheck_all_box_relative_candidates(
        self,
        candidates,
        emit,
        *,
        place_name: str,
    ):
        """Precheck every route; return successes only if all five are safe."""

        successful: dict[str, tuple[object, object, object]] = {}
        failures: list[str] = []
        for candidate in candidates:
            name = str(candidate.box_point_name)
            plan, route, candidate_failures = self._precheck_adaptive_candidates(
                (candidate,),
                emit,
                place_name=f"{place_name}:{name}",
            )
            if plan is None or route is None:
                failures.extend(candidate_failures or (f"{name}: route rejected",))
                continue
            successful[name] = (candidate, plan, route)
        return successful, failures

    def _fresh_state(self, fallback: RobotState) -> RobotState:
        wait_for_state = getattr(self.motion, "wait_for_state", None)
        if wait_for_state is None:
            return self.motion.get_state() or fallback
        state = wait_for_state(1.0)
        return state if state.pose is not None else fallback

    def _same_target(self, first: Detection, second: Detection) -> bool:
        if first.label != second.label:
            return False
        if first.instance_id and second.instance_id:
            return first.instance_id == second.instance_id
        return box_iou(first.bbox, second.bbox) >= float(
            self.config.vision.stabilizer_match_iou
        )

    @staticmethod
    def _candidate_score(selection) -> tuple[float, float, float]:
        detection = selection.detection
        if detection is None:
            return (-1.0, -1.0, -1.0)
        return (
            float(selection.priority),
            float(detection.score()),
            float(detection.confidence),
        )

    @staticmethod
    def _validate_vertical_grasp_plan(
        plan,
        emit,
        place_name: str,
        *,
        require_tool_down: bool = False,
    ) -> None:
        """Verify a zero-lateral descent and optional vertical-down tool axis."""

        if require_tool_down:
            tool_z = rpy_to_matrix(plan.grasp_pose.rpy)[:, 2]
            vertical_error = float(
                np.linalg.norm(tool_z - np.array((0.0, 0.0, -1.0)))
            )
            if vertical_error > 1e-7:
                raise RuntimeError(
                    "vertical grasp tool Z axis is not vertical down; "
                    f"tool_z=({tool_z[0]:.9f}, {tool_z[1]:.9f}, "
                    f"{tool_z[2]:.9f})"
                )
        high_poses = (
            ("pregrasp", plan.pregrasp_pose),
            ("lift", plan.lift_pose),
            ("transfer", plan.transfer_pose),
            ("retreat", plan.retreat_pose),
        )
        for waypoint_name, high_pose in high_poses:
            if high_pose is None:
                raise RuntimeError(
                    f"vertical grasp descent is missing {waypoint_name!r}"
                )
            if not np.allclose(
                high_pose.xyz[:2], plan.grasp_pose.xyz[:2], atol=1e-9
            ) or not np.allclose(
                high_pose.rpy, plan.grasp_pose.rpy, atol=1e-9
            ):
                raise RuntimeError(
                    f"vertical grasp descent {waypoint_name!r} is not "
                    "aligned with the final grasp column"
                )
        descent_m = (
            float(plan.pregrasp_pose.xyz[2])
            - float(plan.grasp_pose.xyz[2])
        )
        emit(
            TaskPhase.PLAN,
            f"{place_name}: vertical grasp column locked; "
            f"xy=({plan.grasp_pose.xyz[0]:.4f}, "
            f"{plan.grasp_pose.xyz[1]:.4f}), "
            f"straight_descent={descent_m:.4f} m, "
            "lateral=0.0000 m, rotation=0.0000 rad",
        )

    def _validate_task_cd_grasp_safety_plan(
        self,
        plan,
        emit,
        place_name: str,
        expected_grasp_rpy: Sequence[float] | None = None,
    ):
        """Apply the C/D flange-to-jaw-tip envelope before route precheck."""

        if self.config.task.mode not in {TaskMode.C, TaskMode.D}:
            return None
        if not self.config.motion.grasp_safety.enabled:
            return None
        report = self.planner.validate_grasp_safety_plan(
            plan,
            expected_grasp_rpy=expected_grasp_rpy,
        )
        tip = report.grasp_tip_xyz_m
        flange = report.grasp_flange_xyz_m
        release_tip = report.release_tip_xyz_m
        release_flange = report.release_flange_xyz_m
        if (
            tip is None
            or flange is None
            or release_tip is None
            or release_flange is None
        ):
            raise RuntimeError("C/D grasp safety report is missing grasp geometry")
        emit(
            TaskPhase.PLAN,
            f"{place_name}: C/D grasp safety passed; "
            f"place_pose_frame={report.place_pose_frame}; "
            f"grasp_tip_xyz=({tip[0]:.6f}, {tip[1]:.6f}, {tip[2]:.6f}) "
            f"grasp_flange_xyz=({flange[0]:.6f}, {flange[1]:.6f}, "
            f"{flange[2]:.6f}) m; "
            f"release_tip_xyz=({release_tip[0]:.6f}, "
            f"{release_tip[1]:.6f}, {release_tip[2]:.6f}) m; "
            f"release_flange_xyz=({release_flange[0]:.6f}, "
            f"{release_flange[1]:.6f}, {release_flange[2]:.6f}) m; "
            f"route_flange_z=[{report.minimum_route_flange_z_m:.6f}, "
            f"{report.maximum_route_flange_z_m:.6f}] m; "
            f"route_tip_z=[{report.minimum_route_tip_z_m:.6f}, "
            f"{report.maximum_route_tip_z_m:.6f}] m; "
            f"approach_height={report.approach_height_m:.6f} m; "
            f"approach_xy_error={report.approach_xy_error_m:.6f} m; "
            f"max_tool_tilt={report.maximum_tool_tilt_deg:.6f} deg; "
            "max_grasp_tool_tilt="
            f"{report.maximum_grasp_tool_tilt_deg:.6f} deg; "
            "max_place_tool_tilt="
            f"{report.maximum_place_tool_tilt_deg:.6f} deg; "
            f"max_horizontal_reach={report.maximum_horizontal_reach_m:.6f} m",
        )
        return report

    def _validate_tool_axis_grasp_plan(
        self,
        plan,
        emit=None,
        place_name: str = "",
    ) -> None:
        """Verify a tilted visual plan approaches along the final tool Z axis.

        For an in-place Task B pick the geometry is intentionally strict::

            pregrasp = grasp - approach_height * R[:, 2]
            release  = grasp
            lift = transfer = retreat = pregrasp

        This validator never relaxes IK, joint, singularity, collision or route
        prechecks.  It only rejects a malformed plan before those checks run.
        """

        required = (
            ("pregrasp", plan.pregrasp_pose),
            ("grasp", plan.grasp_pose),
            ("lift", plan.lift_pose),
            ("transfer", plan.transfer_pose),
            ("release", plan.release_pose),
            ("retreat", plan.retreat_pose),
        )
        for waypoint_name, pose in required:
            if pose is None:
                raise RuntimeError(
                    f"tool-axis grasp is missing {waypoint_name!r}"
                )
            values = np.asarray((*pose.xyz, *pose.rpy), dtype=float)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"tool-axis waypoint {waypoint_name!r} must contain finite "
                    "XYZ and RPY values"
                )

        grasp_pose = plan.grasp_pose
        pregrasp_pose = plan.pregrasp_pose
        rotation = rpy_to_matrix(grasp_pose.rpy)
        if (
            not np.all(np.isfinite(rotation))
            or not np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=1e-8,
                rtol=0.0,
            )
            or float(np.linalg.det(rotation)) <= 0.0
        ):
            raise RuntimeError(
                "tool-axis grasp rotation is not a proper orthonormal matrix"
            )
        tool_z = rotation[:, 2]
        if float(tool_z[2]) >= -1e-6:
            raise RuntimeError(
                "tool-axis grasp must point downward; "
                f"tool_z=({tool_z[0]:.9f}, {tool_z[1]:.9f}, "
                f"{tool_z[2]:.9f})"
            )

        expected_height = self.config.adaptive_inverse.candidates.approach_height_m
        if expected_height is None:
            expected_height = (
                self.config.task_b.vertical_grasp_approach_height_m
            )
        if (
            expected_height is None
            or not np.isfinite(float(expected_height))
            or float(expected_height) <= 0.0
        ):
            raise RuntimeError(
                "tool-axis grasp needs a positive finite configured approach "
                "height"
            )
        expected_height = float(expected_height)
        approach_vector = np.asarray(pregrasp_pose.xyz) - np.asarray(
            grasp_pose.xyz
        )
        expected_vector = -expected_height * tool_z
        if not np.allclose(
            approach_vector,
            expected_vector,
            atol=1e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                "tool-axis pregrasp is not the configured approach distance "
                "opposite the final tool Z axis"
            )
        if float(np.linalg.norm(approach_vector)) <= 0.0:
            raise RuntimeError("tool-axis pregrasp approach distance is zero")

        for waypoint_name, pose in (
            ("pregrasp", pregrasp_pose),
            ("lift", plan.lift_pose),
            ("transfer", plan.transfer_pose),
            ("retreat", plan.retreat_pose),
        ):
            if not np.allclose(
                pose.rpy, grasp_pose.rpy, atol=1e-9, rtol=0.0
            ):
                raise RuntimeError(
                    f"tool-axis waypoint {waypoint_name!r} does not retain the "
                    "final grasp orientation"
                )
            if not np.allclose(
                pose.xyz, pregrasp_pose.xyz, atol=1e-9, rtol=0.0
            ):
                raise RuntimeError(
                    f"tool-axis waypoint {waypoint_name!r} did not return to "
                    "the tool-axis high pose"
                )

        if not np.allclose(
            plan.release_pose.xyz,
            grasp_pose.xyz,
            atol=1e-9,
            rtol=0.0,
        ) or not np.allclose(
            plan.release_pose.rpy,
            grasp_pose.rpy,
            atol=1e-9,
            rtol=0.0,
        ):
            raise RuntimeError(
                "tool-axis release pose is not identical to the final grasp pose"
            )

        if emit is not None:
            lateral_m = float(np.linalg.norm(approach_vector[:2]))
            vertical_m = float(approach_vector[2])
            emit(
                TaskPhase.PLAN,
                f"{place_name}: tool-axis approach validated; "
                f"distance={expected_height:.4f} m, "
                f"lateral={lateral_m:.4f} m, "
                f"vertical={vertical_m:.4f} m, "
                f"tool_z=({tool_z[0]:.4f}, {tool_z[1]:.4f}, "
                f"{tool_z[2]:.4f})",
            )

    def _adaptive_route_metrics(self, route) -> tuple[float, float]:
        """Return minimum Jacobian sigma and total joint travel for a route."""

        sigmas = [
            float(segment.min_jacobian_singular_value)
            for segment in route.segments
            if np.isfinite(float(segment.min_jacobian_singular_value))
        ]
        minimum_sigma = min(sigmas) if sigmas else 0.0
        total_travel_deg = 0.0
        for segment in route.segments:
            metrics = getattr(segment, "joint_metrics", None)
            if metrics is not None:
                total_travel_deg += float(
                    np.sum(np.abs(np.asarray(metrics.travel_deg, dtype=float)))
                )
                continue
            joint_waypoints = tuple(getattr(segment, "joint_waypoints", ()) or ())
            if len(joint_waypoints) >= 2:
                start = np.asarray(joint_waypoints[0], dtype=float)
                end = np.asarray(joint_waypoints[-1], dtype=float)
                total_travel_deg += float(
                    np.sum(np.abs(np.degrees(end - start)))
                )
        return minimum_sigma, total_travel_deg

    def _validate_adaptive_candidate_plan(self, candidate) -> None:
        """Reject a malformed or unsafe candidate before route prechecking."""

        plan = getattr(candidate, "plan", None)
        if plan is None:
            raise RuntimeError("adaptive candidate is missing its motion plan")
        waypoints = plan.ordered_waypoints()
        if not waypoints:
            raise RuntimeError("adaptive candidate has no waypoints")
        for waypoint_name, pose in waypoints:
            if pose is None:
                raise RuntimeError(
                    f"adaptive candidate is missing waypoint {waypoint_name!r}"
                )
            pose_values = np.asarray((*pose.xyz, *pose.rpy), dtype=float)
            if pose_values.shape != (6,) or not np.all(np.isfinite(pose_values)):
                raise RuntimeError(
                    f"adaptive waypoint {waypoint_name!r} must contain finite "
                    "XYZ and RPY values"
                )

        if self._tilted_visual_grasp:
            self._validate_tool_axis_grasp_plan(plan)
            rotation = rpy_to_matrix(plan.grasp_pose.rpy)
            try:
                clearance_config = getattr(
                    candidate,
                    "dynamic_box_clearance",
                    self.config.adaptive_inverse.box_clearance,
                )
                clearance = check_box_clearance(
                    clearance_config,
                    flange_xy_m=plan.grasp_pose.xyz[:2],
                    rotation=rotation,
                )
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"box-clearance check failed closed: {exc}"
                ) from exc
            if not clearance.allowed:
                raise RuntimeError(clearance.reason)
            return

        grasp_pose = plan.grasp_pose
        pregrasp_pose = plan.pregrasp_pose
        if not np.allclose(
            pregrasp_pose.xyz[:2], grasp_pose.xyz[:2], atol=1e-9, rtol=0.0
        ) or not np.allclose(
            pregrasp_pose.rpy, grasp_pose.rpy, atol=1e-9, rtol=0.0
        ):
            raise RuntimeError(
                "adaptive pregrasp is not aligned with the final grasp column"
            )

        if plan.release_pose is not None and (
            not np.allclose(
                plan.release_pose.xyz[:2],
                grasp_pose.xyz[:2],
                atol=1e-9,
                rtol=0.0,
            )
            or not np.allclose(
                plan.release_pose.rpy,
                grasp_pose.rpy,
                atol=1e-9,
                rtol=0.0,
            )
        ):
            raise RuntimeError(
                "adaptive release pose is not aligned with the final grasp "
                "column"
            )

        rotation = rpy_to_matrix(grasp_pose.rpy)
        if not np.all(np.isfinite(rotation)):
            raise RuntimeError("adaptive grasp rotation contains non-finite values")
        if self._arbitrary_placement or self._box_relative_five_point:
            tool_z = rotation[:, 2]
            vertical_error = float(
                np.linalg.norm(tool_z - np.array((0.0, 0.0, -1.0)))
            )
            if vertical_error > 1e-7:
                mode = (
                    "box-relative"
                    if self._box_relative_five_point
                    else "arbitrary-placement"
                )
                raise RuntimeError(
                    f"{mode} grasp tool Z axis is not vertical down; "
                    f"tool_z=({tool_z[0]:.9f}, {tool_z[1]:.9f}, "
                    f"{tool_z[2]:.9f})"
                )

        try:
            clearance_config = getattr(
                candidate,
                "dynamic_box_clearance",
                self.config.adaptive_inverse.box_clearance,
            )
            clearance = check_box_clearance(
                clearance_config,
                flange_xy_m=grasp_pose.xyz[:2],
                rotation=rotation,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"box-clearance check failed closed: {exc}"
            ) from exc
        if not clearance.allowed:
            raise RuntimeError(clearance.reason)

    def _precheck_adaptive_candidates(
        self,
        candidates,
        emit,
        *,
        place_name: str,
    ):
        """Precheck safety-ordered candidates and select one complete route."""

        failures: list[str] = []
        successful: list[tuple[object, object, float, float]] = []
        selection = self.config.adaptive_inverse.selection
        minimum_required_sigma = (
            float(self.config.motion.cartesian_min_jacobian_singular_value)
            + float(selection.min_sigma_margin)
        )

        for index, candidate in enumerate(candidates, start=1):
            try:
                self._validate_adaptive_candidate_plan(candidate)
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                message = (
                    f"candidate {index}: defensive plan/box-clearance check "
                    f"rejected: {exc}"
                )
                failures.append(message)
                emit(TaskPhase.PLAN, message)
                continue
            plan = candidate.plan
            try:
                # Adaptive planners derive the flange target from the visual
                # jaw-tip point themselves (``use_visual_grasp_z``).  Do not
                # apply the fixed-plan tip conversion a second time here, but
                # do apply the same fail-closed C/D geometry envelope to every
                # candidate before any IK/FK or motion precheck.
                self._validate_task_cd_grasp_safety_plan(
                    plan,
                    emit,
                    f"{place_name}:candidate {index}",
                )
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                message = (
                    f"candidate {index}: C/D grasp safety check rejected: {exc}"
                )
                failures.append(message)
                emit(TaskPhase.PLAN, message)
                continue
            waypoints = plan.ordered_waypoints()
            identity_parts: list[str] = []
            object_center_base_xyz = getattr(
                candidate, "object_center_base_xyz", None
            )
            box_point_name = getattr(candidate, "box_point_name", None)
            if object_center_base_xyz is not None:
                identity_parts.append(
                    "object_center_base="
                    f"{tuple(round(float(value), 6) for value in object_center_base_xyz)}"
                )
                major_axis_base_xyz = getattr(
                    candidate, "major_axis_base", None
                )
                if major_axis_base_xyz is not None:
                    identity_parts.append(
                        "major_axis_base="
                        f"{tuple(round(float(value), 6) for value in major_axis_base_xyz)}"
                    )
                major_axis_horizontal_base = getattr(
                    candidate, "major_axis_horizontal_base", None
                )
                if major_axis_horizontal_base is not None:
                    identity_parts.append(
                        "major_axis_xy="
                        f"{tuple(round(float(value), 6) for value in major_axis_horizontal_base)}"
                    )
                jaw_tip_object_offset_m = getattr(
                    candidate, "jaw_tip_object_offset_m", None
                )
                if jaw_tip_object_offset_m is not None:
                    identity_parts.append(
                        "jaw_tip_object_offset_m="
                        f"{tuple(round(float(value), 6) for value in jaw_tip_object_offset_m)}"
                    )
                commanded_jaw_tip_base_xyz = getattr(
                    candidate, "commanded_jaw_tip_base_xyz", None
                )
                if commanded_jaw_tip_base_xyz is not None:
                    identity_parts.append(
                        "commanded_jaw_tip_base="
                        f"{tuple(round(float(value), 6) for value in commanded_jaw_tip_base_xyz)}"
                    )
            elif box_point_name is not None:
                identity_parts.append(f"box_point={box_point_name}")
            if hasattr(candidate, "yaw_offset_deg"):
                identity_parts.append(
                    f"yaw_offset={float(candidate.yaw_offset_deg):.3f} deg"
                )
            if hasattr(candidate, "pitch_offset_deg"):
                identity_parts.append(
                    f"pitch_offset={float(candidate.pitch_offset_deg):.3f} deg"
                )
            identity = " ".join(identity_parts)
            emit(
                TaskPhase.PLAN,
                f"candidate {index}: {identity} "
                f"grasp_xyz={tuple(round(float(value), 6) for value in plan.grasp_pose.xyz)} "
                f"grasp_rpy={tuple(round(float(value), 6) for value in plan.grasp_pose.rpy)} "
                f"pregrasp_xyz={tuple(round(float(value), 6) for value in plan.pregrasp_pose.xyz)} "
                f"pregrasp_rpy={tuple(round(float(value), 6) for value in plan.pregrasp_pose.rpy)}",
            )

            # First solve just the high approach endpoint.  A near-singular or
            # unreachable wrist posture can never yield a safe continuous
            # route, while discovering that fact through repeated Cartesian
            # refinement costs tens of seconds.  Backends without kinematics
            # return None and fall through to the unchanged full-route check.
            condition_check = getattr(
                self.motion,
                "precheck_pose_condition",
                None,
            )
            if callable(condition_check) and waypoints:
                try:
                    endpoint_sigma = condition_check(
                        waypoints[0][1],
                        label=f"adaptive candidate {index}",
                    )
                except (ValueError, RuntimeError) as exc:
                    message = (
                        f"candidate {index}: pregrasp endpoint condition check "
                        f"rejected: {exc}"
                    )
                    failures.append(message)
                    emit(TaskPhase.PLAN, message)
                    continue
                if endpoint_sigma is not None:
                    endpoint_sigma = float(endpoint_sigma)
                    if not np.isfinite(endpoint_sigma):
                        message = (
                            f"candidate {index}: pregrasp endpoint returned a "
                            "non-finite Jacobian sigma"
                        )
                        failures.append(message)
                        emit(TaskPhase.PLAN, message)
                        continue
                    if endpoint_sigma < minimum_required_sigma:
                        message = (
                            f"candidate {index}: pregrasp endpoint minimum "
                            f"Jacobian sigma {endpoint_sigma:.6f} is below "
                            f"required {minimum_required_sigma:.6f}"
                        )
                        failures.append(message)
                        emit(TaskPhase.PLAN, message)
                        continue
                    emit(
                        TaskPhase.PLAN,
                        f"adaptive candidate {index}/{len(candidates)} passed "
                        f"pregrasp endpoint condition: "
                        f"min_sigma={endpoint_sigma:.6f}",
                    )

            try:
                route = self.motion.precheck_linear_targets(
                    tuple(pose for _name, pose in waypoints),
                    step_m=float(self.config.motion.cartesian_step_m),
                )
            except (ValueError, RuntimeError) as exc:
                message = self._task_b_route_rejection_message(
                    place_name,
                    plan,
                    waypoints,
                    exc,
                )
                failures.append(f"candidate {index}: {message}")
                emit(
                    TaskPhase.PLAN,
                    f"adaptive candidate {index}/{len(candidates)} rejected: {exc}",
                )
                continue

            if len(route.segments) != len(waypoints):
                message = (
                    f"candidate {index}: full-route precheck returned "
                    f"{len(route.segments)} segment(s), expected {len(waypoints)}"
                )
                failures.append(message)
                emit(TaskPhase.PLAN, message)
                continue

            minimum_sigma, joint_travel_deg = self._adaptive_route_metrics(route)
            # The live backend already enforces the configured singularity
            # threshold.  The adaptive selection additionally requires the
            # operator-set safety margin.  Mock/dry-run routes intentionally do
            # not expose physical Jacobian values and therefore cannot pretend
            # to pass that live-only metric.
            if not self.motion.dry_run and minimum_sigma < minimum_required_sigma:
                message = (
                    f"candidate {index}: minimum Jacobian sigma "
                    f"{minimum_sigma:.6f} is below required "
                    f"{minimum_required_sigma:.6f}"
                )
                failures.append(message)
                emit(TaskPhase.PLAN, message)
                continue

            successful.append(
                (candidate, route, minimum_sigma, joint_travel_deg)
            )
            box_point_name = getattr(candidate, "box_point_name", None)
            if box_point_name is not None:
                candidate_description = (
                    f"box point {box_point_name!r}, "
                    f"box yaw={np.degrees(candidate.box_pose.yaw_rad):.2f} deg"
                )
            else:
                candidate_description = (
                    f"yaw_offset={candidate.yaw_offset_deg:.2f} deg, "
                    f"pitch_offset={candidate.pitch_offset_deg:.2f} deg, "
                    "orientation_change="
                    f"{np.degrees(candidate.orientation_change_rad):.2f} deg"
                )
            emit(
                TaskPhase.PLAN,
                f"adaptive candidate {index}/{len(candidates)} passed full-route "
                f"precheck: {candidate_description}, "
                f"min_sigma={minimum_sigma:.6f}, "
                f"joint_travel={joint_travel_deg:.2f} deg",
            )
            if bool(selection.stop_after_first_valid):
                emit(
                    TaskPhase.PLAN,
                    f"adaptive winner: {candidate_description}, "
                    "selection=first_safe_candidate, "
                    f"min_sigma={minimum_sigma:.6f}, "
                    f"joint_travel={joint_travel_deg:.2f} deg",
                )
                return candidate.plan, route, failures

        if not successful:
            return None, None, failures

        live_sigmas = [
            float(item[2]) for item in successful if not self.motion.dry_run
        ]
        travels = [float(item[3]) for item in successful]
        orientations = [
            float(item[0].orientation_change_rad) for item in successful
        ]
        sigma_floor = (
            minimum_required_sigma
            if live_sigmas
            else 0.0
        )
        sigma_span = max(
            max(live_sigmas, default=sigma_floor) - sigma_floor,
            1e-9,
        )
        travel_min = min(travels)
        travel_span = max(max(travels) - travel_min, 1e-9)
        orientation_min = min(orientations)
        orientation_span = max(max(orientations) - orientation_min, 1e-9)

        ranked: list[tuple[float, int, object, object, float, float]] = []
        for index, (candidate, route, minimum_sigma, travel_deg) in enumerate(
            successful,
            start=1,
        ):
            sigma_quality = (
                0.0
                if self.motion.dry_run
                else max(0.0, (minimum_sigma - sigma_floor) / sigma_span)
            )
            travel_quality = 1.0 - (
                (travel_deg - travel_min) / travel_span
            )
            orientation_quality = 1.0 - (
                (
                    float(candidate.orientation_change_rad)
                    - orientation_min
                )
                / orientation_span
            )
            score = (
                float(selection.sigma_weight) * sigma_quality
                + float(selection.joint_travel_weight) * travel_quality
                + float(selection.orientation_weight) * orientation_quality
            )
            ranked.append(
                (
                    score,
                    -index,
                    candidate,
                    route,
                    minimum_sigma,
                    travel_deg,
                )
            )

        ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
        (
            score,
            _negative_index,
            candidate,
            route,
            minimum_sigma,
            travel_deg,
        ) = ranked[0]
        emit(
            TaskPhase.PLAN,
            f"adaptive winner: yaw_offset="
            f"{candidate.yaw_offset_deg:.2f} deg, "
            f"pitch_offset={candidate.pitch_offset_deg:.2f} deg, "
            f"selection=ranked, score={score:.4f}, "
            f"min_sigma={minimum_sigma:.6f}, joint_travel={travel_deg:.2f} deg",
        )
        return candidate.plan, route, failures

    def _is_usable(self, detection: Detection) -> bool:
        if detection.confidence < self.config.vision.min_confidence:
            return False
        if self.config.vision.require_depth and detection.depth_m is None:
            return False
        if detection.base_xyz is None and detection.camera_xyz is None:
            return False
        short_edge = self.config.task_cd.short_edge_grasp
        if short_edge is not None and short_edge.enabled:
            if detection.attributes.get("held") is True:
                return False
            if (
                detection.object_center_camera_xyz is None
                or detection.major_axis_camera is None
                or detection.minor_axis_camera is None
                or detection.major_axis_length_m is None
                or detection.minor_axis_length_m is None
                or detection.pose_confidence is None
            ):
                return False
            major_axis = np.asarray(detection.major_axis_camera, dtype=float)
            minor_axis = np.asarray(detection.minor_axis_camera, dtype=float)
            if (
                major_axis.shape != (3,)
                or minor_axis.shape != (3,)
                or not np.all(np.isfinite(major_axis))
                or not np.all(np.isfinite(minor_axis))
                or float(np.linalg.norm(major_axis)) <= 1e-12
                or float(np.linalg.norm(minor_axis)) <= 1e-12
            ):
                return False
            if float(detection.pose_confidence) < float(
                short_edge.min_pose_confidence
            ):
                return False
            axis_ratio = float(detection.major_axis_length_m) / float(
                detection.minor_axis_length_m
            )
            if (
                not np.isfinite(axis_ratio)
                or axis_ratio + 1e-12
                < float(short_edge.pca.min_axis_ratio)
            ):
                return False
        adaptive = self.config.adaptive_inverse
        if adaptive.enabled:
            if detection.attributes.get("held") is True:
                return False
            if detection.object_center_camera_xyz is None:
                return False
            if detection.major_axis_camera is None:
                return False
            if detection.pose_confidence is None:
                return False
            if float(detection.pose_confidence) < float(
                adaptive.min_pose_confidence
            ):
                return False
            if not np.all(np.isfinite(detection.major_axis_camera)):
                return False
            if float(np.linalg.norm(detection.major_axis_camera)) <= 1e-12:
                return False
        return True

    def _set_vision_status(self, text: str) -> None:
        """Best-effort preview status update without coupling mocks to runtime."""

        set_status = getattr(self.vision, "set_status", None)
        if set_status is None:
            return
        try:
            set_status(text)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._log(f"[vision:live] status update failed: {exc}")

    def _recover_after_failed_grasp(
        self, plan, events: list[PipelineEvent], reason: str
    ) -> None:
        self._log(f"[recovery] {reason}; open jaws and retreat")
        self._set_vision_status(f"RECOVERY: {reason}")
        try:
            self.gripper.open()
            self.motion.move_to(
                plan.pregrasp_pose,
                name="recovery-retreat",
                speed_mps=self.config.motion.speed_mps,
            )
        except Exception as exc:
            events.append(
                PipelineEvent(TaskPhase.ABORT, f"recovery move failed: {exc}")
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for component in (self.vision, self.motion, self.gripper):
            # Grippers use close() for the jaws and close_backend() for
            # resource teardown; never close the jaws during shutdown.
            close = getattr(component, "close_backend", None)
            if close is None:
                close = getattr(component, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    self._log(f"[shutdown] {type(component).__name__}: {exc}")
