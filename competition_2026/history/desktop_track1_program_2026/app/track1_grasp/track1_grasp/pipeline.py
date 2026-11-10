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
from typing import Sequence

import numpy as np

from .config import Track1Config
from .detection_stabilizer import box_iou
from .domain import (
    Detection,
    Frame,
    GripperState,
    PickResult,
    PickStatus,
    PipelineEvent,
    RobotState,
    Selection,
    RunReport,
    TaskMode,
    TaskPhase,
)
from .kinematics import joint_scan_target, joints_within_limits
from .policy import TargetPolicy


_PHASE_FOR_WAYPOINT = {
    "pregrasp": TaskPhase.APPROACH,
    "grasp": TaskPhase.GRASP,
    "lift": TaskPhase.LIFT,
    "transfer": TaskPhase.TRANSFER,
    "release": TaskPhase.RELEASE,
    "retreat": TaskPhase.RETREAT,
}


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
        log=None,
    ) -> None:
        self.config = config
        self.vision = vision
        self.policy = policy
        self.planner = planner
        self.motion = motion
        self.gripper = gripper
        self.hold_verifier = hold_verifier
        self._log = log or (lambda message: None)
        self._closed = False
        self._task_b_pick_index = 0
        #: Actual base-frame XY locations where completed objects were released.
        #: The configured reference points are only expected windows; the
        #: observed release positions are what must be excluded in later rounds.
        self._task_b_completed_xy: list[tuple[float, float]] = []
        self._task_b_target_label: str | None = (
            config.task_b.target_class.strip() or None
        )

    def run(self, max_picks: int | None = None) -> RunReport:
        default_picks = (
            self.config.task_b.rounds
            if self.config.task.mode is TaskMode.B
            else self.config.task.max_picks
        )
        requested = int(max_picks or default_picks)
        if requested < 1:
            raise ValueError("max_picks must be >= 1")
        if self.config.task.mode is TaskMode.B:
            requested = min(
                requested,
                self.config.task_b.rounds,
                len(self.config.place_poses),
            )

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
                continue
            if result.status == PickStatus.NO_TARGET:
                break
            if self.config.task.stop_on_error:
                break
        report = RunReport(requested, attempts, successes, tuple(results))
        self._log(
            f"[run] requested={requested} attempts={attempts} "
            f"successes={successes} rate={report.success_rate:.2%}"
        )
        return report

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

    def run_once(self) -> PickResult:
        events: list[PipelineEvent] = [
            PipelineEvent(TaskPhase.IDLE, "starting pick cycle")
        ]

        def emit(phase: TaskPhase, message: str) -> None:
            events.append(PipelineEvent(phase, message))
            self._log(f"[{phase.value}] {message}")
            self._set_vision_status(f"{phase.value.upper()}: {message}")

        state = self.motion.get_state()
        if not self.motion.dry_run and not state.ready:
            emit(TaskPhase.ABORT, f"robot not ready: {state.message}")
            return PickResult(PickStatus.ABORTED, None, state.message, tuple(events))

        place_name, place_pose = self._next_place()
        expected_label = (
            self._expected_task_b_label()
            if self.config.task.mode is TaskMode.B
            else None
        )
        completed_points = (
            tuple(self._task_b_completed_xy)
            if self.config.task.mode is TaskMode.B
            else ()
        )
        try:
            if self.config.task.mode is TaskMode.B:
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
        try:
            plan = self.planner.plan(
                target,
                state,
                place_pose=None if return_to_pick else place_pose,
                place_name=place_name,
                return_to_pick=return_to_pick,
            )
        except (ValueError, RuntimeError) as exc:
            emit(TaskPhase.PLAN, f"rejected: {exc}")
            self.policy.commit_failure()
            return PickResult(PickStatus.REJECTED, target, str(exc), tuple(events))
        emit(TaskPhase.PLAN, f"planned {len(plan.ordered_waypoints())} waypoint(s)")

        if self.config.gripper.open_before_approach and self.gripper.state != GripperState.OPEN:
            emit(TaskPhase.OPEN_GRIPPER, "opening jaws before entering approach corridor")
            try:
                self.gripper.open()
            except Exception as exc:
                emit(TaskPhase.ABORT, f"gripper open failed: {exc}")
                return PickResult(PickStatus.ABORTED, target, str(exc), tuple(events))

        try:
            for name, pose in plan.ordered_waypoints():
                phase = _PHASE_FOR_WAYPOINT.get(name, TaskPhase.APPROACH)
                emit(phase, f"move to {name}: xyz={tuple(round(v, 4) for v in pose.xyz)}")
                self.motion.move_to(
                    pose,
                    name=name,
                    speed_mps=self.config.motion.speed_mps,
                )
                if name == "grasp":
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
            # Advance only after every waypoint, including release and the
            # final retreat, has completed without raising.  Record the actual
            # released XY rather than the configured reference point so a
            # returned object cannot be mistaken for the next point later.
            self._task_b_completed_xy.append(
                (float(plan.grasp_pose.xyz[0]), float(plan.grasp_pose.xyz[1]))
            )
            self._task_b_pick_index = completed
            if completed < self.config.task_b.rounds:
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
        """Confirm a target in the current view, then search wrist views if needed."""

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

        views = self._search_views()
        scan_views = views[1:]
        if not scan_views:
            return None, frame, self._fresh_state(initial_state), reason

        if self.motion.dry_run:
            self._validate_search_views(initial_state, scan_views, emit)
            return (
                None,
                frame,
                self._fresh_state(initial_state),
                f"{reason}; dry-run checked {len(scan_views)} search view(s) but "
                "did not move the arm",
            )

        if not initial_state.ready or initial_state.joints is None:
            raise RuntimeError(
                "automatic target search needs AutoReady with joint feedback: "
                f"{initial_state.message}"
            )
        if not hasattr(self.motion, "move_joints"):
            raise RuntimeError("configured motion backend cannot run joint-space search")

        q_initial = np.asarray(initial_state.joints, dtype=float)
        state = initial_state
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
                f"MOVING: search view {index}/{len(scan_views)} "
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

        if self.config.search.return_to_initial_on_miss:
            self._set_vision_status("MOVING: returning to initial search pose")
            self.motion.move_joints(q_initial, name="task_search_return")
        return (
            None,
            frame,
            self._fresh_state(state),
            f"target not confirmed in the current view or {len(scan_views)} search views",
        )

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
            frame = self.vision.capture()
            latest_frame = frame
            usable = [
                detection
                for detection in frame.detections
                if self._is_usable(detection)
            ]
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
                selection = self.policy.select(usable)
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
                    f"no confirmed target",
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

        For rounds after the first, ``completed_points`` contains every point
        whose pick-and-return cycle already finished.  A candidate must be
        closer to the current point than to all of those points by at least
        ``point_assignment_margin_m``.  This is stronger than a simple
        tolerance check: even when two tolerance circles overlap, the object
        left at an old point cannot start the next round.
        """

        expected_xy = np.asarray(expected_point[:2], dtype=float)
        if expected_xy.shape != (2,) or not np.all(np.isfinite(expected_xy)):
            return [], f"invalid Task B reference point {expected_point!r}"

        excluded: list[tuple[int, np.ndarray]] = []
        for index, point in enumerate(completed_points, start=1):
            xy = np.asarray(point[:2], dtype=float)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                return [], f"invalid completed Task B reference point {index}: {point!r}"
            excluded.append((index, xy))

        tolerance_m = float(self.config.task_b.pick_pose_tolerance_m)
        assignment_margin_m = float(self.config.task_b.point_assignment_margin_m)
        accepted: list[tuple[Detection, float]] = []
        ambiguous: list[str] = []
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
                f"assignment_margin={assignment_margin_m:.3f} m, "
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

    def _is_usable(self, detection: Detection) -> bool:
        if detection.confidence < self.config.vision.min_confidence:
            return False
        if self.config.vision.require_depth and detection.depth_m is None:
            return False
        if detection.base_xyz is None and detection.camera_xyz is None:
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
