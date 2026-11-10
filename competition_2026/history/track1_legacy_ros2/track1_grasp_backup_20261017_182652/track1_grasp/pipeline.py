"""The hardware-independent Track 1 pick state machine.

Tasks B/C/D/E all use the same observable sequence::

    scan -> detect -> select -> plan -> open_gripper
         -> approach -> grasp -> close -> verify_hold
         -> lift -> transfer -> release -> retreat -> done

Task B additionally walks a list of five place slots (the bin's four
corners plus its centre), one completed pick per slot.
"""

from __future__ import annotations

from .config import Track1Config
from .domain import (
    Detection,
    GripperState,
    PickResult,
    PickStatus,
    PipelineEvent,
    RunReport,
    TaskMode,
    TaskPhase,
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
        self._place_index = 0

    def run(self, max_picks: int | None = None) -> RunReport:
        default_picks = (
            self.config.task_b.rounds
            if self.config.task.mode is TaskMode.B
            else self.config.task.max_picks
        )
        requested = int(max_picks or default_picks)
        if requested < 1:
            raise ValueError("max_picks must be >= 1")

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
        """Return the place slot for this cycle (task B walks a list)."""

        poses = self.config.place_poses if self.config.task.mode is TaskMode.B else ()
        if not poses:
            return "", None
        names = self.config.task_b.place_names
        pose = poses[self._place_index % len(poses)]
        if names:
            name = names[self._place_index % len(names)]
        else:
            default = ("corner_1", "corner_2", "corner_3", "corner_4", "centre")
            name = default[self._place_index % len(default)]
        return name, pose

    def run_once(self) -> PickResult:
        events: list[PipelineEvent] = [
            PipelineEvent(TaskPhase.IDLE, "starting pick cycle")
        ]

        def emit(phase: TaskPhase, message: str) -> None:
            events.append(PipelineEvent(phase, message))
            self._log(f"[{phase.value}] {message}")

        state = self.motion.get_state()
        if not self.motion.dry_run and not state.ready:
            emit(TaskPhase.ABORT, f"robot not ready: {state.message}")
            return PickResult(PickStatus.ABORTED, None, state.message, tuple(events))

        emit(TaskPhase.SCAN, "capturing a frame")
        try:
            frame = self.vision.capture()
        except StopIteration as exc:
            emit(TaskPhase.SCAN, str(exc))
            return PickResult(PickStatus.NO_TARGET, None, str(exc), tuple(events))

        emit(TaskPhase.DETECT, f"frame {frame.frame_id!r}: {len(frame.detections)} raw detection(s)")
        usable = [d for d in frame.detections if self._is_usable(d)]
        selection = self.policy.select(usable)
        if not selection.accepted or selection.detection is None:
            emit(TaskPhase.SELECT, selection.reason)
            return PickResult(
                PickStatus.NO_TARGET,
                None,
                selection.reason,
                tuple(events),
            )

        target = selection.detection
        emit(
            TaskPhase.SELECT,
            f"selected {target.label!r} conf={target.confidence:.2f} "
            f"graspability={target.graspability:.2f}: {selection.reason}",
        )

        place_name, place_pose = self._next_place()
        try:
            plan = self.planner.plan(
                target, state, place_pose=place_pose, place_name=place_name
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

        self.policy.commit_success(selection)
        self._place_index += 1
        if plan.release_pose is None:
            message = "object lifted and held; no place pose configured"
        else:
            message = "object picked, transferred and released"
        emit(TaskPhase.DONE, message)
        return PickResult(
            PickStatus.SUCCESS, target, message, tuple(events), place_name=plan.place_name
        )

    def _is_usable(self, detection: Detection) -> bool:
        if detection.confidence < self.config.vision.min_confidence:
            return False
        if self.config.vision.require_depth and detection.depth_m is None:
            return False
        if detection.base_xyz is None and detection.camera_xyz is None:
            return False
        return True

    def _recover_after_failed_grasp(
        self, plan, events: list[PipelineEvent], reason: str
    ) -> None:
        self._log(f"[recovery] {reason}; open jaws and retreat")
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
