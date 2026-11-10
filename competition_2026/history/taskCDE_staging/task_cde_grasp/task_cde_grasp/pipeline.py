"""The hardware-independent CDE pick state machine."""

from __future__ import annotations

from .config import CDEConfig
from .domain import (
    Detection,
    GripperState,
    PickPhase,
    PickResult,
    PickStatus,
    PipelineEvent,
    RunReport,
)
from .policy import TargetPolicy


_PHASE_FOR_WAYPOINT = {
    "pregrasp": PickPhase.APPROACH,
    "grasp": PickPhase.GRASP,
    "lift": PickPhase.LIFT,
    "transfer": PickPhase.TRANSFER,
    "release": PickPhase.RELEASE,
    "retreat": PickPhase.RETREAT,
}


class PickPipeline:
    """One object at a time: scan -> select -> open -> move -> close -> lift.

    The jaws are opened before entering the narrow approach corridor and are
    closed only after the tool reaches the planned grasp pose.  A failed hold
    check triggers an open-and-retreat recovery instead of lifting blindly.
    """

    def __init__(
        self,
        config: CDEConfig,
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

    def run(self, max_picks: int | None = None) -> RunReport:
        requested = int(max_picks or self.config.task.max_picks)
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
                    (PipelineEvent(PickPhase.SCAN, "no more frames"),),
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

    def run_once(self) -> PickResult:
        events: list[PipelineEvent] = [
            PipelineEvent(PickPhase.IDLE, "starting pick cycle")
        ]

        def emit(phase: PickPhase, message: str) -> None:
            events.append(PipelineEvent(phase, message))
            self._log(f"[{phase.value}] {message}")

        state = self.motion.get_state()
        if not self.motion.dry_run and not state.ready:
            emit(PickPhase.ABORT, f"robot not ready: {state.message}")
            return PickResult(PickStatus.ABORTED, None, state.message, tuple(events))

        emit(PickPhase.SCAN, "capturing a frame")
        try:
            frame = self.vision.capture()
        except StopIteration as exc:
            emit(PickPhase.SCAN, str(exc))
            return PickResult(PickStatus.NO_TARGET, None, str(exc), tuple(events))

        emit(PickPhase.DETECT, f"frame {frame.frame_id!r}: {len(frame.detections)} raw detection(s)")
        usable = [d for d in frame.detections if self._is_usable(d)]
        selection = self.policy.select(usable)
        if not selection.accepted or selection.detection is None:
            emit(PickPhase.SELECT, selection.reason)
            return PickResult(
                PickStatus.NO_TARGET,
                None,
                selection.reason,
                tuple(events),
            )

        target = selection.detection
        emit(
            PickPhase.SELECT,
            f"selected {target.label!r} conf={target.confidence:.2f} "
            f"graspability={target.graspability:.2f}: {selection.reason}",
        )

        try:
            plan = self.planner.plan(target, state)
        except (ValueError, RuntimeError) as exc:
            emit(PickPhase.PLAN, f"rejected: {exc}")
            self.policy.commit_failure()
            return PickResult(PickStatus.REJECTED, target, str(exc), tuple(events))
        emit(PickPhase.PLAN, f"planned {len(plan.ordered_waypoints())} waypoint(s)")

        if self.config.gripper.open_before_approach and self.gripper.state != GripperState.OPEN:
            emit(PickPhase.OPEN_GRIPPER, "opening jaws before entering approach corridor")
            try:
                self.gripper.open()
            except Exception as exc:
                emit(PickPhase.ABORT, f"gripper open failed: {exc}")
                return PickResult(PickStatus.ABORTED, target, str(exc), tuple(events))

        try:
            for name, pose in plan.ordered_waypoints():
                phase = _PHASE_FOR_WAYPOINT.get(name, PickPhase.APPROACH)
                emit(phase, f"move to {name}: xyz={tuple(round(v, 4) for v in pose.xyz)}")
                self.motion.move_to(
                    pose,
                    name=name,
                    speed_mps=self.config.motion.speed_mps,
                )
                if name == "grasp":
                    emit(PickPhase.GRASP, "reached grasp pose; closing jaws")
                    self.gripper.close()
                    emit(PickPhase.VERIFY_HOLD, "checking whether the object is held")
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
                    emit(PickPhase.VERIFY_HOLD, reason)
                elif name == "release":
                    emit(PickPhase.RELEASE, "opening jaws to release the object")
                    self.gripper.open()
        except Exception as exc:
            emit(PickPhase.ABORT, f"motion failed: {exc}")
            self.policy.commit_failure()
            return PickResult(PickStatus.ABORTED, target, str(exc), tuple(events))

        self.policy.commit_success(selection)
        if plan.release_pose is None:
            message = "object lifted and held; no place pose configured"
        else:
            message = "object picked, transferred and released"
        emit(PickPhase.DONE, message)
        return PickResult(PickStatus.SUCCESS, target, message, tuple(events))

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
                PipelineEvent(PickPhase.ABORT, f"recovery move failed: {exc}")
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
