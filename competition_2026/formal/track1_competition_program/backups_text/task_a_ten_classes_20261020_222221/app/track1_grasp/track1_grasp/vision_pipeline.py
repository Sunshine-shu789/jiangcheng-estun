"""Task A pipeline: capture -> detect -> annotate -> score.

The pipeline keeps its historical vision-only mode and adds an optional,
strictly pre-checked Cartesian patrol used by the ten-position competition
demonstration.  The live vision source is never closed between moves, so YOLO
and the preview continue updating while the arm is moving.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import time
from typing import Sequence

from .annotator import Annotator, load_ground_truth, merge_reports, score_frame
from .config import Track1Config
from .domain import Frame, IouMatch, Pose, RobotState, TaskPhase, VisionReport
from .motion import LinearRoutePlan


class VisionDemoPipeline:
    """Run the task A detection/annotation/IoU loop over a vision source."""

    def __init__(
        self,
        config: Track1Config,
        *,
        vision,
        motion=None,
        annotator: Annotator | None = None,
        log=None,
        clock=None,
        sleep=None,
    ) -> None:
        self.config = config
        self.vision = vision
        self._log = log or (lambda message: None)
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        if motion is not None:
            self.motion = motion
        self.annotator = annotator or Annotator(config.task_a, log=log)
        self._ground_truth = (
            load_ground_truth(config.task_a.ground_truth_file)
            if config.task_a.ground_truth_file
            else {}
        )
        if self._ground_truth:
            self._log(
                f"[scan] loaded {len(self._ground_truth)} ground-truth entr(ies) "
                f"from {config.task_a.ground_truth_file}"
            )
        self._closed = False

    def run(self) -> VisionReport:
        if self.config.task_a.patrol.enabled:
            return self._run_patrol()
        return self._run_scans()

    def _run_scans(self) -> VisionReport:
        threshold = float(self.config.task_a.iou_threshold)
        reports: list[VisionReport] = []
        index = 0
        self._log(
            f"[run] task A detection demo: iou_threshold={threshold:.2f} "
            f"max_frames={self.config.task_a.max_frames or 'all'}"
        )
        self._set_vision_status("TASK A: scanning")
        while True:
            if self.config.task_a.max_frames and index >= self.config.task_a.max_frames:
                break
            try:
                frame, matches, report = self.run_once(index)
            except StopIteration as exc:
                self._log(f"[scan] {exc}")
                self._set_vision_status(f"TASK A DONE: {exc}")
                break
            reports.append(report)
            self._record_observation(frame, matches, index)
            index += 1
            if self.config.task.sample_period_s:
                time.sleep(0.0)  # placeholder: real sources pace themselves

        return self._finish_report(reports, threshold)

    def _run_patrol(self) -> VisionReport:
        threshold = float(self.config.task_a.iou_threshold)
        motion = getattr(self, "motion", None)
        if motion is None:
            raise RuntimeError(
                "task A patrol is enabled but no motion backend was built"
            )

        patrol = self.config.task_a.patrol
        self._set_vision_status("TASK A PATROL: waiting for RobotInfo")
        wait_for_state = getattr(motion, "wait_for_state", None)
        if callable(wait_for_state):
            state = wait_for_state(10.0)
        else:
            state = motion.get_state()
        if state.pose is None:
            raise RuntimeError(
                "task A patrol cannot start: no fresh /RobotInfo with "
                "end_positions after 10 s. Check that the Codroid UDP bridge "
                "is running, the controller is AutoReady, and 设置 -> ROS "
                "targets this PC on UDP port 9006."
            )

        route = self._precheck_patrol_route(state)
        distances = route.distances_m
        if len(route.targets) != patrol.position_count:
            raise RuntimeError(
                "patrol pre-check returned an unexpected target count: "
                f"{len(route.targets)} != {patrol.position_count}"
            )
        too_short = [
            (index, distance)
            for index, distance in enumerate(distances, start=1)
            if distance + 1e-9 < patrol.min_move_distance_m
        ]
        if too_short:
            raise RuntimeError(f"patrol route contains a short move: {too_short}")

        self._log(
            f"[patrol] route ready: {len(route.targets)} move(s), "
            f"minimum distance={min(distances):.3f} m"
        )
        for index, target in enumerate(route.targets, start=1):
            self._log(
                f"[patrol] position {index}/10 target xyz={target.xyz} "
                f"distance={distances[index - 1]:.3f} m"
            )

        reports: list[VisionReport] = []
        visible_positions: list[int] = []
        for index, segment in enumerate(route.segments, start=1):
            self._set_vision_status(
                f"TASK A PATROL: moving {index}/{patrol.position_count} "
                f"({segment.distance_m:.2f} m straight)"
            )
            motion.execute_linear_route(
                LinearRoutePlan(start_pose=route.start_pose, segments=(segment,)),
                names=(f"task-a-patrol-{index:02d}",),
                speed_mps=self.config.motion.speed_mps,
            )
            if patrol.settle_s:
                self._sleep(patrol.settle_s)

            self._set_vision_status(
                f"TASK A PATROL: observing {index}/{patrol.position_count}"
            )
            visibility = patrol.object_visibility
            target_visible = False
            if visibility.enabled:
                observed_frame, target_visible = self._observe_target_at_position(
                    index
                )
                frame, matches, report = self._evaluate_frame(
                    observed_frame,
                    index - 1,
                )
                if target_visible:
                    visible_positions.append(index)
            else:
                frame, matches, report = self.run_once(index - 1)
            reports.append(report)
            self._record_observation(frame, matches, index - 1)
            if visibility.enabled:
                if target_visible:
                    if index < len(route.segments):
                        self._log(
                            f"[patrol] position {index}: target confirmed; "
                            "advancing to the next position"
                        )
                    else:
                        self._log(
                            "[patrol] final position target confirmed; "
                            "patrol complete"
                        )
                elif index < len(route.segments):
                    self._log(
                        f"[patrol] position {index}: target not confirmed; "
                        "continuing to the next position"
                    )
                else:
                    self._log(
                        "[patrol] final position target not confirmed; "
                        "patrol complete"
                    )

        summary = self._finish_report(reports, threshold)
        if visibility.enabled:
            self._log(
                "[patrol] target confirmed at position(s): "
                f"{visible_positions or 'none'}"
            )
        return replace(
            summary,
            patrol_targets=route.targets,
            patrol_distances_m=distances,
            patrol_visible_positions=tuple(visible_positions),
        )

    def _target_is_visible(self, frame: Frame) -> bool:
        target_class = self.config.vision.target_class.strip()
        minimum = float(self.config.vision.min_confidence)
        return any(
            detection.confidence >= minimum
            and (not target_class or detection.label == target_class)
            for detection in frame.detections
        )

    def _observe_target_at_position(
        self, position_index: int
    ) -> tuple[Frame, bool]:
        """Observe a stopped patrol position without treating absence as fatal.

        Three consecutive target frames confirm presence.  If the bounded
        observation window expires first, the latest frame is returned with
        ``False`` so the caller can record the miss and continue the patrol.
        Vision transport failures still propagate as errors.
        """

        visibility = self.config.task_a.patrol.object_visibility
        required = visibility.visible_confirm_frames
        target = self.config.vision.target_class or "目标物体"
        deadline = self._clock() + visibility.observation_window_s
        confirmations = 0
        latest_frame: Frame | None = None
        best_visible_frame: Frame | None = None
        while True:
            frame = self.vision.capture()
            latest_frame = frame
            if self._target_is_visible(frame):
                confirmations += 1
                best_visible_frame = frame
                self._log(
                    f"[patrol] position {position_index}: target visible "
                    f"{confirmations}/{required}"
                )
                self._set_vision_status(
                    f"TASK A PATROL: target visible at position "
                    f"{position_index}/{self.config.task_a.patrol.position_count} "
                    f"({confirmations}/{required})"
                )
                if confirmations >= required:
                    self._log(
                        f"[patrol] position {position_index}: target confirmed"
                    )
                    return frame, True
            elif confirmations:
                self._log(
                    f"[patrol] position {position_index}: target disappeared "
                    "before visible confirmation; counter reset"
                )
                confirmations = 0
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                break
            self._sleep(min(visibility.poll_s, remaining))

        if latest_frame is None:  # pragma: no cover - capture always runs once
            raise RuntimeError("patrol observation produced no frame")
        self._log(
            f"[patrol] position {position_index}: {target} not confirmed within "
            f"{visibility.observation_window_s:g} s; continuing"
        )
        return best_visible_frame or latest_frame, False

    def _precheck_patrol_route(self, state: RobotState) -> LinearRoutePlan:
        motion = self.motion
        patrol = self.config.task_a.patrol
        errors: list[str] = []
        for name, targets in self._patrol_candidates(state.pose):
            try:
                route = motion.precheck_linear_targets(
                    targets, step_m=patrol.linear_step_m
                )
            except Exception as exc:  # try the next safe route candidate
                errors.append(f"{name}: {exc}")
                self._log(
                    f"[patrol] route {name} rejected during pre-check: {exc}"
                )
                continue
            if len(route.targets) != patrol.position_count:
                errors.append(
                    f"{name}: pre-check returned {len(route.targets)} targets"
                )
                continue
            if any(
                distance + 1e-9 < patrol.min_move_distance_m
                for distance in route.distances_m
            ):
                errors.append(
                    f"{name}: route includes a move shorter than minimum"
                )
                continue
            return route
        detail = "; ".join(errors) if errors else "no route candidates were built"
        raise RuntimeError(f"task A patrol route failed pre-check: {detail}")

    def _patrol_candidates(
        self, start: Pose
    ) -> tuple[tuple[str, tuple[Pose, ...]], ...]:
        """Build deterministic candidates without moving the robot.

        A full x/y/z patrol walks around one small box.  Every edge changes
        only one coordinate, is exactly one stride long, and the cycle uses
        both signs on every axis.  Different box octants are tried because
        the locally reachable Cartesian region depends on the current joint
        configuration.  No candidate bypasses the motion backend pre-check.
        """

        patrol = self.config.task_a.patrol
        axes = tuple(dict.fromkeys(patrol.axes))
        if set(axes) == {"x", "y", "z"}:
            return self._three_axis_patrol_candidates(start, axes)
        return self._single_axis_patrol_candidates(start, axes)

    def _three_axis_patrol_candidates(
        self, start: Pose, axes: Sequence[str]
    ) -> tuple[tuple[str, tuple[Pose, ...]], ...]:
        """Return equivalent box tours for all eight axis-sign octants."""

        patrol = self.config.task_a.patrol
        axis_index = {"x": 0, "y": 1, "z": 2}
        low = tuple(
            float(value) + float(self.config.motion.workspace_margin)
            for value in self.config.motion.workspace_min
        )
        high = tuple(
            float(value) - float(self.config.motion.workspace_margin)
            for value in self.config.motion.workspace_max
        )
        center = tuple((low[index] + high[index]) / 2.0 for index in range(3))
        preferred = tuple(
            1.0 if float(start.xyz[axis_index[axis]]) <= center[axis_index[axis]]
            else -1.0
            for axis in axes
        )

        sign_options = [
            (sx, sy, sz)
            for sx in (1.0, -1.0)
            for sy in (1.0, -1.0)
            for sz in (1.0, -1.0)
        ]
        sign_options.sort(
            key=lambda signs: sum(
                sign != preferred[index] for index, sign in enumerate(signs)
            )
        )

        # The eight-edge cycle closes the box.  Repeating it fills any
        # position_count while keeping every commanded edge at one stride.
        edge_cycle = (
            (0, 1.0),
            (1, 1.0),
            (0, -1.0),
            (2, 1.0),
            (0, 1.0),
            (1, -1.0),
            (0, -1.0),
            (2, -1.0),
        )
        candidates: list[tuple[str, tuple[Pose, ...]]] = []
        for signs in sign_options:
            position = [float(value) for value in start.xyz]
            targets: list[Pose] = []
            for target_number in range(patrol.position_count):
                relative_axis, edge_direction = edge_cycle[
                    target_number % len(edge_cycle)
                ]
                axis = axes[relative_axis]
                position[axis_index[axis]] += (
                    signs[relative_axis]
                    * edge_direction
                    * float(patrol.stride_m)
                )
                targets.append(
                    Pose(tuple(position), start.rpy)
                )
            label = "xyz box " + " ".join(
                f"{axis}{'+' if sign > 0.0 else '-'}"
                for axis, sign in zip(axes, signs)
            )
            candidates.append((label, tuple(targets)))
        return tuple(candidates)

    def _single_axis_patrol_candidates(
        self, start: Pose, axes: Sequence[str]
    ) -> tuple[tuple[str, tuple[Pose, ...]], ...]:
        """Legacy fallback for configurations that intentionally expose
        fewer than three axes.  Full x/y/z patrols never use this fallback,
        because that would silently omit requested directions."""

        patrol = self.config.task_a.patrol
        low = tuple(
            float(value) + float(self.config.motion.workspace_margin)
            for value in self.config.motion.workspace_min
        )
        high = tuple(
            float(value) - float(self.config.motion.workspace_margin)
            for value in self.config.motion.workspace_max
        )

        def inside(point: tuple[float, float, float]) -> bool:
            return all(
                low[index] - 1e-9 <= point[index] <= high[index] + 1e-9
                for index in range(3)
            )

        candidates: list[tuple[str, tuple[Pose, ...]]] = []
        for axis in axes:
            axis_index = {"x": 0, "y": 1, "z": 2}[axis]
            endpoints: list[tuple[float, float, float]] = []
            for sign in (1.0, -1.0):
                point = list(start.xyz)
                point[axis_index] += sign * patrol.stride_m
                candidate = tuple(float(value) for value in point)
                if inside(candidate):  # type: ignore[arg-type]
                    endpoints.append(candidate)  # type: ignore[arg-type]
            if not endpoints:
                continue

            if len(endpoints) == 2:
                pattern = (endpoints[0], endpoints[1])
            else:
                pattern = (endpoints[0], tuple(float(v) for v in start.xyz))
            targets = tuple(
                Pose(pattern[index % len(pattern)], start.rpy)
                for index in range(patrol.position_count)
            )
            distances = self._patrol_distances(start, targets)
            if all(
                distance + 1e-9 >= patrol.min_move_distance_m
                for distance in distances
            ):
                candidates.append((f"axis {axis}", targets))
        return tuple(candidates)

    @staticmethod
    def _patrol_distances(start: Pose, targets: Sequence[Pose]) -> tuple[float, ...]:
        previous = start
        distances: list[float] = []
        for target in targets:
            distances.append(
                math.dist(previous.xyz, target.xyz)
            )
            previous = target
        return tuple(distances)

    def _record_observation(
        self, frame: Frame, matches: tuple[IouMatch, ...], index: int
    ) -> None:
        if self.config.task_a.save_annotated:
            image = self.annotator.render(frame)
            path = self.annotator.save(frame, image, index)
            if path:
                self._log(f"[annotate] wrote {path}")
        else:
            self.annotator.render(frame)
        self._log_match_summary(frame, matches)

    def _finish_report(
        self, reports: Sequence[VisionReport], threshold: float
    ) -> VisionReport:
        summary = merge_reports(reports, threshold=threshold)
        self._log(
            f"[report] frames={summary.frames} detections={summary.detections} "
            f"matched={summary.matched}/{summary.total_annotations} "
            f"passed={summary.passed} pass_rate={summary.pass_rate:.1%} "
            f"mean_iou={summary.mean_iou:.3f}"
        )
        return summary

    def run_once(self, index: int = 0) -> tuple[Frame, tuple[IouMatch, ...], VisionReport]:
        """Capture, filter, annotate and score exactly one frame."""

        self._log(f"[{TaskPhase.SCAN.value}] capturing frame {index}")
        self._set_vision_status(f"TASK A: capturing frame {index + 1}")
        frame = self.vision.capture()
        return self._evaluate_frame(frame, index)

    def _evaluate_frame(
        self,
        frame: Frame,
        index: int,
    ) -> tuple[Frame, tuple[IouMatch, ...], VisionReport]:
        """Filter and score one already captured frame."""

        frame = self._apply_ground_truth(frame)
        usable = tuple(
            detection
            for detection in frame.detections
            if detection.confidence >= self.config.vision.min_confidence
        )
        scored_frame = Frame(
            frame_id=frame.frame_id,
            detections=usable,
            stamp_s=frame.stamp_s,
            annotations=frame.annotations,
            image_path=frame.image_path,
            image=frame.image,
        )
        self._log(
            f"[{TaskPhase.DETECT.value}] frame {frame.frame_id!r}: "
            f"{len(frame.detections)} raw, {len(usable)} above min_confidence"
        )
        matches, report = score_frame(
            scored_frame, threshold=self.config.task_a.iou_threshold
        )
        self._log(
            f"[{TaskPhase.ANNOTATE.value}] drew {len(usable)} prediction(s) "
            f"and {len(frame.annotations)} ground-truth box(es)"
        )
        self._log(
            f"[{TaskPhase.EVALUATE.value}] passed={report.passed}/"
            f"{report.total_annotations} mean_iou={report.mean_iou:.3f}"
        )
        self._set_vision_status(
            f"TASK A: frame {index + 1}, {len(usable)} detection(s), "
            f"IoU passed {report.passed}/{report.total_annotations}"
        )
        return scored_frame, matches, report

    def _set_vision_status(self, text: str) -> None:
        set_status = getattr(self.vision, "set_status", None)
        if set_status is None:
            return
        try:
            set_status(text)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._log(f"[vision:live] status update failed: {exc}")

    def _apply_ground_truth(self, frame: Frame) -> Frame:
        """Attach externally loaded ground truth when the source has none."""

        if not self._ground_truth or frame.annotations:
            return frame
        boxes = self._ground_truth.get(frame.frame_id)
        if boxes is None and frame.image_path:
            boxes = self._ground_truth.get(Path(frame.image_path).name) or self._ground_truth.get(
                Path(frame.image_path).stem
            )
        if boxes is None:
            return frame
        return Frame(
            frame_id=frame.frame_id,
            detections=frame.detections,
            stamp_s=frame.stamp_s,
            annotations=boxes,
            image_path=frame.image_path,
            image=frame.image,
        )

    def _log_match_summary(self, frame: Frame, matches: tuple[IouMatch, ...]) -> None:
        for match in matches:
            if match.annotation is None:
                label = match.detection.label if match.detection else "-"
                self._log(f"    {label:12s} iou=0.000 (no ground truth) miss")
                continue
            label = match.label
            if match.detection is None:
                self._log(f"    {label:12s} iou=0.000 miss")
                continue
            flag = "ok" if match.passed else "low"
            self._log(f"    {label:12s} iou={match.iou:.3f} {flag}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.vision, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # pragma: no cover - defensive
                self._log(f"[shutdown] vision: {exc}")
        motion = getattr(self, "motion", None)
        if motion is not None:
            close_motion = getattr(motion, "close", None)
            if close_motion is not None:
                try:
                    close_motion()
                except Exception as exc:  # pragma: no cover - defensive
                    self._log(f"[shutdown] motion: {exc}")
