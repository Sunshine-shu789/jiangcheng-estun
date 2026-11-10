"""Gripper backends and the post-close hold-verification boundary."""

from __future__ import annotations

import time

from .config import GripperConfig
from .domain import Detection, GripperState, MotionPlan


class MockGripperBackend:
    """In-memory gripper that records every open/close command."""

    def __init__(self, config: GripperConfig, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self._state = GripperState.UNKNOWN
        self.events: list[str] = []

    @property
    def state(self) -> GripperState:
        return self._state

    def open(self) -> None:
        self._state = GripperState.OPEN
        self.events.append("open")
        self._log("[gripper:mock] open")

    def close(self) -> None:
        self._state = GripperState.CLOSED
        self.events.append("close")
        self._log("[gripper:mock] close")

    def close_backend(self) -> None:
        return None


class TaskAGripperBackend:
    """Adapter over the existing pluggable gripper in taskA_grasp."""

    def __init__(self, config: GripperConfig, node=None, log=None) -> None:
        if not config.taska_config:
            raise ValueError("taska gripper needs gripper.taska_config")
        from taskA_grasp.config import load_config
        from taskA_grasp.gripper import make_gripper

        self.config = config
        self._log = log or (lambda message: None)
        self._taska_config = load_config(config.taska_config)
        self._gripper = make_gripper(
            self._taska_config.gripper, log=self._log, node=node
        )
        self._state = GripperState.UNKNOWN

    @property
    def state(self) -> GripperState:
        name = str(getattr(self._gripper, "state", "")).lower()
        if name == "open":
            return GripperState.OPEN
        if name in {"closed", "close"}:
            return GripperState.CLOSED
        if name == "moving":
            return GripperState.MOVING
        if name == "fault":
            return GripperState.FAULT
        return self._state

    def open(self) -> None:
        self._gripper.open()
        self._state = GripperState.OPEN
        time.sleep(self.config.settle_time_s)

    def close(self) -> None:
        self._gripper.close()
        self._state = GripperState.CLOSED
        time.sleep(self.config.settle_time_s)

    def close_backend(self) -> None:
        self._gripper.shutdown()


class MockHoldVerifier:
    """Optimistic verifier for no-hardware runs.

    A real implementation should combine one or more of: jaw position,
    motor current, object displacement in the wrist camera, or a short
    lift-and-reobserve check.
    """

    def verify(self, detection: Detection, plan: MotionPlan) -> tuple[bool, str]:
        return True, "mock hold verifier accepted"


def build_gripper(config: GripperConfig, node=None, log=None):
    if config.backend == "mock":
        return MockGripperBackend(config, log=log)
    if config.backend == "taska":
        return TaskAGripperBackend(config, node=node, log=log)
    raise ValueError(f"unsupported gripper backend: {config.backend!r}")
