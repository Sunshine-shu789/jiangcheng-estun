"""Factory that wires a Track 1 config to interchangeable adapters."""

from __future__ import annotations

from dataclasses import replace

from .annotator import Annotator
from .config import Track1Config
from .domain import TaskMode
from .gripper import build_gripper, build_hold_verifier
from .motion import build_motion
from .pipeline import PickPipeline
from .planner import PosePlanner
from .policy import TargetPolicy
from .vision import build_vision
from .vision_pipeline import VisionDemoPipeline


def _vision_for(config: Track1Config):
    """Build the vision source, honouring per-task mock scenarios."""

    vision_config = replace(
        config.vision, scenario_file=config.vision.scenario_for(config.task.mode)
    )
    return vision_config


def build_pick_pipeline(
    config: Track1Config,
    *,
    node=None,
    log=None,
    operator_gate=None,
) -> PickPipeline:
    """Assemble the motion pipeline used by tasks B/C/D/E."""

    log = log or (lambda message: None)
    vision = build_vision(
        _vision_for(config),
        log=log,
        adaptive_inverse=config.adaptive_inverse,
    )
    motion = build_motion(config.motion, node=node, log=log)
    gripper = build_gripper(config.gripper, node=node, log=log)
    hold_verifier = build_hold_verifier(config.gripper, gripper, log=log)
    planner = PosePlanner(config, log=log)
    policy = TargetPolicy(config.target, order=config.order)
    return PickPipeline(
        config,
        vision=vision,
        policy=policy,
        planner=planner,
        motion=motion,
        gripper=gripper,
        hold_verifier=hold_verifier,
        operator_gate=operator_gate,
        log=log,
    )


def build_vision_pipeline(
    config: Track1Config,
    *,
    node=None,
    log=None,
    patrol_operator_gate=None,
) -> VisionDemoPipeline:
    """Assemble task A, adding motion only when its patrol is enabled."""

    log = log or (lambda message: None)
    vision = build_vision(
        _vision_for(config),
        log=log,
        adaptive_inverse=config.adaptive_inverse,
    )
    annotator = Annotator(config.task_a, log=log)
    motion = (
        build_motion(config.motion, node=node, log=log)
        if config.task_a.patrol.enabled
        else None
    )
    return VisionDemoPipeline(
        config,
        vision=vision,
        motion=motion,
        annotator=annotator,
        log=log,
        patrol_operator_gate=patrol_operator_gate,
    )


def build_pipeline(
    config: Track1Config,
    *,
    node=None,
    log=None,
    operator_gate=None,
    patrol_operator_gate=None,
):
    """Return the pipeline that matches ``config.task.mode``.

    Task A returns a :class:`VisionDemoPipeline`; every other task returns a
    :class:`PickPipeline`.  Callers only rely on ``run()`` and ``close()``, so
    the two are interchangeable from the node's point of view.
    """

    if config.task.mode is TaskMode.A:
        return build_vision_pipeline(
            config,
            node=node,
            log=log,
            patrol_operator_gate=patrol_operator_gate,
        )
    return build_pick_pipeline(
        config,
        node=node,
        log=log,
        operator_gate=operator_gate,
    )
