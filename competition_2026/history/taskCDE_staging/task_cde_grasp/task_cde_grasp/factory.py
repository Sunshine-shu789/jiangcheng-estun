"""Factory that wires configuration to interchangeable adapters."""

from __future__ import annotations

from .config import CDEConfig
from .gripper import MockHoldVerifier, build_gripper
from .motion import build_motion
from .pipeline import PickPipeline
from .planner import PosePlanner
from .policy import TargetPolicy
from .vision import build_vision


def build_pipeline(config: CDEConfig, *, node=None, log=None):
    log = log or (lambda message: None)
    vision = build_vision(config.vision, log=log)
    motion = build_motion(config.motion, node=node, log=log)
    gripper = build_gripper(config.gripper, node=node, log=log)
    planner = PosePlanner(config, log=log)
    policy = TargetPolicy(config.target, order=config.order)
    return PickPipeline(
        config,
        vision=vision,
        policy=policy,
        planner=planner,
        motion=motion,
        gripper=gripper,
        hold_verifier=MockHoldVerifier(),
        log=log,
    )
