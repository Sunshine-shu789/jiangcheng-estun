from track1_grasp.config import TargetConfig
from track1_grasp.domain import Detection
from track1_grasp.policy import TargetPolicy


def detection(label, confidence, graspability=1.0, instance=""):
    return Detection(
        label=label,
        confidence=confidence,
        bbox=(0, 0, 10, 10),
        base_xyz=(0.1, 0.1, 0.05),
        graspability=graspability,
        instance_id=instance,
    )


def test_any_policy_prefers_combined_confidence_and_graspability():
    policy = TargetPolicy(TargetConfig(policy="any"))
    selected = policy.select(
        [
            detection("mug", 0.80, 0.20),
            detection("bowl", 0.70, 1.00),
        ]
    )
    assert selected.accepted
    assert selected.detection.label == "bowl"


def test_d_policy_accepts_graspable_unknown_and_rejects_weak_unknown():
    policy = TargetPolicy(
        TargetConfig(
            policy="known_and_unseen",
            known_classes=("mug",),
            unknown_labels=("object",),
            unknown_min_graspability=0.70,
        )
    )
    accepted = policy.select([detection("object", 0.80, 0.75)])
    rejected = policy.select([detection("object", 0.95, 0.60)])
    assert accepted.accepted and accepted.detection.label == "object"
    assert not rejected.accepted


def test_e_policy_advances_only_after_success():
    policy = TargetPolicy(
        TargetConfig(policy="ordered", max_attempts_per_target=2),
        order=("mug", "bowl"),
    )
    first = policy.select([detection("bowl", 0.9), detection("mug", 0.8)])
    assert first.detection.label == "mug"
    assert policy.select([detection("bowl", 0.9)]).accepted is False
    policy.commit_success(first)
    second = policy.select([detection("bowl", 0.9)])
    assert second.detection.label == "bowl"
