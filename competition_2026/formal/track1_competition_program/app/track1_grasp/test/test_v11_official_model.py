from pathlib import Path
import hashlib
import json

import pytest
import yaml

from track1_grasp.config import VisionConfig, apply_mode, load_config
from track1_grasp.domain import TaskMode
from track1_grasp.vision import RealSenseYoloVision


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models/official/track1_40class_v11_official_f43234a9.pt"
SHA_FILE = ROOT / "models/official/track1_40class_v11_official.sha256"
METADATA = ROOT / "models/official/track1_40class_v11_official.json"
CONFIG = ROOT / "config/track1_task_cd_40class_v11_official.yaml"
TASK_C_CONFIG = ROOT / "config/track1_task_c_v11_official_20261102.yaml"
TASK_D_CONFIG = ROOT / "config/track1_task_d_v11_official_20261102.yaml"
EXPECTED_SHA256 = "f43234a94f1d71b24cb187082c18eb22fea84c88922c605df5f7b1a79ffcaaa6"
EXPECTED_CLASSES = (
    "1-Cheez-it",
    "2-Starkist_Tuna",
    "3-Scissors",
    "4-Frenchs_Mustard",
    "5-Tomato_Soup",
    "6-Foam_Brick",
    "7-Clamp",
    "8-Plastic_Banana",
    "9-Mug",
    "10-meat_can",
    "11-Padlock",
    "12-Baseball",
    "13-Bowl",
    "14-Sleeve",
    "15-Coca_cola_half",
    "16-ICRA_Duckie",
    "17-Eggs_Plush",
    "18-Dice",
    "19-Elmers_Glue",
    "20-Scotch_Tape",
    "21-Haribo_Golden_Bear",
    "22-E27_Adapter",
    "23-GU24_Adapter",
    "24-Spoon",
    "25-Flat_Screwdriver",
    "26-Plastic_Strawberry",
    "27-Coca_cola_full",
    "28-Kong_Duck_Dog_Toy",
    "29-Paper_Towels",
    "30-Stabilo_OHPen",
    "31-Plastic_White_Cup",
    "32-Wine_Glass",
    "33-Key",
    "34-Nail",
    "35-Book",
    "36-Adjustable_Wrench",
    "37-T-shirt",
    "38-Jumbo_Pencil_Cup",
    "39-Glove",
    "40-Pringles",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_official_v11_weight_hash_and_metadata_match():
    assert MODEL.is_file()
    assert file_sha256(MODEL) == EXPECTED_SHA256
    sha_document = SHA_FILE.read_text(encoding="utf-8").split()
    assert sha_document == [EXPECTED_SHA256, MODEL.name]

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["sha256"] == EXPECTED_SHA256
    assert metadata["class_count"] == 40
    assert tuple(metadata["class_names"]) == EXPECTED_CLASSES
    assert metadata["default_for_tasks"] == ["C", "D"]
    assert metadata["task_a_b_defaults_changed"] is False


def test_cd_v11_config_uses_exact_40_class_order_and_official_weight():
    config = load_config(CONFIG)
    assert config.task.max_picks == 40
    assert Path(config.vision.weights).resolve() == MODEL.resolve()
    assert config.vision.weights_sha256 == EXPECTED_SHA256
    assert config.vision.classes == EXPECTED_CLASSES
    assert config.target.known_classes == EXPECTED_CLASSES
    assert config.target.order == EXPECTED_CLASSES
    assert "17-Eggs_Plush" in config.vision.classes
    assert "26-Plastic_Strawberry" in config.vision.classes
    assert len(config.vision.classes) == len(set(config.vision.classes)) == 40


def test_cd_v11_uses_authorized_live_motion_and_feedback_gripper_for_c_and_d():
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert payload["motion"]["backend"] == "robotmove"
    assert payload["motion"]["dry_run"] is False
    assert payload["motion"]["grasp_safety"]["maximum_grasp_flange_z_m"] is None
    assert payload["motion"]["grasp_safety"]["maximum_grasp_tip_z_m"] is None
    assert payload["gripper"]["backend"] == "ms42dc"
    assert payload["gripper"]["require_feedback"] is True
    assert tuple(payload["vision"]["classes"]) == EXPECTED_CLASSES

    # The shared test fixture supplies only offline box geometry and actuator
    # overrides.  Use it to verify that the same profile applies to both modes
    # without treating the unmeasured official box data as valid.
    base = load_config(CONFIG)
    for mode in (TaskMode.C, TaskMode.D):
        config = apply_mode(base, mode)
        config.validate()
        assert config.task.mode is mode
        assert config.vision.classes == EXPECTED_CLASSES
    assert apply_mode(base, TaskMode.D).target.policy == "known_and_unseen"


def test_task_c_and_d_use_mode_specific_configs_and_one_shared_launcher():
    shared = (ROOT / "bin/run_task_cd").read_text(encoding="utf-8")
    cases = (
        (TASK_C_CONFIG, TaskMode.C, "known"),
        (TASK_D_CONFIG, TaskMode.D, "known_and_unseen"),
    )
    for config_path, mode, policy in cases:
        config = load_config(config_path)
        assert config.task.mode is mode
        assert config.target.policy == policy
        assert config.vision.classes == EXPECTED_CLASSES
        assert Path(config.vision.weights).resolve() == MODEL.resolve()
        assert f"config/{config_path.name}" in shared
        assert "track1_task_cd_40class_v11_official.yaml" not in shared
        assert "track1_task_cd_38class_safe_20261029.yaml" not in shared

    for task, wrapper_name in (("C", "run_task_c"), ("D", "run_task_d")):
        wrapper = (ROOT / "bin" / wrapper_name).read_text(encoding="utf-8")
        assert 'exec "$ROOT/bin/run_task_cd" ' + task in wrapper


def _mapping_differences(left, right, prefix=()):
    differences = set()
    if isinstance(left, dict) and isinstance(right, dict):
        for key in set(left) | set(right):
            path = prefix + (key,)
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(_mapping_differences(left[key], right[key], path))
    elif left != right:
        differences.add(prefix)
    return differences


def test_cd_official_configs_differ_only_in_mode_specific_logic():
    c_payload = yaml.safe_load(TASK_C_CONFIG.read_text(encoding="utf-8"))
    d_payload = yaml.safe_load(TASK_D_CONFIG.read_text(encoding="utf-8"))
    # Task C retains independently validated field fixes for support-plane
    # detection and clear-cycle convergence.  Keep the allowed differences
    # exact so unrelated configuration drift still fails this contract test.
    assert _mapping_differences(c_payload, d_payload) == {
        ("task", "mode"),
        ("target", "policy"),
        ("vision", "unknown_proposer", "enabled"),
        ("task_cd", "fixed_view_no_progress_cycles"),
        (
            "task_cd",
            "short_edge_grasp",
            "pca",
            "support_plane_roi_far_min_local_normal_z",
        ),
        (
            "task_cd",
            "short_edge_grasp",
            "pca",
            "support_plane_roi_far_max_local_depth_span_m",
        ),
        (
            "task_cd",
            "short_edge_grasp",
            "pca",
            "support_plane_target_depth_center_fraction",
        ),
    }


def test_task_mode_gate_keeps_unknown_proposals_out_of_task_c():
    c_config = apply_mode(load_config(TASK_C_CONFIG), TaskMode.C)
    d_config = apply_mode(load_config(TASK_D_CONFIG), TaskMode.D)
    assert c_config.vision.unknown_proposer.enabled is False
    assert d_config.vision.unknown_proposer.enabled is True


def test_yolo_hash_mismatch_is_rejected_before_model_loading(tmp_path):
    bad_weights = tmp_path / "bad.pt"
    bad_weights.write_bytes(b"not a YOLO model")
    config = VisionConfig(
        backend="realsense_yolo",
        weights=str(bad_weights),
        weights_sha256="0" * 64,
        continuous=False,
    )
    source = RealSenseYoloVision(config)
    with pytest.raises(RuntimeError, match="YOLO weights SHA256 mismatch"):
        source._start()
