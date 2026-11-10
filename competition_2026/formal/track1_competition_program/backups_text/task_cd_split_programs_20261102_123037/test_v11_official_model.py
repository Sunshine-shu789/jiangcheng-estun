from pathlib import Path
import hashlib
import json

import pytest

from track1_grasp.config import VisionConfig, apply_mode, load_config
from track1_grasp.domain import TaskMode
from track1_grasp.vision import RealSenseYoloVision


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models/official/track1_40class_v11_official_f43234a9.pt"
SHA_FILE = ROOT / "models/official/track1_40class_v11_official.sha256"
METADATA = ROOT / "models/official/track1_40class_v11_official.json"
CONFIG = ROOT / "config/track1_task_cd_40class_v11_official.yaml"
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
    base = load_config(CONFIG)
    for mode in (TaskMode.C, TaskMode.D):
        config = apply_mode(base, mode)
        config.validate()
        assert config.task.mode is mode
        assert config.motion.backend == "robotmove"
        assert config.motion.dry_run is False
        assert config.gripper.backend == "ms42dc"
        assert config.gripper.require_feedback is True
        assert config.motion.grasp_safety.maximum_grasp_flange_z_m is None
        assert config.motion.grasp_safety.maximum_grasp_tip_z_m is None
        assert config.vision.classes == EXPECTED_CLASSES
    assert apply_mode(base, TaskMode.D).target.policy == "known_and_unseen"


def test_cd_launchers_default_to_v11_official_config():
    for launcher_name in ("run_task_c", "run_task_d"):
        text = (ROOT / "bin" / launcher_name).read_text(encoding="utf-8")
        assert f"config/{CONFIG.name}" in text
        assert "track1_task_cd_38class_safe_20261029.yaml" not in text


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
