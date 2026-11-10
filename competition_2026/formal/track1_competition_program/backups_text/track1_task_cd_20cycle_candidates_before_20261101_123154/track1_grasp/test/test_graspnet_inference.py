import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from track1_grasp import graspnet_inference
from track1_grasp.graspnet_inference import (
    GraspNetInferenceResult,
    sha256_file,
)
from track1_grasp.jaw_grasp_scoring import JawGraspCandidate


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT.parents[1]
CHECKPOINT = (
    DATA_ROOT
    / "models"
    / "graspnet"
    / "checkpoint_thirdparty_freax_epoch18.tar"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868"
)


def inference_result(count: int = 2) -> GraspNetInferenceResult:
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
    return GraspNetInferenceResult(
        translations=np.asarray(
            [[0.1, -0.02, 0.3], [0.2, 0.01, 0.4]][:count],
            dtype=np.float32,
        ),
        rotations=rotations,
        scores=np.linspace(0.9, 0.5, count, dtype=np.float32),
        network_scores=np.linspace(1.0, 0.6, count, dtype=np.float32),
        tolerances=np.linspace(0.02, 0.01, count, dtype=np.float32),
        widths=np.linspace(0.04, 0.03, count, dtype=np.float32),
        heights=np.full(count, 0.02, dtype=np.float32),
        depths=np.full(count, 0.04, dtype=np.float32),
        approaches=np.tile([-1.0, 0.0, 0.0], (count, 1)).astype(np.float32),
        angles_rad=np.zeros(count, dtype=np.float32),
        seed_indices=np.arange(count, dtype=np.int64),
        source_point_indices=np.arange(count, dtype=np.int64) + 17,
        metadata={"quality_validated": False},
    )


def test_verified_checkpoint_hash_is_persisted():
    assert CHECKPOINT.is_file()
    assert sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256
    assert graspnet_inference.DEFAULT_CHECKPOINT_SHA256 == EXPECTED_CHECKPOINT_SHA256


def test_sha256_file_matches_known_content(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"graspnet-offline-check")
    assert sha256_file(path) == (
        "631dce487b7b00e5b087c315199b5809d681836d85e80fcf9b0c1128b1321ffb"
    )


def test_checkpoint_sha_mismatch_is_rejected_before_model_build(tmp_path: Path, monkeypatch):
    path = tmp_path / "bad-checkpoint.tar"
    path.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(graspnet_inference, "_require_torch", lambda: object())

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        graspnet_inference.load_inference_model(
            path,
            expected_sha256="0" * 64,
        )


def test_import_does_not_import_torch():
    env = os.environ.copy()
    source_path = str(PACKAGE_ROOT)
    env["PYTHONPATH"] = (
        source_path
        if not env.get("PYTHONPATH")
        else source_path + os.pathsep + env["PYTHONPATH"]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import track1_grasp.graspnet_inference; "
            "print('torch' in sys.modules)",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_result_round_trip_and_candidate_conversion():
    result = inference_result()
    payload = result.to_dict()
    assert payload["count"] == 2
    assert payload["metadata"]["quality_validated"] is False
    candidate = JawGraspCandidate.from_graspnet_result(result, 1)
    assert candidate.candidate_id == "graspnet_00001"
    assert candidate.depth_m == pytest.approx(0.04)
    assert candidate.source_point_index == 18
    assert candidate.metadata["graspnet_translation_semantics"].startswith(
        "surface seed/reference"
    )


def test_empty_result_is_valid():
    result = GraspNetInferenceResult(
        translations=np.empty((0, 3), dtype=np.float32),
        rotations=np.empty((0, 3, 3), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        network_scores=np.empty((0,), dtype=np.float32),
        tolerances=np.empty((0,), dtype=np.float32),
        widths=np.empty((0,), dtype=np.float32),
        heights=np.empty((0,), dtype=np.float32),
        depths=np.empty((0,), dtype=np.float32),
        approaches=np.empty((0, 3), dtype=np.float32),
        angles_rad=np.empty((0,), dtype=np.float32),
        seed_indices=np.empty((0,), dtype=np.int64),
        source_point_indices=np.empty((0,), dtype=np.int64),
    )
    assert len(result) == 0
    assert result.to_dict()["count"] == 0
    with pytest.raises(IndexError):
        JawGraspCandidate.from_graspnet_result(result, 0)
