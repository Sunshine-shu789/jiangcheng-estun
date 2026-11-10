"""Isolated PyTorch/CUDA inference bridge for the GraspNet baseline model.

This module deliberately implements only the inference path.  The official
``models/graspnet.py`` imports label-generation and training KNN modules at
module import time, which makes a ROS/offline deployment unnecessarily depend
on training-only CUDA extensions.  The Stage-1/Stage-2 classes below reuse the
upstream PointNet++ backbone and grasp modules while excluding those training
imports.

The source checkpoint is a third-party mirror.  Its structure and forward pass
have been verified, but its real-world grasp quality has not.  Every result
therefore carries provenance and ``quality_validated = False`` metadata.

Typical use from the data-disk workspace::

    export PYTHONPATH="$PWD/app/track1_grasp:$PWD:$PYTHONPATH"
    tooling/graspnet_venv/bin/python -m \
        track1_grasp.tools.offline_graspnet ...

Importing this module with the system Python is safe; PyTorch is imported only
when a model/inference operation is requested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import importlib
import json
import math
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party" / "graspnet-baseline"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "models"
    / "graspnet"
    / "checkpoint_thirdparty_freax_epoch18.tar"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868"
)
GRASP_MAX_WIDTH_M = 0.1
GRASP_MAX_TOLERANCE_M = 0.05
NUM_VIEWS = 300
NUM_ANGLES = 12
NUM_DEPTHS = 4


@dataclass(frozen=True)
class GraspNetInferenceResult:
    """Decoded GraspNet candidates in the input point-cloud frame.

    ``rotation`` maps the GraspNet local frame to the point-cloud frame.  The
    translation is the network seed/reference point, not the URDF
    ``grasp_tcp`` or gripper body origin.  Use :mod:`jaw_profile` to convert it
    into a physical TCP pose.
    """

    translations: np.ndarray
    rotations: np.ndarray
    scores: np.ndarray
    network_scores: np.ndarray
    tolerances: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    depths: np.ndarray
    approaches: np.ndarray
    angles_rad: np.ndarray
    seed_indices: np.ndarray
    source_point_indices: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        translations = np.asarray(self.translations, dtype=np.float32)
        rotations = np.asarray(self.rotations, dtype=np.float32)
        scalar_arrays = {
            "scores": self.scores,
            "network_scores": self.network_scores,
            "tolerances": self.tolerances,
            "widths": self.widths,
            "heights": self.heights,
            "depths": self.depths,
            "angles_rad": self.angles_rad,
            "seed_indices": self.seed_indices,
            "source_point_indices": self.source_point_indices,
        }
        normalized_scalars: dict[str, np.ndarray] = {}
        count = len(translations)
        if translations.ndim != 2 or translations.shape[1] != 3:
            raise ValueError("translations must have shape (N, 3)")
        if rotations.shape != (count, 3, 3):
            raise ValueError("rotations must have shape (N, 3, 3)")
        approaches = np.asarray(self.approaches, dtype=np.float32)
        if approaches.shape != (count, 3):
            raise ValueError("approaches must have shape (N, 3)")
        for name, values in scalar_arrays.items():
            array = np.asarray(values)
            if array.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            normalized_scalars[name] = array
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "approaches", approaches)
        for name, values in normalized_scalars.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def __len__(self) -> int:
        return int(len(self.translations))

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "count": len(self),
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            for name in (
                "translations",
                "rotations",
                "scores",
                "network_scores",
                "tolerances",
                "widths",
                "heights",
                "depths",
                "approaches",
                "angles_rad",
                "seed_indices",
                "source_point_indices",
            ):
                payload[name] = np.asarray(getattr(self, name)).tolist()
        return payload


@dataclass(frozen=True)
class GraspNetModelBundle:
    """Loaded model plus the exact source/checkpoint metadata used to build it."""

    model: Any
    device: Any
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_epoch: int | None
    checkpoint_loss: float | None
    parameter_count: int
    torch_version: str
    cuda_version: str | None
    cuda_device_name: str | None
    quality_validated: bool = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - exercised by system Python
        raise RuntimeError(
            "GraspNet inference requires the dedicated environment: "
            "source the data-disk workspace and use "
            "tooling/graspnet_venv/bin/python"
        ) from exc


def _explicit_viewpoint_matrix(torch: Any):
    def convert(batch_towards: Any, batch_angle: Any) -> Any:
        axis_x = batch_towards
        ones = torch.ones(
            axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device
        )
        zeros = torch.zeros(
            axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device
        )
        axis_y = torch.stack(
            [-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1
        )
        mask_y = torch.norm(axis_y, dim=-1) == 0
        if bool(torch.any(mask_y)):
            axis_y = axis_y.clone()
            axis_y[mask_y, 1] = 1.0
        axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
        axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
        axis_z = torch.linalg.cross(axis_x, axis_y, dim=-1)
        sine = torch.sin(batch_angle)
        cosine = torch.cos(batch_angle)
        rot_roll = torch.stack(
            (
                ones,
                zeros,
                zeros,
                zeros,
                cosine,
                -sine,
                zeros,
                sine,
                cosine,
            ),
            dim=-1,
        ).reshape(-1, 3, 3)
        base = torch.stack((axis_x, axis_y, axis_z), dim=-1)
        return torch.matmul(base, rot_roll)

    return convert


def _import_third_party_modules() -> tuple[Any, Any, Any, Any]:
    """Import the inference-only upstream modules with local paths first."""

    required = (
        THIRD_PARTY_ROOT / "models" / "backbone.py",
        THIRD_PARTY_ROOT / "models" / "modules.py",
        THIRD_PARTY_ROOT / "utils" / "loss_utils.py",
        THIRD_PARTY_ROOT / "pointnet2" / "pointnet2",
    )
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "GraspNet inference source is incomplete: " + ", ".join(missing)
        )
    for path in (
        THIRD_PARTY_ROOT / "pointnet2",
        THIRD_PARTY_ROOT / "utils",
        THIRD_PARTY_ROOT / "models",
    ):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)

    backbone = importlib.import_module("backbone")
    modules = importlib.import_module("modules")
    loss_utils = importlib.import_module("loss_utils")
    pointnet2 = importlib.import_module("pointnet2")
    return backbone, modules, loss_utils, pointnet2


def _build_model_classes() -> tuple[Any, Any, Any, Any]:
    torch = _require_torch()
    nn = torch.nn
    backbone_module, modules, loss_utils, _pointnet2 = (
        _import_third_party_modules()
    )
    explicit_matrix = _explicit_viewpoint_matrix(torch)
    # The upstream helper still calls torch.cross without a dim argument.  The
    # explicit implementation avoids that deprecated/ambiguous call in both
    # Stage 1 and our decoder while preserving its numerical convention.
    modules.batch_viewpoint_params_to_matrix = explicit_matrix
    loss_utils.batch_viewpoint_params_to_matrix = explicit_matrix

    class GraspNetStage1(nn.Module):
        def __init__(self, input_feature_dim: int = 0, num_view: int = NUM_VIEWS):
            super().__init__()
            self.backbone = backbone_module.Pointnet2Backbone(input_feature_dim)
            self.vpmodule = modules.ApproachNet(num_view, 256)

        def forward(self, end_points: dict[str, Any]) -> dict[str, Any]:
            pointcloud = end_points["point_clouds"]
            seed_features, seed_xyz, end_points = self.backbone(
                pointcloud, end_points
            )
            return self.vpmodule(seed_xyz, seed_features, end_points)

    class GraspNetStage2(nn.Module):
        def __init__(
            self,
            num_angle: int = NUM_ANGLES,
            num_depth: int = NUM_DEPTHS,
            cylinder_radius: float = 0.05,
            hmin: float = -0.02,
            hmax_list: Sequence[float] = (0.01, 0.02, 0.03, 0.04),
        ):
            super().__init__()
            self.num_angle = num_angle
            self.num_depth = num_depth
            self.crop = modules.CloudCrop(
                64, 3, cylinder_radius, hmin, list(hmax_list)
            )
            self.operation = modules.OperationNet(num_angle, num_depth)
            self.tolerance = modules.ToleranceNet(num_angle, num_depth)

        def forward(self, end_points: dict[str, Any]) -> dict[str, Any]:
            pointcloud = end_points["input_xyz"]
            seed_xyz = end_points["fp2_xyz"]
            vp_rot = end_points["grasp_top_view_rot"]
            vp_features = self.crop(seed_xyz, pointcloud, vp_rot)
            end_points = self.operation(vp_features, end_points)
            return self.tolerance(vp_features, end_points)

    class GraspNetInference(nn.Module):
        def __init__(self, input_feature_dim: int = 0):
            super().__init__()
            self.view_estimator = GraspNetStage1(input_feature_dim)
            self.grasp_generator = GraspNetStage2()

        def forward(self, end_points: dict[str, Any]) -> dict[str, Any]:
            end_points = self.view_estimator(end_points)
            return self.grasp_generator(end_points)

    return GraspNetInference, explicit_matrix, modules, loss_utils


def _checkpoint_tensor_count(state_dict: Mapping[str, Any]) -> int:
    return sum(1 for value in state_dict.values() if hasattr(value, "shape"))


def load_inference_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str = "cuda",
    expected_sha256: str | None = DEFAULT_CHECKPOINT_SHA256,
    verify_hash: bool = True,
) -> GraspNetModelBundle:
    """Build the isolated Stage-1/Stage-2 model and strictly load a checkpoint."""

    torch = _require_torch()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GraspNet checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if verify_hash and expected_sha256 is not None:
        if checkpoint_sha256.lower() != str(expected_sha256).lower():
            raise ValueError(
                "GraspNet checkpoint SHA256 mismatch: "
                f"expected {expected_sha256}, got {checkpoint_sha256}"
            )

    model_class, _matrix, _modules, _loss_utils = _build_model_classes()
    model = model_class(input_feature_dim=0)

    # PyTorch >=2.6 defaults to weights_only=True.  The checkpoint contains
    # only tensors and scalar metadata; keep that safe default explicitly.
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError("checkpoint does not contain model_state_dict")
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    incompatible = model.load_state_dict(state_dict, strict=True)
    if getattr(incompatible, "missing_keys", None) or getattr(
        incompatible, "unexpected_keys", None
    ):
        raise RuntimeError(
            "strict checkpoint load produced missing/unexpected keys: "
            f"missing={getattr(incompatible, 'missing_keys', None)!r}, "
            f"unexpected={getattr(incompatible, 'unexpected_keys', None)!r}"
        )

    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    resolved_device = torch.device(requested)
    model.to(resolved_device)
    model.eval()
    epoch = payload.get("epoch")
    loss = payload.get("loss")
    return GraspNetModelBundle(
        model=model,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=int(epoch) if epoch is not None else None,
        checkpoint_loss=float(loss) if loss is not None else None,
        parameter_count=int(sum(p.numel() for p in model.parameters())),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
        cuda_device_name=(
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else None
        ),
    )


def _decode_predictions(
    end_points: Mapping[str, Any],
    *,
    max_candidates: int | None = None,
    min_score: float | None = None,
    explicit_matrix: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    torch = _require_torch()
    batch_size = len(end_points["point_clouds"])
    if batch_size != 1:
        raise ValueError("offline decoder currently accepts batch size 1")
    objectness_score = end_points["objectness_score"][0].float()
    grasp_score = end_points["grasp_score_pred"][0].float()
    grasp_center = end_points["fp2_xyz"][0].float()
    approaching = -end_points["grasp_top_view_xyz"][0].float()
    angle_logits = end_points["grasp_angle_cls_pred"][0].float()
    width_pred = 1.2 * end_points["grasp_width_pred"][0].float()
    width_pred = torch.clamp(width_pred, min=0.0, max=GRASP_MAX_WIDTH_M)
    tolerance_pred = end_points["grasp_tolerance_pred"][0].float()

    angle_class = torch.argmax(angle_logits, dim=0)
    angle_rad = angle_class.float() / NUM_ANGLES * math.pi
    angle_index = angle_class.unsqueeze(0)
    selected_score = torch.gather(grasp_score, 0, angle_index).squeeze(0)
    selected_width = torch.gather(width_pred, 0, angle_index).squeeze(0)
    selected_tolerance = torch.gather(
        tolerance_pred, 0, angle_index
    ).squeeze(0)

    depth_class = torch.argmax(selected_score, dim=1, keepdim=True)
    depth_m = (depth_class.float() + 1.0) * 0.01
    selected_score = torch.gather(selected_score, 1, depth_class)
    selected_angle = torch.gather(angle_rad, 1, depth_class)
    selected_width = torch.gather(selected_width, 1, depth_class)
    selected_tolerance = torch.gather(
        selected_tolerance, 1, depth_class
    )
    selected_score = selected_score.squeeze(1)
    selected_angle = selected_angle.squeeze(1)
    selected_width = selected_width.squeeze(1)
    selected_tolerance = selected_tolerance.squeeze(1)
    depth_m = depth_m.squeeze(1)

    objectness_pred = torch.argmax(objectness_score, dim=0)
    objectness_mask = objectness_pred == 1
    seed_indices = torch.nonzero(objectness_mask, as_tuple=False).squeeze(1)
    objectness_positive_count = int(seed_indices.numel())
    selected_score = selected_score[objectness_mask]
    selected_width = selected_width[objectness_mask]
    selected_tolerance = selected_tolerance[objectness_mask]
    depth_m = depth_m[objectness_mask]
    approaching = approaching[objectness_mask]
    selected_angle = selected_angle[objectness_mask]
    grasp_center = grasp_center[objectness_mask]
    network_score = selected_score
    adjusted_score = network_score * selected_tolerance / GRASP_MAX_TOLERANCE_M
    selected_width = torch.clamp(selected_width, min=0.0, max=GRASP_MAX_WIDTH_M)
    selected_tolerance = torch.clamp(
        selected_tolerance, min=0.0, max=GRASP_MAX_TOLERANCE_M
    )

    rotation = explicit_matrix(approaching, selected_angle)
    fp2_inds = end_points.get("fp2_inds")
    if fp2_inds is None:
        source_point_indices = seed_indices
    else:
        source_point_indices = fp2_inds[0][objectness_mask]

    if max_candidates is not None:
        max_candidates = int(max_candidates)
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if len(adjusted_score) > max_candidates:
            top_indices = torch.topk(adjusted_score, max_candidates).indices
            selected_score = selected_score[top_indices]
            selected_width = selected_width[top_indices]
            selected_tolerance = selected_tolerance[top_indices]
            depth_m = depth_m[top_indices]
            approaching = approaching[top_indices]
            selected_angle = selected_angle[top_indices]
            grasp_center = grasp_center[top_indices]
            network_score = network_score[top_indices]
            adjusted_score = adjusted_score[top_indices]
            rotation = rotation[top_indices]
            seed_indices = seed_indices[top_indices]
            source_point_indices = source_point_indices[top_indices]

    if len(adjusted_score) == 0:
        arrays = {
            "translations": np.empty((0, 3), dtype=np.float32),
            "rotations": np.empty((0, 3, 3), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "network_scores": np.empty((0,), dtype=np.float32),
            "tolerances": np.empty((0,), dtype=np.float32),
            "widths": np.empty((0,), dtype=np.float32),
            "heights": np.empty((0,), dtype=np.float32),
            "depths": np.empty((0,), dtype=np.float32),
            "approaches": np.empty((0, 3), dtype=np.float32),
            "angles_rad": np.empty((0,), dtype=np.float32),
            "seed_indices": np.empty((0,), dtype=np.int64),
            "source_point_indices": np.empty((0,), dtype=np.int64),
        }
        return arrays, {
            "objectness_positive_seeds": objectness_positive_count,
            "selected_before_max_candidates": 0,
            "selected_after_max_candidates": 0,
        }

    # Sort deterministically by score and then by source index so repeated
    # runs on the same cloud are reproducible even when scores tie.
    order = torch.argsort(
        adjusted_score,
        descending=True,
        stable=True,
    )
    selected_score = selected_score[order]
    selected_width = selected_width[order]
    selected_tolerance = selected_tolerance[order]
    depth_m = depth_m[order]
    approaching = approaching[order]
    selected_angle = selected_angle[order]
    grasp_center = grasp_center[order]
    network_score = network_score[order]
    adjusted_score = adjusted_score[order]
    rotation = rotation[order]
    seed_indices = seed_indices[order]
    source_point_indices = source_point_indices[order]

    arrays = {
        "translations": grasp_center.detach().cpu().numpy().astype(np.float32),
        "rotations": rotation.detach().cpu().numpy().astype(np.float32),
        "scores": adjusted_score.detach().cpu().numpy().astype(np.float32),
        "network_scores": network_score.detach().cpu().numpy().astype(np.float32),
        "tolerances": selected_tolerance.detach().cpu().numpy().astype(np.float32),
        "widths": selected_width.detach().cpu().numpy().astype(np.float32),
        "heights": np.full(
            (len(adjusted_score),), 0.02, dtype=np.float32
        ),
        "depths": depth_m.detach().cpu().numpy().astype(np.float32),
        "approaches": approaching.detach().cpu().numpy().astype(np.float32),
        "angles_rad": selected_angle.detach().cpu().numpy().astype(np.float32),
        "seed_indices": seed_indices.detach().cpu().numpy().astype(np.int64),
        "source_point_indices": source_point_indices.detach().cpu().numpy().astype(np.int64),
    }
    stats = {
        "objectness_positive_seeds": objectness_positive_count,
        "selected_before_max_candidates": objectness_positive_count,
        "selected_after_max_candidates": int(len(adjusted_score)),
    }
    if min_score is not None:
        threshold = float(min_score)
        if not math.isfinite(threshold):
            raise ValueError("min_score must be finite")
        keep = arrays["scores"] >= threshold
        arrays = {key: value[keep] for key, value in arrays.items()}
        stats["selected_after_min_score"] = int(np.count_nonzero(keep))
    return arrays, stats


def run_inference(
    points_xyz: np.ndarray | Sequence[Sequence[float]],
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    max_candidates: int | None = 256,
    min_score: float | None = None,
    bundle: GraspNetModelBundle | None = None,
) -> GraspNetInferenceResult:
    """Run one 20k-point cloud through the local GraspNet inference model."""

    torch = _require_torch()
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise ValueError("points_xyz must have shape (N, 3) or (N, 6)")
    if points.shape[0] < 2048:
        raise ValueError("GraspNet inference needs at least 2048 points")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xyz contains non-finite values")
    if bundle is None:
        bundle = load_inference_model(checkpoint_path, device=device)
    if max_candidates is not None and int(max_candidates) <= 0:
        raise ValueError("max_candidates must be positive if provided")

    # The loaded checkpoint has a 3-channel input.  Reject feature columns
    # instead of silently dropping them and producing an ambiguous result.
    if points.shape[1] != 3:
        raise ValueError("the verified checkpoint expects exactly xyz input")

    start = time.perf_counter()
    tensor = torch.from_numpy(points.copy()).to(bundle.device)
    tensor = tensor.unsqueeze(0)
    end_points: dict[str, Any] = {"point_clouds": tensor}
    with torch.inference_mode():
        end_points = bundle.model(end_points)
        arrays, stats = _decode_predictions(
            end_points,
            max_candidates=max_candidates,
            min_score=min_score,
            explicit_matrix=_explicit_viewpoint_matrix(torch),
        )
        if bundle.device.type == "cuda":
            torch.cuda.synchronize(bundle.device)
    elapsed = time.perf_counter() - start
    metadata = {
        "engine": "graspnet_baseline_isolated_inference",
        "checkpoint_path": str(bundle.checkpoint_path),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "checkpoint_epoch": bundle.checkpoint_epoch,
        "checkpoint_loss": bundle.checkpoint_loss,
        "checkpoint_provenance": "third_party_mirror",
        "quality_validated": False,
        "strict_load": True,
        "input_points": int(points.shape[0]),
        "device": str(bundle.device),
        "torch_version": bundle.torch_version,
        "cuda_version": bundle.cuda_version,
        "cuda_device_name": bundle.cuda_device_name,
        "parameter_count": bundle.parameter_count,
        "elapsed_seconds": float(elapsed),
        "max_candidates": max_candidates,
        "min_score": min_score,
        "stats": stats,
        "output_layout": [
            "score",
            "width",
            "height",
            "depth",
            "rotation(3x3)",
            "translation(3)",
            "tolerance",
            "object_id=-1",
        ],
    }
    return GraspNetInferenceResult(
        translations=arrays["translations"],
        rotations=arrays["rotations"],
        scores=arrays["scores"],
        network_scores=arrays["network_scores"],
        tolerances=arrays["tolerances"],
        widths=arrays["widths"],
        heights=arrays["heights"],
        depths=arrays["depths"],
        approaches=arrays["approaches"],
        angles_rad=arrays["angles_rad"],
        seed_indices=arrays["seed_indices"],
        source_point_indices=arrays["source_point_indices"],
        metadata=metadata,
    )


def provenance_payload() -> dict[str, Any]:
    """Return the provenance file if present, otherwise a compact fallback."""

    path = DEFAULT_CHECKPOINT.parent / "checkpoint_provenance.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "checkpoint_sha256": DEFAULT_CHECKPOINT_SHA256,
        "quality_validated": False,
    }


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_CHECKPOINT_SHA256",
    "GraspNetInferenceResult",
    "GraspNetModelBundle",
    "load_inference_model",
    "provenance_payload",
    "run_inference",
    "sha256_file",
]
