#!/usr/bin/env python3
"""Create a candidate checkpoint by copying selected YOLO class rows.

All non-target parameters are taken bit-for-bit from the base checkpoint. The
donor checkpoint contributes only the selected rows of the three-scale
classification heads. The output is a candidate only; it never replaces the
production model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


TARGET_TENSOR_SUFFIXES = tuple(
    f"model.23.cv3.{level}.2.{parameter}"
    for level in range(3)
    for parameter in ("weight", "bias")
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> tuple[dict[str, Any], Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise SystemExit(f"不是受支持的 Ultralytics checkpoint: {path}")
    model = checkpoint["model"]
    if not hasattr(model, "state_dict"):
        raise SystemExit(f"checkpoint 中 model 无 state_dict: {path}")
    return checkpoint, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-index", type=int, action="append", required=True)
    args = parser.parse_args()

    base_path = args.base.expanduser().resolve()
    donor_path = args.donor.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    class_indices = sorted(set(args.class_index))
    if not base_path.is_file() or not donor_path.is_file():
        raise SystemExit("base 或 donor 权重不存在")
    if not class_indices or any(index < 0 for index in class_indices):
        raise SystemExit("class-index 必须是非负整数")

    base_checkpoint, base_model = load_checkpoint(base_path)
    donor_checkpoint, donor_model = load_checkpoint(donor_path)
    base_state = base_model.float().state_dict()
    donor_state = donor_model.float().state_dict()
    if sorted(base_state) != sorted(donor_state):
        raise SystemExit("base 与 donor 的 state_dict 键不一致")
    for key in base_state:
        if tuple(base_state[key].shape) != tuple(donor_state[key].shape):
            raise SystemExit(f"base 与 donor 张量形状不一致: {key}")
    target_tensor_names = [key for key in base_state if key in TARGET_TENSOR_SUFFIXES]
    if len(target_tensor_names) != 6:
        raise SystemExit(f"应找到 6 个三尺度分类目标张量，实际 {len(target_tensor_names)}")

    nc = int(getattr(base_model, "nc", 0))
    if nc != 40 or int(getattr(donor_model, "nc", 0)) != 40:
        raise SystemExit("base/donor 必须都是 40 类模型")
    if any(index >= nc for index in class_indices):
        raise SystemExit(f"class-index 超出类别范围: {class_indices}")

    for key in target_tensor_names:
        target = base_state[key]
        if target.ndim == 0 or int(target.shape[0]) != nc:
            raise SystemExit(f"分类头张量第 0 维不是类别数: {key} {tuple(target.shape)}")
        for class_index in class_indices:
            changed = not torch.equal(target[class_index], donor_state[key][class_index])
            target[class_index].copy_(donor_state[key][class_index])
            if not changed:
                print(f"警告：{key} 第 {class_index} 行与 donor 完全相同")

    base_model.load_state_dict(base_state, strict=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True)
    candidate_path = weights_dir / "best.pt"
    base_checkpoint["model"] = base_model
    base_checkpoint["date"] = now_iso()
    base_checkpoint["candidate_provenance"] = {
        "created_at": now_iso(),
        "base_weights": str(base_path),
        "donor_weights": str(donor_path),
        "class_indices": class_indices,
        "official_ids": [index + 1 for index in class_indices],
        "class_names": [str(base_model.names[index]) for index in class_indices],
        "copied_tensors": target_tensor_names,
        "non_target_parameters_unchanged": True,
    }
    torch.save(base_checkpoint, candidate_path)

    # Verify all non-target state tensors remain exactly equal to the base.
    saved_checkpoint, saved_model = load_checkpoint(candidate_path)
    saved_state = saved_model.float().state_dict()
    non_target_diffs = [
        key for key in base_state if key not in target_tensor_names and not torch.equal(base_state[key], saved_state[key])
    ]
    if non_target_diffs:
        raise SystemExit(f"候选权重出现非目标变化: {non_target_diffs[:10]}")
    selected_rows_unchanged: list[str] = []
    for key in target_tensor_names:
        for class_index in range(nc):
            if class_index in class_indices:
                continue
            if not torch.equal(base_state[key][class_index], saved_state[key][class_index]):
                selected_rows_unchanged.append(f"{key}[{class_index}]")
    if selected_rows_unchanged:
        raise SystemExit(f"同层未选中类别行发生变化: {selected_rows_unchanged[:10]}")

    from ultralytics import YOLO

    wrapper = YOLO(str(candidate_path), task="detect")
    if int(wrapper.model.nc) != nc:
        raise SystemExit("候选权重重新加载后类别数不一致")

    report = {
        "schema_version": 1,
        "created_at": now_iso(),
        "mode": "candidate_class_row_transplant",
        "production_model_replaced": False,
        "base_weights": str(base_path),
        "base_sha256": sha256_file(base_path),
        "donor_weights": str(donor_path),
        "donor_sha256": sha256_file(donor_path),
        "candidate_weights": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "class_indices": class_indices,
        "official_ids": [index + 1 for index in class_indices],
        "class_names": [str(base_model.names[index]) for index in class_indices],
        "copied_tensors": target_tensor_names,
        "checks": {
            "state_dict_keys_equal": True,
            "all_tensor_shapes_equal": True,
            "non_target_parameters_unchanged": True,
            "unselected_rows_in_target_tensors_unchanged": True,
            "candidate_reload_nc_40": True,
        },
    }
    (output_dir / "candidate_provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
