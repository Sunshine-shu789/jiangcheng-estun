#!/usr/bin/env python3
"""Track-1 40-class training preflight and guarded training launcher.

Default behavior is read-only preflight. Training only starts when --execute is
explicitly supplied. All generated reports, caches and runs stay on the data
volume under the Track-1 project directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import yaml

DATA_ROOT = Path("/media/sunshine/新加卷").resolve()
DEFAULT_PROJECT_ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
PROJECT_ROOT = Path(os.environ.get("TRACK1_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")
SCHEMA_VERSION = 1


class PreflightError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def resolve_path(value: Any, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def names_to_list(names: Any) -> list[str]:
    if isinstance(names, dict):
        try:
            return [str(names[i]) for i in range(len(names))]
        except KeyError as exc:
            raise PreflightError(f"names 不是从 0 开始的连续映射：{exc}") from exc
    if isinstance(names, list):
        return [str(item) for item in names]
    raise PreflightError(f"names 类型非法：{type(names).__name__}")


def add_check(checks: dict[str, Any], name: str, ok: bool, detail: Any) -> None:
    checks[name] = {"ok": bool(ok), "detail": detail}


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"训练配置不存在：{path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PreflightError(f"训练配置顶层必须是字典：{path}")
    required = ("dataset_yaml", "base_weights", "output_root", "preflight_report", "expected", "train")
    missing = [key for key in required if key not in payload]
    if missing:
        raise PreflightError(f"训练配置缺少字段：{missing}")
    if not isinstance(payload["expected"], dict):
        raise PreflightError("expected 必须是字典")
    if not isinstance(payload["train"], dict):
        raise PreflightError("train 必须是字典")
    return payload


def configure_data_cache() -> dict[str, Path]:
    paths = {
        "cache_root": PROJECT_ROOT / "cache" / "track1_training",
        "tmp": PROJECT_ROOT / "cache" / "track1_training" / "tmp",
        "matplotlib": PROJECT_ROOT / "cache" / "track1_training" / "matplotlib",
        "numba": PROJECT_ROOT / "cache" / "track1_training" / "numba",
        "cuda": PROJECT_ROOT / "cache" / "track1_training" / "cuda",
        "torch": PROJECT_ROOT / "cache" / "track1_training" / "torch",
        "xdg": PROJECT_ROOT / "cache" / "track1_training" / "xdg",
        "ultralytics": PROJECT_ROOT / "config" / "ultralytics",
        "state": PROJECT_ROOT / "state",
        "logs": PROJECT_ROOT / "logs",
        "reports": PROJECT_ROOT / "reports",
    }
    for key, path in paths.items():
        if key != "cache_root":
            path.mkdir(parents=True, exist_ok=True)

    # PyTorch dataloader workers create AF_UNIX sockets under TMPDIR. The
    # canonical data-volume path is longer than the 108-byte socket limit, so
    # expose the same directory through a short /tmp symlink. Files still live
    # on the data volume; only the symlink entry itself is in /tmp.
    tmp_alias = Path("/tmp") / f"track1_training_tmp_{os.getuid()}"
    try:
        tmp_target = paths["tmp"].resolve(strict=True)
        if tmp_alias.is_symlink():
            if tmp_alias.resolve(strict=True) != tmp_target:
                raise PreflightError(f"训练临时目录别名指向错误位置：{tmp_alias}")
        elif tmp_alias.exists():
            raise PreflightError(f"训练临时目录别名位置已被普通文件或目录占用：{tmp_alias}")
        else:
            tmp_alias.symlink_to(tmp_target, target_is_directory=True)
        if tmp_alias.resolve(strict=True) != tmp_target:
            raise PreflightError(f"训练临时目录别名未解析到数据盘：{tmp_alias}")
    except Exception as exc:
        if isinstance(exc, PreflightError):
            raise
        raise PreflightError(f"无法建立短训练临时目录：{type(exc).__name__}: {exc}") from exc
    paths["tmp_alias"] = tmp_alias

    os.environ["TMPDIR"] = str(tmp_alias)
    os.environ["TEMP"] = str(tmp_alias)
    os.environ["TMP"] = str(tmp_alias)
    os.environ["MPLCONFIGDIR"] = str(paths["matplotlib"])
    os.environ["NUMBA_CACHE_DIR"] = str(paths["numba"])
    os.environ["CUDA_CACHE_PATH"] = str(paths["cuda"])
    os.environ["TORCH_HOME"] = str(paths["torch"])
    os.environ["XDG_CACHE_HOME"] = str(paths["xdg"])
    os.environ["YOLO_CONFIG_DIR"] = str(paths["ultralytics"])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    return paths


def parse_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str], bool]:
    rows: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    if not path.is_file():
        return rows, [f"标签不存在：{path}"], False
    content = path.read_text(encoding="utf-8", errors="strict").splitlines()
    for line_no, raw in enumerate(content, 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{line_no} 不是 5 列")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{line_no} 含非数字")
            continue
        if not 0 <= class_id < 40:
            errors.append(f"{path.name}:{line_no} 类别越界：{class_id}")
            continue
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            errors.append(f"{path.name}:{line_no} 坐标不在 0..1")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path.name}:{line_no} 宽高必须大于 0")
            continue
        eps = 1e-6
        if x - width / 2 < -eps or x + width / 2 > 1 + eps or y - height / 2 < -eps or y + height / 2 > 1 + eps:
            errors.append(f"{path.name}:{line_no} 框超出图片边界")
            continue
        if width >= 0.98 and height >= 0.98:
            errors.append(f"{path.name}:{line_no} 疑似整图伪框")
            continue
        rows.append((class_id, x, y, width, height))
    return rows, errors, len(content) == 0


def check_postclasses_review(audit: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ready": False,
        "review_audit_path": None,
        "reviewed": None,
        "with_boxes": None,
        "empty": None,
        "skipped": None,
        "boxes": None,
        "errors": [],
    }
    try:
        post_root = Path(str(audit["source_summary"]["post28"]["root"]))
        review_audit_path = post_root / "audit_summary.json"
        result["review_audit_path"] = str(review_audit_path)
        review = json.loads(review_audit_path.read_text(encoding="utf-8"))
        reviewed = int(review.get("item_count", -1))
        empty = int(review.get("empty_background_count", -1))
        with_boxes = reviewed - empty
        skipped = int(review.get("skipped_count", -1))
        boxes = int(review.get("box_count", -1))
        result.update(reviewed=reviewed, with_boxes=with_boxes, empty=empty, skipped=skipped, boxes=boxes)
        expected_values = {
            "reviewed": int(expected.get("review_records", 348)),
            "with_boxes": int(expected.get("reviewed_with_boxes", 346)),
            "empty": int(expected.get("empty_review_records", 2)),
            "skipped": int(expected.get("skipped_review_records", 0)),
            "boxes": int(expected.get("postclass_boxes", 602)),
        }
        failures = []
        if review.get("complete_review") is not True:
            failures.append("complete_review 不是 true")
        if review.get("postclasses_merge_ready") is not True:
            failures.append("postclasses_merge_ready 不是 true")
        if int(review.get("unreviewed_count", -1)) != 0:
            failures.append("仍有未复核记录")
        if skipped != 0:
            failures.append("仍有跳过记录")
        if with_boxes != expected_values["with_boxes"]:
            failures.append(f"有框记录应为 {expected_values['with_boxes']}，实际 {with_boxes}")
        for key in ("reviewed", "empty", "boxes"):
            if result[key] != expected_values[key]:
                failures.append(f"{key} 应为 {expected_values[key]}，实际 {result[key]}")
        result["errors"] = failures
        result["ready"] = not failures
    except Exception as exc:
        result["errors"].append(f"无法核对后类复核审计：{type(exc).__name__}: {exc}")
    return result


def check_dataset(dataset_yaml: Path, expected: dict[str, Any], decode_mode: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    class_counts = {split: {} for split in SPLITS}
    empty_labels = {split: 0 for split in SPLITS}
    image_counts: dict[str, int] = {}
    boxes_total = 0
    dataset_root = dataset_yaml.parent

    if not dataset_yaml.is_file():
        errors.append(f"dataset.yaml 不存在：{dataset_yaml}")
    if not is_within(dataset_yaml, DATA_ROOT):
        errors.append(f"dataset.yaml 不在数据盘：{dataset_yaml}")
    if not is_within(dataset_root, DATA_ROOT):
        errors.append(f"数据集目录不在数据盘：{dataset_root}")

    payload: dict[str, Any] = {}
    names: list[str] = []
    audit: dict[str, Any] = {}
    audit_path = dataset_root / "audit_summary.json"
    try:
        payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PreflightError("dataset.yaml 顶层不是字典")
        names = names_to_list(payload.get("names"))
        add_check(checks, "dataset_yaml_readable", True, str(dataset_yaml))
    except Exception as exc:
        errors.append(f"读取 dataset.yaml 失败：{type(exc).__name__}: {exc}")
        add_check(checks, "dataset_yaml_readable", False, str(exc))

    if payload:
        configured_root = resolve_path(payload.get("path", dataset_root), dataset_root)
        add_check(checks, "dataset_root_matches_yaml", configured_root == dataset_root.resolve(), {
            "yaml_path": str(configured_root), "actual_root": str(dataset_root.resolve())
        })
        if configured_root != dataset_root.resolve():
            errors.append("dataset.yaml 的 path 与文件所在目录不一致")
        nc = int(payload.get("nc", -1))
        expected_nc = int(expected.get("nc", 40))
        add_check(checks, "nc_matches_expected", nc == expected_nc, {"expected": expected_nc, "actual": nc})
        if nc != expected_nc:
            errors.append(f"nc 应为 {expected_nc}，实际 {nc}")
        add_check(checks, "names_count_matches_nc", len(names) == expected_nc, {"expected": expected_nc, "actual": len(names)})
        if len(names) != expected_nc:
            errors.append(f"names 数量应为 {expected_nc}，实际 {len(names)}")
        front10_names = [str(item) for item in expected.get("front10_names", [])]
        front10_match = len(front10_names) == 10 and names[:10] == front10_names
        add_check(checks, "front10_names_preserved", front10_match, names[:10])
        if not front10_match:
            errors.append("前 10 类名称或顺序与确认版本不一致")

    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_ready = audit.get("training_ready") is True
        audit_checks = audit.get("checks", {})
        all_checks_true = isinstance(audit_checks, dict) and all(value is True for value in audit_checks.values())
        add_check(checks, "merged_audit_training_ready", audit_ready, {
            "audit_path": str(audit_path), "training_ready": audit.get("training_ready"),
            "training_started": audit.get("training_started")
        })
        add_check(checks, "merged_audit_all_checks_true", all_checks_true, audit_checks)
        if not audit_ready:
            errors.append("合并审计 training_ready 不是 true")
        if not all_checks_true:
            errors.append("合并审计存在未通过的检查项")
    except Exception as exc:
        errors.append(f"读取合并审计失败：{type(exc).__name__}: {exc}")
        add_check(checks, "merged_audit_training_ready", False, str(exc))

    review_result = check_postclasses_review(audit, expected) if audit else {
        "ready": False, "errors": ["合并审计不可用，无法定位后类复核审计"]
    }
    add_check(checks, "review_348_complete", bool(review_result.get("ready")), {
        "reviewed": review_result.get("reviewed"), "with_boxes": review_result.get("with_boxes"),
        "empty": review_result.get("empty"), "skipped": review_result.get("skipped"),
        "errors": review_result.get("errors", [])
    })
    if not review_result.get("ready"):
        errors.extend(review_result.get("errors", []))

    try:
        from ultralytics.data.utils import check_det_dataset
        ultra_data = check_det_dataset(str(dataset_yaml))
        ultra_names = names_to_list(ultra_data.get("names"))
        ultra_nc = int(ultra_data.get("nc", -1))
        ultra_ok = ultra_nc == int(expected.get("nc", 40)) and len(ultra_names) == ultra_nc
        add_check(checks, "ultralytics_dataset_check", ultra_ok, {"nc": ultra_nc, "names": len(ultra_names)})
        if not ultra_ok:
            errors.append("Ultralytics 数据集检查未通过")
    except Exception as exc:
        add_check(checks, "ultralytics_dataset_check", False, f"{type(exc).__name__}: {exc}")
        errors.append(f"Ultralytics 数据集检查失败：{type(exc).__name__}: {exc}")

    expected_images = expected.get("images", {}) or {}
    empty_slots = {int(item) for item in expected.get("empty_slots", [16, 25])}
    active_indices = set(range(int(expected.get("nc", 40)))) - empty_slots
    all_sample_images: list[Path] = []

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split} 图片或标签目录不存在")
            image_counts[split] = 0
            class_counts[split] = {}
            continue

        broken_images = sorted(str(p) for p in image_dir.iterdir() if p.is_symlink() and not p.exists())
        broken_labels = sorted(str(p) for p in label_dir.iterdir() if p.is_symlink() and not p.exists())
        if broken_images:
            errors.append(f"{split} 存在坏图片链接，例如 {broken_images[:3]}")
        if broken_labels:
            errors.append(f"{split} 存在坏标签链接，例如 {broken_labels[:3]}")

        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        image_counts[split] = len(images)
        image_map = {p.stem: p for p in images}
        label_map = {p.stem: p for p in labels}
        if len(image_map) != len(images):
            errors.append(f"{split} 图片存在重复 stem")
        if len(label_map) != len(labels):
            errors.append(f"{split} 标签存在重复 stem")
        missing_labels = sorted(set(image_map) - set(label_map))
        extra_labels = sorted(set(label_map) - set(image_map))
        if missing_labels:
            errors.append(f"{split} 有图片缺标签，例如 {missing_labels[:5]}")
        if extra_labels:
            errors.append(f"{split} 有标签缺图片，例如 {extra_labels[:5]}")

        split_class_counts: dict[int, int] = {}
        split_boxes = 0
        for stem in sorted(set(image_map) & set(label_map)):
            image_path = image_map[stem]
            label_path = label_map[stem]
            if not image_path.is_symlink() or not label_path.is_symlink():
                errors.append(f"{split}/{stem} 不是绝对符号链接")
            for kind, path in (("图片", image_path), ("标签", label_path)):
                try:
                    target = path.resolve(strict=True)
                    if not is_within(target, DATA_ROOT):
                        errors.append(f"{split}/{path.name} 的{kind}目标不在数据盘：{target}")
                except Exception as exc:
                    errors.append(f"{split}/{path.name} 的{kind}链接无法解析：{exc}")
            try:
                rows, label_errors, physically_empty = parse_label_file(label_path)
            except Exception as exc:
                errors.append(f"{split}/{label_path.name} 无法读取：{type(exc).__name__}: {exc}")
                continue
            if label_errors:
                errors.extend(f"{split}/{item}" for item in label_errors[:10])
            if physically_empty or not rows:
                empty_labels[split] += 1
            split_boxes += len(rows)
            for class_id, *_ in rows:
                split_class_counts[class_id] = split_class_counts.get(class_id, 0) + 1

        class_counts[split] = {str(key): int(value) for key, value in sorted(split_class_counts.items())}
        boxes_total += split_boxes
        expected_count = int(expected_images.get(split, -1))
        add_check(checks, f"{split}_image_count", len(images) == expected_count, {"expected": expected_count, "actual": len(images)})
        if len(images) != expected_count:
            errors.append(f"{split} 图片数应为 {expected_count}，实际 {len(images)}")
        all_sample_images.extend(images)

        missing_active = sorted(index for index in active_indices if split_class_counts.get(index, 0) <= 0)
        missing_front10 = sorted(index for index in range(10) if split_class_counts.get(index, 0) <= 0)
        bad_empty_slots = sorted(index for index in empty_slots if split_class_counts.get(index, 0) > 0)
        add_check(checks, f"{split}_all_38_active_classes_present", not missing_active, {
            "missing": missing_active, "covered": len(active_indices) - len(missing_active)
        })
        add_check(checks, f"{split}_front10_present", not missing_front10, {
            "missing": missing_front10, "counts": {str(index): split_class_counts.get(index, 0) for index in range(10)}
        })
        add_check(checks, f"{split}_empty_slots_17_26", not bad_empty_slots, {
            "empty_slots": sorted(empty_slots), "unexpectedly_labeled": bad_empty_slots
        })
        if missing_active:
            errors.append(f"{split} 缺少实际类别索引：{missing_active}")
        if missing_front10:
            errors.append(f"{split} 前 10 类缺样本：{missing_front10}")
        if bad_empty_slots:
            errors.append(f"{split} 空槽 17/26 出现了标签：{bad_empty_slots}")

    expected_boxes = int(expected.get("boxes", 2238))
    add_check(checks, "total_box_count", boxes_total == expected_boxes, {"expected": expected_boxes, "actual": boxes_total})
    if boxes_total != expected_boxes:
        errors.append(f"bbox 总数应为 {expected_boxes}，实际 {boxes_total}")

    decoded = 0
    decode_errors: list[str] = []
    if decode_mode != "none" and all_sample_images:
        try:
            import cv2
            limit = len(all_sample_images) if decode_mode == "all" else min(64, len(all_sample_images))
            step = max(1, len(all_sample_images) // limit)
            for image_path in all_sample_images[::step][:limit]:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    decode_errors.append(f"无法解码：{image_path}")
                else:
                    decoded += 1
        except Exception as exc:
            decode_errors.append(f"OpenCV 解码检查失败：{type(exc).__name__}: {exc}")
    add_check(checks, "image_decode_sample", not decode_errors, {
        "mode": decode_mode, "decoded": decoded, "errors": decode_errors[:10]
    })
    if decode_errors:
        errors.extend(decode_errors[:10])

    unique_errors = list(dict.fromkeys(errors))
    warnings = [warning for warning in warnings if warning]
    return {
        "ready": not unique_errors,
        "dataset_root": str(dataset_root),
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha256": sha256_file(dataset_yaml) if dataset_yaml.is_file() else None,
        "audit_path": str(audit_path),
        "audit_training_ready": bool(audit.get("training_ready")) if audit else False,
        "audit_training_started": audit.get("training_started") if audit else None,
        "review": review_result,
        "images": image_counts,
        "empty_labels": empty_labels,
        "boxes": boxes_total,
        "class_counts": class_counts,
        "empty_slots_indices": sorted(empty_slots),
        "empty_slots_official_ids": [index + 1 for index in sorted(empty_slots)],
        "checks": checks,
        "warnings": warnings,
        "errors": unique_errors[:50],
        "error_count": len(unique_errors),
    }


def final_conv(sequence: Any) -> Any:
    for module in reversed(list(sequence.children())):
        if hasattr(module, "out_channels"):
            return module
    raise PreflightError("无法定位检测头分类卷积层")


def check_weights(weights_path: Path, expected: dict[str, Any], dataset_names: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}
    result: dict[str, Any] = {
        "ready": False, "weights_path": str(weights_path), "weights_sha256": None,
        "source_nc": None, "target_nc": None, "source_names": [],
        "front10_rows_transferred": False, "checks": checks, "errors": errors
    }
    if not weights_path.is_file():
        errors.append(f"起始权重不存在：{weights_path}")
        add_check(checks, "weights_exist", False, str(weights_path))
        return result
    if not is_within(weights_path, DATA_ROOT):
        errors.append(f"起始权重不在数据盘：{weights_path}")
    add_check(checks, "weights_exist", True, str(weights_path))

    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel

        result["weights_sha256"] = sha256_file(weights_path)
        source_wrapper = YOLO(str(weights_path), task="detect")
        source = source_wrapper.model
        source_head = source.model[-1]
        source_nc = int(getattr(source_head, "nc", getattr(source, "nc", -1)))
        source_names = names_to_list(getattr(source, "names", None))
        result["source_nc"] = source_nc
        result["source_names"] = source_names
        expected_source_nc = int(expected.get("source_nc", 10))
        expected_nc = int(expected.get("nc", 40))
        result["target_nc"] = expected_nc

        add_check(checks, "source_nc_10", source_nc == expected_source_nc, {"expected": expected_source_nc, "actual": source_nc})
        if source_nc != expected_source_nc:
            errors.append(f"起始权重应为 {expected_source_nc} 类，实际 {source_nc}")
        front10_match = len(source_names) >= 10 and source_names[:10] == dataset_names[:10]
        add_check(checks, "source_front10_names_match_dataset", front10_match, source_names[:10])
        if not front10_match:
            errors.append("起始权重前 10 类名称/顺序与数据集不一致")
        try:
            fused = bool(source.is_fused())
        except Exception:
            fused = False
        add_check(checks, "source_weights_unfused", not fused, {"fused": fused})
        if fused:
            errors.append("起始权重是 fused 推理权重，不能用于可靠微调")

        target = DetectionModel(deepcopy(source.yaml), nc=expected_nc, ch=3, verbose=False)
        target.names = {index: name for index, name in enumerate(dataset_names)}
        target.load({"model": source}, verbose=False)
        target_head = target.model[-1]
        target_head_nc = int(getattr(target_head, "nc", -1))
        add_check(checks, "expanded_head_nc_40", target_head_nc == expected_nc, {"expected": expected_nc, "actual": target_head_nc})
        if target_head_nc != expected_nc:
            errors.append(f"扩展后检测头应为 {expected_nc} 类，实际 {target_head_nc}")

        row_results = []
        front10_rows_equal = True
        new_rows_initialized = True
        with torch.no_grad():
            for level, (source_seq, target_seq) in enumerate(zip(source_head.cv3, target_head.cv3)):
                source_conv = final_conv(source_seq)
                target_conv = final_conv(target_seq)
                source_bias = source_conv.bias.detach()
                target_bias = target_conv.bias.detach()
                source_weight = source_conv.weight.detach()
                target_weight = target_conv.weight.detach()
                bias_equal = bool(torch.equal(source_bias[:10], target_bias[:10]))
                weight_equal = bool(torch.equal(source_weight[:10], target_weight[:10]))
                rows_equal = bias_equal and weight_equal
                front10_rows_equal = front10_rows_equal and rows_equal
                new_nonzero = bool(torch.count_nonzero(target_bias[10:]).item() or torch.count_nonzero(target_weight[10:]).item())
                new_rows_initialized = new_rows_initialized and new_nonzero
                row_results.append({
                    "level": level, "source_out_channels": int(source_conv.out_channels),
                    "target_out_channels": int(target_conv.out_channels),
                    "front10_bias_equal": bias_equal, "front10_weight_equal": weight_equal,
                    "new_rows_initialized": new_nonzero
                })
        add_check(checks, "front10_head_rows_transferred", front10_rows_equal, row_results)
        add_check(checks, "new_class_rows_initialized", new_rows_initialized, {"new_rows": expected_nc - source_nc})
        if not front10_rows_equal:
            errors.append("40 类扩展后，前 10 类分类头没有按名称完整继承")
        if not new_rows_initialized:
            errors.append("新增类别分类头没有初始化")

        result["front10_rows_transferred"] = front10_rows_equal
        result["class_head_levels"] = row_results
        result["torch_version"] = torch.__version__
        result["torch_cuda_build"] = torch.version.cuda
        result["ultralytics_version"] = package_version("ultralytics")
        result["parameter_count_source"] = int(sum(parameter.numel() for parameter in source.parameters()))
        result["parameter_count_target"] = int(sum(parameter.numel() for parameter in target.parameters()))
    except Exception as exc:
        errors.append(f"权重扩展检查失败：{type(exc).__name__}: {exc}")
        add_check(checks, "weights_load_and_expand", False, str(exc))

    result["ready"] = not errors
    result["errors"] = list(dict.fromkeys(errors))
    return result


def check_storage(output_root: Path, report_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}
    min_free_gb = float(expected.get("min_free_gb", 5))
    usage = shutil.disk_usage(DATA_ROOT)
    free_gb = usage.free / (1024 ** 3)
    output_on_data = is_within(output_root, DATA_ROOT)
    report_on_data = is_within(report_path, DATA_ROOT)
    add_check(checks, "output_on_data_volume", output_on_data, str(output_root))
    add_check(checks, "report_on_data_volume", report_on_data, str(report_path))
    add_check(checks, "free_space", free_gb >= min_free_gb, {"free_gb": round(free_gb, 2), "minimum_gb": min_free_gb})
    if not output_on_data:
        errors.append(f"训练输出目录不在数据盘：{output_root}")
    if not report_on_data:
        errors.append(f"预检报告不在数据盘：{report_path}")
    if free_gb < min_free_gb:
        errors.append(f"数据盘剩余空间不足：{free_gb:.2f} GiB < {min_free_gb:.2f} GiB")

    probe_ok = False
    probe_error = None
    probe_path = output_root / f".track1_preflight_write_probe_{os.getpid()}"
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with probe_path.open("x", encoding="utf-8") as handle:
            handle.write(now_iso())
        probe_path.unlink()
        probe_ok = True
    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
        try:
            if probe_path.exists():
                probe_path.unlink()
        except Exception:
            pass
    add_check(checks, "data_volume_writable", probe_ok, probe_error or str(output_root))
    if not probe_ok:
        errors.append(f"数据盘不可写：{probe_error}")

    return {
        "ready": not errors, "data_root": str(DATA_ROOT), "output_root": str(output_root),
        "report_path": str(report_path), "free_gb": round(free_gb, 2),
        "minimum_free_gb": min_free_gb, "writable": probe_ok, "checks": checks, "errors": errors
    }


def check_runtime(expected: dict[str, Any], train_config: dict[str, Any], allow_cpu: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    result: dict[str, Any] = {
        "ready": False, "allow_cpu": allow_cpu, "gpu_required": False, "gpu_ready": False,
        "torch_cuda_available": False, "torch_cuda_device_count": 0, "gpu_memory_gb": None,
        "nvidia_smi_ok": False, "nvidia_smi": None, "kernel_driver_version": None,
        "checks": checks, "errors": errors, "warnings": warnings
    }
    device = str(train_config.get("device", "0")).strip().lower()
    result["gpu_required"] = not allow_cpu and device != "cpu"

    try:
        import torch
        result["torch_version"] = torch.__version__
        result["torch_cuda_build"] = torch.version.cuda
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        result["torch_cuda_available"] = available
        result["torch_cuda_device_count"] = count
        add_check(checks, "torch_cuda_available", available, {
            "available": available, "device_count": count, "torch": torch.__version__, "cuda_build": torch.version.cuda
        })
        if available and count > 0:
            props = torch.cuda.get_device_properties(0)
            result["gpu_name"] = props.name
            result["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            result["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        add_check(checks, "torch_cuda_available", False, f"{type(exc).__name__}: {exc}")
        errors.append(f"CUDA 检测失败：{type(exc).__name__}: {exc}")

    driver_version_path = Path("/proc/driver/nvidia/version")
    if driver_version_path.is_file():
        try:
            result["kernel_driver_version"] = driver_version_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            pass

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            process = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=False
            )
            result["nvidia_smi"] = {"returncode": process.returncode, "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
            result["nvidia_smi_ok"] = process.returncode == 0
        except Exception as exc:
            result["nvidia_smi"] = {"returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    else:
        result["nvidia_smi"] = {"returncode": None, "stdout": "", "stderr": "未找到 nvidia-smi"}
    add_check(checks, "nvidia_smi", result["nvidia_smi_ok"], result["nvidia_smi"])
    if not result["nvidia_smi_ok"]:
        warnings.append("nvidia-smi 不可用；最终以 torch.cuda.is_available() 为准")

    min_gpu_gb = float(expected.get("min_gpu_memory_gb", 6))
    if result["gpu_required"]:
        memory_ok = result["gpu_memory_gb"] is not None and float(result["gpu_memory_gb"]) >= min_gpu_gb
        add_check(checks, "gpu_memory_sufficient", memory_ok, {"required_gb": min_gpu_gb, "actual_gb": result["gpu_memory_gb"]})
        result["gpu_ready"] = bool(result["torch_cuda_available"] and memory_ok)
        if not result["torch_cuda_available"]:
            errors.append("GPU 不可用：torch.cuda.is_available() 为 false")
        if result["torch_cuda_available"] and not memory_ok:
            errors.append(f"GPU 显存不足：需要至少 {min_gpu_gb:.1f} GiB，实际 {result['gpu_memory_gb']} GiB")
        result["ready"] = result["gpu_ready"]
    else:
        result["gpu_ready"] = True
        result["ready"] = True
        warnings.append("将使用 CPU；只有显式添加 --allow-cpu 时才允许此模式")

    if not result["ready"]:
        result["recommended_action"] = "先修复 NVIDIA 驱动/设备节点，并确认 nvidia-smi 与 torch.cuda 都正常；不要在 GPU 未就绪时直接训练。"
    return result


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def print_summary(report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    weights = report["weights"]
    storage = report["storage"]
    runtime = report["runtime"]
    overall = report["overall"]
    print("赛道一 40 类训练预检")
    print(f"- 数据集：{'通过' if dataset['ready'] else '未通过'}（{dataset.get('images')}，bbox={dataset.get('boxes')}）")
    print(f"- 后类复核：348→有框 {dataset.get('review', {}).get('with_boxes')} / 空标签 {dataset.get('review', {}).get('empty')} / 跳过 {dataset.get('review', {}).get('skipped')}")
    print(f"- 起始权重：{'通过' if weights['ready'] else '未通过'}（{weights.get('source_nc')} 类 → {weights.get('target_nc')} 类，前 10 类头继承={weights.get('front10_rows_transferred')}）")
    print(f"- 数据盘写权限：{'通过' if storage['ready'] else '未通过'}（剩余 {storage.get('free_gb')} GiB）")
    print(f"- GPU/CUDA：{'通过' if runtime['ready'] else '未通过'}（CUDA={runtime.get('torch_cuda_available')}，显存={runtime.get('gpu_memory_gb')} GiB）")
    print(f"- 训练准备：{'完成' if overall['preparation_ready'] else '未完成'}")
    print(f"- 可以开始训练：{'是' if overall['can_start_training'] else '否'}")
    print("- 本次启动训练：否")
    if overall["blockers"]:
        print("- 阻塞项：")
        for blocker in overall["blockers"][:10]:
            print(f"  - {blocker}")


def run_post_train_audit(
    best_weights: Path, dataset_yaml: Path, train_config: dict[str, Any],
    expected_nc: int, run_dir: Path
) -> dict[str, Any]:
    errors: list[str] = []
    from ultralytics import YOLO

    model_wrapper = YOLO(str(best_weights), task="detect")
    head = model_wrapper.model.model[-1]
    model_nc = int(getattr(head, "nc", -1))
    model_names = names_to_list(model_wrapper.model.names)
    if model_nc != expected_nc or len(model_names) != expected_nc:
        errors.append(f"训练后类别槽异常：nc={model_nc}, names={len(model_names)}")

    device = str(train_config.get("device", "0"))
    if device == "cpu":
        batch = int(train_config.get("batch", 2))
        workers = int(train_config.get("workers", 2))
        amp = False
    else:
        batch = int(train_config.get("batch", 8))
        workers = int(train_config.get("workers", 4))
        amp = bool(train_config.get("amp", True))

    metrics = model_wrapper.val(
        data=str(dataset_yaml), split="val", imgsz=int(train_config.get("imgsz", 640)),
        batch=batch, device=device, workers=workers, amp=amp, conf=0.001, iou=0.6,
        plots=False, verbose=True, project=str(run_dir / "post_train_val"), name="val", exist_ok=True
    )
    box = metrics.box
    class_indices = [int(value) for value in box.ap_class_index.tolist()]
    ap50_values = [float(value) for value in box.ap50.tolist()]
    precision_values = [float(value) for value in box.p.tolist()]
    recall_values = [float(value) for value in box.r.tolist()]
    per_class: dict[str, Any] = {}
    for class_id, ap50, precision, recall in zip(class_indices, ap50_values, precision_values, recall_values):
        per_class[str(class_id)] = {
            "official_id": class_id + 1, "name": model_names[class_id],
            "precision": precision, "recall": recall, "ap50": ap50
        }
    front10_ok = all(
        index in class_indices and math.isfinite(per_class[str(index)]["ap50"]) and per_class[str(index)]["ap50"] > 0.0
        for index in range(10)
    )
    if not front10_ok:
        errors.append("训练后验证中，前 10 类至少有一类没有任何正 AP50，不能验收")
    result = {
        "created_at": now_iso(), "best_weights": str(best_weights), "model_nc": model_nc,
        "model_names_count": len(model_names), "validated_class_indices": class_indices,
        "front10_ap50_ok": front10_ok,
        "front10_metrics": {str(index): per_class.get(str(index)) for index in range(10)},
        "per_class": per_class, "map50": float(box.map50), "map": float(box.map),
        "errors": errors, "accepted": not errors
    }
    write_json_atomic(run_dir / "post_train_audit.json", result)
    return result


def execute_training(
    config: dict[str, Any], dataset_yaml: Path, weights_path: Path, output_root: Path,
    preflight_report: dict[str, Any], run_name_arg: str | None, allow_cpu: bool
) -> int:
    import fcntl
    from ultralytics import YOLO

    if not preflight_report["overall"]["can_start_training"]:
        print("预检未通过，拒绝启动训练。", file=sys.stderr)
        for blocker in preflight_report["overall"]["blockers"]:
            print(f"- {blocker}", file=sys.stderr)
        return 2

    run_name = run_name_arg or f"track1_40class_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        print(f"非法 run name：{run_name}", file=sys.stderr)
        return 2
    run_dir = output_root / run_name
    if run_dir.exists():
        print(f"拒绝覆盖已有训练输出：{run_dir}", file=sys.stderr)
        return 2

    lock_path = PROJECT_ROOT / "state" / "track1_40class_training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("已有另一个赛道一 40 类训练进程在运行，拒绝并发启动。", file=sys.stderr)
            return 2

        train_args = dict(config["train"])
        if allow_cpu:
            cpu_profile = config.get("cpu_profile", {}) or {}
            train_args.update(cpu_profile)
            train_args["device"] = "cpu"
            train_args["amp"] = False
        train_args.update({
            "data": str(dataset_yaml), "project": str(output_root), "name": run_name,
            "exist_ok": False, "resume": False
        })

        history_path = PROJECT_ROOT / "state" / "track1_training_runs.jsonl"
        start_record = {
            "created_at": now_iso(), "run_name": run_name, "run_dir": str(run_dir),
            "weights": str(weights_path), "dataset_yaml": str(dataset_yaml),
            "config_preflight": preflight_report["overall"]
        }
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(start_record, ensure_ascii=False) + "\n")

        print(f"开始训练：run_name={run_name}")
        print(f"输出目录：{run_dir}")
        print(f"起始权重：{weights_path}")
        print(f"数据集：{dataset_yaml}")
        model = YOLO(str(weights_path), task="detect")
        model.train(**train_args)

        best_weights = run_dir / "weights" / "best.pt"
        if not best_weights.is_file():
            print(f"训练结束但未找到 best.pt：{best_weights}", file=sys.stderr)
            return 2
        post_audit = run_post_train_audit(
            best_weights=best_weights, dataset_yaml=dataset_yaml, train_config=train_args,
            expected_nc=int(config["expected"].get("nc", 40)), run_dir=run_dir
        )
        if not post_audit["accepted"]:
            print("训练后验收未通过，模型不能作为最终 40 类模型使用。", file=sys.stderr)
            for error in post_audit["errors"]:
                print(f"- {error}", file=sys.stderr)
            return 2
        print(f"训练后验收通过：{run_dir / 'post_train_audit.json'}")
        return 0


def build_report(
    config_path: Path, config: dict[str, Any], dataset_yaml: Path, weights_path: Path,
    output_root: Path, report_path: Path, decode_mode: str, allow_cpu: bool
) -> dict[str, Any]:
    expected = config["expected"]
    dataset_result = check_dataset(dataset_yaml, expected, decode_mode)
    dataset_names = names_to_list(yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")).get("names")) if dataset_yaml.is_file() else []
    weights_result = check_weights(weights_path, expected, dataset_names)
    storage_result = check_storage(output_root, report_path, expected)
    runtime_result = check_runtime(expected, config["train"], allow_cpu)
    preparation_ready = bool(dataset_result["ready"] and weights_result["ready"] and storage_result["ready"])
    can_start = bool(preparation_ready and runtime_result["ready"])
    blockers: list[str] = []
    if not dataset_result["ready"]:
        blockers.append(f"数据集未通过：{dataset_result['errors'][:3]}")
    if not weights_result["ready"]:
        blockers.append(f"起始权重未通过：{weights_result['errors'][:3]}")
    if not storage_result["ready"]:
        blockers.extend(storage_result["errors"])
    if not runtime_result["ready"]:
        blockers.extend(runtime_result["errors"])
    return {
        "schema_version": SCHEMA_VERSION, "created_at": now_iso(), "config_path": str(config_path),
        "training_started": False, "scope": "仅训练前预检；默认不会启动训练",
        "dataset": dataset_result, "weights": weights_result, "storage": storage_result,
        "runtime": runtime_result,
        "overall": {
            "preparation_ready": preparation_ready, "runtime_ready": runtime_result["ready"],
            "gpu_ready": runtime_result["gpu_ready"], "can_start_training": can_start,
            "blockers": blockers
        }
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="赛道一 40 类训练预检；只有 --execute 才启动训练")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "track1_train_40class_v1.yaml"))
    parser.add_argument("--report", default=None, help="覆盖预检报告路径")
    parser.add_argument("--no-report", action="store_true", help="不写预检报告")
    parser.add_argument("--decode-images", choices=("sample", "all", "none"), default="sample")
    parser.add_argument("--require-gpu", action="store_true", help="GPU/CUDA 未就绪时返回非零退出码")
    parser.add_argument("--allow-cpu", action="store_true", help="仅在明确同意 CPU 训练时使用")
    parser.add_argument("--execute", action="store_true", help="明确启动训练；不加此参数只预检")
    parser.add_argument("--run-name", default=None, help="训练输出子目录名")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        configure_data_cache()
    except Exception as exc:
        print(f"缓存/临时目录配置失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    config_path = resolve_path(args.config)
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"配置错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    dataset_yaml = resolve_path(config["dataset_yaml"])
    weights_path = resolve_path(config["base_weights"])
    output_root = resolve_path(config["output_root"])
    report_path = resolve_path(args.report) if args.report else resolve_path(config["preflight_report"])

    report = build_report(
        config_path=config_path, config=config, dataset_yaml=dataset_yaml,
        weights_path=weights_path, output_root=output_root, report_path=report_path,
        decode_mode=args.decode_images, allow_cpu=args.allow_cpu
    )
    if not args.no_report:
        write_json_atomic(report_path, report)
    print_summary(report)
    print(f"- 预检报告：{report_path if not args.no_report else '未写入'}")

    if args.execute:
        return execute_training(
            config=config, dataset_yaml=dataset_yaml, weights_path=weights_path,
            output_root=output_root, preflight_report=report,
            run_name_arg=args.run_name, allow_cpu=args.allow_cpu
        )

    if args.require_gpu and not report["overall"]["can_start_training"]:
        return 2
    return 0 if report["overall"]["preparation_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
