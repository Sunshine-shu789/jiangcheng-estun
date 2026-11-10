#!/usr/bin/env python3
"""Audit and build the self-collected Track-1 front-10 dataset.

This is a separate pipeline from the existing post-class curation.  It only
accepts the newly collected 2026-10-29 range, never imports official front-10
images, and never treats a missing RGB file as an empty-box frame.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import numpy as np
    import yaml
except ImportError as exc:
    raise SystemExit(f"缺少审计依赖：{exc}")

DEFAULT_ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
DEFAULT_SOURCE = Path("/media/sunshine/新加卷/赛道一_C_D自采照片/20261027_session_001")
DATASET_NAME = "track1_front10_selfcollected_v1"
FRAME_START = 1290
FRAME_END = 1564
TARGET_IDS = tuple(range(1, 11))
FILE_KINDS = ("rgb", "depth", "metadata", "overlay")
TRIPLE_A = (1547, 1555, "triple-can-layout-20261029-a")
TRIPLE_B = (1556, 1564, "triple-can-layout-20261029-b")
EXPECTED_SHAPE = (640, 480)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_catalog(path: Path) -> tuple[dict[int, dict[str, Any]], str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = payload.get("classes") or {}
    catalog: dict[int, dict[str, Any]] = {}
    for key, value in classes.items():
        class_id = int(key)
        if class_id in TARGET_IDS:
            item = dict(value)
            item["class_id"] = class_id
            catalog[class_id] = item
    if sorted(catalog) != list(TARGET_IDS):
        raise RuntimeError(f"类别表前10类不完整：{sorted(catalog)}")
    return catalog, sha256_file(path)


def physical_group(row: dict[str, Any]) -> str:
    frame_id = int(row["frame_id"])
    for start, end, name in (TRIPLE_A, TRIPLE_B):
        if start <= frame_id <= end:
            return name
    capture_group = str(row.get("capture_group") or "")
    if capture_group:
        return f"class-{int(row['class_id'])}:{capture_group}"
    return f"class-{int(row['class_id'])}:single-{row['frame_id']}"


def load_rows(source: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    checksums: dict[str, str] = {}
    for line in (source / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        checksums[rel.strip()] = digest.strip()
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate((source / "frames.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if FRAME_START <= int(row["frame_id"]) <= FRAME_END:
            row["_line_no"] = line_no
            rows.append(row)
    return rows, checksums


def audit_row(row: dict[str, Any], source: Path, checksums: dict[str, str]) -> dict[str, Any]:
    record = {
        "line_no": row["_line_no"],
        "frame_id": str(row["frame_id"]),
        "class_id": int(row["class_id"]),
        "scene_id": str(row.get("scene_id") or ""),
        "captured_at": str(row.get("captured_at") or ""),
        "capture_group": str(row.get("capture_group") or ""),
        "scene_type": str(row.get("scene_type") or ""),
        "physical_group": physical_group(row),
        "source_files": {kind: str((row.get("files") or {}).get(kind) or "") for kind in FILE_KINDS},
        "original_files": dict(row.get("files") or {}),
        "issues": [],
        "warnings": [],
        "quality": {},
    }
    if record["class_id"] not in TARGET_IDS:
        record["issues"].append("class_not_in_front10")
    if record["scene_type"] == "box_11l_empty":
        record["warnings"].append("box_11l_empty_scene_hint_ignored")
    if not record["capture_group"]:
        record["warnings"].append("empty_capture_group")
    for kind in FILE_KINDS:
        rel = record["source_files"][kind]
        path = source / rel
        if not rel or not path.is_file():
            record["issues"].append(f"{kind}_missing")
            continue
        if path.stat().st_size <= 0:
            record["issues"].append(f"{kind}_zero_size")
            continue
        if rel not in checksums:
            record["issues"].append(f"{kind}_not_in_sha256sums")
        elif sha256_file(path) != checksums[rel]:
            record["issues"].append(f"{kind}_sha256_mismatch")
    rgb_path = source / record["source_files"]["rgb"]
    if record["source_files"]["rgb"] and rgb_path.is_file():
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            record["issues"].append("rgb_unreadable")
        else:
            height, width = rgb.shape[:2]
            if (width, height) != EXPECTED_SHAPE:
                record["issues"].append(f"rgb_size_{width}x{height}")
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            record["quality"].update(
                rgb_width=width,
                rgb_height=height,
                rgb_mean=round(float(gray.mean()), 3),
                rgb_std=round(float(gray.std()), 3),
                rgb_sharpness=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
            )
    depth_path = source / record["source_files"]["depth"]
    if record["source_files"]["depth"] and depth_path.is_file():
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            record["issues"].append("depth_unreadable")
        else:
            height, width = depth.shape[:2]
            if (width, height) != EXPECTED_SHAPE:
                record["issues"].append(f"depth_size_{width}x{height}")
            if depth.dtype != np.uint16:
                record["issues"].append(f"depth_dtype_{depth.dtype}")
            nonzero = int((depth > 0).sum())
            record["quality"]["depth_nonzero_ratio"] = round(nonzero / depth.size, 6)
            if nonzero == 0:
                record["issues"].append("depth_all_zero")
    meta_path = source / record["source_files"]["metadata"]
    if record["source_files"]["metadata"] and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("frame_id")) != record["frame_id"]:
                record["issues"].append("metadata_frame_id_mismatch")
            if int(meta.get("class_id", -1)) != record["class_id"]:
                record["issues"].append("metadata_class_id_mismatch")
            if str((meta.get("files") or {}).get("rgb", "")) != record["source_files"]["rgb"]:
                record["issues"].append("metadata_rgb_path_mismatch")
        except Exception:
            record["issues"].append("metadata_invalid_json")
    overlay_path = source / record["source_files"]["overlay"]
    if record["source_files"]["overlay"] and overlay_path.is_file():
        if cv2.imread(str(overlay_path), cv2.IMREAD_COLOR) is None:
            record["issues"].append("overlay_unreadable")
    record["accepted"] = not record["issues"]
    return record


def choose_splits(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["physical_group"]].append(record)
    ordered_groups = sorted(
        by_group,
        key=lambda group: (min(str(r["captured_at"]) for r in by_group[group]), group),
    )
    classes_by_group = {
        group: {int(r["class_id"]) for r in by_group[group]} for group in ordered_groups
    }
    val_groups: set[str] = set()
    # At least two well-spaced physical groups per class.  Triple-can groups can
    # satisfy several classes at once, so the union size usually stays near 20%.
    for class_id in TARGET_IDS:
        candidates = [group for group in ordered_groups if class_id in classes_by_group[group]]
        if len(candidates) < 2:
            raise RuntimeError(f"类别 {class_id} 的物理采集组少于2，无法稳定划分 train/val")
        desired = max(2, math.ceil(len(candidates) * 0.20))
        desired = min(desired, len(candidates) - 1)
        for ordinal in range(1, desired + 1):
            if sum(class_id in classes_by_group[g] for g in val_groups) >= desired:
                break
            index = math.floor(ordinal * len(candidates) / (desired + 1))
            group = candidates[min(max(index, 0), len(candidates) - 1)]
            val_groups.add(group)
    split_by_group = {group: ("val" if group in val_groups else "train") for group in ordered_groups}
    selected_by_group: dict[str, dict[str, Any]] = {}
    selected_frames: set[str] = set()
    for group in ordered_groups:
        group_rows = by_group[group]
        selected = max(
            group_rows,
            key=lambda r: (float(r["quality"].get("rgb_sharpness", -math.inf)), r["frame_id"]),
        )
        selected_by_group[group] = selected
        selected_frames.add(f"{selected['class_id']}:{selected['frame_id']}:{selected['source_files']['rgb']}")
    for record in records:
        record["split"] = split_by_group[record["physical_group"]]
        record["review_selected"] = (
            f"{record['class_id']}:{record['frame_id']}:{record['source_files']['rgb']}" in selected_frames
        )
    counts = {
        split: Counter(int(r["class_id"]) for r in records if r["split"] == split)
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        missing = [class_id for class_id in TARGET_IDS if counts[split][class_id] == 0]
        if missing:
            raise RuntimeError(f"{split} 缺少类别：{missing}")
    return {
        "split_by_group": split_by_group,
        "selected_by_group": selected_by_group,
        "group_count": len(ordered_groups),
        "val_group_count": len(val_groups),
        "class_counts": {
            split: {str(class_id): int(counts[split][class_id]) for class_id in TARGET_IDS}
            for split in ("train", "val")
        },
        "val_groups": sorted(val_groups),
    }


def copy_files(output: Path, source: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        for kind in FILE_KINDS:
            rel = record["source_files"][kind]
            src = source / rel
            dst = output / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if dst.stat().st_size != src.stat().st_size or sha256_file(src) != sha256_file(dst):
                raise RuntimeError(f"复制校验失败：{rel}")


def write_sha256s(root: Path) -> int:
    checksum_path = root / "SHA256SUMS"
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum_path:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def build(source: Path, output: Path, class_map: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"输出目录已存在，拒绝覆盖：{output}")
    raw_rows, checksums = load_rows(source)
    records = [audit_row(row, source, checksums) for row in raw_rows]
    accepted = [record for record in records if record["accepted"]]
    excluded = [record for record in records if not record["accepted"]]
    if not accepted:
        raise RuntimeError("没有可接受的前10类照片")
    split_info = choose_splits(accepted)
    catalog, class_map_sha = load_catalog(class_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"临时目录已存在：{staging}")
    staging.mkdir(parents=True)
    try:
        copy_files(staging, source, accepted)
        accepted_sorted = sorted(
            accepted,
            key=lambda r: (r["captured_at"], r["frame_id"], r["class_id"], r["source_files"]["rgb"]),
        )
        write_jsonl(staging / "frames.jsonl", accepted_sorted)
        (staging / "manifests").mkdir(parents=True, exist_ok=True)
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "split", "physical_group", "image", "frame_id", "class_id",
                    "capture_group", "captured_at", "review_selected",
                ],
            )
            writer.writeheader()
            for record in accepted_sorted:
                writer.writerow({
                    "split": record["split"],
                    "physical_group": record["physical_group"],
                    "image": Path(record["source_files"]["rgb"]).name,
                    "frame_id": record["frame_id"],
                    "class_id": record["class_id"],
                    "capture_group": record["capture_group"],
                    "captured_at": record["captured_at"],
                    "review_selected": str(record["review_selected"]).lower(),
                })
        review_rows = [
            r for r in accepted_sorted if r["review_selected"]
        ]
        with (staging / "manifests" / "review_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "split", "physical_group", "image", "frame_id", "class_id",
                    "capture_group", "captured_at", "split_unit",
                ],
            )
            writer.writeheader()
            for index, record in enumerate(review_rows, 1):
                writer.writerow({
                    "split": record["split"],
                    "physical_group": record["physical_group"],
                    "image": Path(record["source_files"]["rgb"]).name,
                    "frame_id": record["frame_id"],
                    "class_id": record["class_id"],
                    "capture_group": record["capture_group"],
                    "captured_at": record["captured_at"],
                    "split_unit": f"front10-unit-{index:04d}",
                })
        names = [catalog[class_id]["canonical_label"] for class_id in TARGET_IDS]
        dataset_yaml = {
            "path": str(output),
            "train": "rgb",
            "val": "rgb",
            "nc": 10,
            "names": names,
        }
        (staging / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        summary = {
            "schema_version": 1,
            "created_at": now_iso(),
            "dataset_name": DATASET_NAME,
            "source": str(source),
            "frame_range": [f"{FRAME_START:06d}", f"{FRAME_END:06d}"],
            "class_map": str(class_map),
            "class_map_sha256": class_map_sha,
            "raw_record_count": len(records),
            "accepted_record_count": len(accepted),
            "excluded_record_count": len(excluded),
            "accepted_unique_frame_count": len({r["frame_id"] for r in accepted}),
            "accepted_class_counts": {
                str(class_id): sum(r["class_id"] == class_id for r in accepted)
                for class_id in TARGET_IDS
            },
            "physical_group_count": split_info["group_count"],
            "review_selected_count": sum(bool(r["review_selected"]) for r in accepted),
            "split": {
                "method": "physical_group_stratified",
                "val_group_count": split_info["val_group_count"],
                "class_counts": split_info["class_counts"],
                "val_groups": split_info["val_groups"],
            },
            "excluded": [
                {
                    "frame_id": r["frame_id"],
                    "class_id": r["class_id"],
                    "physical_group": r["physical_group"],
                    "rgb": r["source_files"]["rgb"],
                    "issues": r["issues"],
                }
                for r in excluded
            ],
            "policies": {
                "official_front10_used": False,
                "missing_rgb_excluded_not_empty": True,
                "depth_only_record_not_accepted": True,
                "small_objects_not_filtered_by_area": True,
                "scene_type_is_not_a_class": True,
                "triple_can_layouts_are_atomic_groups": True,
                "bursts_stay_in_one_split": True,
                "empty_or_missing_target_is_not_box_11l_empty": True,
            },
        }
        write_json(staging / "manifests" / "audit_summary.json", summary)
        write_json(staging / "dataset_metadata.json", summary)
        (staging / "README.md").write_text(
            f"""# {DATASET_NAME}

这是赛道一前 10 类的新自采派生集，来源仅为：
`{source}`

- 原始新增记录：{len(records)}
- 完整四件套并校验通过：{len(accepted)}
- 排除：{len(excluded)}（Mug RGB 缺失的 001517–001519）
- 物理采集组：{split_info['group_count']}
- 人工复核代表帧：{sum(bool(r['review_selected']) for r in accepted)}
- 官方前 10 类图片：未使用
- 17、26：本数据集不涉及，最终仍保留 YOLO 空槽

三罐画面 `001547–001555` 和 `001556–001564` 分别作为两个物理组，
两组内部不会跨 train/val。缺失 RGB 的帧明确排除，绝不当作空箱负样本。
""",
            encoding="utf-8",
        )
        write_sha256s(staging)
        staging.rename(output)
    except Exception:
        print(f"构建失败，保留诊断目录：{staging}", file=sys.stderr)
        raise
    return {
        "ok": True,
        "output": str(output),
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计并构建赛道一前10类新自采派生集")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_ROOT / "datasets/track1_selfcollected_v1/manifests/class_map.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / f"datasets/{DATASET_NAME}")
    args = parser.parse_args(argv)
    result = build(args.source.resolve(), args.output.resolve(), args.class_map.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
