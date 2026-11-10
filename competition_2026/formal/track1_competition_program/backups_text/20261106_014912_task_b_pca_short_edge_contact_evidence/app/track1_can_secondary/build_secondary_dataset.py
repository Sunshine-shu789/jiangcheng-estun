#!/usr/bin/env python3
"""Build the isolated 3-class can detector dataset.

The source images are real self-collected photographs already present on the
data volume.  Images and labels are physically copied; no symlinks are created.
Only YOLO classes 0/1/2 are emitted:
  0 = 2-Starkist_Tuna
  1 = 5-Tomato_Soup
  2 = 10-meat_can
Everything else is represented by an empty label file (hard negative).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
OLD_MERGE = ROOT / "datasets/track1_training_merged_v1/merge_manifest.jsonl"
LATEST_EXPORT = ROOT / "datasets/track1_front10_bbox_export_v1"
LATEST_SPLITS = ROOT / "datasets/track1_front10_bbox_review_source_v1/manifests/split_manifest.csv"
POST_EXPORT = ROOT / "datasets/track1_postclasses_bbox_export_v1"
HOLDOUT_RAW = ROOT / "logs/track1_acceptance_web/three_cans_current_raw_20261031.png"
HOLDOUT_DEBUG = Path("/tmp/three_can_now.jpg")

CAN_NAMES = ["2-Starkist_Tuna", "5-Tomato_Soup", "10-meat_can"]
REMAP = {1: 0, 4: 1, 9: 2}
EXPECTED_SOURCE_SPLITS = {"train", "val"}


@dataclass(frozen=True)
class Item:
    source: Path
    source_label: Path
    source_kind: str
    source_split: str
    physical_group: str
    output_name: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 fields, got {len(fields)}")
        cls = int(float(fields[0]))
        vals = tuple(float(x) for x in fields[1:])
        if any(x < 0.0 or x > 1.0 for x in vals):
            raise ValueError(f"{path}:{line_no}: normalized bbox outside [0,1]: {vals}")
        x, y, w, h = vals
        if w <= 0 or h <= 0 or x - w / 2 < -1e-5 or y - h / 2 < -1e-5 or x + w / 2 > 1.00001 or y + h / 2 > 1.00001:
            raise ValueError(f"{path}:{line_no}: invalid bbox {vals}")
        rows.append((cls, x, y, w, h))
    return rows


def load_old_front10() -> tuple[list[Item], int]:
    items: list[Item] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    for line_no, line in enumerate(OLD_MERGE.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source_kind") != "front10":
            continue
        source = Path(row["source_image"])
        label = Path(row["source_label"])
        split = str(row["source_split"])
        group = str(row["group"])
        if not source.is_file() or not label.is_file():
            skipped += 1
            continue
        key = (str(source.resolve()), split)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        items.append(Item(source, label, "old_front10", split, f"old:{group}", f"old_front10__{source.name}"))
    return items, skipped


def load_latest_front10() -> tuple[list[Item], int]:
    group_by_unit: dict[str, str] = {}
    split_by_unit: dict[str, str] = {}
    with LATEST_SPLITS.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            unit = row["split_unit"]
            group_by_unit.setdefault(unit, f"latest:{unit}")
            split_by_unit.setdefault(unit, row["split"])

    items: list[Item] = []
    skipped = 0
    for split in sorted(EXPECTED_SOURCE_SPLITS):
        for image in sorted((LATEST_EXPORT / "images" / split).glob("*")):
            if not image.is_file():
                skipped += 1
                continue
            label = LATEST_EXPORT / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                skipped += 1
                continue
            unit = image.stem.split("__", 1)[0]
            if unit not in group_by_unit:
                raise RuntimeError(f"latest review unit missing from split manifest: {unit}")
            if split_by_unit[unit] != split:
                raise RuntimeError(f"latest split mismatch for {unit}")
            items.append(Item(image, label, "latest_front10", split, group_by_unit[unit], f"latest_front10__{image.name}"))
    return items, skipped


def load_post30() -> tuple[list[Item], int]:
    items: list[Item] = []
    skipped = 0
    for split in sorted(EXPECTED_SOURCE_SPLITS):
        for image in sorted((POST_EXPORT / "images" / split).glob("*")):
            if not image.is_file():
                skipped += 1
                continue
            label = POST_EXPORT / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                skipped += 1
                continue
            # Each exported review frame is already one physical group.
            group = image.stem.split("__", 1)[0]
            items.append(Item(image, label, "post30", split, f"post30:{group}", f"post30__{image.name}"))
    return items, skipped


def write_yaml(build_root: Path, dataset_root: Path) -> None:
    text = (
        f"path: {dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"- {CAN_NAMES[0]}\n"
        f"- {CAN_NAMES[1]}\n"
        f"- {CAN_NAMES[2]}\n"
        "nc: 3\n"
    )
    (build_root / "dataset.yaml").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/track1_can_secondary_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    build = output.with_name(output.name + ".building")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {output}")
    if build.exists():
        raise SystemExit(f"stale build directory exists; remove after inspection: {build}")

    old, old_skipped = load_old_front10()
    latest, latest_skipped = load_latest_front10()
    post, post_skipped = load_post30()
    items = old + latest + post

    if not old or not latest or not post:
        raise SystemExit(f"source set is incomplete: old={len(old)} latest={len(latest)} post={len(post)}")
    if old_skipped or latest_skipped or post_skipped:
        raise SystemExit(f"source files missing: old={old_skipped} latest={latest_skipped} post={post_skipped}")

    names = [x.output_name for x in items]
    if len(names) != len(set(names)):
        dupes = [k for k, v in Counter(names).items() if v > 1]
        raise SystemExit(f"output filename collision: {dupes[:10]}")

    groups_by_split: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.source_split not in EXPECTED_SOURCE_SPLITS:
            raise SystemExit(f"invalid source split: {item.source_split}")
        groups_by_split[item.source_split].add(item.physical_group)
        group_splits[item.physical_group].add(item.source_split)
    overlap = sorted(g for g, splits in group_splits.items() if len(splits) > 1)
    if overlap:
        raise SystemExit(f"physical groups leak across train/val: {overlap[:20]}")

    for split in sorted(EXPECTED_SOURCE_SPLITS):
        (build / "images" / split).mkdir(parents=True, exist_ok=True)
        (build / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats: dict[str, Counter] = {s: Counter() for s in EXPECTED_SOURCE_SPLITS}
    class_instances: dict[str, Counter] = {s: Counter() for s in EXPECTED_SOURCE_SPLITS}
    source_images: dict[str, Counter] = {s: Counter() for s in EXPECTED_SOURCE_SPLITS}
    positive_images: dict[str, Counter] = {s: Counter() for s in EXPECTED_SOURCE_SPLITS}
    negative_images: dict[str, Counter] = {s: Counter() for s in EXPECTED_SOURCE_SPLITS}
    manifest_rows: list[dict] = []

    total = len(items)
    for idx, item in enumerate(items, 1):
        dest_image = build / "images" / item.source_split / item.output_name
        dest_label = build / "labels" / item.source_split / f"{Path(item.output_name).stem}.txt"
        shutil.copyfile(item.source, dest_image)
        raw_rows = parse_yolo_label(item.source_label)
        mapped: list[str] = []
        mapped_classes: list[int] = []
        for cls, x, y, w, h in raw_rows:
            if cls in REMAP:
                out_cls = REMAP[cls]
                mapped.append(f"{out_cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
                mapped_classes.append(out_cls)
        dest_label.write_text(("\n".join(mapped) + "\n") if mapped else "", encoding="utf-8")

        split = item.source_split
        stats[split]["images"] += 1
        stats[split]["groups"] = len(groups_by_split[split])
        source_images[split][item.source_kind] += 1
        if mapped_classes:
            positive_images[split][item.source_kind] += 1
        else:
            negative_images[split][item.source_kind] += 1
        class_instances[split].update(mapped_classes)
        manifest_rows.append({
            "split": split,
            "physical_group": item.physical_group,
            "source_kind": item.source_kind,
            "source_image": str(item.source.resolve()),
            "source_label": str(item.source_label.resolve()),
            "output_image": str(Path("images") / split / item.output_name),
            "output_label": str(Path("labels") / split / dest_label.name),
            "class_ids": sorted(mapped_classes),
            "sha256": sha256(dest_image),
        })
        if idx % 250 == 0 or idx == total:
            print(f"materialized {idx}/{total}", flush=True)

    if set(class_instances["train"]) != {0, 1, 2} or set(class_instances["val"]) != {0, 1, 2}:
        raise SystemExit(f"missing can class in split: train={class_instances['train']} val={class_instances['val']}")

    write_yaml(build, output)
    with (build / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (build / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    holdout = build / "holdout"
    holdout.mkdir(parents=True, exist_ok=True)
    holdout_files = []
    for src in [HOLDOUT_RAW, HOLDOUT_DEBUG]:
        if src.is_file():
            dst = holdout / f"three_cans_current__{src.name}"
            shutil.copyfile(src, dst)
            holdout_files.append({"source": str(src), "file": str(dst.relative_to(build)), "sha256": sha256(dst)})
    (holdout / "README.md").write_text(
        "# 外部 holdout\n\n当前现场三罐摆位截图，只用于最终外部验收，不参与训练或验证。\n",
        encoding="utf-8",
    )
    (holdout / "manifest.json").write_text(json.dumps(holdout_files, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(output),
        "dataset_yaml": str(output / "dataset.yaml"),
        "materialization": "physical_copy",
        "classes": {str(i): n for i, n in enumerate(CAN_NAMES)},
        "source_policy": {
            "official_competition_dataset_used": False,
            "old_front10_selfcollected_source": str(OLD_MERGE),
            "latest_front10_selfcollected_source": str(LATEST_EXPORT),
            "hard_negative_source": str(POST_EXPORT),
            "same_physical_group_across_train_val": False,
        },
        "images": {s: stats[s]["images"] for s in EXPECTED_SOURCE_SPLITS},
        "groups": {s: len(groups_by_split[s]) for s in EXPECTED_SOURCE_SPLITS},
        "source_images": {s: dict(source_images[s]) for s in EXPECTED_SOURCE_SPLITS},
        "positive_images": {s: dict(positive_images[s]) for s in EXPECTED_SOURCE_SPLITS},
        "negative_images": {s: dict(negative_images[s]) for s in EXPECTED_SOURCE_SPLITS},
        "instances": {s: {str(i): class_instances[s][i] for i in range(3)} for s in EXPECTED_SOURCE_SPLITS},
        "holdout_images": len(holdout_files),
        "training_ready": True,
        "training_ready_reason": "all images physically copied; labels limited to 3 can classes; groups disjoint",
    }
    (build / "audit_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (build / "README.md").write_text(
        "# 赛道一三罐头隔离数据集 v1\n\n"
        "- 用途：只训练 `2-Starkist_Tuna`、`5-Tomato_Soup`、`10-meat_can` 三类。\n"
        "- 来源：用户自采前10类真实照片（旧批次 + 最新复核批次）以及后30类真实照片硬负样本。\n"
        "- 未使用官方比赛数据集。\n"
        "- 图片和标签均为实体文件，不使用符号链接。\n"
        "- 非罐头照片保留空标签作为硬负样本。\n"
        "- train/val 按原始采集组隔离，同一组不跨集合。\n"
        "- `holdout/` 是当前现场三罐截图，禁止用于训练或调参。\n",
        encoding="utf-8",
    )

    build.rename(output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
