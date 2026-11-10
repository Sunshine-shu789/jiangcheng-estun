#!/usr/bin/env python3
"""Build Track-1 40-class v3 by preserving v2 and adding verified scissors data.

Pixels are linked from the data-volume sources; v2 is read-only. The only new
labels are the 19 reviewed scissors frames. The build is atomic and refuses to
overwrite an existing v3 dataset.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
V2_ROOT = ROOT / "datasets/track1_training_merged_v2"
VERIFIED_JSON = ROOT / "config/track1_scissors_v3_verified_boxes.json"
OUTPUT = ROOT / "datasets/track1_training_merged_v3"
TEMP = ROOT / "datasets/.track1_training_merged_v3.partial"
LABEL_SOURCE = ROOT / "datasets/track1_scissors_v3_verified_label_sources"
SPLITS = ("train", "val")
EMPTY_SLOTS = {16, 25}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BuildError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def names_from_yaml(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = payload.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or len(names) != 40:
        raise BuildError(f"类别表不是 40 项：{path}")
    return [str(x) for x in names]


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise BuildError(f"标签不是 5 列：{path}:{line_no}")
        cls = int(parts[0])
        values = tuple(float(v) for v in parts[1:])
        if not 0 <= cls < 40:
            raise BuildError(f"类别越界：{path}:{line_no}: {cls}")
        if not all(0.0 <= v <= 1.0 for v in values):
            raise BuildError(f"坐标越界：{path}:{line_no}: {values}")
        x, y, w, h = values
        if w <= 0 or h <= 0 or x-w/2 < -1e-6 or x+w/2 > 1+1e-6 or y-h/2 < -1e-6 or y+h/2 > 1+1e-6:
            raise BuildError(f"框非法：{path}:{line_no}")
        rows.append((cls, x, y, w, h))
    return rows


def symlink(target: Path, link: Path) -> None:
    target = target.resolve(strict=True)
    if not target.is_file():
        raise BuildError(f"链接目标不是文件：{target}")
    os.symlink(target, link)


def main() -> int:
    if OUTPUT.exists():
        raise BuildError(f"拒绝覆盖已有 v3 数据集：{OUTPUT}")
    if TEMP.exists():
        raise BuildError(f"临时目录已存在，请人工检查：{TEMP}")
    if not (V2_ROOT / ".complete").is_file():
        raise BuildError(f"v2 数据集未完成：{V2_ROOT}")
    names = names_from_yaml(V2_ROOT / "dataset.yaml")
    v2_audit = json.loads((V2_ROOT / "audit_summary.json").read_text(encoding="utf-8"))
    if v2_audit.get("training_ready") is not True:
        raise BuildError("v2 training_ready 不是 true")

    v2_groups: dict[str, str] = {}
    for raw in (V2_ROOT / "merge_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        rel = str(row["output_image"])
        if rel in v2_groups:
            raise BuildError(f"v2 manifest 重复：{rel}")
        v2_groups[rel] = str(row["group"])

    verified = json.loads(VERIFIED_JSON.read_text(encoding="utf-8"))
    records = verified.get("records", [])
    if len(records) != 21:
        raise BuildError(f"v3 剪刀复核记录应为 21，实际 {len(records)}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        group_splits[row["group"]].add(row["split"])
        if row["split"] not in SPLITS or int(row["class_id"]) != 2:
            raise BuildError(f"剪刀复核记录字段非法：{row}")
    if any(len(v) != 1 for v in group_splits.values()):
        raise BuildError(f"剪刀 capture group 跨 split：{group_splits}")

    expected_labels: dict[str, str] = {}
    for row in records:
        frame = int(row["frame"])
        if frame in (1334, 1351):
            continue
        name = f"scissors_v3__{row['group'].lower()}__{frame:06d}.txt"
        x, y, w, h = (float(v) for v in row["yolo_xywh_normalized"])
        expected_labels[name] = f"2 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
    if LABEL_SOURCE.exists():
        actual_labels = {p.name: p.read_text(encoding="utf-8") for p in LABEL_SOURCE.rglob("*.txt")}
        if actual_labels != expected_labels:
            raise BuildError(f"已有 v3 标签源与复核记录不一致：{LABEL_SOURCE}")
    else:
        for row in records:
            frame = int(row["frame"])
            if frame in (1334, 1351):
                continue
            split = str(row["split"])
            (LABEL_SOURCE / split).mkdir(parents=True, exist_ok=True)
            name = f"scissors_v3__{row['group'].lower()}__{frame:06d}.txt"
            (LABEL_SOURCE / split / name).write_text(expected_labels[name], encoding="utf-8")
        (LABEL_SOURCE / "verified_boxes.json").write_text(
            json.dumps(verified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (LABEL_SOURCE / "README.md").write_text(
            "v3 新增剪刀标签源。每个标签均由 21 帧最终叠加复核页确认；训练集通过绝对软链接引用，避免在合并目录内散落实体标签。\n",
            encoding="utf-8",
        )

    TEMP.mkdir(parents=True)
    for split in SPLITS:
        (TEMP / "images" / split).mkdir(parents=True)
        (TEMP / "labels" / split).mkdir(parents=True)

    manifest_rows: list[dict[str, Any]] = []
    class_counts = {split: Counter() for split in SPLITS}
    empty_counts = Counter()
    group_split_all: dict[str, set[str]] = defaultdict(set)

    # Preserve every v2 pair through absolute symlinks to its resolved source.
    for split in SPLITS:
        image_dir = V2_ROOT / "images" / split
        label_dir = V2_ROOT / "labels" / split
        images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.stem: p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
        if set(images) != set(labels):
            raise BuildError(f"v2 {split} 图片/标签不配对")
        for stem in sorted(images):
            image = images[stem]
            label = labels[stem]
            rel_image = f"images/{split}/{image.name}"
            rel_label = f"labels/{split}/{label.name}"
            symlink(image, TEMP / rel_image)
            symlink(label, TEMP / rel_label)
            group = v2_groups.get(rel_image)
            if not group:
                raise BuildError(f"v2 manifest 缺少：{rel_image}")
            rows = parse_label(label)
            if not rows:
                empty_counts[split] += 1
            for cls, *_ in rows:
                class_counts[split][cls] += 1
            group_split_all[group].add(split)
            manifest_rows.append({
                "source_kind": "v2_preserved", "source_split": split,
                "source_image": str(image.resolve(strict=True)),
                "source_label": str(label.resolve(strict=True)),
                "output_image": rel_image, "output_label": rel_label,
                "group": group, "class_ids": [r[0] for r in rows],
            })

    existing_by_frame: dict[str, tuple[str, str, Path, Path]] = {}
    for split in SPLITS:
        for image in (TEMP / "images" / split).iterdir():
            for frame in range(1334, 1355):
                if image.name.endswith(f"__{frame:06d}{image.suffix}"):
                    if frame in existing_by_frame:
                        raise BuildError(f"v2 中帧 {frame} 重复")
                    label = TEMP / "labels" / split / f"{image.stem}.txt"
                    existing_by_frame[frame] = (split, image.name, image, label)
    if set(existing_by_frame) != {1334, 1351}:
        raise BuildError(f"v2 中应只有剪刀 1334 和 1351，实际 {sorted(existing_by_frame)}")

    added = []
    for row in records:
        frame = int(row["frame"])
        split = str(row["split"])
        group = f"scissors-v3-{row['group'].lower()}"
        if frame in existing_by_frame:
            old_split, _, image, label = existing_by_frame[frame]
            if old_split != split:
                raise BuildError(f"帧 {frame} split 冲突：v2={old_split}, verified={split}")
            old_classes = [r[0] for r in parse_label(label)]
            if old_classes != [2]:
                raise BuildError(f"v2 帧 {frame} 不是单剪刀标签：{old_classes}")
            continue

        image_source = Path(row["image"])
        suffix = image_source.suffix.lower()
        stem = f"scissors_v3__{row['group'].lower()}__{frame:06d}"
        rel_image = f"images/{split}/{stem}{suffix}"
        rel_label = f"labels/{split}/{stem}.txt"
        symlink(image_source, TEMP / rel_image)
        x, y, w, h = (float(v) for v in row["yolo_xywh_normalized"])
        label_name = f"scissors_v3__{row['group'].lower()}__{frame:06d}.txt"
        label_source = LABEL_SOURCE / split / label_name
        if not label_source.is_file():
            raise BuildError(f"缺少已验证标签源：{label_source}")
        symlink(label_source, TEMP / rel_label)
        class_counts[split][2] += 1
        group_split_all[group].add(split)
        manifest_rows.append({
            "source_kind": "v3_verified_scissors", "source_split": split,
            "source_image": str(image_source.resolve(strict=True)), "source_label": str(label_source.resolve(strict=True)),
            "output_image": rel_image, "output_label": rel_label,
            "group": group, "class_ids": [2], "frame": f"{frame:06d}",
            "label_origin": row["label_origin"],
            "xyxy_pixels": row["xyxy_pixels"], "image_size": row["image_size"],
        })
        added.append({"split": split, "group": group, "frame": f"{frame:06d}", "output_image": rel_image})

    crossings = {g: sorted(s) for g, s in group_split_all.items() if len(s) > 1}
    if crossings:
        raise BuildError(f"合并后 group 跨 split：{crossings}")
    active = set(range(40)) - EMPTY_SLOTS
    for split in SPLITS:
        missing = sorted(i for i in active if class_counts[split][i] == 0)
        forbidden = sorted(i for i in EMPTY_SLOTS if class_counts[split][i])
        if missing or forbidden:
            raise BuildError(f"{split} 类别覆盖失败：missing={missing}, forbidden={forbidden}")

    image_counts = {s: sum(1 for p in (TEMP / 'images' / s).iterdir() if p.is_file()) for s in SPLITS}
    label_counts = {s: sum(1 for p in (TEMP / 'labels' / s).iterdir() if p.is_file()) for s in SPLITS}
    boxes = sum(sum(c.values()) for c in class_counts.values())
    if image_counts != {"train": 335, "val": 103}:
        raise BuildError(f"v3 图片数不符合预期：{image_counts}")
    if label_counts != image_counts:
        raise BuildError(f"v3 图片/标签数不一致：{image_counts} vs {label_counts}")
    if boxes != 696:
        raise BuildError(f"v3 bbox 数应为 696，实际 {boxes}")
    if len(added) != 19:
        raise BuildError(f"应新增 19 帧剪刀，实际 {len(added)}")

    with (TEMP / "merge_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (TEMP / "class_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class_index", "class_name", "boxes"])
        for split in SPLITS:
            for cls in range(40):
                if class_counts[split][cls]:
                    writer.writerow([split, cls, names[cls], class_counts[split][cls]])
    with (TEMP / "v3_added_scissors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "group", "frame", "output_image"])
        writer.writeheader()
        writer.writerows(added)

    dataset_yaml = {
        "path": str(OUTPUT),
        "train": "images/train",
        "val": "images/val",
        "nc": 40,
        "names": names,
    }
    (TEMP / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "created_at": now_iso(),
        "dataset_root": str(OUTPUT),
        "dataset_yaml": str(OUTPUT / "dataset.yaml"),
        "materialization": "absolute_symlink",
        "source_summary": v2_audit["source_summary"],
        "source": {
            "preserved_dataset": str(V2_ROOT),
            "preserved_images": {s: sum(1 for p in (V2_ROOT/'images'/s).iterdir() if p.is_file()) for s in SPLITS},
            "preserved_boxes": int(v2_audit["output_audit"]["box_count"]),
            "scissors_verified_boxes": str(VERIFIED_JSON),
            "scissors_verified_labels": str(LABEL_SOURCE),
            "scissors_visual_contact_sheet": verified["contact_sheet"],
        },
        "images": image_counts,
        "labels": label_counts,
        "box_count": boxes,
        "empty_labels": dict(empty_counts),
        "class_counts": {s: {str(k): v for k, v in sorted(class_counts[s].items())} for s in SPLITS},
        "front10_boxes": {s: sum(class_counts[s][i] for i in range(10)) for s in SPLITS},
        "added_scissors": added,
        "added_scissors_count": len(added),
        "scissors_total_frames_reviewed": len(records),
        "scissors_existing_v2_frames": sorted(existing_by_frame),
        "scissors_groups": sorted(group_splits),
        "capture_group_split_counts": {
            g: {s: sum(1 for r in manifest_rows if r['group'] == g and r['source_split'] == s) for s in SPLITS}
            for g in sorted(group_splits)
        },
        "checks": {
            "v2_preserved": True,
            "image_label_pairing_complete": True,
            "nc_40": True,
            "empty_slots_17_26": True,
            "all_38_active_classes_have_train_and_val": True,
            "no_capture_group_split_crossing": True,
            "scissors_all_21_frames_reviewed": True,
            "scissors_new_frames_added": len(added) == 19,
        },
        "training_ready": True,
        "training_started": False,
        "note": "前10类继续全部来自自采数据；未使用官方前10类图片。17/26仅作为空槽，没有背景类或空箱场景类。",
    }
    (TEMP / "audit_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# track1_training_merged_v3

由 v2 完整保留并增加 19 张已逐帧复核的剪刀图片，用于修复剪刀被识别成 Sleeve/Book/Adjustable_Wrench 的问题。

- 图片：train {image_counts['train']} / val {image_counts['val']}
- bbox：{boxes}
- 类别：40 槽，17/26 为空槽
- 新增剪刀：{len(added)} 张，7 个拍摄组；同组不会跨 train/val
- 所有图片和标签均为数据盘绝对软链接；新增标签源位于 `track1_scissors_v3_verified_label_sources`
- 未使用官方前 10 类图片
- 本数据集没有开始训练

见 `audit_summary.json`、`merge_manifest.jsonl` 和 `v3_added_scissors.csv`。
"""
    (TEMP / "README.md").write_text(readme, encoding="utf-8")
    (TEMP / ".complete").write_text(now_iso() + "\n", encoding="utf-8")
    TEMP.rename(OUTPUT)
    print(json.dumps({
        "ok": True, "dataset_root": str(OUTPUT), "images": image_counts,
        "boxes": boxes, "added_scissors": len(added),
        "scissors_split_counts": {s: sum(1 for r in records if r["split"] == s) for s in SPLITS},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if TEMP.exists():
            shutil.rmtree(TEMP, ignore_errors=True)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
