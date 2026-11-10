#!/usr/bin/env python3
"""Build a non-destructive v4 dataset with controlled train-only oversampling.

v3 remains the source of truth.  The v4 dataset re-links every v3 image and
label, then creates extra train-only replicas of the few front-ten frames that
contain Tomato Soup (index 4) and meat_can (index 9).  Validation files are
never copied into training.  No source pixels or labels are modified.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
V3_ROOT = ROOT / "datasets/track1_training_merged_v3"
OUTPUT = ROOT / "datasets/track1_training_merged_v4"
TEMP = ROOT / "datasets/.track1_training_merged_v4.partial"
SPLITS = ("train", "val")
EMPTY_SLOTS = {16, 25}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Every selected frame gets two extra copies.  These three hard cases get
# additional copies because they directly cover the observed failure modes:
# top-view Tomato, gold meat_can, and the mixed three-can scene.
SPECIAL_EXTRA = {
    "front10__front10-review-unit-0023__001378": 2,  # top-view Tomato Soup
    "front10__front10-review-unit-0067__001532": 2,  # gold meat_can, front view
    "front10__front10-review-unit-0071__001559": 3,  # StarKist + Tomato + gold meat_can
}
BASE_EXTRA = 2


class BuildError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_names(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = payload.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or len(names) != 40:
        raise BuildError(f"类别表不是 40 项：{path}")
    return [str(item) for item in names]


def link_target(path: Path) -> Path:
    try:
        target = path.resolve(strict=True)
    except Exception as exc:
        raise BuildError(f"链接目标无法解析：{path}: {exc}") from exc
    if not target.is_file():
        raise BuildError(f"链接目标不是文件：{target}")
    return target


def symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        raise BuildError(f"拒绝覆盖已有文件：{link}")
    os.symlink(str(target), link)


def count_labels(paths: list[Path]) -> tuple[int, dict[int, int], int]:
    boxes = 0
    counts: Counter[int] = Counter()
    empty = 0
    for path in paths:
        content = path.read_text(encoding="utf-8").splitlines()
        rows = [line.strip() for line in content if line.strip()]
        if not rows:
            empty += 1
            continue
        for line_no, row in enumerate(rows, 1):
            parts = row.split()
            if len(parts) != 5:
                raise BuildError(f"标签不是 5 列：{path}:{line_no}")
            try:
                cls = int(parts[0])
                x, y, w, h = (float(value) for value in parts[1:])
            except ValueError as exc:
                raise BuildError(f"标签含非数字：{path}:{line_no}") from exc
            if not 0 <= cls < 40 or not all(0.0 <= value <= 1.0 for value in (x, y, w, h)):
                raise BuildError(f"标签类别或坐标非法：{path}:{line_no}")
            if w <= 0 or h <= 0 or x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                raise BuildError(f"标签框非法：{path}:{line_no}")
            counts[cls] += 1
            boxes += 1
    return boxes, dict(counts), empty


def main() -> int:
    if OUTPUT.exists():
        raise BuildError(f"拒绝覆盖已有 v4 数据集：{OUTPUT}")
    if TEMP.exists():
        raise BuildError(f"临时目录已存在，请人工检查：{TEMP}")
    if not (V3_ROOT / ".complete").is_file():
        raise BuildError(f"v3 数据集未完成：{V3_ROOT}")
    v3_audit = json.loads((V3_ROOT / "audit_summary.json").read_text(encoding="utf-8"))
    if v3_audit.get("training_ready") is not True:
        raise BuildError("v3 training_ready 不是 true")
    names = load_names(V3_ROOT / "dataset.yaml")
    if len(names) != 40:
        raise BuildError("v5 类别表异常")

    manifest_path = V3_ROOT / "merge_manifest.jsonl"
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("source_split") not in SPLITS:
            raise BuildError(f"v3 manifest split 非法：{line_no}: {row.get('source_split')}")
        rows.append(row)
    if len(rows) != 438:
        raise BuildError(f"v3 manifest 应为 438 条，实际 {len(rows)}")
    if len({row["output_image"] for row in rows}) != len(rows):
        raise BuildError("v3 manifest output_image 有重复")

    # Verify source group separation before adding anything.
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["group"])].add(str(row["source_split"]))
    crossing = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) != 1}
    if crossing:
        raise BuildError(f"v3 存在跨 split 的采集组：{crossing}")

    selected: list[tuple[dict[str, Any], int]] = []
    for row in rows:
        if row["source_split"] != "train":
            continue
        classes = {int(value) for value in row.get("class_ids", [])}
        if not classes.intersection({4, 9}):
            continue
        stem = Path(str(row["output_image"])).stem
        extra = BASE_EXTRA + int(SPECIAL_EXTRA.get(stem, 0))
        selected.append((row, extra))

    # Do not accidentally oversample a validation or duplicate the same source
    # more than once for a mixed frame.
    selected_keys = {(row["output_image"], extra) for row, extra in selected}
    if len(selected_keys) != len(selected):
        raise BuildError("v4 选中的训练帧出现重复")
    if any(row["source_split"] != "train" for row, _ in selected):
        raise BuildError("验证帧被错误选入过采样")

    temp_dir = Path(tempfile.mkdtemp(prefix=".track1_training_merged_v4.partial.", dir=OUTPUT.parent))
    try:
        for split in SPLITS:
            (temp_dir / "images" / split).mkdir(parents=True, exist_ok=False)
            (temp_dir / "labels" / split).mkdir(parents=True, exist_ok=False)

        out_rows: list[dict[str, Any]] = []
        for row in rows:
            rel_image = Path(str(row["output_image"]))
            rel_label = Path(str(row["output_label"]))
            source_image = link_target(V3_ROOT / rel_image)
            source_label = link_target(V3_ROOT / rel_label)
            symlink(source_image, temp_dir / rel_image)
            symlink(source_label, temp_dir / rel_label)
            copied = dict(row)
            copied["source_image"] = str(source_image)
            copied["source_label"] = str(source_label)
            copied["v4_role"] = "preserved_v3"
            out_rows.append(copied)

        added: list[dict[str, Any]] = []
        for row, extra in selected:
            rel_image = Path(str(row["output_image"]))
            rel_label = Path(str(row["output_label"]))
            source_image = link_target(V3_ROOT / rel_image)
            source_label = link_target(V3_ROOT / rel_label)
            suffix = rel_image.suffix.lower()
            source_stem = rel_image.stem
            tag = "tomato-soup" if 4 in {int(v) for v in row.get("class_ids", [])} else "meat-can"
            if {int(v) for v in row.get("class_ids", [])}.intersection({4, 9}) == {4, 9}:
                tag = "tomato-and-meat-can"
            for index in range(1, extra + 1):
                new_stem = f"v4_oversample__{tag}__{source_stem}__r{index:02d}"
                new_image_rel = Path("images") / "train" / f"{new_stem}{suffix}"
                new_label_rel = Path("labels") / "train" / f"{new_stem}.txt"
                symlink(source_image, temp_dir / new_image_rel)
                symlink(source_label, temp_dir / new_label_rel)
                item = dict(row)
                item.update({
                    "source_image": str(source_image),
                    "source_label": str(source_label),
                    "output_image": new_image_rel.as_posix(),
                    "output_label": new_label_rel.as_posix(),
                    "v4_role": "train_only_oversample",
                    "replica_of": str(row["output_image"]),
                })
                out_rows.append(item)
                added.append(item)

        # Validate pairing and group separation after materialization.
        for split in SPLITS:
            image_stems = {p.stem for p in (temp_dir / "images" / split).iterdir() if p.is_file()}
            label_stems = {p.stem for p in (temp_dir / "labels" / split).iterdir() if p.is_file()}
            if image_stems != label_stems:
                raise BuildError(f"{split} 图片/标签不配对，缺标签={sorted(image_stems-label_stems)[:5]}，多标签={sorted(label_stems-image_stems)[:5]}")
        post_group_splits: dict[str, set[str]] = defaultdict(set)
        for row in out_rows:
            post_group_splits[str(row["group"])].add(str(row["source_split"]))
        post_crossing = {group: sorted(splits) for group, splits in post_group_splits.items() if len(splits) != 1}
        if post_crossing:
            raise BuildError(f"v4 出现跨 split 的采集组：{post_crossing}")
        if any(str(row["v4_role"]) == "train_only_oversample" and str(row["source_split"]) != "train" for row in out_rows):
            raise BuildError("v4 过采样记录不是 train")

        split_images: dict[str, list[Path]] = {}
        split_boxes: dict[str, int] = {}
        split_counts: dict[str, dict[int, int]] = {}
        split_empty: dict[str, int] = {}
        for split in SPLITS:
            images = sorted(p for p in (temp_dir / "images" / split).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            labels = [temp_dir / "labels" / split / f"{p.stem}.txt" for p in images]
            boxes, counts, empty = count_labels(labels)
            split_images[split] = images
            split_boxes[split] = boxes
            split_counts[split] = counts
            split_empty[split] = empty
            if any(cls in EMPTY_SLOTS for cls in counts):
                raise BuildError(f"{split} 的 17/26 空槽出现标签")
            missing_active = [idx for idx in set(range(40)) - EMPTY_SLOTS if counts.get(idx, 0) <= 0]
            if missing_active:
                raise BuildError(f"{split} 缺少 38 个活动类中的：{missing_active}")

        # Exactly preserve the v3 validation set, and only add train replicas.
        v3_val = sorted(p.stem for p in (V3_ROOT / "images" / "val").iterdir() if p.is_file())
        v4_val = sorted(p.stem for p in (temp_dir / "images" / "val").iterdir() if p.is_file())
        if v3_val != v4_val:
            raise BuildError("v4 的验证集与 v3 不一致")
        if len(split_images["train"]) != 335 + len(added):
            raise BuildError("训练图片数与预期不符")
        if len(split_images["val"]) != 103:
            raise BuildError("验证图片数应为 103")

        dataset_yaml = {
            "path": str(OUTPUT),
            "train": "images/train",
            "val": "images/val",
            "nc": 40,
            "names": names,
        }
        (temp_dir / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        with (temp_dir / "merge_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        with (temp_dir / "oversampling.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_image", "class_ids", "extra_copies", "reason"])
            writer.writeheader()
            for row, extra in selected:
                classes = [int(value) for value in row.get("class_ids", [])]
                reason = "hard_case" if Path(str(row["output_image"])).stem in SPECIAL_EXTRA else "target_class"
                writer.writerow({
                    "source_image": row["output_image"],
                    "class_ids": " ".join(str(value) for value in classes),
                    "extra_copies": extra,
                    "reason": reason,
                })

        with (temp_dir / "class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "class_index", "class_name", "boxes"])
            for split in SPLITS:
                for cls in range(40):
                    if split_counts[split].get(cls, 0):
                        writer.writerow([split, cls, names[cls], split_counts[split][cls]])

        # Keep the original post-class review pointers intact so the existing
        # preflight can verify the 348 reviewed records without touching v3.
        source_summary = json.loads(json.dumps(v3_audit.get("source_summary", {}), ensure_ascii=False))
        source_summary["v4_oversampling"] = {
            "source_dataset": str(V3_ROOT),
            "selected_train_frames": len(selected),
            "added_train_replicas": len(added),
            "base_extra_copies": BASE_EXTRA,
            "special_extra": SPECIAL_EXTRA,
        }
        audit = {
            "schema_version": 1,
            "created_at": now_iso(),
            "dataset_root": str(OUTPUT),
            "dataset_yaml": str(OUTPUT / "dataset.yaml"),
            "materialization": "absolute_symlink",
            "source_summary": source_summary,
            "source": {
                "preserved_dataset": str(V3_ROOT),
                "preserved_images": {"train": 335, "val": 103},
                "preserved_boxes": int(v3_audit.get("box_count", 696)),
                "oversampling_csv": str(OUTPUT / "oversampling.csv"),
            },
            "images": {split: len(split_images[split]) for split in SPLITS},
            "labels": {split: len(split_images[split]) for split in SPLITS},
            "box_count": sum(split_boxes.values()),
            "boxes_by_split": split_boxes,
            "empty_labels": split_empty,
            "class_counts": {split: {str(k): v for k, v in sorted(split_counts[split].items())} for split in SPLITS},
            "front10_boxes": {split: sum(split_counts[split].get(i, 0) for i in range(10)) for split in SPLITS},
            "oversampling": {
                "base_extra_copies": BASE_EXTRA,
                "special_extra": SPECIAL_EXTRA,
                "selected_train_frames": len(selected),
                "added_train_replicas": len(added),
                "validation_unchanged": True,
            },
            "checks": {
                "v3_preserved": True,
                "v4_validation_matches_v3": True,
                "image_label_pairing_complete": True,
                "nc_40": True,
                "empty_slots_17_26": True,
                "all_38_active_classes_have_train_and_val": True,
                "no_capture_group_split_crossing": True,
                "oversampling_train_only": True,
                "no_validation_images_added_to_train": True,
                "front10_present": True,
            },
            "training_ready": True,
            "training_started": False,
            "note": "v4 只新增训练集的番茄汤和午餐肉罐头重复副本；验证集与 v3 完全一致。",
        }
        (temp_dir / "audit_summary.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        readme = f"""# track1_training_merged_v4

由 v3 原样保留，追加训练集受控重复采样，用于修复前 10 类中的番茄汤罐头和午餐肉罐头混淆。

- 图片：train {len(split_images['train'])} / val {len(split_images['val'])}
- bbox：{sum(split_boxes.values())}
- 类别：40 槽，17/26 为空槽
- v3 每一张图片/标签都保留；新增内容只出现在 train
- 番茄汤和午餐肉罐头按 `oversampling.csv` 重复；验证集与 v3 完全一致
- 未修改 v3、旧配置、旧桌面入口；本数据集尚未开始训练

见 `audit_summary.json`、`merge_manifest.jsonl`、`oversampling.csv`。
"""
        (temp_dir / "README.md").write_text(readme, encoding="utf-8")
        (temp_dir / ".complete").write_text(now_iso() + "\n", encoding="utf-8")
        temp_dir.rename(OUTPUT)

        print(json.dumps({
            "ok": True,
            "dataset_root": str(OUTPUT),
            "images": {split: len(split_images[split]) for split in SPLITS},
            "boxes": sum(split_boxes.values()),
            "added_train_replicas": len(added),
            "class_counts": {split: {str(k): v for k, v in sorted(split_counts[split].items())} for split in SPLITS},
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
