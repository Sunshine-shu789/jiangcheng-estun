#!/usr/bin/env python3
"""Build an evaluation-only historical holdout from the v25 audit.

Selection rule:
  1. original split == val;
  2. physical_group contains no original train image;
  3. SHA256 never occurs in original train (extra exact-duplicate firewall).

The resulting dataset is for measuring old checkpoints. It must never be used
for training, hyperparameter selection, or label relabeling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_AUDIT = Path("/media/sunshine/新加卷/taskA/reports/v25_audit_20261016.csv")
DEFAULT_OUT = Path("/media/sunshine/新加卷/taskA/datasets/real_ft_v25_hist_holdout_20261016")
DEFAULT_REPORT = Path("/media/sunshine/新加卷/taskA/reports/v25_hist_holdout_20261016.md")
CLASS_NAMES = (
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
)


def read_audit(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    required = {"split", "image", "physical_group", "sha256", "class_names", "instance_count", "image_path", "label_path"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"invalid audit CSV: {path}")
    return rows


def link_file(src: Path, dst: Path, overwrite: bool) -> str:
    if not src.is_file():
        raise SystemExit(f"missing source file: {src}")
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise SystemExit(f"destination already exists: {dst}")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--copy", action="store_true", help="copy instead of symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audit_path = args.audit.expanduser().resolve()
    out = args.out.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    rows = read_audit(audit_path)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["physical_group"]].append(row)
    pure_val_groups = {
        group
        for group, members in groups.items()
        if any(row["split"] == "val" for row in members)
        and not any(row["split"] == "train" for row in members)
    }
    train_hashes = {row["sha256"] for row in rows if row["split"] == "train"}
    selected = [
        row
        for row in rows
        if row["split"] == "val"
        and row["physical_group"] in pure_val_groups
        and row["sha256"] not in train_hashes
    ]

    # One image per SHA is sufficient here; exact duplicates carry no extra signal.
    by_hash: dict[str, dict[str, str]] = {}
    for row in selected:
        by_hash.setdefault(row["sha256"], row)
    selected = sorted(by_hash.values(), key=lambda row: (row["physical_group"], row["image"]))
    if not selected:
        raise SystemExit("no historical holdout images selected")

    if out.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to replace: {out}")
    if out.exists():
        shutil.rmtree(out)

    manifest: list[dict[str, object]] = []
    link_modes: Counter[str] = Counter()
    for row in selected:
        src_image = Path(row["image_path"])
        src_label = Path(row["label_path"])
        dst_image = out / "images/val" / src_image.name
        dst_label = out / "labels/val" / src_label.name
        if args.copy:
            if dst_image.exists():
                dst_image.unlink()
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, dst_image)
            mode = "copy"
        else:
            mode = link_file(src_image, dst_image, overwrite=True)
        if args.copy:
            if dst_label.exists():
                dst_label.unlink()
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_label, dst_label)
        else:
            link_file(src_label, dst_label, overwrite=True)
        link_modes[mode] += 1
        manifest.append(
            {
                "physical_group": row["physical_group"],
                "image": src_image.name,
                "sha256": row["sha256"],
                "classes": row["class_names"],
                "instance_count": row["instance_count"],
                "source_image": str(src_image),
                "source_label": str(src_label),
            }
        )

    (out / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out}",
                "val: images/val",
                "nc: 10",
                "names:",
                *[f"- {name}" for name in CLASS_NAMES],
                "",
            ]
        ),
        encoding="utf-8",
    )
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    class_images: Counter[str] = Counter()
    class_instances: Counter[str] = Counter()
    for row in selected:
        names = [name for name in row["class_names"].split(";") if name]
        for name in names:
            class_images[name] += 1
        class_instances.update(names)
    group_counts = Counter(row["physical_group"] for row in selected)
    safe_groups = len(group_counts)
    summary = {
        "audit": str(audit_path),
        "dataset": str(out),
        "evaluation_only": True,
        "selection": "original val groups with no original train member and no train SHA256",
        "images": len(selected),
        "groups": safe_groups,
        "instances": sum(int(row["instance_count"]) for row in selected),
        "class_images": dict(sorted(class_images.items())),
        "class_instances": dict(sorted(class_instances.items())),
        "group_size_histogram": dict(sorted(Counter(group_counts.values()).items())),
        "link_modes": dict(link_modes),
        "missing_classes": [name for name in CLASS_NAMES if class_images[name] == 0],
    }
    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# v25 历史未见 holdout（仅评估）",
        "",
        f"- 审计清单：`{audit_path}`",
        f"- 数据集：`{out}`",
        "- 选择规则：原始 `val` 物理组不得含任何原始 `train` 成员；同时排除出现在原始 train 的 SHA256。",
        "- **用途限制：只用于旧检查点评估，禁止训练、调参或用于选择新模型。**",
        "",
        "## 规模",
        "",
        f"- 图片：{summary['images']} 张",
        f"- 物理组：{summary['groups']} 组",
        f"- GT 实例：{summary['instances']}",
        f"- 唯一 SHA256：{len(set(row['sha256'] for row in selected))}",
        f"- 缺类：{', '.join(summary['missing_classes']) if summary['missing_classes'] else '无'}",
        "",
        "## 每类覆盖",
        "",
        "| 类别 | 含该类图片 | 实例数 |",
        "|---|---:|---:|",
    ]
    for name in CLASS_NAMES:
        lines.append(f"| {name} | {class_images[name]} | {class_instances[name]} |")
    lines.extend(
        [
            "",
            "## 物理组规模",
            "",
            "| 每个物理组图片数 | 组数 |",
            "|---:|---:|",
        ]
    )
    for size, count in sorted(summary["group_size_histogram"].items()):
        lines.append(f"| {size} | {count} |")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 该 holdout 只覆盖原始 val 中与 train 物理隔离的部分；Tuna、Foam_Brick、Mug 的真实历史未见样本不足，不能用它评价这三类。",
            "- 下一阶段可直接用 v18/v21/v22/v23 在同一 holdout 上比较，重点看 Cheez-it、Scissors、Mustard、Clamp、Plastic_Banana、meat_can。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {out}")
    print(f"[saved] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
