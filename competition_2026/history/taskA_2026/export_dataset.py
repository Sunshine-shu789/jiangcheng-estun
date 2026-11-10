#!/usr/bin/env python3
"""把 CEPB Subset 1 的一个切片导出成 YOLO 检测数据集。

默认：500 个场景（1..5000 等距抽）× left 相机 × 3 种光照 = 1500 张带标注图。
train/val 按场景 8:2 划分，同一场景的 3 种光照只会出现在同一侧。

产物目录（默认 /media/sunshine/新加卷/taskA/dataset_v1）：
    images/train|val/*.png
    labels/train|val/*.txt        YOLO 格式: cls cx cy w h（归一化）
    classes.txt                   类别表（10 类）
    data.yaml                     ultralytics 数据配置
    instances.csv                 每个目标的可见度/框/是否越界
    export_report.md
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

import cepb


def build_scenes(total: int, count: int) -> List[int]:
    """从 1..total 里等距抽 count 个场景号。"""
    if count > total:
        raise ValueError(f"最多 {total} 个场景")
    step = total / count
    scenes = [int(round(1 + i * step)) for i in range(count)]
    return sorted(set(scenes))


def class_id_from_name(name: str) -> int:
    match = re.match(r"^(\d+)[-_]", name)
    if not match:
        raise ValueError(f"物体名没有编号前缀：{name}")
    return int(match.group(1))


def yolo_line(class_id: int, box: Tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / cepb.IMAGE_WIDTH
    cy = (y1 + y2) / 2 / cepb.IMAGE_HEIGHT
    w = (x2 - x1) / cepb.IMAGE_WIDTH
    h = (y2 - y1) / cepb.IMAGE_HEIGHT
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def clip_box(box: Tuple[float, float, float, float]):
    x1 = min(max(box[0], 0.0), float(cepb.IMAGE_WIDTH))
    y1 = min(max(box[1], 0.0), float(cepb.IMAGE_HEIGHT))
    x2 = min(max(box[2], 0.0), float(cepb.IMAGE_WIDTH))
    y2 = min(max(box[3], 0.0), float(cepb.IMAGE_HEIGHT))
    return x1, y1, x2, y2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=500, help="抽多少个场景")
    parser.add_argument(
        "--camera",
        default="left",
        choices=list(cepb.CAMERAS) + ["all"],
        help="单个相机名，或 all（left/middle/right 三个都导出）",
    )
    parser.add_argument("--val-every", type=int, default=5, help="每 N 个场景取 1 个做 val")
    parser.add_argument("--out", default=str(cepb.WORKDIR / "dataset_v1"))
    args = parser.parse_args()

    out = Path(args.out)
    scenes = build_scenes(5000, args.scenes)
    val_scenes = set(scenes[:: args.val_every])
    cameras = list(cepb.CAMERAS) if args.camera == "all" else [args.camera]
    print(f"场景 {len(scenes)} 个（val {len(val_scenes)} / train {len(scenes) - len(val_scenes)}）")
    print(
        f"相机 {cameras}，光照 {cepb.LIGHTS}，"
        f"图片 {len(scenes) * len(cameras) * len(cepb.LIGHTS)} 张"
    )

    names: List[str] = []
    for camera in cameras:
        for scene in scenes:
            names.append(cepb.gt_name(camera, scene))
            for light in cepb.LIGHTS:
                names.append(cepb.rgb_name(camera, light, scene))

    tmp = Path(tempfile.mkdtemp(prefix="cepb_export_", dir=str(cepb.WORKDIR / "tmp")))
    # 全量（5000 场景 = 60000 个文件）时一次 argv 会超 Linux ARG_MAX（≈2 MB），
    # 因此分批解压：每批 2000 个名字，同一 -d 目录，效果等价。
    batch = 2000
    print(f"解压到 {tmp} …（{len(names)} 个文件，分 {ceil(len(names) / batch)} 批）")
    for i in range(0, len(names), batch):
        subprocess.run(
            ["unzip", "-o", "-j", "-d", str(tmp), cepb.ARCHIVE] + names[i : i + batch],
            check=True,
            capture_output=True,
        )
        print(f"  已解压 {min(i + batch, len(names))}/{len(names)}", flush=True)
    missing = [n for n in names if not (tmp / n).exists()]
    if missing:
        raise RuntimeError(f"解压后缺少 {len(missing)} 个文件，例如 {missing[:3]}")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    class_names: Dict[int, str] = {}
    class_counter: Counter = Counter()
    split_counter: Counter = Counter()
    clipped_counter = 0
    skipped = 0

    for scene in scenes:
        split = "val" if scene in val_scenes else "train"
        for camera in cameras:
            frame = cepb.parse_gt_frame(
                camera, scene, (tmp / cepb.gt_name(camera, scene)).read_bytes()
            )
            templates: List[Tuple[int, str, Tuple[float, float, float, float], bool, float]] = []
            for obj in frame.objects:
                class_id = class_id_from_name(obj.name)
                class_names[class_id] = obj.name
                box_raw = obj.aabb
                box = clip_box(box_raw)
                width, height = box[2] - box[0], box[3] - box[1]
                if width < 1 or height < 1:
                    skipped += 1
                    continue
                clipped = box != tuple(box_raw)
                templates.append((class_id, obj.name, box, clipped, obj.visibility))
            labels = [yolo_line(cid - 1, box) for cid, _, box, _, _ in templates]
            for light in cepb.LIGHTS:
                image_name = cepb.rgb_name(camera, light, scene)
                label_name = image_name.rsplit(".", 1)[0] + ".txt"
                shutil.move(str(tmp / image_name), str(out / "images" / split / image_name))
                (out / "labels" / split / label_name).write_text(
                    "\n".join(labels) + "\n", encoding="utf-8"
                )
                split_counter[split] += 1
                for class_id, class_name, box, clipped, visibility in templates:
                    clipped_counter += int(clipped)
                    class_counter[class_id] += 1
                    rows.append(
                        {
                            "split": split,
                            "scene": scene,
                            "camera": camera,
                            "light": light,
                            "image": image_name,
                            "class_id": class_id - 1,
                            "class_name": class_name,
                            "visibility": f"{visibility:.4f}",
                            "x1": f"{box[0]:.2f}",
                            "y1": f"{box[1]:.2f}",
                            "x2": f"{box[2]:.2f}",
                            "y2": f"{box[3]:.2f}",
                            "area_px": f"{(box[2] - box[0]) * (box[3] - box[1]):.0f}",
                            "clipped": int(clipped),
                        }
                    )

    shutil.rmtree(tmp, ignore_errors=True)

    classes_path = out / "classes.txt"
    classes_path.write_text(
        "\n".join(class_names[i] for i in sorted(class_names)) + "\n", encoding="utf-8"
    )
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out),
                "train": "images/train",
                "val": "images/val",
                "nc": len(class_names),
                "names": [class_names[i] for i in sorted(class_names)],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    csv_path = out / "instances.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    visibility = np.asarray([float(r["visibility"]) for r in rows])
    report = [
        f"# Subset 1 切片导出（{split_counter['train'] + split_counter['val']} 张）",
        "",
        f"- 数据包：`{cepb.ARCHIVE}`",
        f"- 场景：{len(scenes)} 个（等距取自 1..5000）",
        f"- 相机/光照：`{'/'.join(cameras)}` × {len(cepb.LIGHTS)} 种",
        f"- 图片：train {split_counter['train']} / val {split_counter['val']}",
        f"- 目标实例：{len(rows)}（每类统计见表）",
        f"- 裁到画面内的框：{clipped_counter}；裁剪后面积过小被跳过：{skipped}",
        f"- 可见度：中位 {np.median(visibility):.2f}，<0.5 占 {(visibility < 0.5).mean() * 100:.1f}%",
        "",
        "## 类别分布（实例数）",
        "",
        "| class_id | 类别 | 实例数 |",
        "| --- | --- | --- |",
    ]
    for cid in sorted(class_names):
        report.append(f"| {cid - 1} | {class_names[cid]} | {class_counter[cid]} |")
    report += [
        "",
        "## 备注",
        "",
        "- 标签为 AABB（projected_cuboid 的轴对齐外接框）转成的 YOLO 归一化格式。",
        "- 同一场景的 3 种光照共用一份 GT，且被分到同一侧（train/val 按场景划分）。",
        "- `instances.csv` 带每个实例的 visibility 和是否被画面裁剪，后续可按可见度过滤或分层评估。",
        "",
    ]
    (out / "export_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"数据集目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
