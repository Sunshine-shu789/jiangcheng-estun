#!/usr/bin/env python3
"""Subset 1 全量数据体检：文件结构、类别分布、遮挡与框大小。

产出：
    taskA/reports/dataset_stats.md
    taskA/reports/class_stats.csv
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

import cepb

ROOT = Path(__file__).resolve().parents[1]
REPORTS = cepb.WORKDIR / "reports"


def count_entries(entries) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    pat_gt = re.compile(r"^GT_(left|middle|right)_camera_\d+\.yaml$")
    pat_rgb = re.compile(
        r"^(left|middle|right)_camera_(directional_light|point_light|spot_light)"
        r"_scene_\d+_rgb\.png$"
    )
    pat_depth = re.compile(
        r"^(left|middle|right)_camera_directional_light_scene_\d+_(depth|normals)\.png$"
    )
    pat_seg = re.compile(
        r"^(left|middle|right)_camera_scene_\d+_segmentation\.png$"
    )
    for name in entries:
        m = pat_gt.match(name)
        if m:
            counts[f"GT {m.group(1)}"] += 1
            continue
        m = pat_rgb.match(name)
        if m:
            counts[f"RGB {m.group(1)} / {m.group(2)}"] += 1
            continue
        m = pat_depth.match(name)
        if m:
            counts[f"IMG {m.group(1)} / {m.group(2)}"] += 1
            continue
        if pat_seg.match(name):
            counts["IMG segmentation"] += 1
            continue
        counts["其它"] += 1
    return dict(counts)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    entries = cepb.list_entries()
    counts = count_entries(entries)

    tmp_root = cepb.WORKDIR / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cepb_gt_", dir=str(tmp_root)))
    try:
        print(f"解压 {sum(v for k, v in counts.items() if k.startswith('GT'))} 个 GT yaml 到 {tmp} …")
        subprocess.run(
            ["unzip", "-o", "-j", cepb.ARCHIVE, "GT_*.yaml", "-d", str(tmp)],
            check=True,
            capture_output=True,
        )
        gt_files = sorted(tmp.glob("GT_*.yaml"))
        print(f"开始解析 {len(gt_files)} 个 GT yaml …")

        class_count: Dict[str, int] = defaultdict(int)
        class_vis: Dict[str, List[float]] = defaultdict(list)
        class_area: Dict[str, List[float]] = defaultdict(list)
        class_camera: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        frame_sizes: List[int] = []
        out_of_frame = 0
        parse_errors: List[str] = []

        for idx, path in enumerate(gt_files, 1):
            try:
                data = yaml.safe_load(path.read_bytes())
            except Exception as exc:  # noqa: BLE001
                parse_errors.append(f"{path.name}: {exc}")
                continue
            match = re.match(r"GT_(left|middle|right)_camera_(\d+)\.yaml", path.name)
            camera = match.group(1) if match else "unknown"
            objects = data.get("objects") or {}
            frame_sizes.append(len(objects))
            for name, raw in objects.items():
                visibility = raw.get("visibility")
                if isinstance(visibility, (list, tuple)):
                    visibility = float(visibility[0]) if visibility else 0.0
                else:
                    visibility = float(visibility or 0.0)
                points = [
                    (float(x), float(cepb.IMAGE_HEIGHT - y))
                    for x, y in raw["projected_cuboid"]
                ]
                box = cepb.aabb_of_polygon(points)
                area = cepb.aabb_area(box)
                class_count[name] += 1
                class_vis[name].append(visibility)
                class_area[name].append(area)
                class_camera[name][camera] += 1
                if any(
                    x < 0 or x > cepb.IMAGE_WIDTH or y < 0 or y > cepb.IMAGE_HEIGHT
                    for x, y in points
                ):
                    out_of_frame += 1
            if idx % 3000 == 0:
                print(f"  {idx}/{len(gt_files)}")

        total_objects = sum(class_count.values())
        frame_arr = np.asarray(frame_sizes)
        full_area = cepb.IMAGE_WIDTH * cepb.IMAGE_HEIGHT

        rows: List[Dict[str, object]] = []
        for name in sorted(class_count, key=lambda n: (-class_count[n], n)):
            vis = np.asarray(class_vis[name])
            area = np.asarray(class_area[name])
            rows.append(
                {
                    "class": name,
                    "instances": class_count[name],
                    "visibility_median": float(np.median(vis)),
                    "visibility_p10": float(np.percentile(vis, 10)),
                    "occluded_lt_0.5": int((vis < 0.5).sum()),
                    "box_area_median_px": float(np.median(area)),
                    "box_coverage_median": float(np.median(area) / full_area),
                    "left": class_camera[name].get("left", 0),
                    "middle": class_camera[name].get("middle", 0),
                    "right": class_camera[name].get("right", 0),
                }
            )

        csv_path = REPORTS / "class_stats.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        report: List[str] = [
            "# Subset 1 数据体检报告",
            "",
            f"- 数据包：`{cepb.ARCHIVE}`",
            f"- 压缩包条目：{len(entries)}",
            f"- GT 帧数：{len(gt_files)}（应为 5000 场景 × 3 相机）",
            f"- 目标实例总数：{total_objects}",
            f"- 每帧目标数：min={frame_arr.min()}，中位={np.median(frame_arr):.0f}，max={frame_arr.max()}",
            f"- 投影点越出画面的实例：{out_of_frame} / {total_objects}",
            f"- 解析失败：{len(parse_errors)}",
            "",
            "## 文件结构",
            "",
            "| 类型 | 数量 |",
            "| --- | --- |",
        ]
        for key in sorted(counts):
            report.append(f"| {key} | {counts[key]} |")
        report += [
            "",
            "## 类别统计",
            "",
            "| 类别 | 实例数 | 可见度中位 | 可见度 P10 | 遮挡<0.5 | 框面积中位(px²) | 框占画面中位 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            report.append(
                f"| {row['class']} | {row['instances']} | {row['visibility_median']:.2f} | "
                f"{row['visibility_p10']:.2f} | {row['occluded_lt_0.5']} | "
                f"{row['box_area_median_px']:.0f} | {row['box_coverage_median'] * 100:.2f}% |"
            )
        report += [
            "",
            "## 结论与注意",
            "",
            "- 每个场景的每台相机有一份 GT；三种光照共用同一份 GT。",
            "- 训练时可把 3 种光照的 RGB 都配上同一台相机的 GT，等于 15,000 → 45,000 张带标注图。",
            "- 可见度低的目标遮挡严重，任务 A 的 IoU 评估里要决定是否计入、或按可见度分层报告。",
            "- 投影点越出画面属于正常现象（物体贴在图像边缘），评估时要先把框裁到画面内。",
            "",
        ]
        if parse_errors:
            report.append("### 解析失败")
            report += [f"- {err}" for err in parse_errors[:20]]
            report.append("")

        out = REPORTS / "dataset_stats.md"
        out.write_text("\n".join(report), encoding="utf-8")
        print("\n".join(report[:40]))
        print(f"\n报告：{out}\nCSV：{csv_path}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
