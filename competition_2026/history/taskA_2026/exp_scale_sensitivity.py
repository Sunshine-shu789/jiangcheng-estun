#!/usr/bin/env python3
"""实验：物体在网络输入里的尺度偏离训练值多少才会掉点？

做法：把 val 图整体缩小 s 倍后居中放回原尺寸画布（等价于"相机离得更远/视场更宽"，
物体在网络输入里变成 s 倍），标签同步缩放，然后推理 + 按 IoU≥0.75 评估。

配合已知的上侧（imgsz 960 → 1.5×、imgsz 1280 → 2×），可得到完整的尺度敏感度曲线，
用来定真机相机的安装高度、镜头视场和 ROI 尺寸。

用法：python exp_scale_sensitivity.py --weights runs/yolo11n_v2/weights/best.pt --scales 0.5,0.7,0.85
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import cepb


def build_scaled_view(data: Path, out: Path, scale: float) -> Path:
    src_img = data / "images" / "val"
    src_lbl = data / "labels" / "val"
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    images = sorted(src_img.glob("*.png"))
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        nh, nw = int(round(h * scale)), int(round(w * scale))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (64, 64, 64)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = small
        cv2.imwrite(str(out / "images" / "val" / img_path.name), canvas)

        ox, oy = (1.0 - scale) / 2.0, (1.0 - scale) / 2.0
        lines = []
        for line in (src_lbl / f"{img_path.stem}.txt").read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls, cx, cy, bw, bh = p[0], *(float(v) for v in p[1:5])
            lines.append(
                f"{cls} {ox + cx * scale:.6f} {oy + cy * scale:.6f} "
                f"{bw * scale:.6f} {bh * scale:.6f}"
                + (" " + " ".join(p[5:]) if len(p) > 5 else "")
            )
        (out / "labels" / "val" / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    keep = {p.name for p in images}
    with (data / "instances.csv").open(encoding="utf-8") as fh, (
        out / "instances.csv"
    ).open("w", newline="", encoding="utf-8") as fw:
        reader = csv.DictReader(fh)
        writer = csv.DictWriter(fw, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["image"] in keep and row["split"] == "val":
                writer.writerow(row)
    shutil.copy(data / "classes.txt", out / "classes.txt")
    return out


def parse_pass(report: Path) -> dict:
    text = report.read_text(encoding="utf-8")
    out = {}
    for key, pat in (
        ("pass", r"通过率（IoU≥0\.75） \| ([\d.]+)%"),
        ("iou", r"平均 best IoU \| ([\d.]+)"),
        ("recall", r"Recall@IoU0\.5 \| ([\d.]+)%"),
        ("fp", r"误检（FP 框） \| (\d+)"),
    ):
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1)) if key != "fp" else int(m.group(1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(cepb.WORKDIR / "dataset_v2"))
    parser.add_argument("--weights", default=str(cepb.WORKDIR / "runs/yolo11n_v2/weights/best.pt"))
    parser.add_argument("--work", default="/tmp/scale_sensitivity")
    parser.add_argument("--scales", default="0.5,0.7,0.85")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    task_dir = Path(__file__).resolve().parent
    data = Path(args.data).resolve()
    work = Path(args.work).resolve()
    if work.exists():
        shutil.rmtree(work)
    rows = []
    for s in [float(x) for x in args.scales.split(",")]:
        tag = f"s{int(round(s * 100)):03d}"
        ds = work / tag / "ds"
        build_scaled_view(data, ds, s)
        preds = work / tag / "preds"
        subprocess.run(
            [sys.executable, str(task_dir / "infer_imgsz.py"), "--weights", args.weights,
             "--data", str(ds), "--split", "val", "--imgsz", str(args.imgsz), "--out", str(preds)],
            check=True, capture_output=True,
        )
        report = work / tag / "eval"
        subprocess.run(
            [sys.executable, str(task_dir / "eval_detection.py"), "--data", str(ds),
             "--pred-labels", str(preds / "labels"), "--split", "val", "--out", str(report)],
            check=True, capture_output=True,
        )
        stats = parse_pass(report.with_suffix(".md"))
        rows.append((s, stats))
        print(f"物体缩小 {s:.2f}× -> 通过率 {stats.get('pass')}%  IoU {stats.get('iou')}  误检 {stats.get('fp')}", flush=True)
        for sub in ("images", "labels"):
            shutil.rmtree(ds / sub)

    lines = [
        "# 物体尺度的敏感度曲线（决定真机相机怎么装）",
        "",
        "把 val 图整体缩放后居中放回原画布（等价于相机离得更远 / 视场更宽），标签同步缩放。",
        "权重 `runs/yolo11n_v2/weights/best.pt`，imgsz 640，val 900 张 / 9000 实例，IoU≥0.75。",
        "",
        "| 物体在网络输入里的尺度 | 通过率 | 平均 best IoU | Recall@0.5 | 误检 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s, st in rows:
        lines.append(f"| {s:.2f}× | {st.get('pass')}% | {st.get('iou')} | {st.get('recall')}% | {st.get('fp')} |")
    lines += [
        "| 1.00×（训练尺度，基线） | 39.4% | 0.492 | 59.6% | 3124 |",
        "| 1.50×（imgsz 960 推理） | 33.3% | 0.444 | 54.2% | 3362 |",
        "| 2.00×（imgsz 1280 推理） | 4.1% | 0.244 | 27.1% | 5734 |",
        "",
        "> 仿真训练时物体长边在网络输入里是 80–190 px（中位 138 px）。",
        "> 真机装相机时，应让物体的网络输入像素落在这条曲线的平坦区间内。",
        "",
    ]
    text = "\n".join(lines)
    (work / "summary.md").write_text(text, encoding="utf-8")
    print(f"汇总：{work / 'summary.md'}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
