#!/usr/bin/env python3
"""Evaluate detectors on the synthetic mesh holdout, with pose/scale breakdown.

Reads manifest.json (az / el / dist per image) and reports, per model:
  - per-class IoU>=0.75 pass rate (competition criterion, category must be right)
  - pass rate bucketed by azimuth, elevation and distance
This isolates "does the model break at extreme pose / small scale" for every
class, including Mug / Tuna which have no historically-unseen real image.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CLASS_NAMES = {
    0: "1-Cheez-it", 1: "2-Starkist_Tuna", 2: "3-Scissors", 3: "4-Frenchs_Mustard",
    4: "5-Tomato_Soup", 5: "6-Foam_Brick", 6: "7-Clamp", 7: "8-Plastic_Banana",
    8: "9-Mug", 9: "10-meat_can",
}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def read_label(path: Path, w: int, h: int):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def el_bucket(el):
    return "el<=20(平视)" if el <= 20 else ("el21-40" if el <= 40 else ("el41-60" if el <= 60 else "el>60(俯视)"))


def dist_bucket(d):
    return "近(d<=0.7)" if d <= 0.70 else ("中(0.7-0.9)" if d <= 0.90 else ("远(0.9-1.2)" if d <= 1.2 else "很远(d>1.2)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    root = Path(args.data).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    img_dir = root / "images" / "val"
    lbl_dir = root / "labels" / "val"

    frames, metas = [], []
    for rec in manifest:
        im = cv2.imread(str(img_dir / rec["image"]))
        if im is None:
            continue
        h, w = im.shape[:2]
        frames.append(im)
        gts = read_label(lbl_dir / (Path(rec["image"]).stem + ".txt"), w, h)
        metas.append({**rec, "w": w, "h": h, "gt": gts})

    rows = []
    for weight in args.weights:
        wp = Path(weight)
        if not wp.is_file():
            print(f"[skip] {wp}")
            continue
        name = wp.parent.parent.name
        print(f"[eval] {name}", flush=True)
        model = YOLO(str(wp))
        results = []
        for i in range(0, len(frames), args.batch):
            chunk = frames[i:i + args.batch]
            results.extend(model.predict(chunk, imgsz=args.imgsz, conf=args.conf,
                                         verbose=False, device=args.device or None))
        for rec, res in zip(metas, results):
            dets = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                cf = res.boxes.conf.cpu().numpy()
                dets = list(zip(cls.tolist(), cf.tolist(), xyxy.tolist()))
            for gi, (gc, gbox) in enumerate(rec["gt"]):
                best, bestc, bestconf = 0.0, None, 0.0
                for dc, dconf, dbox in dets:
                    v = iou(gbox, dbox)
                    if v > best:
                        best, bestc, bestconf = v, dc, dconf
                rows.append({
                    "model": name, "image": rec["image"], "class": CLASS_NAMES[gc],
                    "class_id": gc, "az": rec["az"], "el": rec["el"], "dist": rec["dist"],
                    "az_bucket": f"az{int(rec['az'])}", "el_bucket": el_bucket(rec["el"]),
                    "dist_bucket": dist_bucket(rec["dist"]),
                    "best_iou": round(best, 6), "pred_class": CLASS_NAMES.get(bestc, "none"),
                    "correct": int(bestc == gc), "pass_iou75": int(best >= 0.75 and bestc == gc),
                    "pass_iou50": int(best >= 0.50 and bestc == gc), "pred_conf": round(float(bestconf), 4),
                })
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    models = sorted({r["model"] for r in rows})
    classes = [CLASS_NAMES[i] for i in range(10)]
    lines = ["# 合成网格 holdout：角度/距离鲁棒性评估", "",
             f"- 数据：`{root}`（合成，仅评估，不进训练）",
             f"- imgsz={args.imgsz}，入口 conf={args.conf}，IoU75 且类别正确 = 达标", "",
             "## 每类 IoU75 通过率", "",
             "| 模型 | " + " | ".join(c.split("-")[0] for c in classes) + " | 总体 |",
             "|---|" + "---:|" * 11]
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        cells, tot_gt, tot_pass = [], 0, 0
        for c in classes:
            cr = [r for r in mr if r["class"] == c]
            g, p = len(cr), sum(r["pass_iou75"] for r in cr)
            tot_gt += g; tot_pass += p
            cells.append(f"{p}/{g}" if g else "-")
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {tot_pass}/{tot_gt} ({tot_pass/max(1,tot_gt):.2f}) |")

    for dim, key in (("仰角", "el_bucket"), ("距离", "dist_bucket")):
        lines += ["", f"## 按{dim}拆分（IoU75 通过率）", "",
                  "| 模型 | 桶 | 通过/GT | 通过率 |", "|---|---|---:|---:|"]
        for m in models:
            buckets = sorted({r[key] for r in rows if r["model"] == m})
            for b in buckets:
                br = [r for r in rows if r["model"] == m and r[key] == b]
                lines.append(f"| {m} | {b} | {sum(r['pass_iou75'] for r in br)}/{len(br)} | "
                             f"{sum(r['pass_iou75'] for r in br)/max(1,len(br)):.3f} |")

    lines += ["", "## 最易失败的角度（每模型最差 az）", "", "| 模型 | az | 通过/GT | 通过率 |", "|---|---|---:|---:|"]
    for m in models:
        azs = sorted({r["az_bucket"] for r in rows if r["model"] == m})
        stats = []
        for a in azs:
            ar = [r for r in rows if r["model"] == m and r["az_bucket"] == a]
            stats.append((sum(r["pass_iou75"] for r in ar) / max(1, len(ar)), a, sum(r["pass_iou75"] for r in ar), len(ar)))
        for rate, a, p, g in sorted(stats)[:3]:
            lines.append(f"| {m} | {a} | {p}/{g} | {rate:.3f} |")

    lines += ["", "## 说明与边界", "",
              "- 合成贴图来自比赛官方 mesh，背景是真实空桌照片；用于探测**结构性的角度/距离**弱点。",
              "- 合成→真实存在域差，绝对数值不能当作真实场景识别率；真实数值仍需新拍的、物理隔离的照片集。",
              "- 本次仅离线评估：未训练、未替换 `camera_recognition.py` 权重、未控制机械臂/机械爪。", ""]
    Path(args.summary).write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {args.out_csv}")
    print(f"[saved] {args.summary}")


if __name__ == "__main__":
    raise SystemExit(main())
