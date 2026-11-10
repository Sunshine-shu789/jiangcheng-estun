#!/usr/bin/env python3
"""Scan existing 10-class checkpoints on existing labels for mug/tuna/scissors.

No images are captured or relabelled.  The script separates classification errors
from localisation errors and evaluates 9100-style multi-scale inference.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TARGETS = (1, 2, 8)
CLASS_NAMES = {
    0: "Cheez-it",
    1: "Tuna",
    2: "Scissors",
    3: "Mustard",
    4: "Tomato_Soup",
    5: "Foam_Brick",
    6: "Clamp",
    7: "Plastic_Banana",
    8: "Mug",
    9: "meat_can",
}


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def prepare(frame: np.ndarray, scale: float) -> tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    sw, sh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((h, w, 3), 114, np.uint8)
    xo, yo = (w - sw) // 2, (h - sh) // 2
    canvas[yo:yo + sh, xo:xo + sw] = cv2.resize(
        frame, (sw, sh), interpolation=cv2.INTER_AREA
    )
    return canvas, xo, yo


def nms(dets, threshold=0.55):
    kept = []
    for det in sorted(dets, key=lambda item: item[1], reverse=True):
        if all(det[0] != other[0] or iou(det[2], other[2]) < threshold for other in kept):
            kept.append(det)
    return kept


def labels_for(image: Path, root: Path):
    parts = list(image.parts)
    split = image.parent.name
    label = root / "labels" / split / f"{image.stem}.txt"
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError(f"cannot read {image}")
    h, w = frame.shape[:2]
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
        out.append((int(cls), ((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h)))
    return frame, out


def predict_all(model, images, scales, conf, batch):
    per_image = [[] for _ in images]
    for scale in scales:
        prepared = [prepare(image, scale) for image in images]
        for start in range(0, len(images), batch):
            chunk = prepared[start:start+batch]
            sources = [item[0] for item in chunk]
            results = model.predict(
                source=sources, imgsz=640, conf=conf, iou=0.7,
                max_det=300, device=0, verbose=False,
            )
            for idx, result in enumerate(results, start=start):
                _, xo, yo = chunk[idx-start]
                boxes = result.boxes
                if boxes is None:
                    continue
                for box, cls, score in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    per_image[idx].append((
                        int(cls), float(score),
                        ((x1-xo)/scale, (y1-yo)/scale, (x2-xo)/scale, (y2-yo)/scale),
                    ))
    return [nms(dets) for dets in per_image]


def evaluate(per_image, gts_per_image, thresholds, match_ious):
    rows = []
    for threshold in thresholds:
        for match_iou in match_ious:
            tp = Counter(); fp = Counter(); fn = Counter(); confused = defaultdict(Counter)
            best_ious = defaultdict(list)
            for dets, gts in zip(per_image, gts_per_image):
                dets = [d for d in dets if d[1] >= threshold]
                matched = set()
                for cls, conf, box in sorted(dets, key=lambda item: item[1], reverse=True):
                    best_idx, best_iou = -1, 0.0
                    for idx, (gt_cls, gt_box) in enumerate(gts):
                        if idx in matched or gt_cls != cls:
                            continue
                        value = iou(box, gt_box)
                        if value > best_iou:
                            best_idx, best_iou = idx, value
                    if best_idx >= 0 and best_iou >= match_iou:
                        matched.add(best_idx); tp[cls] += 1
                    elif cls in TARGETS:
                        fp[cls] += 1
                for idx, (gt_cls, gt_box) in enumerate(gts):
                    if gt_cls not in TARGETS:
                        continue
                    if idx in matched:
                        best_ious[gt_cls].append(1.0)
                        continue
                    fn[gt_cls] += 1
                    choices = []
                    for cls, conf, box in dets:
                        value = iou(box, gt_box)
                        if value > 0:
                            choices.append((value, cls, conf))
                    if choices:
                        value, cls, conf = max(choices)
                        confused[gt_cls][CLASS_NAMES.get(cls, str(cls))] += 1
                        best_ious[gt_cls].append(value)
                    else:
                        confused[gt_cls]["missing"] += 1
                        best_ious[gt_cls].append(0.0)
            for cls in TARGETS:
                p = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] else 1.0
                r = tp[cls] / (tp[cls] + fn[cls]) if tp[cls] + fn[cls] else 1.0
                values = best_ious[cls]
                rows.append({
                    "threshold": threshold,
                    "match_iou": match_iou,
                    "class_id": cls,
                    "class": CLASS_NAMES[cls],
                    "tp": tp[cls],
                    "fp": fp[cls],
                    "fn": fn[cls],
                    "precision": p,
                    "recall": r,
                    "mean_best_iou": float(np.mean(values)) if values else 0.0,
                    "median_best_iou": float(np.median(values)) if values else 0.0,
                    "iou75_hits": sum(v >= 0.75 for v in values),
                    "confusion": json.dumps(confused[cls], ensure_ascii=False, sort_keys=True),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="+")
    parser.add_argument("--data", default="/media/sunshine/新加卷/taskA/datasets/real_ft_v22_20261016")
    parser.add_argument("--scales", default="1.0,0.85")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--thresholds", default="0.15,0.25,0.35")
    parser.add_argument("--match-ious", default="0.3,0.5,0.75")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.data)
    images = sorted((root / "images/val").glob("*.jpg"))
    loaded = [labels_for(path, root) for path in images]
    frames = [item[0] for item in loaded]
    gts = [item[1] for item in loaded]
    scales = tuple(float(v) for v in args.scales.split(","))
    thresholds = tuple(float(v) for v in args.thresholds.split(","))
    match_ious = tuple(float(v) for v in args.match_ious.split(","))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for weights in args.weights:
        path = Path(weights)
        if not path.is_file():
            print(f"[skip missing] {path}")
            continue
        print(f"[scan] {path}", flush=True)
        model = YOLO(str(path))
        predictions = predict_all(model, frames, scales, args.conf, args.batch)
        for row in evaluate(predictions, gts, thresholds, match_ious):
            row.update({"model": path.name, "run": path.parent.parent.name, "weights": str(path)})
            all_rows.append(row)
        del model
    fields = [
        "model", "run", "weights", "threshold", "match_iou", "class_id", "class",
        "tp", "fp", "fn", "precision", "recall", "mean_best_iou",
        "median_best_iou", "iou75_hits", "confusion",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"[saved] {out} rows={len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
