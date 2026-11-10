#!/usr/bin/env python3
"""按 9100 服务的双尺度口径统计验证集对象级 TP/FP/FN。"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCALES = (0.65, 0.75)


def prep(frame, scale):
    h, w = frame.shape[:2]
    sw, sh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((h, w, 3), 114, np.uint8)
    xo, yo = (w - sw) // 2, (h - sh) // 2
    canvas[yo:yo + sh, xo:xo + sw] = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    return canvas, xo, yo


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def predict(model, frame, conf):
    out = []
    for scale in SCALES:
        canvas, xo, yo = prep(frame, scale)
        result = model.predict(source=canvas, imgsz=640, conf=conf, device=0, verbose=False)[0]
        for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (float(v) for v in box)
            out.append((int(cls), float(score), ((x1 - xo) / scale, (y1 - yo) / scale, (x2 - xo) / scale, (y2 - yo) / scale)))
    return out


def nms(dets, threshold=0.55):
    kept = []
    for det in sorted(dets, key=lambda item: item[1], reverse=True):
        if all(det[0] != other[0] or iou(det[2], other[2]) < threshold for other in kept):
            kept.append(det)
    return kept


def labels_for(path):
    h, w = cv2.imread(str(path)).shape[:2]
    label = path.parents[2] / "labels" / path.parent.name / f"{path.stem}.txt"
    out = []
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
        out.append((int(cls), ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def score(preds, gts, threshold):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    matched = set()
    for cls, conf, box in sorted(nms(preds), key=lambda item: item[1], reverse=True):
        if conf < threshold:
            continue
        best_idx, best_iou = -1, 0.0
        for idx, (gt_cls, gt_box) in enumerate(gts):
            if idx in matched or gt_cls != cls:
                continue
            value = iou(box, gt_box)
            if value > best_iou:
                best_idx, best_iou = idx, value
        if best_idx >= 0 and best_iou >= 0.3:
            matched.add(best_idx)
            tp[cls] += 1
        else:
            fp[cls] += 1
    for idx, (cls, _) in enumerate(gts):
        if idx not in matched:
            fn[cls] += 1
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("--data", default="/media/sunshine/新加卷/taskA/datasets/real_ft_v22_20261016")
    parser.add_argument("--thresholds", default="0.25,0.35")
    parser.add_argument("--raw-conf", type=float, default=0.05)
    args = parser.parse_args()
    root = Path(args.data)
    images = sorted((root / "images/val").glob("*.jpg"))
    model = YOLO(args.weights)
    cache = [(path, labels_for(path), predict(model, cv2.imread(str(path)), args.raw_conf)) for path in images]
    print(f"weights={args.weights} images={len(images)} scales={SCALES}")
    for threshold in (float(v) for v in args.thresholds.split(",")):
        total_tp = defaultdict(int); total_fp = defaultdict(int); total_fn = defaultdict(int)
        for _, gts, preds in cache:
            tp, fp, fn = score(preds, gts, threshold)
            for cls in set(tp) | set(fp) | set(fn):
                total_tp[cls] += tp[cls]; total_fp[cls] += fp[cls]; total_fn[cls] += fn[cls]
        print(f"[{threshold:.2f}]")
        for cls in range(10):
            tp, fp, fn = total_tp[cls], total_fp[cls], total_fn[cls]
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 1.0
            print(f"  {cls:2d} TP={tp:3d} FP={fp:3d} FN={fn:3d} P={precision:.3f} R={recall:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
