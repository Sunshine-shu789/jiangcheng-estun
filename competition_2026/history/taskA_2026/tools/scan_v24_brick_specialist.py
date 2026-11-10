#!/usr/bin/env python3
"""扫描泡沫砖专家模型 checkpoints，按目标姿态与误报门禁选模型。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

T = Path("/media/sunshine/新加卷/taskA")
SCENES = Path("/home/sunshine/codex/easton/taskA/datasets/random_real_scenes_near_supp_20261013")
BRICK_BOXES = Path("/home/sunshine/codex/easton/taskA/prelabels_v21/manual_boxes_v2.json")
BRICK_RAW = T / "real_raw/6-Foam_Brick"
EMPTY = T / "real_raw/empty_background"
EMPTY_HOLDOUT = Path("/home/sunshine/codex/easton/taskA/scratch/v21_empty_holdout_20261016.txt")
SPECIALIST_MANIFEST = Path("/home/sunshine/codex/easton/taskA/scratch/v24_brick_specialist_manifest_20261016.json")
V22_VAL_IMAGES = T / "datasets/real_ft_v22_20261016/images/val"
V22_VAL_LABELS = T / "datasets/real_ft_v22_20261016/labels/val"
# 专家模型部署时使用原尺寸和轻微缩放；0.65/0.75 会把远处小目标压得过小。
SCALES = (1.0, 0.85)
CLASS_FOAM = 5
VAL_TAGS = {
    "015559_273953", "015652_181086", "015946_992055", "020108_464841",
    "020158_025628", "020244_213967", "021652_180906", "021736_469985",
    "022141_233823", "022314_004847",
}


def prep(frame: np.ndarray, scale: float) -> tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    sw, sh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((h, w, 3), 114, np.uint8)
    xo, yo = (w - sw) // 2, (h - sh) // 2
    canvas[yo:yo + sh, xo:xo + sw] = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    return canvas, xo, yo


def predict_boxes(model, frame: np.ndarray):
    out = []
    for scale in SCALES:
        canvas, xo, yo = prep(frame, scale)
        result = model.predict(source=canvas, imgsz=640, conf=0.03, device=0, verbose=False)[0]
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            out.append((cls, conf, ((x1 - xo) / scale, (y1 - yo) / scale,
                                    (x2 - xo) / scale, (y2 - yo) / scale)))
    return out


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def hit_box(det, gt) -> bool:
    cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
    return iou(det, gt) >= 0.3 or (gt[0] < cx < gt[2] and gt[1] < cy < gt[3])


def load_new_holdout():
    manifest = json.loads(SPECIALIST_MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for stem in manifest["holdout"]:
        image = cv2.imread(str(SCENES / "images" / f"{stem}.jpg"))
        if image is None:
            raise RuntimeError(f"missing new holdout image: {stem}")
        line = (SCENES / "labels" / f"{stem}.txt").read_text(encoding="utf-8").strip().split()
        if int(line[0]) != CLASS_FOAM:
            raise RuntimeError(f"unexpected class in {stem}")
        _, x, y, w, h = map(float, line)
        ih, iw = image.shape[:2]
        gt = ((x - w / 2) * iw, (y - h / 2) * ih,
              (x + w / 2) * iw, (y + h / 2) * ih)
        rows.append((stem, image, gt))
    return rows


def old_brick_metrics(model, threshold: float):
    boxes = json.loads(BRICK_BOXES.read_text(encoding="utf-8"))
    results = {}
    for image_path in sorted(BRICK_RAW.glob("*.jpg")):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        gts = [tuple(box[1:]) for box in boxes.get(image_path.name, []) if int(box[0]) == CLASS_FOAM]
        if not gts:
            continue
        dets = [d for d in predict_boxes(model, frame)
                if d[0] == CLASS_FOAM and d[1] >= threshold]
        ok = any(hit_box(d[2], gt) for d in dets for gt in gts)
        tag = image_path.stem[-13:]
        results[tag] = {"hit": bool(ok), "best_conf": max((d[1] for d in dets), default=0.0)}
    old_hits = sum(v["hit"] for v in results.values())
    old_val_hits = sum(v["hit"] for tag, v in results.items() if tag in VAL_TAGS)
    return old_hits, old_val_hits, results


def negative_metrics(model, threshold: float):
    hold_names = {line.strip() for line in EMPTY_HOLDOUT.read_text(encoding="utf-8").splitlines() if line.strip()}
    rows = []
    for image_path in sorted(EMPTY.glob("*.jpg")):
        frame = cv2.imread(str(image_path))
        dets = [d for d in predict_boxes(model, frame)
                if d[0] == CLASS_FOAM and d[1] >= threshold]
        best = max((d[1] for d in dets), default=0.0)
        rows.append((image_path.name, image_path.name in hold_names, bool(dets), best))
    fp_hold = sum(1 for _, hold, fp, _ in rows if hold and fp)
    fp_train = sum(1 for _, hold, fp, _ in rows if not hold and fp)
    fp_all = sum(1 for _, _, fp, _ in rows if fp)
    return fp_all, fp_hold, fp_train, rows


def other_class_negative_metrics(model, threshold: float):
    """检查非泡沫砖的旧类图片是否被专家模型误报成 class5。

    这些图不参与专家模型的梯度更新，但专门覆盖香蕉、罐头、剪刀、芥末等
    容易与浅色长方体混淆的物品。
    """
    rows = []
    for image_path in sorted(V22_VAL_IMAGES.glob("*.jpg")):
        label_path = V22_VAL_LABELS / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise RuntimeError(f"missing validation label: {label_path}")
        gt_classes = {
            int(line.split()[0])
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if CLASS_FOAM in gt_classes:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"failed to read validation image: {image_path}")
        dets = [
            d for d in predict_boxes(model, frame)
            if d[0] == CLASS_FOAM and d[1] >= threshold
        ]
        best = max((d[1] for d in dets), default=0.0)
        rows.append((image_path.name, bool(dets), best))
    fp = sum(1 for _, is_fp, _ in rows if is_fp)
    worst = max((conf for _, is_fp, conf in rows if is_fp), default=0.0)
    return fp, len(rows), worst, rows


def evaluate(model_path: Path, threshold: float):
    model = YOLO(str(model_path))
    new_rows = load_new_holdout()
    new_hits = 0
    new_conf = []
    for stem, frame, gt in new_rows:
        dets = [d for d in predict_boxes(model, frame)
                if d[0] == CLASS_FOAM and d[1] >= threshold]
        ok = any(hit_box(d[2], gt) for d in dets)
        new_hits += int(ok)
        new_conf.append(max((d[1] for d in dets), default=0.0))
    old_hits, old_val_hits, old_details = old_brick_metrics(model, threshold)
    fp_all, fp_hold, fp_train, neg_rows = negative_metrics(model, threshold)
    other_fp, other_total, other_worst, _ = other_class_negative_metrics(model, threshold)
    worst_fp = max((conf for _, _, fp, conf in neg_rows if fp), default=0.0)
    del model
    return {
        "checkpoint": model_path.name,
        "new_hits": new_hits,
        "new_total": len(new_rows),
        "new_min_conf": min(new_conf),
        "new_max_conf": max(new_conf),
        "old_hits": old_hits,
        "old_total": len(old_details),
        "old_val_hits": old_val_hits,
        "old_val_total": len(VAL_TAGS),
        "empty_fp": fp_all,
        "empty_hold_fp": fp_hold,
        "empty_train_fp": fp_train,
        "empty_total": len(neg_rows),
        "worst_empty_conf": worst_fp,
        "other_fp": other_fp,
        "other_total": other_total,
        "other_worst_conf": other_worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    run = Path(args.run_dir)
    checkpoints = sorted(run.glob("weights/epoch*.pt"), key=lambda p: int(p.stem[5:]))
    if (run / "weights/last.pt").is_file():
        checkpoints.append(run / "weights/last.pt")
    if not checkpoints:
        raise SystemExit("no checkpoints")
    rows = []
    for checkpoint in checkpoints:
        row = evaluate(checkpoint, args.threshold)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    fieldnames = list(rows[0])
    if args.out:
        with Path(args.out).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
