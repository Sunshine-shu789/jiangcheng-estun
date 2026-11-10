#!/usr/bin/env python3
"""v22 汤罐专项复测：14 张姿态留出 + 32 张负样本留出。

推理口径与服务一致：0.75/0.65 两次缩放居中贴到 640x480，imgsz=640。
命中: class4 与手工框 IoU>=0.3，或框中心落在手工框内。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

T = Path("/media/sunshine/新加卷/taskA")
RAW_SOUP = T / "real_raw/5-Tomato_Soup"
RAW_EMPTY = T / "real_raw/empty_background"
BOXJSON = Path("/home/sunshine/codex/easton/taskA/prelabels_soup/manual_boxes_soup.json")
MANIFEST = Path("/home/sunshine/codex/easton/taskA/scratch/v22_build_manifest_20261016.json")
GROUPS_JSON = Path("/home/sunshine/codex/easton/taskA/scratch/soup_groups.json")
SCALES = (0.75, 0.65)
CLASS_NAMES = [
    "1-Cheez-it", "2-Starkist_Tuna", "3-Scissors", "4-Frenchs_Mustard",
    "5-Tomato_Soup", "6-Foam_Brick", "7-Clamp", "8-Plastic_Banana",
    "9-Mug", "10-meat_can",
]


def prep(frame: np.ndarray, scale: float) -> tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    sw, sh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((h, w, 3), 114, np.uint8)
    xo, yo = (w - sw) // 2, (h - sh) // 2
    canvas[yo:yo + sh, xo:xo + sw] = cv2.resize(
        frame, (sw, sh), interpolation=cv2.INTER_AREA
    )
    return canvas, xo, yo


def predict_boxes(model: YOLO, frame: np.ndarray, conf: float) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    detections: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for scale in SCALES:
        canvas, xo, yo = prep(frame, scale)
        result = model.predict(
            source=canvas, imgsz=640, conf=conf, device=0, verbose=False
        )[0]
        for raw_box, raw_class, raw_conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = [float(v) for v in raw_box]
            detections.append(
                (
                    int(raw_class),
                    float(raw_conf),
                    (
                        max(0.0, (x1 - xo) / scale),
                        max(0.0, (y1 - yo) / scale),
                        min(640.0, (x2 - xo) / scale),
                        min(480.0, (y2 - yo) / scale),
                    ),
                )
            )
    return detections


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def hit(det: tuple[float, float, float, float], gt: tuple[float, float, float, float]) -> bool:
    if iou(det, gt) >= 0.3:
        return True
    cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
    return gt[0] < cx < gt[2] and gt[1] < cy < gt[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("--out", default="")
    parser.add_argument("--raw-conf", type=float, default=0.03)
    parser.add_argument("--thresholds", default="0.20,0.25,0.35")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")
    thresholds = [float(v) for v in args.thresholds.split(",")]
    boxes = json.loads(BOXJSON.read_text(encoding="utf-8"))
    groups = json.loads(GROUPS_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    soup_val = manifest["new_soup_val"]
    neg_holdout = manifest["negative_holdout"]

    model = YOLO(str(weights))
    soup_dets: dict[str, list[tuple[int, float, tuple[float, float, float, float]]]] = {}
    for name in soup_val:
        frame = cv2.imread(str(RAW_SOUP / name))
        if frame is None:
            raise SystemExit(f"cannot read {name}")
        soup_dets[name] = predict_boxes(model, frame, args.raw_conf)

    neg_dets: dict[str, list[tuple[int, float, tuple[float, float, float, float]]]] = {}
    for name in neg_holdout:
        frame = cv2.imread(str(RAW_EMPTY / name))
        if frame is None:
            raise SystemExit(f"cannot read {name}")
        neg_dets[name] = predict_boxes(model, frame, args.raw_conf)

    pose_of = {name: group for group, names in groups.items() for name in names}
    lines = [f"weights={weights}", f"scales={SCALES}", f"raw_conf={args.raw_conf}"]
    for threshold in thresholds:
        hits = 0
        per_pose = []
        for name in soup_val:
            gts = [tuple(box[1:]) for box in boxes[name] if box[0] == 4]
            dets = [d for d in soup_dets[name] if d[0] == 4 and d[1] >= threshold]
            best = max((d[1] for d in dets), default=0.0)
            ok = any(hit(d[2], gt) for d in dets for gt in gts)
            hits += int(ok)
            per_pose.append((pose_of[name], "命中" if ok else "漏检", best))
        neg_hit = 0
        neg_class = Counter()
        neg_max = 0.0
        for name in neg_holdout:
            dets = [d for d in neg_dets[name] if d[1] >= threshold]
            if dets:
                neg_hit += 1
                neg_max = max(neg_max, max(d[1] for d in dets))
                for d in dets:
                    neg_class[CLASS_NAMES[d[0]]] += 1
        lines.append(
            f"[conf {threshold:.2f}] soup {hits}/{len(soup_val)}; "
            f"negative FP {neg_hit}/{len(neg_holdout)}"
            + (f" max={neg_max:.2f}" if neg_hit else "")
        )
        lines.extend(
            f"  {pose:22s} {status:4s} best={best:.3f}"
            for pose, status, best in per_pose
        )

    # 全量低阈值明细，便于查误报来源。
    for threshold in thresholds:
        lines.append(f"[negative detail conf {threshold:.2f}]")
        for name in neg_holdout:
            dets = sorted(
                (d for d in neg_dets[name] if d[1] >= threshold),
                key=lambda d: d[1],
                reverse=True,
            )
            if dets:
                lines.append(
                    "  " + name + " " + ", ".join(
                        f"{CLASS_NAMES[d[0]]}={d[1]:.3f}" for d in dets[:4]
                    )
                )

    report = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else weights.parent / "soup_eval.txt"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
