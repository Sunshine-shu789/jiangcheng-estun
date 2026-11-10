#!/usr/bin/env python3
"""Fit and cross-validate per-class box calibration from saved detections.

The calibration is applied around each predicted box center:
  w' = sx*w, h' = sy*h, cx' = cx + dx*image_width, cy' = cy + dy*image_height.
It is evaluated with 5-fold cross-validation on the fixed camera set.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

CLASSES = {
    "1-Cheez-it": 0,
    "2-Starkist_Tuna": 1,
    "3-Scissors": 2,
    "4-Frenchs_Mustard": 3,
    "5-Tomato_Soup": 4,
    "6-Foam_Brick": 5,
    "7-Clamp": 6,
    "9-Mug": 8,
    "10-meat_can": 9,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--report-root", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=25000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--detection-source", choices=("base", "tta", "best"), default="best")
    return p.parse_args()


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter) if aa + ab - inter > 0 else 0.0


def load_gt(path: Path, width: int, height: int) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        p = raw.split()
        if len(p) < 5:
            continue
        _, cx, cy, w, h = (float(v) for v in p[:5])
        rows.append([(cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height])
    return rows


def best(dets: list[dict[str, Any]], gt: list[float], class_id: int) -> dict[str, Any] | None:
    candidates = [d for d in dets if int(d["class_id"]) == class_id]
    if not candidates:
        return None
    return max(candidates, key=lambda d: iou(gt, d["xyxy"]))


def calibrated(box: list[float], width: int, height: int, params: tuple[float, float, float, float]) -> list[float]:
    sx, sy, dx, dy = params
    cx = (box[0] + box[2]) / 2 + dx * width
    cy = (box[1] + box[3]) / 2 + dy * height
    w = max(1.0, (box[2] - box[0]) * sx)
    h = max(1.0, (box[3] - box[1]) * sy)
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def score(rows: list[dict[str, Any]], params: tuple[float, float, float, float], threshold: float = 0.75) -> tuple[float, int]:
    vals = []
    for r in rows:
        b = calibrated(r["box"], r["width"], r["height"], params)
        vals.append(iou(b, r["gt"]))
    return sum(vals) / len(vals), sum(v >= threshold for v in vals)


def optimize(rows: list[dict[str, Any]], samples: int, seed: int) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    best_p = (1.0, 1.0, 0.0, 0.0)
    best_key = score(rows, best_p)
    # Broad random search around the existing box geometry.
    for _ in range(samples):
        p = (
            rng.uniform(0.72, 1.28),
            rng.uniform(0.72, 1.28),
            rng.uniform(-0.10, 0.10),
            rng.uniform(-0.10, 0.10),
        )
        s = score(rows, p)
        key = (s[1], s[0])
        if key > (best_key[1], best_key[0]):
            best_p, best_key = p, s
    # Coordinate refinement around the random-search winner.
    step = (0.08, 0.08, 0.03, 0.03)
    p = list(best_p)
    for _ in range(40):
        improved = False
        for j, st in enumerate(step):
            for delta in (-st, st):
                q = p.copy()
                q[j] = min(1.5, max(0.5, q[j] + delta)) if j < 2 else min(0.2, max(-0.2, q[j] + delta))
                s = score(rows, tuple(q))
                if (s[1], s[0]) > (best_key[1], best_key[0]):
                    p, best_key, improved = q, s, True
        if not improved:
            step = tuple(v * 0.5 for v in step)
    return tuple(p)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    all_data: dict[str, list[dict[str, Any]]] = {}
    for label, class_id in CLASSES.items():
        summary_path = args.report_root / f"{label}_predictions_640" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        image_dir = args.report_root.parents[1] / "eval_capture_20261016" / label
        rows = []
        for row in summary["rows"]:
            image_path = image_dir / row["file"]
            with Image.open(image_path) as im:
                width, height = im.size
            gts = load_gt(args.gt_dir / f'{image_path.stem}.txt', width, height)
            if not gts:
                continue
            gt = gts[0]
            b = best(row.get("detections") or [], gt, class_id)
            t = best(row.get("tta_detections") or [], gt, class_id)
            if b is None and t is None:
                continue
            # Use the better stored source as the fixed input to calibration.
            if args.detection_source == "base":
                source = b
            elif args.detection_source == "tta":
                source = t
            else:
                source = b if b is not None and (t is None or iou(b["xyxy"], gt) >= iou(t["xyxy"], gt)) else t
            if source is None:
                continue
            rows.append({"file": row["file"], "width": width, "height": height,
                         "gt": gt, "box": source["xyxy"], "source": args.detection_source})
        all_data[label] = rows

    result: dict[str, Any] = {"classes": {}, "folds": args.folds, "samples": args.samples}
    for label, rows in all_data.items():
        before = score(rows, (1.0, 1.0, 0.0, 0.0))
        params = optimize(rows, args.samples, args.seed)
        after = score(rows, params)
        # 5-fold out-of-fold predictions using per-fold fitted parameters.
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        folds = [indices[i::args.folds] for i in range(args.folds)]
        oof_iou: list[float] = []
        oof_pass = 0
        fold_params = []
        for fold in folds:
            test_ids = set(fold)
            train = [r for i, r in enumerate(rows) if i not in test_ids]
            test = [r for i, r in enumerate(rows) if i in test_ids]
            p = optimize(train, args.samples, args.seed + len(fold_params))
            fold_params.append(p)
            for r in test:
                v = iou(calibrated(r["box"], r["width"], r["height"], p), r["gt"])
                oof_iou.append(v)
                oof_pass += int(v >= 0.75)
        result["classes"][label] = {
            "n": len(rows),
            "before_mean_iou": round(before[0], 6),
            "before_passed": before[1],
            "fitted_params": [round(v, 6) for v in params],
            "fitted_mean_iou": round(after[0], 6),
            "fitted_passed": after[1],
            "oof_mean_iou": round(sum(oof_iou) / len(oof_iou), 6) if oof_iou else None,
            "oof_passed": oof_pass,
            "fold_params": [[round(v, 6) for v in p] for p in fold_params],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["classes"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
