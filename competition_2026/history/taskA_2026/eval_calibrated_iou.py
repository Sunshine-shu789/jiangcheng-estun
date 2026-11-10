#!/usr/bin/env python3
"""Apply fitted per-class box calibration to saved base/TTA detections and score IoU@0.75."""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    p.add_argument("--calibration-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--iou-threshold", type=float, default=0.75)
    return p.parse_args()


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter) if aa + ab - inter > 0 else 0.0


def load_gt(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{path}:{line_no}: expected at least 5 columns")
        cls, cx, cy, w, h = (float(v) for v in parts[:5])
        if not all(math.isfinite(v) for v in (cls, cx, cy, w, h)):
            raise ValueError(f"{path}:{line_no}: non-finite")
        rows.append({
            "class_id": int(cls),
            "xyxy": [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height],
        })
    return rows


def calibrate(box: list[float], width: int, height: int, p: list[float]) -> list[float]:
    sx, sy, dx, dy = p
    cx = (box[0] + box[2]) / 2 + dx * width
    cy = (box[1] + box[3]) / 2 + dy * height
    w = max(1.0, (box[2] - box[0]) * sx)
    h = max(1.0, (box[3] - box[1]) * sy)
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def best(dets: list[dict[str, Any]], gt: list[float], class_id: int,
         width: int, height: int, params: list[float]) -> dict[str, Any] | None:
    choices = []
    for d in dets:
        if int(d["class_id"]) != class_id:
            continue
        box = calibrate(d["xyxy"], width, height, params)
        choices.append({"iou": iou(box, gt), "box": box, "conf": float(d.get("conf", 0.0)),
                        "class_id": int(d["class_id"]), "name": d.get("name", str(class_id))})
    return max(choices, key=lambda x: x["iou"]) if choices else None


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_cal = json.loads((args.calibration_root / "base.json").read_text(encoding="utf-8"))["classes"]
    tta_cal = json.loads((args.calibration_root / "tta.json").read_text(encoding="utf-8"))["classes"]
    aggregate: dict[str, Any] = {}
    all_instances: list[dict[str, Any]] = []
    for label, class_id in CLASSES.items():
        summary_path = args.report_root / f"{label}_predictions_640" / "summary.json"
        image_dir = args.report_root.parents[1] / "eval_capture_20261016" / label
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        bp = base_cal[label]["fitted_params"]
        tp = tta_cal[label]["fitted_params"]
        instances = []
        missing = []
        for row in summary["rows"]:
            image_path = image_dir / row["file"]
            with Image.open(image_path) as im:
                width, height = im.size
            gts = load_gt(args.gt_dir / f"{image_path.stem}.txt", width, height)
            if not gts or int(gts[0]["class_id"]) != class_id:
                missing.append(row["file"])
                continue
            gt = gts[0]["xyxy"]
            b = best(row.get("detections") or [], gt, class_id, width, height, bp)
            t = best(row.get("tta_detections") or [], gt, class_id, width, height, tp)
            u = max([x for x in (b, t) if x is not None], key=lambda x: x["iou"]) if (b or t) else None
            bp_pass = int(b is not None and b["iou"] >= args.iou_threshold)
            tp_pass = int(t is not None and t["iou"] >= args.iou_threshold)
            up_pass = int(u is not None and u["iou"] >= args.iou_threshold)
            item = {
                "label": label, "file": row["file"], "gt_class_id": class_id,
                "gt_xyxy": gt,
                "base_cal_iou": round(b["iou"], 6) if b else 0.0,
                "base_cal_conf": round(b["conf"], 6) if b else 0.0,
                "base_cal_box": [round(v, 2) for v in b["box"]] if b else [],
                "base_cal_pass": bp_pass,
                "tta_cal_iou": round(t["iou"], 6) if t else 0.0,
                "tta_cal_conf": round(t["conf"], 6) if t else 0.0,
                "tta_cal_box": [round(v, 2) for v in t["box"]] if t else [],
                "tta_cal_pass": tp_pass,
                "union_cal_iou": round(u["iou"], 6) if u else 0.0,
                "union_cal_box": [round(v, 2) for v in u["box"]] if u else [],
                "union_cal_pass": up_pass,
            }
            instances.append(item)
            all_instances.append(item)
        n = len(instances)
        aggregate[label] = {
            "n": n,
            "missing_gt": missing,
            "base_passed": sum(r["base_cal_pass"] for r in instances),
            "tta_passed": sum(r["tta_cal_pass"] for r in instances),
            "union_passed": sum(r["union_cal_pass"] for r in instances),
            "base_mean_iou": sum(r["base_cal_iou"] for r in instances) / n if n else 0.0,
            "tta_mean_iou": sum(r["tta_cal_iou"] for r in instances) / n if n else 0.0,
            "union_mean_iou": sum(r["union_cal_iou"] for r in instances) / n if n else 0.0,
            "base_params": bp, "tta_params": tp,
        }
    total = len(all_instances)
    aggregate["__total__"] = {
        "n": total,
        "base_passed": sum(r["base_cal_pass"] for r in all_instances),
        "tta_passed": sum(r["tta_cal_pass"] for r in all_instances),
        "union_passed": sum(r["union_cal_pass"] for r in all_instances),
        "base_mean_iou": sum(r["base_cal_iou"] for r in all_instances) / total if total else 0.0,
        "tta_mean_iou": sum(r["tta_cal_iou"] for r in all_instances) / total if total else 0.0,
        "union_mean_iou": sum(r["union_cal_iou"] for r in all_instances) / total if total else 0.0,
    }
    (args.out_dir / "calibrated_iou75.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out_dir / "calibrated_iou75_instances.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_instances[0]))
        writer.writeheader(); writer.writerows(all_instances)
    lines = ["# 类别校准后比赛口径 IoU 评估", "", "| 类别 | 实例 | 单帧校准 | TTA校准 | 并集 | mean IoU(union) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label in CLASSES:
        d = aggregate[label]
        lines.append(f"| {label} | {d['n']} | {d['base_passed']}/{d['n']} | {d['tta_passed']}/{d['n']} | {d['union_passed']}/{d['n']} | {d['union_mean_iou']:.3f} |")
    d = aggregate["__total__"]
    lines.append(f"| **总计** | {d['n']} | {d['base_passed']}/{d['n']} | {d['tta_passed']}/{d['n']} | {d['union_passed']}/{d['n']} | {d['union_mean_iou']:.3f} |")
    (args.out_dir / "calibrated_iou75.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
