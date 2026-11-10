#!/usr/bin/env python3
"""Acceptance test: fusion_postprocess.fuse_views must reproduce the offline result
on the fixed 2026-10-16 set (205/225 IoU@0.75, 29/225 frames with a wrong-class box).

Default GT uses the reviewed v4 Mug rule: one box around the whole mug, including its handle.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from fusion_postprocess import box_iou, fuse_views, load_calibration
from arbiter_experiment_20261016 import CLASSES, REP, EVAL, load_gt, size

SRC = REP / "multiview_predictions_20261016"
CAL = REP / "box_calibration_v3_20261016"
VIEWS = ["v1.00", "v1.00_tta", "v0.75", "v0.75_tta", "v0.65", "v0.65_tta"]
CLASS_NAMES = ["1-Cheez-it", "2-Starkist_Tuna", "3-Scissors", "4-Frenchs_Mustard",
               "5-Tomato_Soup", "6-Foam_Brick", "7-Clamp", "8-Plastic_Banana",
               "9-Mug", "10-meat_can"]
EXPECT_PASS, EXPECT_TOTAL, EXPECT_WRONG_FRAMES = 205, 225, 29
DEFAULT_GT = REP / "gt_reviewed_v4_mug_handle_20261016" / "labels"


def load_gt_from(gt_dir: Path, stem: str, w: int, h: int) -> list[dict]:
    p = gt_dir / (stem + ".txt")
    if not p.exists():
        return []
    rows = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        rows.append({"class_id": cid,
                     "xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    ap.add_argument("--expect-pass", type=int, default=None)
    ap.add_argument("--expect-wrong-frames", type=int, default=None)
    args = ap.parse_args()
    gt_dir = args.gt_dir
    expect_pass = EXPECT_PASS if args.expect_pass is None else args.expect_pass
    expect_wrong = EXPECT_WRONG_FRAMES if args.expect_wrong_frames is None else args.expect_wrong_frames
    cal = {"v1.00": load_calibration(CAL / "base.json"),
           "v1.00_tta": load_calibration(CAL / "tta.json")}
    passed = tot = wrong_frames = wrong_boxes = 0
    per_pass, per_tot = Counter(), Counter()
    for cid, folder, _ in CLASSES:
        p = SRC / folder / "summary.json"
        if not p.exists():
            continue
        for row in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            fname = row["file"]
            w, h = size(EVAL / folder / fname)
            gt = [g for g in load_gt_from(gt_dir, Path(fname).stem, w, h) if g["class_id"] == cid]
            if not gt:
                continue
            by_view = {v: [{"class_id": d["class_id"],
                            "class": CLASS_NAMES[d["class_id"]],
                            "confidence": d["conf"], "bbox": d["xyxy"]}
                           for d in (row.get(v) or [])] for v in VIEWS}
            fused = fuse_views(by_view, min_views=3, width=w, height=h,
                               calibration_by_view=cal, class_names=CLASS_NAMES)
            ok = sum(1 for g in gt if max(
                [box_iou(g["xyxy"], x["bbox"]) for x in fused if x["class_id"] == cid] + [0.0]) >= 0.75)
            passed += ok
            tot += len(gt)
            per_pass[folder] += ok
            per_tot[folder] += len(gt)
            wrong = [x["class_id"] for x in fused if x["class_id"] != cid]
            wrong_frames += bool(wrong)
            wrong_boxes += len(wrong)
    print(f"pass {passed}/{tot} ({100*passed/tot:.1f}%) | wrong-class frames {wrong_frames} | wrong boxes {wrong_boxes}")
    for f in sorted(per_tot):
        print(f"  {f}: {per_pass[f]}/{per_tot[f]}")
    print("GT:", gt_dir)
    ok = (passed, tot, wrong_frames) == (expect_pass, EXPECT_TOTAL, expect_wrong)
    print("ACCEPTANCE:", "PASS" if ok else
          f"FAIL (expected {expect_pass}/{EXPECT_TOTAL}, {expect_wrong} wrong frames)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
