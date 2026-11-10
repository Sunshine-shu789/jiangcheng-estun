#!/usr/bin/env python3
"""6-view fusion / arbitration experiment on the fixed 2026-10-16 set.

Consumes multiview_predictions_20261016/<class>/summary.json (6 views) and the
fixed merged GT. Reports honest IoU@0.75 (one output box per object) plus the
cross-class false-positive count for each candidate deployment pipeline.

Variants
--------
  single          v1.00 detections
  single*         v1.00 + per-class box calibration
  tta             v1.00_tta detections
  tta*            v1.00_tta + per-class box calibration
  wbf2*           WBF of the two calibrated 1.0 views
  wbf6            WBF of all six raw views
  wbf6*           WBF of two calibrated views + four raw crop views
  wbf6+arb        wbf6 + attractor suppression
  cons3           box must appear (same class, IoU>=0.5) in >=3 of 6 views
  cons3+arb       cons3 + attractor suppression
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from arbiter_experiment_20261016 import (
    CLASSES, HUMAN, IOU_PASS, REP, EVAL,
    attractor_suppress, best_iou, calibrate, det, iou, load_gt, merge_views,
    size, wbf,
)

SRC = REP / "multiview_predictions_20261016"
OUT = REP / "multiview_arbiter_20261016"

VIEWS = ["v1.00", "v1.00_tta", "v0.75", "v0.75_tta", "v0.65", "v0.65_tta"]
CONSIST_IOU = 0.5
CONSIST_MIN_VIEWS = 3


def as_dets(items: list[dict], src: str) -> list[dict]:
    return [det(d["class_id"], d["conf"], d["xyxy"], src) for d in items or []]


def corroborated(view_dets: dict[str, list[dict]], min_views: int) -> list[dict]:
    """Keep a box only if a same-class overlapping box appears in >= min_views views."""
    votes: list[dict] = []
    for name in VIEWS:
        for d in view_dets.get(name, []):
            matched = None
            for v in votes:
                if v["class_id"] == d["class_id"] and iou(v["xyxy"], d["xyxy"]) >= CONSIST_IOU:
                    matched = v
                    break
            if matched is None:
                votes.append(dict(d, views={name}, n=1))
            else:
                matched["n"] += 1
                matched["views"].add(name)
                if d["conf"] > matched["conf"]:
                    matched.update(conf=d["conf"], xyxy=d["xyxy"])
    kept = [dict(v, src="+".join(sorted(v["views"]))) for v in votes if v["n"] >= min_views]
    return merge_views(kept)


def vote_filter(view_dets: dict[str, list[dict]], min_views: int) -> list[dict]:
    """Keep every box that has a same-class overlapping partner in >= min_views-1
    other views. Returns the surviving boxes (not yet merged)."""
    names = list(view_dets)
    kept: list[dict] = []
    for i, name in enumerate(names):
        for d in view_dets[name]:
            votes = 1
            for j, other in enumerate(names):
                if i == j:
                    continue
                if any(x["class_id"] == d["class_id"] and iou(x["xyxy"], d["xyxy"]) >= CONSIST_IOU
                       for x in view_dets[other]):
                    votes += 1
                    if votes >= min_views:
                        break
            if votes >= min_views:
                kept.append(d)
    return kept


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variants = ["single", "tta*", "wbf2*", "wbf6*", "vote2+wbf*", "vote3+wbf*",
                "vote2+wbf+arb*", "vote3+wbf+arb*", "cons3"]
    per_var: dict[str, list[dict]] = {v: [] for v in variants}
    confusion: dict[str, Counter] = {v: Counter() for v in variants}
    name = HUMAN

    for cid, folder, _human in CLASSES:
        summ = SRC / folder / "summary.json"
        if not summ.exists():
            print(f"[skip] {folder}")
            continue
        rows = json.loads(summ.read_text(encoding="utf-8"))["rows"]
        img_dir = EVAL / folder
        for row in rows:
            fname = row["file"]
            w, h = size(img_dir / fname)
            gt = [g for g in load_gt(Path(fname).stem, w, h) if g["class_id"] == cid]
            if not gt:
                continue
            vd = {v: as_dets(row.get(v) or [], v) for v in VIEWS}
            cb = calibrate(vd["v1.00"], "base", w, h)
            ct = calibrate(vd["v1.00_tta"], "tta", w, h)
            views = {
                "single": vd["v1.00"],
                "single*": cb,
                "tta": vd["v1.00_tta"],
                "tta*": ct,
                "wbf2*": wbf(cb + ct),
            }
            vd_cal = dict(vd)
            vd_cal["v1.00"] = cb
            vd_cal["v1.00_tta"] = ct
            views["wbf6*"] = wbf([d for v in VIEWS for d in vd_cal[v]])
            v2 = vote_filter(vd_cal, 2)
            v3 = vote_filter(vd_cal, 3)
            views["vote2+wbf*"] = wbf(v2)
            views["vote3+wbf*"] = wbf(v3)
            views["vote2+wbf+arb*"] = attractor_suppress(views["vote2+wbf*"])
            views["vote3+wbf+arb*"] = attractor_suppress(views["vote3+wbf*"])
            views["cons3"] = corroborated(vd, CONSIST_MIN_VIEWS)
            for v in variants:
                dets = views[v]
                passed = sum(1 for g in gt if best_iou(g["xyxy"], dets, cid) >= IOU_PASS)
                ious = [best_iou(g["xyxy"], dets, cid) for g in gt]
                wrong = [d["class_id"] for d in dets if d["class_id"] != cid]
                per_var[v].append({"class_id": cid, "human": name[cid], "file": fname,
                                   "n_gt": len(gt), "n_pass": passed,
                                   "mean_iou": sum(ious) / len(ious),
                                   "wrong_classes": wrong})
                confusion[v].update(wrong)

    totals = {v: [0, 0] for v in variants}
    for cid in [c[0] for c in CLASSES]:
        for v in variants:
            totals[v][0] += sum(r["n_pass"] for r in per_var[v] if r["class_id"] == cid)
            totals[v][1] += sum(r["n_gt"] for r in per_var[v] if r["class_id"] == cid)

    md = ["# 6 视角融合 / 类别仲裁实验（离线，固定验收集）", "",
          "- 权重：v23 epoch3；视角：imgsz640 × {原图, TTA} × {1.00, 0.75, 0.65 灰底缩放}",
          f"- 通过：IoU@{IOU_PASS}，GT = gt_reviewed_v4_mug_handle_20261016（固定版）",
          f"- 一致性参数：consist_iou={CONSIST_IOU}, min_views={CONSIST_MIN_VIEWS}",
          "- `*` = 使用按类拟合的框校准（仅 1.00 视角有拟合参数）", ""]
    md.append("| 类别 | 实例 | " + " | ".join(variants) + " |")
    md.append("| --- | ---: | " + " | ".join("---:" for _ in variants) + " |")
    for cid in [c[0] for c in CLASSES]:
        n = sum(r["n_gt"] for r in per_var["single"] if r["class_id"] == cid)
        row = [name[cid], str(n)]
        for v in variants:
            row.append(f"{sum(r['n_pass'] for r in per_var[v] if r['class_id'] == cid)}/{n}")
        md.append("| " + " | ".join(row) + " |")
    row = ["**总计**", str(totals['single'][1])]
    for v in variants:
        row.append(f"**{totals[v][0]}/{totals[v][1]} ({100*totals[v][0]/totals[v][1]:.1f}%)**")
    md.append("| " + " | ".join(row) + " |")

    md += ["", "## 可靠性：出现任何非本类框的帧数 / 非本类框总数", "",
           "（这一组才是用户真正看到的\"看错\"问题；帧数越少越好）", "",
           "| 候选管线 | 通过率 | 有错类框的帧 | 错类框总数 |",
           "| --- | ---: | ---: | ---: |"]
    for v in variants:
        rows_all = per_var[v]
        wf = sum(1 for r in rows_all if r["wrong_classes"])
        wt = sum(len(r["wrong_classes"]) for r in rows_all)
        md.append(f"| {v} | {totals[v][0]}/{totals[v][1]} ({100*totals[v][0]/totals[v][1]:.1f}%) "
                  f"| {wf}/{len(rows_all)} | {wt} |")

    md += ["", "## 平均 IoU", "", "| 类别 | " + " | ".join(variants) + " |",
           "| --- | " + " | ".join("---:" for _ in variants) + " |"]
    for cid in [c[0] for c in CLASSES]:
        row = [name[cid]]
        for v in variants:
            rs = [r["mean_iou"] for r in per_var[v] if r["class_id"] == cid]
            row.append(f"{sum(rs)/len(rs):.3f}" if rs else "-")
        md.append("| " + " | ".join(row) + " |")

    md += ["", "## 跨类误报（非本类框计数）", ""]
    for v in variants:
        txt = ", ".join(f"{name.get(k,k)}×{n}" for k, n in confusion[v].most_common(6)) or "无"
        md.append(f"- **{v}**：{txt}")

    report = {"iou_threshold": IOU_PASS, "variants": {}}
    for v in variants:
        report["variants"][v] = {
            "passed": totals[v][0], "total": totals[v][1],
            "pass_rate": round(totals[v][0] / totals[v][1], 4),
            "wrong_frames": sum(1 for r in per_var[v] if r["wrong_classes"]),
            "wrong_boxes": sum(len(r["wrong_classes"]) for r in per_var[v]),
            "confusions": {name.get(k, str(k)): n for k, n in confusion[v].most_common()},
            "per_class": {name[c]: {
                "instances": sum(r["n_gt"] for r in per_var[v] if r["class_id"] == c),
                "passed": sum(r["n_pass"] for r in per_var[v] if r["class_id"] == c),
            } for c in [x[0] for x in CLASSES]},
        }
    (OUT / "multiview_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "multiview_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (OUT / "multiview_instances.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["variant", "class", "file", "n_gt", "n_pass", "mean_iou", "wrong_classes"])
        for v in variants:
            for r in per_var[v]:
                wr.writerow([v, r["human"], r["file"], r["n_gt"], r["n_pass"],
                             f"{r['mean_iou']:.4f}",
                             " ".join(name.get(c, str(c)) for c in r["wrong_classes"])])
    print("\n".join(md))


if __name__ == "__main__":
    main()
