#!/usr/bin/env python3
"""Category-arbitration / multi-view fusion experiment on the fixed 2026-10-16 set.

Runs entirely offline on the already-saved base + TTA detections in each
``<class>_predictions_640/summary.json``. No model, no GPU, no weight changes.

Variants compared
-----------------
  base        : single-frame 640 detections
  tta         : test-time-augmented 640 detections
  union       : base ∪ TTA, class-wise greedy NMS merge (combined score = max)
  union+arb   : union + "attractor" suppression (meat_can vs specific classes)
  consist     : keep only boxes corroborated in BOTH views (IoU>=0.5)
  consist+arb : consist + attractor suppression

GT side: fixed merged set (gt_reviewed_v4_mug_handle_20261016/labels).
Threshold: IoU@0.75, matching the competition-style pass criterion.

Writes a per-variant JSON + markdown report and a per-instance CSV.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path("/media/sunshine/新加卷/taskA")
EVAL = ROOT / "eval_capture_20261016"
REP = ROOT / "reports" / "camera_eval_20261016"
GT_DIR = REP / "gt_reviewed_v4_mug_handle_20261016" / "labels"
OUT = REP / "arbiter_experiment_20261016"

IOU_PASS = 0.75
MERGE_IOU = 0.5        # duplicate suppression across views
CONSIST_IOU = 0.5      # corroboration match across views
CONSIST_MIN_CONF = 0.90  # a single-view box survives alone only at this conf
ATTRACTOR_IOU = 0.55   # overlap needed to treat meat_can as a duplicate label
ATTRACTOR_MARGIN = 0.15  # specific class may be this much less confident

# class_id -> (folder name, human name)
CLASSES = [
    (0, "1-Cheez-it", "Cheez-it"),
    (1, "2-Starkist_Tuna", "Tuna"),
    (2, "3-Scissors", "Scissors"),
    (3, "4-Frenchs_Mustard", "Mustard"),
    (4, "5-Tomato_Soup", "Tomato Soup"),
    (5, "6-Foam_Brick", "Foam Brick"),
    (6, "7-Clamp", "Clamp"),
    (8, "9-Mug", "Mug"),
    (9, "10-meat_can", "meat_can"),
]
HUMAN = {cid: human for cid, _, human in CLASSES}
CAL_NAME = {cid: folder for cid, folder, _ in CLASSES}
SPECIFIC = {0, 1, 2, 3, 4, 5, 6, 8}   # everything except meat_can (9)
ATTRACTOR = 9

CALIB_ROOT = REP / "box_calibration_v3_20261016"


def _load_calib(fname: str) -> dict[str, list[float]]:
    p = CALIB_ROOT / fname
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: v["fitted_params"] for k, v in data.get("classes", {}).items()}


CALIB = {"base": _load_calib("base.json"), "tta": _load_calib("tta.json")}


def calibrate(dets: list[dict[str, Any]], view: str, w: int, h: int) -> list[dict[str, Any]]:
    """Rescale each box by the per-class fitted params for its source view."""
    params = CALIB.get(view, {})
    out = []
    for d in dets:
        key = CAL_NAME.get(d["class_id"])
        p = params.get(key) if key else None
        if not p:
            out.append(d)
            continue
        sx, sy, dx, dy = p
        x1, y1, x2, y2 = d["xyxy"]
        cx, cy = (x1 + x2) / 2 + dx * w, (y1 + y2) / 2 + dy * h
        bw, bh = max(1.0, (x2 - x1) * sx), max(1.0, (y2 - y1) * sy)
        out.append(dict(d, xyxy=[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
    return out


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = ua + ub - inter
    return inter / denom if denom > 0 else 0.0


def det(cid: int, conf: float, xyxy: list[float], src: str) -> dict[str, Any]:
    return {"class_id": int(cid), "conf": float(conf),
            "xyxy": [float(v) for v in xyxy], "src": src}


def merge_views(dets: list[dict[str, Any]], thr: float = MERGE_IOU) -> list[dict[str, Any]]:
    """Class-wise greedy merge; combined score = max over the merged group."""
    kept: list[dict[str, Any]] = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        hit = None
        for k in kept:
            if k["class_id"] == d["class_id"] and iou(k["xyxy"], d["xyxy"]) >= thr:
                hit = k
                break
        if hit is None:
            kept.append(dict(d))
        else:
            hit["conf"] = max(hit["conf"], d["conf"])
            hit["src"] = hit["src"] + "+" + d["src"] if d["src"] not in hit["src"] else hit["src"]
    return kept


def wbf(dets: list[dict[str, Any]], thr: float = MERGE_IOU) -> list[dict[str, Any]]:
    """Confidence-weighted box fusion: same-class overlapping boxes merge into one
    confidence-weighted average box (better localisation than pick-the-max)."""
    clusters: list[list[dict[str, Any]]] = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        placed = False
        for cl in clusters:
            if cl[0]["class_id"] == d["class_id"] and any(iou(m["xyxy"], d["xyxy"]) >= thr for m in cl):
                cl.append(d)
                placed = True
                break
        if not placed:
            clusters.append([d])
    out = []
    for cl in clusters:
        wsum = sum(m["conf"] for m in cl)
        coords = [sum(m["xyxy"][i] * m["conf"] for m in cl) / wsum for i in range(4)]
        srcs = []
        for m in cl:
            for s_ in m["src"].split("+"):
                if s_ not in srcs:
                    srcs.append(s_)
        out.append({"class_id": cl[0]["class_id"], "conf": max(m["conf"] for m in cl),
                    "xyxy": coords, "src": "+".join(srcs), "n": len(cl)})
    return out


def consistency(base: list[dict], tta: list[dict]) -> list[dict]:
    """Keep boxes corroborated across views; lonely boxes only if very confident."""
    out: list[dict] = []
    for d in base:
        if any(x["class_id"] == d["class_id"] and iou(x["xyxy"], d["xyxy"]) >= CONSIST_IOU for x in tta):
            out.append(dict(d, src="base+tta"))
        elif d["conf"] >= CONSIST_MIN_CONF:
            out.append(dict(d, src="base"))
    for d in tta:
        if not any(x["class_id"] == d["class_id"] and iou(x["xyxy"], d["xyxy"]) >= CONSIST_IOU for x in base):
            if d["conf"] >= CONSIST_MIN_CONF:
                out.append(dict(d, src="tta"))
    return merge_views(out)


def attractor_suppress(dets: list[dict]) -> list[dict]:
    """Drop a meat_can box when a specific-class box labels the same object."""
    kept: list[dict] = []
    for d in dets:
        if d["class_id"] != ATTRACTOR:
            kept.append(d)
            continue
        clash = any(
            k["class_id"] in SPECIFIC
            and iou(k["xyxy"], d["xyxy"]) >= ATTRACTOR_IOU
            and k["conf"] >= d["conf"] - ATTRACTOR_MARGIN
            for k in dets
        )
        if not clash:
            kept.append(d)
    return kept


def load_gt(name: str, w: int, h: int) -> list[dict]:
    p = GT_DIR / (name + ".txt")
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


def best_iou(gt: list[float], dets: list[dict], cid: int) -> float:
    best = 0.0
    for d in dets:
        if d["class_id"] != cid:
            continue
        best = max(best, iou(gt, d["xyxy"]))
    return best


def size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variants = ["base", "tta", "union", "union+arb", "consist",
                "base*", "tta*", "union*", "union+arb*", "consist*", "wbf*", "wbf+arb*"]
    per_var: dict[str, list[dict]] = {v: [] for v in variants}
    confusion: dict[str, Counter] = {v: Counter() for v in variants}

    for cid, folder, human in CLASSES:
        summary = REP / f"{folder}_predictions_640" / "summary.json"
        if not summary.exists():
            continue
        rows = json.loads(summary.read_text(encoding="utf-8"))["rows"]
        img_dir = EVAL / folder
        for row in rows:
            fname = row["file"]
            w, h = size(img_dir / fname)
            gt = load_gt(Path(fname).stem, w, h)
            gt_here = [g for g in gt if g["class_id"] == cid]
            if not gt_here:
                continue
            base = [det(d["class_id"], d["conf"], d["xyxy"], "base") for d in row.get("detections") or []]
            tta = [det(d["class_id"], d["conf"], d["xyxy"], "tta") for d in row.get("tta_detections") or []]
            views = {"base": base, "tta": tta}
            views["union"] = merge_views(base + tta)
            views["union+arb"] = attractor_suppress(views["union"])
            views["consist"] = consistency(base, tta)
            # calibrated pipelines ("*" = box-calibrated inputs)
            cb = calibrate(base, "base", w, h)
            ct = calibrate(tta, "tta", w, h)
            views["base*"] = cb
            views["tta*"] = ct
            views["union*"] = merge_views(cb + ct)
            views["union+arb*"] = attractor_suppress(views["union*"])
            views["consist*"] = consistency(cb, ct)
            views["wbf*"] = wbf(cb + ct)
            views["wbf+arb*"] = attractor_suppress(views["wbf*"])
            for v, dets in views.items():
                passed = sum(1 for g in gt_here if best_iou(g["xyxy"], dets, cid) >= IOU_PASS)
                ious = [best_iou(g["xyxy"], dets, cid) for g in gt_here]
                wrong = [d["class_id"] for d in dets if d["class_id"] != cid]
                per_var[v].append({
                    "class_id": cid, "folder": folder, "human": human, "file": fname,
                    "n_gt": len(gt_here), "n_pass": passed,
                    "mean_iou": sum(ious) / len(ious),
                    "wrong_classes": wrong,
                })
                confusion[v].update(wrong)

    report: dict[str, Any] = {"iou_threshold": IOU_PASS, "variants": {}}
    md = ["# 类别仲裁 / 多视角融合实验（离线，固定验收集）", "",
          "- 权重与推理结果：v23 epoch3，base@640 与 TTA@640（复用已存 summary.json，未重跑模型）",
          f"- 通过口径：IoU@{IOU_PASS}，GT = gt_reviewed_v4_mug_handle_20261016",
          f"- 融合/仲裁参数：merge_iou={MERGE_IOU}, consist_iou={CONSIST_IOU}, "
          f"consist_min_conf={CONSIST_MIN_CONF}, attractor_iou={ATTRACTOR_IOU}, margin={ATTRACTOR_MARGIN}",
          ""]
    md.append("| 类别 | 实例 | " + " | ".join(variants) + " |")
    md.append("| --- | ---: | " + " | ".join("---:" for _ in variants) + " |")
    name = HUMAN
    order = [c[0] for c in CLASSES]
    totals = {v: [0, 0] for v in variants}
    for cid in order:
        row = [f"{name[cid]}"]
        n = sum(r["n_gt"] for r in per_var["base"] if r["class_id"] == cid)
        row.append(str(n))
        for v in variants:
            p = sum(r["n_pass"] for r in per_var[v] if r["class_id"] == cid)
            totals[v][0] += p
            totals[v][1] += n
            row.append(f"{p}/{n}")
        md.append("| " + " | ".join(row) + " |")
    row = ["**总计**", str(totals['base'][1])]
    for v in variants:
        row.append(f"**{totals[v][0]}/{totals[v][1]} ({100*totals[v][0]/totals[v][1]:.1f}%)**")
    md.append("| " + " | ".join(row) + " |")

    md += ["", "## 平均 IoU（每类，union 口径）", ""]
    md.append("| 类别 | " + " | ".join(variants) + " |")
    md.append("| --- | " + " | ".join("---:" for _ in variants) + " |")
    for cid in order:
        row = [name[cid]]
        for v in variants:
            rs = [r["mean_iou"] for r in per_var[v] if r["class_id"] == cid]
            row.append(f"{sum(rs)/len(rs):.3f}" if rs else "-")
        md.append("| " + " | ".join(row) + " |")

    md += ["", "## 跨类误报（所有非本类框计数）", ""]
    for v in variants:
        c = confusion[v]
        txt = ", ".join(f"{name.get(k,k)}×{n}" for k, n in c.most_common(6)) or "无"
        md.append(f"- **{v}**：{txt}")

    for v in variants:
        report["variants"][v] = {
            "passed": totals[v][0], "total": totals[v][1],
            "pass_rate": round(totals[v][0] / totals[v][1], 4),
            "per_class": {
                name[cid]: {
                    "instances": sum(r["n_gt"] for r in per_var[v] if r["class_id"] == cid),
                    "passed": sum(r["n_pass"] for r in per_var[v] if r["class_id"] == cid),
                    "mean_iou": round(sum(r["mean_iou"] for r in per_var[v] if r["class_id"] == cid) /
                                      max(1, sum(1 for r in per_var[v] if r["class_id"] == cid)), 4),
                } for cid in order
            },
            "confusions": {name.get(k, str(k)): n for k, n in confusion[v].most_common()},
        }
    (OUT / "arbiter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "arbiter_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with (OUT / "arbiter_instances.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "class", "file", "n_gt", "n_pass", "mean_iou", "wrong_classes"])
        for v in variants:
            for r in per_var[v]:
                w.writerow([v, r["human"], r["file"], r["n_gt"], r["n_pass"],
                            f"{r['mean_iou']:.4f}", " ".join(name.get(c, str(c)) for c in r["wrong_classes"])])
    print("\n".join(md))


if __name__ == "__main__":
    main()
