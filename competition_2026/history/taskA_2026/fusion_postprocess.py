#!/usr/bin/env python3
"""Deployable multi-view fusion / category arbitration for task-A detection.

Validated offline on the fixed 2026-10-16 set (225 instances, 9 classes):
  single 1.00 view .................. 151/225 (67.1%), 44/225 frames with a wrong-class box
  TTA 1.00 + box calibration ........ 203/225 (90.2%), 44/225
  3-of-6 vote + WBF + arbitration ... 205/225 (91.1%), 29/225   <- recommended

Pipeline (per frame):
  1. collect one detection list per VIEW (scale x {plain, tta})
  2. optionally apply per-view per-class box calibration (sx, sy, dx, dy)
  3. vote gate: keep a box only if a same-class IoU>=0.5 partner appears in
     >= min_views distinct views
  4. confidence-weighted box fusion (WBF) over the survivors
  5. attractor arbitration: drop an over-predicted class box when a
     specific-class box labels the same object with comparable confidence

Pure post-processing: no model, no weights, no GPU. Side-effect free.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# over-predicted "attractor" class -> classes allowed to displace it
ATTRACTOR_CLASSES = {9}          # 10-meat_can
DISPLACERS = {0, 1, 2, 3, 4, 5, 6, 8}
VOTE_IOU = 0.5
WBF_IOU = 0.5
ATTRACTOR_IOU = 0.55
ATTRACTOR_MARGIN = 0.15


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = aa + ab - inter
    return inter / denom if denom > 0 else 0.0


def load_calibration(path: str | Path) -> dict[str, list[float]]:
    """Read box_calibration_v3_20261016/<base|tta>.json -> {class_name: params}."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: v["fitted_params"] for k, v in data.get("classes", {}).items()}


def calibrate(
    detections: Sequence[Mapping[str, Any]],
    params_by_class: Mapping[str, Sequence[float]],
    class_names: Sequence[str],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Apply per-class (sx, sy, dx, dy) box calibration in pixel space."""
    out: list[dict[str, Any]] = []
    for d in detections:
        cid = int(d["class_id"])
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        p = params_by_class.get(name)
        box = [float(v) for v in d["bbox"]]
        if p:
            sx, sy, dx, dy = p
            cx = (box[0] + box[2]) / 2 + dx * width
            cy = (box[1] + box[3]) / 2 + dy * height
            bw = max(1.0, (box[2] - box[0]) * sx)
            bh = max(1.0, (box[3] - box[1]) * sy)
            box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
        out.append({**d, "bbox": box})
    return out


def vote_gate(
    detections_by_view: Mapping[str, Sequence[Mapping[str, Any]]],
    min_views: int = 3,
    vote_iou: float = VOTE_IOU,
) -> list[dict[str, Any]]:
    """Keep every box that has a same-class IoU>=vote_iou partner in enough views."""
    views = list(detections_by_view)
    kept: list[dict[str, Any]] = []
    for i, view in enumerate(views):
        for d in detections_by_view[view]:
            votes = 1
            for j, other in enumerate(views):
                if i == j:
                    continue
                if any(
                    int(x["class_id"]) == int(d["class_id"])
                    and box_iou(x["bbox"], d["bbox"]) >= vote_iou
                    for x in detections_by_view[other]
                ):
                    votes += 1
                    if votes >= min_views:
                        break
            if votes >= min_views:
                kept.append(dict(d))
    return kept


def wbf(detections: Iterable[Mapping[str, Any]], wbf_iou: float = WBF_IOU) -> list[dict[str, Any]]:
    """Confidence-weighted box fusion: same-class overlapping boxes -> weighted mean box."""
    clusters: list[list[dict[str, Any]]] = []
    for d in sorted(detections, key=lambda x: -float(x["confidence"])):
        placed = False
        for cl in clusters:
            if int(cl[0]["class_id"]) == int(d["class_id"]) and any(
                box_iou(m["bbox"], d["bbox"]) >= wbf_iou for m in cl
            ):
                cl.append(d)
                placed = True
                break
        if not placed:
            clusters.append([dict(d)])
    out: list[dict[str, Any]] = []
    for cl in clusters:
        wsum = sum(float(m["confidence"]) for m in cl) or 1.0
        coords = [sum(float(m["bbox"][i]) * float(m["confidence"]) for m in cl) / wsum
                  for i in range(4)]
        out.append({
            "class_id": int(cl[0]["class_id"]),
            "class": cl[0].get("class"),
            "confidence": round(max(float(m["confidence"]) for m in cl), 4),
            "bbox": [round(v, 1) for v in coords],
            "fusion_votes": len(cl),
        })
    return out


def attractor_suppress(
    detections: Sequence[Mapping[str, Any]],
    attractors: set[int] = ATTRACTOR_CLASSES,
    displacers: set[int] = DISPLACERS,
    attractor_iou: float = ATTRACTOR_IOU,
    margin: float = ATTRACTOR_MARGIN,
) -> list[dict[str, Any]]:
    """Drop an attractor-class box when a specific class labels the same object."""
    out: list[dict[str, Any]] = []
    for d in detections:
        if int(d["class_id"]) not in attractors:
            out.append(dict(d))
            continue
        clash = any(
            int(k["class_id"]) in displacers
            and box_iou(k["bbox"], d["bbox"]) >= attractor_iou
            and float(k["confidence"]) >= float(d["confidence"]) - margin
            for k in detections
        )
        if not clash:
            out.append(dict(d))
    return out


def fuse_views(
    detections_by_view: Mapping[str, Sequence[Mapping[str, Any]]],
    min_views: int = 3,
    width: int | None = None,
    height: int | None = None,
    calibration_by_view: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    class_names: Sequence[str] | None = None,
    arbitration: bool = True,
) -> list[dict[str, Any]]:
    """Full recommended pipeline. Returns the fused detection list for one frame."""
    prepared: dict[str, list[dict[str, Any]]] = {}
    for view, dets in detections_by_view.items():
        rows = [dict(d) for d in dets]
        cal = (calibration_by_view or {}).get(view)
        if cal and class_names and width and height:
            rows = calibrate(rows, cal, class_names, width, height)
        prepared[view] = rows
    fused = wbf(vote_gate(prepared, min_views=min_views))
    if arbitration:
        fused = attractor_suppress(fused)
    return fused
