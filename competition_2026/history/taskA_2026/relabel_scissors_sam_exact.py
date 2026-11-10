#!/usr/bin/env python3
"""剪刀逐帧重标 v4：用**原标签框本身**作为 MobileSAM bbox prompt，取最大连通域外接框。

为什么不用扩张框：09-26 的旧框本身较松，扩张后 SAM 会把桌面/背景当成主体；
用原框做 prompt 时，SAM 能在框内正确分出剪刀（已逐帧可视化验证）。
输出到独立目录，原标签不动。
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

CLASS_ID = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default="mobile_sam.pt")
    p.add_argument("--pad", type=int, default=2)
    # bbox 面积比 = SAM 外接框面积 / 旧框面积，低于/高于该范围视为分割失败，保留旧框
    p.add_argument("--min-bbox-ratio", type=float, default=0.15)
    p.add_argument("--max-bbox-ratio", type=float, default=1.30)
    # mask 外接框相对旧框向外溢出的容忍比例
    p.add_argument("--outside-tol", type=float, default=0.25)
    p.add_argument("--expand", type=float, default=0.0,
                   help="先试用旧框外扩比例；0 表示只试原框")
    p.add_argument("--fallback-expand", type=float, default=0.10,
                   help="原框分割失败时再尝试的外扩比例（0 禁用）")
    return p.parse_args()


def read_label(path: Path):
    rows = [l.split() for l in path.read_text().strip().splitlines() if l.strip()]
    assert len(rows) == 1 and int(rows[0][0]) == CLASS_ID, path
    return [float(v) for v in rows[0][1:5]]


def box_to_px(box, W, H):
    cx, cy, w, h = box
    return [float((cx-w/2)*W), float((cy-h/2)*H), float((cx+w/2)*W), float((cy+h/2)*H)]


def px_to_box(b, W, H):
    x1, y1, x2, y2 = b
    return [((x1+x2)/2)/W, ((y1+y2)/2)/H, (x2-x1)/W, (y2-y1)/H]


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, ix2-ix1)*max(0.0, iy2-iy1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/u if u > 0 else 0.0


def expand_box(b, r, W, H):
    x1, y1, x2, y2 = b
    dx, dy = (x2-x1)*r, (y2-y1)*r
    return [max(0.0, x1-dx), max(0.0, y1-dy), min(W-1.0, x2+dx), min(H-1.0, y2+dy)]


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max())+1, float(ys.max())+1]


def biggest_component(m):
    m = (m > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    i = 1 + int(np.argmax(stats[1:, 4]))
    return (lab == i).astype(np.uint8)


def main() -> int:
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    (out/"images").mkdir(parents=True, exist_ok=True)
    (out/"labels").mkdir(parents=True, exist_ok=True)
    model = SAM(args.weights)
    rows = []
    for img_path in sorted((src/"images").glob("*.jpg")):
        lab_path = src/"labels"/(img_path.stem+".txt")
        if not lab_path.exists():
            continue
        frame = cv2.imread(str(img_path)); H, W = frame.shape[:2]
        old_box = read_label(lab_path); old_px = box_to_px(old_box, W, H)
        ow, oh = old_px[2]-old_px[0], old_px[3]-old_px[1]
        old_area = max(1.0, ow*oh)

        prompts = []
        if args.expand > 0:
            prompts.append(("expand", expand_box(old_px, args.expand, W, H)))
        prompts.append(("exact", list(old_px)))
        if args.fallback_expand > 0:
            prompts.append(("fallback_expand", expand_box(old_px, args.fallback_expand, W, H)))

        chosen = None
        for tag, prompt in prompts:
            res = model(str(img_path), bboxes=[prompt], verbose=False)[0]
            if res.masks is None or len(res.masks.data) == 0:
                continue
            m = biggest_component(res.masks.data[0].cpu().numpy())
            if m.sum() == 0:
                continue
            mb = mask_bbox(m)
            bw, bh = mb[2]-mb[0], mb[3]-mb[1]
            bbox_ratio = (bw*bh)/old_area
            outside = 0.0
            if bw > 0 and bh > 0:
                left = max(0.0, old_px[0]-mb[0])/max(1.0, ow)
                top = max(0.0, old_px[1]-mb[1])/max(1.0, oh)
                right = max(0.0, mb[2]-old_px[2])/max(1.0, ow)
                bottom = max(0.0, mb[3]-old_px[3])/max(1.0, oh)
                outside = max(left, top, right, bottom)
            valid = (args.min_bbox_ratio <= bbox_ratio <= args.max_bbox_ratio and outside <= args.outside_tol)
            cand = dict(tag=tag, prompt=[round(v,1) for v in prompt], mask=m, mbox=mb,
                        bbox_ratio=round(bbox_ratio,4), outside=round(outside,4), valid=valid)
            if valid:
                chosen = cand
                break
            if chosen is None:
                chosen = cand  # keep first for diagnostics, but mark invalid
        warn = []
        if chosen is not None and chosen["valid"]:
            p = args.pad
            new_px = [max(0.0, chosen["mbox"][0]-p), max(0.0, chosen["mbox"][1]-p),
                      min(float(W-1), chosen["mbox"][2]+p), min(float(H-1), chosen["mbox"][3]+p)]
            mask_area = int(chosen["mask"].sum())
            bbox_ratio = chosen["bbox_ratio"]
            outside = chosen["outside"]
            used = chosen["tag"]
        else:
            new_px = list(old_px); mask_area = 0
            bbox_ratio = chosen["bbox_ratio"] if chosen else 0.0
            outside = chosen["outside"] if chosen else 0.0
            used = chosen["tag"] if chosen else "none"
            warn.append("sam_invalid_keep_old")
        new_area = max(1.0, (new_px[2]-new_px[0])*(new_px[3]-new_px[1]))
        ratio = new_area/old_area
        new_box = px_to_box(new_px, W, H)
        (out/"labels"/(img_path.stem+".txt")).write_text(
            f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in new_box) + "\n")
        link = out/"images"/img_path.name
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(img_path.resolve())
        rows.append(dict(image=img_path.name, old=[round(v, 6) for v in old_box],
                         new=[round(v, 6) for v in new_box],
                         old_px=[int(v) for v in old_px], new_px=[int(round(v)) for v in new_px],
                         iou_old_new=round(iou_xyxy(old_px, new_px), 3), area_ratio=round(ratio, 3),
                         mask_area=mask_area, bbox_ratio=bbox_ratio, outside=outside,
                         prompt=used, warn=";".join(warn)))
    (out/"review.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    ar = [r["area_ratio"] for r in rows]
    print(f"{src.parent.parent.name}: {len(rows)} 帧，告警 {sum(1 for r in rows if r['warn'])} 帧")
    for r in rows:
        if r["warn"]: print("  WARN", r["image"], r["warn"])
    print("  面积比 中位 %.2f 范围 %.2f-%.2f | IoU(旧,新) 中位 %.2f | bbox_ratio 中位 %.2f" % (
        float(np.median(ar)), min(ar), max(ar), float(np.median([r["iou_old_new"] for r in rows])),
        float(np.median([r["bbox_ratio"] for r in rows]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
