#!/usr/bin/env python3
"""剪刀逐帧重标 v2：把"固定模板框"换成"按物体像素收紧的框"。

背景：`labeled_group_color_v1/3-Scissors` 的框是每个姿态组一个**固定框**、整组复制，
不跟随物体。已确认 09-26 的 "134253" 组框上沿切过手柄、下沿多出 0.117 图高空白；
其余组框也普遍偏松（左侧留 50-60 px 空桌面）。

v2 算法（纯 RGB，逐帧自适应）：
  1. ROI = 旧框外扩 35%
  2. 前景 = 局部对比暗区(median blur 背景 - V > max(18, 0.12*bg)) ∪ 高饱和区(手柄橙/蓝)
  3. 连通域筛选：面积≥60、质心落在旧框外扩 15% 内、包围盒≤1.5×旧框、不贴边
  4. 取保留连通域并集外接框 + 3px padding
  5. 可信度闸门：新/旧面积比 ∈[0.45,1.8]，否则回退旧框并告警

输出到独立目录，不动原标签。用法见 --help。
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import cv2
import numpy as np

CLASS_ID = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pad-roi", type=float, default=0.35)
    p.add_argument("--pad-box", type=int, default=3)
    p.add_argument("--min-area", type=int, default=60)
    p.add_argument("--dark-rel", type=float, default=0.12)
    p.add_argument("--dark-abs", type=float, default=18.0)
    p.add_argument("--sat-abs", type=float, default=75.0)
    return p.parse_args()


def read_label(path: Path):
    rows = [l.split() for l in path.read_text().strip().splitlines() if l.strip()]
    assert len(rows) == 1, f"{path}: 期望 1 行，实得 {len(rows)}"
    c = int(rows[0][0]); assert c == CLASS_ID, f"{path}: class={c}"
    return [float(v) for v in rows[0][1:5]]


def box_to_px(box, W, H):
    cx, cy, w, h = box
    return np.array([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H], dtype=float)


def px_to_box(b, W, H):
    x1, y1, x2, y2 = b
    return [((x1+x2)/2)/W, ((y1+y2)/2)/H, (x2-x1)/W, (y2-y1)/H]


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, ix2-ix1)*max(0.0, iy2-iy1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/u if u > 0 else 0.0


def segment(frame, old_px, args):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in old_px]
    w, h = x2-x1, y2-y1
    rx1 = int(max(0, x1-args.pad_roi*w)); rx2 = int(min(W, x2+args.pad_roi*w))
    ry1 = int(max(0, y1-args.pad_roi*h)); ry2 = int(min(H, y2+args.pad_roi*h))
    roi = frame[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:, :, 2].astype(np.float32), (5, 5), 0)
    s = hsv[:, :, 1].astype(np.float32)
    bg = cv2.medianBlur(v.astype(np.uint8), 31).astype(np.float32)
    dark = (bg - v) > np.maximum(args.dark_abs, args.dark_rel*bg)
    sat = s > max(args.sat_abs, float(np.percentile(s, 85)) + 25.0)
    mask = (dark | sat).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    ex = 0.15
    ox1, oy1, ox2, oy2 = x1-ex*w, y1-ex*h, x2+ex*w, y2+ex*h
    old_area = max(1.0, w*h)
    keep = []
    for i in range(1, n):
        bx, by, bw, bh, area = (int(t) for t in stats[i])
        if area < args.min_area:
            continue
        gx1, gy1 = rx1+bx, ry1+by
        gx2, gy2 = rx1+bx+bw, ry1+by+bh
        ccx, ccy = (gx1+gx2)/2, (gy1+gy2)/2
        if not (ox1 <= ccx <= ox2 and oy1 <= ccy <= oy2):
            continue
        if bw > 1.5*w or bh > 1.5*h:
            continue
        if bw*bh > 1.8*old_area:
            continue
        if gx1 <= 0 or gy1 <= 0 or gx2 >= W or gy2 >= H:
            continue
        keep.append((gx1, gy1, gx2, gy2, area))
    if not keep:
        return None, dict(note="no_component")
    p = args.pad_box
    box = [max(0, min(k[0] for k in keep)-p), max(0, min(k[1] for k in keep)-p),
           min(W-1, max(k[2] for k in keep)+p), min(H-1, max(k[3] for k in keep)+p)]
    return box, dict(note="ok", comps=len(keep),
                     mask_area=int(sum(k[4] for k in keep)))


def main() -> int:
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    (out/"images").mkdir(parents=True, exist_ok=True)
    (out/"labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path in sorted((src/"images").glob("*.jpg")):
        lab_path = src/"labels"/(img_path.stem+".txt")
        if not lab_path.exists():
            print("跳过（无标签）", img_path.name); continue
        frame = cv2.imread(str(img_path)); H, W = frame.shape[:2]
        old_box = read_label(lab_path)
        old_px = box_to_px(old_box, W, H)
        new_px, info = segment(frame, old_px, args)
        warn = []
        if new_px is None:
            new_px = [float(v) for v in old_px]; warn.append("keep_old:"+info["note"])
        old_area = max(1.0, (old_px[2]-old_px[0])*(old_px[3]-old_px[1]))
        new_area = max(1.0, (new_px[2]-new_px[0])*(new_px[3]-new_px[1]))
        ratio = new_area/old_area
        if ratio < 0.45 or ratio > 1.8:
            new_px = [float(v) for v in old_px]
            new_area = old_area; ratio = 1.0
            warn.append("revert_ratio")
        new_box = px_to_box(new_px, W, H)
        (out/"labels"/(img_path.stem+".txt")).write_text(
            f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in new_box) + "\n")
        link = out/"images"/img_path.name
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(img_path.resolve())
        rows.append(dict(image=img_path.name,
                         old=[round(v, 6) for v in old_box], new=[round(v, 6) for v in new_box],
                         old_px=[int(v) for v in old_px], new_px=[int(round(v)) for v in new_px],
                         iou_old_new=round(iou_xyxy(old_px, new_px), 3), area_ratio=round(ratio, 3),
                         new_size_px=[int(new_px[2]-new_px[0]), int(new_px[3]-new_px[1])],
                         **info, warn=";".join(warn)))
    (out/"review.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    n_warn = sum(1 for r in rows if r["warn"])
    print(f"{src.parent.parent.name}: {len(rows)} 帧，告警 {n_warn} 帧")
    for r in rows:
        if r["warn"]: print("  WARN", r["image"], r["warn"])
    ar = [r["area_ratio"] for r in rows]
    print("  面积比 中位 %.2f  范围 %.2f-%.2f  IoU(旧,新) 中位 %.2f" % (
        float(np.median(ar)), min(ar), max(ar), float(np.median([r["iou_old_new"] for r in rows]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
