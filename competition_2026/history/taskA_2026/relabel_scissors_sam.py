#!/usr/bin/env python3
"""剪刀逐帧重标 v3（SAM）：用 old 框当 prompt 让 MobileSAM 分割出整把剪刀，取其外接框。

旧标签的问题：每个姿态组一个**固定框**整组复制 → 框整体偏移（09-26 的 134253 组
上沿切过手柄、下沿多出 85 px 空桌面）。纯 RGB 阈值法抓不到反光刀刃，SAM 可以。

流程：prompt = 旧框外扩 30% → MobileSAM → 取最大连通域外接框 → +2px padding。
若 mask 贴到 prompt 边界（说明被 prompt 截断）则扩大 prompt 重跑（最多 3 次）。
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
    p.add_argument("--expand", type=float, default=0.30)
    p.add_argument("--max-expand", type=int, default=3)
    p.add_argument("--pad", type=int, default=2)
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
        ew, eh = args.expand, args.expand
        mask = None; prompt = None; it = 0
        while it < args.max_expand:
            it += 1
            prompt = [max(0.0, old_px[0]-ew*ow), max(0.0, old_px[1]-eh*oh),
                      min(float(W-1), old_px[2]+ew*ow), min(float(H-1), old_px[3]+eh*oh)]
            res = model(str(img_path), bboxes=[prompt], verbose=False)[0]
            if res.masks is None or len(res.masks.data) == 0:
                ew += 0.2; eh += 0.2; continue
            m = (res.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            if n <= 1:
                ew += 0.2; eh += 0.2; continue
            i = 1 + int(np.argmax(stats[1:, 4]))
            m = (lab == i).astype(np.uint8)
            ys, xs = np.where(m > 0)
            mb = [float(xs.min()), float(ys.min()), float(xs.max())+1, float(ys.max())+1]
            # mask 贴 prompt 边 → 可能被截断，扩大 prompt 重跑
            touch = (mb[0] <= prompt[0]+1.5 or mb[1] <= prompt[1]+1.5 or
                     mb[2] >= prompt[2]-1.5 or mb[3] >= prompt[3]-1.5)
            mask = m; mbox = mb
            if not touch:
                break
            ew += 0.25; eh += 0.25
        warn = []
        if mask is None:
            new_px = list(old_px); warn.append("sam_failed_keep_old")
        else:
            p = args.pad
            new_px = [max(0, mbox[0]-p), max(0, mbox[1]-p),
                      min(W-1, mbox[2]+p), min(H-1, mbox[3]+p)]
        old_area = max(1.0, ow*oh)
        new_area = max(1.0, (new_px[2]-new_px[0])*(new_px[3]-new_px[1]))
        ratio = new_area/old_area
        if ratio < 0.35 or ratio > 1.6:
            new_px = list(old_px); ratio = 1.0; warn.append("revert_ratio")
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
                         passes=it, mask_area=int(mask.sum()) if mask is not None else 0,
                         new_size_px=[int(new_px[2]-new_px[0]), int(new_px[3]-new_px[1])],
                         warn=";".join(warn)))
    (out/"review.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    ar = [r["area_ratio"] for r in rows]
    print(f"{src.parent.parent.name}: {len(rows)} 帧，告警 {sum(1 for r in rows if r['warn'])} 帧")
    for r in rows:
        if r["warn"]: print("  WARN", r["image"], r["warn"])
    print("  面积比 中位 %.2f 范围 %.2f-%.2f | IoU(旧,新) 中位 %.2f | 多轮重跑 %d 帧" % (
        float(np.median(ar)), min(ar), max(ar), float(np.median([r["iou_old_new"] for r in rows])),
        sum(1 for r in rows if r["passes"] > 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
