#!/usr/bin/env python3
"""剪刀标注口径核查：量化真值框(GT)与实际物体像素的偏差。

做法（纯只读）：
  1. 读 val 集 32 张剪刀图的 YOLO 标注
  2. 在 GT 外扩 25% 的 ROI 内用 Otsu 取暗色像素（物体+阴影），保留最大连通域
  3. 该连通域的外接矩形当作"贴合物体的理想框(object bbox)"
  4. 比较 GT 与 object bbox：IoU 上限、各边偏移（占图宽/高的比例）

IoU(GT, object bbox) 就是"完美检测器最多能拿到的 IoU"，也就是评分口径下的天花板。
"""
import cv2, numpy as np, os, glob, re, json

DS = "/media/sunshine/新加卷/taskA/datasets/real_ft_v16_mug_20261013"
OUT = "/media/sunshine/新加卷/taskA/reports/scissors_diag_20261016/label_check.md"

def load_gt(txt):
    with open(txt) as f:
        v = f.readline().split()
    c, cx, cy, w, h = int(v[0]), *map(float, v[1:5])
    return c, np.array([cx, cy, w, h])

def iou_xywh(a, b):  # normalized xywh
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0, min(ax2, bx2)-max(ax1, bx1)); iy = max(0, min(ay2, by2)-max(ay1, by1))
    inter = ix*iy
    ua = a[2]*a[3]+b[2]*b[3]-inter
    return inter/ua if ua > 0 else 0.0

def object_bbox(img, gt):
    H, W = img.shape[:2]
    cx, cy, w, h = gt[0]*W, gt[1]*H, gt[2]*W, gt[3]*H
    pad = 0.25
    x1 = int(max(0, cx-w/2 - pad*w)); x2 = int(min(W, cx+w/2 + pad*w))
    y1 = int(max(0, cy-h/2 - pad*h)); y2 = int(min(H, cy+h/2 + pad*h))
    roi = img[y1:y2, x1:x2]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(( (_ > 0).astype(np.uint8) ), 8)
    if n <= 1:
        return None, t
    # 最大连通域（排除背景 label 0）
    idx = 1 + int(np.argmax(stats[1:, 4]))
    bx, by, bw, bh = stats[idx, 0], stats[idx, 1], stats[idx, 2], stats[idx, 3]
    # 回全图归一化
    nb = np.array([(x1+bx+bw/2)/W, (y1+by+bh/2)/H, bw/W, bh/H])
    return nb, t

rows = []
for txt in sorted(glob.glob(os.path.join(DS, "labels/val/*scissors*"))):
    stem = os.path.basename(txt)[:-4]
    imgp = os.path.join(DS, "images/val", stem + ".jpg")
    img = cv2.imread(imgp)
    if img is None:
        continue
    c, gt = load_gt(txt)
    ob, t = object_bbox(img, gt)
    if ob is None:
        continue
    H, W = img.shape[:2]
    gx1, gy1 = gt[0]-gt[2]/2, gt[1]-gt[3]/2
    gx2, gy2 = gt[0]+gt[2]/2, gt[1]+gt[3]/2
    ox1, oy1 = ob[0]-ob[2]/2, ob[1]-ob[3]/2
    ox2, oy2 = ob[0]+ob[2]/2, ob[1]+ob[3]/2
    batch = "v5_0926" if "20260926" in stem else "v9_0927"
    rows.append(dict(
        stem=stem, batch=batch,
        gt=[round(float(x), 4) for x in gt], obj=[round(float(x), 4) for x in ob],
        iou_ceiling=round(iou_xywh(gt, ob), 3),
        d_left=round((ox1-gx1), 3), d_right=round((gx2-ox2), 3),
        d_top=round((oy1-gy1), 3), d_bottom=round((gy2-oy2), 3),
        gt_area=round(float(gt[2]*gt[3]), 3), obj_area=round(float(ob[2]*ob[3]), 3),
    ))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
def agg(batch):
    r = [x for x in rows if x["batch"] == batch]
    ce = [x["iou_ceiling"] for x in r]
    return dict(n=len(r), ceil_median=round(float(np.median(ce)), 3), ceil_min=round(float(np.min(ce)), 3),
                ceil_max=round(float(np.max(ce)), 3),
                n_ge_075=int(sum(1 for x in ce if x >= 0.75)),
                obj_area_median=round(float(np.median([x["obj_area"] for x in r])), 3),
                gt_area_median=round(float(np.median([x["gt_area"] for x in r])), 3),
                d_bottom_median=round(float(np.median([x["d_bottom"] for x in r])), 3),
                d_top_median=round(float(np.median([x["d_top"] for x in r])), 3))

summ = {b: agg(b) for b in ("v5_0926", "v9_0927")}
with open(OUT, "w") as f:
    f.write("# 剪刀真值框 vs 物体像素（标注口径核查）\n\n")
    f.write("`iou_ceiling` = 贴合物体的理想框与真值框的 IoU，即评分口径下完美检测器的上限。\n\n")
    for b, s in summ.items():
        f.write(f"## {b}\n{json.dumps(s, ensure_ascii=False, indent=2)}\n\n")
    f.write("| 图像 | 批次 | GT(w,h) | 物体(w,h) | iou_ceiling | 左/右/上/下边 真值超出量 |\n|---|---|---|---|---|---|\n")
    for x in rows:
        f.write(f"| {x['stem'][-38:]} | {x['batch']} | {x['gt'][2]:.3f},{x['gt'][3]:.3f} | {x['obj'][2]:.3f},{x['obj'][3]:.3f} | "
                f"{x['iou_ceiling']:.3f} | {x['d_left']:.3f}/{x['d_right']:.3f}/{x['d_top']:.3f}/{x['d_bottom']:.3f} |\n")
json.dump(dict(summary=summ, rows=rows), open(os.path.join(os.path.dirname(OUT), "label_check.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=2))
