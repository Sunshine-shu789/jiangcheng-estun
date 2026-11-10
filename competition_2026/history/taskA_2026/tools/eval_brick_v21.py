#!/usr/bin/env python3
"""泡沫砖姿态复测（v20 vs v21 用同一把尺子）。

协议（与 9100 服务一致的双尺度稳定检测）:
  对每张 640x480 原图，分别按 scale=0.75 / 0.65 缩放居中贴到 640x480 灰底画布，
  imgsz=640 推理，conf>=阈值；把框映射回原图坐标后合并两尺度结果。
  砖命中判据: class5 框与人工框 IoU>=0.3（中心点也认）。
空背景: 12 张训练未见的留出图（另有 14 张已进训练，单独标注）。
用法: eval_brick_v21.py <weights> [conf]
"""
import cv2, numpy as np, json, sys, os
from collections import defaultdict
from ultralytics import YOLO

T = "/media/sunshine/新加卷/taskA"
RAW = f"{T}/real_raw/6-Foam_Brick"
EMPTY = f"{T}/real_raw/empty_background"
BOXJSON = "/home/sunshine/codex/easton/taskA/prelabels_v21/manual_boxes_v2.json"
HOLDOUT = "/home/sunshine/codex/easton/taskA/scratch/v21_empty_holdout_20261016.txt"
POSE = {
    "中心竖放": ["015555_855884","015556_244603","015556_632185","015557_006862","015557_378440","015557_758709","015558_141195","015558_518728","015558_891405","015559_273953"],
    "中心横放": ["015651_427267","015651_806864","015652_181086"],
    "左上":     ["015946_619787","015946_992055"],
    "右上":     ["020108_081864","020108_464841"],
    "左下":     ["020157_646343","020158_025628"],
    "右下":     ["020243_834170","020244_213967"],
    "立放":     ["021651_419155","021651_797070","021652_180906"],
    "斜靠":     ["021735_717901","021736_092395","021736_469985"],
    "并排罐头": ["022140_850009","022141_233823"],
    "罐头遮挡": ["022313_628794","022314_004847"],
}
VAL_HELD = {g[-1] for g in POSE.values()}          # 进 val 的 10 张（held-out）
SCALES = (0.75, 0.65)
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
W = sys.argv[1]

def prep(frame, s):
    h, w = frame.shape[:2]
    sw, sh = int(round(w * s)), int(round(h * s))
    canvas = np.full((h, w, 3), 114, np.uint8)
    xo, yo = (w - sw) // 2, (h - sh) // 2
    canvas[yo:yo+sh, xo:xo+sw] = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    return canvas, xo, yo

def predict_boxes(model, frame):
    out = []
    for s in SCALES:
        canvas, xo, yo = prep(frame, s)
        r = model.predict(source=canvas, imgsz=640, conf=CONF, device=0, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append((int(b.cls.item()), float(b.conf.item()),
                        (x1-xo)/s, (y1-yo)/s, (x2-xo)/s, (y2-yo)/s))
    return out

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.

boxes = json.load(open(BOXJSON))
model = YOLO(W)
print(f"### weights={W} conf={CONF} scales={SCALES}")
tot_hit = tot_img = 0
rows = []
for pose, tags in POSE.items():
    hit = 0
    for tag in tags:
        fn = [f for f in os.listdir(RAW) if f.endswith(".jpg") and f[:-4][-13:] == tag][0]
        img = cv2.imread(os.path.join(RAW, fn)); assert img is not None, fn
        gt = [tuple(b[1:]) for b in boxes[fn] if b[0] == 5]
        det = [b for b in predict_boxes(model, img) if b[0] == 5]
        ok = any(iou(d[2:], g) >= 0.3 or
                 (g[0] < (d[2]+d[4])/2 < g[2] and g[1] < (d[3]+d[5])/2 < g[3])
                 for d in det for g in gt)
        hit += ok
        if tag not in VAL_HELD:
            pass
    tot_hit += hit; tot_img += len(tags)
    rows.append((pose, hit, len(tags)))
    mark = sum(1 for t in tags if t in VAL_HELD)
    print(f"{pose:6s} {hit}/{len(tags)}   其中留出评测 {mark} 张")

# held-out 口径：每姿态只取进 val 的那 1 张
print("\n--- held-out（每姿态 1 张，共 10 张，未参与训练）---")
h_hit = 0; h_img = 0
for pose, tags in POSE.items():
    tag = tags[-1]
    fn = [f for f in os.listdir(RAW) if f.endswith(".jpg") and f[:-4][-13:] == tag][0]
    img = cv2.imread(os.path.join(RAW, fn))
    gt = [tuple(b[1:]) for b in boxes[fn] if b[0] == 5]
    det = [b for b in predict_boxes(model, img) if b[0] == 5]
    ok = any(iou(d[2:], g) >= 0.3 for d in det for g in gt)
    h_hit += ok; h_img += 1
    print(f"  {pose:6s} {'命中' if ok else '漏检'}")
print(f"  合计 {h_hit}/{h_img} = {100*h_hit/h_img:.1f}%")

print("\n--- 空背景误检（class5）---")
hold = [l.strip() for l in open(HOLDOUT) if l.strip()]
for tag, names in (("留出12张(未训练)", hold),
                   ("训练内14张", sorted(n for n in os.listdir(EMPTY) if n not in hold))):
    fp = 0; conf_max = 0.0; details = []
    for n in names:
        img = cv2.imread(os.path.join(EMPTY, n))
        d = [b for b in predict_boxes(model, img) if b[0] == 5]
        if d:
            fp += 1
            best = max(d, key=lambda x: x[1])
            conf_max = max(conf_max, best[1])
            details.append((n, best[1], best[2:]))
    print(f"  {tag}: {fp}/{len(names)} = {100*fp/len(names):.1f}%" +
          (f"（最高conf {conf_max:.2f}）" if fp else ""))
    for n, c, box in sorted(details, key=lambda x: -x[1])[:15]:
        print(f"      {n} conf={c:.3f} box={tuple(round(v) for v in box)}")
        ann = cv2.imread(os.path.join(EMPTY, n))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(ann, f"brick {c:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        os.makedirs("/tmp/brick_fp_v22", exist_ok=True)
        cv2.imwrite(f"/tmp/brick_fp_v22/{n}", ann)
print(f"\n全部31张（训练+val 混合）: {tot_hit}/{tot_img} = {100*tot_hit/tot_img:.1f}%")
