#!/usr/bin/env python3
"""从 manual_boxes_v2.json 重生成 YOLO labels + 可视化 montage（31 张泡沫砖图）。

输出:
  prelabels_v21/labels/*.txt        覆盖 v1 结果
  prelabels_v21/vis_v2/*.jpg        单图可视化
  /tmp/final_ds/labels_v2_*.jpg     8图/版 montage 复核
"""
import cv2, numpy as np, os, json

ROOT = "/home/sunshine/codex/easton/taskA"
RAW  = "/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick"
J = os.path.join(ROOT, "prelabels_v21/manual_boxes_v2.json")
LBL = os.path.join(ROOT, "prelabels_v21/labels")
VIS = os.path.join(ROOT, "prelabels_v21/vis_v2")
MONT = "/tmp/final_ds"

d = json.load(open(J))
os.makedirs(LBL, exist_ok=True)
os.makedirs(VIS, exist_ok=True)
os.makedirs(MONT, exist_ok=True)

names = {5: "brick", 9: "can"}
n_boxes = {5: 0, 9: 0}
tiles, tags = [], []
for fn in sorted(d):
    p = os.path.join(RAW, fn)
    img = cv2.imread(p)
    assert img is not None, p
    h, w = img.shape[:2]
    lines = []
    for cls, x1, y1, x2, y2 in d[fn]:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        assert x2 > x1 and y2 > y1, (fn, cls, x1, y1, x2, y2)
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        n_boxes[cls] += 1
        col = (0, 255, 0) if cls == 5 else (255, 140, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        cv2.putText(img, names[cls], (x1 + 2, max(11, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    with open(os.path.join(LBL, fn[:-4] + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    cv2.imwrite(os.path.join(VIS, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    tiles.append(cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
    tags.append(fn[-17:-4])

n = 0
for i in range(0, len(tiles), 8):
    chunk = tiles[i:i+8]
    while len(chunk) < 8:
        chunk.append(np.zeros_like(tiles[0]))
    rows = [cv2.hconcat(chunk[j:j+2]) for j in range(0, 8, 2)]
    cv2.imwrite(f"{MONT}/labels_v2_{i//8+1}.jpg", cv2.vconcat(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"labels_v2_{i//8+1}.jpg", tags[i:i+8])
    n += 1

print(f"images={len(d)} brick={n_boxes[5]} can={n_boxes[9]} montage={n}")
