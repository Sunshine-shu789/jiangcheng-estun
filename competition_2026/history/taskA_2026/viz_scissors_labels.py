#!/usr/bin/env python3
"""可视化：真值框 vs 暗像素外接框（用于人工核对标注口径）"""
import cv2, numpy as np, os, glob, json, sys
sys.path.insert(0, "/home/sunshine/codex/easton/taskA")
from diag_scissors_labels import load_gt, object_bbox, iou_xywh
DS = "/media/sunshine/新加卷/taskA/datasets/real_ft_v16_mug_20261013"
OUT = "/media/sunshine/新加卷/taskA/reports/scissors_diag_20261016/label_check_overlay.jpg"
imgs = []
for txt in sorted(glob.glob(os.path.join(DS, "labels/val/*scissors*"))):
    stem = os.path.basename(txt)[:-4]
    img = cv2.imread(os.path.join(DS, "images/val", stem + ".jpg"))
    if img is None: continue
    c, gt = load_gt(txt); H, W = img.shape[:2]
    ob, _ = object_bbox(img, gt)
    if ob is None: continue
    box = [[int(gt[0]*W), int(gt[1]*H), int(gt[2]*W), int(gt[3]*H)], [str(round(iou_xywh(gt, ob),3))]]
    imgs.append((stem, img.copy(), gt, ob, box[1][0]))
tiles = []
for stem, img, gt, ob, ce in imgs:
    H, W = img.shape[:2]
    g = np.array([int(gt[0]*W), int(gt[1]*H), int(gt[2]*W), int(gt[3]*H)])
    o = np.array([int(ob[0]*W), int(ob[1]*H), int(ob[2]*W), int(ob[3]*H)])
    cv2.rectangle(img, (int(g[0]-g[2]/2), int(g[1]-g[3]/2)), (int(g[0]+g[2]/2), int(g[1]+g[3]/2)), (0,255,0), 2)
    cv2.rectangle(img, (int(o[0]-o[2]/2), int(o[1]-o[3]/2)), (int(o[0]+o[2]/2), int(o[1]+o[3]/2)), (255,0,0), 2)
    cv2.putText(img, f"{stem[-30:]} ceil={ce}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    tiles.append(cv2.resize(img, (400, 300)))
rows = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
rows = [r for r in rows if r.shape[1] == 1200]
cv2.imwrite(OUT, np.vstack(rows))
print("saved", OUT, len(tiles), "tiles")
