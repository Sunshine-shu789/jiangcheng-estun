#!/usr/bin/env python3
"""对比拼图：红=旧（模板）框，绿=新（逐帧收紧）框。"""
import cv2, numpy as np, sys, json
from pathlib import Path
src = Path(sys.argv[1]); out = Path(sys.argv[2]); cols = int(sys.argv[3]) if len(sys.argv)>3 else 4
step = int(sys.argv[4]) if len(sys.argv)>4 else 1
rows = json.loads((src/"review.json").read_text())[::step]
tiles=[]
for r in rows:
    img = cv2.imread(str((src/"images"/r["image"]).resolve()))
    H,W = img.shape[:2]
    cv2.rectangle(img, (int(r["old_px"][0]),int(r["old_px"][1])), (int(r["old_px"][2]),int(r["old_px"][3])), (0,0,255), 2)
    cv2.rectangle(img, (int(r["new_px"][0]),int(r["new_px"][1])), (int(r["new_px"][2]),int(r["new_px"][3])), (0,255,0), 2)
    tag = f'{r["image"].split("_")[-2]}-{r["image"][-10:-4]} r={r["area_ratio"]:.2f}'
    cv2.putText(img, tag, (5,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
    tiles.append(cv2.resize(img,(320,240)))
rowsimg=[np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]
rowsimg=[r for r in rowsimg if r.shape[1]==320*cols]
cv2.imwrite(str(out), np.vstack(rowsimg))
print("saved", out, len(tiles), "tiles")
