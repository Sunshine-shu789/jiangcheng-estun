#!/usr/bin/env python3
"""框贴合度审计：以人工框内主色为种子，全图同色连通域 -> 紧致外接矩形，与人工框比对 IoU。"""
import cv2, numpy as np, json, sys, os

RAW = "/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick"
JSONP = "/home/sunshine/codex/easton/taskA/prelabels_v21/manual_boxes.json"
OUTDIR = "/tmp/audit"

def dom_color(hsv, box):
    x1,y1,x2,y2 = box
    roi = hsv[max(0,y1):y2, max(0,x1):x2].reshape(-1,3)
    m = (roi[:,1] > 35) & (roi[:,2] > 35)
    r = roi[m]
    if len(r) < 50: return None
    h = r[:,0]
    # 只在"偏红/粉"一侧找峰（OpenCV H: 0-179，砖大约 0-15 或 165-179 的粉色系）
    hist = np.bincount(h, minlength=180)
    # 环形平滑
    hist = (np.roll(hist,1)+hist+np.roll(hist,-1))
    peak = int(np.argmax(hist))
    sel = r[np.minimum(np.abs(h.astype(int)-peak), 180-np.abs(h.astype(int)-peak)) <= 8]
    if len(sel) < 50: sel = r
    hlo, hhi = np.percentile(sel[:,0], [5,95])
    slo, shi = np.percentile(sel[:,1], [3,97])
    vlo, vhi = np.percentile(sel[:,2], [3,97])
    return peak, hlo, hhi, slo, shi, vlo, vhi

def tight_bbox(hsv, box, seed):
    peak,hlo,hhi,slo,shi,vlo,vhi = seed
    h = hsv[:,:,0].astype(int); s = hsv[:,:,1]; v = hsv[:,:,2]
    dh = np.minimum(np.abs(h-peak), 180-np.abs(h-peak))
    m = ((dh <= 10) & (s>=max(25,slo-30)) & (s<=min(255,shi+30)) &
         (v>=max(25,vlo-35)) & (v<=min(255,vhi+25))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    x1,y1,x2,y2 = box
    best = None
    for i in range(1,n):
        x,y,w,hh,area = stats[i]
        if area < 200: continue
        ix1,iy1 = max(x,x1), max(y,y1); ix2,iy2 = min(x+w,x2), min(y+hh,y2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        boxarea = (x2-x1)*(y2-y1)
        if inter/boxarea > 0.45 and inter/area > 0.6:
            if best is None or area > best[1]: best = (i, area, (x,y,x+w,y+hh))
    if best is None: return None
    return best[2]

def iou(a,b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1]); ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua>0 else 0

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    boxes = json.load(open(JSONP))
    rows = []
    for name, ann in boxes.items():
        p = os.path.join(RAW, name)
        img = cv2.imread(p)
        if img is None: print("MISS", name); continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for item in ann:
            cls = item[0]; b = item[1:5]
            if cls != 5: continue
            seed = dom_color(hsv, b)
            tb = tight_bbox(hsv, b, seed) if seed else None
            if tb is None:
                print(f"{name}  MANUAL={b}  TIGHT=None"); rows.append((name,None)); continue
            v = iou(b, list(tb))
            print(f"{name}  MANUAL={[int(x) for x in b]}  TIGHT={list(map(int,tb))}  IoU={v:.3f}")
            rows.append((name,(tb,v)))
    # 汇总
    bad = [(n,t,v) for n,t in [(r[0],r[1][1]) if r[1] and isinstance(r[1],tuple) else (r[0],0.0) for r in rows] for v in [t]]
    print("\nIoU<0.85: ", [f"{n}({v:.2f})" for n,_,v in bad if v<0.85])

if __name__ == "__main__":
    main()
