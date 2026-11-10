#!/usr/bin/env python3
"""从砖体内部种子做局部颜色泛洪 -> 紧致外接框。用于人工框的定量校验/精修建议。"""
import cv2, numpy as np, json, os, sys

RAW="/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick"
J=json.load(open("/home/sunshine/codex/easton/taskA/prelabels_v21/manual_boxes.json"))

def brick_bbox(img, box, tol=(16,14,14), debug=False):
    x1,y1,x2,y2=[int(v) for v in box]
    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h,w=lab.shape[:2]
    # 在框内找种子：取框内 3x3 平均最接近框内中位数颜色的点，且远离边界
    m=cv2.inRange(np.zeros((1,1),np.uint8),0,0)
    roi=lab[y1:y2,x1:x2].reshape(-1,3).astype(int)
    med=np.median(roi,axis=0)
    d=np.abs(roi-med).sum(1)
    idx=np.argsort(d)[:max(1,len(d)//50)]      # 最接近中位色的 2% 像素
    ys,xs=np.divmod(idx, x2-x1)
    # 取这些点里最靠近框中心的
    cx,cy=(x2-x1)/2,(y2-y1)/2
    k=np.argmin((xs-cx)**2+(ys-cy)**2)
    sx,sy=int(x1+xs[k]), int(y1+ys[k])
    mask=np.zeros((h+2,w+2),np.uint8)
    flags=4|cv2.FLOODFILL_MASK_ONLY|(255<<8)
    lo,up=tol[0],tol[1]
    cv2.floodFill(lab.copy(), mask, (sx,sy), 0, (int(lo),int(lo),int(lo)), (int(up),int(up),int(up)), flags)
    m=mask[1:-1,1:-1]
    m=cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    m=cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))
    n,labl,stats,cent=cv2.connectedComponentsWithStats(m,8)
    i=labl[sy,sx]
    if i==0:
        return None
    x,y,ww,hh,area=stats[i]
    return [int(x),int(y),int(x+ww),int(y+hh)], (sx,sy), int(area)

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua else 0

def main(names=None, tol=(16,14,14), save=None):
    out={}
    for name,ann in J.items():
        if names and not any(t in name for t in names): continue
        img=cv2.imread(os.path.join(RAW,name))
        if img is None: print("MISS",name); continue
        for item in ann:
            cls=item[0]; b=item[1:5]
            r=brick_bbox(img,b,tol)
            if r is None: print(f"{name} cls{cls} box={b} -> FAIL"); continue
            tb,seed,area=r
            v=iou(b,list(tb))
            print(f"{name[-13:-4]} cls{cls} box={b} flood={list(tb)} seed={seed} area={area} IoU={v:.3f}")
            out.setdefault(name,[]).append((cls,b,tb,v))
    if save:
        json.dump({k:[(c,b,t) for c,b,t,v in v_] for k,v_ in out.items()}, open(save,'w'), indent=1)
    return out

if __name__=="__main__":
    names=sys.argv[1:] or None
    main(names)
