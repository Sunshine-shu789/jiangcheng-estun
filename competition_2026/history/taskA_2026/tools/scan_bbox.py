#!/usr/bin/env python3
"""按颜色分类逐行扫描，输出砖/罐的可见外接框（含每行 run，便于人工确认）。"""
import cv2, numpy as np, os, sys
RAW="/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick"

def masks(img):
    b,g,r=[c.astype(int) for c in cv2.split(img)]
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v=[c.astype(int) for c in cv2.split(hsv)]
    # 砖：粉棕，R>G 且 R>B，饱和中等，亮度中等
    brick=(r>g+8)&(r>b+2)&(s>28)&(s<170)&(v>60)&(v<215)&(g-b>-12)
    # 罐头：蓝色标签 或 高亮低饱和金属
    blue=(h>92)&(h<135)&(s>60)&(v>45)
    return brick, blue

def scan(name, kind, y0,y1, step=8, xr=(0,640)):
    f=[n for n in os.listdir(RAW) if name in n][0]
    img=cv2.imread(os.path.join(RAW,f))
    brick,blue=masks(img)
    m = brick if kind=='brick' else blue
    X0,X1=xr
    print(f"--- {name} [{kind}]")
    ys=[]
    for y in range(y0,y1,step):
        row=m[y,X0:X1]
        idx=np.where(row)[0]
        if len(idx)==0: continue
        # 取最长连续 run
        runs=np.split(idx,np.where(np.diff(idx)>12)[0]+1)
        run=max(runs,key=len)
        if len(run)<20: continue
        print(f"  y={y:3d}: x {X0+int(run[0]):3d} .. {X0+int(run[-1]):3d}  (len {len(run)})")
        ys.append(y)
    if ys: print(f"  => y range {min(ys)}..{max(ys)}")

if __name__=="__main__":
    a=sys.argv[1:]
    scan(a[0],a[1],int(a[2]),int(a[3]),int(a[4]) if len(a)>4 else 8)
