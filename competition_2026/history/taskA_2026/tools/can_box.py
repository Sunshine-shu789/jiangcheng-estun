#!/usr/bin/env python3
import cv2, numpy as np, os
RAW="/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick"
def can_bbox(img, roi=None, pad=8):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); h,s,v=[c.astype(int) for c in cv2.split(hsv)]
    blue=((h>90)&(h<140)&(s>55)&(v>40)).astype(np.uint8)*255
    if roi is not None:
        x1,y1,x2,y2=roi; m=np.zeros_like(blue); m[y1:y2,x1:x2]=blue[y1:y2,x1:x2]; blue=m
    blue=cv2.morphologyEx(blue,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    n,l,st,c=cv2.connectedComponentsWithStats(blue,8)
    if n<2: return None,0
    i=1+int(np.argmax(st[1:,4])); x,y,w,hh,a=st[i]
    x1=max(0,x-pad); y1=max(0,y-pad); x2=min(img.shape[1],x+w+pad); y2=min(img.shape[0],y+hh+pad)
    return [int(x1),int(y1),int(x2),int(y2)], int(a)
ROIS={"022140_850009":(110,30,330,230),"022141_233823":(90,30,320,230),
      "022313_628794":(320,30,600,330),"022314_004847":(320,20,600,330)}
res={}
for tag,roi in ROIS.items():
    f=[n for n in os.listdir(RAW) if tag in n][0]
    img=cv2.imread(os.path.join(RAW,f))
    b,a=can_bbox(img,roi)
    res[tag]=b
    print(tag,"can=",b,"blueArea=",a)
