#!/usr/bin/env python3
"""按姿态取代表帧，2x放大 + 32px坐标网格 + 现框，用于人工精确复核。"""
import cv2, numpy as np, json, os
RAW="/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick"
J=json.load(open("/home/sunshine/codex/easton/taskA/prelabels_v21/manual_boxes.json"))
reps=["015555_855884","015651_427267","015946_619787","020108_081864","020157_646343",
      "020243_834170","020244_213967","021651_419155","021735_717901","022140_850009","022313_628794"]
os.makedirs("/tmp/zoom",exist_ok=True)
for i,tag in enumerate(reps,1):
    name=[k for k in J if tag in k][0]
    img=cv2.imread(os.path.join(RAW,name))
    for item in J[name]:
        cls,x1,y1,x2,y2=item
        c=(0,255,0) if cls==5 else (255,120,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),c,1)
        cv2.putText(img,f"{'brick' if cls==5 else 'can'}",(x1+2,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1)
    big=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    for x in range(0,640,32):
        cv2.line(big,(x*2,0),(x*2,960),(0,90,255),1)
        if x%64==0: cv2.putText(big,str(x),(x*2+2,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,90,255),1)
    for y in range(0,480,32):
        cv2.line(big,(0,y*2),(1280,y*2),(0,90,255),1)
        if y%64==0: cv2.putText(big,str(y),(2,y*2+16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,90,255),1)
    cv2.putText(big,f"#{i} {tag}  box={J[name]}",(6,952),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imwrite(f"/tmp/zoom/{i:02d}_{tag}.jpg",big,[cv2.IMWRITE_JPEG_QUALITY,92])
    print(i,tag,J[name])
