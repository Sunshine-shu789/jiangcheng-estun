#!/usr/bin/env python3
import cv2, os, sys, json
RAW="/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick"
# tag -> [(cls,x1,y1,x2,y2),...]
TRY={
 "022140_850009":[(5,58,244,368,406),(9,148,48,330,215)],
 "022141_233823":[(5,58,244,368,406),(9,148,48,330,215)],
 "022313_628794":[(5,162,130,400,448),(9,350,40,570,250)],
 "022314_004847":[(5,162,130,400,448),(9,350,40,570,250)],
}
tiles=[]
for tag,boxes in TRY.items():
    f=[n for n in os.listdir(RAW) if tag in n][0]
    img=cv2.imread(os.path.join(RAW,f))
    for cls,x1,y1,x2,y2 in boxes:
        col=(0,255,0) if cls==5 else (255,120,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),col,1)
        cv2.putText(img,f"{'brick' if cls==5 else 'can'}",(x1+2,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
    for x in range(0,640,64): cv2.line(img,(x,0),(x,480),(0,90,255),1)
    for y in range(0,480,64): cv2.line(img,(0,y),(640,y),(0,90,255),1)
    for x in range(0,640,128): cv2.putText(img,str(x),(x+2,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,90,255),1)
    for y in range(0,480,64): cv2.putText(img,str(y),(2,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,90,255),1)
    cv2.putText(img,tag,(340,470),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    tiles.append(cv2.resize(img,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC))
top=cv2.hconcat(tiles[:2]); bot=cv2.hconcat(tiles[2:])
cv2.imwrite("/tmp/try_dual.jpg",cv2.vconcat([top,bot]),[cv2.IMWRITE_JPEG_QUALITY,92])
print("ok")
