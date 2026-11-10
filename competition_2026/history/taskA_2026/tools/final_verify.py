#!/usr/bin/env python3
"""对 31 张泡沫砖图绘制最终框，生成复核montage。"""
import cv2, numpy as np, os, json
RAW="/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick"
G={
 "015555_855884":[(5,205,61,363,364)],"015556_244603":[(5,205,61,363,364)],
 "015556_632185":[(5,205,61,363,364)],"015557_006862":[(5,205,61,363,364)],
 "015557_378440":[(5,205,61,363,364)],"015557_758709":[(5,205,61,363,364)],
 "015558_141195":[(5,205,61,363,364)],"015558_518728":[(5,205,61,363,364)],
 "015558_891405":[(5,205,61,363,364)],"015559_273953":[(5,205,61,363,364)],
 "015651_427267":[(5,112,200,461,359)],"015651_806864":[(5,112,200,461,359)],
 "015652_181086":[(5,112,200,461,359)],
 "015946_619787":[(5,12,118,331,275)],"015946_992055":[(5,12,118,331,275)],
 "020108_081864":[(5,202,105,569,293)],"020108_464841":[(5,202,105,569,293)],
 "020157_646343":[(5,0,256,368,424)],"020158_025628":[(5,0,256,368,424)],
 "020243_834170":[(5,303,247,640,425)],"020244_213967":[(5,303,247,640,425)],
 "021651_419155":[(5,258,140,456,431)],"021651_797070":[(5,258,140,456,431)],
 "021652_180906":[(5,258,140,456,431)],
 "021735_717901":[(5,253,100,375,377)],"021736_092395":[(5,253,100,375,377)],
 "021736_469985":[(5,253,100,375,377)],
 "022140_850009":[(5,65,237,363,427),(9,163,48,338,205)],
 "022141_233823":[(5,65,237,363,427),(9,163,48,338,205)],
 "022313_628794":[(5,162,122,372,372),(9,244,40,415,190)],
 "022314_004847":[(5,162,122,372,372),(9,244,40,415,190)],
}
files=sorted(os.listdir(RAW))
files=[f for f in files if "015220_297811" not in f]
assert len(files)==31, len(files)
tiles=[]
for f in files:
    tag=f[-17:-4]
    assert tag in G, tag
    img=cv2.imread(os.path.join(RAW,f))
    for cls,x1,y1,x2,y2 in G[tag]:
        col=(0,255,0) if cls==5 else (255,140,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),col,1)
        cv2.putText(img,f"{'brick' if cls==5 else 'can'}",(x1+2,max(11,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)
    for x in range(0,640,64): cv2.line(img,(x,0),(x,480),(0,90,255),1)
    for y in range(0,480,64): cv2.line(img,(0,y),(640,y),(0,90,255),1)
    cv2.rectangle(img,(0,0),(70,12),(0,0,0),-1)
    cv2.putText(img,tag[:8],(2,10),cv2.FONT_HERSHEY_SIMPLEX,0.32,(255,255,0),1)
    tiles.append(cv2.resize(img,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC))
os.makedirs("/tmp/final",exist_ok=True)
for i in range(0,len(tiles),8):
    chunk=tiles[i:i+8]
    while len(chunk)<8: chunk.append(np.zeros_like(tiles[0]))
    rows=[cv2.hconcat(chunk[j:j+2]) for j in range(0,8,2)]
    m=cv2.vconcat(rows)
    cv2.imwrite(f"/tmp/final/verify_{i//8+1}.jpg",m,[cv2.IMWRITE_JPEG_QUALITY,92])
    print(f"verify_{i//8+1}.jpg",[f[-17:-4] for f in files[i:i+8]])
json.dump({f:[[int(v) for v in item] for item in G[f[-17:-4]]] for f in files},
          open("/home/sunshine/codex/easton/taskA/prelabels_v21/manual_boxes_v2.json","w"), indent=1)
print("saved manual_boxes_v2.json")
