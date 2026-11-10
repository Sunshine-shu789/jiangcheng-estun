#!/usr/bin/env python3
"""Probe MobileSAM prompts on a few scissors frames. Not a production script."""
import sys, cv2, numpy as np
from pathlib import Path
from ultralytics import SAM

base = Path('/media/sunshine/新加卷/taskA/real_batch_v5_20260926/labeled_group_color_v1/3-Scissors')
outdir = Path('/media/sunshine/新加卷/taskA/reports/scissors_relabel_20261016/probe')
outdir.mkdir(parents=True, exist_ok=True)
names = [
 '3-Scissors_20260926_133902_853748.jpg',
 '3-Scissors_20260926_133923_340283.jpg',
 '3-Scissors_20260926_134253_096711.jpg',
]
model = SAM('/home/sunshine/codex/easton/taskA/mobile_sam.pt')

def read_box(p, W, H):
    a=p.read_text().split(); cx,cy,w,h=map(float,a[1:5])
    return np.array([(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H], np.float32)

def biggest(m):
    m=(m>0.5).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n<=1: return m
    i=1+int(np.argmax(stats[1:,4]))
    return (lab==i).astype(np.uint8)

def save_mask(img, m, box, name, tag):
    vis=img.copy()
    if m is not None:
        ov=vis.copy(); ov[m>0]=(0,255,0); vis=cv2.addWeighted(vis,0.65,ov,0.35,0)
    if box is not None:
        b=np.round(box).astype(int); cv2.rectangle(vis,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
    cv2.putText(vis,tag,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
    cv2.imwrite(str(outdir/f'{name}_{tag}.jpg'),vis)

for name in names:
    img=cv2.imread(str(base/'images'/name)); H,W=img.shape[:2]
    old=read_box(base/'labels'/(Path(name).stem+'.txt'),W,H)
    x1,y1,x2,y2=old
    cx,cy=(x1+x2)/2,(y1+y2)/2
    # orange handle point: pixels close to orange hue
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    orange=((hsv[:,:,0]>5)&(hsv[:,:,0]<35)&(hsv[:,:,1]>90)&(hsv[:,:,2]>120))
    # only inside a moderate expansion of old box; pick largest orange component (handle rims likely)
    roi=np.zeros((H,W),np.uint8); 
    ex=0.25*(x2-x1); ey=0.25*(y2-y1)
    roi[max(0,int(y1-ey)):min(H,int(y2+ey)), max(0,int(x1-ex)):min(W,int(x2+ex))]=1
    orange=(orange&roi.astype(bool)).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(orange,8)
    pts=[]; labels=[]
    if n>1:
        # top 2 orange components by area; skip tiny
        order=np.argsort(stats[1:,4])[::-1]+1
        for i in order[:2]:
            if stats[i,4] < 30: continue
            ys,xs=np.where(lab==i)
            px,py=float(np.median(xs)),float(np.median(ys))
            pts.append([px,py]); labels.append(1)
    pts.insert(0,[cx,cy]); labels.insert(0,1)
    # negative points at corners of old box (desk)
    neg=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]
    pts_neg=pts+neg; labels_neg=labels+[0]*len(neg)
    variants=[]
    # bbox exact
    variants.append(('bbox_exact',dict(bboxes=[old.tolist()])))
    # bbox shrunk 20% around center
    sh=0.2
    bb=[cx-sh*(x2-x1)/2,cy-sh*(y2-y1)/2,cx+sh*(x2-x1)/2,cy+sh*(y2-y1)/2]
    variants.append(('bbox_shrink20',dict(bboxes=[bb])))
    # points
    variants.append(('point_center',dict(points=[pts[0]],labels=[1])))
    # points handle(s)
    if len(pts)>1:
        variants.append(('points_handle',dict(points=pts[1:],labels=[1]*len(pts[1:]))))
        variants.append(('bbox_handle',dict(bboxes=[old.tolist()],points=pts[1:],labels=[1]*len(pts[1:]))))
    variants.append(('points_multi_neg',dict(points=pts_neg,labels=labels_neg)))
    for tag,kw in variants:
        try:
            res=model(str(base/'images'/name),verbose=False,**kw)[0]
            if res.masks is None or len(res.masks.data)==0:
                save_mask(img,None,None,name,tag+'_NOMASK'); continue
            # pick highest confidence mask (only one usually)
            m=biggest(res.masks.data[0].cpu().numpy())
            ys,xs=np.where(m>0)
            if len(xs)==0:
                save_mask(img,m,None,name,tag+'_EMPTY'); continue
            mb=[xs.min(),ys.min(),xs.max()+1,ys.max()+1]
            save_mask(img,m,mb,name,tag)
            print(name,tag,'area',int(m.sum()),'box',mb)
        except Exception as e:
            print(name,tag,'ERR',repr(e))
