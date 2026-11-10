import cv2, numpy as np
from pathlib import Path
from ultralytics import SAM
base=Path('/media/sunshine/新加卷/taskA/real_batch_v5_20260926/labeled_group_color_v1/3-Scissors')
out=Path('/media/sunshine/新加卷/taskA/reports/scissors_relabel_20261016/probe_asym'); out.mkdir(parents=True,exist_ok=True)
model=SAM('/home/sunshine/codex/easton/taskA/mobile_sam.pt')

def comps(m,minarea=20):
    n,lab,stats,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8)
    o=[]
    for i in range(1,n):
        x,y,w,h,a=stats[i]
        if a>=minarea: o.append((int(a),(int(x),int(y),int(x+w),int(y+h))))
    o.sort(reverse=True); return o

for name in ['3-Scissors_20260926_133926_694196.jpg','3-Scissors_20260926_133928_827796.jpg','3-Scissors_20260926_133927_913399.jpg']:
    img=cv2.imread(str(base/'images'/name)); H,W=img.shape[:2]
    a=(base/'labels'/(Path(name).stem+'.txt')).read_text().split(); cx,cy,w,h=map(float,a[1:5])
    x1,y1,x2,y2=(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H
    ow,oh=x2-x1,y2-y1
    variants=[]
    for t in [0.10,0.15,0.20,0.25,0.30,0.40]:
        variants.append((f'top{int(t*100)}',[x1,y1-t*oh,x2,y2]))
    variants.append(('top20_right5',[x1,y1-0.20*oh,x2+0.05*ow,y2]))
    variants.append(('top20_bottom5',[x1,y1-0.20*oh,x2,y2+0.05*oh]))
    for tag,p in variants:
        p=[max(0.0,p[0]),max(0.0,p[1]),min(W-1.0,p[2]),min(H-1.0,p[3])]
        res=model(str(base/'images'/name),bboxes=[p],verbose=False)[0]
        if res.masks is None or len(res.masks.data)==0: continue
        m=(res.masks.data[0].cpu().numpy()>0.5).astype(np.uint8)
        cs=comps(m,50)
        # union components with area>80 within prompt-ish; get largest component and union significant
        mb=None
        if m.sum():
            ys,xs=np.where(m>0); mb=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
        # all significant comps
        n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
        keep=np.zeros_like(m)
        for i in range(1,n):
            if stats[i,4]>=80: keep|=(lab==i).astype(np.uint8)
        ub=None
        if keep.sum():
            ys,xs=np.where(keep>0); ub=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
        print(name,tag,'prompt',[round(v) for v in p],'maskarea',int(m.sum()),'bbox',mb,'union',ub,'components',cs[:4])
        vis=img.copy(); ov=vis.copy(); ov[keep>0]=(0,255,0); vis=cv2.addWeighted(vis,0.55,ov,0.45,0)
        cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        cv2.rectangle(vis,(int(p[0]),int(p[1])),(int(p[2]),int(p[3])),(255,255,0),1)
        if ub: cv2.rectangle(vis,(ub[0],ub[1]),(ub[2],ub[3]),(255,0,0),2)
        cv2.putText(vis,tag,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        cv2.imwrite(str(out/f'{Path(name).stem}_{tag}.jpg'),vis)
