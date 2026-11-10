import cv2, numpy as np
from pathlib import Path
from ultralytics import SAM
base=Path('/media/sunshine/新加卷/taskA/real_batch_v5_20260926/labeled_group_color_v1/3-Scissors')
out=Path('/media/sunshine/新加卷/taskA/reports/scissors_relabel_20261016/probe_expand2'); out.mkdir(parents=True,exist_ok=True)
model=SAM('/home/sunshine/codex/easton/taskA/mobile_sam.pt')

def stat(m):
    n,lab,stats,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8)
    keep=np.zeros_like(m)
    for i in range(1,n):
        if stats[i,4]>=80: keep|=(lab==i).astype(np.uint8)
    if keep.sum()==0: return None
    ys,xs=np.where(keep>0)
    return keep,(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1),int(keep.sum())

for name in ['3-Scissors_20260926_133902_853748.jpg','3-Scissors_20260926_133928_827796.jpg','3-Scissors_20260926_134253_096711.jpg']:
    img=cv2.imread(str(base/'images'/name)); H,W=img.shape[:2]
    a=(base/'labels'/(Path(name).stem+'.txt')).read_text().split(); cx,cy,w,h=map(float,a[1:5])
    x1,y1,x2,y2=(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H
    ow,oh=x2-x1,y2-y1
    variants=[('exact',[x1,y1,x2,y2])]
    for r in [0.08,0.12,0.15,0.20]:
        variants.append((f'uni{int(r*100)}',[x1-r*ow,y1-r*oh,x2+r*ow,y2+r*oh]))
    # asymmetric recommended
    variants += [
      ('t10r5',[x1,y1-0.10*oh,x2+0.05*ow,y2]),
      ('t15r8',[x1,y1-0.15*oh,x2+0.08*ow,y2+0.03*oh]),
      ('t20r10l3b3',[x1-0.03*ow,y1-0.20*oh,x2+0.10*ow,y2+0.03*oh]),
    ]
    for tag,p in variants:
        p=[max(0.0,p[0]),max(0.0,p[1]),min(W-1.0,p[2]),min(H-1.0,p[3])]
        res=model(str(base/'images'/name),bboxes=[p],verbose=False)[0]
        if res.masks is None or len(res.masks.data)==0: 
            print(name,tag,'NOMASK'); continue
        m=(res.masks.data[0].cpu().numpy()>0.5).astype(np.uint8)
        st=stat(m)
        if st is None: print(name,tag,'EMPTY'); continue
        keep,bb,area=st
        print(name,tag,'prompt',[round(v) for v in p],'bbox',bb,'area',area)
        vis=img.copy(); ov=vis.copy(); ov[keep>0]=(0,255,0); vis=cv2.addWeighted(vis,0.55,ov,0.45,0)
        cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        cv2.rectangle(vis,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2)
        cv2.putText(vis,tag,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        cv2.imwrite(str(out/f'{Path(name).stem}_{tag}.jpg'),vis)
