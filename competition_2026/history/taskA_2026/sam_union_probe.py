import cv2, numpy as np
from pathlib import Path
from ultralytics import SAM
base=Path('/media/sunshine/新加卷/taskA/real_batch_v5_20260926/labeled_group_color_v1/3-Scissors')
out=Path('/media/sunshine/新加卷/taskA/reports/scissors_relabel_20261016/probe_union'); out.mkdir(parents=True,exist_ok=True)
model=SAM('/home/sunshine/codex/easton/taskA/mobile_sam.pt')

def comps(m,minarea=20):
    n,lab,stats,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8)
    out=[]
    for i in range(1,n):
        x,y,w,h,a=stats[i]
        if a>=minarea: out.append((i,int(a),(int(x),int(y),int(x+w),int(y+h))))
    out.sort(key=lambda t:-t[1]); return out

for name in ['3-Scissors_20260926_133926_694196.jpg','3-Scissors_20260926_133928_827796.jpg','3-Scissors_20260926_133927_913399.jpg','3-Scissors_20260926_133902_853748.jpg']:
    img=cv2.imread(str(base/'images'/name)); H,W=img.shape[:2]
    a=(base/'labels'/(Path(name).stem+'.txt')).read_text().split(); cx,cy,w,h=map(float,a[1:5])
    box=[(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H]
    # exact bbox mask
    res=model(str(base/'images'/name),bboxes=[box],verbose=False)[0]
    m_bbox=(res.masks.data[0].cpu().numpy()>0.5).astype(np.uint8)
    cons=comps(m_bbox,20)
    # keep all components > 80 px, union
    keep=[i for i,ar,bb in cons if ar>80]
    m1=np.isin(np.argwhere(np.ones_like(m_bbox)),[])  # dummy
    m_union=np.zeros_like(m_bbox)
    n,lab,stats,_=cv2.connectedComponentsWithStats(m_bbox,8)
    for i,ar,bb in cons:
        if ar>80: m_union|=(lab==i).astype(np.uint8)
    # orange handle point(s)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    orange=((hsv[:,:,0]>5)&(hsv[:,:,0]<35)&(hsv[:,:,1]>80)&(hsv[:,:,2]>100)).astype(np.uint8)
    roi=np.zeros((H,W),np.uint8); ex=0.25*(box[2]-box[0]); ey=0.25*(box[3]-box[1])
    roi[max(0,int(box[1]-ey)):min(H,int(box[3]+ey)),max(0,int(box[0]-ex)):min(W,int(box[2]+ex))]=1
    orange &= roi
    oc=comps(orange,25)
    print('\n',name,'bbox comps',[(a,b) for _,a,b in cons[:5]],'orange comps',[(a,b) for _,a,b in oc[:5]])
    m_handle=None; pts=[]
    for i,ar,bb in oc[:3]:
        ys,xs=np.where((cv2.connectedComponentsWithStats(orange,8)[1])==i)
        if len(xs)==0: continue
        px,py=float(np.median(xs)),float(np.median(ys)); pts.append((px,py))
    if pts:
        # point prompt one by one, union masks
        m_handle=np.zeros_like(m_bbox)
        for px,py in pts:
            rr=model(str(base/'images'/name),points=[[px,py]],labels=[1],verbose=False)[0]
            if rr.masks is not None and len(rr.masks.data):
                mm=(rr.masks.data[0].cpu().numpy()>0.5).astype(np.uint8)
                cc=comps(mm,20)
                if cc:
                    # keep only components overlapping the orange point (near it) or any significant
                    m_handle |= mm
    m2=m_union.copy()
    if m_handle is not None: m2 |= m_handle
    # bboxes
    for tag,m in [('union_bbox',m_union),('union_handle',m2)]:
        if m.sum()==0: continue
        ys,xs=np.where(m>0); bb=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
        vis=img.copy(); ov=vis.copy(); ov[m>0]=(0,255,0); vis=cv2.addWeighted(vis,0.55,ov,0.45,0)
        cv2.rectangle(vis,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2)
        cv2.rectangle(vis,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
        cv2.putText(vis,tag,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        cv2.imwrite(str(out/f'{Path(name).stem}_{tag}.jpg'),vis)
        print(' ',tag,'area',int(m.sum()),'bbox',bb)
