import cv2, numpy as np
from pathlib import Path
from ultralytics import SAM
base=Path('/media/sunshine/新加卷/taskA/real_batch_v5_20260926/labeled_group_color_v1/3-Scissors')
out=Path('/media/sunshine/新加卷/taskA/reports/scissors_relabel_20261016/probe'); out.mkdir(parents=True,exist_ok=True)
model=SAM('/home/sunshine/codex/easton/taskA/mobile_sam.pt')
for name in ['3-Scissors_20260926_133926_694196.jpg','3-Scissors_20260926_133928_827796.jpg','3-Scissors_20260926_133927_913399.jpg','3-Scissors_20260926_134253_707784.jpg']:
    img=cv2.imread(str(base/'images'/name)); H,W=img.shape[:2]
    a=(base/'labels'/(Path(name).stem+'.txt')).read_text().split(); cx,cy,w,h=map(float,a[1:5])
    box=[(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H]
    res=model(str(base/'images'/name),bboxes=[box],verbose=False)[0]
    print('\n',name,'box',[round(v) for v in box],'nmasks',len(res.masks.data) if res.masks is not None else 0)
    if res.masks is None: continue
    raw=(res.masks.data[0].cpu().numpy()>0.5).astype(np.uint8)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(raw,8)
    comps=[]
    for i in range(1,n):
        x,y,ww,hh,area=stats[i]
        if area<20: continue
        comps.append((i,area,(int(x),int(y),int(x+ww),int(y+hh))))
    comps.sort(key=lambda t:-t[1])
    print(' components:',comps)
    for order,(i,area,bb) in enumerate(comps[:8]):
        m=(lab==i).astype(np.uint8)
        vis=img.copy(); ov=vis.copy(); ov[m>0]=(0,255,0); vis=cv2.addWeighted(vis,0.55,ov,0.45,0)
        cv2.rectangle(vis,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
        cv2.rectangle(vis,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2)
        cv2.putText(vis,f'comp{order} area{area} {bb}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,0,255),2)
        cv2.imwrite(str(out/f'{Path(name).stem}_comp{order}.jpg'),vis)
    # all significant union
    sig=[i for i,area,bb in comps if area>50]
    if sig:
        m=np.isin(lab,sig).astype(np.uint8)
        ys,xs=np.where(m>0); mb=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
        print(' union significant bbox',mb,'area',int(m.sum()))
        vis=img.copy(); ov=vis.copy(); ov[m>0]=(0,255,0); vis=cv2.addWeighted(vis,0.55,ov,0.45,0)
        cv2.rectangle(vis,(mb[0],mb[1]),(mb[2],mb[3]),(255,0,0),2)
        cv2.rectangle(vis,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
        cv2.putText(vis,'union significant',(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,0,255),2)
        cv2.imwrite(str(out/f'{Path(name).stem}_union.jpg'),vis)
