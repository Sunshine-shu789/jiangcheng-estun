#!/usr/bin/env python3
"""用 v18+v20 对泡沫砖新采 31 张做低阈值预标注，输出 YOLO txt + 可视化。"""
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

SRC = Path('/media/sunshine/新加卷/taskA/real_raw/6-Foam_Brick')
OUT = Path('/home/sunshine/codex/easton/taskA/prelabels_v21')
V18 = '/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v18_20261014/weights/best.pt'
V20 = '/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v20_20261015/weights/best.pt'
NAMES = ['1-Cheez-it','2-Starkist_Tuna','3-Scissors','4-Frenchs_Mustard','5-Tomato_Soup',
         '6-Foam_Brick','7-Clamp','8-Plastic_Banana','9-Mug','10-meat_can']

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def main():
    (OUT/'labels').mkdir(parents=True, exist_ok=True)
    (OUT/'vis').mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in SRC.glob('6-Foam_Brick_20261016_*.jpg') if '015220' not in f.name)
    print('待标注:', len(files))
    results = {}
    for tag, wp in (('v18', V18), ('v20', V20)):
        m = YOLO(wp)
        rs = m.predict([str(f) for f in files], imgsz=1280, conf=0.03, iou=0.7,
                       device='cpu', verbose=False, max_det=30)
        results[tag] = rs
        print(tag, '推理完成', len(rs))
    summary = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        boxes = []
        for tag in ('v18', 'v20'):
            r = results[tag][i]
            for b in r.boxes:
                cls = int(b.cls.item()); conf = float(b.conf.item())
                xyxy = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append((cls, conf, xyxy))
        boxes.sort(key=lambda t: -t[1])
        keep = []
        for cls, conf, xyxy in boxes:
            if any(cls == k[0] and iou(xyxy, k[2]) > 0.55 for k in keep):
                continue
            keep.append((cls, conf, xyxy))
        lines, drawn = [], []
        for cls, conf, (x1, y1, x2, y2) in keep:
            cx, cy = (x1+x2)/2/w, (y1+y2)/2/h
            bw, bh = (x2-x1)/w, (y2-y1)/h
            lines.append(f'{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            drawn.append((cls, conf, x1, y1, x2, y2))
        (OUT/'labels'/(f.stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''))
        for cls, conf, x1, y1, x2, y2 in drawn:
            color = (0,255,0) if cls == 5 else (255,160,0)
            cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            cv2.putText(img, f'{NAMES[cls]} {conf:.2f}', (int(x1), max(12,int(y1)-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(str(OUT/'vis'/(f.stem+'.jpg')), img)
        desc = ', '.join(f'{NAMES[c]}:{cf:.2f}' for c,cf,*_ in drawn) or '无框'
        summary.append(f'{f.name}  {desc}')
    (OUT/'summary.txt').write_text('\n'.join(summary)+'\n')
    for s in summary: print(s)

if __name__ == '__main__':
    main()
