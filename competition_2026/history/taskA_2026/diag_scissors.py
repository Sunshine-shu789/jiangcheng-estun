#!/usr/bin/env python3
"""剪刀失败模式诊断：对比当前生产权重在剪刀留出帧上的预测框与真值框。

只读：不修改权重、数据集、服务。输出 overlays + summary md。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

CLASSES = ['1-Cheez-it','2-Starkist_Tuna','3-Scissors','4-Frenchs_Mustard','5-Tomato_Soup',
           '6-Foam_Brick','7-Clamp','8-Plastic_Banana','9-Mug','10-meat_can']
SCISSORS = 2


def read_gt(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) >= 5:
            out.append((int(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4])))
    return out


def to_xyxy(box, w, h):
    _, x, y, bw, bh = box
    return np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return (inter/ua if ua > 0 else 0.0), inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v18_20261014/weights/best.pt')
    ap.add_argument('--data', default='/media/sunshine/新加卷/taskA/datasets/real_ft_v16_mug_20261013')
    ap.add_argument('--split', default='val')
    ap.add_argument('--pattern', default='*scissors*')
    ap.add_argument('--imgsz', type=int, action='append', default=None)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--device', default='0')
    ap.add_argument('--out', default='/media/sunshine/新加卷/taskA/reports/scissors_diag_20261016')
    args = ap.parse_args()
    sizes = args.imgsz or [640, 960]

    data = Path(args.data)
    img_dir = data / 'images' / args.split
    lbl_dir = data / 'labels' / args.split
    images = sorted(img_dir.glob(f'{args.pattern}.jpg')) + sorted(img_dir.glob(f'{args.pattern}.png'))
    if not images:
        raise SystemExit(f'没有匹配到图片：{img_dir}/{args.pattern}')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f'权重 {Path(args.weights).name} | 图片 {len(images)} 张 | imgsz {sizes}')

    model = YOLO(str(args.weights))
    lines = []
    montage_items = []

    for imgsz in sizes:
        results = model.predict(source=[str(p) for p in images], imgsz=imgsz, conf=args.conf,
                               iou=args.iou, device=args.device, verbose=False)
        n_ok = n_all = 0
        rows = []
        for img_path, res in zip(images, results):
            pil = Image.open(img_path).convert('RGB')
            W, H = pil.size
            gts = [g for g in read_gt(lbl_dir / f'{img_path.stem}.txt') if g[0] == SCISSORS]
            if not gts:
                continue
            preds = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                preds = [(xyxy[i], int(cls[i]), float(conf[i])) for i in range(len(cls))]
            for g in gts:
                gbox = to_xyxy(g, W, H)
                n_all += 1
                best_same = (0.0, None); best_any = (0.0, None)
                for box, c, cf in preds:
                    I, inter = iou(gbox, box)
                    if I > best_any[0]:
                        best_any = (I, (box, c, cf, inter))
                    if c == SCISSORS and I > best_same[0]:
                        best_same = (I, (box, c, cf, inter))
                passed = best_same[0] >= 0.75
                n_ok += int(passed)
                box, c, cf, inter = best_any[1] if best_any[1] is not None else (None, -1, 0.0, 0.0)
                rec = dict(image=img_path.name, imgsz=imgsz, gt=[round(v,1) for v in gbox.tolist()],
                           best_iou=round(best_same[0],3), best_iou_any=round(best_any[0],3),
                           pred_class=CLASSES[c] if c >= 0 else '无', pred_conf=round(cf,3),
                           passed=passed)
                if box is not None:
                    pw, ph = box[2]-box[0], box[3]-box[1]
                    gw, gh = gbox[2]-gbox[0], gbox[3]-gbox[1]
                    rec.update(pred=[round(v,1) for v in box.tolist()], w_ratio=round(pw/gw,3), h_ratio=round(ph/gh,3),
                               cover_gt=round(inter/(gw*gh),3), in_gt=round(inter/(pw*ph),3))
                rows.append(rec)
                if not passed and imgsz == sizes[0]:
                    d = ImageDraw.Draw(pil)
                    d.rectangle(gbox.tolist(), outline=(0,255,0), width=3)
                    d.text((gbox[0]+4, max(0, gbox[1]-16)), f'GT scissors', fill=(0,255,0))
                    if box is not None:
                        col = (255,60,60) if c == SCISSORS else (255,160,0)
                        d.rectangle(box.tolist(), outline=col, width=3)
                        d.text((box[0]+4, max(0, box[1]+4)), f'pred {CLASSES[c]} {cf:.2f} IoU{best_any[0]:.2f}', fill=col)
                    else:
                        d.text((10, 10), 'NO PREDICTION', fill=(255,0,0))
            if not passed and imgsz == sizes[0]:
                montage_items.append((img_path.stem, pil))
        pct = 100.0*n_ok/n_all if n_all else 0.0
        print(f'imgsz {imgsz}: 通过 {n_ok}/{n_all} = {pct:.1f}%')
        lines.append((imgsz, n_ok, n_all, pct, rows))

    # 汇总
    report = ['# 剪刀失败模式诊断（3-Scissors）', '',
              f'- 权重：`{args.weights}`（与 9100 生产服务同一份 v18）',
              f'- 数据：`{data}/images/{args.split}` 中匹配 `{args.pattern}` 的 {len(images)} 张',
              f'- 阈值：conf {args.conf}，IoU 匹配 0.5，通过阈值 0.75', '']
    for imgsz, n_ok, n_all, pct, rows in lines:
        report += [f'## imgsz {imgsz}', '', f'**通过率：{n_ok}/{n_all} = {pct:.1f}%**', '',
                   '| 图片 | best IoU | 通过 | 预测类 | 置信 | 宽比 | 高比 | 覆盖GT | 框内属GT |',
                   '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in rows:
            report.append('| {image} | {best_iou} | {p} | {pred_class} | {pred_conf} | {w} | {h} | {cg} | {ig} |'.format(
                image=r['image'][-42:], best_iou=r['best_iou'], p='✅' if r['passed'] else '❌',
                pred_class=r['pred_class'], pred_conf=r['pred_conf'],
                w=r.get('w_ratio','-'), h=r.get('h_ratio','-'), cg=r.get('cover_gt','-'), ig=r.get('in_gt','-')))
        report.append('')
    rpt = out / 'summary.md'
    rpt.write_text('\n'.join(report), encoding='utf-8')
    print('报告:', rpt)

    if montage_items:
        H = 300
        tiles = []
        for name, im in montage_items[:12]:
            tiles.append(im.resize((int(im.width*H/im.height), H)))
        cols = min(3, len(tiles))
        rows_n = (len(tiles)+cols-1)//cols
        W0 = max(t.width for t in tiles)
        canvas = Image.new('RGB', (cols*W0, rows_n*(H+18)), (255,255,255))
        d = ImageDraw.Draw(canvas)
        for i, t in enumerate(tiles):
            x = (i % cols)*W0; y = (i//cols)*(H+18)
            canvas.paste(t, (x, y+18))
            d.text((x+4, y+3), montage_items[i][0][-52:], fill=(0,0,0))
        mont = out / 'montage_failures.jpg'
        canvas.save(mont, quality=85)
        print('拼图:', mont)
    print('overlays 数:', len(montage_items))


if __name__ == '__main__':
    main()
