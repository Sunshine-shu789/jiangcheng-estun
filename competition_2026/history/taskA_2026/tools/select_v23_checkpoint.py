#!/usr/bin/env python3
"""扫描 v23 各 epoch checkpoint，按同一套专项指标选模型。

输出每个 checkpoint：
  phone_fp/74     手机等杂物误报为泡沫砖的图片数（越低越好）
  brick_val       10 张泡沫砖留出姿态命中数（要求 10）
  soup_val        14 张新汤罐留出姿态命中数（要求 14）
  soup_neg_fp/32  汤罐负样本留出误报数（要求 0）
  mAP50 / 关键类  v21 224张同集回归（用于发现旧类退化）
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2

sys.path.insert(0, "/home/sunshine/codex/easton/taskA/tools")
import eval_soup_v22 as soup

T = Path("/media/sunshine/新加卷/taskA")
BRICK_RAW = T / "real_raw/6-Foam_Brick"
BRICK_BOXES = Path("/home/sunshine/codex/easton/taskA/prelabels_v21/manual_boxes_v2.json")
EMPTY_HOLDOUT = Path("/home/sunshine/codex/easton/taskA/scratch/v21_empty_holdout_20261016.txt")
VAL_TAGS = {
    "015559_273953", "015652_181086", "015946_992055", "020108_464841",
    "020158_025628", "020244_213967", "021652_180906", "021736_469985",
    "022141_233823", "022314_004847",
}


def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def soup_metrics(model):
    boxes = json.loads(soup.BOXJSON.read_text(encoding="utf-8"))
    manifest = json.loads(soup.MANIFEST.read_text(encoding="utf-8"))
    hits = 0
    best = []
    for name in manifest["new_soup_val"]:
        frame = cv2.imread(str(soup.RAW_SOUP / name))
        dets = [d for d in soup.predict_boxes(model, frame, 0.03)
                if d[0] == 4 and d[1] >= 0.35]
        gts = [tuple(box[1:]) for box in boxes[name] if box[0] == 4]
        ok = any(soup.hit(d[2], gt) for d in dets for gt in gts)
        hits += int(ok)
        best.append(max((d[1] for d in dets), default=0.0))
    neg_fp = 0
    for name in manifest["negative_holdout"]:
        frame = cv2.imread(str(soup.RAW_EMPTY / name))
        if any(d[1] >= 0.25 for d in soup.predict_boxes(model, frame, 0.03)):
            neg_fp += 1
    return hits, neg_fp, (min(best) if best else 0.0)


def brick_metrics(model):
    gt = json.loads(BRICK_BOXES.read_text(encoding="utf-8"))
    brick_hits = 0
    for tag in sorted(VAL_TAGS):
        cands = list(BRICK_RAW.glob(f"*{tag}.jpg"))
        if not cands:
            raise RuntimeError(f"missing brick image {tag}")
        name = cands[0].name
        frame = cv2.imread(str(cands[0]))
        dets = [d for d in soup.predict_boxes(model, frame, 0.03)
                if d[0] == 5 and d[1] >= 0.25]
        gts = [tuple(box[1:]) for box in gt[name] if box[0] == 5]
        ok = any(
            soup.iou(d[2], g) >= 0.3
            or (g[0] < (d[2][0] + d[2][2]) / 2 < g[2]
                and g[1] < (d[2][1] + d[2][3]) / 2 < g[3])
            for d in dets for g in gts
        )
        brick_hits += int(ok)
    hold = {line.strip() for line in EMPTY_HOLDOUT.read_text(encoding="utf-8").splitlines() if line.strip()}
    phone_fp = 0
    max_conf = 0.0
    for path in sorted((T / "real_raw/empty_background").glob("*.jpg")):
        if path.name in hold:
            continue
        frame = cv2.imread(str(path))
        dets = [d for d in soup.predict_boxes(model, frame, 0.03)
                if d[0] == 5 and d[1] >= 0.25]
        if dets:
            phone_fp += 1
            max_conf = max(max_conf, max(d[1] for d in dets))
    return brick_hits, phone_fp, max_conf


def regression(model, name: str):
    result = model.val(
        data=str(T / "datasets/real_ft_v21_20261016/data.yaml"),
        imgsz=640,
        batch=8,
        device=0,
        verbose=False,
        plots=False,
        project="/tmp/v23_checkpoint_scan",
        name=name,
        exist_ok=True,
    )
    ap50 = [float(v) for v in result.box.ap50]
    return float(result.box.map50), float(result.box.map), ap50


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    run = Path(args.run_dir)
    ckpts = sorted(run.glob("weights/epoch*.pt"), key=lambda p: int(p.stem[5:]))
    if (run / "weights/last.pt").is_file():
        ckpts.append(run / "weights/last.pt")
    if not ckpts:
        raise SystemExit("no epoch checkpoints")
    lines = ["checkpoint,brick_val,phone_fp,phone_max,soup_val,soup_min,soup_neg_fp,map50,map50_95,cheezit,scissors,mustard,foam,banana,meat"]
    for ckpt in ckpts:
        model = load_model(str(ckpt))
        brick_hits, phone_fp, phone_max = brick_metrics(model)
        soup_hits, soup_neg, soup_min = soup_metrics(model)
        map50, map5095, ap50 = regression(model, ckpt.stem)
        row = [
            ckpt.name,
            brick_hits,
            phone_fp,
            f"{phone_max:.3f}",
            soup_hits,
            f"{soup_min:.3f}",
            soup_neg,
            f"{map50:.4f}",
            f"{map5095:.4f}",
            *[f"{ap50[i]:.4f}" for i in (0, 2, 3, 5, 7, 9)],
        ]
        lines.append(",".join(str(v) for v in row))
        print(lines[-1], flush=True)
        del model
    report = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
