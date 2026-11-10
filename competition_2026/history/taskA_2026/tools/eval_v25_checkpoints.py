#!/usr/bin/env python3
"""用 v25 物理组隔离 holdout 统一评估 10 类 YOLO 检查点。

评价口径按比赛任务 A 的实际需求：
  - 类别必须正确；
  - 同类别检测框与 GT 的 IoU >= 0.75 才记为达标；
  - 同时报告 IoU 0.50，便于区分“完全没识别”和“框定位不准”。
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CLASS_NAMES = {
    0: "1-Cheez-it",
    1: "2-Starkist_Tuna",
    2: "3-Scissors",
    3: "4-Frenchs_Mustard",
    4: "5-Tomato_Soup",
    5: "6-Foam_Brick",
    6: "7-Clamp",
    7: "8-Plastic_Banana",
    8: "9-Mug",
    9: "10-meat_can",
}


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def prepare(frame: np.ndarray, scale: float) -> tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    sw, sh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((h, w, 3), 114, np.uint8)
    xo, yo = (w - sw) // 2, (h - sh) // 2
    canvas[yo : yo + sh, xo : xo + sw] = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    return canvas, xo, yo


def nms(dets: list[tuple[int, float, tuple[float, float, float, float]]], threshold: float = 0.55):
    kept = []
    for det in sorted(dets, key=lambda item: item[1], reverse=True):
        if all(det[0] != other[0] or iou(det[2], other[2]) < threshold for other in kept):
            kept.append(det)
    return kept


def read_ground_truth(image: Path, root: Path):
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError(f"cannot read image: {image}")
    h, w = frame.shape[:2]
    label = root / "labels" / image.parent.name / f"{image.stem}.txt"
    gts: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
        gts.append(
            (
                int(cls),
                ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h),
            )
        )
    return frame, gts


def predict_all(
    model: YOLO,
    frames: list[np.ndarray],
    scales: tuple[float, ...],
    conf: float,
    batch: int,
    imgsz: int,
):
    per_image: list[list[tuple[int, float, tuple[float, float, float, float]]]] = [[] for _ in frames]
    for scale in scales:
        prepared = [prepare(frame, scale) for frame in frames]
        for start in range(0, len(prepared), batch):
            chunk = prepared[start : start + batch]
            sources = [item[0] for item in chunk]
            results = model.predict(
                source=sources,
                imgsz=imgsz,
                conf=conf,
                iou=0.7,
                max_det=300,
                device=0,
                verbose=False,
            )
            for index, result in enumerate(results, start=start):
                _, xo, yo = chunk[index - start]
                boxes = result.boxes
                if boxes is None:
                    continue
                for box, cls, score in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    per_image[index].append(
                        (
                            int(cls),
                            float(score),
                            ((x1 - xo) / scale, (y1 - yo) / scale, (x2 - xo) / scale, (y2 - yo) / scale),
                        )
                    )
    return [nms(dets) for dets in per_image]


def evaluate(
    per_image,
    gts_per_image,
    thresholds: tuple[float, ...],
    match_ious: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        for match_iou in match_ious:
            tp: Counter[int] = Counter()
            fp: Counter[int] = Counter()
            fn: Counter[int] = Counter()
            gt_count: Counter[int] = Counter()
            best_iou_values: dict[int, list[float]] = defaultdict(list)
            confusion: dict[int, Counter[str]] = defaultdict(Counter)
            fp_classes: Counter[int] = Counter()

            for dets, gts in zip(per_image, gts_per_image):
                for cls, _ in gts:
                    gt_count[cls] += 1
                dets_conf = [d for d in dets if d[1] >= threshold]
                matched: set[int] = set()

                for cls, conf, box in sorted(dets_conf, key=lambda item: item[1], reverse=True):
                    best_index = -1
                    best_value = 0.0
                    for index, (gt_cls, gt_box) in enumerate(gts):
                        if index in matched or gt_cls != cls:
                            continue
                        value = iou(box, gt_box)
                        if value > best_value:
                            best_index, best_value = index, value
                    if best_index >= 0 and best_value >= match_iou:
                        matched.add(best_index)
                        tp[cls] += 1
                    else:
                        fp[cls] += 1
                        fp_classes[cls] += 1

                for index, (gt_cls, gt_box) in enumerate(gts):
                    best_value = 0.0
                    for cls, _, box in dets_conf:
                        if cls == gt_cls:
                            best_value = max(best_value, iou(box, gt_box))
                    if index in matched:
                        best_iou_values[gt_cls].append(max(best_value, match_iou))
                        continue
                    fn[gt_cls] += 1
                    choices: list[tuple[float, int, float]] = []
                    for cls, conf, box in dets_conf:
                        value = iou(box, gt_box)
                        if value > 0:
                            choices.append((value, cls, conf))
                    if choices:
                        value, cls, _ = max(choices)
                        confusion[gt_cls][CLASS_NAMES.get(cls, str(cls))] += 1
                        best_iou_values[gt_cls].append(value)
                    else:
                        confusion[gt_cls]["missing"] += 1
                        best_iou_values[gt_cls].append(0.0)

            for cls in sorted(CLASS_NAMES):
                values = best_iou_values[cls]
                precision = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] else 1.0
                recall = tp[cls] / gt_count[cls] if gt_count[cls] else 1.0
                rows.append(
                    {
                        "threshold": threshold,
                        "match_iou": match_iou,
                        "class_id": cls,
                        "class": CLASS_NAMES[cls],
                        "gt": gt_count[cls],
                        "tp": tp[cls],
                        "fp": fp[cls],
                        "fn": fn[cls],
                        "precision": precision,
                        "recall": recall,
                        "mean_best_iou": float(np.mean(values)) if values else 0.0,
                        "median_best_iou": float(np.median(values)) if values else 0.0,
                        "iou75_hits": sum(value >= 0.75 for value in values),
                        "confusion": json.dumps(confusion[cls], ensure_ascii=False, sort_keys=True),
                    }
                )
    return rows


def macro_summary(rows: list[dict[str, object]], threshold: float, match_iou: float) -> dict[str, object]:
    selected = [row for row in rows if row["threshold"] == threshold and row["match_iou"] == match_iou]
    total_gt = sum(int(row["gt"]) for row in selected)
    total_tp = sum(int(row["tp"]) for row in selected)
    total_fp = sum(int(row["fp"]) for row in selected)
    total_fn = sum(int(row["fn"]) for row in selected)
    macro_p = float(np.mean([float(row["precision"]) for row in selected])) if selected else 0.0
    macro_r = float(np.mean([float(row["recall"]) for row in selected])) if selected else 0.0
    return {
        "threshold": threshold,
        "match_iou": match_iou,
        "gt": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "recall": total_tp / total_gt if total_gt else 0.0,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "iou75_hit_rate": total_tp / total_gt if total_gt else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        nargs="+",
        default=[
            "/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v18_20261014/weights/best.pt",
            "/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v21_20261016/weights/best.pt",
            "/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v22_20261016/weights/best.pt",
            "/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v23_hardneg_20261016/weights/best.pt",
        ],
    )
    parser.add_argument("--data", default="/media/sunshine/新加卷/taskA/datasets/real_ft_v25_clean_20261016")
    parser.add_argument("--out", default="/media/sunshine/新加卷/taskA/reports/v25_model_eval_20261016.csv")
    parser.add_argument("--summary", default="/media/sunshine/新加卷/taskA/reports/v25_model_eval_20261016.md")
    parser.add_argument("--scales", default="1.0,0.85")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--thresholds", default="0.15,0.25,0.35,0.5")
    parser.add_argument("--match-ious", default="0.5,0.75")
    args = parser.parse_args()

    root = Path(args.data).expanduser().resolve()
    images = sorted((root / "images/val").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no val images: {root}")
    frames, gts = [], []
    for image in images:
        frame, gt = read_ground_truth(image, root)
        frames.append(frame)
        gts.append(gt)
    scales = tuple(float(value) for value in args.scales.split(","))
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    match_ious = tuple(float(value) for value in args.match_ious.split(","))

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for weight in args.weights:
        path = Path(weight)
        if not path.is_file():
            print(f"[skip] missing {path}", flush=True)
            continue
        print(f"[eval] {path}", flush=True)
        model = YOLO(str(path))
        predictions = predict_all(model, frames, scales, args.conf, args.batch, args.imgsz)
        rows = evaluate(predictions, gts, thresholds, match_ious)
        run_name = path.parent.parent.name
        for row in rows:
            row.update({"model": run_name, "weights": str(path)})
            all_rows.append(row)
        summaries.append(
            {
                "model": run_name,
                "weights": str(path),
                **macro_summary(rows, 0.25, 0.75),
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    fields = [
        "model", "weights", "threshold", "match_iou", "class_id", "class", "gt", "tp", "fp",
        "fn", "precision", "recall", "mean_best_iou", "median_best_iou", "iou75_hits", "confusion",
    ]
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = Path(args.summary).expanduser().resolve()
    lines = ["# v25 物理组隔离 holdout：检查点统一评估", ""]
    lines.append(f"- 数据集：`{root}`")
    total_gt_instances = sum(len(items) for items in gts)
    lines.append(f"- val：{len(images)} 张，{total_gt_instances} 个 GT 实例")
    lines.append(f"- 推理：imgsz={args.imgsz}，scales={','.join(args.scales.split(','))}，conf 入口={args.conf}")
    lines.append("- 比赛口径主表：conf>=0.25，IoU>=0.75")
    lines.append("")
    lines.append("| 模型 | GT | IoU75 TP | IoU75 Recall | IoU75 Precision | FP | IoU50 Recall | Macro IoU75 Recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        model = str(summary["model"])
        rows = [row for row in all_rows if row["model"] == model]
        iou75 = macro_summary(rows, 0.25, 0.75)
        iou50 = macro_summary(rows, 0.25, 0.5)
        lines.append(
            f"| {model} | {iou75['gt']} | {iou75['tp']} | {float(iou75['recall']):.3f} | "
            f"{float(iou75['precision']):.3f} | {iou75['fp']} | {float(iou50['recall']):.3f} | "
            f"{float(iou75['macro_recall']):.3f} |"
        )
    lines.append("")
    lines.append("## 每类 IoU75（conf>=0.25）")
    lines.append("")
    for summary in summaries:
        model = str(summary["model"])
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| 类别 | GT | TP | FP | FN | Precision | Recall | Mean IoU | Median IoU | 混淆 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        rows = [row for row in all_rows if row["model"] == model and row["threshold"] == 0.25 and row["match_iou"] == 0.75]
        for row in sorted(rows, key=lambda item: int(item["class_id"])):
            lines.append(
                f"| {row['class']} | {row['gt']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                f"{float(row['precision']):.3f} | {float(row['recall']):.3f} | "
                f"{float(row['mean_best_iou']):.3f} | {float(row['median_best_iou']):.3f} | "
                f"`{row['confusion']}` |"
            )
        lines.append("")
    lines.append("## 注意事项")
    lines.append("")
    lines.append("- 这是物理组隔离结果，不能和旧 v22 val 的 Tuna/Mug 100% 直接比较；旧 val 存在精确重复和同组跨 split。")
    lines.append("- `missing` 表示 GT 没有同类检测；其他类别名表示框存在但类别混淆；同类低 IoU 归入 `fn`。")
    lines.append("- 只有 IoU75 达标率，才是赛道 A 的帧级/实例级风险指标；IoU50 仅用于定位诊断。")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out}")
    print(f"[saved] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
