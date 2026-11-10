#!/usr/bin/env python3
"""Offline, read-only acceptance audit for Track-1 front-ten classes.

The audit compares annotated labels against a YOLO checkpoint, records every
miss / wrong-class / localization error, and writes overlays and a report under
the data volume.  It never opens a camera and never imports ROS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(os.environ.get("TRACK1_PROJECT_ROOT", "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")).resolve()
DATA_ROOT = Path("/media/sunshine/新加卷").resolve()
FRONT10 = tuple(range(10))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {
    "gt": (60, 220, 60),
    "tp": (40, 220, 255),
    "fp": (40, 40, 230),
    "wrong": (230, 80, 40),
    "text": (255, 255, 255),
}


@dataclass
class Box:
    xyxy: tuple[float, float, float, float]
    class_id: int
    score: float | None = None


@dataclass
class MatchError:
    image: str
    gt_index: int
    gt_class_id: int
    gt_name: str
    gt_xyxy: list[float]
    reason: str
    prediction_class_id: int | None = None
    prediction_name: str | None = None
    prediction_score: float | None = None
    prediction_xyxy: list[float] | None = None
    iou: float | None = None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(value[i]) for i in range(len(value))]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("dataset names 不是列表或字典")


def image_paths(dataset_root: Path, split: str) -> list[Path]:
    root = dataset_root / "images" / split
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path(image_path: Path, dataset_root: Path, split: str) -> Path:
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def read_labels(path: Path, width: int, height: int, names: list[str]) -> list[Box]:
    boxes: list[Box] = []
    if not path.is_file():
        return boxes
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no} 不是 5 列")
        class_id = int(parts[0])
        if not 0 <= class_id < len(names):
            raise ValueError(f"{path}:{line_no} 类别越界: {class_id}")
        x, y, w, h = (float(value) for value in parts[1:])
        x1 = max(0.0, min(width, (x - w / 2.0) * width))
        y1 = max(0.0, min(height, (y - h / 2.0) * height))
        x2 = max(0.0, min(width, (x + w / 2.0) * width))
        y2 = max(0.0, min(height, (y + h / 2.0) * height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{path}:{line_no} 框尺寸无效")
        boxes.append(Box((x1, y1, x2, y2), class_id))
    return boxes


def iou(a: Box | tuple[float, float, float, float], b: Box | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy if isinstance(a, Box) else a
    bx1, by1, bx2, by2 = b.xyxy if isinstance(b, Box) else b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def draw_box(image: np.ndarray, box: Box, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    text = label
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    ty = max(th + 3, y1 - 4)
    cv2.rectangle(image, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(image, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text"], 1, cv2.LINE_AA)


def predict_boxes(model: Any, image_path: Path, image: np.ndarray, *, imgsz: int, conf: float, iou_threshold: float, device: str, max_det: int) -> list[Box]:
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou_threshold,
        device=device,
        max_det=max_det,
        verbose=False,
    )[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    return [
        Box(tuple(float(v) for v in row.tolist()), int(cls), float(score))
        for row, score, cls in zip(xyxy, scores, classes)
    ]


def match_front10(
    image_ref: str,
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    names: list[str],
    match_iou: float,
) -> tuple[list[Box], list[Box], list[MatchError], dict[str, int]]:
    """Greedily match same-class boxes, then classify each unmatched GT error."""
    gt_indices = [i for i, box in enumerate(gt_boxes) if box.class_id in FRONT10]
    candidate_pairs: list[tuple[float, int, int]] = []
    for gi in gt_indices:
        gt = gt_boxes[gi]
        for pi, pred in enumerate(pred_boxes):
            if pred.class_id != gt.class_id:
                continue
            score = iou(gt, pred)
            if score >= match_iou:
                candidate_pairs.append((score, gi, pi))
    candidate_pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched_gt: list[Box] = []
    matches: list[tuple[Box, Box]] = []
    for overlap, gi, pi in candidate_pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched_gt.append(gt_boxes[gi])
        matches.append((gt_boxes[gi], pred_boxes[pi]))

    errors: list[MatchError] = []
    counts: Counter[str] = Counter()
    for gi in gt_indices:
        if gi in used_gt:
            continue
        gt = gt_boxes[gi]
        # Find the most overlapping prediction, regardless of class.
        candidates = sorted(
            ((iou(gt, pred), pi, pred) for pi, pred in enumerate(pred_boxes)),
            key=lambda item: item[0],
            reverse=True,
        )
        best_iou, best_pi, best_pred = candidates[0] if candidates else (0.0, -1, None)
        if best_pred is not None and best_pred.class_id != gt.class_id and best_iou >= match_iou:
            reason = "wrong_class"
            counts[reason] += 1
        elif best_pred is not None and best_iou >= match_iou:
            reason = "localization"
            counts[reason] += 1
        elif best_pred is not None and best_iou >= 0.20 and best_pred.class_id != gt.class_id:
            reason = "wrong_class_low_iou"
            counts["wrong_class"] += 1
        elif best_iou >= 0.20:
            reason = "localization"
            counts[reason] += 1
        else:
            reason = "missing"
            counts[reason] += 1
        errors.append(
            MatchError(
                image=image_ref,
                gt_index=gi,
                gt_class_id=gt.class_id,
                gt_name=names[gt.class_id],
                gt_xyxy=[round(v, 2) for v in gt.xyxy],
                reason=reason,
                prediction_class_id=best_pred.class_id if best_pred else None,
                prediction_name=names[best_pred.class_id] if best_pred else None,
                prediction_score=round(best_pred.score or 0.0, 5) if best_pred and best_pred.score is not None else None,
                prediction_xyxy=[round(v, 2) for v in best_pred.xyxy] if best_pred else None,
                iou=round(best_iou, 4),
            )
        )

    # Unmatched predictions are false positives, including wrong-class boxes
    # already associated with an unmatched GT above.  Keep them visible.
    unmatched_preds = [pred for pi, pred in enumerate(pred_boxes) if pi not in used_pred]
    return matched_gt, unmatched_preds, errors, dict(counts)


def make_contact_sheet(paths: list[Path], destination: Path, title: str, cell_w: int = 360, cell_h: int = 300) -> None:
    if not paths:
        return
    cols = min(3, len(paths))
    rows = math.ceil(len(paths) / cols)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = min((cell_w - 8) / image.shape[1], (cell_h - 32) / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        row, col = divmod(idx, cols)
        x = col * cell_w + (cell_w - resized.shape[1]) // 2
        y = row * cell_h + 28
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(sheet, path.stem[:42], (col * cell_w + 6, row * cell_h + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(sheet, title[:100], (8, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), sheet)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "track1_front10_offline_audit_v2")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05, help="最低预测置信度，报告和叠加图均使用此值")
    parser.add_argument("--iou", type=float, default=0.60, help="YOLO NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.50, help="GT 与预测匹配 IoU")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    weights = args.weights.expanduser().resolve()
    dataset_yaml = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"权重不存在: {weights}")
    if not dataset_yaml.is_file():
        raise SystemExit(f"数据集 YAML 不存在: {dataset_yaml}")
    if not str(output).startswith(str(DATA_ROOT) + os.sep):
        raise SystemExit(f"审计输出必须位于数据盘 {DATA_ROOT}: {output}")
    output.mkdir(parents=True, exist_ok=True)

    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("dataset.yaml 不是对象")
    names = as_names(payload.get("names"))
    if len(names) < 40:
        raise SystemExit(f"类别数不足 40: {len(names)}")
    dataset_root = dataset_yaml.parent
    split = args.split
    images = image_paths(dataset_root, split)
    if not images:
        raise SystemExit(f"{split} 图片目录为空: {dataset_root / 'images' / split}")

    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    model_names = as_names(model.names) if hasattr(model, "names") else names
    if len(model_names) != len(names):
        raise SystemExit(f"模型类别数 {len(model_names)} 与数据集类别数 {len(names)} 不一致")

    records: list[dict[str, Any]] = []
    all_errors: list[MatchError] = []
    totals = Counter()
    per_class = {class_id: Counter() for class_id in FRONT10}
    overlays_by_class: dict[int, list[Path]] = defaultdict(list)
    overlay_dir = output / "overlays" / split
    if not args.no_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"无法读取图片: {image_path}")
        height, width = image.shape[:2]
        gt_all = read_labels(label_path(image_path, dataset_root, split), width, height, names)
        gt_front = [box for box in gt_all if box.class_id in FRONT10]
        if not gt_front:
            continue
        pred_boxes = predict_boxes(
            model,
            image_path,
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            max_det=args.max_det,
        )
        matched, unmatched, errors, error_counts = match_front10(
            str(image_path), gt_all, pred_boxes, names, args.match_iou
        )
        # Per-class counts: every GT and every prediction are counted, but only
        # front-ten GT boxes contribute to the acceptance denominator.
        for gt in gt_front:
            per_class[gt.class_id]["gt"] += 1
        for pred in pred_boxes:
            if pred.class_id in FRONT10:
                per_class[pred.class_id]["pred"] += 1
        for box in matched:
            per_class[box.class_id]["tp"] += 1
            totals["tp"] += 1
        for error in errors:
            all_errors.append(error)
            per_class[error.gt_class_id][error.reason] += 1
            totals[error.reason] += 1
        for pred in unmatched:
            if pred.class_id in FRONT10:
                per_class[pred.class_id]["fp"] += 1
            totals["fp"] += 1
        totals["gt"] += len(gt_front)
        records.append(
            {
                "image": str(image_path),
                "width": width,
                "height": height,
                "front10_gt": len(gt_front),
                "all_gt": len(gt_all),
                "predictions": len(pred_boxes),
                "matched": len(matched),
                "unmatched_predictions": len(unmatched),
                "errors": [asdict(item) for item in errors],
            }
        )
        if not args.no_overlays:
            overlay = image.copy()
            for gt in gt_front:
                draw_box(overlay, gt, COLORS["gt"], f"GT {names[gt.class_id]}", 2)
            for pred in pred_boxes:
                color = COLORS["wrong"] if pred.class_id not in FRONT10 else COLORS["tp"]
                draw_box(overlay, pred, color, f"P {names[pred.class_id]} {pred.score:.2f}", 2)
            error_text = f"errors={len(errors)} unmatched_pred={len(unmatched)}"
            cv2.putText(overlay, error_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, error_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            overlay_path = overlay_dir / image_path.name
            cv2.imwrite(str(overlay_path), overlay)
            for gt in gt_front:
                overlays_by_class[gt.class_id].append(overlay_path)
        if index % 10 == 0 or index == len(images):
            print(f"[{index}/{len(images)}] {image_path.name} gt_front10={len(gt_front)} errors={len(errors)} fp={len(unmatched)}", flush=True)

    class_rows: list[dict[str, Any]] = []
    for class_id in FRONT10:
        row = per_class[class_id]
        gt_count = row["gt"]
        tp = row["tp"]
        fp = row["fp"]
        recall = tp / gt_count if gt_count else None
        precision = tp / (tp + fp) if (tp + fp) else None
        class_rows.append(
            {
                "class_id": class_id,
                "official_id": class_id + 1,
                "name": names[class_id],
                "gt": gt_count,
                "tp": tp,
                "fp": fp,
                "recall": None if recall is None else round(recall, 5),
                "precision": None if precision is None else round(precision, 5),
                "missing": row["missing"],
                "wrong_class": row["wrong_class"] + row["wrong_class_low_iou"],
                "localization": row["localization"],
            }
        )

    accepted = (
        not all_errors
        and totals["fp"] == 0
        and all((row["gt"] > 0 and row["tp"] == row["gt"]) for row in class_rows)
    )
    summary = {
        "created_at": now_iso(),
        "mode": "offline_front10_acceptance",
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "dataset_yaml": str(dataset_yaml),
        "split": split,
        "image_count_total": len(images),
        "image_count_with_front10_gt": len(records),
        "conf_threshold": args.conf,
        "nms_iou": args.iou,
        "match_iou": args.match_iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "class_count": len(names),
        "front10_gt": totals["gt"],
        "front10_tp": totals["tp"],
        "front10_fn": totals["missing"] + totals["wrong_class"] + totals["wrong_class_low_iou"] + totals["localization"],
        "missing": totals["missing"],
        "wrong_class": totals["wrong_class"] + totals["wrong_class_low_iou"],
        "localization": totals["localization"],
        "false_positive_predictions": totals["fp"],
        "error_count": len(all_errors),
        "front10_recall": round(totals["tp"] / totals["gt"], 5) if totals["gt"] else None,
        "accepted": bool(accepted),
        "acceptance_rule": "无漏检、无错类、无定位错误、无未匹配预测，且前10类验证集每类至少一个 GT；该规则偏严格，用于现场前验收。",
        "classes": class_rows,
        "errors": [asdict(item) for item in all_errors],
        "images": records,
    }
    write_json(output / f"front10_offline_audit_{split}.json", summary)
    write_csv(output / f"front10_offline_audit_{split}_classes.csv", class_rows, list(class_rows[0]))
    write_csv(output / f"front10_offline_audit_{split}_errors.csv", [asdict(item) for item in all_errors], list(MatchError.__dataclass_fields__))
    if not args.no_overlays:
        for class_id, paths in overlays_by_class.items():
            make_contact_sheet(paths, output / f"contact_{split}_{class_id + 1:02d}_{names[class_id].replace('/', '_')}.jpg", f"{names[class_id]} | GT green / predictions yellow-red")
        make_contact_sheet([Path(record["image"]) for record in records], output / f"contact_{split}_all_front10_originals.jpg", f"{split} front10 images")

    lines = [
        "# 赛道一前10类离线验收报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 权重：`{weights}`",
        f"- 权重 SHA256：`{summary['weights_sha256']}`",
        f"- 数据集：`{dataset_yaml}`",
        f"- 划分：`{split}`",
        f"- 参数：conf={args.conf}, NMS IoU={args.iou}, match IoU={args.match_iou}, imgsz={args.imgsz}, device={args.device}",
        f"- 前10类 GT 框：{summary['front10_gt']}；匹配 TP：{summary['front10_tp']}；FN：{summary['front10_fn']}",
        f"- 漏检：{summary['missing']}；错类：{summary['wrong_class']}；定位错误：{summary['localization']}；未匹配预测：{summary['false_positive_predictions']}",
        f"- **严格验收结果：{'通过' if accepted else '未通过'}**",
        "",
        "## 逐类结果",
        "",
        "| 类别 | GT | TP | FP | Recall | Precision | 漏检 | 错类 | 定位 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in class_rows:
        lines.append(f"| {row['official_id']} {row['name']} | {row['gt']} | {row['tp']} | {row['fp']} | {row['recall'] if row['recall'] is not None else '-'} | {row['precision'] if row['precision'] is not None else '-'} | {row['missing']} | {row['wrong_class']} | {row['localization']} |")
    lines += ["", "## 错误明细", ""]
    if not all_errors:
        lines.append("没有记录到前10类 GT 的错误。")
    else:
        lines += ["| 图片 | GT | 原因 | 预测 | 置信度 | IoU |", "|---|---|---|---|---:|---:|"]
        for error in all_errors:
            lines.append(f"| {Path(error.image).name} | {error.gt_class_id + 1}-{error.gt_name} | {error.reason} | {error.prediction_class_id + 1 if error.prediction_class_id is not None else '-'}-{error.prediction_name or '-'} | {error.prediction_score if error.prediction_score is not None else '-'} | {error.iou if error.iou is not None else '-'} |")
    lines += [
        "",
        "## 输出文件",
        "",
        f"- JSON：`{output / f'front10_offline_audit_{split}.json'}`",
        f"- 逐类 CSV：`{output / f'front10_offline_audit_{split}_classes.csv'}`",
        f"- 错误 CSV：`{output / f'front10_offline_audit_{split}_errors.csv'}`",
        "- 叠加图目录：`overlays/`。绿色为 GT，黄色为预测，红色为预测类别不在前10类。",
        "",
        "> 说明：当前验证集每个类别样本很少，尤其剪刀等类别只有 1 个验证框；本报告是发现问题的严格离线筛查，不把总体 mAP 当作现场可用证明。",
    ]
    (output / f"front10_offline_audit_{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("front10_gt", "front10_tp", "front10_fn", "missing", "wrong_class", "localization", "false_positive_predictions", "accepted")}, ensure_ascii=False, indent=2))
    print(f"报告目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
