#!/usr/bin/env python3
"""Offline v7 audit for front ten classes and official classes 17/26.

The tool reads a YOLO checkpoint and a static validation split only. It never
opens a camera, imports ROS, moves the robot, or controls the gripper.
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

PROJECT_ROOT = Path(
    os.environ.get(
        "TRACK1_PROJECT_ROOT",
        "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序",
    )
).resolve()
DATA_ROOT = Path("/media/sunshine/新加卷").resolve()
FRONT10 = tuple(range(10))
NEW2 = (16, 25)
TARGET_CLASSES = FRONT10 + NEW2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    xyxy: tuple[float, float, float, float]
    class_id: int
    score: float | None = None


@dataclass
class ErrorRecord:
    image: str
    gt_class_id: int
    gt_name: str
    gt_xyxy: list[float]
    reason: str
    prediction_class_id: int | None = None
    prediction_name: str | None = None
    prediction_score: float | None = None
    prediction_xyxy: list[float] | None = None
    iou: float | None = None


@dataclass
class OverlapWarning:
    image: str
    gt_class_id: int
    gt_name: str
    overlapping_class_id: int
    overlapping_name: str
    score: float
    iou: float


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(value[key]) for key in range(len(value))]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("dataset names 不是列表或字典")


def image_paths(dataset_root: Path, split: str) -> list[Path]:
    root = dataset_root / "images" / split
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


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


def box_iou(a: Box | tuple[float, float, float, float], b: Box | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy if isinstance(a, Box) else a
    bx1, by1, bx2, by2 = b.xyxy if isinstance(b, Box) else b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def predict_boxes(
    model: Any,
    image_path: Path,
    *,
    imgsz: int,
    conf: float,
    iou_threshold: float,
    device: str,
    max_det: int,
) -> list[Box]:
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
        Box(tuple(float(value) for value in row.tolist()), int(class_id), float(score))
        for row, score, class_id in zip(xyxy, scores, classes)
    ]


def draw_box(image: np.ndarray, box: Box, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    top = max(text_height + 3, y1 - 4)
    cv2.rectangle(image, (x1, top - text_height - 3), (x1 + text_width + 4, top + 2), color, -1)
    cv2.putText(image, label, (x1 + 2, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render_overlay(
    destination: Path,
    image: np.ndarray,
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    names: list[str],
) -> None:
    canvas = image.copy()
    for box in gt_boxes:
        draw_box(canvas, box, (60, 210, 60), f"GT {box.class_id + 1}-{names[box.class_id]}")
    for box in pred_boxes:
        draw_box(canvas, box, (30, 170, 255), f"P {box.class_id + 1}-{names[box.class_id]} {box.score or 0.0:.3f}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"无法写出叠加图: {destination}")


def contact_sheet(paths: list[Path], destination: Path, title: str, cell_w: int = 360, cell_h: int = 300) -> None:
    if not paths:
        return
    cols = min(3, len(paths))
    rows = math.ceil(len(paths) / cols)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = min((cell_w - 8) / image.shape[1], (cell_h - 34) / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        row, col = divmod(index, cols)
        x = col * cell_w + (cell_w - resized.shape[1]) // 2
        y = row * cell_h + 28
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(sheet, path.stem[:42], (col * cell_w + 6, row * cell_h + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(sheet, title[:110], (8, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"无法写出联系表: {destination}")


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


def load_post_train_metrics(path: Path | None, names: list[str]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_class = payload.get("per_class", {})
    metrics: dict[str, Any] = {
        "post_train_audit": str(path),
        "post_train_accepted": bool(payload.get("accepted")),
        "map50": payload.get("map50"),
        "map50_95": payload.get("map"),
        "front10_baseline_drops": payload.get("front10_baseline_drops", {}),
        "classes": {},
    }
    for class_id in TARGET_CLASSES:
        metric = per_class.get(str(class_id), {})
        metrics["classes"][str(class_id)] = {
            "official_id": class_id + 1,
            "name": names[class_id],
            "ap50": metric.get("ap50"),
            "precision": metric.get("precision"),
            "recall": metric.get("recall"),
        }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-train-audit", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--overlap-warning-iou", type=float, default=0.30)
    parser.add_argument("--device", default="cpu")
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
        raise SystemExit(f"输出必须位于数据盘 {DATA_ROOT}: {output}")

    raw_dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw_dataset, dict):
        raise SystemExit("dataset.yaml 不是对象")
    names = as_names(raw_dataset.get("names"))
    if len(names) != 40:
        raise SystemExit(f"数据集应为 40 类，实际 {len(names)} 类")

    dataset_root = dataset_yaml.parent
    split = args.split
    images = image_paths(dataset_root, split)
    if not images:
        raise SystemExit(f"{split} 图片目录为空")

    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    model_names = as_names(model.names)
    if model_names != names:
        raise SystemExit("模型与数据集类别表不一致")

    output.mkdir(parents=True, exist_ok=True)
    overlay_dir = output / "overlays" / split
    per_class: dict[int, Counter[str]] = {class_id: Counter() for class_id in TARGET_CLASSES}
    errors: list[ErrorRecord] = []
    warnings: list[OverlapWarning] = []
    records: list[dict[str, Any]] = []
    overlay_paths_by_class: dict[int, list[Path]] = defaultdict(list)
    all_prediction_counts: Counter[str] = Counter()

    for image_index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"无法读取图片: {image_path}")
        height, width = image.shape[:2]
        gt_all = read_labels(label_path(image_path, dataset_root, split), width, height, names)
        gt_targets = [box for box in gt_all if box.class_id in TARGET_CLASSES]
        preds = predict_boxes(
            model,
            image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            max_det=args.max_det,
        )
        for prediction in preds:
            all_prediction_counts[names[prediction.class_id]] += 1

        # Match same-class boxes greedily by IoU.
        candidates: list[tuple[float, int, int]] = []
        for gt_index, gt in enumerate(gt_targets):
            for pred_index, prediction in enumerate(preds):
                if prediction.class_id != gt.class_id:
                    continue
                overlap = box_iou(gt, prediction)
                if overlap >= args.match_iou:
                    candidates.append((overlap, gt_index, pred_index))
        candidates.sort(reverse=True)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for _, gt_index, pred_index in candidates:
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            gt = gt_targets[gt_index]
            prediction = preds[pred_index]
            per_class[gt.class_id]["gt"] += 1
            per_class[gt.class_id]["tp"] += 1
            per_class[gt.class_id]["matched_score_sum"] += prediction.score or 0.0
            per_class[gt.class_id]["matched_score_count"] += 1

        for gt_index, gt in enumerate(gt_targets):
            if gt_index in matched_gt:
                continue
            per_class[gt.class_id]["gt"] += 1
            best_overlap = 0.0
            best_prediction: Box | None = None
            for prediction in preds:
                overlap = box_iou(gt, prediction)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_prediction = prediction
            if best_prediction is not None and best_overlap >= args.match_iou and best_prediction.class_id != gt.class_id:
                reason = "wrong_class"
            elif best_prediction is not None and best_overlap >= args.match_iou:
                reason = "localization"
            elif best_prediction is not None and best_overlap >= 0.20 and best_prediction.class_id != gt.class_id:
                reason = "wrong_class_low_iou"
            elif best_overlap >= 0.20:
                reason = "localization"
            else:
                reason = "missing"
            per_class[gt.class_id][reason] += 1
            errors.append(
                ErrorRecord(
                    image=str(image_path),
                    gt_class_id=gt.class_id,
                    gt_name=names[gt.class_id],
                    gt_xyxy=[round(value, 2) for value in gt.xyxy],
                    reason=reason,
                    prediction_class_id=best_prediction.class_id if best_prediction else None,
                    prediction_name=names[best_prediction.class_id] if best_prediction else None,
                    prediction_score=round(best_prediction.score or 0.0, 5) if best_prediction else None,
                    prediction_xyxy=[round(value, 2) for value in best_prediction.xyxy] if best_prediction else None,
                    iou=round(best_overlap, 4),
                )
            )

        # Any selected-class prediction not matched to a same-class GT is an FP.
        for pred_index, prediction in enumerate(preds):
            if prediction.class_id not in TARGET_CLASSES or pred_index in matched_pred:
                continue
            per_class[prediction.class_id]["fp"] += 1
            errors.append(
                ErrorRecord(
                    image=str(image_path),
                    gt_class_id=prediction.class_id,
                    gt_name=names[prediction.class_id],
                    gt_xyxy=[],
                    reason="false_positive",
                    prediction_class_id=prediction.class_id,
                    prediction_name=names[prediction.class_id],
                    prediction_score=round(prediction.score or 0.0, 5),
                    prediction_xyxy=[round(value, 2) for value in prediction.xyxy],
                    iou=None,
                )
            )

        # Record same-object overlapping predictions from another class. This
        # does not itself count as a GT-level failure, matching prior readiness.
        for gt in gt_targets:
            for prediction in preds:
                if prediction.class_id == gt.class_id:
                    continue
                overlap = box_iou(gt, prediction)
                if overlap >= args.overlap_warning_iou:
                    warnings.append(
                        OverlapWarning(
                            image=str(image_path),
                            gt_class_id=gt.class_id,
                            gt_name=names[gt.class_id],
                            overlapping_class_id=prediction.class_id,
                            overlapping_name=names[prediction.class_id],
                            score=round(prediction.score or 0.0, 5),
                            iou=round(overlap, 4),
                        )
                    )

        if not args.no_overlays and (gt_targets or any(box.class_id in TARGET_CLASSES for box in preds)):
            overlay_path = overlay_dir / image_path.name
            render_overlay(overlay_path, image, gt_targets, preds, names)
            for class_id in {box.class_id for box in gt_targets}:
                overlay_paths_by_class[class_id].append(overlay_path)

        records.append(
            {
                "image": str(image_path),
                "target_gt_count": len(gt_targets),
                "target_gt_classes": [box.class_id for box in gt_targets],
                "target_prediction_count": sum(box.class_id in TARGET_CLASSES for box in preds),
            }
        )
        if image_index % 20 == 0 or image_index == len(images):
            print(f"进度 {image_index}/{len(images)}", flush=True)

    class_rows: list[dict[str, Any]] = []
    for class_id in TARGET_CLASSES:
        row = per_class[class_id]
        gt_count = row["gt"]
        tp = row["tp"]
        fp = row["fp"]
        fn = row["missing"] + row["wrong_class"] + row["wrong_class_low_iou"] + row["localization"]
        if fn > 0:
            status = "FAIL"
            note = "存在 GT 级漏检、错类或定位错误"
        elif fp > 0:
            status = "CAUTION"
            note = "GT 均可检出，但存在额外同类预测"
        elif gt_count > 0:
            status = "PASS"
            note = "无 GT 级错误，无额外同类预测"
        else:
            status = "NO_GT"
            note = "验证集中没有该类 GT"
        scores = row["matched_score_sum"] / row["matched_score_count"] if row["matched_score_count"] else None
        class_rows.append(
            {
                "class_id": class_id,
                "official_id": class_id + 1,
                "group": "front10" if class_id < 10 else "new2",
                "name": names[class_id],
                "gt": gt_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "missing": row["missing"],
                "wrong_class": row["wrong_class"] + row["wrong_class_low_iou"],
                "localization": row["localization"],
                "recall": round(tp / gt_count, 5) if gt_count else None,
                "precision": round(tp / (tp + fp), 5) if (tp + fp) else None,
                "matched_mean_score": round(scores, 5) if scores is not None else None,
                "status": status,
                "note": note,
            }
        )

    target_errors = [error for error in errors if error.reason != "false_positive"]
    target_fps = [error for error in errors if error.reason == "false_positive"]
    new2_rows = [row for row in class_rows if row["group"] == "new2"]
    front10_rows = [row for row in class_rows if row["group"] == "front10"]
    new2_core_ok = all(row["gt"] > 0 and row["fn"] == 0 and row["tp"] == row["gt"] for row in new2_rows)
    front10_core_ok = all(row["gt"] > 0 and row["fn"] == 0 and row["tp"] == row["gt"] for row in front10_rows)
    strict_ok = not errors and all(row["gt"] > 0 and row["tp"] == row["gt"] for row in class_rows)
    post_metrics = load_post_train_metrics(args.post_train_audit, names)
    front10_drops = post_metrics.get("front10_baseline_drops", {})
    front10_drop_ok = all(float(item.get("drop", 0.0)) <= 0.03 for item in front10_drops.values())
    new2_ap_ok = all(
        post_metrics.get("classes", {}).get(str(class_id), {}).get("ap50", -1.0) >= 0.30
        for class_id in NEW2
    )

    summary = {
        "schema_version": 1,
        "created_at": now_iso(),
        "mode": "v7_offline_targeted_acceptance",
        "read_only_static_inference": True,
        "opens_camera": False,
        "imports_ros": False,
        "robot_control": False,
        "gripper_control": False,
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "dataset_yaml": str(dataset_yaml),
        "split": split,
        "image_count": len(images),
        "target_classes": [class_id + 1 for class_id in TARGET_CLASSES],
        "target_class_names": [names[class_id] for class_id in TARGET_CLASSES],
        "parameters": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "nms_iou": args.iou,
            "match_iou": args.match_iou,
            "overlap_warning_iou": args.overlap_warning_iou,
            "device": args.device,
            "max_det": args.max_det,
        },
        "post_train_metrics": post_metrics,
        "decision": {
            "new2_core_ok": new2_core_ok,
            "front10_core_ok": front10_core_ok,
            "strict_zero_error_ok": strict_ok,
            "front10_vs_v6_drop_within_0_03": front10_drop_ok,
            "new2_ap50_at_least_0_30": new2_ap_ok,
            "deployment_candidate_ready_pending_onsite": bool(
                new2_core_ok and front10_core_ok and front10_drop_ok and new2_ap_ok
            ),
            "note": "通过以验证集 GT 召回/错类/定位为核心；额外同类预测和跨类重叠单列，不自动覆盖生产模型。",
        },
        "counts": {
            "gt": sum(int(row["gt"]) for row in class_rows),
            "tp": sum(int(row["tp"]) for row in class_rows),
            "fp": len(target_fps),
            "gt_level_errors": len(target_errors),
            "overlap_warnings": len(warnings),
        },
        "classes": class_rows,
        "errors": [asdict(error) for error in errors],
        "overlap_warnings": [asdict(warning) for warning in warnings],
        "images": records,
        "all_prediction_counts_by_class": dict(sorted(all_prediction_counts.items())),
    }
    write_json(output / "v7_targeted_offline_audit_val.json", summary)
    write_csv(output / "v7_targeted_offline_audit_val_classes.csv", class_rows, list(class_rows[0]))
    write_csv(output / "v7_targeted_offline_audit_val_errors.csv", [asdict(error) for error in errors], list(ErrorRecord.__dataclass_fields__))
    write_csv(output / "v7_targeted_offline_audit_val_overlap_warnings.csv", [asdict(warning) for warning in warnings], list(OverlapWarning.__dataclass_fields__))
    if not args.no_overlays:
        for class_id, paths in overlay_paths_by_class.items():
            deduplicated: list[Path] = []
            seen: set[Path] = set()
            for path in paths:
                if path not in seen:
                    seen.add(path)
                    deduplicated.append(path)
            contact_sheet(
                deduplicated,
                output / f"contact_{split}_{class_id + 1:02d}_{names[class_id]}.jpg",
                f"{names[class_id]} | GT green / predictions orange",
            )

    lines = [
        "# 赛道一 v7 前 10 类 + 第 17/26 类离线定向验收",
        "",
        f"- 时间：{summary['created_at']}",
        f"- 权重：`{weights}`",
        f"- 权重 SHA256：`{summary['weights_sha256']}`",
        f"- 数据：`{dataset_yaml}`，split=`{split}`，图片数={len(images)}",
        f"- 参数：conf={args.conf}, NMS IoU={args.iou}, match IoU={args.match_iou}, overlap warning IoU={args.overlap_warning_iou}, imgsz={args.imgsz}, device={args.device}",
        "- 安全属性：只读静态图片；不打开相机、不导入 ROS、不控制机械臂或夹爪。",
        "",
        "## 结论",
        "",
        f"- 前 10 类 GT 级核心检查：`{'通过' if front10_core_ok else '未通过'}`",
        f"- 第 17/26 类 GT 级核心检查：`{'通过' if new2_core_ok else '未通过'}`",
        f"- 前 10 类相对 v6 AP50 回退不超过 0.03：`{'通过' if front10_drop_ok else '未通过'}`",
        f"- 第 17/26 类 AP50 >= 0.30：`{'通过' if new2_ap_ok else '未通过'}`",
        f"- 零错误严格检查：`{'通过' if strict_ok else '未通过'}`（额外预测会使其失败，但不等于 GT 漏检）",
        f"- 可进入现场视觉验收：`{'是，仍需现场确认' if summary['decision']['deployment_candidate_ready_pending_onsite'] else '否'}`",
        "- 生产模型：**未替换**。",
        "",
        "## 逐类结果",
        "",
        "| 类别 | GT | TP | FP | FN | 漏检 | 错类 | 定位 | Recall | Precision | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in class_rows:
        lines.append(
            f"| {row['official_id']} {row['name']} | {row['gt']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['missing']} | {row['wrong_class']} | {row['localization']} | "
            f"{row['recall'] if row['recall'] is not None else '-'} | "
            f"{row['precision'] if row['precision'] is not None else '-'} | {row['status']} |"
        )
    lines += ["", "## 跨类重叠告警", ""]
    if not warnings:
        lines.append("没有记录到 IoU 达到告警阈值的跨类重叠。")
    else:
        lines += ["| 图片 | GT 类别 | 重叠预测类别 | 置信度 | IoU |", "|---|---|---|---:|---:|"]
        for warning in warnings:
            lines.append(
                f"| {Path(warning.image).name} | {warning.gt_class_id + 1}-{warning.gt_name} | "
                f"{warning.overlapping_class_id + 1}-{warning.overlapping_name} | {warning.score:.5f} | {warning.iou:.4f} |"
            )
    lines += ["", "## 错误明细", ""]
    if not errors:
        lines.append("没有记录到目标类别的 GT 级错误或额外同类预测。")
    else:
        lines += ["| 图片 | 目标类别 | 原因 | 预测类别 | 置信度 | IoU |", "|---|---|---|---|---:|---:|"]
        for error in errors:
            lines.append(
                f"| {Path(error.image).name} | {error.gt_class_id + 1}-{error.gt_name} | {error.reason} | "
                f"{error.prediction_class_id + 1 if error.prediction_class_id is not None else '-'}-{error.prediction_name or '-'} | "
                f"{error.prediction_score if error.prediction_score is not None else '-'} | {error.iou if error.iou is not None else '-'} |"
            )
    lines += [
        "",
        "## 输出文件",
        "",
        f"- JSON：`{output / 'v7_targeted_offline_audit_val.json'}`",
        f"- 逐类 CSV：`{output / 'v7_targeted_offline_audit_val_classes.csv'}`",
        f"- 错误 CSV：`{output / 'v7_targeted_offline_audit_val_errors.csv'}`",
        f"- 跨类重叠 CSV：`{output / 'v7_targeted_offline_audit_val_overlap_warnings.csv'}`",
        f"- 叠加图目录：`{overlay_dir}`",
        "",
        "> 说明：验证集样本量较少。本报告用于离线回归和现场前筛查，不代替真实相机、多物品、反光和遮挡条件下的现场验收。",
    ]
    (output / "v7_targeted_offline_audit_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "counts": summary["counts"],
        "decision": summary["decision"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["decision"]["deployment_candidate_ready_pending_onsite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
