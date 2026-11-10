#!/usr/bin/env python3
"""Build a camera-free visual acceptance gallery for the Track 1 v11 model."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
DEFAULT_DATASET = PROJECT_ROOT / "datasets/track1_training_merged_v10_all_multiclass_augmented_20261031"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs/track1_train_40class_v11_full_v10aug_from_yolo11s"
    / "track1_40class_v11_20261031_234506/weights/best.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/track1_v11_offline_acceptance_20261101"
EXPECTED_SHA256 = "f43234a94f1d71b24cb187082c18eb22fea84c88922c605df5f7b1a79ffcaaa6"
WEAK_OFFICIAL_IDS = [3, 7, 18, 24, 25, 32, 33, 37]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_color(class_index: int) -> tuple[int, int, int]:
    hue = int((class_index * 37) % 180)
    hsv = np.uint8([[[hue, 215, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def read_yolo_label(path: Path) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if not path.is_file():
        return boxes
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"标签格式错误 {path}:{line_number}: {raw!r}")
        class_index = int(parts[0])
        cx, cy, width, height = (float(value) for value in parts[1:5])
        boxes.append(
            {
                "class_index": class_index,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
            }
        )
    return boxes


def yolo_to_xyxy(box: dict[str, Any], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = int(round((box["cx"] - box["width"] / 2) * image_width))
    y1 = int(round((box["cy"] - box["height"] / 2) * image_height))
    x2 = int(round((box["cx"] + box["width"] / 2) * image_width))
    y2 = int(round((box["cy"] + box["height"] / 2) * image_height))
    return (
        max(0, min(image_width - 1, x1)),
        max(0, min(image_height - 1, y1)),
        max(0, min(image_width - 1, x2)),
        max(0, min(image_height - 1, y2)),
    )


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: float = 0.42,
    thickness: int = 1,
    filled: bool = True,
) -> None:
    height, width = image.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    label_height = text_height + baseline + 5
    label_width = text_width + 6
    label_x = max(0, min(width - label_width, x))
    label_y = y - label_height
    if label_y < 0:
        label_y = min(height - label_height, y + 2)
    if filled:
        cv2.rectangle(
            image,
            (label_x, label_y),
            (label_x + label_width, label_y + label_height),
            color,
            -1,
            cv2.LINE_AA,
        )
        luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = (10, 10, 10) if luminance > 145 else (255, 255, 255)
        cv2.putText(
            image,
            text,
            (label_x + 3, label_y + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            image,
            text,
            (label_x + 2, max(text_height + 2, label_y + text_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def annotate_ground_truth(image: np.ndarray, boxes: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for box in boxes:
        class_index = int(box["class_index"])
        x1, y1, x2, y2 = yolo_to_xyxy(box, width, height)
        color = (60, 220, 60)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_label(output, f"GT {class_index + 1:02d} {names[class_index]}", x1, y1, color)
    return output


def annotate_predictions(
    image: np.ndarray,
    predictions: list[dict[str, Any]],
    names: list[str],
    accepted_conf: float,
) -> np.ndarray:
    output = image.copy()
    for prediction in predictions:
        confidence = float(prediction["confidence"])
        class_index = int(prediction["class_index"])
        x1, y1, x2, y2 = (int(round(value)) for value in prediction["xyxy"])
        if confidence >= accepted_conf:
            color = class_color(class_index)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            draw_label(output, f"{class_index + 1:02d} {names[class_index]} {confidence:.2f}", x1, y1, color)
        else:
            color = (145, 145, 145)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            draw_label(
                output,
                f"低 {class_index + 1:02d} {names[class_index]} {confidence:.2f}",
                x1,
                y1,
                color,
                scale=0.38,
                filled=False,
            )
    return output


def add_panel_header(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    canvas = np.full((height + 38, width, 3), 28, dtype=np.uint8)
    canvas[38:, :, :] = image
    cv2.putText(
        canvas,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_comparison(
    image_path: Path,
    label_path: Path,
    names: list[str],
    predictions: list[dict[str, Any]],
    accepted_conf: float,
    output_path: Path,
) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    gt_boxes = read_yolo_label(label_path)
    accepted = [item for item in predictions if float(item["confidence"]) >= accepted_conf]
    candidates = [item for item in predictions if float(item["confidence"]) < accepted_conf]
    left = annotate_ground_truth(image, gt_boxes, names)
    right = annotate_predictions(image, predictions, names, accepted_conf)
    left = add_panel_header(
        left,
        f"人工标注 GT | boxes={len(gt_boxes)}",
        (70, 235, 70),
    )
    issue_count = count_obvious_issues(gt_boxes, accepted, image.shape[1], image.shape[0])
    header_color = (70, 220, 70) if issue_count == 0 else (70, 190, 255)
    right = add_panel_header(
        right,
        f"V11 预测 | accepted>={accepted_conf:.2f}: {len(accepted)} | 灰线候选: {len(candidates)}",
        header_color,
    )
    divider = np.full((max(left.shape[0], right.shape[0]), 8, 3), 15, dtype=np.uint8)
    height = max(left.shape[0], right.shape[0])

    def pad_panel(panel: np.ndarray) -> np.ndarray:
        if panel.shape[0] == height:
            return panel
        return cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))

    comparison = cv2.hconcat([pad_panel(left), divider, pad_panel(right)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise ValueError(f"无法写出图片：{output_path}")
    return len(gt_boxes), len(accepted)


def match_boxes(
    gt_boxes: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gt_xyxy = [yolo_to_xyxy(box, image_width, image_height) for box in gt_boxes]
    used_predictions: set[int] = set()
    matched_gt: set[int] = set()
    wrong_pairs: list[dict[str, Any]] = []
    for gt_index, gt_box in enumerate(gt_boxes):
        candidates = []
        for pred_index, prediction in enumerate(predictions):
            if pred_index in used_predictions:
                continue
            iou = box_iou(gt_xyxy[gt_index], tuple(prediction["xyxy"]))
            if iou >= iou_threshold:
                candidates.append((iou, pred_index))
        candidates.sort(reverse=True)
        same_class = next(
            (
                (iou, pred_index)
                for iou, pred_index in candidates
                if int(predictions[pred_index]["class_index"]) == int(gt_box["class_index"])
            ),
            None,
        )
        if same_class is not None:
            used_predictions.add(same_class[1])
            matched_gt.add(gt_index)
        elif candidates:
            wrong_pairs.append({"gt_index": gt_index, "pred_index": candidates[0][1], "iou": candidates[0][0]})
    missing = [gt_boxes[index] for index in range(len(gt_boxes)) if index not in matched_gt]
    extra = [prediction for index, prediction in enumerate(predictions) if index not in used_predictions]
    return missing, extra, wrong_pairs


def count_obvious_issues(
    gt_boxes: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> int:
    missing, extra, wrong = match_boxes(gt_boxes, predictions, image_width, image_height)
    return len(missing) + len(extra) + len(wrong)


def format_classes(boxes: list[dict[str, Any]], names: list[str]) -> list[str]:
    return sorted({names[int(item["class_index"])] for item in boxes})


def build_gallery_html(records: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    summary = metadata["summary"]
    weak_names = [metadata["names"][index - 1] for index in WEAK_OFFICIAL_IDS]
    cards: list[str] = []
    for record in records:
        groups = ["all"]
        if record["is_multiobject"]:
            groups.append("multi")
        if record["weak_hit"]:
            groups.append("weak")
        if record["issue_count"]:
            groups.append("issue")
        group_value = " ".join(groups)
        missing_html = "、".join(html.escape(value) for value in record["missing_classes"]) or "无"
        extra_html = "、".join(html.escape(value) for value in record["extra_classes"]) or "无"
        wrong_html = "、".join(html.escape(value) for value in record["wrong_pairs"]) or "无"
        image_rel = html.escape(record["image_rel"])
        file_name = html.escape(record["file_name"])
        cards.append(
            f"""
            <article class="card" data-groups="{group_value}" data-name="{file_name.lower()}">
              <div class="card-head">
                <div><strong>{file_name}</strong><span class="badge">{'多物品' if record['is_multiobject'] else '单物品/组合'}</span></div>
                <div class="counts">GT {record['gt_count']} / V11 {record['prediction_count']}</div>
              </div>
              <a href="{image_rel}" target="_blank"><img loading="lazy" src="{image_rel}" alt="{file_name}"></a>
              <div class="issues">
                <span>疑似漏检：{missing_html}</span>
                <span>疑似错类：{wrong_html}</span>
                <span>额外/重复：{extra_html}</span>
              </div>
              <div class="review" data-file="{file_name}">
                <button data-status="pass">通过</button>
                <button data-status="issue">有问题</button>
                <button data-status="pending">待复核</button>
                <span class="saved">未标记</span>
              </div>
            </article>
            """
        )

    cards_html = "\n".join(cards)
    conf = metadata["accepted_conf"]
    inference_conf = metadata["inference_conf"]
    model_sha = metadata["model_sha256"]
    generated = html.escape(metadata["generated_at"])
    weak_html = "、".join(html.escape(value) for value in weak_names)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>赛道一 v11 离线视觉验收</title>
<style>
:root {{ color-scheme: dark; --bg:#111418; --panel:#1b2027; --line:#343b46; --text:#eef2f7; --muted:#a8b3c1; --ok:#40c77a; --warn:#ffbd4a; --bad:#ff6b6b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,"Noto Sans CJK SC",sans-serif; }}
header {{ position:sticky; top:0; z-index:10; background:rgba(17,20,24,.96); border-bottom:1px solid var(--line); padding:16px 22px; backdrop-filter:blur(8px); }}
h1 {{ margin:0 0 8px; font-size:22px; }}
.meta {{ color:var(--muted); overflow-wrap:anywhere; }}
.metrics {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:12px; }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 12px; min-width:150px; }}
.metric b {{ display:block; font-size:18px; color:var(--ok); }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
button {{ border:1px solid var(--line); background:var(--panel); color:var(--text); border-radius:7px; padding:7px 12px; cursor:pointer; }}
button:hover,button.active {{ border-color:#6aa9ff; background:#233149; }}
input {{ width:min(420px,100%); border:1px solid var(--line); background:var(--panel); color:var(--text); border-radius:7px; padding:8px 11px; }}
main {{ padding:18px 22px 40px; display:grid; gap:18px; grid-template-columns:repeat(auto-fit,minmax(min(100%,620px),1fr)); }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; min-width:0; }}
.card[hidden] {{ display:none; }}
.card-head {{ display:flex; justify-content:space-between; gap:10px; align-items:center; margin-bottom:8px; }}
.badge {{ display:inline-block; margin-left:8px; padding:1px 7px; border-radius:99px; background:#32445e; color:#cfe4ff; font-size:12px; }}
.counts {{ color:var(--muted); white-space:nowrap; }}
img {{ display:block; width:100%; height:auto; border-radius:6px; background:#000; }}
.issues {{ display:flex; flex-direction:column; gap:2px; color:var(--muted); font-size:12px; margin:8px 2px; }}
.review {{ display:flex; gap:7px; align-items:center; }}
.review .saved {{ margin-left:auto; color:var(--muted); }}
.review button[data-status="pass"].active {{ background:#174d31; border-color:var(--ok); }}
.review button[data-status="issue"].active {{ background:#572224; border-color:var(--bad); }}
.review button[data-status="pending"].active {{ background:#4f3c16; border-color:var(--warn); }}
.note {{ color:var(--muted); margin-top:10px; }}
@media (max-width:700px) {{ header {{ padding:12px; }} main {{ padding:12px; }} .card-head {{ align-items:flex-start; flex-direction:column; }} }}
</style>
</head>
<body>
<header>
  <h1>赛道一 40 类 v11 离线视觉验收</h1>
  <div class="meta">模型 SHA256: {html.escape(model_sha)}<br>生成时间: {generated} · 实线预测阈值 {conf:.2f} · 灰线候选阈值 {inference_conf:.2f}</div>
  <div class="metrics">
    <div class="metric">全量验证 mAP50<b>{summary['all_map50']:.4f}</b></div>
    <div class="metric">全量验证 mAP50-95<b>{summary['all_map50_95']:.4f}</b></div>
    <div class="metric">多物品 mAP50<b>{summary['multi_map50']:.4f}</b></div>
    <div class="metric">多物品 mAP50-95<b>{summary['multi_map50_95']:.4f}</b></div>
    <div class="metric">图片数量<b>{len(records)}</b></div>
  </div>
  <div class="note">重点类别：{weak_html}。左图为人工标注，右图为 v11；灰色框是低于正式阈值的候选，不代表正式输出。页面中的“疑似”项按 IoU 0.50 自动预筛，最终以人工复核为准。</div>
  <div class="toolbar">
    <button class="filter active" data-filter="multi">多物品优先</button>
    <button class="filter" data-filter="weak">重点类别</button>
    <button class="filter" data-filter="issue">疑似问题</button>
    <button class="filter" data-filter="all">全部图片</button>
    <input id="search" placeholder="搜索文件名">
    <button id="export">导出复核结果 CSV</button>
  </div>
</header>
<main>{cards_html}</main>
<script>
const STORAGE_KEY = 'track1_v11_offline_review_v1';
const reviewState = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
const cards = [...document.querySelectorAll('.card')];
const filterButtons = [...document.querySelectorAll('.filter')];
const search = document.getElementById('search');
let activeFilter = 'multi';

function applyState(card) {{
  const file = card.querySelector('.review').dataset.file;
  const value = reviewState[file] || '';
  card.querySelectorAll('.review button').forEach(button => button.classList.toggle('active', button.dataset.status === value));
  card.querySelector('.saved').textContent = value === 'pass' ? '已标记通过' : value === 'issue' ? '已标记问题' : value === 'pending' ? '待复核' : '未标记';
}}
function applyFilters() {{
  const query = search.value.trim().toLowerCase();
  cards.forEach(card => {{
    const matchesFilter = card.dataset.groups.split(' ').includes(activeFilter);
    const matchesSearch = !query || card.dataset.name.includes(query);
    card.hidden = !(matchesFilter && matchesSearch);
  }});
}}
filterButtons.forEach(button => button.addEventListener('click', () => {{
  activeFilter = button.dataset.filter;
  filterButtons.forEach(item => item.classList.toggle('active', item === button));
  applyFilters();
}}));
search.addEventListener('input', applyFilters);
document.querySelectorAll('.review button').forEach(button => button.addEventListener('click', () => {{
  const review = button.closest('.review');
  const file = review.dataset.file;
  const current = reviewState[file];
  if (current === button.dataset.status) delete reviewState[file]; else reviewState[file] = button.dataset.status;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(reviewState));
  applyState(button.closest('.card'));
}}));
document.getElementById('export').addEventListener('click', () => {{
  const rows = [['file', 'review_status']];
  cards.forEach(card => {{
    const file = card.querySelector('.review').dataset.file;
    rows.push([file, reviewState[file] || 'pending']);
  }});
  const csv = '\\ufeff' + rows.map(row => row.map(value => '"' + String(value).replaceAll('"', '""') + '"').join(',')).join('\\r\\n');
  const blob = new Blob([csv], {{type:'text/csv;charset=utf-8'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'track1_v11_offline_review.csv';
  link.click();
  URL.revokeObjectURL(link.href);
}});
cards.forEach(applyState);
applyFilters();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成赛道一 v11 无相机离线视觉验收页面")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--accepted-conf", type=float, default=0.15)
    parser.add_argument("--inference-conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = args.dataset.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    images_dir = dataset / "images/val"
    labels_dir = dataset / "labels/val"
    dataset_yaml = dataset / "dataset.yaml"
    if not images_dir.is_dir() or not labels_dir.is_dir() or not dataset_yaml.is_file():
        raise FileNotFoundError(f"验证集目录不完整：{dataset}")
    if not weights.is_file():
        raise FileNotFoundError(f"模型不存在：{weights}")
    if not 0 < args.inference_conf <= args.accepted_conf < 1:
        raise ValueError("必须满足 0 < inference_conf <= accepted_conf < 1")
    model_sha = sha256_file(weights)
    if model_sha != EXPECTED_SHA256:
        raise ValueError(f"模型哈希异常：{model_sha} != {EXPECTED_SHA256}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在；使用 --overwrite 重建：{output}")
        shutil.rmtree(output)
    annotated_dir = output / "images"
    annotated_dir.mkdir(parents=True)

    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = [str(value) for value in config["names"]]
    model = YOLO(str(weights), task="detect")
    model_names = [str(model.names[index]) for index in sorted(model.names)]
    if model_names != names or len(names) != 40:
        raise ValueError("模型和数据集类别表不一致")

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError("验证集没有图片")

    records: list[dict[str, Any]] = []
    predicted_class_counts: Counter[str] = Counter()
    gt_class_counts: Counter[str] = Counter()
    for index, image_path in enumerate(image_paths, 1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        result = model.predict(
            source=str(image_path),
            conf=args.inference_conf,
            iou=0.60,
            imgsz=args.imgsz,
            device=args.device,
            max_det=100,
            verbose=False,
        )[0]
        predictions: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, confidence, class_index in zip(xyxy, confidences, classes):
                predictions.append(
                    {
                        "xyxy": [float(value) for value in box],
                        "confidence": float(confidence),
                        "class_index": int(class_index),
                    }
                )
        predictions.sort(key=lambda item: -float(item["confidence"]))
        gt_boxes = read_yolo_label(label_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片：{image_path}")
        image_height, image_width = image.shape[:2]
        accepted = [item for item in predictions if float(item["confidence"]) >= args.accepted_conf]
        missing, extra, wrong = match_boxes(gt_boxes, accepted, image_width, image_height)
        missing_classes = format_classes(missing, names)
        extra_classes = format_classes(extra, names)
        wrong_pairs = sorted(
            {
                f"{names[int(gt_boxes[pair['gt_index']]['class_index'])]}→"
                f"{names[int(accepted[pair['pred_index']]['class_index'])]}"
                for pair in wrong
            }
        )
        weak_indices = {index - 1 for index in WEAK_OFFICIAL_IDS}
        weak_hit = any(int(item["class_index"]) in weak_indices for item in gt_boxes) or any(
            int(item["class_index"]) in weak_indices for item in accepted
        )
        output_name = f"{index:03d}_{image_path.stem}.jpg"
        output_image = annotated_dir / output_name
        gt_count, prediction_count = make_comparison(
            image_path,
            label_path,
            names,
            predictions,
            args.accepted_conf,
            output_image,
        )
        for box in gt_boxes:
            gt_class_counts[names[int(box["class_index"])]] += 1
        for prediction in accepted:
            predicted_class_counts[names[int(prediction["class_index"])]] += 1
        records.append(
            {
                "file_name": image_path.name,
                "image_rel": f"images/{output_name}",
                "source_image": str(image_path),
                "source_label": str(label_path),
                "is_multiobject": image_path.name.startswith("multiobj__"),
                "weak_hit": weak_hit,
                "gt_count": gt_count,
                "prediction_count": prediction_count,
                "candidate_count_below_threshold": len(predictions) - prediction_count,
                "missing_classes": missing_classes,
                "extra_classes": extra_classes,
                "wrong_pairs": wrong_pairs,
                "issue_count": len(missing) + len(extra) + len(wrong),
                "predictions": [
                    {
                        "class_index": item["class_index"],
                        "class_name": names[int(item["class_index"])],
                        "confidence": round(float(item["confidence"]), 6),
                        "xyxy": [round(float(value), 2) for value in item["xyxy"]],
                    }
                    for item in predictions
                ],
                "ground_truth": gt_boxes,
            }
        )
        print(f"[{index:03d}/{len(image_paths)}] {image_path.name} GT={gt_count} pred={prediction_count}", flush=True)

    all_metric = json.loads(
        (PROJECT_ROOT / "reports/track1_training_v11_full_v10aug_post_train_latest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "赛道一 v11 无相机离线视觉验收",
        "weights": str(weights),
        "model_sha256": model_sha,
        "dataset": str(dataset),
        "accepted_conf": args.accepted_conf,
        "inference_conf": args.inference_conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "names": names,
        "weak_official_ids": WEAK_OFFICIAL_IDS,
        "summary": {
            "images": len(records),
            "multiobject_images": sum(1 for record in records if record["is_multiobject"]),
            "ground_truth_boxes": sum(record["gt_count"] for record in records),
            "predictions_at_threshold": sum(record["prediction_count"] for record in records),
            "images_with_obvious_issue_signal": sum(1 for record in records if record["issue_count"]),
            "all_map50": float(all_metric["all_val"]["map50"]),
            "all_map50_95": float(all_metric["all_val"]["map50_95"]),
            "multi_map50": float(all_metric["multiobject_val"]["map50"]),
            "multi_map50_95": float(all_metric["multiobject_val"]["map50_95"]),
        },
        "ground_truth_class_counts": dict(sorted(gt_class_counts.items())),
        "prediction_class_counts": dict(sorted(predicted_class_counts.items())),
        "notes": [
            "左图为人工标注，右图为模型预测。",
            "灰线为低于正式阈值的候选，不代表正式输出。",
            "疑似漏检/错类/额外框按 IoU 0.50 自动预筛，仅供人工复核。",
            "本页面不控制相机、机械臂或夹爪。",
        ],
    }
    (output / "review_data.json").write_text(
        json.dumps({"metadata": metadata, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "review_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_name",
                "multiobject",
                "weak_class_image",
                "gt_count",
                "prediction_count",
                "candidate_below_threshold",
                "missing_classes",
                "wrong_pairs",
                "extra_classes",
                "review_status",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["file_name"],
                    int(record["is_multiobject"]),
                    int(record["weak_hit"]),
                    record["gt_count"],
                    record["prediction_count"],
                    record["candidate_count_below_threshold"],
                    "|".join(record["missing_classes"]),
                    "|".join(record["wrong_pairs"]),
                    "|".join(record["extra_classes"]),
                    "pending",
                ]
            )
    (output / "index.html").write_text(build_gallery_html(records, metadata), encoding="utf-8")
    (output / "README.txt").write_text(
        "赛道一 v11 无相机离线视觉验收\n"
        f"模型：{weights}\n"
        f"SHA256：{model_sha}\n"
        f"数据集：{dataset}\n"
        f"正式阈值：{args.accepted_conf:.2f}\n"
        f"灰线候选阈值：{args.inference_conf:.2f}\n"
        f"打开：{output / 'index.html'}\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(f"gallery={output / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
