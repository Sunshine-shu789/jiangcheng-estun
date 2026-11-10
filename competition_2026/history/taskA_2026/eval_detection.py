#!/usr/bin/env python3
"""任务 A 检测评估：预测框 vs GT。

输入二选一：
    --pred-labels DIR   YOLO 格式预测目录（文件名与图片同名，.txt）
    --fake MODE         用 GT 造假的预测做自测
                        perfect / shift / shrink / jitter / drop / extra / random

输出：
    reports/eval_<split>_<tag>.md
    reports/eval_<split>_<tag>_instances.csv
    reports/eval_<split>_<tag>_overlays/*.png（可选，默认画前 6 个失败案例）
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import cepb

IMAGE_W = cepb.IMAGE_WIDTH
IMAGE_H = cepb.IMAGE_HEIGHT


@dataclass(frozen=True)
class Box:
    class_id: int
    box: Tuple[float, float, float, float]
    score: float = 1.0


def load_yolo_labels(
    path: Path,
    image_width: int = IMAGE_W,
    image_height: int = IMAGE_H,
) -> List[Box]:
    boxes: List[Box] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        score = float(parts[5]) if len(parts) >= 6 else 1.0
        x1 = (cx - w / 2) * image_width
        y1 = (cy - h / 2) * image_height
        x2 = (cx + w / 2) * image_width
        y2 = (cy + h / 2) * image_height
        boxes.append(Box(class_id, (x1, y1, x2, y2), score))
    return boxes


def save_yolo_labels(
    path: Path,
    boxes: Sequence[Box],
    image_width: int = IMAGE_W,
    image_height: int = IMAGE_H,
) -> None:
    lines = []
    for item in boxes:
        x1, y1, x2, y2 = item.box
        cx = (x1 + x2) / 2 / image_width
        cy = (y1 + y2) / 2 / image_height
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height
        lines.append(f"{item.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_camera(image_name: str) -> str:
    real_match = re.match(r"real_.*_v(\d+)_", image_name)
    if real_match:
        return f"real_v{real_match.group(1)}"
    return image_name.split("_camera_", 1)[0]


def parse_light(image_name: str) -> str:
    match = re.search(
        r"_camera_(directional_light|point_light|spot_light)_", image_name
    )
    return match.group(1) if match else "unknown"


def parse_scene(image_name: str) -> Optional[int]:
    match = re.search(r"_scene_(\d+)_", image_name)
    return int(match.group(1)) if match else None


def load_visibility(instances_csv: Path) -> Dict[Tuple[str, int], float]:
    table: Dict[Tuple[str, int], float] = {}
    if not instances_csv.exists():
        return table
    with instances_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["image"], int(row["class_id"]))
            table[key] = float(row["visibility"])
    return table


def fake_boxes(
    gt: Sequence[Box],
    mode: str,
    param: float,
    rng: random.Random,
    image_width: int = IMAGE_W,
    image_height: int = IMAGE_H,
) -> List[Box]:
    if mode == "perfect":
        return list(gt)
    result: List[Box] = []
    for item in gt:
        x1, y1, x2, y2 = item.box
        if mode == "shift":
            dx = dy = param
        elif mode == "shrink":
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dx = (x2 - x1) * (param - 1) / 2
            dy = (y2 - y1) * (param - 1) / 2
        elif mode in ("jitter", "random"):
            dx = rng.uniform(-param, param) * (x2 - x1)
            dy = rng.uniform(-param, param) * (y2 - y1)
        else:
            dx = dy = 0.0
        box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        if mode == "shrink":
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half_w = (x2 - x1) * param / 2
            half_h = (y2 - y1) * param / 2
            box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        if mode == "drop" and rng.random() < param:
            continue
        result.append(Box(item.class_id, box))
    if mode in ("extra", "random"):
        count = max(1, int(param)) if mode == "extra" else 1
        for _ in range(count):
            cx = rng.uniform(0.15, 0.85) * image_width
            cy = rng.uniform(0.15, 0.85) * image_height
            w = rng.uniform(0.05, 0.25) * image_width
            h = rng.uniform(0.05, 0.25) * image_height
            result.append(
                Box(rng.randrange(10), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            )
    return result


@dataclass
class ImageResult:
    image: str
    rows: List[Dict[str, object]]
    false_positives: int


def evaluate_image(
    image: str,
    gt: Sequence[Box],
    pred: Sequence[Box],
    visibility: Dict[Tuple[str, int], float],
    match_iou: float,
) -> ImageResult:
    best_iou = [0.0] * len(gt)
    best_pred: List[Optional[int]] = [None] * len(gt)
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if p.class_id != g.class_id:
                continue
            iou = cepb.iou_aabb(g.box, p.box)
            if iou > best_iou[gi]:
                best_iou[gi] = iou
                best_pred[gi] = pi

    pairs = sorted(
        (
            (cepb.iou_aabb(g.box, p.box), gi, pi)
            for gi, g in enumerate(gt)
            for pi, p in enumerate(pred)
            if g.class_id == p.class_id
        ),
        reverse=True,
    )
    matched_gt, matched_pred = set(), set()
    for iou, gi, pi in pairs:
        if iou < match_iou:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)

    camera, light, scene = parse_camera(image), parse_light(image), parse_scene(image)
    rows: List[Dict[str, object]] = []
    for gi, g in enumerate(gt):
        rows.append(
            {
                "image": image,
                "scene": scene,
                "camera": camera,
                "light": light,
                "class_id": g.class_id,
                "visibility": visibility.get((image, g.class_id)),
                "best_iou": best_iou[gi],
                "pred_index": best_pred[gi],
                "matched": int(gi in matched_gt),
            }
        )
    return ImageResult(image, rows, len(pred) - len(matched_pred))


def visibility_bin(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "0.00-0.25"
    if value < 0.5:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def aggregate(rows: Sequence[Dict[str, object]], iou_threshold: float) -> Dict[str, object]:
    total = len(rows)
    passed = sum(1 for r in rows if float(r["best_iou"]) >= iou_threshold)
    matched = sum(int(r["matched"]) for r in rows)
    ious = np.asarray([float(r["best_iou"]) for r in rows])
    return {
        "n": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "mean_iou": float(ious.mean()) if total else 0.0,
        "recall@match": matched / total if total else 0.0,
    }


def table(
    title: str, groups: Dict[str, List[Dict[str, object]]], iou_threshold: float
) -> List[str]:
    lines = [
        f"## {title}",
        "",
        "| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in sorted(groups):
        stats = aggregate(groups[key], iou_threshold)
        lines.append(
            f"| {key} | {stats['n']} | {stats['passed']} | "
            f"{stats['pass_rate'] * 100:.1f}% | {stats['mean_iou']:.3f} |"
        )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(cepb.WORKDIR / "dataset_v1"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--pred-labels", default=None)
    parser.add_argument(
        "--fake",
        default=None,
        choices=["perfect", "shift", "shrink", "jitter", "drop", "extra", "random"],
    )
    parser.add_argument("--fake-param", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.75)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--save-fake", default=None)
    parser.add_argument("--overlays", type=int, default=6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not args.pred_labels and not args.fake:
        parser.error("必须给 --pred-labels 或 --fake")
    data = Path(args.data)
    image_dir = data / "images" / args.split
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise SystemExit(f"{image_dir} 下没有图片")
    visibility = load_visibility(data / "instances.csv")

    defaults = {
        "shift": 10.0,
        "shrink": 0.5,
        "jitter": 0.1,
        "drop": 0.3,
        "extra": 2.0,
        "random": 0.2,
        "perfect": 0.0,
    }
    param = args.fake_param if args.fake_param is not None else defaults.get(args.fake, 0.0)
    rng = random.Random(args.seed)
    tag = Path(args.pred_labels).name if args.pred_labels else f"fake_{args.fake}"
    reports = cepb.WORKDIR / "reports"
    out_prefix = Path(args.out) if args.out else reports / f"eval_{args.split}_{tag}"

    all_rows: List[Dict[str, object]] = []
    false_positives = 0
    fail_cases: List[Tuple[float, str, Dict[str, object]]] = []
    pred_by_image: Dict[str, List[Box]] = {}

    for image_path in images:
        image_name = image_path.name
        with Image.open(image_path) as image_info:
            image_width, image_height = image_info.size
        gt = load_yolo_labels(
            data / "labels" / args.split / (image_path.stem + ".txt"),
            image_width,
            image_height,
        )
        if args.pred_labels:
            pred = load_yolo_labels(
                Path(args.pred_labels) / (image_path.stem + ".txt"),
                image_width,
                image_height,
            )
        else:
            pred = fake_boxes(
                gt,
                args.fake,
                param,
                rng,
                image_width,
                image_height,
            )
            if args.save_fake:
                save_yolo_labels(
                    Path(args.save_fake) / (image_path.stem + ".txt"),
                    pred,
                    image_width,
                    image_height,
                )
        result = evaluate_image(image_name, gt, pred, visibility, args.match_iou)
        pred_by_image[image_name] = list(pred)
        for row in result.rows:
            row["passed"] = int(float(row["best_iou"]) >= args.iou_threshold)
        all_rows.extend(result.rows)
        false_positives += result.false_positives
        for row in result.rows:
            if not row["passed"]:
                fail_cases.append((float(row["best_iou"]), image_name, row))

    groups_class: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    groups_vis: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    groups_camera: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    groups_light: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        groups_class[f"class {row['class_id']}"].append(row)
        groups_vis[visibility_bin(row["visibility"])].append(row)
        groups_camera[str(row["camera"])].append(row)
        groups_light[str(row["light"])].append(row)

    overall = aggregate(all_rows, args.iou_threshold)
    report = [
        f"# 检测评估：{args.split} / {tag}",
        "",
        f"- 数据：`{data}`",
        f"- 图片：{len(images)}，GT 实例：{len(all_rows)}",
        f"- 通过阈值：IoU ≥ {args.iou_threshold}",
        f"- 匹配阈值：IoU ≥ {args.match_iou}（用于 recall）",
        f"- 假预测参数：{param}（seed={args.seed}）" if args.fake else "- 预测来源：外部 YOLO 标签",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 通过率（IoU≥{args.iou_threshold}） | {overall['pass_rate'] * 100:.1f}% |",
        f"| 平均 best IoU | {overall['mean_iou']:.3f} |",
        f"| Recall@IoU{args.match_iou} | {overall['recall@match'] * 100:.1f}% |",
        f"| 误检（FP 框） | {false_positives} |",
        "",
    ]
    report += table("按类别", groups_class, args.iou_threshold)
    report += table("按可见度", groups_vis, args.iou_threshold)
    report += table("按相机", groups_camera, args.iou_threshold)
    report += table("按光照", groups_light, args.iou_threshold)

    fail_cases.sort(key=lambda item: item[0])
    report += [
        "## 最差的 10 个实例",
        "",
        "| 图片 | 类别 | 可见度 | best IoU |",
        "| --- | --- | --- | --- |",
    ]
    for iou, image_name, row in fail_cases[:10]:
        vis = row["visibility"]
        vis_text = f"{vis:.2f}" if isinstance(vis, float) else "-"
        report.append(
            f"| {image_name} | class {row['class_id']} | {vis_text} | {iou:.3f} |"
        )
    report.append("")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".md")).write_text("\n".join(report), encoding="utf-8")
    csv_path = Path(str(out_prefix) + "_instances.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image",
                "scene",
                "camera",
                "light",
                "class_id",
                "visibility",
                "best_iou",
                "pred_index",
                "matched",
                "passed",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    if args.overlays > 0 and fail_cases:
        overlay_dir = Path(str(out_prefix) + "_overlays")
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for iou, image_name, row in fail_cases[: args.overlays]:
            image = Image.open(data / "images" / args.split / image_name).convert("RGB")
            draw = ImageDraw.Draw(image)
            gt = load_yolo_labels(
                data / "labels" / args.split / (Path(image_name).stem + ".txt"),
                image.width,
                image.height,
            )
            for item in gt:
                if item.class_id == row["class_id"]:
                    draw.rectangle(item.box, outline=(0, 255, 0), width=6)
            pred = pred_by_image.get(image_name, [])
            for item in pred:
                if item.class_id == row["class_id"]:
                    draw.rectangle(item.box, outline=(255, 0, 0), width=6)
            draw.text((20, 20), f"class {row['class_id']} IoU={iou:.3f}", fill=(255, 255, 0))
            image.save(overlay_dir / image_name)

    print("\n".join(report[:30]))
    print(f"\n报告：{out_prefix.with_suffix('.md')}")
    print(f"实例 CSV：{csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
