#!/usr/bin/env python3
"""Evaluate competition-style IoU from an eval_camera_class_20261016.py summary.

This script does not run a model and does not modify weights. It compares the
already-saved base/TTA detections in ``summary.json`` with manually reviewed
YOLO-format ground-truth labels.

A GT instance passes only when:
  1. the predicted class is the requested class, and
  2. the best matching prediction has IoU >= --iou-threshold.

The union result answers "did base OR TTA pass", not "is the deployed model
stable". Keep the three columns separate in reports.

Example:
  python eval_iou_from_summary.py \
    --summary reports/.../6-Foam_Brick_predictions_640/summary.json \
    --gt-dir labels/6-Foam_Brick \
    --image-dir eval_capture_20261016/6-Foam_Brick \
    --class-id 5 \
    --out-prefix reports/.../6-Foam_Brick_iou75
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=None,
                        help="只评估该类 GT；不填则评估标签中的全部类")
    parser.add_argument("--iou-threshold", type=float, default=0.75)
    parser.add_argument("--out-prefix", type=Path, default=None)
    parser.add_argument("--allow-missing-gt", action="store_true",
                        help="缺 GT 时跳过后继续；默认直接失败")
    return parser.parse_args()


def load_gt(label_path: Path) -> list[dict[str, Any]]:
    """Read YOLO cx/cy/w/h labels. Extra columns are ignored."""
    if not label_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{label_path}:{line_no}: expected at least 5 columns")
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            raise ValueError(f"{label_path}:{line_no}: non-finite coordinate")
        rows.append({
            "class_id": class_id,
            "cx": cx, "cy": cy, "w": w, "h": h,
            "line_no": line_no,
        })
    return rows


def yolo_to_xyxy(row: dict[str, Any], width: int, height: int) -> list[float]:
    cx, cy, w, h = row["cx"] * width, row["cy"] * height, row["w"] * width, row["h"] * height
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def best_match(gt_xyxy: list[float], detections: list[dict[str, Any]],
               class_id: int | None) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for det in detections:
        det_class = int(det["class_id"])
        if class_id is not None and det_class != class_id:
            continue
        overlap = iou_xyxy(gt_xyxy, det["xyxy"])
        if best is None or overlap > best["iou"]:
            best = {
                "iou": overlap,
                "class_id": det_class,
                "name": det.get("name", str(det_class)),
                "conf": float(det.get("conf", 0.0)),
                "xyxy": [float(v) for v in det["xyxy"]],
            }
    return best


def dimension(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise SystemExit("--iou-threshold must be in [0, 1]")
    if not args.summary.exists():
        raise SystemExit(f"summary not found: {args.summary}")
    if not args.image_dir.is_dir():
        raise SystemExit(f"image directory not found: {args.image_dir}")
    if not args.gt_dir.is_dir():
        raise SystemExit(f"GT directory not found: {args.gt_dir}")

    report = json.loads(args.summary.read_text(encoding="utf-8"))
    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list):
        raise SystemExit("summary.json has no rows list")

    out_prefix = args.out_prefix or args.summary.with_name(args.summary.stem + "_iou75")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    instances: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in raw_rows:
        filename = str(row.get("file", ""))
        if not filename:
            raise ValueError("summary row has no file")
        image_path = args.image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        width, height = dimension(image_path)
        label_path = args.gt_dir / (Path(filename).stem + ".txt")
        gt_rows = load_gt(label_path)
        if not gt_rows:
            missing.append(filename)
            if args.allow_missing_gt:
                continue
            raise SystemExit(f"GT label missing or empty: {label_path}")

        base = list(row.get("detections") or [])
        tta = list(row.get("tta_detections") or [])
        union = base + tta
        for gt_index, gt in enumerate(gt_rows, 1):
            gt_class = int(gt["class_id"])
            if args.class_id is not None and gt_class != args.class_id:
                continue
            gt_xyxy = yolo_to_xyxy(gt, width, height)
            base_best = best_match(gt_xyxy, base, args.class_id)
            tta_best = best_match(gt_xyxy, tta, args.class_id)
            union_best = best_match(gt_xyxy, union, args.class_id)
            any_best = best_match(gt_xyxy, union, None)

            def value(match: dict[str, Any] | None, key: str, default: Any = 0.0) -> Any:
                return default if match is None else match[key]

            item = {
                "file": filename,
                "gt_index": gt_index,
                "gt_class_id": gt_class,
                "gt_xyxy": [round(v, 2) for v in gt_xyxy],
                "base_iou": round(float(value(base_best, "iou")), 6),
                "base_conf": round(float(value(base_best, "conf")), 6),
                "base_source": "base" if base_best else "",
                "base_pass": int(base_best is not None and base_best["iou"] >= args.iou_threshold),
                "tta_iou": round(float(value(tta_best, "iou")), 6),
                "tta_conf": round(float(value(tta_best, "conf")), 6),
                "tta_source": "tta" if tta_best else "",
                "tta_pass": int(tta_best is not None and tta_best["iou"] >= args.iou_threshold),
                "union_iou": round(float(value(union_best, "iou")), 6),
                "union_conf": round(float(value(union_best, "conf")), 6),
                "union_pass": int(union_best is not None and union_best["iou"] >= args.iou_threshold),
                "any_iou": round(float(value(any_best, "iou")), 6),
                "any_class_id": int(value(any_best, "class_id", -1)),
                "any_name": str(value(any_best, "name", "")),
                "any_conf": round(float(value(any_best, "conf")), 6),
            }
            instances.append(item)

    if not instances:
        raise SystemExit("no GT instances selected for evaluation")

    base_passed = sum(r["base_pass"] for r in instances)
    tta_passed = sum(r["tta_pass"] for r in instances)
    union_passed = sum(r["union_pass"] for r in instances)
    mean_base = sum(r["base_iou"] for r in instances) / len(instances)
    mean_tta = sum(r["tta_iou"] for r in instances) / len(instances)
    mean_union = sum(r["union_iou"] for r in instances) / len(instances)
    summary = {
        "summary": str(args.summary),
        "gt_dir": str(args.gt_dir),
        "image_dir": str(args.image_dir),
        "class_id": args.class_id,
        "iou_threshold": args.iou_threshold,
        "instances": len(instances),
        "base_passed": base_passed,
        "tta_passed": tta_passed,
        "union_passed": union_passed,
        "base_pass_rate": base_passed / len(instances),
        "tta_pass_rate": tta_passed / len(instances),
        "union_pass_rate": union_passed / len(instances),
        "mean_base_iou": mean_base,
        "mean_tta_iou": mean_tta,
        "mean_union_iou": mean_union,
        "missing_gt": missing,
    }

    label = report.get("label", args.summary.parent.name)
    lines = [
        f"# {label} 比赛口径 IoU 评估",
        "",
        f"- summary：`{args.summary}`",
        f"- GT：`{args.gt_dir}`",
        f"- 类别：{args.class_id if args.class_id is not None else '全部'}",
        f"- 图像：`{args.image_dir}`",
        f"- 通过条件：类别正确且 IoU≥{args.iou_threshold}",
        f"- GT 实例：{len(instances)}",
        "",
        "| 路径 | 通过 | 通过率 | 平均 best IoU |",
        "| --- | ---: | ---: | ---: |",
        f"| 单帧 | {base_passed}/{len(instances)} | {base_passed / len(instances) * 100:.1f}% | {mean_base:.3f} |",
        f"| TTA | {tta_passed}/{len(instances)} | {tta_passed / len(instances) * 100:.1f}% | {mean_tta:.3f} |",
        f"| 单帧∪TTA | {union_passed}/{len(instances)} | {union_passed / len(instances) * 100:.1f}% | {mean_union:.3f} |",
        "",
    ]

    def table(title: str, passed_key: str, iou_key: str) -> None:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in instances:
            grouped.setdefault(item["gt_class_id"], []).append(item)
        lines.extend([f"## {title}", "", "| 类别 | 实例 | 通过 | 通过率 | 平均 best IoU |", "| ---: | ---: | ---: | ---: | ---: |"])
        for class_id in sorted(grouped):
            group = grouped[class_id]
            passed = sum(int(r[passed_key]) for r in group)
            mean = sum(float(r[iou_key]) for r in group) / len(group)
            lines.append(f"| {class_id} | {len(group)} | {passed} | {passed / len(group) * 100:.1f}% | {mean:.3f} |")
        lines.append("")

    table("按类别", "base_pass", "base_iou")
    lines.extend([
        "## 说明",
        "",
        "- 本脚本只读取已经保存的检测结果，不运行模型、不改权重、不修改服务。",
        "- 单帧和 TTA 分开报告；并集只表示至少一条路径通过，不代表稳定部署结果。",
        "- 如果 GT 尚未人工复核，本报告的数值只能作为临时诊断，不能作为比赛最终成绩。",
    ])
    if missing:
        lines.append(f"- 缺失或空 GT：{len(missing)} 张：" + ", ".join(missing))
    lines.append("")

    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    csv_path = Path(str(out_prefix) + "_instances.csv")
    json_path.write_text(json.dumps({"summary": summary, "instances": instances}, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(instances[0]))
        writer.writeheader()
        writer.writerows(instances)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Instances CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
