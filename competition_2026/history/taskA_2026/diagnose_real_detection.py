#!/usr/bin/env python3
"""Diagnose live-preview preprocessing on all reviewed real images.

The script runs the exact center-pad operation used by live_camera_preview.py
and compares detections against the single-object YOLO labels. It is intended
for diagnosing scale and confidence sensitivity, not as a clean validation set:
the real images were also used to train the staged model.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class GroundTruth:
    class_id: int
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--v2-weights", required=True)
    parser.add_argument("--v3-weights", required=True)
    parser.add_argument("--v2-name", default="v2")
    parser.add_argument("--v3-name", default="v3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--image-prefix",
        default=None,
        help="Only evaluate image stems starting with this prefix.",
    )
    parser.add_argument("--scales", default="1.0,0.65")
    parser.add_argument(
        "--conf-thresholds",
        default="0.05,0.10,0.15,0.20,0.25",
    )
    return parser.parse_args()


def load_labels(label_dir: Path, stems: Iterable[str]) -> dict[str, list[GroundTruth]]:
    labels: dict[str, list[GroundTruth]] = {}
    for stem in stems:
        path = label_dir / f"{stem}.txt"
        rows: list[GroundTruth] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            class_id = int(float(fields[0]))
            cx, cy, width, height = (float(value) for value in fields[1:5])
            rows.append(
                GroundTruth(
                    class_id,
                    (
                        (cx - width / 2) * 640.0,
                        (cy - height / 2) * 480.0,
                        (cx + width / 2) * 640.0,
                        (cy + height / 2) * 480.0,
                    ),
                )
            )
        labels[stem] = rows
    return labels


def iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def center_scale(frame: np.ndarray, scale: float) -> tuple[np.ndarray, int, int]:
    if scale == 1.0:
        return frame, 0, 0
    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    x_offset = (width - scaled_width) // 2
    y_offset = (height - scaled_height) // 2
    canvas[
        y_offset : y_offset + scaled_height,
        x_offset : x_offset + scaled_width,
    ] = resized
    return canvas, x_offset, y_offset


def chunks(items: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def infer(
    model: YOLO,
    paths: list[Path],
    scale: float,
    x_offset: int,
    y_offset: int,
    imgsz: int,
    device: str,
    batch: int,
) -> dict[str, list[Prediction]]:
    output: dict[str, list[Prediction]] = {}
    for path_chunk in chunks(paths, batch):
        frames = []
        for path in path_chunk:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"cannot read image: {path}")
            frames.append(frame)

        if scale == 1.0:
            sources = frames
            offsets = [(0, 0)] * len(frames)
        else:
            prepared = [center_scale(frame, scale) for frame in frames]
            sources = [item[0] for item in prepared]
            offsets = [(item[1], item[2]) for item in prepared]

        results = model.predict(
            source=sources,
            imgsz=imgsz,
            conf=0.01,
            iou=0.7,
            max_det=100,
            device=device,
            verbose=False,
        )
        for path, result, offset in zip(path_chunk, results, offsets):
            x_offset, y_offset = offset
            predictions: list[Prediction] = []
            boxes = result.boxes
            if boxes is not None:
                for raw_box, confidence, class_id in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                ):
                    x1 = (float(raw_box[0]) - x_offset) / scale
                    y1 = (float(raw_box[1]) - y_offset) / scale
                    x2 = (float(raw_box[2]) - x_offset) / scale
                    y2 = (float(raw_box[3]) - y_offset) / scale
                    predictions.append(
                        Prediction(
                            int(class_id),
                            float(confidence),
                            (x1, y1, x2, y2),
                        )
                    )
            output[path.stem] = predictions
    return output


def summarize(
    ground_truth: dict[str, list[GroundTruth]],
    predictions: dict[str, list[Prediction]],
    threshold: float,
) -> dict[str, object]:
    total = 0
    correct_class_hits = 0
    matched_hits = 0
    correct_p75_hits = 0
    matched_p75_hits = 0
    false_positives = 0
    no_predictions = 0
    mean_correct_iou = 0.0
    mean_best_iou = 0.0
    per_class: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {
            "total": 0,
            "correct_hits": 0,
            "matched_hits": 0,
            "false_positives": 0,
            "no_predictions": 0,
            "iou_sum": 0.0,
        }
    )

    for stem, ground_truth_rows in ground_truth.items():
        active = [
            prediction
            for prediction in predictions.get(stem, [])
            if prediction.confidence >= threshold
        ]
        for truth in ground_truth_rows:
            total += 1
            class_scores = [
                iou(truth.box, prediction.box)
                for prediction in active
                if prediction.class_id == truth.class_id
            ]
            all_scores = [iou(truth.box, prediction.box) for prediction in active]
            correct_iou = max(class_scores, default=0.0)
            best_iou = max(all_scores, default=0.0)
            correct_hit = int(correct_iou >= 0.5)
            matched_hit = int(best_iou >= 0.5)
            correct_p75_hit = int(correct_iou >= 0.75)
            matched_p75_hit = int(best_iou >= 0.75)
            correct_class_hits += correct_hit
            matched_hits += matched_hit
            correct_p75_hits += correct_p75_hit
            matched_p75_hits += matched_p75_hit
            no_predictions += int(not active)
            mean_correct_iou += correct_iou
            mean_best_iou += best_iou

            row = per_class[truth.class_id]
            row["total"] += 1
            row["correct_hits"] += correct_hit
            row["matched_hits"] += matched_hit
            row["no_predictions"] += int(not active)
            row["iou_sum"] += correct_iou

        matched = set()
        for truth in ground_truth_rows:
            best_index = None
            best_value = 0.0
            for index, prediction in enumerate(active):
                if index in matched or prediction.class_id != truth.class_id:
                    continue
                value = iou(truth.box, prediction.box)
                if value > best_value:
                    best_index = index
                    best_value = value
            if best_index is not None and best_value >= 0.5:
                matched.add(best_index)
            else:
                fc = per_class[truth.class_id]
                fc["false_positives"] += 0
        false_positives += max(0, len(active) - len(matched))
        per_class[ground_truth_rows[0].class_id]["false_positives"] += max(
            0,
            len(active) - len(matched),
        )

    return {
        "total": total,
        "correct_class_hits": correct_class_hits,
        "matched_hits": matched_hits,
        "correct_p75_hits": correct_p75_hits,
        "matched_p75_hits": matched_p75_hits,
        "false_positives": false_positives,
        "no_predictions": no_predictions,
        "mean_correct_iou": mean_correct_iou / total if total else 0.0,
        "mean_best_iou": mean_best_iou / total if total else 0.0,
        "per_class": per_class,
    }


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images).resolve()
    label_dir = Path(args.labels).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.image_prefix:
        paths = [
            path
            for path in paths
            if path.stem.startswith(args.image_prefix)
        ]
    if not paths:
        raise SystemExit(f"no images in {image_dir}")
    ground_truth = load_labels(label_dir, (path.stem for path in paths))
    scales = [float(value) for value in args.scales.split(",")]
    thresholds = [float(value) for value in args.conf_thresholds.split(",")]
    models: dict[str, tuple[YOLO, dict[float, dict[str, list[Prediction]]]]] = {}
    for model_name, weights in (
        (args.v2_name, args.v2_weights),
        (args.v3_name, args.v3_weights),
    ):
        model = YOLO(weights)
        models[model_name] = (
            model,
            {
                scale: infer(
                    model,
                    paths,
                    scale,
                    0,
                    0,
                    args.imgsz,
                    args.device,
                    args.batch,
                )
                for scale in scales
            },
        )

    rows: list[dict[str, object]] = []
    for model_name, (_, model_predictions) in models.items():
        for scale in scales:
            for threshold in thresholds:
                stats = summarize(
                    ground_truth,
                    model_predictions[scale],
                    threshold,
                )
                rows.append(
                    {
                        "model": model_name,
                        "scale": scale,
                        "conf": threshold,
                        "images": len(paths),
                        "instances": stats["total"],
                        "correct_hits": stats["correct_class_hits"],
                        "correct_recall": (
                            stats["correct_class_hits"] / stats["total"]
                            if stats["total"]
                            else 0.0
                        ),
                        "matched_hits": stats["matched_hits"],
                        "matched_recall": (
                            stats["matched_hits"] / stats["total"]
                            if stats["total"]
                            else 0.0
                        ),
                        "correct_recall_iou75": (
                            stats["correct_p75_hits"] / stats["total"]
                            if stats["total"]
                            else 0.0
                        ),
                        "matched_recall_iou75": (
                            stats["matched_p75_hits"] / stats["total"]
                            if stats["total"]
                            else 0.0
                        ),
                        "false_positives": stats["false_positives"],
                        "no_predictions": stats["no_predictions"],
                        "mean_correct_iou": stats["mean_correct_iou"],
                        "mean_best_iou": stats["mean_best_iou"],
                    }
                )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Real-image preprocessing diagnosis",
        "",
        "This is a diagnostic on training images, not a clean validation result.",
        "",
        "| model | scale | conf | recall@0.5 | recall@0.75 | no pred | false positives | mean correct IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {row['model']} | {row['scale']:.2f} | {row['conf']:.2f} | "
            f"{row['correct_recall'] * 100:.1f}% | "
            f"{row['correct_recall_iou75'] * 100:.1f}% | "
            f"{row['no_predictions']} | {row['false_positives']} | "
            f"{row['mean_correct_iou']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Per-class recall at scale 1.0 and conf 0.10",
            "",
            "| model | class | total | correct hit | recall | no pred | mean IoU |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_name, (_, model_predictions) in models.items():
        stats = summarize(ground_truth, model_predictions[1.0], 0.10)
        for class_id in sorted(stats["per_class"]):
            item = stats["per_class"][class_id]
            report.append(
                f"| {model_name} | {class_id} | {item['total']} | "
                f"{item['correct_hits']} | "
                f"{item['correct_hits'] / item['total'] * 100:.1f}% | "
                f"{item['no_predictions']} | "
                f"{item['iou_sum'] / item['total']:.3f} |"
            )
    report_path = out_dir / "summary.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
