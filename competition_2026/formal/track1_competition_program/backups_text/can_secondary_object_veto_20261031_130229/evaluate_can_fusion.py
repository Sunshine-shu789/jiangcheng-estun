#!/usr/bin/env python3
"""Offline threshold and regression audit for primary + dedicated-can fusion."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from can_fusion import (
    CanFusion,
    CanFusionConfig,
    PRIMARY_CAN_IDS,
    bbox_iou,
    same_physical_box,
)


CAN_CLASS_NAMES = [
    "2-Starkist_Tuna",
    "5-Tomato_Soup",
    "10-meat_can",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-config", required=True, type=Path)
    parser.add_argument("--secondary-weights", required=True, type=Path)
    parser.add_argument("--dataset-yaml", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"not a YAML mapping: {path}")
    return value


def dataset_paths(dataset_yaml: Path, split: str) -> tuple[Path, Path, list[str]]:
    data = load_yaml(dataset_yaml)
    root = Path(str(data.get("path", dataset_yaml.parent)))
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    split_value = str(data[split])
    image_dir = root / split_value
    if not image_dir.is_dir():
        image_dir = dataset_yaml.parent / split_value
    label_dir = image_dir.parent.parent / "labels" / image_dir.name
    if not label_dir.is_dir():
        label_dir = dataset_yaml.parent / "labels" / split
    raw_names = data.get("names")
    if isinstance(raw_names, dict):
        names = [str(raw_names[i]) for i in sorted(raw_names)]
    elif isinstance(raw_names, list):
        names = [str(v) for v in raw_names]
    else:
        raise ValueError(f"dataset names missing: {dataset_yaml}")
    return image_dir.resolve(), label_dir.resolve(), names


def image_files(directory: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes
    )


def label_path(label_dir: Path, image: Path) -> Path:
    return label_dir / f"{image.stem}.txt"


def read_gt(
    path: Path,
    dataset_names: Sequence[str],
    primary_names: Sequence[str],
    frame_shape: Sequence[int],
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    name_to_primary = {name: index for index, name in enumerate(primary_names)}
    class_map: dict[int, int] = {}
    for class_id, name in enumerate(dataset_names):
        if name not in name_to_primary:
            raise ValueError(f"ground-truth class not in primary model: {class_id}:{name}")
        class_map[class_id] = name_to_primary[name]
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"invalid label {path}:{line_number}")
        source_id = int(float(fields[0]))
        cx, cy, bw, bh = (float(v) for v in fields[1:5])
        class_id = class_map[source_id]
        result.append(
            {
                "id": class_id,
                "officialId": class_id + 1,
                "name": primary_names[class_id],
                "bbox": [
                    (cx - bw / 2.0) * width,
                    (cy - bh / 2.0) * height,
                    (cx + bw / 2.0) * width,
                    (cy + bh / 2.0) * height,
                ],
            }
        )
    return result


def predict(model: YOLO, frame: np.ndarray, args: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = model.predict(
        source=frame,
        conf=float(args["conf"]),
        iou=float(args["iou"]),
        imgsz=int(args["imgsz"]),
        device=str(args["device"]),
        max_det=int(args["max_det"]),
        verbose=False,
    )
    output: list[dict[str, Any]] = []
    if not results:
        return output
    boxes = results[0].boxes
    if boxes is None:
        return output
    for box, score, class_id in zip(
        boxes.xyxy.detach().cpu().numpy(),
        boxes.conf.detach().cpu().numpy(),
        boxes.cls.detach().cpu().numpy().astype(int),
    ):
        output.append(
            {
                "class_id": int(class_id),
                "confidence": float(score),
                "bbox": [float(v) for v in box],
            }
        )
    return output


def canonical_counts(items: Sequence[Mapping[str, Any]], ids: set[int]) -> tuple[int, ...]:
    counts = Counter(int(item["id"]) for item in items if int(item["id"]) in ids)
    return tuple(counts[class_id] for class_id in sorted(ids))


def match_detections(
    predictions: Sequence[Mapping[str, Any]],
    ground_truths: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
    class_aware: bool = True,
) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    unmatched_gt = set(range(len(ground_truths)))
    matched_pred: list[int] = []
    unmatched_pred: list[int] = []
    tp: list[tuple[int, int]] = []
    wrong: list[tuple[int, int]] = []
    for pred_index in sorted(
        range(len(predictions)),
        key=lambda i: -float(predictions[i].get("confidence", 0.0)),
    ):
        pred = predictions[pred_index]
        best_index: int | None = None
        best_iou = -1.0
        for gt_index in unmatched_gt:
            gt = ground_truths[gt_index]
            if class_aware and int(pred["id"]) != int(gt["id"]):
                continue
            overlap = bbox_iou(pred["bbox"], gt["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        if best_index is not None and best_iou >= iou_threshold:
            unmatched_gt.remove(best_index)
            matched_pred.append(pred_index)
            tp.append((pred_index, best_index))
        else:
            unmatched_pred.append(pred_index)

    # For output diagnostics, locate geometrically matching GT for each
    # unmatched prediction even when the predicted class is wrong.
    for pred_index in unmatched_pred:
        best_index: int | None = None
        best_iou = -1.0
        for gt_index, gt in enumerate(ground_truths):
            overlap = bbox_iou(predictions[pred_index]["bbox"], gt["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        if best_index is not None and best_iou >= iou_threshold:
            wrong.append((pred_index, best_index))
    return matched_pred, sorted(unmatched_gt), tp, wrong


def evaluate_threshold(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    threshold_config: CanFusionConfig,
    primary_names: Sequence[str],
) -> dict[str, Any]:
    totals = Counter()
    exact_images = 0
    can_images = 0
    negative_images = 0
    negative_false_images = 0
    mismatches: list[dict[str, Any]] = []
    collateral: list[dict[str, Any]] = []
    confusion: Counter[tuple[int, int]] = Counter()
    unresolved_confusers: Counter[int] = Counter()
    unresolved_examples: list[dict[str, Any]] = []
    for row in rows:
        gt_cans = [item for item in row["gt"] if int(item["id"]) in PRIMARY_CAN_IDS]
        gt_noncan = [item for item in row["gt"] if int(item["id"]) not in PRIMARY_CAN_IDS]
        fusion = CanFusion(primary_names, threshold_config)
        fused, debug = fusion.fuse(
            row["shape"],
            row["primary"],
            row["secondary"],
            final_conf=float(row["final_conf"]),
            use_temporal=False,
        )
        pred_cans = [item for item in fused if int(item["id"]) in PRIMARY_CAN_IDS]
        pred_noncan = [item for item in fused if int(item["id"]) not in PRIMARY_CAN_IDS]
        gt_can_counts = canonical_counts(gt_cans, set(PRIMARY_CAN_IDS))
        pred_can_counts = canonical_counts(pred_cans, set(PRIMARY_CAN_IDS))
        exact = gt_can_counts == pred_can_counts
        exact_images += int(exact)
        can_images += int(bool(gt_cans))
        if not gt_cans:
            negative_images += 1
            negative_false_images += int(bool(pred_cans))

        for iou_threshold, label in ((0.30, "iou30"), (0.50, "iou50")):
            matched, missed, _, wrong = match_detections(
                pred_cans,
                gt_cans,
                iou_threshold=iou_threshold,
                class_aware=True,
            )
            totals[f"{label}_tp"] += len(matched)
            totals[f"{label}_fn"] += len(missed)
            totals[f"{label}_fp"] += len(pred_cans) - len(matched)
            totals[f"{label}_wrong_class"] += len(wrong)
            if iou_threshold == 0.30:
                for pred_index, gt_index in wrong:
                    confusion[(int(pred_cans[pred_index]["id"]), int(gt_cans[gt_index]["id"]))] += 1

        matched_noncan, missed_noncan, _, _ = match_detections(
            pred_noncan,
            gt_noncan,
            iou_threshold=0.50,
            class_aware=True,
        )
        totals["noncan_gt"] += len(gt_noncan)
        totals["noncan_tp"] += len(matched_noncan)
        totals["noncan_fn"] += len(missed_noncan)
        if missed_noncan:
            collateral.append(
                {
                    "image": row["image"],
                    "missed": [gt_noncan[i] for i in missed_noncan],
                }
            )

        # Count confirmed non-can duplicate boxes still left on can pixels.
        # A prediction that matches a real non-can ground truth is not counted.
        for pred in pred_noncan:
            matches_real_noncan = any(
                int(pred["id"]) == int(gt["id"])
                and bbox_iou(pred["bbox"], gt["bbox"]) >= 0.50
                for gt in gt_noncan
            )
            if matches_real_noncan:
                continue
            if any(
                same_physical_box(
                    pred["bbox"],
                    gt["bbox"],
                    min_iou=0.30,
                    min_ios=0.70,
                )
                for gt in gt_cans
            ):
                unresolved_confusers[int(pred["id"])] += 1
                if len(unresolved_examples) < 30:
                    unresolved_examples.append(
                        {
                            "image": row["image"],
                            "prediction": pred,
                            "canGroundTruths": gt_cans,
                        }
                    )
        if not exact:
            mismatches.append(
                {
                    "image": row["image"],
                    "gt": [int(v["id"]) for v in gt_cans],
                    "pred": [int(v["id"]) for v in pred_cans],
                    "debug": {
                        "selected": debug["selectedCans"],
                        "removedPrimaryCans": debug["removedPrimaryCans"],
                        "suppressedConfusers": debug["suppressedConfusers"],
                        "rejectedSecondary": debug["rejectedSecondary"],
                    },
                }
            )

    return {
        "secondaryConf": threshold,
        "thresholds": {
            "secondary_conf": threshold_config.secondary_conf,
            "strong_conf": threshold_config.strong_conf,
            "base_fallback_conf": threshold_config.base_fallback_conf,
        },
        "images": len(rows),
        "canImages": can_images,
        "exactCanCountImages": exact_images,
        "exactCanCountRate": exact_images / len(rows) if rows else 0.0,
        "negativeImages": negative_images,
        "negativeImagesWithFalseCan": negative_false_images,
        "canPrecisionIoU30": totals["iou30_tp"]
        / max(1, totals["iou30_tp"] + totals["iou30_fp"]),
        "canRecallIoU30": totals["iou30_tp"]
        / max(1, totals["iou30_tp"] + totals["iou30_fn"]),
        "canPrecisionIoU50": totals["iou50_tp"]
        / max(1, totals["iou50_tp"] + totals["iou50_fp"]),
        "canRecallIoU50": totals["iou50_tp"]
        / max(1, totals["iou50_tp"] + totals["iou50_fn"]),
        "canWrongClassIoU30": totals["iou30_wrong_class"],
        "nonCanRecallIoU50": totals["noncan_tp"] / max(1, totals["noncan_gt"]),
        "confusionPredToGt": {
            f"{primary_names[pred]}->{primary_names[gt]}": count
            for (pred, gt), count in sorted(confusion.items())
        },
        "unresolvedNonCanOnCan": {
            primary_names[class_id]: count
            for class_id, count in sorted(unresolved_confusers.items())
        },
        "unresolvedNonCanOnCanTotal": sum(unresolved_confusers.values()),
        "unresolvedNonCanOnCanExamples": unresolved_examples,
        "mismatches": mismatches,
        "collateralNonCanMisses": collateral,
        "totals": dict(totals),
    }



def main() -> int:
    args = parse_args()
    started = time.monotonic()
    primary_config = load_yaml(args.primary_config)
    primary_model_cfg = primary_config["model"]
    primary_names = [str(v) for v in primary_model_cfg["class_names"]]
    device = str(args.device or primary_model_cfg.get("device", "cpu"))
    primary_weights = Path(str(primary_model_cfg["weights"]))
    secondary_weights = args.secondary_weights.resolve()
    if not primary_weights.is_file() or not secondary_weights.is_file():
        raise FileNotFoundError("primary or secondary model weights are missing")

    image_dir, label_dir, dataset_names = dataset_paths(args.dataset_yaml, args.split)
    images = image_files(image_dir)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise ValueError(f"no images in {image_dir}")

    secondary_model = YOLO(str(secondary_weights))
    secondary_names = [
        str(secondary_model.names[i]) for i in sorted(secondary_model.names)
    ]
    if secondary_names != CAN_CLASS_NAMES:
        raise ValueError(f"secondary model classes mismatch: {secondary_names}")

    primary_model = YOLO(str(primary_weights))
    primary_predict_args = {
        "conf": float(primary_config.get("can_postprocess", {}).get("inference_conf", 0.005)),
        "iou": float(primary_model_cfg.get("iou", 0.60)),
        "imgsz": int(primary_model_cfg.get("imgsz", 640)),
        "device": device,
        "max_det": int(primary_model_cfg.get("max_det", 100)),
    }
    secondary_predict_args = {
        "conf": 0.01,
        "iou": float(primary_config.get("can_secondary", {}).get("iou", 0.50)),
        "imgsz": int(primary_config.get("can_secondary", {}).get("imgsz", 640)),
        "device": device,
        "max_det": int(primary_config.get("can_secondary", {}).get("max_det", 20)),
    }
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, 1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"cannot read {image_path}")
        gt = read_gt(label_path(label_dir, image_path), dataset_names, primary_names, frame.shape)
        rows.append(
            {
                "image": image_path.name,
                "shape": list(frame.shape),
                "gt": gt,
                "primary": predict(primary_model, frame, primary_predict_args),
                "secondary": predict(secondary_model, frame, secondary_predict_args),
                "final_conf": float(primary_model_cfg.get("default_conf", 0.15)),
            }
        )
        if index % 50 == 0 or index == len(images):
            print(f"inference {index}/{len(images)}", file=sys.stderr, flush=True)

    base_config = CanFusionConfig.from_mapping(primary_config.get("can_secondary", {}))
    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]
    evaluations = []
    for threshold in thresholds:
        cfg = CanFusionConfig(
            enabled=base_config.enabled,
            secondary_conf=threshold,
            # Threshold scanning changes only the acceptance threshold.  Keep
            # the stronger confidence gate fixed so confuser suppression is
            # evaluated with the production policy rather than becoming more
            # aggressive at low-threshold candidates.
            strong_conf=base_config.strong_conf,
            base_fallback_conf=base_config.base_fallback_conf,
            match_iou=base_config.match_iou,
            match_ios=base_config.match_ios,
            suppress_iou=base_config.suppress_iou,
            suppress_ios=base_config.suppress_ios,
            min_box_area_ratio=base_config.min_box_area_ratio,
            min_box_side=base_config.min_box_side,
            max_box_aspect=base_config.max_box_aspect,
            max_det_per_class=base_config.max_det_per_class,
            temporal_min_hits=1,
            temporal_hold_inferences=0,
            temporal_match_iou=base_config.temporal_match_iou,
            confusers_by_can_id=base_config.confusers_by_can_id,
        )
        evaluations.append(evaluate_threshold(rows, threshold, cfg, primary_names))

    ranked = sorted(
        evaluations,
        key=lambda item: (
            -item["exactCanCountRate"],
            -item["canRecallIoU50"],
            item["canWrongClassIoU30"],
            item["unresolvedNonCanOnCanTotal"],
            item["negativeImagesWithFalseCan"],
            -item["nonCanRecallIoU50"],
        ),
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "primaryConfig": str(args.primary_config.resolve()),
        "primaryWeights": str(primary_weights.resolve()),
        "secondaryWeights": str(secondary_weights),
        "datasetYaml": str(args.dataset_yaml.resolve()),
        "split": args.split,
        "imageDir": str(image_dir),
        "labelDir": str(label_dir),
        "imageCount": len(rows),
        "device": device,
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "recommendedSecondaryConf": ranked[0]["secondaryConf"] if ranked else None,
        "evaluations": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = [
        {
            key: evaluation[key]
            for key in (
                "secondaryConf",
                "exactCanCountRate",
                "canPrecisionIoU50",
                "canRecallIoU50",
                "canWrongClassIoU30",
                "unresolvedNonCanOnCanTotal",
                "negativeImagesWithFalseCan",
                "nonCanRecallIoU50",
            )
        }
        for evaluation in evaluations
    ]
    print(json.dumps({"recommended": ranked[0]["secondaryConf"] if ranked else None, "summary": summary}, ensure_ascii=False, indent=2))
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
