#!/usr/bin/env python3
"""Repeatable regression audit for can object-level post-processing.

The audit runs the production model at the production low inference threshold,
then compares the normal confidence-filtered model output with the post-processed
output. It is intentionally report-only by default: model errors are reported as
model errors instead of making the post-processor look successful by assumption.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from can_postprocess import CAN_IDS, CanPostProcessor


FRONT_TEN_IDS = set(range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None, help="Override model.device, e.g. cpu or 0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"配置不是 YAML 对象：{path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def label_path_for(image_path: Path) -> Path:
    return image_path.parent.parent.parent / "labels" / "val" / f"{image_path.stem}.txt"


def read_ground_truth(path: Path) -> Counter[int]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少验证集标签：{path}")
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"无效标签行 {path}:{line_number}: {line!r}")
        counts[int(fields[0])] += 1
    return counts


def detection_counts(detections: Sequence[dict[str, Any]], ids: set[int] | None = None) -> Counter[int]:
    return Counter(
        int(item["id"])
        for item in detections
        if ids is None or int(item["id"]) in ids
    )


def without_ids(counts: Counter[int], excluded: set[int]) -> Counter[int]:
    return Counter(
        {class_id: count for class_id, count in counts.items() if class_id not in excluded}
    )


def counter_delta(actual: Counter[int], expected: Counter[int]) -> dict[str, dict[str, int]]:
    missing = expected - actual
    extra = actual - expected
    return {
        "missing": {str(key): value for key, value in sorted(missing.items())},
        "extra": {str(key): value for key, value in sorted(extra.items())},
    }


def listed_counter(value: Counter[int]) -> list[int]:
    result: list[int] = []
    for class_id in sorted(value):
        result.extend([class_id] * value[class_id])
    return result


def predict_chunk(
    model: YOLO,
    frames: Sequence[np.ndarray],
    *,
    names: Sequence[str],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    max_det: int,
) -> list[list[dict[str, Any]]]:
    results = model.predict(
        source=list(frames),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        verbose=False,
        batch=len(frames),
    )
    chunks: list[list[dict[str, Any]]] = []
    for result in results:
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None:
            chunks.append(detections)
            continue
        for bbox, score, class_id in zip(
            boxes.xyxy.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
            boxes.cls.detach().cpu().numpy().astype(int),
        ):
            class_id = int(class_id)
            if not 0 <= class_id < len(names):
                continue
            detections.append(
                {
                    "id": class_id,
                    "officialId": class_id + 1,
                    "name": names[class_id],
                    "confidence": round(float(score), 4),
                    "bbox": [round(float(value), 1) for value in bbox],
                }
            )
        chunks.append(detections)
    return chunks


def compact_can_debug(debug: dict[str, Any]) -> dict[str, Any]:
    return {
        "selectedCans": debug.get("selectedCans", []),
        "suppressed": debug.get("suppressed", []),
        "tomatoRedRatio": debug.get("tomatoRedRatio"),
        "tomatoTunaOverlap": debug.get("tomatoTunaOverlap"),
        "tomatoDecision": debug.get("tomatoDecision"),
        "rejectedImplausibleCanCandidates": debug.get(
            "rejectedImplausibleCanCandidates", []
        ),
        "rejectedCansByStrongerConfuser": debug.get(
            "rejectedCansByStrongerConfuser", []
        ),
        "goldFallbackActive": debug.get("goldFallbackActive", False),
        "goldFallback": debug.get("goldFallback"),
        "goldRejected": debug.get("goldRejected", []),
    }


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    config = load_config(args.config)
    model_config = config["model"]
    names = list(model_config["class_names"])
    weights = Path(model_config["weights"])
    expected_hash = str(model_config.get("sha256", "")).strip().lower()
    actual_hash = sha256(weights)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"模型 SHA256 不匹配：期望 {expected_hash}，实际 {actual_hash}")

    images = image_files(args.images)
    if not images:
        raise ValueError(f"验证集目录没有图片：{args.images}")

    device = str(args.device if args.device is not None else model_config.get("device", "cpu"))
    final_conf = float(model_config["default_conf"])
    inference_conf = float(config.get("can_postprocess", {}).get("inference_conf", 0.005))
    postprocess_config = dict(config.get("can_postprocess", {}))
    # This audit checks one image at a time. Temporal confirmation is deliberately
    # disabled so a failure cannot be hidden by a previously held track.
    postprocess_config["temporal_min_hits"] = 1
    postprocess_config["temporal_hold_inferences"] = 0

    model = YOLO(str(weights))
    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch))

    for offset in range(0, len(images), batch_size):
        batch_paths = images[offset : offset + batch_size]
        frames: list[np.ndarray] = []
        ground_truths: list[Counter[int]] = []
        for path in batch_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"无法读取图片：{path}")
            frames.append(frame)
            ground_truths.append(read_ground_truth(label_path_for(path)))

        raw_batches = predict_chunk(
            model,
            frames,
            names=names,
            conf=inference_conf,
            iou=float(model_config.get("iou", 0.60)),
            imgsz=int(model_config.get("imgsz", 640)),
            device=device,
            max_det=int(model_config.get("max_det", 100)),
        )

        for path, frame, ground_truth, raw in zip(
            batch_paths, frames, ground_truths, raw_batches
        ):
            processor = CanPostProcessor(names, postprocess_config)
            processed, debug = processor.process(frame, raw, final_conf=final_conf)
            raw_filtered = [
                item for item in raw if float(item["confidence"]) >= final_conf
            ]

            gt_counts = ground_truth
            raw_counts = detection_counts(raw_filtered)
            post_counts = detection_counts(processed)
            gt_cans = gt_counts & Counter({class_id: 10**9 for class_id in CAN_IDS})
            raw_cans = raw_counts & Counter({class_id: 10**9 for class_id in CAN_IDS})
            post_cans = post_counts & Counter({class_id: 10**9 for class_id in CAN_IDS})
            gt_front = gt_counts & Counter({class_id: 10**9 for class_id in FRONT_TEN_IDS})
            raw_front = raw_counts & Counter(
                {class_id: 10**9 for class_id in FRONT_TEN_IDS}
            )
            post_front = post_counts & Counter(
                {class_id: 10**9 for class_id in FRONT_TEN_IDS}
            )

            noncan_ids = set(range(len(names))) - CAN_IDS
            gt_noncan = gt_counts & Counter({class_id: 10**9 for class_id in noncan_ids})
            raw_noncan = raw_counts & Counter({class_id: 10**9 for class_id in noncan_ids})
            post_noncan = post_counts & Counter(
                {class_id: 10**9 for class_id in noncan_ids}
            )

            target_exact = post_cans == gt_cans
            target_changed = post_cans != raw_cans
            gold_fallback = debug.get("goldFallback")
            false_gold_fallback = bool(
                debug.get("goldFallbackActive") and gt_counts.get(9, 0) == 0
            )
            lost_noncan_gt = (gt_noncan & raw_noncan) - post_noncan
            lost_front10_gt = (gt_front & raw_front) - post_front
            changed_noncan_front10 = without_ids(raw_front, CAN_IDS) != without_ids(
                post_front, CAN_IDS
            )

            row = {
                "image": path.name,
                "label": str(label_path_for(path).resolve()),
                "gt": listed_counter(gt_counts),
                "raw": listed_counter(raw_counts),
                "post": listed_counter(post_counts),
                "gtCans": listed_counter(gt_cans),
                "rawCans": listed_counter(raw_cans),
                "postCans": listed_counter(post_cans),
                "targetExact": target_exact,
                "targetDelta": counter_delta(post_cans, gt_cans),
                "targetChangedByPostprocess": target_changed,
                "modelRawTargetExact": raw_cans == gt_cans,
                "fixedByPostprocess": raw_cans != gt_cans and target_exact,
                "regressedByPostprocess": raw_cans == gt_cans and not target_exact,
                "lostNonCanGroundTruth": listed_counter(lost_noncan_gt),
                "lostFrontTenGroundTruth": listed_counter(lost_front10_gt),
                "changedNonCanFrontTen": changed_noncan_front10,
                "falseGoldFallback": false_gold_fallback,
                "goldFallbackBox": gold_fallback.get("bbox") if gold_fallback else None,
                "debug": compact_can_debug(debug),
            }
            rows.append(row)

        completed = min(offset + len(batch_paths), len(images))
        print(f"audit {completed}/{len(images)}", file=sys.stderr, flush=True)

    fixed = [row["image"] for row in rows if row["fixedByPostprocess"]]
    regressed = [row["image"] for row in rows if row["regressedByPostprocess"]]
    persistent = [
        row["image"]
        for row in rows
        if row["targetChangedByPostprocess"]
        and not row["targetExact"]
        and not row["regressedByPostprocess"]
    ]
    false_gold = [row["image"] for row in rows if row["falseGoldFallback"]]
    missed_gold_gt = [
        row["image"]
        for row in rows
        if row["gtCans"].count(9) > 0 and row["postCans"].count(9) == 0
    ]
    lost_noncan = {
        row["image"]: row["lostNonCanGroundTruth"]
        for row in rows
        if row["lostNonCanGroundTruth"]
    }
    lost_front10 = {
        row["image"]: row["lostFrontTenGroundTruth"]
        for row in rows
        if row["lostFrontTenGroundTruth"]
    }
    changed_front10 = [
        row["image"] for row in rows if row["changedNonCanFrontTen"]
    ]

    report = {
        "config": str(args.config.resolve()),
        "imagesDirectory": str(args.images.resolve()),
        "weights": str(weights.resolve()),
        "weightsSha256": actual_hash,
        "device": device,
        "inferenceConf": inference_conf,
        "finalConf": final_conf,
        "imageCount": len(rows),
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "summary": {
            "rawTargetMismatchImages": sum(
                not row["modelRawTargetExact"] for row in rows
            ),
            "postTargetMismatchImages": sum(not row["targetExact"] for row in rows),
            "fixedByPostprocess": fixed,
            "regressedByPostprocess": regressed,
            "persistentTargetMismatch": persistent,
            "falseGoldFallback": false_gold,
            "missedGoldGroundTruth": missed_gold_gt,
            "lostNonCanGroundTruth": lost_noncan,
            "lostFrontTenGroundTruth": lost_front10,
            "changedNonCanFrontTen": changed_front10,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={args.output}")

    regression_exists = bool(
        regressed
        or false_gold
        or lost_noncan
        or lost_front10
        or changed_front10
    )
    if args.fail_on_regression and regression_exists:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
