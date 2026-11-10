#!/usr/bin/env python3
"""Offline acceptance for v11 single-scale and 640+800 view fusion.

The script is read-only with respect to datasets, weights, deployment configs,
Task A and Task B. It caches raw predictions, evaluates AP on the selected
validation split, and writes per-image miss/false-positive diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path('/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序')
RUN_DIR = PROJECT_ROOT / 'runs/track1_train_40class_v11_full_v10aug_from_yolo11s/track1_40class_v11_20261031_234506'
DEFAULT_MODEL = RUN_DIR / 'weights/best.pt'
DEFAULT_DATASET = RUN_DIR / 'post_train_audit/multiobject_val_data/dataset.yaml'
DEFAULT_OUTPUT = PROJECT_ROOT / 'reports/track1_v11_multiscale_fusion_exact_20261101'
DEFAULT_CACHE = PROJECT_ROOT / 'cache/track1_v11_multiscale_fusion_exact_20261101'
OFFICIAL_EXPECTED = {
    71: {'map50': 0.8875452967171717, 'map50_95': 0.5764002732013859},
}
CAPTURE_PROTOCOL_VERSION = 2
DEFAULT_REVIEW = PROJECT_ROOT / 'reports/track1_v11_offline_acceptance_20261101/manual_review_multiobject.json'
IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.webp'}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def dataset_signature(image_paths: Sequence[Path], dataset_yaml: Path) -> str:
    """Fingerprint the exact images and labels used by a capture run."""
    digest = hashlib.sha256()
    digest.update(str(dataset_yaml.resolve()).encode('utf-8'))
    for image in sorted(image_paths, key=lambda p: p.name):
        resolved_image = image.resolve()
        label = image_label_path(image, dataset_yaml.parent).resolve()
        stat = resolved_image.stat()
        digest.update(f'\0{image.name}\0{resolved_image}\0{stat.st_size}'.encode('utf-8'))
        if label.is_file():
            label_stat = label.stat()
            digest.update(f'\0{label}\0{label_stat.st_size}'.encode('utf-8'))
    return digest.hexdigest()


def serializable_ratio_pad(value: Any) -> list[list[float]]:
    return [[float(x) for x in row] for row in value]


def validation_box_to_original(box: Sequence[float], meta: dict[str, Any]) -> list[float]:
    gain_y, gain_x = meta['ratio_pad'][0]
    pad_x, pad_y = meta['ratio_pad'][1]
    width = float(meta['original_width'])
    height = float(meta['original_height'])
    x1 = min(width, max(0.0, (float(box[0]) - pad_x) / gain_x))
    y1 = min(height, max(0.0, (float(box[1]) - pad_y) / gain_y))
    x2 = min(width, max(0.0, (float(box[2]) - pad_x) / gain_x))
    y2 = min(height, max(0.0, (float(box[3]) - pad_y) / gain_y))
    return [x1, y1, x2, y2]


def original_box_to_validation(box: Sequence[float], meta: dict[str, Any]) -> list[float]:
    gain_y, gain_x = meta['ratio_pad'][0]
    pad_x, pad_y = meta['ratio_pad'][1]
    input_height, input_width = meta['input_shape']
    x1 = min(float(input_width), max(0.0, float(box[0]) * gain_x + pad_x))
    y1 = min(float(input_height), max(0.0, float(box[1]) * gain_y + pad_y))
    x2 = min(float(input_width), max(0.0, float(box[2]) * gain_x + pad_x))
    y2 = min(float(input_height), max(0.0, float(box[3]) * gain_y + pad_y))
    return [x1, y1, x2, y2]



def resolve_dataset(dataset_yaml: Path) -> tuple[Path, Path, list[str]]:
    payload = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    root = Path(str(payload.get('path', dataset_yaml.parent))).expanduser().resolve()
    split = payload.get('val')
    if isinstance(split, list):
        val_paths = [(root / str(item)).resolve() for item in split]
    elif isinstance(split, str):
        val_paths = [(root / split).resolve()]
    else:
        raise ValueError(f'dataset val 字段非法: {type(split).__name__}')
    images: list[Path] = []
    for path in val_paths:
        if path.is_file() and path.suffix.lower() == '.txt':
            images.extend(Path(line.strip()).resolve() for line in path.read_text().splitlines() if line.strip())
        elif path.is_dir():
            images.extend(sorted(p.resolve() for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            raise FileNotFoundError(f'验证集路径不存在: {path}')
    names_raw = payload.get('names')
    if isinstance(names_raw, dict):
        names = [str(names_raw[i]) for i in range(len(names_raw))]
    elif isinstance(names_raw, list):
        names = [str(x) for x in names_raw]
    else:
        raise ValueError('dataset names 字段缺失或非法')
    return root, val_paths[0], names


def image_label_path(image: Path, dataset_root: Path) -> Path:
    parts = list(image.parts)
    try:
        images_index = len(parts) - 1 - parts[::-1].index('images')
    except ValueError as exc:
        raise ValueError(f'图片路径不含 images 目录: {image}') from exc
    parts[images_index] = 'labels'
    return Path(*parts).with_suffix('.txt')


def read_ground_truth(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not label_path.is_file():
        return rows
    for line in label_path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cls = int(float(fields[0]))
        cx, cy, bw, bh = (float(value) for value in fields[1:5])
        x1 = max(0.0, (cx - bw / 2) * width)
        y1 = max(0.0, (cy - bh / 2) * height)
        x2 = min(float(width), (cx + bw / 2) * width)
        y2 = min(float(height), (cy + bh / 2) * height)
        rows.append({'class_id': cls, 'bbox': [x1, y1, x2, y2]})
    return rows


def box_iou_one(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    ab = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = aa + ab - inter
    return inter / denom if denom > 0 else 0.0


def box_iou_matrix(gt_boxes: Sequence[Sequence[float]], pred_boxes: Sequence[Sequence[float]]) -> np.ndarray:
    if not gt_boxes or not pred_boxes:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    g = np.asarray(gt_boxes, dtype=np.float32)
    p = np.asarray(pred_boxes, dtype=np.float32)
    ix1 = np.maximum(g[:, None, 0], p[None, :, 0])
    iy1 = np.maximum(g[:, None, 1], p[None, :, 1])
    ix2 = np.minimum(g[:, None, 2], p[None, :, 2])
    iy2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    ga = np.maximum(0.0, g[:, 2] - g[:, 0]) * np.maximum(0.0, g[:, 3] - g[:, 1])
    pa = np.maximum(0.0, p[:, 2] - p[:, 0]) * np.maximum(0.0, p[:, 3] - p[:, 1])
    denom = ga[:, None] + pa[None, :] - inter
    return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)


def match_image(
    gt: Sequence[dict[str, Any]], preds: Sequence[dict[str, Any]], iou_thresholds: Iterable[float] = IOU_THRESHOLDS,
) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    """Match predictions with the same matching rule used by Ultralytics validation.

    The highest-IoU prediction/GT pairs are made unique independently at each IoU
    threshold. This intentionally mirrors the existing v11 post-train metrics so
    single-scale and fused results remain directly comparable.
    """
    thresholds = list(float(x) for x in iou_thresholds)
    tp = np.zeros((len(preds), len(thresholds)), dtype=bool)
    pred_to_gt: list[list[int]] = [[-1] * len(thresholds) for _ in preds]
    gt_to_pred: list[list[int]] = [[-1] * len(thresholds) for _ in gt]
    gt_boxes = [row['bbox'] for row in gt]
    pred_boxes = [row['bbox'] for row in preds]
    iou = box_iou_matrix(gt_boxes, pred_boxes)
    gt_classes = np.asarray([int(row['class_id']) for row in gt], dtype=np.int64)
    pred_classes = np.asarray([int(row['class_id']) for row in preds], dtype=np.int64)
    if len(gt_classes) and len(pred_classes):
        iou = iou * (gt_classes[:, None] == pred_classes[None, :])
    for ti, threshold in enumerate(thresholds):
        matches = np.argwhere(iou >= threshold)
        if len(matches):
            if len(matches) > 1:
                matches = matches[np.argsort(iou[matches[:, 0], matches[:, 1]])[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            for gi, pi in matches:
                gi_i, pi_i = int(gi), int(pi)
                tp[pi_i, ti] = True
                pred_to_gt[pi_i][ti] = gi_i
                gt_to_pred[gi_i][ti] = pi_i
    return tp, pred_to_gt, gt_to_pred


def evaluation_stats(records: Sequence[dict[str, Any]], pred_key: str = 'predictions') -> dict[str, Any]:
    tp_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    pred_cls_parts: list[np.ndarray] = []
    target_cls_parts: list[np.ndarray] = []
    for record in records:
        gt = record['ground_truth']
        preds = record[pred_key]
        tp, _, _ = match_image(gt, preds)
        tp_parts.append(tp)
        conf_parts.append(np.asarray([float(p['confidence']) for p in preds], dtype=np.float32))
        pred_cls_parts.append(np.asarray([int(p['class_id']) for p in preds], dtype=np.float32))
        target_cls_parts.append(np.asarray([int(g['class_id']) for g in gt], dtype=np.float32))
    tp = np.concatenate(tp_parts, axis=0) if tp_parts else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(conf_parts) if conf_parts else np.zeros(0, dtype=np.float32)
    pred_cls = np.concatenate(pred_cls_parts) if pred_cls_parts else np.zeros(0, dtype=np.float32)
    target_cls = np.concatenate(target_cls_parts) if target_cls_parts else np.zeros(0, dtype=np.float32)
    if len(conf) == 0:
        return {'map50': 0.0, 'map50_95': 0.0, 'per_class': {}, 'counts': {'predictions': 0, 'targets': len(target_cls)}}

    from ultralytics.utils.metrics import ap_per_class

    _, _, p, r, _, ap, unique, _, _, _, _, _ = ap_per_class(
        tp, conf, pred_cls, target_cls, plot=False, names={i: str(i) for i in range(int(max(pred_cls.max(initial=0), target_cls.max(initial=0))) + 1)},
    )
    per_class: dict[str, dict[str, float]] = {}
    for idx, cls in enumerate(unique.tolist()):
        per_class[str(int(cls))] = {
            'precision': float(p[idx]),
            'recall': float(r[idx]),
            'ap50': float(ap[idx, 0]),
            'ap50_95': float(ap[idx].mean()),
        }
    return {
        'map50': float(ap[:, 0].mean()) if len(ap) else 0.0,
        'map50_95': float(ap.mean()) if len(ap) else 0.0,
        'per_class': per_class,
        'counts': {'predictions': int(len(conf)), 'targets': int(len(target_cls))},
    }


def diagnostics(
    records: Sequence[dict[str, Any]], names: Sequence[str], pred_key: str, conf_threshold: float,
    focus_files: set[str],
) -> dict[str, Any]:
    totals = Counter()
    focus: dict[str, Any] = {}
    for record in records:
        preds = [dict(p) for p in record[pred_key] if float(p['confidence']) >= conf_threshold]
        _, _, gt_to_pred = match_image(record['ground_truth'], preds, iou_thresholds=[0.5])
        tp, _, _ = match_image(record['ground_truth'], preds, iou_thresholds=[0.5])
        matched_preds = {m[0] for m in gt_to_pred if m[0] >= 0}
        missing = [
            {'class_id': int(g['class_id']), 'class_name': names[int(g['class_id'])], 'bbox': g['bbox']}
            for gi, g in enumerate(record['ground_truth']) if gt_to_pred[gi][0] < 0
        ]
        extra = [
            {
                'class_id': int(p['class_id']), 'class_name': names[int(p['class_id'])],
                'confidence': float(p['confidence']), 'bbox': p['bbox'],
            }
            for pi, p in enumerate(preds) if pi not in matched_preds and not tp[pi, 0]
        ]
        totals['targets'] += len(record['ground_truth'])
        totals['predictions'] += len(preds)
        totals['true_positives'] += len(preds) - len(extra)
        totals['false_negatives'] += len(missing)
        totals['false_positives'] += len(extra)
        if record['file_name'] in focus_files:
            focus[record['file_name']] = {
                'missing': missing,
                'extra': extra,
                'prediction_count': len(preds),
                'gt_count': len(record['ground_truth']),
            }
    if totals['true_positives'] + totals['false_positives'] > 0:
        precision = totals['true_positives'] / (totals['true_positives'] + totals['false_positives'])
    else:
        precision = 0.0
    if totals['true_positives'] + totals['false_negatives'] > 0:
        recall = totals['true_positives'] / (totals['true_positives'] + totals['false_negatives'])
    else:
        recall = 0.0
    return {
        'confidence_threshold': conf_threshold,
        'iou_threshold': 0.5,
        'totals': {**dict(totals), 'precision': precision, 'recall': recall},
        'focus_images': focus,
    }


def load_or_capture(
    image_paths: Sequence[Path], cache_path: Path, model_path: Path, dataset_yaml: Path,
    imgsz: int, device: str, batch: int, conf: float, iou: float, max_det: int,
    force: bool, names: Sequence[str],
) -> dict[str, Any]:
    """Capture the exact predictions used by Ultralytics' official validator.

    The previous implementation used ``model.predict``. That omits the validator's
    ``rect=True`` input geometry and ``multi_label=True`` NMS, so its AP was not
    comparable with the official post-training result. This function captures the
    tensors inside ``DetectionValidator._process_batch`` instead.
    """
    from pathlib import Path as RuntimePath

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator

    model_sha = sha256_file(model_path)
    signature = dataset_signature(image_paths, dataset_yaml)
    metadata = {
        'created_at': now_iso(), 'protocol_version': CAPTURE_PROTOCOL_VERSION,
        'capture_source': 'ultralytics.DetectionValidator._process_batch',
        'rect': True, 'multi_label': True,
        'model': str(model_path), 'model_sha256': model_sha,
        'dataset_yaml': str(dataset_yaml), 'dataset_sha256': sha256_file(dataset_yaml),
        'dataset_signature': signature, 'imgsz': imgsz,
        'conf': conf, 'iou': iou, 'max_det': max_det, 'batch': batch,
        'device': device, 'images': len(image_paths),
    }
    if cache_path.is_file() and not force:
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
        cached = payload.get('metadata', {})
        checks = ('protocol_version', 'model_sha256', 'dataset_signature', 'imgsz', 'conf', 'iou', 'max_det')
        if all(cached.get(key) == metadata[key] for key in checks):
            return payload

    class CaptureDetectionValidator(DetectionValidator):
        last_instance: Any = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.capture_records: list[dict[str, Any]] = []
            type(self).last_instance = self

        def _process_batch(self, preds: dict[str, Any], batch: dict[str, Any]) -> dict[str, np.ndarray]:
            result = super()._process_batch(preds, batch)
            pred_cls = preds['cls'].detach().cpu().numpy()
            pred_conf = preds['conf'].detach().cpu().numpy()
            pred_boxes = preds['bboxes'].detach().cpu().numpy()
            gt_cls = batch['cls'].detach().cpu().numpy()
            gt_boxes = batch['bboxes'].detach().cpu().numpy()
            input_height, input_width = (int(v) for v in batch['imgsz'])
            original_height, original_width = (int(v) for v in batch['ori_shape'])
            ratio_pad = serializable_ratio_pad(batch['ratio_pad'])
            record_meta = {
                'input_shape': [input_height, input_width],
                'original_shape': [original_height, original_width],
                'original_width': original_width,
                'original_height': original_height,
                'ratio_pad': ratio_pad,
            }
            self.capture_records.append({
                'file_name': RuntimePath(batch['im_file']).name,
                'image': str(batch['im_file']),
                **record_meta,
                'ground_truth': [
                    {
                        'class_id': int(cls),
                        'class_name': names[int(cls)],
                        'bbox': [float(v) for v in box],
                        'bbox_original': validation_box_to_original(box, record_meta),
                    }
                    for box, cls in zip(gt_boxes, gt_cls)
                ],
                'predictions': [
                    {
                        'class_id': int(cls),
                        'class_name': names[int(cls)],
                        'confidence': float(score),
                        'bbox': [float(v) for v in box],
                        'bbox_original': validation_box_to_original(box, record_meta),
                    }
                    for box, score, cls in zip(pred_boxes, pred_conf, pred_cls)
                ],
            })
            return result

    CaptureDetectionValidator.last_instance = None
    model = YOLO(str(model_path), task='detect')
    official = model.val(
        validator=CaptureDetectionValidator, data=str(dataset_yaml), split='val', imgsz=imgsz,
        batch=batch, device=device, workers=0, amp=True,
        conf=conf, iou=iou, max_det=max_det, rect=True,
        plots=False, verbose=False,
    )
    validator = CaptureDetectionValidator.last_instance
    if validator is None or not validator.capture_records:
        raise RuntimeError('官方验证器没有捕获到预测记录')
    records = sorted(validator.capture_records, key=lambda row: row['file_name'])
    if len(records) != len(image_paths):
        expected_names = {path.name for path in image_paths}
        captured_names = [str(row['file_name']) for row in records]
        missing_names = sorted(expected_names - set(captured_names))
        duplicate_names = sorted(name for name, count in Counter(captured_names).items() if count > 1)
        raise RuntimeError(
            f'捕获图片数量异常: {len(records)} != {len(image_paths)}; '
            f'missing={missing_names}; duplicate={duplicate_names}'
        )

    custom_metrics = evaluation_stats(records)
    official_metrics = {'map50': float(official.box.map50), 'map50_95': float(official.box.map)}
    delta = {
        'map50': abs(custom_metrics['map50'] - official_metrics['map50']),
        'map50_95': abs(custom_metrics['map50_95'] - official_metrics['map50_95']),
    }
    if max(delta.values()) > 1e-10:
        raise RuntimeError(f'捕获结果与官方验证指标不一致: {official_metrics} {custom_metrics} delta={delta}')
    metadata.update({
        'official_metrics': official_metrics,
        'captured_metrics': {'map50': custom_metrics['map50'], 'map50_95': custom_metrics['map50_95']},
        'metric_delta': delta,
        'predictions': int(sum(len(row['predictions']) for row in records)),
        'targets': int(sum(len(row['ground_truth']) for row in records)),
    })
    payload = {'metadata': metadata, 'records': records}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return payload


def fuse_records(
    records_by_view: dict[int, list[dict[str, Any]]], min_views: int,
) -> list[dict[str, Any]]:
    """Fuse views after mapping each view back to original-image coordinates.

    The fused box is finally placed in the coordinate space of the smallest
    view (normally 640). Its ground truth therefore stays in exactly the same
    letterboxed validation space used by the official validator.
    """
    from fusion_postprocess import fuse_views

    by_name: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for imgsz, records in records_by_view.items():
        for record in records:
            by_name[record['file_name']][imgsz] = record
    fused_records: list[dict[str, Any]] = []
    for file_name, views in sorted(by_name.items()):
        base_size = min(views)
        base = views[base_size]
        detections_by_view = {
            str(imgsz): [
                {
                    'class_id': p['class_id'], 'class': p['class_name'],
                    'confidence': p['confidence'], 'bbox': p['bbox_original'],
                }
                for p in record['predictions']
            ]
            for imgsz, record in views.items()
        }
        fused = fuse_views(
            detections_by_view, min_views=min_views,
            width=base['original_width'], height=base['original_height'], arbitration=False,
        )
        for prediction in fused:
            prediction['bbox'] = original_box_to_validation(prediction['bbox'], base)
            prediction.pop('bbox_original', None)
        fused_records.append({
            **{k: v for k, v in base.items() if k != 'predictions'},
            'predictions': fused,
        })
    return fused_records


def names_for_evaluation(names: Sequence[str], per_class: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, row in per_class.items():
        cls = int(key)
        result[key] = {'class_id': cls, 'class_name': names[cls], **row}
    return result


def write_reports(
    output: Path, methods: dict[str, list[dict[str, Any]]], names: Sequence[str], focus_files: set[str],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    report_methods: dict[str, Any] = {}
    for method, records in methods.items():
        metrics = evaluation_stats(records)
        metrics['per_class'] = names_for_evaluation(names, metrics['per_class'])
        metrics['diagnostics_at_0.15'] = diagnostics(records, names, 'predictions', 0.15, focus_files)
        report_methods[method] = metrics
        with (output / f'{method}_image_diagnostics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['file_name', 'gt_count', 'prediction_count', 'missing_classes', 'extra_classes'])
            for record in records:
                row = metrics['diagnostics_at_0.15']['focus_images'].get(record['file_name'])
                if row is None:
                    continue
                writer.writerow([
                    record['file_name'], row['gt_count'], row['prediction_count'],
                    '|'.join(x['class_name'] for x in row['missing']),
                    '|'.join(f"{x['class_name']}:{x['confidence']:.4f}" for x in row['extra']),
                ])
    report = {
        'schema_version': 1,
        'created_at': now_iso(),
        'images': len(next(iter(methods.values()))),
        'class_count': len(names),
        'focus_images': sorted(focus_files),
        'methods': report_methods,
        'safety': {
            'deployment_config_modified': False,
            'task_a_b_modified': False,
            'robot_or_gripper_control': False,
        },
    }
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# v11 多尺度与融合离线验收', '',
        f"- 图片：{report['images']} 张多物品验证图",
        f"- 类别：{report['class_count']}",
        '- 置信度诊断：0.15',
        '- IoU 诊断：0.50', '',
        '| 方法 | mAP50 | mAP50-95 | P@0.15 | R@0.15 | 漏检 | 误报 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method, row in report_methods.items():
        diag = row['diagnostics_at_0.15']['totals']
        lines.append(
            f"| {method} | {row['map50']:.6f} | {row['map50_95']:.6f} | "
            f"{diag['precision']:.4f} | {diag['recall']:.4f} | {diag['false_negatives']} | {diag['false_positives']} |"
        )
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--review', type=Path, default=DEFAULT_REVIEW)
    parser.add_argument('--prefix', default='multiobj__')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640, 800])
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--force', action='store_true', help='忽略预测缓存并重新推理')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    dataset_root, val_dir, names = resolve_dataset(args.dataset)
    image_paths = sorted(
        p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p.name.startswith(args.prefix)
    )
    if not image_paths:
        raise RuntimeError(f'未找到前缀为 {args.prefix!r} 的验证图片: {val_dir}')
    focus_files: set[str] = set()
    if args.review.is_file():
        review = json.loads(args.review.read_text(encoding='utf-8'))
        focus_files = {str(item['file_name']) for item in review.get('issues', [])}

    args.cache.mkdir(parents=True, exist_ok=True)
    records_by_view: dict[int, list[dict[str, Any]]] = {}
    capture_metadata: dict[str, Any] = {}
    for imgsz in args.imgsz:
        cache_path = args.cache / f'exact_predictions_{args.prefix.rstrip("_")}_{imgsz}.json'
        payload = load_or_capture(
            image_paths, cache_path, args.model, args.dataset, imgsz, args.device, args.batch,
            args.conf, args.iou, args.max_det, args.force, names,
        )
        records_by_view[imgsz] = payload['records']
        capture_metadata[str(imgsz)] = payload['metadata']
        official = payload['metadata']['official_metrics']
        expected = OFFICIAL_EXPECTED.get(len(image_paths))
        if imgsz == 640 and expected is not None:
            actual = {'map50': official['map50'], 'map50_95': official['map50_95']}
            delta = {key: abs(actual[key] - expected[key]) for key in expected}
            if max(delta.values()) > 1e-10:
                raise RuntimeError(f'640 官方基线复现失败: actual={actual} expected={expected} delta={delta}')

    image_sets = [ {r['file_name'] for r in records} for records in records_by_view.values() ]
    if any(image_set != image_sets[0] for image_set in image_sets[1:]):
        raise RuntimeError('不同尺度的图片集合不一致')

    methods: dict[str, list[dict[str, Any]]] = {
        f'{imgsz}_raw': records for imgsz, records in records_by_view.items()
    }
    if len(records_by_view) >= 2:
        methods['640_800_wbf_union'] = fuse_records(records_by_view, min_views=1)
        methods['640_800_wbf_consensus'] = fuse_records(records_by_view, min_views=2)
    report = write_reports(args.output, methods, names, focus_files)
    report['capture_metadata'] = capture_metadata
    report['baseline_verified'] = bool(
        len(image_paths) == 71 and 640 in records_by_view
        and max(abs(capture_metadata['640']['official_metrics']['map50'] - OFFICIAL_EXPECTED[71]['map50']),
                abs(capture_metadata['640']['official_metrics']['map50_95'] - OFFICIAL_EXPECTED[71]['map50_95'])) <= 1e-10
    )
    (args.output / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps({m: {'mAP50': r['map50'], 'mAP50-95': r['map50_95'],
                          **r['diagnostics_at_0.15']['totals']} for m, r in report['methods'].items()},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
