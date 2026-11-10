#!/usr/bin/env python3
"""Evaluate one Track-1 checkpoint on a full val split and an optional filename subset.

The script is read-only for datasets, model weights and Task A/B.  It writes only
into the requested report directory and creates a symlinked subset dataset there.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path('/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序').resolve()
IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.webp'}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def names_to_list(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(value[index]) for index in range(len(value))]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f'非法类别映射: {type(value).__name__}')


def resolve_split(dataset_yaml: Path, split: str) -> tuple[Path, Path, list[str]]:
    payload = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('dataset.yaml 顶层不是对象')
    dataset_root = Path(str(payload.get('path', dataset_yaml.parent))).expanduser().resolve()
    value = payload.get(split)
    if not isinstance(value, str):
        raise ValueError(f'dataset.yaml 的 {split} 必须是路径字符串')
    split_path = (dataset_root / value).resolve()
    if not split_path.is_dir():
        raise FileNotFoundError(f'验证集目录不存在: {split_path}')
    names = names_to_list(payload.get('names'))
    return dataset_root, split_path, names


def image_label_path(image: Path, dataset_root: Path) -> Path:
    try:
        relative = image.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        # Subset images are symlinks inside the report directory.  Preserve the
        # symlink path instead of resolving back to the source dataset.
        relative = image.absolute().relative_to(dataset_root.absolute())
    parts = list(relative.parts)
    if 'images' not in parts:
        raise ValueError(f'图片不在 images/ 子目录下: {image}')
    index = parts.index('images')
    return dataset_root.joinpath(*parts[:index], 'labels', *parts[index + 1:]).with_suffix('.txt')


def make_filtered_dataset(
    dataset_yaml: Path, report_dir: Path, split: str, prefix: str,
) -> tuple[Path, list[Path], list[Path]]:
    dataset_root, image_dir, _ = resolve_split(dataset_yaml, split)
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem.startswith(prefix)
    )
    if not images:
        raise RuntimeError(f'验证集没有前缀 {prefix!r} 的图片: {image_dir}')
    subset_root = report_dir / 'subset_data' / f'{split}_{prefix.rstrip("_")}'
    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_images = subset_root / 'images' / split
    subset_labels = subset_root / 'labels' / split
    subset_images.mkdir(parents=True)
    subset_labels.mkdir(parents=True)
    labels: list[Path] = []
    for image in images:
        label = image_label_path(image, dataset_root)
        if not label.is_file():
            raise FileNotFoundError(f'图片缺标签: {image}')
        (subset_images / image.name).symlink_to(image.resolve())
        (subset_labels / label.name).symlink_to(label.resolve())
        labels.append(label)
    payload = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    payload['path'] = str(subset_root)
    payload['train'] = f'images/{split}'
    payload['val'] = f'images/{split}'
    subset_yaml = subset_root / 'dataset.yaml'
    subset_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return subset_yaml, images, labels



def make_named_dataset(
    dataset_yaml: Path, report_dir: Path, split: str, names: list[str], label: str,
) -> tuple[Path, list[Path], list[Path]]:
    dataset_root, image_dir, _ = resolve_split(dataset_yaml, split)
    available = {
        path.name: path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing = sorted(set(names) - set(available))
    if missing:
        raise FileNotFoundError(f'命名验证图片缺失 {len(missing)} 张: {missing[:5]}')
    images = [available[name] for name in names]
    subset_root = report_dir / 'subset_data' / label
    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_images = subset_root / 'images' / split
    subset_labels = subset_root / 'labels' / split
    subset_images.mkdir(parents=True)
    subset_labels.mkdir(parents=True)
    labels: list[Path] = []
    for image in images:
        label_path = image_label_path(image, dataset_root)
        if not label_path.is_file():
            raise FileNotFoundError(f'图片缺标签: {image}')
        (subset_images / image.name).symlink_to(image.resolve())
        (subset_labels / label_path.name).symlink_to(label_path.resolve())
        labels.append(label_path)
    payload = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    payload['path'] = str(subset_root)
    payload['train'] = f'images/{split}'
    payload['val'] = f'images/{split}'
    subset_yaml = subset_root / 'dataset.yaml'
    subset_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return subset_yaml, images, labels

def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    if box is None:
        return {'class_count': 0, 'classes': [], 'per_class': {}}
    class_indices = [int(value) for value in box.ap_class_index.tolist()]
    per_class: dict[str, Any] = {}
    for class_id, ap, ap50, precision, recall in zip(
        class_indices,
        box.ap.tolist(), box.ap50.tolist(), box.p.tolist(), box.r.tolist(),
    ):
        per_class[str(class_id)] = {
            'official_id': class_id + 1,
            'precision': float(precision),
            'recall': float(recall),
            'ap50': float(ap50),
            'ap50_95': float(ap),
        }
    return {
        'class_count': len(class_indices),
        'classes': class_indices,
        'map50': float(box.map50),
        'map50_95': float(box.map),
        'mean_precision': float(box.mp),
        'mean_recall': float(box.mr),
        'per_class': per_class,
    }


def dataset_digest(dataset_yaml: Path, images: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset_yaml.resolve()).encode())
    for image in sorted(images, key=lambda path: path.name):
        digest.update(f'\0{image.name}\0{sha256_file(image)}'.encode())
    return digest.hexdigest()


def evaluate(
    model: Any, dataset_yaml: Path, output_root: Path, name: str, split: str,
    imgsz: int, batch: int, device: str, workers: int, amp: bool,
    conf: float, iou: float, max_det: int,
) -> tuple[dict[str, Any], list[Path], list[Path]]:
    dataset_root, image_dir, names = resolve_split(dataset_yaml, split)
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = [image_label_path(image, dataset_root) for image in images]
    result = model.val(
        data=str(dataset_yaml), split=split, imgsz=imgsz, batch=batch,
        device=device, workers=workers, amp=amp, conf=conf, iou=iou,
        max_det=max_det, plots=True, project=str(output_root), name=name,
        exist_ok=True, verbose=True,
    )
    metrics = metrics_to_dict(result)
    metrics.update({
        'images': len(images),
        'dataset_yaml': str(dataset_yaml),
        'dataset_sha256': sha256_file(dataset_yaml),
        'dataset_signature': dataset_digest(dataset_yaml, images),
        'all_40_classes': metrics.get('class_count') == 40,
        'names_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest(),
    })
    return metrics, images, labels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-name', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--prefix', default='multiobj__')
    parser.add_argument('--expected-full-images', type=int, default=170)
    parser.add_argument('--expected-subset-images', type=int, default=72)
    parser.add_argument('--train-overlap-subset', action='store_true')
    parser.add_argument('--reviewed-csv', type=Path, help='可选的 72 张复核清单；用于 train-overlap 诊断')
    parser.add_argument('--reviewed-split', default='train')
    parser.add_argument('--reviewed-expected-images', type=int, default=72)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--max-det', type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model.expanduser().resolve()
    dataset_yaml = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError(f'输出必须位于正式项目目录: {output}')
    output.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    wrapper = YOLO(str(model_path), task='detect')
    _, _, names = resolve_split(dataset_yaml, args.split)
    model_names = names_to_list(wrapper.model.names)
    if int(wrapper.model.nc) != 40 or model_names != names:
        raise RuntimeError('检查点类别槽或类别顺序与数据集不一致')

    full_metrics, full_images, full_labels = evaluate(
        wrapper, dataset_yaml, output / 'results', 'full_val', args.split,
        args.imgsz, args.batch, args.device, args.workers, args.amp,
        args.conf, args.iou, args.max_det,
    )
    if full_metrics['images'] != args.expected_full_images:
        raise RuntimeError(f"全 val 图片数异常: {full_metrics['images']} != {args.expected_full_images}")

    subset_yaml, subset_images, subset_labels = make_filtered_dataset(
        dataset_yaml, output, args.split, args.prefix,
    )
    if len(subset_images) != args.expected_subset_images:
        raise RuntimeError(f'子集图片数异常: {len(subset_images)} != {args.expected_subset_images}')
    subset_metrics, _, _ = evaluate(
        YOLO(str(model_path), task='detect'), subset_yaml, output / 'results', 'multiobject_subset', args.split,
        args.imgsz, args.batch, args.device, args.workers, args.amp,
        args.conf, args.iou, args.max_det,
    )

    reviewed_metrics: dict[str, Any] | None = None
    reviewed_images: list[Path] = []
    reviewed_labels: list[Path] = []
    reviewed_csv = args.reviewed_csv.expanduser().resolve() if args.reviewed_csv else None
    if reviewed_csv is not None:
        if not reviewed_csv.is_file():
            raise FileNotFoundError(reviewed_csv)
        with reviewed_csv.open('r', newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        reviewed_names = [str(row['file_name']) for row in rows]
        if len(reviewed_names) != args.reviewed_expected_images or len(set(reviewed_names)) != len(reviewed_names):
            raise RuntimeError(f'复核清单数量或唯一性异常: {len(reviewed_names)}')
        reviewed_yaml, reviewed_images, reviewed_labels = make_named_dataset(
            dataset_yaml, output, args.reviewed_split, reviewed_names, 'reviewed_72_train_overlap',
        )
        reviewed_metrics, _, _ = evaluate(
            YOLO(str(model_path), task='detect'), reviewed_yaml, output / 'results',
            'reviewed_train_overlap', args.reviewed_split,
            args.imgsz, args.batch, args.device, args.workers, args.amp,
            args.conf, args.iou, args.max_det,
        )
        reviewed_metrics.update({
            'train_overlap': True,
            'holdout': False,
            'role': 'diagnostic_only_not_holdout',
            'review_csv': str(reviewed_csv),
            'review_csv_sha256': sha256_file(reviewed_csv),
        })

    report = {
        'schema_version': 1,
        'created_at': now_iso(),
        'report_name': args.report_name,
        'model': str(model_path),
        'model_sha256': sha256_file(model_path),
        'dataset_yaml': str(dataset_yaml),
        'dataset_yaml_sha256': sha256_file(dataset_yaml),
        'validation_protocol': {
            'imgsz': args.imgsz, 'conf': args.conf, 'iou': args.iou,
            'max_det': args.max_det, 'batch': args.batch, 'device': args.device,
        },
        'full_val': {**full_metrics, 'train_overlap': False},
        'multiobject_val': {
            **subset_metrics,
            'prefix': args.prefix,
            'train_overlap': bool(args.train_overlap_subset),
            'holdout': not bool(args.train_overlap_subset),
        },
        'source_images': {
            'full_val': [str(path) for path in full_images],
            'multiobject_val': [str(path) for path in subset_images],
            'reviewed_train_overlap': [str(path) for path in reviewed_images],
        },
        'source_labels_exist': {
            'full_val': all(path.is_file() for path in full_labels),
            'multiobject_val': all(path.is_file() for path in subset_labels),
            'reviewed_train_overlap': all(path.is_file() for path in reviewed_labels),
        },
        'reviewed_train_overlap': reviewed_metrics,
        'all_40_classes': full_metrics['all_40_classes'] and subset_metrics['all_40_classes'] and (
            reviewed_metrics is None or reviewed_metrics['all_40_classes']
        ),
        'safety': {
            'dataset_modified': False,
            'weights_modified': False,
            'deployment_config_modified': False,
            'task_a_b_modified': False,
            'robot_or_gripper_control': False,
        },
    }
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    markdown = [
        f"# {args.report_name}", '',
        f"- 模型：`{model_path.name}`",
        f"- 模型 SHA256：`{report['model_sha256']}`",
        f"- 全验证集：{full_metrics['images']} 张，mAP50 **{full_metrics['map50']:.6f}**，mAP50-95 **{full_metrics['map50_95']:.6f}**",
        f"- 多物品隔离集：{subset_metrics['images']} 张，mAP50 **{subset_metrics['map50']:.6f}**，mAP50-95 **{subset_metrics['map50_95']:.6f}**",
        f"- 多物品集训练重叠：`{report['multiobject_val']['train_overlap']}`",
        f"- 40 类覆盖：`{report['all_40_classes']}`", '',
        '| 数据集 | 图片 | mAP50 | mAP50-95 | P | R |',
        '|---|---:|---:|---:|---:|---:|',
        f"| 全 val | {full_metrics['images']} | {full_metrics['map50']:.6f} | {full_metrics['map50_95']:.6f} | {full_metrics['mean_precision']:.4f} | {full_metrics['mean_recall']:.4f} |",
        f"| 多物品 | {subset_metrics['images']} | {subset_metrics['map50']:.6f} | {subset_metrics['map50_95']:.6f} | {subset_metrics['mean_precision']:.4f} | {subset_metrics['mean_recall']:.4f} |",
    ]
    if reviewed_metrics is not None:
        markdown.extend([
            '',
            f"- 复核 72 张诊断（训练重叠，不作 holdout）：mAP50 **{reviewed_metrics['map50']:.6f}**，mAP50-95 **{reviewed_metrics['map50_95']:.6f}**",
            f"| 复核 72 张（train overlap） | {reviewed_metrics['images']} | {reviewed_metrics['map50']:.6f} | {reviewed_metrics['map50_95']:.6f} | {reviewed_metrics['mean_precision']:.4f} | {reviewed_metrics['mean_recall']:.4f} |",
        ])
    (output / 'report.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'evaluation_complete', 'report': str(report_path),
        'full_val': {key: full_metrics[key] for key in ('images', 'map50', 'map50_95', 'mean_precision', 'mean_recall')},
        'multiobject_val': {key: subset_metrics[key] for key in ('images', 'map50', 'map50_95', 'mean_precision', 'mean_recall')},
        'reviewed_train_overlap': None if reviewed_metrics is None else {key: reviewed_metrics[key] for key in ('images', 'map50', 'map50_95', 'mean_precision', 'mean_recall')},
        'all_40_classes': report['all_40_classes'],
        'multiobject_train_overlap': report['multiobject_val']['train_overlap'],
    }, ensure_ascii=False, indent=2))
    return 0 if report['all_40_classes'] else 5


if __name__ == '__main__':
    raise SystemExit(main())
