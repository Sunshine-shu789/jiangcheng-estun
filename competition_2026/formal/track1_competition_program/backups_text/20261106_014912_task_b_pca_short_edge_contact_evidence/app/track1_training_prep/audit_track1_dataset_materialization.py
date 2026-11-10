#!/usr/bin/env python3
"""Audit a materialized Track-1 dataset before training.

The audit is intentionally independent from the builder. It verifies that all
dataset entries are absolute symlinks into the data volume, that train and
validation contain no shared resolved image, that validation is unchanged from
v6, that every image decodes, and that every synthetic green meat-can variant
is train-derived and still carries class index 9.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

DATA_ROOT = Path('/media/sunshine/新加卷').resolve()
PROJECT_ROOT = DATA_ROOT / '埃斯顿比赛_2026/赛道一比赛程序'
DEFAULT_DATASET = PROJECT_ROOT / 'datasets/track1_training_merged_v9_hardneg'
DEFAULT_SOURCE = PROJECT_ROOT / 'datasets/track1_training_merged_v6'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MEAT_CAN = 9


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def is_within(path: Path, parent: Path) -> bool:
    try:
        common = os.path.commonpath((os.path.realpath(path), os.path.realpath(parent)))
        return common == os.path.realpath(parent)
    except (TypeError, ValueError):
        return False


def real_path(path: Path) -> Path:
    """Return the normalized link target without requiring a slow stat call."""
    return Path(os.path.realpath(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def regular_image_entries(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    ) if directory.is_dir() else []


def all_entries(directory: Path) -> list[Path]:
    return sorted(directory.iterdir()) if directory.is_dir() else []


def parse_label(path: Path) -> tuple[list[tuple[int, str]], list[str]]:
    rows: list[tuple[int, str]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except Exception as exc:
        return [], [f'{path}: 无法读取: {type(exc).__name__}: {exc}']
    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f'{path}:{line_no}: 不是五列 YOLO 标签')
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f'{path}:{line_no}: 标签不是数字')
            continue
        if not 0 <= class_id < 40:
            errors.append(f'{path}:{line_no}: 类别越界 {class_id}')
            continue
        if any(not 0.0 <= value <= 1.0 for value in values):
            errors.append(f'{path}:{line_no}: 坐标不在 [0,1]')
            continue
        rows.append((class_id, line))
    return rows, errors


def audit_link(path: Path, errors: list[str], target_errors: list[str]) -> Path | None:
    if not path.is_symlink():
        errors.append(f'不是符号链接: {path}')
        return None
    raw_target = os.readlink(path)
    if not os.path.isabs(raw_target):
        errors.append(f'链接目标不是绝对路径: {path} -> {raw_target}')
    target = real_path(path)
    if not is_within(target, DATA_ROOT):
        target_errors.append(f'链接目标不在数据盘: {path} -> {target}')
    return target


def validate_split_pairing(dataset: Path, split: str, errors: list[str]) -> None:
    image_dir = dataset / 'images' / split
    label_dir = dataset / 'labels' / split
    image_stems = {path.stem for path in regular_image_entries(image_dir)}
    label_stems = {path.stem for path in all_entries(label_dir) if path.suffix.lower() == '.txt'}
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)
    if missing_labels:
        errors.append(f'{split} 有图片缺标签: {missing_labels[:10]}')
    if extra_labels:
        errors.append(f'{split} 有标签缺图片: {extra_labels[:10]}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    dataset = real_path(args.dataset.expanduser().absolute())
    source = real_path(args.source.expanduser().absolute())
    output = real_path(args.output.expanduser().absolute()) if args.output else dataset / 'materialization_audit.json'
    pending = dataset / '.materialization_pending'
    complete = dataset / '.complete'
    audit_summary_path = dataset / 'audit_summary.json'

    errors: list[str] = []
    target_errors: list[str] = []
    if not is_within(dataset, DATA_ROOT):
        errors.append(f'数据集不在数据盘: {dataset}')
    if not is_within(source, DATA_ROOT):
        errors.append(f'源数据集不在数据盘: {source}')
    if not (source / '.complete').is_file():
        errors.append(f'源 v6 数据集未完成: {source}')
    if not dataset.is_dir():
        errors.append(f'数据集目录不存在: {dataset}')
    if not pending.is_file():
        errors.append(f'缺少 .materialization_pending: {pending}')

    split_targets: dict[str, dict[str, list[Path]]] = {'train': {'images': [], 'labels': []}, 'val': {'images': [], 'labels': []}}
    all_links_absolute = True
    decoded_images = 0
    decode_errors: list[str] = []
    label_errors: list[str] = []
    class_counts: dict[str, Counter[int]] = {'train': Counter(), 'val': Counter()}

    for split in ('train', 'val'):
        image_dir = dataset / 'images' / split
        label_dir = dataset / 'labels' / split
        validate_split_pairing(dataset, split, errors)

        image_entries = all_entries(image_dir)
        label_entries = all_entries(label_dir)
        if any(not path.is_symlink() for path in image_entries + label_entries):
            all_links_absolute = False

        before_errors = len(errors)
        for image_path in image_entries:
            target = audit_link(image_path, errors, target_errors)
            if target is None:
                continue
            split_targets[split]['images'].append(target)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                decode_errors.append(f'{split}/{image_path.name}: 无法解码')
            else:
                decoded_images += 1

        for label_path in label_entries:
            target = audit_link(label_path, errors, target_errors)
            if target is None:
                continue
            split_targets[split]['labels'].append(target)
            rows, row_errors = parse_label(label_path)
            label_errors.extend(row_errors)
            for class_id, _ in rows:
                class_counts[split][class_id] += 1

        if len(errors) > before_errors:
            all_links_absolute = False

    train_resolved = set(split_targets['train']['images'])
    val_resolved = set(split_targets['val']['images'])
    overlap = sorted(str(path) for path in train_resolved & val_resolved)
    if overlap:
        errors.append(f'训练集与验证集真实图片路径重叠 {len(overlap)} 个: {overlap[:5]}')

    source_val_images = {path.name: real_path(path) for path in regular_image_entries(source / 'images/val')}
    source_val_labels = {path.name: real_path(path) for path in all_entries(source / 'labels/val') if path.suffix.lower() == '.txt'}
    dest_val_images = {path.name: real_path(path) for path in regular_image_entries(dataset / 'images/val')}
    dest_val_labels = {path.name: real_path(path) for path in all_entries(dataset / 'labels/val') if path.suffix.lower() == '.txt'}
    validation_one_to_one = (
        source_val_images == dest_val_images
        and source_val_labels == dest_val_labels
        and len(source_val_images) == 121
        and len(source_val_labels) == 121
    )
    if not validation_one_to_one:
        errors.append('验证集不是 v6 的 121 图/121 标签一对一链接')

    source_train_images = {real_path(path) for path in regular_image_entries(source / 'images/train')}
    manifest_path = dataset / 'synthetic_green_meat_can_manifest.jsonl'
    manifest_items: list[dict[str, Any]] = []
    if manifest_path.is_file():
        for line_no, raw in enumerate(manifest_path.read_text(encoding='utf-8').splitlines(), 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f'合成清单第 {line_no} 行 JSON 无效: {exc}')
                continue
            manifest_items.append(item)
    else:
        errors.append(f'缺少合成清单: {manifest_path}')

    synthetic_ok = bool(manifest_items)
    train_green_variant_links: set[Path] = set()
    for item in manifest_items:
        try:
            source_image = real_path(Path(str(item['source_image'])))
            source_label = real_path(Path(str(item['source_label'])))
            output_image = real_path(Path(str(item['output_image'])))
            destination_link = dataset / 'images/train' / output_image.name
            destination_label_link = dataset / 'labels/train' / f'{output_image.stem}.txt'
        except Exception as exc:
            errors.append(f'合成样本路径无效: {type(exc).__name__}: {exc}')
            synthetic_ok = False
            continue
        if item.get('train_only') is not True:
            errors.append(f'合成样本未标记 train_only: {output_image}')
            synthetic_ok = False
        if source_image not in source_train_images or source_image in source_val_images.values():
            errors.append(f'合成样本来源不是 v6 train: {source_image}')
            synthetic_ok = False
        if not is_within(output_image, DATA_ROOT):
            errors.append(f'合成图片不在数据盘: {output_image}')
            synthetic_ok = False
        if not destination_link.is_symlink() or real_path(destination_link) != output_image:
            errors.append(f'合成图片未正确链接进 train: {destination_link}')
            synthetic_ok = False
        else:
            train_green_variant_links.add(real_path(destination_link))
        if not destination_label_link.is_symlink() or real_path(destination_label_link) != source_label:
            errors.append(f'合成标签未正确链接到 train: {destination_label_link}')
            synthetic_ok = False
        rows, row_errors = parse_label(destination_label_link) if destination_label_link.is_file() else ([], ['标签不可读'])
        label_errors.extend(row_errors)
        if not any(class_id == MEAT_CAN for class_id, _ in rows):
            errors.append(f'合成样本缺少 class 9 标签: {destination_label_link}')
            synthetic_ok = False

    if len(train_green_variant_links) != len(manifest_items):
        errors.append(f'合成训练链接数不一致: manifest={len(manifest_items)}, links={len(train_green_variant_links)}')
        synthetic_ok = False
    if not train_green_variant_links:
        errors.append('没有合成绿色肉罐头训练样本')
        synthetic_ok = False

    all_images_decode = not decode_errors
    all_labels_valid = not label_errors
    train_val_disjoint = not overlap
    targets_inside_data = not target_errors
    ready = all((
        not errors,
        all_images_decode,
        all_labels_valid,
        train_val_disjoint,
        targets_inside_data,
        validation_one_to_one,
        synthetic_ok,
        all_links_absolute,
    ))

    audit = {
        'schema_version': 1,
        'created_at': now_iso(),
        'dataset': str(dataset),
        'source_dataset': str(source),
        'dataset_yaml_sha256': sha256_file(dataset / 'dataset.yaml') if (dataset / 'dataset.yaml').is_file() else None,
        'train_images': len(regular_image_entries(dataset / 'images/train')),
        'val_images': len(regular_image_entries(dataset / 'images/val')),
        'train_unique_resolved_images': len(train_resolved),
        'val_unique_resolved_images': len(val_resolved),
        'train_val_resolved_overlap': len(overlap),
        'overlap_examples': overlap[:10],
        'all_entries_are_absolute_symlinks': all_links_absolute,
        'all_link_targets_inside_data_volume': targets_inside_data,
        'decoded_images': decoded_images,
        'decode_errors': decode_errors,
        'label_errors': label_errors[:20],
        'class_counts': {split: {str(key): int(value) for key, value in sorted(counts.items())} for split, counts in class_counts.items()},
        'synthetic_variants': len(manifest_items),
        'all_synthetic_variants_label_class9': synthetic_ok,
        'validation_one_to_one_from_v6': validation_one_to_one,
        'checks': {
            'train_val_disjoint': train_val_disjoint,
            'absolute_symlinks': all_links_absolute,
            'targets_inside_data_volume': targets_inside_data,
            'all_images_decode': all_images_decode,
            'all_labels_valid': all_labels_valid,
            'synthetic_label_preserved': synthetic_ok,
            'validation_unchanged_count': validation_one_to_one,
        },
        'errors': list(dict.fromkeys(errors))[:50],
        'ready': ready,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    if ready:
        summary = json.loads(audit_summary_path.read_text(encoding='utf-8'))
        summary['training_ready'] = True
        summary['training_started'] = False
        summary['materialization_ready'] = True
        summary['materialization_audit'] = str(output)
        summary['materialization_audit_sha256'] = sha256_file(output)
        audit_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (dataset / '.complete').write_text(json.dumps({
            'created_at': now_iso(),
            'materialization_audit': str(output),
            'materialization_audit_sha256': sha256_file(output),
            'train_images': audit['train_images'],
            'val_images': audit['val_images'],
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        pending.unlink(missing_ok=True)
    else:
        complete.unlink(missing_ok=True)
        if audit_summary_path.is_file():
            summary = json.loads(audit_summary_path.read_text(encoding='utf-8'))
            summary['training_ready'] = False
            summary['materialization_ready'] = False
            summary['materialization_audit'] = str(output)
            audit_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps({
        'ready': ready,
        'audit': str(output),
        'train_images': audit['train_images'],
        'val_images': audit['val_images'],
        'train_val_resolved_overlap': audit['train_val_resolved_overlap'],
        'decoded_images': decoded_images,
        'synthetic_variants': len(manifest_items),
        'errors': audit['errors'],
    }, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
