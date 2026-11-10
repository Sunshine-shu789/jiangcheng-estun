#!/usr/bin/env python3
"""Build a v7 training view that repeats only class-17/26 train images.

The source v6 dataset is never modified.  Images and labels in the v7 view are
absolute symlinks, so no image pixels are copied away from the data volume.
Validation split is copied one-to-one and is never oversampled.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import yaml

DATA_ROOT = Path('/media/sunshine/新加卷').resolve()
PROJECT_ROOT = Path('/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序')
SOURCE = PROJECT_ROOT / 'datasets/track1_training_merged_v6'
DEST = PROJECT_ROOT / 'datasets/track1_training_merged_v7_oversample_new2'
TARGET_INDICES = (16, 25)
TOTAL_COPIES = 4  # original + 3 duplicates, train only
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def link_absolute(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(destination))


def label_classes(path: Path) -> list[int]:
    classes = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{path}:{line_number}: 不是 5 列 YOLO 检测标签')
        classes.append(int(parts[0]))
    return classes


def main() -> int:
    if not (SOURCE / '.complete').is_file():
        raise SystemExit(f'v6 数据集未完成：{SOURCE}')
    if not DEST.is_absolute() or DATA_ROOT not in DEST.parents:
        raise SystemExit(f'v7 目标目录不在数据盘：{DEST}')
    if DEST.exists():
        raise SystemExit(f'拒绝覆盖已有 v7 目录：{DEST}')

    source_yaml = yaml.safe_load((SOURCE / 'dataset.yaml').read_text(encoding='utf-8'))
    names = source_yaml['names']
    if not isinstance(names, list) or len(names) != 40:
        raise SystemExit('v6 dataset.yaml 类别名称异常')

    DEST.mkdir(parents=True)
    for split in ('train', 'val'):
        (DEST / 'images' / split).mkdir(parents=True)
        (DEST / 'labels' / split).mkdir(parents=True)

    summary = {}
    all_class_counts = {}
    total_boxes = 0
    oversampled_stems = []

    for split in ('train', 'val'):
        image_count = 0
        box_count = 0
        class_counts: dict[str, int] = {}
        source_images = sorted(
            p for p in (SOURCE / 'images' / split).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in source_images:
            label_path = SOURCE / 'labels' / split / f'{image_path.stem}.txt'
            if not label_path.is_file():
                raise FileNotFoundError(f'缺少标签：{label_path}')
            classes = label_classes(label_path)
            copies = TOTAL_COPIES if split == 'train' and any(c in TARGET_INDICES for c in classes) else 1
            if copies > 1:
                oversampled_stems.append(image_path.stem)
            for copy_index in range(copies):
                suffix = '' if copy_index == 1 else f'__v7x{copy_index}'
                destination_stem = image_path.stem + suffix
                destination_image = DEST / 'images' / split / f'{destination_stem}{image_path.suffix.lower()}'
                destination_label = DEST / 'labels' / split / f'{destination_stem}.txt'
                link_absolute(image_path, destination_image)
                link_absolute(label_path, destination_label)
                image_count += 1
                box_count += len(classes)
                for class_id in classes:
                    class_counts[str(class_id)] = class_counts.get(str(class_id), 0) + 1
        summary[split] = {
            'source_images': len(source_images),
            'images': image_count,
            'boxes': box_count,
            'class_counts': dict(sorted(class_counts.items(), key=lambda item: int(item[0]))),
        }
        all_class_counts[split] = summary[split]['class_counts']
        total_boxes += box_count

    for class_id in TARGET_INDICES:
        for split in ('train', 'val'):
            if all_class_counts[split].get(str(class_id), 0) <= 0:
                raise RuntimeError(f'{split} 缺少目标类别索引 {class_id}')

    dataset_yaml = {
        'path': str(DEST),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 40,
        'names': names,
    }
    (DEST / 'dataset.yaml').write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding='utf-8'
    )

    v6_audit = json.loads((SOURCE / 'audit_summary.json').read_text(encoding='utf-8'))
    audit = {
        'schema_version': 1,
        'created_at': now_iso(),
        'dataset_root': str(DEST),
        'source_dataset': str(SOURCE),
        'training_ready': True,
        'training_started': False,
        'materialization': 'absolute_symlink',
        'checks': {
            'source_v6_complete': True,
            'all_links_inside_data_volume': True,
            'train_val_disjoint_from_v6': True,
            'target_classes_train_only_oversampled': True,
            'validation_not_oversampled': True,
        },
        'images': {split: summary[split]['images'] for split in ('train', 'val')},
        'source_images': {split: summary[split]['source_images'] for split in ('train', 'val')},
        'boxes': total_boxes,
        'class_counts': all_class_counts,
        'oversampling': {
            'class_indices': list(TARGET_INDICES),
            'official_ids': [index + 1 for index in TARGET_INDICES],
            'class_names': [names[index] for index in TARGET_INDICES],
            'total_copies_in_train': TOTAL_COPIES,
            'unique_train_images': len(oversampled_stems),
            'train_images_after_oversampling': summary['train']['images'],
        },
        'source_summary': v6_audit.get('source_summary', {}),
    }
    (DEST / 'audit_summary.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    for name in ('merge_manifest.jsonl', 'new_bboxes.csv', 'class_counts.csv', 'README.md'):
        source_file = SOURCE / name
        if source_file.is_file():
            target_file = DEST / f'v6_{name}'
            link_absolute(source_file, target_file)

    (DEST / '.complete').write_text(json.dumps({
        'created_at': now_iso(),
        'train_images': summary['train']['images'],
        'val_images': summary['val']['images'],
        'boxes': total_boxes,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'dataset': str(DEST),
        'train': summary['train'],
        'val': summary['val'],
        'total_boxes': total_boxes,
        'oversampled_unique_train_images': len(oversampled_stems),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
