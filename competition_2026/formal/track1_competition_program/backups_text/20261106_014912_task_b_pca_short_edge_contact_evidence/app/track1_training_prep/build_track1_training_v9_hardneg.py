#!/usr/bin/env python3
"""Build a train-only v9 hard-negative view without modifying v6/v7.

Validation images stay one-to-one. Train additions target:
- official 10-meat_can as an Eggs hard negative;
- green-front official 17-Eggs_Plush frames;
- official 26-Plastic_Strawberry;
- a small number of Eggs-confusion classes.

All links are absolute. Synthetic green meat-can variants are derived only from
train images and remain labelled as 10-meat_can. No validation frame is copied
into train.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

DATA_ROOT = Path('/media/sunshine/新加卷').resolve()
PROJECT_ROOT = DATA_ROOT / '埃斯顿比赛_2026/赛道一比赛程序'
SOURCE = PROJECT_ROOT / 'datasets/track1_training_merged_v6'
DEST = PROJECT_ROOT / 'datasets/track1_training_merged_v9_hardneg'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MEAT_CAN = 9
EGGS = 16
STRAWBERRY = 25
CONFUSION_CLASSES = (5, 12, 18, 28)
MEAT_CAN_EXTRA_COPIES = 5
GREEN_EGG_EXTRA_COPIES = 4
ORANGE_EGG_EXTRA_COPIES = 2
STRAWBERRY_EXTRA_COPIES = 3
CONFUSION_EXTRA_COPIES = 1
SYNTHETIC_GREEN_VARIANTS = 2


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def link_absolute(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(destination))


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_no, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{path}:{line_no}: 不是五列 YOLO 标签')
        class_id = int(parts[0])
        x, y, width, height = (float(value) for value in parts[1:])
        if not 0 <= class_id < 40:
            raise ValueError(f'{path}:{line_no}: 类别越界 {class_id}')
        rows.append((class_id, x, y, width, height))
    return rows


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def green_fraction(image_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> float | None:
    egg_boxes = [box for box in boxes if box[0] == EGGS]
    if not egg_boxes:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'无法读取图片: {image_path}')
    height, width = image.shape[:2]
    _, x, y, box_width, box_height = max(egg_boxes, key=lambda box: box[3] * box[4])
    x1 = max(0, min(width, int((x - box_width / 2.0) * width)))
    x2 = max(0, min(width, int((x + box_width / 2.0) * width)))
    y1 = max(0, min(height, int((y - box_height / 2.0) * height)))
    y2 = max(0, min(height, int((y + box_height / 2.0) * height)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 30)
        & (hsv[:, :, 0] <= 100)
        & (hsv[:, :, 1] >= 50)
        & (hsv[:, :, 2] >= 40)
    )
    return float(mask.mean())


def make_synthetic_green_variant(
    source_image: Path,
    boxes: list[tuple[int, float, float, float, float]],
    destination: Path,
    target_hue: int,
) -> None:
    image = cv2.imread(str(source_image))
    if image is None:
        raise ValueError(f'无法读取图片: {source_image}')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    touched = np.zeros((height, width), dtype=bool)
    for class_id, x, y, box_width, box_height in boxes:
        if class_id != MEAT_CAN:
            continue
        x1 = max(0, min(width, int((x - box_width / 2.0) * width)))
        x2 = max(0, min(width, int((x + box_width / 2.0) * width)))
        y1 = max(0, min(height, int((y - box_height / 2.0) * height)))
        y2 = max(0, min(height, int((y + box_height / 2.0) * height)))
        local_hsv = hsv[y1:y2, x1:x2]
        local_mask = (local_hsv[:, :, 1] >= 35) & (local_hsv[:, :, 2] >= 30)
        touched[y1:y2, x1:x2] |= local_mask
    if not touched.any():
        raise ValueError(f'肉罐头区域没有可着色像素: {source_image}')
    hsv[touched, 0] = target_hue
    hsv[touched, 1] = np.clip(hsv[touched, 1].astype(np.int16) + 20, 80, 255).astype(np.uint8)
    changed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), changed, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f'无法写合成训练图: {destination}')


def annotated_tile(image_path: Path, boxes: list[tuple[int, float, float, float, float]], caption: str, width: int = 420, height: int = 315) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'无法读取图片: {image_path}')
    h, w = image.shape[:2]
    for class_id, x, y, box_width, box_height in boxes:
        x1 = max(0, min(w - 1, int((x - box_width / 2.0) * w)))
        x2 = max(0, min(w - 1, int((x + box_width / 2.0) * w)))
        y1 = max(0, min(h - 1, int((y - box_height / 2.0) * h)))
        y2 = max(0, min(h - 1, int((y + box_height / 2.0) * h)))
        color = (255, 90, 40) if class_id == MEAT_CAN else (0, 170, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{class_id}', (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    tile = np.full((height + 52, width, 3), 18, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(tile, caption[:60], (8, height + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(tile, 'TRAIN-ONLY SYNTHETIC HARD NEGATIVE; LABEL=9', (8, height + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 220, 150), 1, cv2.LINE_AA)
    return tile


def make_synthetic_sheet(items: list[tuple[Path, list[tuple[int, float, float, float, float]], str]], output: Path) -> None:
    if not items:
        return
    columns = 4
    tiles = [annotated_tile(image, boxes, caption) for image, boxes, caption in items]
    tile_h, tile_w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / columns)
    canvas = np.full((rows * tile_h + 54, columns * tile_w, 3), 8, dtype=np.uint8)
    cv2.putText(canvas, 'v9 train-only synthetic green meat-can hard negatives', (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2, cv2.LINE_AA)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[54 + row * tile_h:54 + (row + 1) * tile_h, column * tile_w:(column + 1) * tile_w] = tile
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f'无法写合成接触表: {output}')


def main() -> int:
    if not (SOURCE / '.complete').is_file():
        raise SystemExit(f'v6 数据集未完成: {SOURCE}')
    if not is_within(DEST, DATA_ROOT):
        raise SystemExit(f'v9 目标不在数据盘: {DEST}')
    if DEST.exists():
        raise SystemExit(f'拒绝覆盖已有 v9 目录: {DEST}')

    source_yaml = yaml.safe_load((SOURCE / 'dataset.yaml').read_text(encoding='utf-8'))
    names = source_yaml.get('names')
    if not isinstance(names, list) or len(names) != 40:
        raise SystemExit('v6 dataset.yaml 类别名称异常')

    DEST.mkdir(parents=True)
    for split in ('train', 'val'):
        (DEST / 'images' / split).mkdir(parents=True)
        (DEST / 'labels' / split).mkdir(parents=True)
    synthetic_dir = DEST / 'assets' / 'synthetic_green_meat_can'
    synthetic_dir.mkdir(parents=True)
    manifests: list[dict[str, Any]] = []

    split_info: dict[str, dict[str, Any]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    image_counts: dict[str, int] = {}
    box_counts: dict[str, int] = {}
    extras_by_kind: Counter[str] = Counter()
    excluded_train_val_overlap: list[dict[str, str]] = []

    # Resolve validation targets before materializing train. A historical v6
    # train alias can point at a validation image even though the split folders
    # themselves are different. Compare real paths so every such alias is
    # excluded, not just one known filename.
    validation_resolved: set[Path] = set()
    for validation_image in image_files(SOURCE / 'images/val'):
        validation_resolved.add(validation_image.resolve(strict=True))
    if not validation_resolved:
        raise RuntimeError('v6 验证集为空，无法建立真实路径隔离集合')

    for split in ('train', 'val'):
        counts: Counter[int] = Counter()
        total_boxes = 0
        image_paths = image_files(SOURCE / 'images' / split)
        unique_sources: dict[Path, tuple[Path, Path, list[tuple[int, float, float, float, float]]]] = {}
        for image_path in image_paths:
            label_path = SOURCE / 'labels' / split / f'{image_path.stem}.txt'
            if not label_path.is_file():
                raise FileNotFoundError(f'缺少标签: {label_path}')
            boxes = parse_label(label_path)
            source_resolved = image_path.resolve(strict=True)
            if not is_within(source_resolved, DATA_ROOT):
                raise ValueError(f'源图片不在数据盘: {source_resolved}')
            if split == 'train' and source_resolved in validation_resolved:
                excluded_train_val_overlap.append({
                    'train_file': str(image_path.relative_to(SOURCE)),
                    'resolved_source': str(source_resolved),
                    'reason': 'train_path_resolves_to_validation_source',
                })
                continue
            unique_sources.setdefault(source_resolved, (image_path, label_path, boxes))

            destination_image = DEST / 'images' / split / image_path.name
            destination_label = DEST / 'labels' / split / f'{image_path.stem}.txt'
            link_absolute(image_path, destination_image)
            link_absolute(label_path, destination_label)
            for class_id, *_ in boxes:
                counts[class_id] += 1
            total_boxes += len(boxes)

            if split != 'train':
                continue

            classes = {box[0] for box in boxes}
            extra_rules: list[tuple[str, int]] = []
            if MEAT_CAN in classes:
                extra_rules.append(('meat_can', MEAT_CAN_EXTRA_COPIES))
            egg_fraction = green_fraction(image_path, boxes)
            if egg_fraction is not None:
                if egg_fraction >= 0.30:
                    extra_rules.append(('green_egg', GREEN_EGG_EXTRA_COPIES))
                else:
                    extra_rules.append(('orange_egg', ORANGE_EGG_EXTRA_COPIES))
            if STRAWBERRY in classes:
                extra_rules.append(('strawberry', STRAWBERRY_EXTRA_COPIES))
            if classes.intersection(CONFUSION_CLASSES):
                extra_rules.append(('egg_confusion_context', CONFUSION_EXTRA_COPIES))

            for kind, copies in extra_rules:
                for copy_index in range(1, copies + 1):
                    suffix = f'__v9{kind.replace("_", "")}{copy_index:02d}'
                    copied_image = DEST / 'images' / split / f'{image_path.stem}{suffix}{image_path.suffix.lower()}'
                    copied_label = DEST / 'labels' / split / f'{image_path.stem}{suffix}.txt'
                    link_absolute(image_path, copied_image)
                    link_absolute(label_path, copied_label)
                    for class_id, *_ in boxes:
                        counts[class_id] += 1
                    total_boxes += len(boxes)
                    extras_by_kind[kind] += 1

        image_counts[split] = len(image_files(DEST / 'images' / split))
        box_counts[split] = total_boxes
        class_counts[split] = {str(class_id): int(counts[class_id]) for class_id in sorted(counts)}
        split_info[split] = {
            'source_images': len(image_paths),
            'unique_resolved_sources': len(unique_sources),
            'images': image_counts[split],
            'boxes': total_boxes,
        }

        if split == 'train':
            for source_resolved, (source_image, source_label, boxes) in sorted(unique_sources.items(), key=lambda item: str(item[0])):
                if not any(box[0] == MEAT_CAN for box in boxes):
                    continue
                for variant_index, target_hue in ((1, 52), (2, 76)):
                    marker = f'__v9greenmeatcan{variant_index:02d}'
                    synthetic_path = synthetic_dir / f'{source_image.stem}{marker}.png'
                    make_synthetic_green_variant(source_image, boxes, synthetic_path, target_hue)
                    destination_image = DEST / 'images' / 'train' / synthetic_path.name
                    destination_label = DEST / 'labels' / 'train' / f'{synthetic_path.stem}.txt'
                    link_absolute(synthetic_path, destination_image)
                    link_absolute(source_label, destination_label)
                    for class_id, *_ in boxes:
                        counts[class_id] += 1
                    total_boxes += len(boxes)
                    extras_by_kind['synthetic_green_meat_can'] += 1
                    manifests.append({
                        'kind': 'synthetic_green_meat_can',
                        'variant': variant_index,
                        'target_hue_opencv': target_hue,
                        'source_image': str(source_resolved),
                        'source_label': str(source_label.resolve(strict=True)),
                        'output_image': str(synthetic_path),
                        'label_class_ids': [box[0] for box in boxes],
                        'train_only': True,
                    })
            image_counts['train'] = len(image_files(DEST / 'images' / 'train'))
            box_counts['train'] = total_boxes
            class_counts['train'] = {str(class_id): int(counts[class_id]) for class_id in sorted(counts)}
            split_info['train'].update({'images': image_counts['train'], 'boxes': total_boxes})

    train_resolved = {path.resolve(strict=True) for path in image_files(DEST / 'images/train')}
    val_resolved = {path.resolve(strict=True) for path in image_files(DEST / 'images/val')}
    remaining_overlap = sorted(str(path) for path in train_resolved & val_resolved)
    if remaining_overlap:
        raise RuntimeError(f'训练集仍与验证集存在真实源路径重叠: {remaining_overlap[:5]}')

    if image_counts['val'] != 121 or box_counts['val'] != 174:
        raise RuntimeError(f'验证集必须保持 121 图/174 框，实际 {image_counts["val"]} 图/{box_counts["val"]} 框')
    for target in (MEAT_CAN, EGGS, STRAWBERRY):
        if class_counts['val'].get(str(target), 0) <= 0 or class_counts['train'].get(str(target), 0) <= 0:
            raise RuntimeError(f'目标类别 {target} 在 train/val 中不完整')

    dataset_yaml = {
        'path': str(DEST),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 40,
        'names': names,
    }
    (DEST / 'dataset.yaml').write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding='utf-8')

    synthetic_sheet_items: list[tuple[Path, list[tuple[int, float, float, float, float]], str]] = []
    for item in manifests:
        source_image = Path(item['source_image'])
        source_label = Path(item['source_label'])
        boxes = parse_label(source_label)
        synthetic_sheet_items.append((Path(item['output_image']), boxes, f'{source_image.stem} green-v{item["variant"]}'))
    make_synthetic_sheet(synthetic_sheet_items, DEST / 'synthetic_green_meat_can_contact_sheet.jpg')

    (DEST / 'synthetic_green_meat_can_manifest.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in manifests), encoding='utf-8'
    )
    (DEST / 'overlay_manifest.json').write_text(json.dumps({
        'source_dataset': str(SOURCE),
        'target_dataset': str(DEST),
        'extras_by_kind': dict(sorted(extras_by_kind.items())),
        'meat_can_extra_copies_per_source_image': MEAT_CAN_EXTRA_COPIES,
        'green_egg_extra_copies_per_source_image': GREEN_EGG_EXTRA_COPIES,
        'orange_egg_extra_copies_per_source_image': ORANGE_EGG_EXTRA_COPIES,
        'strawberry_extra_copies_per_source_image': STRAWBERRY_EXTRA_COPIES,
        'egg_confusion_extra_copies_per_source_image': CONFUSION_EXTRA_COPIES,
        'synthetic_green_meat_can_variants_per_unique_source': SYNTHETIC_GREEN_VARIANTS,
        'validation_one_to_one': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    source_audit = json.loads((SOURCE / 'audit_summary.json').read_text(encoding='utf-8'))
    audit = {
        'schema_version': 1,
        'created_at': now_iso(),
        'dataset_root': str(DEST),
        'source_dataset': str(SOURCE),
        # Promoted to true only after the independent materialization audit
        # verifies path disjointness, symlink targets, labels, and decoding.
        'training_ready': False,
        'training_started': False,
        'materialization': 'absolute_symlink',
        'checks': {
            'source_v6_complete': True,
            'all_dataset_links_absolute': True,
            'all_link_targets_inside_data_volume': True,
            'validation_one_to_one_unchanged': True,
            'validation_not_in_train': True,
            'validation_not_oversampled': True,
            'train_val_resolved_disjoint': True,
            'train_only_hard_negative_augmentation': True,
            'synthetic_variants_from_train_only': True,
            'meat_can_labels_preserved_on_synthetic_variants': True,
        },
        'images': image_counts,
        'source_images': {split: split_info[split]['source_images'] for split in ('train', 'val')},
        'unique_resolved_sources': {split: split_info[split]['unique_resolved_sources'] for split in ('train', 'val')},
        'boxes': sum(box_counts.values()),
        'class_counts': class_counts,
        'front10_counts': {
            'train_boxes': sum(class_counts['train'].get(str(index), 0) for index in range(10)),
            'val_boxes': sum(class_counts['val'].get(str(index), 0) for index in range(10)),
        },
        'extra_images': dict(sorted(extras_by_kind.items())),
        'excluded_train_val_overlap_count': len(excluded_train_val_overlap),
        'excluded_train_val_overlap': excluded_train_val_overlap,
        'source_summary': source_audit.get('source_summary', {}),
    }
    (DEST / 'audit_summary.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    for name in ('merge_manifest.jsonl', 'new_bboxes.csv', 'class_counts.csv', 'README.md'):
        source_file = SOURCE / name
        if source_file.is_file():
            link_absolute(source_file, DEST / f'v6_{name}')

    (DEST / 'README.md').write_text(
        '# v9 训练集视图\n\n'
        '该目录由 `build_track1_training_v9_hardneg.py` 从 v6 创建，只新增内容，不修改 v6/v7。\n\n'
        '- 验证集与 v6 一对一相同，不复制到 train。\n'
        f'- 已按解析后的真实源路径排除 {len(excluded_train_val_overlap)} 个 train/val 重叠别名。\n'
        '- train 中加强 10-meat_can 困难负样本、绿色正面 Eggs、Strawberry 和少量 Eggs 混淆类。\n'
        '- `assets/synthetic_green_meat_can` 是由 train 肉罐头图片颜色变换生成的合成困难负样本，标签仍为官方第 10 类。\n'
        '- 生产模型未替换。\n',
        encoding='utf-8',
    )
    (DEST / '.materialization_pending').write_text(json.dumps({
        'created_at': now_iso(),
        'train_images': image_counts['train'],
        'val_images': image_counts['val'],
        'boxes': sum(box_counts.values()),
        'excluded_train_val_overlap_count': len(excluded_train_val_overlap),
        'train_val_resolved_overlap': 0,
        'next_step': 'run audit_track1_dataset_materialization.py',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    result = {
        'dataset': str(DEST),
        'source': str(SOURCE),
        'images': image_counts,
        'boxes': box_counts,
        'class_counts': class_counts,
        'front10_counts': audit['front10_counts'],
        'extras_by_kind': dict(sorted(extras_by_kind.items())),
        'excluded_train_val_overlap_count': len(excluded_train_val_overlap),
        'synthetic_manifest': str(DEST / 'synthetic_green_meat_can_manifest.jsonl'),
        'contact_sheet': str(DEST / 'synthetic_green_meat_can_contact_sheet.jpg'),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
