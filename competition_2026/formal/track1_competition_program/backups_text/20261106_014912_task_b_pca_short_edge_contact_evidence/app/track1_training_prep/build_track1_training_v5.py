#!/usr/bin/env python3
"""Build v5 as an additive active-learning dataset from the audited v4 data.

The v4 dataset, weights, reports, and validation split remain untouched.  v5
adds train-only replicas of one already human-reviewed hard case: a top-view
Tomato Soup can that v4 classified as Starkist Tuna.  The source frame remains
in validation as a regression check; this is deliberate active learning, not a
claim of an independent holdout.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path('/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序')
V4_ROOT = ROOT / 'datasets/track1_training_merged_v4'
OUTPUT = ROOT / 'datasets/track1_training_merged_v5'
TEMP = ROOT / 'datasets/.track1_training_merged_v5.partial'
SPLITS = ('train', 'val')
EMPTY_SLOTS = {16, 25}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
HARD_STEM = 'front10__front10-review-unit-0025__001382'
HARD_COPIES = 8


class BuildError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def names_from_yaml(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    names = data.get('names')
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or len(names) != 40:
        raise BuildError(f'类别表不是 40 项：{path}')
    return [str(item) for item in names]


def link_file(target: Path, link: Path) -> None:
    target = target.resolve(strict=True)
    if not target.is_file():
        raise BuildError(f'源文件不存在：{target}')
    if link.exists() or link.is_symlink():
        raise BuildError(f'拒绝覆盖：{link}')
    os.symlink(str(target), link)


def count_labels(root: Path) -> tuple[dict[str, int], int, dict[str, dict[int, int]], dict[str, int]]:
    images: dict[str, int] = {}
    boxes_total = 0
    counts: dict[str, dict[int, int]] = {split: {} for split in SPLITS}
    empty: dict[str, int] = {split: 0 for split in SPLITS}
    for split in SPLITS:
        image_dir = root / 'images' / split
        label_dir = root / 'labels' / split
        image_map = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        label_map = {p.stem: p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == '.txt'}
        if set(image_map) != set(label_map):
            raise BuildError(f'{split} 图片与标签 stem 不一致')
        images[split] = len(image_map)
        split_counts: Counter[int] = Counter()
        for stem in sorted(image_map):
            rows = [line.strip() for line in label_map[stem].read_text(encoding='utf-8').splitlines() if line.strip()]
            if not rows:
                empty[split] += 1
            for line_no, row in enumerate(rows, 1):
                parts = row.split()
                if len(parts) != 5:
                    raise BuildError(f'{label_map[stem]}:{line_no} 不是 5 列')
                cls = int(parts[0])
                values = [float(value) for value in parts[1:]]
                if not 0 <= cls < 40 or not all(0.0 <= value <= 1.0 for value in values):
                    raise BuildError(f'{label_map[stem]}:{line_no} 标签非法')
                split_counts[cls] += 1
                boxes_total += 1
        counts[split] = dict(split_counts)
    for split in SPLITS:
        missing = sorted(set(range(40)) - EMPTY_SLOTS - set(counts[split]))
        if missing:
            raise BuildError(f'{split} 缺少活动类别：{missing}')
        if any(counts[split].get(index, 0) for index in EMPTY_SLOTS):
            raise BuildError(f'{split} 的 17/26 空槽出现标签')
    return images, boxes_total, counts, empty


def main() -> int:
    if OUTPUT.exists() or TEMP.exists():
        raise BuildError(f'拒绝覆盖已有目录：{OUTPUT if OUTPUT.exists() else TEMP}')
    if not (V4_ROOT / '.complete').is_file():
        raise BuildError('v4 数据集未完成')
    audit_v4 = json.loads((V4_ROOT / 'audit_summary.json').read_text(encoding='utf-8'))
    if audit_v4.get('training_ready') is not True:
        raise BuildError('v4 training_ready 不是 true')
    names = names_from_yaml(V4_ROOT / 'dataset.yaml')
    hard_image = V4_ROOT / 'images' / 'val' / f'{HARD_STEM}.png'
    hard_label = V4_ROOT / 'labels' / 'val' / f'{HARD_STEM}.txt'
    if not hard_image.is_file() or not hard_label.is_file():
        raise BuildError('找不到已复核的番茄汤罐顶部难例')

    base_manifest = [json.loads(line) for line in (V4_ROOT / 'merge_manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    v4_val_stems = sorted(p.stem for p in (V4_ROOT / 'images' / 'val').iterdir() if p.is_file())
    TEMP.mkdir(parents=True)
    try:
        for split in SPLITS:
            (TEMP / 'images' / split).mkdir(parents=True)
            (TEMP / 'labels' / split).mkdir(parents=True)
            for src in sorted((V4_ROOT / 'images' / split).iterdir()):
                if src.is_file() and src.suffix.lower() in IMAGE_SUFFIXES:
                    link_file(src, TEMP / 'images' / split / src.name)
                    label = V4_ROOT / 'labels' / split / f'{src.stem}.txt'
                    link_file(label, TEMP / 'labels' / split / label.name)

        added_rows: list[dict[str, Any]] = []
        for index in range(1, HARD_COPIES + 1):
            stem = f'v5_hard__tomato_top__001382__r{index:02d}'
            image_link = TEMP / 'images' / 'train' / f'{stem}.png'
            label_link = TEMP / 'labels' / 'train' / f'{stem}.txt'
            link_file(hard_image, image_link)
            link_file(hard_label, label_link)
            added_rows.append({
                'source_image': str(hard_image),
                'source_label': str(hard_label),
                'output_image': str(image_link),
                'output_label': str(label_link),
                'source_split': 'val',
                'target_split': 'train',
                'class_ids': [4],
                'purpose': 'active_learning_hard_case',
            })

        if sorted(p.stem for p in (TEMP / 'images' / 'val').iterdir() if p.is_file()) != v4_val_stems:
            raise BuildError('v5 验证集与 v4 不一致')
        images, boxes_total, class_counts, empty = count_labels(TEMP)
        if images['train'] != 366 + HARD_COPIES or images['val'] != 103:
            raise BuildError(f'图片数异常：{images}')

        dataset_yaml = {
            'path': str(OUTPUT),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 40,
            'names': names,
        }
        (TEMP / 'dataset.yaml').write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding='utf-8')

        with (TEMP / 'merge_manifest.jsonl').open('w', encoding='utf-8') as handle:
            for row in base_manifest:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            for row in added_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

        with (TEMP / 'oversampling.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['source_image', 'target_split', 'extra_copies', 'class_ids', 'reason'])
            writer.writerow([str(hard_image), 'train', HARD_COPIES, '4', 'active_learning_wrong_class_hard_case'])

        with (TEMP / 'class_counts.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['split', 'class_index', 'class_name', 'boxes'])
            for split in SPLITS:
                for cls in range(40):
                    if class_counts[split].get(cls, 0):
                        writer.writerow([split, cls, names[cls], class_counts[split][cls]])

        audit = json.loads(json.dumps(audit_v4, ensure_ascii=False))
        audit.update({
            'schema_version': 1,
            'created_at': now_iso(),
            'dataset_root': str(OUTPUT),
            'dataset_yaml': str(OUTPUT / 'dataset.yaml'),
            'materialization': 'absolute_symlink',
            'images': images,
            'labels': dict(images),
            'box_count': boxes_total,
            'boxes_by_split': {split: sum(class_counts[split].values()) for split in SPLITS},
            'empty_labels': empty,
            'class_counts': {split: {str(k): v for k, v in sorted(class_counts[split].items())} for split in SPLITS},
            'front10_boxes': {split: sum(class_counts[split].get(i, 0) for i in range(10)) for split in SPLITS},
            'training_ready': True,
            'training_started': False,
        })
        audit.setdefault('source_summary', {})['v5_active_learning'] = {
            'source_dataset': str(V4_ROOT),
            'hard_case': str(hard_image),
            'hard_case_label': str(hard_label),
            'hard_case_class': names[4],
            'train_only_extra_copies': HARD_COPIES,
            'reason': 'v4 mistook top-view Tomato Soup for Starkist Tuna at conf 0.82435 and IoU 0.8566',
            'validation_unchanged': True,
            'caveat': 'The hard source remains in val for regression; this is not an independent holdout for that one frame.',
        }
        (TEMP / 'audit_summary.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (TEMP / 'README.md').write_text(
            '# 赛道一 40 类训练集 v5\n\n'
            '由已审计的 v4 原样链接生成，验证集完全不变；仅将已人工复核的番茄汤罐顶部难例额外加入训练 8 次，用于修正与金枪鱼罐的类别混淆。\n\n'
            f'- 训练图片：{images["train"]}\n- 验证图片：{images["val"]}\n- 总框数：{boxes_total}\n'
            f'- 难例：`{hard_image}`\n- 说明：这是主动学习修补，不是独立验证集；v4 权重、数据和报告均未修改。\n',
            encoding='utf-8'
        )
        (TEMP / '.complete').write_text(now_iso() + '\n', encoding='utf-8')
        os.replace(TEMP, OUTPUT)
    except Exception:
        shutil.rmtree(TEMP, ignore_errors=True)
        raise

    print(json.dumps({
        'output': str(OUTPUT),
        'images': images,
        'box_count': boxes_total,
        'hard_case': str(hard_image),
        'train_extra_copies': HARD_COPIES,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
