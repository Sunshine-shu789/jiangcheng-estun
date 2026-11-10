#!/usr/bin/env python3
"""Independent full audit for the Track-1 v10 40-class augmented dataset.

This script does not trust the builder's summary for file integrity. It scans
every image and label, verifies manifests and hashes, checks split isolation,
and writes a machine-readable report before training is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import yaml

DATA_ROOT = Path('/media/sunshine/新加卷').resolve()
PROJECT_ROOT = DATA_ROOT / '埃斯顿比赛_2026/赛道一比赛程序'
DEFAULT_DATASET = PROJECT_ROOT / 'datasets/track1_training_merged_v10_all_multiclass_augmented_20261031'
DEFAULT_REPORT = PROJECT_ROOT / 'reports/track1_v10_dataset_independent_audit_latest.json'
SPLITS = ('train', 'val')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
EXPECTED_IMAGES = {'train': 4248, 'val': 170}
EXPECTED_BOXES = {'train': 9924, 'val': 459}
EXPECTED_SOURCE = {'train': 1062, 'val': 170}
EXPECTED_AUGMENTED = 3186
# One already-audited background-only train source plus its 3 offline variants,
# and one background-only validation source. These are intentional negatives.
EXPECTED_EMPTY_LABELS = {'train': 4, 'val': 1}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'JSON 顶层不是对象：{path}')
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f'{path}:{line_number} JSON 顶层不是对象')
        records.append(value)
    return records


def image_entries(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def label_entries(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == '.txt'
    }


def parse_label(path: Path, context: str, errors: list[str]) -> tuple[list[tuple[int, float, float, float, float]], bool]:
    rows: list[tuple[int, float, float, float, float]] = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except Exception as exc:
        errors.append(f'{context}: 标签无法读取：{type(exc).__name__}: {exc}')
        return rows, False
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f'{context}:{line_number}: 不是 YOLO 五列')
            continue
        try:
            class_index = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f'{context}:{line_number}: 含非数字')
            continue
        if not 0 <= class_index < 40:
            errors.append(f'{context}:{line_number}: 类别越界 {class_index}')
            continue
        values = (x, y, width, height)
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f'{context}:{line_number}: 坐标不在 0..1')
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f'{context}:{line_number}: 宽高必须大于 0')
            continue
        epsilon = 1e-6
        if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon:
            errors.append(f'{context}:{line_number}: 横向越界')
            continue
        if y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
            errors.append(f'{context}:{line_number}: 纵向越界')
            continue
        if width >= 0.98 and height >= 0.98:
            errors.append(f'{context}:{line_number}: 疑似整图伪框')
            continue
        rows.append((class_index, x, y, width, height))
    return rows, bool(lines)


def resolve_output(dataset: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else dataset / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='独立审计 Track-1 v10 全量增强数据集')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    dataset = args.dataset.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks[name] = {'ok': bool(ok), 'detail': detail}
        if not ok:
            errors.append(f'{name}: {detail}')

    check('dataset_on_data_volume', is_within(dataset, DATA_ROOT), str(dataset))
    check('dataset_exists', dataset.is_dir(), str(dataset))
    if not dataset.is_dir():
        raise SystemExit(f'数据集不存在：{dataset}')

    dataset_yaml = dataset / 'dataset.yaml'
    audit_path = dataset / 'audit_summary.json'
    check('complete_marker', (dataset / '.complete').is_file(), str(dataset / '.complete'))
    check('dataset_yaml', dataset_yaml.is_file(), str(dataset_yaml))
    check('audit_summary', audit_path.is_file(), str(audit_path))
    if not dataset_yaml.is_file() or not audit_path.is_file():
        raise SystemExit('数据集基础文件不完整')

    payload = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise SystemExit('dataset.yaml 顶层不是对象')
    names = payload.get('names')
    if isinstance(names, dict):
        names = [names.get(index) for index in range(len(names))]
    names = [str(item) for item in names] if isinstance(names, list) else []
    check('dataset_root_yaml', Path(str(payload.get('path', ''))).resolve() == dataset, str(payload.get('path')))
    check('class_count_40', int(payload.get('nc', -1)) == 40 and len(names) == 40, {'nc': payload.get('nc'), 'names': len(names)})

    audit = read_json(audit_path)
    audit_checks = audit.get('checks', {})
    check('builder_training_ready', audit.get('training_ready') is True, audit.get('training_ready'))
    check('builder_all_checks_true', isinstance(audit_checks, dict) and all(value is True for value in audit_checks.values()), len(audit_checks) if isinstance(audit_checks, dict) else None)
    check('build_did_not_train', audit.get('training_started') is False, audit.get('training_started'))
    check('build_did_not_modify_model', audit.get('model_modified') is False, audit.get('model_modified'))
    check('build_did_not_modify_ab', audit.get('task_a_b_modified') is False, audit.get('task_a_b_modified'))

    reconciliation = audit.get('source_split_reconciliation', {})
    reviewed_labels = audit.get('reviewed_overlap_labeling', {})
    check('split_reconciliation_entries_6', reconciliation.get('entries_changed') == 6, reconciliation.get('entries_changed'))
    check('split_reconciliation_groups_6', sum(len(items) for items in reconciliation.get('group_changes', {}).values()) == 6, reconciliation.get('group_changes'))
    check('split_reconciliation_no_cross', reconciliation.get('final_group_cross_split_count') == 0, reconciliation.get('final_group_cross_split_count'))
    check('reviewed_overlap_groups_99', reviewed_labels.get('reviewed_label_authority_groups') == 99, reviewed_labels.get('reviewed_label_authority_groups'))
    check('reviewed_prior_entries_166', reviewed_labels.get('prior_entries_relabeled') == 166, reviewed_labels.get('prior_entries_relabeled'))
    check('validation_dedup_entries_22', reviewed_labels.get('validation_entries_dropped') == 22, reviewed_labels.get('validation_entries_dropped'))
    check('validation_dedup_records_22', len(reviewed_labels.get('validation_deduplication', [])) == 22, len(reviewed_labels.get('validation_deduplication', [])))
    check('physical_groups_790', audit.get('physical_groups_checked') == 790, audit.get('physical_groups_checked'))

    images: dict[str, dict[str, Path]] = {}
    labels: dict[str, dict[str, Path]] = {}
    image_hashes: dict[str, dict[str, str]] = {split: {} for split in SPLITS}
    class_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    class_image_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    empty_labels: dict[str, int] = {split: 0 for split in SPLITS}
    decoded = 0
    for split in SPLITS:
        images[split] = image_entries(dataset / 'images' / split)
        labels[split] = label_entries(dataset / 'labels' / split)
        check(f'{split}_image_count', len(images[split]) == EXPECTED_IMAGES[split], {'expected': EXPECTED_IMAGES[split], 'actual': len(images[split])})
        check(f'{split}_label_count', len(labels[split]) == EXPECTED_IMAGES[split], {'expected': EXPECTED_IMAGES[split], 'actual': len(labels[split])})
        missing_labels = sorted(set(images[split]) - set(labels[split]))
        extra_labels = sorted(set(labels[split]) - set(images[split]))
        check(f'{split}_image_label_pairing', not missing_labels and not extra_labels, {'missing_labels': missing_labels[:10], 'extra_labels': extra_labels[:10]})

        for stem in sorted(set(images[split]) & set(labels[split])):
            image_path = images[split][stem]
            label_path = labels[split][stem]
            check(f'{split}_augment_name_policy_{stem}', split == 'train' or not stem.startswith('aug__'), stem)
            if not is_within(image_path, dataset) or not is_within(label_path, dataset):
                errors.append(f'{split}/{stem}: 文件位于数据集目录之外')
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2:
                errors.append(f'{split}/{stem}: 图片无法解码或尺寸异常')
            else:
                decoded += 1
            image_hashes[split][stem] = sha256_file(image_path)
            rows, physically_nonempty = parse_label(label_path, f'{split}/{stem}', errors)
            if not rows:
                empty_labels[split] += 1
            for class_index, *_ in rows:
                class_counts[split][class_index] += 1
                class_image_counts[split][class_index] += 1

        present = [index for index in range(40) if class_counts[split][index] > 0]
        missing = [index for index in range(40) if class_counts[split][index] <= 0]
        check(f'{split}_all_40_classes', len(present) == 40, {'covered': len(present), 'missing': missing})
        check(f'{split}_no_class_99', class_counts[split][99] == 0, class_counts[split][99])
        check(f'{split}_box_count', sum(class_counts[split].values()) == EXPECTED_BOXES[split], {'expected': EXPECTED_BOXES[split], 'actual': sum(class_counts[split].values())})

    check('all_images_decode', decoded == sum(EXPECTED_IMAGES.values()), {'decoded': decoded, 'expected': sum(EXPECTED_IMAGES.values())})
    check('empty_label_counts', empty_labels == EXPECTED_EMPTY_LABELS, {'expected': EXPECTED_EMPTY_LABELS, 'actual': empty_labels})
    val_hash_values = list(image_hashes['val'].values())
    train_hash_set = set(image_hashes['train'].values())
    val_hash_set = set(val_hash_values)
    check('validation_hashes_unique', len(val_hash_values) == len(val_hash_set), {'images': len(val_hash_values), 'unique': len(val_hash_set)})
    check('train_val_exact_disjoint', not bool(train_hash_set & val_hash_set), len(train_hash_set & val_hash_set))
    check('no_augmented_validation', not any(stem.startswith('aug__') for stem in images['val']), [stem for stem in images['val'] if stem.startswith('aug__')][:10])

    source_manifest_path = dataset / 'source_manifest.jsonl'
    augmentation_manifest_path = dataset / 'augmentation_manifest.jsonl'
    check('source_manifest', source_manifest_path.is_file(), str(source_manifest_path))
    check('augmentation_manifest', augmentation_manifest_path.is_file(), str(augmentation_manifest_path))
    source_records = read_jsonl(source_manifest_path)
    augmentation_records = read_jsonl(augmentation_manifest_path)
    check('source_manifest_count', len(source_records) == 1232, len(source_records))
    check('augmentation_manifest_count', len(augmentation_records) == EXPECTED_AUGMENTED, len(augmentation_records))

    source_by_kind: Counter[str] = Counter()
    source_by_split: Counter[str] = Counter()
    source_outputs: set[str] = set()
    source_box_total = 0
    multi_box_total = 0
    multi_source_images: set[str] = set()
    source_output_split: dict[str, str] = {}
    for index, record in enumerate(source_records, 1):
        source_kind = str(record.get('source_kind'))
        split = str(record.get('split'))
        source_by_kind[source_kind] += 1
        source_by_split[split] += 1
        source_box_total += int(record.get('box_count', -1))
        if source_kind == 'multiobject_v4':
            multi_box_total += int(record.get('box_count', -1))
            source_value = record.get('source_image')
            if source_value:
                multi_source_images.add(str(Path(str(source_value)).resolve()))
            if not str(record.get('label_authority', '')).startswith('multiobject_v4:'):
                errors.append(f'source_manifest:{index}: 多物品记录不是 v4 人工标签权威')
        output_image = resolve_output(dataset, record.get('output_image'))
        output_label = resolve_output(dataset, record.get('output_label'))
        relative = str(output_image.relative_to(dataset)) if is_within(output_image, dataset) else str(output_image)
        if relative in source_outputs:
            errors.append(f'source_manifest:{index}: 输出图片重复 {relative}')
        source_outputs.add(relative)
        source_output_split[relative] = split
        if not output_image.is_file() or not output_label.is_file():
            errors.append(f'source_manifest:{index}: 输出图片或标签不存在')
            continue
        if sha256_file(output_image) != str(record.get('image_sha256')):
            errors.append(f'source_manifest:{index}: 输出图片哈希不符')
        if output_image.stem not in images.get(split, {}):
            errors.append(f'source_manifest:{index}: 图片未纳入对应 split')

    actual_source_outputs = set()
    for split in SPLITS:
        for stem, path in images[split].items():
            if not stem.startswith('aug__'):
                actual_source_outputs.add(str(path.relative_to(dataset)))
    check('source_outputs_unique', len(source_outputs) == len(source_records), {'records': len(source_records), 'unique': len(source_outputs)})
    check('source_outputs_match_filesystem', source_outputs == actual_source_outputs, {'manifest': len(source_outputs), 'filesystem': len(actual_source_outputs)})
    check('source_counts', source_by_split == Counter(EXPECTED_SOURCE), dict(source_by_split))
    check('source_v9_entries_911', source_by_kind['prior_v9'] == 911, source_by_kind['prior_v9'])
    check('source_v4_entries_321', source_by_kind['multiobject_v4'] == 321, source_by_kind['multiobject_v4'])
    check('source_boxes_total_2940', source_box_total == 2940, source_box_total)
    check('source_v4_boxes_1646', multi_box_total == 1646, multi_box_total)
    check('source_v4_unique_images_321', len(multi_source_images) == 321, len(multi_source_images))

    augmentation_outputs: set[str] = set()
    augmented_by_source: Counter[str] = Counter()
    augmented_box_total = 0
    augmented_hashes: set[str] = set()
    for index, record in enumerate(augmentation_records, 1):
        split = str(record.get('split'))
        variant = int(record.get('variant', -1))
        source_output = str(record.get('source_output_image'))
        augmented_by_source[source_output] += 1
        output_image = resolve_output(dataset, record.get('output_image'))
        output_label = resolve_output(dataset, record.get('output_label'))
        relative = str(output_image.relative_to(dataset)) if is_within(output_image, dataset) else str(output_image)
        if relative in augmentation_outputs:
            errors.append(f'augmentation_manifest:{index}: 输出图片重复 {relative}')
        augmentation_outputs.add(relative)
        if split != 'train' or variant not in (1, 2, 3):
            errors.append(f'augmentation_manifest:{index}: split/变体非法 {split}/{variant}')
        if not output_image.is_file() or not output_label.is_file():
            errors.append(f'augmentation_manifest:{index}: 增强图片或标签不存在')
            continue
        if not output_image.stem.startswith('aug__v'):
            errors.append(f'augmentation_manifest:{index}: 文件名不是增强命名')
        output_hash = sha256_file(output_image)
        augmented_hashes.add(output_hash)
        if output_hash != str(record.get('output_sha256')):
            errors.append(f'augmentation_manifest:{index}: 增强图片哈希不符')
        augmented_box_total += int(record.get('output_box_count', -1))

    actual_augmented_outputs = {
        str(path.relative_to(dataset))
        for stem, path in images['train'].items()
        if stem.startswith('aug__')
    }
    check('augmented_outputs_unique', len(augmentation_outputs) == len(augmentation_records), {'records': len(augmentation_records), 'unique': len(augmentation_outputs)})
    check('augmented_outputs_match_filesystem', augmentation_outputs == actual_augmented_outputs, {'manifest': len(augmentation_outputs), 'filesystem': len(actual_augmented_outputs)})
    check('three_variants_per_source', len(augmented_by_source) == EXPECTED_SOURCE['train'] and set(augmented_by_source.values()) == {3}, {'sources': len(augmented_by_source), 'variant_counts': sorted(set(augmented_by_source.values()))})
    check('augmented_box_total_7443', augmented_box_total == 7443, augmented_box_total)
    check('augmented_hashes_unique', len(augmented_hashes) == len(augmentation_records), {'records': len(augmentation_records), 'unique_hashes': len(augmented_hashes)})

    if isinstance(names, list):
        expected_class_counts = audit.get('class_boxes', {})
        expected_class_images = audit.get('class_images', {})
        for split in SPLITS:
            expected_boxes = {int(key): int(value) for key, value in expected_class_counts.get(split, {}).items()}
            expected_images = {int(key): int(value) for key, value in expected_class_images.get(split, {}).items()}
            if expected_boxes != dict(class_counts[split]) or expected_images != dict(class_image_counts[split]):
                errors.append(f'{split}: 重新统计的类别分布与构建审计不一致')

    report = {
        'schema_version': 1,
        'created_at': now_iso(),
        'dataset_root': str(dataset),
        'dataset_yaml_sha256': sha256_file(dataset_yaml),
        'audit_summary_sha256': sha256_file(audit_path),
        'summary': {
            'images': {split: len(images[split]) for split in SPLITS},
            'boxes': {split: sum(class_counts[split].values()) for split in SPLITS},
            'source_images': dict(source_by_split),
            'source_boxes': source_box_total,
            'augmented_images': len(augmentation_records),
            'augmented_boxes': augmented_box_total,
            'decoded_images': decoded,
            'empty_labels': empty_labels,
            'unique_hashes': {
                'train': len(train_hash_set),
                'val': len(val_hash_set),
            },
            'reviewed_overlap_groups': reviewed_labels.get('reviewed_label_authority_groups'),
            'validation_deduplicated_entries': reviewed_labels.get('validation_entries_dropped'),
            'split_reconciled_entries': reconciliation.get('entries_changed'),
            'physical_groups_checked': audit.get('physical_groups_checked'),
        },
        'checks': checks,
        'warnings': warnings,
        'errors': errors,
        'passed': not errors,
        'training_ready': not errors,
        'training_started': False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_name(report_path.name + f'.tmp.{os.getpid()}')
    temp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp_path, report_path)

    print(json.dumps({
        'ok': not errors,
        'dataset_root': str(dataset),
        'report': str(report_path),
        'summary': report['summary'],
        'failed_checks': [name for name, item in checks.items() if not item['ok']],
        'error_count': len(errors),
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
