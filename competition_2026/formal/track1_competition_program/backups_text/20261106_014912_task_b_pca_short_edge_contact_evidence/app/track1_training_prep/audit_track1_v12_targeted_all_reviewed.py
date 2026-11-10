#!/usr/bin/env python3
"""Independent audit for the Track-1 v12 targeted reviewed dataset.

The builder summary is treated as an untrusted hint. This script reconstructs
split isolation from source provenance, scans every image and label, validates
all augmentation counts and hashes, and writes a machine-readable report that
the guarded training preflight can require.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import yaml


PROJECT_ROOT = Path(
    '/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序'
).resolve()
DEFAULT_DATASET = PROJECT_ROOT / (
    'datasets/track1_training_merged_v12_targeted_all_reviewed_20261101'
)
DEFAULT_REPORT = PROJECT_ROOT / (
    'reports/track1_v12_dataset_independent_audit_latest.json'
)
DEFAULT_REVIEW_CSV = PROJECT_ROOT / (
    'reports/track1_v11_offline_acceptance_20261101/manual_review_multiobject.csv'
)
DEFAULT_V4_GROUPS = PROJECT_ROOT / (
    'datasets/track1_all_history_multi_object_review_20261031_v4/'
    'manifests/review_groups.csv'
)
DEFAULT_V6_MANIFEST = PROJECT_ROOT / (
    'datasets/track1_training_merged_v9_hardneg/v6_merge_manifest.jsonl'
)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = ('train', 'val')
EXPECTED_NAMES = [
    '1-Cheez-it',
    '2-Starkist_Tuna',
    '3-Scissors',
    '4-Frenchs_Mustard',
    '5-Tomato_Soup',
    '6-Foam_Brick',
    '7-Clamp',
    '8-Plastic_Banana',
    '9-Mug',
    '10-meat_can',
    '11-Padlock',
    '12-Baseball',
    '13-Bowl',
    '14-Sleeve',
    '15-Coca_cola_half',
    '16-ICRA_Duckie',
    '17-Eggs_Plush',
    '18-Dice',
    '19-Elmers_Glue',
    '20-Scotch_Tape',
    '21-Haribo_Golden_Bear',
    '22-E27_Adapter',
    '23-GU24_Adapter',
    '24-Spoon',
    '25-Flat_Screwdriver',
    '26-Plastic_Strawberry',
    '27-Coca_cola_full',
    '28-Kong_Duck_Dog_Toy',
    '29-Paper_Towels',
    '30-Stabilo_OHPen',
    '31-Plastic_White_Cup',
    '32-Wine_Glass',
    '33-Key',
    '34-Nail',
    '35-Book',
    '36-Adjustable_Wrench',
    '37-T-shirt',
    '38-Jumbo_Pencil_Cup',
    '39-Glove',
    '40-Pringles',
]
EXPECTED_SOURCE = {'train': 1062, 'val': 170}
EXPECTED_SOURCE_KIND_SPLIT = {
    ('prior_v9', 'train'): 813,
    ('prior_v9', 'val'): 98,
    ('multiobject_v4', 'train'): 249,
    ('multiobject_v4', 'val'): 72,
}
EXPECTED_IMAGES = {'train': 5180, 'val': 170}
EXPECTED_BOXES = {'train': 13230, 'val': 499}
EXPECTED_EMPTY = {'train': 4, 'val': 1}
EXPECTED_AUGMENTATION = {'generic': 3186, 'focus': 890, 'issue': 42}
EXPECTED_SAFE_CANDIDATES = 173
EXPECTED_REVIEW = {'pass': 58, 'issue': 14}
FOCUS_INDICES = {
    2, 5, 6, 13, 14, 17, 18, 22, 23, 26, 29, 30, 31, 32, 35, 36,
}
FRAME_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')
UNIT_RE = re.compile(r'(?:front10-review-unit|split-unit|all-history-multi)-\d+')
BURST_RE = re.compile(r'burst:([^|;]+)')
INTXLNK_MAGIC = b'IntxLNK\x01'


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON 顶层不是对象：{path}')
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f'{path}:{line_number} JSON 顶层不是对象')
        rows.append(value)
    return rows


def read_intxlnk_payload(path: Path) -> bytes:
    current = path.absolute()
    seen: set[Path] = set()
    for _ in range(16):
        resolved = current.resolve(strict=True)
        if resolved in seen:
            raise ValueError(f'IntxLNK 链接循环：{resolved}')
        seen.add(resolved)
        payload = resolved.read_bytes()
        if not payload.startswith(INTXLNK_MAGIC):
            return payload
        target_text = payload[len(INTXLNK_MAGIC):].decode('utf-16le').rstrip('\x00')
        target = Path(target_text)
        if not target.is_absolute():
            target = resolved.parent / target
        current = target.absolute()
    raise ValueError(f'IntxLNK 链接层级过深：{path}')


def read_jsonl_intxlnk(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        read_intxlnk_payload(path).decode('utf-8').splitlines(), 1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f'{path}:{line_number} JSON 顶层不是对象')
        rows.append(value)
    return rows


def io_entries(directory: Path) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    errors: list[str] = []
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            if path.stem in images:
                errors.append(f'同 stem 多张图片：{directory}/{path.stem}')
            images[path.stem] = path
        elif suffix == '.txt':
            if path.stem in labels:
                errors.append(f'同 stem 多个标签：{directory}/{path.stem}')
            labels[path.stem] = path
    return images, labels, errors


def parse_label(path: Path, context: str, errors: list[str]) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except Exception as exc:
        errors.append(f'{context}: 标签读取失败：{type(exc).__name__}: {exc}')
        return boxes
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
        values = (x, y, width, height)
        if not 0 <= class_index < 40:
            errors.append(f'{context}:{line_number}: 类别越界 {class_index}')
            continue
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f'{context}:{line_number}: 坐标不在 0..1')
            continue
        if width <= 0 or height <= 0:
            errors.append(f'{context}:{line_number}: 宽高非正')
            continue
        eps = 1e-6
        if x - width / 2 < -eps or x + width / 2 > 1 + eps:
            errors.append(f'{context}:{line_number}: 横向越界')
            continue
        if y - height / 2 < -eps or y + height / 2 > 1 + eps:
            errors.append(f'{context}:{line_number}: 纵向越界')
            continue
        boxes.append((class_index, x, y, width, height))
    return boxes


def boxes_match(
    actual: list[tuple[int, float, float, float, float]],
    expected: Any,
) -> bool:
    if not isinstance(expected, list) or len(actual) != len(expected):
        return False
    for actual_box, expected_box in zip(actual, expected):
        if not isinstance(expected_box, dict):
            return False
        if int(actual_box[0]) != int(expected_box.get('class_index', -1)):
            return False
        values = (
            float(expected_box.get('x', -1)),
            float(expected_box.get('y', -1)),
            float(expected_box.get('width', -1)),
            float(expected_box.get('height', -1)),
        )
        if any(abs(left - right) > 2e-6 for left, right in zip(actual_box[1:], values)):
            return False
    return True


def load_review_csv(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    names = [str(row.get('file_name') or '') for row in rows]
    if len(rows) != 72 or len(set(names)) != 72 or not all(names):
        raise ValueError('人工复核 CSV 不是 72 个唯一文件名')
    statuses = {str(row['file_name']): str(row.get('review_status') or '') for row in rows}
    if dict(Counter(statuses.values())) != EXPECTED_REVIEW:
        raise ValueError(f'人工复核状态错误：{dict(Counter(statuses.values()))}')
    return rows, statuses


def load_v4_groups(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    by_unit = {str(row['unit_id']): row for row in rows}
    by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for frame in str(row.get('frame_ids') or '').split(','):
            if frame.strip():
                by_frame[frame.strip()].append(row)
    return by_unit, by_frame


def load_v6_groups(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in read_jsonl_intxlnk(path):
        group = str(row.get('group') or '').strip()
        source_label = str(row.get('source_label') or '').strip()
        if group and source_label:
            mapping[source_label] = group
    return mapping


def source_tokens(
    record: dict[str, Any],
    v6_groups: dict[str, str],
    v4_by_unit: dict[str, dict[str, str]],
    v4_by_frame: dict[str, list[dict[str, str]]],
) -> set[str]:
    output_name = Path(str(record['output_image'])).name
    source_stem = Path(str(record.get('source_image') or '')).stem
    frames = set(FRAME_RE.findall(output_name))
    frames.update(FRAME_RE.findall(source_stem))
    units = set(UNIT_RE.findall(output_name))
    units.update(UNIT_RE.findall(source_stem))
    for frame in frames:
        for row in v4_by_frame.get(frame, []):
            units.add(str(row['unit_id']))
    tokens = {f'hash:{record["image_sha256"]}', *(f'frame:{frame}' for frame in frames)}
    for unit in units:
        tokens.add(f'unit:{unit}')
        row = v4_by_unit.get(unit)
        if row:
            tokens.update(f'burst:{value}' for value in BURST_RE.findall(str(row.get('raw_group_keys') or '')))
            tokens.update(
                f'frame:{frame}'
                for frame in str(row.get('frame_ids') or '').split(',')
                if frame
            )
    group = v6_groups.get(str(record.get('source_label') or ''))
    if group:
        tokens.add(f'prior-group:{group}')
    return tokens


def build_components(records: list[dict[str, Any]], tokens: list[set[str]]) -> dict[int, int]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    by_token: dict[str, list[int]] = defaultdict(list)
    for index, values in enumerate(tokens):
        for token in values:
            by_token[token].append(index)
    for indices in by_token.values():
        for index in indices[1:]:
            union(indices[0], index)
    return {index: find(index) for index in range(len(records))}


def class_counts_from_labels(
    boxes_by_stem: dict[str, list[tuple[int, float, float, float, float]]]
) -> tuple[Counter[int], Counter[int]]:
    box_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    for boxes in boxes_by_stem.values():
        seen: set[int] = set()
        for class_index, *_ in boxes:
            box_counts[class_index] += 1
            seen.add(class_index)
        for class_index in seen:
            image_counts[class_index] += 1
    return box_counts, image_counts


def as_counter_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(index): int(counter[index]) for index in range(40)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='赛道一 v12 数据集独立审计')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--review-csv', type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument('--v4-groups', type=Path, default=DEFAULT_V4_GROUPS)
    parser.add_argument('--v6-manifest', type=Path, default=DEFAULT_V6_MANIFEST)
    args = parser.parse_args(argv)

    dataset = args.dataset.expanduser().resolve(strict=True)
    report_path = args.report.expanduser().resolve(strict=False)
    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks[name] = {'ok': bool(ok), 'detail': detail}
        if not ok:
            errors.append(name)

    required = [
        dataset / 'dataset.yaml',
        dataset / '.complete',
        dataset / 'source_manifest.jsonl',
        dataset / 'augmentation_manifest.jsonl',
        dataset / 'audit_summary.json',
        dataset / 'augmentation_policy.yaml',
        dataset / 'review/reviewed_72_manifest.csv',
        dataset / 'review/review_provenance.json',
        args.review_csv,
        args.v4_groups,
        args.v6_manifest,
    ]
    missing_required = [str(path) for path in required if not path.is_file()]
    check('required_files_exist', not missing_required, missing_required)
    if missing_required:
        payload = {
            'schema_version': 12,
            'created_at': now_iso(),
            'dataset_root': str(dataset),
            'passed': False,
            'checks': checks,
            'errors': errors,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    dataset_yaml = yaml.safe_load((dataset / 'dataset.yaml').read_text(encoding='utf-8'))
    yaml_path = Path(str(dataset_yaml.get('path') or '')).resolve(strict=False)
    check('dataset_yaml_path', yaml_path == dataset, {'expected': str(dataset), 'actual': str(yaml_path)})
    check('dataset_yaml_nc', int(dataset_yaml.get('nc', -1)) == 40, dataset_yaml.get('nc'))
    check('dataset_yaml_names', list(dataset_yaml.get('names') or []) == EXPECTED_NAMES, len(dataset_yaml.get('names') or []))

    source_records = read_jsonl(dataset / 'source_manifest.jsonl')
    augmentation_records = read_jsonl(dataset / 'augmentation_manifest.jsonl')
    check('source_manifest_count_1232', len(source_records) == 1232, len(source_records))
    check('augmentation_manifest_count_4118', len(augmentation_records) == 4118, len(augmentation_records))

    review_rows, review_status = load_review_csv(args.review_csv)
    review_provenance = read_json(dataset / 'review/review_provenance.json')
    check('review_csv_count_72', len(review_rows) == 72, len(review_rows))
    check('review_status_58_pass_14_issue', dict(Counter(review_status.values())) == EXPECTED_REVIEW, dict(Counter(review_status.values())))
    check('review_provenance_counts', review_provenance.get('reviewed_images') == 72 and review_provenance.get('status_counts') == EXPECTED_REVIEW, review_provenance.get('status_counts'))

    source_names = [Path(str(record['output_image'])).name for record in source_records]
    source_by_name = {name: record for name, record in zip(source_names, source_records)}
    check('source_names_unique', len(source_names) == len(set(source_names)), {'names': len(source_names), 'unique': len(set(source_names))})
    check('reviewed_names_all_in_source', set(review_status).issubset(source_by_name), sorted(set(review_status) - set(source_by_name))[:10])

    source_split = Counter(str(record.get('split')) for record in source_records)
    check('source_split_1062_train_170_val', dict(source_split) == EXPECTED_SOURCE, dict(source_split))
    source_kind_split = Counter((str(record.get('source_kind')), str(record.get('split'))) for record in source_records)
    check('source_kind_split_exact', dict(source_kind_split) == EXPECTED_SOURCE_KIND_SPLIT, {f'{kind}:{split}': count for (kind, split), count in sorted(source_kind_split.items())})

    reviewed_in_train: list[str] = []
    reviewed_in_val: list[str] = []
    review_mismatches: list[str] = []
    for name, status in review_status.items():
        record = source_by_name.get(name)
        if record is None:
            continue
        (reviewed_in_train if record.get('split') == 'train' else reviewed_in_val).append(name)
        if record.get('review_status') != status:
            review_mismatches.append(name)
    check('all_72_reviewed_in_train', len(reviewed_in_train) == 72 and not reviewed_in_val, {'train': len(reviewed_in_train), 'val': len(reviewed_in_val)})
    check('reviewed_manifest_status_match', not review_mismatches, review_mismatches[:10])

    issue_names = {name for name, status in review_status.items() if status == 'issue'}
    pass_names = {name for name, status in review_status.items() if status == 'pass'}
    issue_sources = {Path(str(source_by_name[name]['output_image'])).name for name in issue_names if name in source_by_name}
    check('issue_source_names_14', len(issue_sources) == 14, len(issue_sources))

    v6_groups = load_v6_groups(args.v6_manifest)
    v4_by_unit, v4_by_frame = load_v4_groups(args.v4_groups)
    tokens = [source_tokens(record, v6_groups, v4_by_unit, v4_by_frame) for record in source_records]
    components = build_components(source_records, tokens)
    component_members: dict[int, list[int]] = defaultdict(list)
    for index, component in components.items():
        component_members[component].append(index)
    component_sizes = {index: len(component_members[component]) for index, component in components.items()}
    cross_components = [
        members for members in component_members.values()
        if len({str(source_records[index].get('split')) for index in members}) > 1
    ]
    check('reconstructed_physical_groups_no_cross_split', not cross_components, [len(members) for members in cross_components[:10]])
    metadata_group_mismatches = [
        source_names[index] for index, record in enumerate(source_records)
        if int(record.get('physical_component_size', -1)) != component_sizes[index]
    ]
    check('physical_group_sizes_match_metadata', not metadata_group_mismatches, metadata_group_mismatches[:10])

    reviewed_component_sizes = [component_sizes[source_names.index(name)] for name in reviewed_in_train]
    check('all_reviewed_physical_groups_isolated', len(reviewed_component_sizes) == 72 and max(reviewed_component_sizes, default=0) == 1, Counter(reviewed_component_sizes))

    protected_indices = {
        index for index, record in enumerate(source_records)
        if record.get('original_v10_split') == 'train' or source_names[index] in review_status
    }
    protected_token_counts = Counter(
        token for index in protected_indices for token in tokens[index]
    )
    safe_candidates: list[int] = []
    for index, record in enumerate(source_records):
        if record.get('source_kind') != 'multiobject_v4' or record.get('original_v10_split') != 'train':
            continue
        if source_names[index] in review_status or component_sizes[index] != 1:
            continue
        self_reference = 1 if index in protected_indices else 0
        if any(protected_token_counts[token] > self_reference for token in tokens[index]):
            continue
        safe_candidates.append(index)
    check('safe_validation_candidates_173', len(safe_candidates) == EXPECTED_SAFE_CANDIDATES, len(safe_candidates))

    safe_candidate_names = {source_names[index] for index in safe_candidates}
    new_val_indices = [
        index for index, record in enumerate(source_records)
        if record.get('split_reason') == 'isolated_new_multiobject_validation'
    ]
    new_val_names = {source_names[index] for index in new_val_indices}
    check('new_validation_count_72', len(new_val_indices) == 72, len(new_val_indices))
    check('new_validation_from_safe_candidates', new_val_names.issubset(safe_candidate_names), sorted(new_val_names - safe_candidate_names)[:10])
    check('new_validation_never_reviewed', not bool(new_val_names & set(review_status)), sorted(new_val_names & set(review_status))[:10])
    check('new_validation_groups_isolated', all(component_sizes[index] == 1 for index in new_val_indices), max((component_sizes[index] for index in new_val_indices), default=0))

    prior_val = [
        record for record in source_records
        if record.get('source_kind') == 'prior_v9' and record.get('split') == 'val'
    ]
    multi_val = [
        record for record in source_records
        if record.get('source_kind') == 'multiobject_v4' and record.get('split') == 'val'
    ]
    check('validation_composition_98_prior_72_new_multi', len(prior_val) == 98 and len(multi_val) == 72, {'prior': len(prior_val), 'multi': len(multi_val)})

    source_box_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    new_val_class_boxes: Counter[int] = Counter()
    source_boxes_by_stem: dict[str, list[tuple[int, float, float, float, float]]] = {}
    source_labels = {split: {} for split in SPLITS}
    source_images = {split: {} for split in SPLITS}
    for split in SPLITS:
        images, _, entry_errors = io_entries(dataset / 'images' / split)
        _, labels, label_errors = io_entries(dataset / 'labels' / split)
        errors.extend(entry_errors + label_errors)
        source_images[split] = images
        source_labels[split] = labels
        check(f'{split}_image_label_pairing', set(images) == set(labels), {'images': len(images), 'labels': len(labels), 'missing_labels': sorted(set(images) - set(labels))[:10], 'extra_labels': sorted(set(labels) - set(images))[:10]})

    for record in source_records:
        name = Path(str(record['output_image'])).name
        split = str(record['split'])
        stem = Path(name).stem
        image_path = dataset / str(record['output_image'])
        label_path = dataset / str(record['output_label'])
        if stem not in source_images[split] or stem not in source_labels[split]:
            errors.append(f'源清单文件缺失：{split}/{stem}')
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        check(f'decode_source_{split}_{stem}', image is not None and image.ndim == 3 and min(image.shape[:2]) >= 2, None)
        actual_image_sha = sha256_file(image_path)
        actual_label_sha = sha256_file(label_path)
        check(f'source_image_sha_{split}_{stem}', actual_image_sha == record.get('image_sha256'), actual_image_sha)
        check(f'source_label_sha_{split}_{stem}', actual_label_sha == record.get('label_sha256'), actual_label_sha)
        boxes = parse_label(label_path, f'{split}/{stem}', errors)
        check(f'source_boxes_manifest_{split}_{stem}', boxes_match(boxes, record.get('boxes')), {'actual': len(boxes), 'manifest': record.get('box_count')})
        source_boxes_by_stem[f'{split}/{stem}'] = boxes
        seen: set[int] = set()
        for class_index, *_ in boxes:
            source_box_counts[split][class_index] += 1
            seen.add(class_index)
        if record in (source_records[index] for index in new_val_indices):
            for class_index in seen:
                new_val_class_boxes[class_index] += 1

    new_val_counts = Counter()
    for index in new_val_indices:
        record = source_records[index]
        for box in record.get('boxes') or []:
            new_val_counts[int(box['class_index'])] += 1
    check('new_validation_all_40_classes', len(new_val_counts) == 40, len(new_val_counts))
    check('new_validation_min_7_boxes_per_class', len(new_val_counts) == 40 and min(new_val_counts.values()) >= 7, {str(index): new_val_counts[index] for index in range(40)})

    manifest_source_counts = Counter(str(record.get('split')) for record in source_records)
    check('source_dataset_image_counts', dict(manifest_source_counts) == EXPECTED_SOURCE, dict(manifest_source_counts))
    check('source_box_counts_2441_499', {split: sum(source_box_counts[split].values()) for split in SPLITS} == {'train': 2441, 'val': 499}, {split: sum(source_box_counts[split].values()) for split in SPLITS})
    empty_source = {
        split: sum(not boxes for key, boxes in source_boxes_by_stem.items() if key.startswith(f'{split}/'))
        for split in SPLITS
    }
    check('source_empty_labels_1_1', empty_source == {'train': 1, 'val': 1}, empty_source)

    augmentation_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    augmentation_kind_counts: Counter[str] = Counter()
    for record in augmentation_records:
        augmentation_by_source[Path(str(record.get('source_output_image') or '')).name].append(record)
        augmentation_kind_counts[str(record.get('augmentation_kind'))] += 1
    check('augmentation_kind_counts', dict(augmentation_kind_counts) == EXPECTED_AUGMENTATION, dict(augmentation_kind_counts))

    train_source_names = {
        Path(str(record['output_image'])).name for record in source_records
        if record.get('split') == 'train'
    }
    focus_source_names = {
        name for name in train_source_names
        if any(
            int(box.get('class_index', -1)) in FOCUS_INDICES
            for box in (source_by_name[name].get('boxes') or [])
        )
    }
    generic_count_errors = []
    focus_count_errors = []
    issue_count_errors = []
    non_issue_extra = []
    for name in sorted(train_source_names):
        records = augmentation_by_source.get(name, [])
        kind_counts = Counter(str(record.get('augmentation_kind')) for record in records)
        if kind_counts['generic'] != 3:
            generic_count_errors.append((name, kind_counts['generic']))
        expected_focus = 2 if name in focus_source_names else 0
        if kind_counts['focus'] != expected_focus:
            focus_count_errors.append((name, kind_counts['focus'], expected_focus))
        if name in issue_sources:
            if kind_counts['issue'] != 3:
                issue_count_errors.append((name, kind_counts['issue']))
        elif kind_counts['issue'] != 0:
            non_issue_extra.append((name, kind_counts['issue']))
    check('generic_exactly_3_per_train_source', not generic_count_errors, generic_count_errors[:10])
    check('focus_exactly_2_per_focus_source', not focus_count_errors and len(focus_source_names) == 445, {'focus_sources': len(focus_source_names), 'errors': focus_count_errors[:10]})
    check('issue_exactly_3_per_issue_source', not issue_count_errors and len(issue_sources) == 14, {'issue_sources': len(issue_sources), 'errors': issue_count_errors[:10]})
    check('no_issue_augmentation_for_pass_or_unreviewed', not non_issue_extra, non_issue_extra[:10])
    check('no_augmentation_for_validation_sources', not bool(set(augmentation_by_source) & (set(source_names) - train_source_names)), sorted(set(augmentation_by_source) & (set(source_names) - train_source_names))[:10])

    expected_aug_images: set[str] = set()
    expected_aug_labels: set[str] = set()
    augmentation_hashes: list[str] = []
    augmentation_image_entries, augmentation_label_entries, aug_entry_errors = io_entries(dataset / 'images/train')
    errors.extend(aug_entry_errors)
    _, train_label_entries, train_label_errors = io_entries(dataset / 'labels/train')
    errors.extend(train_label_errors)
    all_aug_boxes: Counter[int] = Counter()
    aug_decode_count = 0
    aug_empty_count = 0
    aug_label_count_error = 0
    aug_hash_error = 0
    aug_manifest_box_error = 0
    for record in augmentation_records:
        output_image_rel = Path(str(record['output_image']))
        output_label_rel = Path(str(record['output_label']))
        expected_aug_images.add(output_image_rel.name)
        expected_aug_labels.add(output_label_rel.name)
        image_path = dataset / output_image_rel
        label_path = dataset / output_label_rel
        if not image_path.is_file() or not label_path.is_file():
            errors.append(f'增强文件缺失：{output_image_rel} / {output_label_rel}')
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or min(image.shape[:2]) < 2:
            errors.append(f'增强图无法解码：{image_path}')
        else:
            aug_decode_count += 1
        actual_sha = sha256_file(image_path)
        if actual_sha != record.get('output_sha256'):
            aug_hash_error += 1
        augmentation_hashes.append(actual_sha)
        boxes = parse_label(label_path, str(output_label_rel), errors)
        if not boxes:
            aug_empty_count += 1
        if len(boxes) != int(record.get('output_box_count', -1)):
            aug_label_count_error += 1
        if len(boxes) != int(record.get('source_box_count', -1)):
            aug_manifest_box_error += 1
        for class_index, *_ in boxes:
            all_aug_boxes[class_index] += 1
    check('augmentation_paths_unique', len(expected_aug_images) == len(augmentation_records) and len(expected_aug_labels) == len(augmentation_records), {'images': len(expected_aug_images), 'labels': len(expected_aug_labels)})
    check('augmentation_files_exist', not any('增强文件缺失' in error for error in errors), len(augmentation_records))
    check('augmentation_images_decode', aug_decode_count == len(augmentation_records), aug_decode_count)
    check('augmentation_manifest_hashes_match', aug_hash_error == 0, aug_hash_error)
    check('augmentation_label_box_counts_match_manifest', aug_label_count_error == 0, aug_label_count_error)
    check('augmentation_preserves_source_box_count', aug_manifest_box_error == 0, aug_manifest_box_error)
    check('augmentation_pixel_hashes_unique', len(augmentation_hashes) == len(set(augmentation_hashes)), {'hashes': len(augmentation_hashes), 'unique': len(set(augmentation_hashes))})

    source_train_names = {Path(str(record['output_image'])).name for record in source_records if record.get('split') == 'train'}
    source_train_stems = {Path(name).stem for name in source_train_names}
    source_val_names = {Path(str(record['output_image'])).name for record in source_records if record.get('split') == 'val'}
    actual_train_image_names = set(augmentation_image_entries)
    actual_train_label_names = set(train_label_entries)
    expected_train_image_names = source_train_stems | {Path(name).stem for name in expected_aug_images}
    check('train_filesystem_exact_source_plus_augmentation', actual_train_image_names == expected_train_image_names, {'actual': len(actual_train_image_names), 'expected': len(expected_train_image_names), 'extra': sorted(actual_train_image_names - expected_train_image_names)[:10], 'missing': sorted(expected_train_image_names - actual_train_image_names)[:10]})
    check('train_label_filesystem_exact', actual_train_label_names == {Path(name).stem for name in source_train_names} | {Path(name).stem for name in expected_aug_labels}, {'actual': len(actual_train_label_names), 'expected': len(source_train_names | expected_aug_images)})
    check('validation_filename_set_exact', set(source_images['val']) == {Path(name).stem for name in source_val_names}, {'actual': len(source_images['val']), 'expected': len(source_val_names)})
    check('validation_has_no_augmentation_names', not any(name.startswith('aug__') for name in source_images['val']), [name for name in source_images['val'] if name.startswith('aug__')][:10])

    source_train_hashes = {
        str(record['image_sha256']) for record in source_records if record.get('split') == 'train'
    }
    source_val_hashes = {
        str(record['image_sha256']) for record in source_records if record.get('split') == 'val'
    }
    augmentation_hash_set = set(augmentation_hashes)
    check('source_train_val_pixel_disjoint', not bool(source_train_hashes & source_val_hashes), len(source_train_hashes & source_val_hashes))
    check('augmentation_does_not_repeat_validation', not bool(augmentation_hash_set & source_val_hashes), len(augmentation_hash_set & source_val_hashes))
    check('augmentation_does_not_repeat_source_train', not bool(augmentation_hash_set & source_train_hashes), len(augmentation_hash_set & source_train_hashes))
    check('validation_pixel_hashes_unique', len(source_val_hashes) == 170, len(source_val_hashes))

    output_box_counts = {
        'train': source_box_counts['train'] + all_aug_boxes,
        'val': source_box_counts['val'],
    }
    output_empty_labels = {
        'train': empty_source['train'] + aug_empty_count,
        'val': empty_source['val'],
    }
    check('output_image_counts_5180_170', {split: len(source_images[split]) for split in SPLITS} == {'train': 5180, 'val': 170}, {split: len(source_images[split]) for split in SPLITS})
    check('output_box_counts_13230_499', {split: sum(output_box_counts[split].values()) for split in SPLITS} == EXPECTED_BOXES, {split: sum(output_box_counts[split].values()) for split in SPLITS})
    check('output_empty_labels_4_1', output_empty_labels == {'train': 4, 'val': 1}, output_empty_labels)
    check('train_all_40_classes', len([index for index in range(40) if output_box_counts['train'][index] > 0]) == 40, len([index for index in range(40) if output_box_counts['train'][index] > 0]))
    check('val_all_40_classes', len([index for index in range(40) if output_box_counts['val'][index] > 0]) == 40, len([index for index in range(40) if output_box_counts['val'][index] > 0]))

    builder_summary = read_json(dataset / 'audit_summary.json')
    check('builder_training_ready', builder_summary.get('training_ready') is True, builder_summary.get('training_ready'))
    check('builder_training_not_started', builder_summary.get('training_started') is False, builder_summary.get('training_started'))
    check('builder_counts_match_independent', builder_summary.get('images') == EXPECTED_IMAGES and builder_summary.get('boxes') == EXPECTED_BOXES, {'images': builder_summary.get('images'), 'boxes': builder_summary.get('boxes')})
    check('builder_multiobject_321_images_1646_boxes', sum(record.get('source_kind') == 'multiobject_v4' for record in source_records) == 321 and sum(int(record.get('box_count', 0)) for record in source_records if record.get('source_kind') == 'multiobject_v4') == 1646, None)

    decoder_mismatches = [name for name, item in checks.items() if name.startswith('decode_source_') and not item['ok']]
    check('all_source_images_decode', not decoder_mismatches, len(decoder_mismatches))
    sha_mismatches = [
        name for name, item in checks.items()
        if (name.startswith('source_image_sha_') or name.startswith('source_label_sha_')) and not item['ok']
    ]
    check('all_source_hashes_match_manifest', not sha_mismatches, len(sha_mismatches))
    box_mismatches = [
        name for name, item in checks.items()
        if name.startswith('source_boxes_manifest_') and not item['ok']
    ]
    check('all_source_labels_match_manifest', not box_mismatches, len(box_mismatches))

    failed_checks = [name for name, item in checks.items() if not item['ok']]
    passed = not failed_checks
    payload = {
        'schema_version': 12,
        'created_at': now_iso(),
        'dataset_root': str(dataset),
        'dataset_yaml': str(dataset / 'dataset.yaml'),
        'report_role': 'independent_scan_not_builder_summary',
        'passed': passed,
        'training_ready': passed,
        'errors': failed_checks,
        'metrics': {
            'classes': 40,
            'source_images': dict(source_split),
            'images': {split: len(source_images[split]) for split in SPLITS},
            'boxes': {split: sum(output_box_counts[split].values()) for split in SPLITS},
            'empty_labels': output_empty_labels,
            'class_boxes': {split: as_counter_dict(output_box_counts[split]) for split in SPLITS},
            'reviewed_images': len(reviewed_in_train),
            'review_status': dict(Counter(review_status.values())),
            'new_validation_images': len(new_val_indices),
            'new_validation_min_class_boxes': min(new_val_counts.values()) if new_val_counts else 0,
            'new_validation_class_boxes': {str(index): new_val_counts[index] for index in range(40)},
            'safe_candidate_count': len(safe_candidates),
            'physical_components': len(component_members),
            'physical_components_cross_split': len(cross_components),
            'augmentation': dict(augmentation_kind_counts),
            'focus_sources': len(focus_source_names),
            'issue_sources': len(issue_sources),
            'augmentation_unique_hashes': len(set(augmentation_hashes)),
            'decoded_images': sum(len(source_images[split]) for split in SPLITS),
        },
        'provenance': {
            'review_csv': str(args.review_csv.resolve()),
            'review_csv_sha256': sha256_file(args.review_csv),
            'v4_groups': str(args.v4_groups.resolve()),
            'v4_groups_sha256': sha256_file(args.v4_groups),
            'v6_manifest': str(args.v6_manifest.resolve()),
            'v6_manifest_sha256': sha256_file(args.v6_manifest),
            'source_manifest_sha256': sha256_file(dataset / 'source_manifest.jsonl'),
            'augmentation_manifest_sha256': sha256_file(dataset / 'augmentation_manifest.jsonl'),
        },
        'checks': checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'passed': passed,
        'report': str(report_path),
        'failed_checks': failed_checks,
        'metrics': payload['metrics'],
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'审计异常：{type(exc).__name__}: {exc}', file=sys.stderr)
        raise
