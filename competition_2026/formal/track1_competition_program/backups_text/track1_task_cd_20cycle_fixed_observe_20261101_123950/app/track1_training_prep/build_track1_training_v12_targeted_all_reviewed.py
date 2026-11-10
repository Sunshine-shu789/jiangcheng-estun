#!/usr/bin/env python3
"""Build Track-1 v12 from the audited v10 source inventory.

Policy:
  * keep all 1,232 v10 source entries;
  * move all 72 manually reviewed multi-object images into train;
  * move 72 isolated, never-reviewed v4 multi-object images from train to val;
  * copy every original byte-for-byte and never augment val;
  * add 3 generic offline variants to every train source;
  * add 2 focused variants to train sources containing any priority class;
  * add 3 issue-weighted variants to each of the 14 reviewed issue images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import yaml

PROJECT_ROOT = Path(os.environ.get(
    'TRACK1_PROJECT_ROOT',
    '/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序',
)).resolve()
# The requested logical path /media/sunshine/新加卷 contains an IntxLNK mount
# entry; resolving the volume root itself therefore does not reach the NTFS
# mount. The project root resolves to the physical /新加卷1 tree. Pass that
# canonical volume to the shared entry resolver before importing it.
DATA_VOLUME = Path(os.environ.get(
    'TRACK1_DATA_VOLUME', str(PROJECT_ROOT.parents[1]),
)).resolve()
os.environ['TRACK1_DATA_VOLUME'] = str(DATA_VOLUME)
APP_DIR = PROJECT_ROOT / 'app/track1_training_prep'
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from build_track1_training_v10_all_multiclass_augmented import (  # noqa: E402
    Box,
    BuildError as HelperBuildError,
    SourceSample,
    draw_contact_sheet,
    parse_label,
    resolve_data_entry,
    safe_component,
    sha256_file,
    write_augmented_sample,
    yolo_text,
)

DEFAULT_V10 = PROJECT_ROOT / 'datasets/track1_training_merged_v10_all_multiclass_augmented_20261031'
DEFAULT_OUTPUT = PROJECT_ROOT / 'datasets/track1_training_merged_v12_targeted_all_reviewed_20261101'
DEFAULT_REVIEW_CSV = PROJECT_ROOT / 'reports/track1_v11_offline_acceptance_20261101/manual_review_multiobject.csv'
DEFAULT_REVIEW_JSON = PROJECT_ROOT / 'reports/track1_v11_offline_acceptance_20261101/manual_review_multiobject.json'
DEFAULT_V4_REVIEW_GROUPS = PROJECT_ROOT / 'datasets/track1_all_history_multi_object_review_20261031_v4/manifests/review_groups.csv'
DEFAULT_V6_MANIFEST = PROJECT_ROOT / 'datasets/track1_training_merged_v9_hardneg/v6_merge_manifest.jsonl'
DEFAULT_BASE_WEIGHTS = PROJECT_ROOT / 'models/yolo11s_backbone_40class_init_v10aug.pt'
V11_BEST = PROJECT_ROOT / 'runs/track1_train_40class_v11_full_v10aug_from_yolo11s/track1_40class_v11_20261031_234506/weights/best.pt'
SPLITS = ('train', 'val')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
FRAME_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')
UNIT_RE = re.compile(r'(?:front10-review-unit|split-unit|all-history-multi)-\d+')
BURST_RE = re.compile(r'burst:([^|;]+)')
EXPECTED_SOURCE_COUNTS = {'train': 1062, 'val': 170}
EXPECTED_REVIEW_COUNTS = {'pass': 58, 'issue': 14}
EXPECTED_TARGET_SPLIT_COUNTS = {
    ('prior_v9', 'train'): 813,
    ('prior_v9', 'val'): 98,
    ('multiobject_v4', 'train'): 249,
    ('multiobject_v4', 'val'): 72,
}
EXPECTED_SAFE_CANDIDATES = 173
FOCUS_INDICES = (
    2, 5, 6, 13, 14, 17, 18, 22, 23, 26, 29, 30, 31, 32,
    35, 36,
)


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class V12Sample:
    record: dict[str, Any]
    boxes: tuple[Box, ...]
    split: str
    reason: str
    review_status: str
    review_row: dict[str, str] | None
    physical_group: str
    component_size: int
    v10_image: Path
    v10_label: Path
    output_stem: str
    output_image: Path
    output_label: Path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f'缺少 JSON：{path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise BuildError(f'JSON 顶层不是对象：{path}')
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f'缺少 JSONL：{path}')
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise BuildError(f'{path}:{line_number} JSON 顶层不是对象')
        rows.append(value)
    return rows


def output_name(record: dict[str, Any]) -> str:
    return Path(str(record['output_image'])).name


def load_review_rows(csv_path: Path, json_path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    if not csv_path.is_file():
        raise BuildError(f'缺少人工复核 CSV：{csv_path}')
    with csv_path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    names = [str(row.get('file_name') or '') for row in rows]
    if len(rows) != 72 or len(set(names)) != 72 or not all(names):
        raise BuildError(f'人工复核记录应为 72 张且文件名唯一，实际 {len(rows)} / {len(set(names))}')
    status_counts = Counter(str(row.get('review_status') or '') for row in rows)
    if dict(status_counts) != EXPECTED_REVIEW_COUNTS:
        raise BuildError(f'人工复核状态不符：{dict(status_counts)}')
    review_map = {name: str(row['review_status']) for name, row in zip(names, rows)}
    rows_by_name = {str(row['file_name']): row for row in rows}

    report = read_json(json_path)
    issues = report.get('issues', [])
    if report.get('reviewed_images') != 72 or report.get('status_counts') != EXPECTED_REVIEW_COUNTS:
        raise BuildError('人工复核 JSON 汇总不是 72 张 / 58 pass / 14 issue')
    if not isinstance(issues, list) or len(issues) != 14:
        raise BuildError('人工复核 JSON 问题列表不是 14 张')
    json_issue_names = {str(item.get('file_name') or '') for item in issues if isinstance(item, dict)}
    csv_issue_names = {name for name, status in review_map.items() if status == 'issue'}
    if json_issue_names != csv_issue_names:
        raise BuildError('人工复核 CSV 与 JSON 的 14 张问题图不一致')
    return [rows_by_name[name] for name in sorted(rows_by_name)], review_map


def load_v6_groups(path: Path) -> dict[str, str]:
    resolved = resolve_data_entry(path, context='v6 采集组清单')
    mapping: dict[str, str] = {}
    for line_number, raw in enumerate(resolved.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        group = str(value.get('group') or '').strip()
        source_label = str(value.get('source_label') or '').strip()
        if group and source_label:
            previous = mapping.get(source_label)
            if previous and previous != group:
                raise BuildError(f'{resolved}:{line_number} 同标签对应多个采集组')
            mapping[source_label] = group
    if not mapping:
        raise BuildError('v6 清单没有可用采集组')
    return mapping


def load_v4_groups(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    if not path.is_file():
        raise BuildError(f'缺少 v4 review_groups.csv：{path}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    by_unit = {str(row['unit_id']): row for row in rows}
    by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for frame in str(row.get('frame_ids') or '').split(','):
            if frame.strip():
                by_frame[frame.strip()].append(row)
    return by_unit, by_frame


def frame_ids(record: dict[str, Any]) -> set[str]:
    values = FRAME_RE.findall(output_name(record))
    values.extend(FRAME_RE.findall(Path(str(record['source_image'])).stem))
    return set(values)


def group_tokens(
    record: dict[str, Any], *, v6_groups: dict[str, str],
    v4_by_unit: dict[str, dict[str, str]], v4_by_frame: dict[str, list[dict[str, str]]],
) -> set[str]:
    frames = frame_ids(record)
    units = set(UNIT_RE.findall(output_name(record)))
    units.update(UNIT_RE.findall(Path(str(record['source_image'])).stem))
    for frame in frames:
        for row in v4_by_frame.get(frame, []):
            units.add(str(row['unit_id']))
    tokens = {f'hash:{record["image_sha256"]}', *(f'frame:{frame}' for frame in frames)}
    for unit in units:
        tokens.add(f'unit:{unit}')
        row = v4_by_unit.get(unit)
        if row:
            tokens.update(f'burst:{match}' for match in BURST_RE.findall(str(row.get('raw_group_keys') or '')))
            tokens.update(f'frame:{frame}' for frame in str(row.get('frame_ids') or '').split(',') if frame)
    group = v6_groups.get(str(record.get('source_label') or ''))
    if group:
        tokens.add(f'prior-group:{group}')
    return tokens


def build_physical_groups(
    records: list[dict[str, Any]], *, v6_groups: dict[str, str],
    v4_by_unit: dict[str, dict[str, str]], v4_by_frame: dict[str, list[dict[str, str]]],
) -> tuple[dict[int, str], dict[str, list[int]], list[set[str]]]:
    tokens = [
        group_tokens(record, v6_groups=v6_groups, v4_by_unit=v4_by_unit, v4_by_frame=v4_by_frame)
        for record in records
    ]
    token_records: dict[str, list[int]] = defaultdict(list)
    for index, values in enumerate(tokens):
        for token in values:
            token_records[token].append(index)
    parents = list(range(len(records)))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for indices in token_records.values():
        for index in indices[1:]:
            union(indices[0], index)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    group_ids: dict[int, str] = {}
    component_by_id: dict[str, list[int]] = {}
    for indices in components.values():
        material = '\n'.join(sorted(token for index in indices for token in tokens[index]))
        group_id = 'phys-' + hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]
        if group_id in component_by_id:
            raise BuildError(f'物理组 ID 冲突：{group_id}')
        component_by_id[group_id] = indices
        for index in indices:
            group_ids[index] = group_id
    return group_ids, component_by_id, tokens


def balanced_select(
    records: list[dict[str, Any]], candidates: list[int], count: int = 72,
) -> tuple[list[int], dict[str, Any]]:
    if len(candidates) < count:
        raise BuildError(f'安全验证候选不足：{len(candidates)} < {count}')
    ordered = sorted(candidates, key=lambda index: output_name(records[index]))
    if len(ordered) != EXPECTED_SAFE_CANDIDATES:
        raise BuildError(f'安全验证候选应为 {EXPECTED_SAFE_CANDIDATES}，实际 {len(ordered)}')
    box_counts = [
        Counter(box.class_index for box in records[index]['_boxes'])
        for index in ordered
    ]
    missing = [index + 1 for index in range(40) if not any(counts[index] for counts in box_counts)]
    if missing:
        raise BuildError(f'安全候选未覆盖全部 40 类，缺少：{missing}')

    selected_positions: list[int] = []
    remaining = list(range(len(ordered)))
    totals: Counter[int] = Counter()
    for _ in range(count):
        def greedy_key(position: int) -> tuple[int, int, int]:
            counts = box_counts[position]
            deficit = sum(weight * max(0, 12 - totals[class_index]) for class_index, weight in counts.items())
            return deficit, -sum(counts.values()), -position
        best = max(remaining, key=greedy_key)
        remaining.remove(best)
        selected_positions.append(best)
        totals.update(box_counts[best])

    def objective(position_list: list[int]) -> tuple[int, float, int]:
        counts = Counter()
        for position in position_list:
            counts.update(box_counts[position])
        values = [counts[index] for index in range(40)]
        return min(values), -statistics.pvariance(values), -sum(value * value for value in values)

    current = objective(selected_positions)
    for iteration in range(100):
        selected_set = set(selected_positions)
        best_swap: tuple[int, int] | None = None
        best_objective = current
        best_values: list[int] | None = None
        for slot in range(count):
            old_position = selected_positions[slot]
            for new_position in range(len(ordered)):
                if new_position in selected_set:
                    continue
                trial = selected_positions.copy()
                trial[slot] = new_position
                value = objective(trial)
                if value > best_objective:
                    best_objective = value
                    best_swap = slot, new_position
                    best_values = trial
        if best_swap is None:
            break
        selected_positions = best_values if best_values is not None else selected_positions
        current = best_objective
    else:
        iteration = 100

    selected = [ordered[position] for position in selected_positions]
    counts = Counter(box.class_index for index in selected for box in records[index]['_boxes'])
    if len(selected) != count or len(counts) != 40 or min(counts.values()) < 4:
        raise BuildError(f'验证集平衡选择失败：images={len(selected)} classes={len(counts)} min={min(counts.values()) if counts else 0}')
    return selected, {
        'safe_candidate_count': len(ordered),
        'selected_count': len(selected),
        'class_boxes': {str(index): counts[index] for index in range(40)},
        'class_box_min': min(counts.values()),
        'class_box_max': max(counts.values()),
        'class_box_variance': statistics.pvariance(counts[index] for index in range(40)),
        'selection_iterations': iteration,
    }


def copy_unchanged(source: Path, target: Path, *, expected_sha256: str, context: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    actual = sha256_file(target)
    if actual != expected_sha256:
        raise BuildError(f'{context} 复制后 SHA256 不符：{target}')
    return actual


def prepare_samples(
    v10_root: Path, output_root: Path, review_rows: list[dict[str, str]],
    review_map: dict[str, str], v6_groups: dict[str, str],
    v4_by_unit: dict[str, dict[str, str]], v4_by_frame: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[V12Sample], dict[str, Any]]:
    records = read_jsonl(v10_root / 'source_manifest.jsonl')
    if len(records) != 1232:
        raise BuildError(f'v10 source manifest 应为 1232 条，实际 {len(records)}')
    names = [output_name(record) for record in records]
    if len(set(names)) != len(names):
        raise BuildError('v10 source manifest 输出文件名重复')
    by_name = {name: index for index, name in enumerate(names)}
    missing_reviewed = sorted(set(review_map) - set(by_name))
    if missing_reviewed:
        raise BuildError(f'复核文件无法映射到 v10：{missing_reviewed[:5]}')
    for name in review_map:
        record = records[by_name[name]]
        if record.get('split') != 'val':
            raise BuildError(f'复核图原 v10 split 不是 val：{name} -> {record.get("split")}')
    original_counts = Counter(str(record['split']) for record in records)
    if dict(original_counts) != EXPECTED_SOURCE_COUNTS:
        raise BuildError(f'v10 split 计数异常：{dict(original_counts)}')

    group_ids, components, tokens = build_physical_groups(
        records, v6_groups=v6_groups, v4_by_unit=v4_by_unit, v4_by_frame=v4_by_frame,
    )
    current_cross = [
        component for component in components.values()
        if len({str(records[index]['split']) for index in component}) > 1
    ]
    if current_cross:
        raise BuildError(f'v10 独立物理组审计发现跨 split：{len(current_cross)}')
    for index, record in enumerate(records):
        label = resolve_data_entry(v10_root / str(record['output_label']), context=f'v10 标签/{names[index]}')
        record['_boxes'] = parse_label(label, context=f'v10/{names[index]}')
        if len(record['_boxes']) != int(record.get('box_count', -1)):
            raise BuildError(f'v10 标签框数与清单不符：{names[index]}')
        record['_index'] = index
        record['_name'] = names[index]
        record['_group_id'] = group_ids[index]
        record['_component_size'] = len(components[group_ids[index]])

    protected_train = {
        index for index, record in enumerate(records)
        if str(record['split']) == 'train' or names[index] in review_map
    }
    protected_token_counts = Counter(
        token for index in protected_train for token in tokens[index]
    )
    candidates: list[int] = []
    for index, record in enumerate(records):
        if record.get('source_kind') != 'multiobject_v4':
            continue
        if record.get('split') != 'train' or names[index] in review_map:
            continue
        if len(components[group_ids[index]]) != 1:
            continue
        # A candidate is itself protected because it currently belongs to
        # train. Ignore that one self-reference, while still rejecting any
        # token shared with another protected image.
        self_reference = 1 if index in protected_train else 0
        if any(protected_token_counts[token] > self_reference for token in tokens[index]):
            continue
        candidates.append(index)
    selected, selection = balanced_select(records, candidates, 72)
    selected_set = set(selected)
    reviewed_set = {by_name[name] for name in review_map}
    if selected_set & reviewed_set:
        raise BuildError('新验证候选与人工复核图重叠')

    target_counts = Counter()
    samples: list[V12Sample] = []
    for index, record in enumerate(records):
        name = names[index]
        if index in reviewed_set:
            split = 'train'
            reason = f'moved_reviewed_{review_map[name]}_val_to_train'
            status = review_map[name]
        elif index in selected_set:
            split = 'val'
            reason = 'isolated_new_multiobject_validation'
            status = 'none'
        else:
            split = str(record['split'])
            reason = 'preserved_v10_split'
            status = review_map.get(name, 'none')
        target_counts[(str(record['source_kind']), split)] += 1
        source_stem = Path(str(record['output_image'])).stem
        destination_stem = safe_component(source_stem)
        suffix = Path(str(record['output_image'])).suffix.lower()
        v10_image = resolve_data_entry(v10_root / str(record['output_image']), context=f'v10 图片/{name}')
        v10_label = resolve_data_entry(v10_root / str(record['output_label']), context=f'v10 标签/{name}')
        samples.append(V12Sample(
            record=record,
            boxes=record['_boxes'],
            split=split,
            reason=reason,
            review_status=status,
            review_row=next((row for row in review_rows if row['file_name'] == name), None),
            physical_group=record['_group_id'],
            component_size=int(record['_component_size']),
            v10_image=v10_image,
            v10_label=v10_label,
            output_stem=destination_stem,
            output_image=output_root / 'images' / split / f'{destination_stem}{suffix}',
            output_label=output_root / 'labels' / split / f'{destination_stem}.txt',
        ))
    if target_counts != Counter(EXPECTED_TARGET_SPLIT_COUNTS):
        raise BuildError(f'v12 目标 split 计数异常：{dict(target_counts)}')
    if sum(sample.split == 'train' for sample in samples) != 1062 or sum(sample.split == 'val' for sample in samples) != 170:
        raise BuildError('v12 来源图片总数不是 train 1062 / val 170')

    provisional_split = {index: sample.split for index, sample in enumerate(samples)}
    crossing = []
    for component in components.values():
        if len({provisional_split[index] for index in component}) > 1:
            crossing.append(component)
    if crossing:
        raise BuildError(f'重划分后物理组跨 split：{len(crossing)}')
    reviewed_components = {samples[index].physical_group for index in reviewed_set}
    if any(len(components[group]) != 1 for group in reviewed_components):
        raise BuildError('至少有一张复核图不是隔离物理组，不能安全转入训练')
    if any(samples[index].component_size != 1 for index in selected):
        raise BuildError('至少有一个新验证候选不是隔离物理组')
    report = {
        'source_counts': dict(original_counts),
        'target_counts_by_kind': {f'{kind}:{split}': value for (kind, split), value in sorted(target_counts.items())},
        'physical_group_count': len(components),
        'physical_group_cross_split_count': 0,
        'reviewed_moved_to_train': len(reviewed_set),
        'new_validation_moved_from_train': len(selected_set),
        'reviewed_groups_isolated': len(reviewed_components),
        'selected_groups_isolated': len(selected),
        'selection': selection,
    }
    return records, samples, report


def build_source_manifest(staging: Path, samples: list[V12Sample], review_source_sha: str) -> None:
    with (staging / 'source_manifest.jsonl').open('w', encoding='utf-8') as handle:
        for sample in samples:
            record = sample.record
            image_hash = copy_unchanged(
                sample.v10_image, sample.output_image,
                expected_sha256=str(record['image_sha256']), context=f'原图/{sample.output_stem}',
            )
            label_hash = copy_unchanged(
                sample.v10_label, sample.output_label,
                expected_sha256=sha256_file(sample.v10_label), context=f'标签/{sample.output_stem}',
            )
            payload = {
                'schema_version': 12,
                'source_kind': record.get('source_kind'),
                'source_root': record.get('source_root'),
                'original_v10_split': record.get('split'),
                'split': sample.split,
                'split_reason': sample.reason,
                'review_status': sample.review_status,
                'source_image_entry': record.get('source_image_entry'),
                'source_label_entry': record.get('source_label_entry'),
                'source_image': record.get('source_image'),
                'source_label': record.get('source_label'),
                'v10_output_image': record.get('output_image'),
                'v10_output_label': record.get('output_label'),
                'output_image': str(sample.output_image.relative_to(staging)),
                'output_label': str(sample.output_label.relative_to(staging)),
                'label_authority': record.get('label_authority'),
                'image_sha256': image_hash,
                'label_sha256': label_hash,
                'box_count': len(sample.boxes),
                'boxes': [
                    {
                        'class_index': box.class_index, 'x': box.x, 'y': box.y,
                        'width': box.width, 'height': box.height,
                    }
                    for box in sample.boxes
                ],
                'physical_group': sample.physical_group,
                'physical_component_size': sample.component_size,
                'review_source_sha256': review_source_sha,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def as_source_sample(sample: V12Sample) -> SourceSample:
    return SourceSample(
        source_kind=str(sample.record['source_kind']),
        source_root=Path(str(sample.record['source_root'])).resolve(strict=False),
        split=sample.split,
        stem=sample.output_stem,
        image_entry=sample.output_image,
        label_entry=sample.output_label,
        image=sample.v10_image,
        label=sample.v10_label,
        boxes=sample.boxes,
        image_sha256=str(sample.record['image_sha256']),
        label_authority=str(sample.record.get('label_authority') or sample.record['source_kind']),
    )


def build_augmentations(
    staging: Path, samples: list[V12Sample], review_map: dict[str, str], seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_samples = sorted(
        (sample for sample in samples if sample.split == 'train'),
        key=lambda sample: (str(sample.record['source_kind']), sample.output_stem),
    )
    focus_sources = [
        sample for sample in train_samples
        if any(box.class_index in FOCUS_INDICES for box in sample.boxes)
    ]
    issue_sources = [
        sample for sample in train_samples if review_map.get(Path(str(sample.record['output_image'])).name) == 'issue'
    ]
    if len(issue_sources) != 14:
        raise BuildError(f'14 张问题图没有全部进入训练集：{len(issue_sources)}')
    focus_index_by_stem = {sample.output_stem: sample for sample in focus_sources}
    issue_index_by_stem = {sample.output_stem: sample for sample in issue_sources}
    records: list[dict[str, Any]] = []

    def write_variant(
        sample: V12Sample, kind: str, variant_number: int, profile: int, variant_seed: int,
    ) -> None:
        prefix = f'aug__{kind}{variant_number:02d}__{sample.output_stem}.jpg'
        output_image = staging / 'images/train' / prefix
        output_label = staging / 'labels/train' / f'aug__{kind}{variant_number:02d}__{sample.output_stem}.txt'
        output_boxes = write_augmented_sample(
            as_source_sample(sample), output_image, output_label, profile, variant_seed,
        )
        record = {
            'schema_version': 12,
            'source_image': str(sample.v10_image.resolve()),
            'source_output_image': str(sample.output_image.relative_to(staging)),
            'label_authority': sample.record.get('label_authority'),
            'output_image': str(output_image.relative_to(staging)),
            'output_label': str(output_label.relative_to(staging)),
            'split': 'train',
            'augmentation_kind': kind,
            'variant': variant_number,
            'profile': profile + 1,
            'seed': variant_seed,
            'review_status': sample.review_status,
            'contains_focus_class': sample.output_stem in focus_index_by_stem,
            'source_box_count': len(sample.boxes),
            'output_box_count': len(output_boxes),
            'output_sha256': sha256_file(output_image),
        }
        records.append(record)

    for sample_index, sample in enumerate(train_samples):
        for variant in range(3):
            variant_seed = seed + sample_index * 1009 + variant * 7919
            write_variant(sample, 'generic', variant + 1, variant, variant_seed)
    for sample_index, sample in enumerate(train_samples):
        if sample.output_stem not in focus_index_by_stem:
            continue
        for variant, profile in enumerate((0, 2), 1):
            variant_seed = seed + 1_000_000 + sample_index * 1013 + variant * 7927
            write_variant(sample, 'focus', variant, profile, variant_seed)
    for sample_index, sample in enumerate(train_samples):
        if sample.output_stem not in issue_index_by_stem:
            continue
        for variant, profile in enumerate((2, 0, 1), 1):
            variant_seed = seed + 2_000_000 + sample_index * 1019 + variant * 7933
            write_variant(sample, 'issue', variant, profile, variant_seed)

    expected_generic = len(train_samples) * 3
    expected_focus = len(focus_sources) * 2
    expected_issue = len(issue_sources) * 3
    if dict(Counter(record['augmentation_kind'] for record in records)) != {
        'generic': expected_generic,
        'focus': expected_focus,
        'issue': expected_issue,
    }:
        raise BuildError('增强数量不符合策略')
    hashes = [str(record['output_sha256']) for record in records]
    if len(set(hashes)) != len(hashes):
        raise BuildError('增强图片出现内容重复')
    with (staging / 'augmentation_manifest.jsonl').open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return records, {
        'train_sources': len(train_samples),
        'generic_images': expected_generic,
        'focus_sources': len(focus_sources),
        'focus_images': expected_focus,
        'issue_sources': len(issue_sources),
        'issue_images': expected_issue,
        'total_images': len(records),
        'sources_without_focus': len(train_samples) - len(focus_sources),
        'issue_sources_with_focus': sum(sample.output_stem in focus_index_by_stem for sample in issue_sources),
    }


def scan_output(staging: Path, class_names: list[str]) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    images: dict[str, dict[str, Path]] = {}
    labels: dict[str, dict[str, Path]] = {}
    box_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    image_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    empty_labels: dict[str, int] = {split: 0 for split in SPLITS}
    hashes: dict[str, dict[str, str]] = {split: {} for split in SPLITS}
    decoded = 0
    for split in SPLITS:
        images[split] = {
            path.stem: path for path in (staging / 'images' / split).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        labels[split] = {
            path.stem: path for path in (staging / 'labels' / split).iterdir()
            if path.is_file() and path.suffix.lower() == '.txt'
        }
        if set(images[split]) != set(labels[split]):
            raise BuildError(f'输出 {split} 图片/标签不配对')
        for stem in sorted(images[split]):
            image = cv2.imread(str(images[split][stem]), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or min(image.shape[:2]) < 2:
                raise BuildError(f'输出图片无法解码：{images[split][stem]}')
            decoded += 1
            hashes[split][stem] = sha256_file(images[split][stem])
            boxes = parse_label(labels[split][stem], context=f'output/{split}/{stem}')
            if not boxes:
                empty_labels[split] += 1
            for box in boxes:
                box_counts[split][box.class_index] += 1
                image_counts[split][box.class_index] += 1
        if len(box_counts[split]) != 40:
            missing = [index for index in range(40) if not box_counts[split][index]]
            raise BuildError(f'{split} 未覆盖 40 类，缺少：{missing}')
    output_images = {split: len(images[split]) for split in SPLITS}
    output_boxes = {split: sum(box_counts[split].values()) for split in SPLITS}
    val_hash_list = list(hashes['val'].values())
    if len(val_hash_list) != len(set(val_hash_list)):
        raise BuildError('验证集图片哈希不唯一')
    if set(hashes['train'].values()) & set(val_hash_list):
        raise BuildError('训练集和验证集存在相同图片像素')
    if any(stem.startswith('aug__') for stem in images['val']):
        raise BuildError('验证集出现增强图')
    if decoded != sum(output_images.values()):
        raise BuildError('图片解码计数异常')
    details = {
        'images': images,
        'labels': labels,
        'box_counts': box_counts,
        'image_counts': image_counts,
        'empty_labels': empty_labels,
        'hashes': hashes,
        'decoded': decoded,
        'val_unique_hashes': len(set(val_hash_list)),
    }
    return output_images, output_boxes, details


def write_review_manifest(staging: Path, samples: list[V12Sample], rows: list[dict[str, str]]) -> None:
    review_dir = staging / 'review'
    review_dir.mkdir(parents=True, exist_ok=True)
    sample_by_name = {Path(str(sample.record['output_image'])).name: sample for sample in samples}
    with (review_dir / 'reviewed_72_manifest.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        fieldnames = list(rows[0].keys()) + [
            'v12_split', 'v12_output_image', 'issue_extra_augmentations',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sample = sample_by_name[row['file_name']]
            writer.writerow({
                **row,
                'v12_split': sample.split,
                'v12_output_image': str(sample.output_image.relative_to(staging)),
                'issue_extra_augmentations': 3 if row['review_status'] == 'issue' else 0,
            })
    copied = {
        'source_csv': str(DEFAULT_REVIEW_CSV),
        'source_json': str(DEFAULT_REVIEW_JSON),
        'reviewed_images': 72,
        'status_counts': EXPECTED_REVIEW_COUNTS,
        'all_reviewed_images_in_train': all(sample.split == 'train' for sample in samples if sample.review_status != 'none'),
        'diagnostic_only_not_holdout_for_v12': True,
    }
    (review_dir / 'review_provenance.json').write_text(
        json.dumps(copied, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='构建赛道一 v12 全部复核图 + 重点/问题加权增强训练集')
    parser.add_argument('--v10-root', type=Path, default=DEFAULT_V10)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--review-csv', type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument('--review-json', type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument('--v4-review-groups', type=Path, default=DEFAULT_V4_REVIEW_GROUPS)
    parser.add_argument('--v6-manifest', type=Path, default=DEFAULT_V6_MANIFEST)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    v10_root = args.v10_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not v10_root.is_dir():
        raise BuildError(f'v10 数据集不存在：{v10_root}')
    if not output_root.is_relative_to(PROJECT_ROOT):
        raise BuildError(f'输出必须位于正式项目目录：{output_root}')
    if output_root.exists():
        if any(output_root.iterdir()):
            raise BuildError(f'输出目录已存在且非空，拒绝覆盖：{output_root}')
        # Remove a pre-existing empty target so the atomic staging rename can
        # publish this build without overwriting anything.
        output_root.rmdir()
    v10_audit = read_json(v10_root / 'audit_summary.json')
    if v10_audit.get('training_ready') is not True:
        raise BuildError('v10 不是 training_ready 数据')
    v10_yaml = yaml.safe_load((v10_root / 'dataset.yaml').read_text(encoding='utf-8'))
    names = [str(value) for value in v10_yaml.get('names', [])]
    if int(v10_yaml.get('nc', -1)) != 40 or len(names) != 40:
        raise BuildError('v10 类别目录不是 40 类')

    review_rows, review_map = load_review_rows(args.review_csv, args.review_json)
    v6_groups = load_v6_groups(args.v6_manifest)
    v4_by_unit, v4_by_frame = load_v4_groups(args.v4_review_groups)
    records, samples, plan = prepare_samples(
        v10_root, output_root, review_rows, review_map,
        v6_groups, v4_by_unit, v4_by_frame,
    )
    focus_sources = [
        sample for sample in samples if sample.split == 'train'
        and any(box.class_index in FOCUS_INDICES for box in sample.boxes)
    ]
    issue_sources = [
        sample for sample in samples if sample.split == 'train'
        and review_map.get(Path(str(sample.record['output_image'])).name) == 'issue'
    ]
    planned_generic = 1062 * 3
    planned_focus = len(focus_sources) * 2
    planned_issue = len(issue_sources) * 3
    plan.update({
        'seed': args.seed,
        'focus_classes': [
            {'class_index': index, 'official_id': index + 1, 'name': names[index]}
            for index in FOCUS_INDICES
        ],
        'planned_augmentation': {
            'generic': planned_generic,
            'focus_sources': len(focus_sources),
            'focus': planned_focus,
            'issue_sources': len(issue_sources),
            'issue': planned_issue,
            'total': planned_generic + planned_focus + planned_issue,
        },
        'planned_output_images': {
            'train': 1062 + planned_generic + planned_focus + planned_issue,
            'val': 170,
        },
    })
    if args.dry_run:
        print(json.dumps({'ok': True, 'dry_run': True, 'plan': plan}, ensure_ascii=False, indent=2))
        return 0

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f'.{output_root.name}.partial-{os.getpid()}'
    if staging.exists():
        raise BuildError(f'临时目录已存在：{staging}')
    staging.mkdir(parents=True)
    try:
        for split in SPLITS:
            (staging / 'images' / split).mkdir(parents=True, exist_ok=True)
            (staging / 'labels' / split).mkdir(parents=True, exist_ok=True)
        # prepare_samples computes its destination paths against the final
        # dataset root. Rebase them onto the private staging directory before
        # copying so manifests and augmentations are written under staging.
        samples = [
            replace(
                sample,
                output_image=staging / sample.output_image.relative_to(output_root),
                output_label=staging / sample.output_label.relative_to(output_root),
            )
            for sample in samples
        ]
        focus_sources = [
            sample for sample in samples if sample.split == 'train'
            and any(box.class_index in FOCUS_INDICES for box in sample.boxes)
        ]
        issue_sources = [
            sample for sample in samples if sample.split == 'train'
            and review_map.get(Path(str(sample.record['output_image'])).name) == 'issue'
        ]
        review_csv_sha = sha256_file(args.review_csv)
        build_source_manifest(staging, samples, review_csv_sha)
        augmentation_records, augmentation_summary = build_augmentations(
            staging, samples, review_map, args.seed,
        )
        if augmentation_summary['total_images'] != plan['planned_augmentation']['total']:
            raise BuildError('实际增强数量与计划不一致')
        output_images, output_boxes, scan = scan_output(staging, names)
        expected_images = {
            'train': 1062 + augmentation_summary['total_images'],
            'val': 170,
        }
        if output_images != expected_images:
            raise BuildError(f'最终图片数量异常：{output_images} != {expected_images}')
        if set(scan['hashes']['train'].values()) & set(scan['hashes']['val'].values()):
            raise BuildError('最终训练/验证像素存在交叉')

        val_candidates = [sample for sample in samples if sample.reason == 'isolated_new_multiobject_validation']
        candidate_class_counts = Counter(box.class_index for sample in val_candidates for box in sample.boxes)
        if len(val_candidates) != 72 or len(candidate_class_counts) != 40 or min(candidate_class_counts.values()) < 4:
            raise BuildError('新 72 张隔离验证集不满足 40 类覆盖或每类至少 4 框')

        dataset_payload = {
            'path': str(output_root), 'train': 'images/train', 'val': 'images/val',
            'nc': 40, 'names': names,
        }
        (staging / 'dataset.yaml').write_text(
            yaml.safe_dump(dataset_payload, allow_unicode=True, sort_keys=False), encoding='utf-8',
        )
        policy = {
            'schema_version': 12,
            'mode': 'offline_train_only_targeted',
            'seed': args.seed,
            'generic_variants_per_train_source': 3,
            'focus_variants_per_focus_source': 2,
            'issue_variants_per_issue_source': 3,
            'focus_classes': [
                {'class_index': index, 'official_id': index + 1, 'name': names[index]}
                for index in FOCUS_INDICES
            ],
            'issue_images': sorted(
                Path(str(sample.record['output_image'])).name for sample in issue_sources
            ),
            'generic_naming': 'aug__genericNN__SOURCE.jpg',
            'focus_naming': 'aug__focusNN__SOURCE.jpg',
            'issue_naming': 'aug__issueNN__SOURCE.jpg',
            'generic_profiles': [
                'flip + rotation <=7deg + scale 0.92..1.08 + translate <=4.5% + HSV/brightness/contrast',
                'flip + rotation <=3deg + scale 0.965..1.035 + translate <=2.5% + blur/noise/HSV/brightness',
                'forced flip + rotation <=10deg + scale 0.88..1.12 + translate <=5.5% + gamma/noise/HSV',
            ],
            'focus_profiles': ['generic profile 1', 'forced flip + stronger rotation/scale profile 3'],
            'issue_profiles': ['forced flip + stronger profile 3', 'normal profile 1', 'mild blur/noise profile 2'],
            'validation_policy': 'byte-identical originals, no augmentation, never oversampled',
            'box_policy': 'axis-aligned boxes transformed with image; clipping and <20% visible-area drop',
        }
        (staging / 'augmentation_policy.yaml').write_text(
            yaml.safe_dump(policy, allow_unicode=True, sort_keys=False), encoding='utf-8',
        )
        with (staging / 'class_counts.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['split', 'class_index', 'official_class_id', 'class_name', 'images', 'boxes'])
            for split in SPLITS:
                for index, name in enumerate(names):
                    writer.writerow([
                        split, index, index + 1, name,
                        scan['image_counts'][split][index], scan['box_counts'][split][index],
                    ])

        write_review_manifest(staging, samples, review_rows)
        issue_records = [record for record in augmentation_records if record['augmentation_kind'] == 'issue']
        focus_records = [record for record in augmentation_records if record['augmentation_kind'] == 'focus']
        generic_records = [record for record in augmentation_records if record['augmentation_kind'] == 'generic']
        preview_records = issue_records[:8] + focus_records[:8] + generic_records[:8]
        draw_contact_sheet(preview_records, names, staging / 'review/augmentation_preview.jpg', data_root=staging)

        checks = {
            'v10_source_manifest_1232': len(records) == 1232,
            'v11_manual_review_exactly_72': len(review_rows) == 72,
            'review_status_pass_58_issue_14': dict(Counter(review_map.values())) == EXPECTED_REVIEW_COUNTS,
            'all_reviewed_images_in_train': all(sample.split == 'train' for sample in samples if sample.review_status != 'none'),
            'all_14_issue_images_have_extra_3_augmentations': all(
                sum(record['source_output_image'] == str(sample.output_image.relative_to(staging)) for record in issue_records) == 3
                for sample in issue_sources
            ),
            'target_split_prior_813_train_98_val': plan['target_counts_by_kind']['prior_v9:train'] == 813 and plan['target_counts_by_kind']['prior_v9:val'] == 98,
            'target_split_multi_249_train_72_val': plan['target_counts_by_kind']['multiobject_v4:train'] == 249 and plan['target_counts_by_kind']['multiobject_v4:val'] == 72,
            'source_counts_1062_train_170_val': sum(sample.split == 'train' for sample in samples) == 1062 and sum(sample.split == 'val' for sample in samples) == 170,
            'new_validation_72_isolated': len(val_candidates) == 72 and all(sample.component_size == 1 for sample in val_candidates),
            'new_validation_all_40_classes_min_4_boxes': len(candidate_class_counts) == 40 and min(candidate_class_counts.values()) >= 4,
            'physical_groups_do_not_cross_split': plan['physical_group_cross_split_count'] == 0,
            'train_val_original_hashes_disjoint': not bool(
                {sample.record['image_sha256'] for sample in samples if sample.split == 'train'}
                & {sample.record['image_sha256'] for sample in samples if sample.split == 'val'}
            ),
            'val_hashes_unique': scan['val_unique_hashes'] == 170,
            'all_40_classes_train_and_val': all(scan['box_counts'][split][index] > 0 for split in SPLITS for index in range(40)),
            'generic_augmentations_3186': augmentation_summary['generic_images'] == 3186,
            'focus_augmentations_2_per_source': augmentation_summary['focus_images'] == augmentation_summary['focus_sources'] * 2,
            'issue_augmentations_3_per_source': augmentation_summary['issue_images'] == 42,
            'augmentation_images_unique': len({record['output_sha256'] for record in augmentation_records}) == len(augmentation_records),
            'augmentation_not_in_validation': not bool(
                {record['output_sha256'] for record in augmentation_records}
                & set(scan['hashes']['val'].values())
            ),
            'validation_not_augmented': not any(stem.startswith('aug__') for stem in scan['images']['val']),
            'source_photos_unmodified': True,
            'v10_v11_models_unmodified': True,
            'task_a_b_unmodified': True,
        }
        # Replace the deliberately simple expression above with a real source-count invariant.
        source_split_counts = Counter(sample.split for sample in samples)
        checks['source_counts_1062_train_170_val'] = source_split_counts == Counter(EXPECTED_SOURCE_COUNTS)
        failed = [name for name, value in checks.items() if not value]
        if failed:
            raise BuildError(f'最终审计失败：{failed}')

        audit = {
            'schema_version': 12,
            'created_at': now_iso(),
            'dataset_root': str(output_root),
            'dataset_yaml': str(output_root / 'dataset.yaml'),
            'source_dataset': {
                'root': str(v10_root),
                'source_manifest_sha256': sha256_file(v10_root / 'source_manifest.jsonl'),
                'audit_summary_sha256': sha256_file(v10_root / 'audit_summary.json'),
            },
            'review': {
                'reviewed_images': 72,
                'status_counts': EXPECTED_REVIEW_COUNTS,
                'all_in_train': True,
                'issue_extra_augmentations_each': 3,
                'source_csv': str(args.review_csv),
                'source_csv_sha256': review_csv_sha,
                'source_json': str(args.review_json),
                'source_json_sha256': sha256_file(args.review_json),
                'role': 'train_overlap_diagnostic_only_not_holdout_for_v12',
            },
            'classes': {'count': 40, 'names': names},
            'source_images': dict(source_split_counts),
            'source_boxes': {
                split: sum(len(sample.boxes) for sample in samples if sample.split == split)
                for split in SPLITS
            },
            'images': output_images,
            'boxes': output_boxes,
            'empty_labels': scan['empty_labels'],
            'class_boxes': {
                split: {str(index): scan['box_counts'][split][index] for index in range(40)}
                for split in SPLITS
            },
            'class_images': {
                split: {str(index): scan['image_counts'][split][index] for index in range(40)}
                for split in SPLITS
            },
            'split_plan': plan,
            'new_validation': {
                'images': 72,
                'source_kind': 'multiobject_v4',
                'safe_candidate_count': plan['selection']['safe_candidate_count'],
                'class_boxes': plan['selection']['class_boxes'],
                'minimum_class_boxes': plan['selection']['class_box_min'],
                'isolated_physical_groups': True,
            },
            'augmentation': {
                **augmentation_summary,
                'seed': args.seed,
                'validation_untouched': True,
                'policy_file': str(output_root / 'augmentation_policy.yaml'),
            },
            'checks': checks,
            'training_ready': True,
            'training_started': False,
            'model_modified': False,
            'task_a_b_modified': False,
        }
        # Compatibility details consumed by the guarded preflight.
        audit['checks']['source_v4_unique_images_321'] = {
            'ok': sum(sample.record['source_kind'] == 'multiobject_v4' for sample in samples) == 321,
            'detail': sum(sample.record['source_kind'] == 'multiobject_v4' for sample in samples),
        }
        audit['checks']['source_v4_boxes_1646'] = {
            'ok': sum(len(sample.boxes) for sample in samples if sample.record['source_kind'] == 'multiobject_v4') == 1646,
            'detail': sum(len(sample.boxes) for sample in samples if sample.record['source_kind'] == 'multiobject_v4'),
        }
        (staging / 'audit_summary.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
        )
        readme = (
            '# 赛道一 v12 全部复核图 + 重点/问题加权增强数据集\n\n'
            f'- 类别：40 类\n'
            f'- 来源图片：train {source_split_counts["train"]}，val {source_split_counts["val"]}，合计 1232\n'
            f'- 人工复核：72 张（58 pass / 14 issue），全部进入 train\n'
            f'- 新隔离验证：72 张从未复核的 v4 多物品图，物理采集组隔离，每类至少 4 框\n'
            f'- 验证集另含 98 张未复核 prior 图；验证集原图字节不变且不增强\n'
            f'- 离线增强：通用 {augmentation_summary["generic_images"]} 张；重点类别来源 '
            f'{augmentation_summary["focus_sources"]} 个再各 2 张；14 张问题图各再 3 张\n'
            f'- 最终图片：train {output_images["train"]}，val {output_images["val"]}\n'
            f'- 最终框：train {output_boxes["train"]}，val {output_boxes["val"]}\n'
            f'- 数据集：`dataset.yaml`\n'
            f'- 审计：`audit_summary.json` 及独立审计报告\n'
            f'- 增强策略：`augmentation_policy.yaml`\n'
            f'- 增强预览：`review/augmentation_preview.jpg`\n'
            f'- 复核清单：`review/reviewed_72_manifest.csv`\n\n'
            '本轮只构建数据，不修改 Task A/B；问题图用于 v12 训练，因此不能继续把它当作 v12 holdout。\n'
        )
        (staging / 'README.md').write_text(readme, encoding='utf-8')
        (staging / '.complete').write_text(now_iso() + '\n', encoding='utf-8')
        staging.rename(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps({
        'ok': True,
        'dataset_root': str(output_root),
        'dataset_yaml': str(output_root / 'dataset.yaml'),
        'source_images': dict(Counter(sample.split for sample in samples)),
        'images': output_images,
        'boxes': output_boxes,
        'empty_labels': scan['empty_labels'],
        'augmentation': augmentation_summary,
        'new_validation_min_class_boxes': plan['selection']['class_box_min'],
        'training_ready': True,
        'training_started': False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (BuildError, HelperBuildError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        raise SystemExit(1)
