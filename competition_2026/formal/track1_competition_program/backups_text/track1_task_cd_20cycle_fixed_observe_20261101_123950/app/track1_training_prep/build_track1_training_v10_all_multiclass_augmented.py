#!/usr/bin/env python3
"""Build Track-1 v10: merge all verified 40-class data and augment train only.

Inputs:
  * v9 hard-negative/oversampled dataset (latest audited 40-class dataset)
  * v4 all-history multi-object human-reviewed YOLO export

The validation images are copied byte-for-byte and are never augmented.
Every training image receives three deterministic offline variants in addition
to the original training entry. Source photos, reviews, prior datasets and
models are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(os.environ.get(
    'TRACK1_PROJECT_ROOT',
    '/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序',
))
DATA_VOLUME = Path(os.environ.get('TRACK1_DATA_VOLUME', '/media/sunshine/新加卷'))
DEFAULT_PRIOR = PROJECT_ROOT / 'datasets/track1_training_merged_v9_hardneg'
DEFAULT_MULTI = PROJECT_ROOT / 'datasets/track1_all_history_multi_object_bbox_export_20261031_v4'
DEFAULT_OUTPUT = PROJECT_ROOT / 'datasets/track1_training_merged_v10_all_multiclass_augmented_20261031'
SPLITS = ('train', 'val')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SAFE_RE = re.compile(r'[^0-9A-Za-z._-]+')
INTXLNK_MAGIC = b'IntxLNK\x01'
FRAME_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')
BURST_RE = re.compile(r'(?:^|\|)burst:([^|;]+)')


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class Box:
    class_index: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class SourceSample:
    source_kind: str
    source_root: Path
    split: str
    stem: str
    image_entry: Path
    label_entry: Path
    image: Path
    label: Path
    boxes: tuple[Box, ...]
    image_sha256: str
    label_authority: str


@dataclass(frozen=True)
class CopiedSample:
    source: SourceSample
    output_stem: str
    output_image: Path
    output_label: Path
    output_sha256: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_entry(path: Path, *, context: str) -> Path:
    """Resolve real symlinks and WSL IntxLNK files to their physical target.

    The exFAT data volume stores links created through WSL as small regular
    files with an ``IntxLNK\\x01`` header followed by a UTF-16LE absolute
    path. Opening those files directly yields the link record rather than the
    image/label bytes, so every source entry must be resolved explicitly.
    """
    current = path.absolute()
    seen: set[str] = set()
    volume = DATA_VOLUME.resolve(strict=True)
    for _depth in range(16):
        try:
            resolved = current.resolve(strict=True)
        except FileNotFoundError as exc:
            raise BuildError(f'{context} 指向不存在的目标：{current}') from exc
        if not resolved.is_file():
            raise BuildError(f'{context} 不是普通文件：{resolved}')
        if not resolved.is_relative_to(volume):
            raise BuildError(f'{context} 目标位于数据盘之外：{resolved}')
        key = str(resolved)
        if key in seen:
            raise BuildError(f'{context} 出现链接循环：{resolved}')
        seen.add(key)
        with resolved.open('rb') as handle:
            magic = handle.read(len(INTXLNK_MAGIC))
        if magic != INTXLNK_MAGIC:
            return resolved
        payload = resolved.read_bytes()[len(INTXLNK_MAGIC):]
        try:
            decoded = payload.decode('utf-16le').rstrip('\x00')
        except UnicodeDecodeError as exc:
            raise BuildError(f'{context} 的 IntxLNK 目标无法解码：{resolved}') from exc
        if not decoded:
            raise BuildError(f'{context} 的 IntxLNK 目标为空：{resolved}')
        target = Path(decoded)
        if not target.is_absolute():
            target = resolved.parent / target
        current = target.absolute()
    raise BuildError(f'{context} 的链接层级过深：{path}')


def safe_component(value: str, *, limit: int = 190) -> str:
    cleaned = SAFE_RE.sub('_', value).strip('._')
    if not cleaned:
        raise BuildError(f'无法生成安全文件名：{value!r}')
    return cleaned[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f'缺少 JSON：{path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise BuildError(f'JSON 顶层不是对象：{path}')
    return value


def load_dataset_yaml(root: Path) -> tuple[Path, list[str]]:
    path = root / 'dataset.yaml'
    if not path.is_file():
        path = root / 'data.yaml'
    if not path.is_file():
        raise BuildError(f'缺少 dataset.yaml/data.yaml：{root}')
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    names = payload.get('names') if isinstance(payload, dict) else None
    if isinstance(names, dict):
        try:
            names = [str(names[index]) for index in range(len(names))]
        except KeyError as exc:
            raise BuildError(f'{path} 的 names 索引不连续') from exc
    if not isinstance(names, list) or len(names) != 40:
        raise BuildError(f'{path} 应为 40 类，实际 {len(names) if isinstance(names, list) else type(names)}')
    return path, [str(item) for item in names]


def parse_label(path: Path, *, context: str) -> tuple[Box, ...]:
    rows: list[Box] = []
    for line_number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise BuildError(f'{context}:{line_number} 不是 YOLO 五列')
        try:
            class_index = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise BuildError(f'{context}:{line_number} 含非数字') from exc
        if not 0 <= class_index < 40:
            raise BuildError(f'{context}:{line_number} 类别越界：{class_index}')
        values = (x, y, width, height)
        if not all(np.isfinite(value) for value in values):
            raise BuildError(f'{context}:{line_number} 坐标不是有限数')
        if not all(0.0 <= value <= 1.0 for value in values):
            raise BuildError(f'{context}:{line_number} 坐标不在 0..1：{values}')
        if width <= 0.0 or height <= 0.0:
            raise BuildError(f'{context}:{line_number} 宽高必须大于 0')
        epsilon = 1e-6
        if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon:
            raise BuildError(f'{context}:{line_number} 横向越界')
        if y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
            raise BuildError(f'{context}:{line_number} 纵向越界')
        if width >= 0.98 and height >= 0.98:
            raise BuildError(f'{context}:{line_number} 疑似整图伪框')
        rows.append(Box(class_index, x, y, width, height))
    return tuple(rows)


def discover_source(root: Path, source_kind: str) -> tuple[list[str], list[SourceSample]]:
    yaml_path, names = load_dataset_yaml(root)
    del yaml_path
    samples: list[SourceSample] = []
    for split in SPLITS:
        image_dir = root / 'images' / split
        label_dir = root / 'labels' / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise BuildError(f'{source_kind}/{split} 缺少 images 或 labels 目录')
        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        labels = {
            path.stem: path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == '.txt'
        }
        if set(images) != set(labels):
            missing_images = sorted(set(labels) - set(images))
            missing_labels = sorted(set(images) - set(labels))
            raise BuildError(
                f'{source_kind}/{split} 图片标签不配对；缺图={missing_images[:5]} '
                f'缺标签={missing_labels[:5]}'
            )
        for stem in sorted(images):
            image_entry = images[stem]
            label_entry = labels[stem]
            image = resolve_data_entry(image_entry, context=f'{source_kind}/{split}/image/{stem}')
            label = resolve_data_entry(label_entry, context=f'{source_kind}/{split}/label/{stem}')
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                raise BuildError(f'{source_kind}/{split}/{stem} 解析后不是支持图片：{image}')
            if label.suffix.lower() != '.txt':
                raise BuildError(f'{source_kind}/{split}/{stem} 解析后不是标签文件：{label}')
            boxes = parse_label(label, context=f'{source_kind}/{split}/{label.name}')
            samples.append(SourceSample(
                source_kind=source_kind,
                source_root=root,
                split=split,
                stem=stem,
                image_entry=image_entry,
                label_entry=label_entry,
                image=image,
                label=label,
                boxes=boxes,
                image_sha256=sha256_file(image),
                label_authority=source_kind,
            ))
    return names, samples


def audit_group_splits(
    prior_root: Path,
    multi_root: Path,
    samples: Iterable[SourceSample],
) -> tuple[int, int, dict[str, set[str]]]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    prior_manifest_records = 0
    prior_manifest_groups: set[str] = set()
    for name in ('merge_manifest.jsonl', 'v6_merge_manifest.jsonl'):
        prior_manifest = prior_root / name
        if not prior_manifest.is_file():
            continue
        resolved_manifest = resolve_data_entry(prior_manifest, context=f'采集组清单/{name}')
        for line_number, line in enumerate(resolved_manifest.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f'{resolved_manifest}:{line_number} JSON 无效') from exc
            prior_manifest_records += 1
            group = str(record.get('group') or '').strip()
            split = str(record.get('source_split') or '').strip()
            if group and split in SPLITS:
                named_group = f'prior:{group}'
                group_splits[named_group].add(split)
                prior_manifest_groups.add(named_group)
    if prior_manifest_records == 0 or not prior_manifest_groups:
        raise BuildError('无法从 v9 来源清单读取物理采集组')
    review_groups = multi_root.parent / 'track1_all_history_multi_object_review_20261031_v4/manifests/review_groups.csv'
    if review_groups.is_file():
        with review_groups.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                split = str(row.get('split') or '').strip()
                if split not in SPLITS:
                    continue
                for group in BURST_RE.findall(str(row.get('raw_group_keys') or '')):
                    group_splits[f'multi:{group}'].add(split)
    crossing = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if crossing:
        raise BuildError(f'物理采集组跨 train/val：{list(crossing.items())[:10]}')
    covered_sources = {
        str(sample.image.resolve())
        for sample in samples
        if sample.source_kind == 'prior_v9'
    }
    if not covered_sources:
        raise BuildError('基础训练集来源为空')
    if not group_splits:
        raise BuildError('无法读取任何物理采集组')
    return len(group_splits), len(covered_sources), dict(group_splits)


def sample_frame_id(sample: SourceSample) -> str:
    frame_ids = sorted(set(FRAME_RE.findall(sample.image.stem)))
    if len(frame_ids) != 1:
        raise BuildError(
            f'无法从图片来源唯一提取 6 位帧号：{sample.source_kind}/{sample.image.name} -> {frame_ids}'
        )
    return frame_ids[0]


def reconcile_multiobject_splits(
    prior_samples: list[SourceSample],
    multi_samples: list[SourceSample],
    multi_root: Path,
) -> tuple[list[SourceSample], dict[str, Any]]:
    """Keep reviewed v4 photos on the split already used by audited v9.

    Several raw frames occur in both source datasets. Their old v4 review unit
    split can disagree with v9. The same physical burst must not cross
    train/val, so v9's audited split is authoritative and the complete v4
    burst is moved together when an overlap exists.
    """
    frame_to_splits: dict[str, set[str]] = defaultdict(set)
    for sample in prior_samples:
        frame_to_splits[sample_frame_id(sample)].add(sample.split)
    frame_conflicts = {frame: splits for frame, splits in frame_to_splits.items() if len(splits) > 1}
    if frame_conflicts:
        raise BuildError(f'v9 同一帧号跨 train/val：{list(frame_conflicts.items())[:10]}')

    review_groups = multi_root.parent / 'track1_all_history_multi_object_review_20261031_v4/manifests/review_groups.csv'
    review_groups = resolve_data_entry(review_groups, context='多物品复核采集组清单')
    frame_to_groups: dict[str, set[str]] = defaultdict(set)
    group_original_splits: dict[str, set[str]] = defaultdict(set)
    with review_groups.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            split = str(row.get('split') or '').strip()
            if split not in SPLITS:
                raise BuildError(f'多物品复核采集组含非法 split：{split!r}')
            frame_ids = {item.strip() for item in str(row.get('frame_ids') or '').split(',') if item.strip()}
            groups = set(BURST_RE.findall(str(row.get('raw_group_keys') or '')))
            if not frame_ids or not groups:
                raise BuildError(f'多物品复核采集组缺帧号或 burst：{row.get("unit_id")!r}')
            for group in groups:
                group_original_splits[group].add(split)
                for frame_id in frame_ids:
                    frame_to_groups[frame_id].add(group)
    if not group_original_splits:
        raise BuildError('多物品复核采集组为空')

    group_v9_splits: dict[str, set[str]] = defaultdict(set)
    for frame_id, groups in frame_to_groups.items():
        for split in frame_to_splits.get(frame_id, set()):
            for group in groups:
                group_v9_splits[group].add(split)
    conflicted_groups = {group: splits for group, splits in group_v9_splits.items() if len(splits) > 1}
    if conflicted_groups:
        raise BuildError(f'同一多物品 burst 对应 v9 多个 split：{list(conflicted_groups.items())[:10]}')

    reconciled: list[SourceSample] = []
    changed_entries: list[dict[str, str]] = []
    changed_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in multi_samples:
        frame_id = sample_frame_id(sample)
        groups = frame_to_groups.get(frame_id)
        if not groups:
            raise BuildError(f'多物品照片不在复核采集组中：{sample.image}')
        targets = {next(iter(group_v9_splits[group])) for group in groups if group in group_v9_splits}
        if len(targets) > 1:
            raise BuildError(f'多物品照片对应多个目标 split：{sample.image} -> {sorted(targets)}')
        target_split = next(iter(targets)) if targets else sample.split
        if target_split != sample.split:
            changed_entries.append({
                'image': sample.image_entry.name,
                'original_split': sample.split,
                'final_split': target_split,
                'groups': ','.join(sorted(groups)),
            })
            for group in groups:
                changed_groups[(sample.split, target_split)].add(group)
            sample = replace(sample, split=target_split)
        reconciled.append(sample)

    final_group_splits: dict[str, set[str]] = defaultdict(set)
    for sample in reconciled:
        for group in frame_to_groups[sample_frame_id(sample)]:
            final_group_splits[group].add(sample.split)
    cross_groups = {group: splits for group, splits in final_group_splits.items() if len(splits) > 1}
    if cross_groups:
        raise BuildError(f'最终多物品 burst 跨 train/val：{list(cross_groups.items())[:10]}')

    report = {
        'policy': 'v9 audited split wins for overlapping physical captures; entire v4 burst follows it',
        'reviewed_groups': len(group_original_splits),
        'reviewed_frames': len(frame_to_groups),
        'v9_frames': len(frame_to_splits),
        'entries_changed': len(changed_entries),
        'group_changes': {
            f'{source_split}->{target_split}': sorted(groups)
            for (source_split, target_split), groups in sorted(changed_groups.items())
        },
        'changed_entries': changed_entries,
        'final_group_cross_split_count': 0,
    }
    return reconciled, report


def apply_reviewed_overlap_labels(
    prior_samples: list[SourceSample],
    multi_samples: list[SourceSample],
) -> tuple[list[SourceSample], dict[str, Any]]:
    """Use v4 human-reviewed boxes for every exact-pixel overlap.

    Training keeps the existing v9 oversampling entries and gives all copies
    the reviewed label. Validation keeps exactly one representative per
    physical image, choosing the v4 reviewed export when available.
    """
    by_hash: dict[str, list[SourceSample]] = defaultdict(list)
    for sample in prior_samples + multi_samples:
        by_hash[sample.image_sha256].append(sample)

    output: list[SourceSample] = []
    authority_changes: list[dict[str, Any]] = []
    validation_deduplicated: list[dict[str, Any]] = []
    for digest, rows in sorted(by_hash.items()):
        splits = {sample.split for sample in rows}
        if len(splits) != 1:
            raise BuildError(f'相同图片在划分校正后仍跨 train/val：{digest} -> {sorted(splits)}')
        split = next(iter(splits))
        reviewed = sorted(
            (sample for sample in rows if sample.source_kind == 'multiobject_v4'),
            key=lambda sample: sample.stem,
        )
        if not reviewed:
            output.extend(sorted(rows, key=lambda sample: (sample.source_kind, sample.stem)))
            continue

        reviewed_box_variants = {tuple(sample.boxes) for sample in reviewed}
        if len(reviewed_box_variants) != 1:
            raise BuildError(f'同一图片存在多套 v4 人工框：{[sample.stem for sample in reviewed]}')
        representative = reviewed[0]
        reviewed_boxes = representative.boxes
        authority = f'multiobject_v4:{representative.stem}'

        labeled_prior = [
            replace(sample, boxes=reviewed_boxes, label_authority=authority)
            for sample in sorted((sample for sample in rows if sample.source_kind == 'prior_v9'), key=lambda sample: sample.stem)
        ]
        labeled_multi = [
            replace(sample, boxes=reviewed_boxes, label_authority=authority)
            for sample in reviewed
        ]
        if labeled_prior:
            authority_changes.append({
                'image_sha256': digest,
                'split': split,
                'label_authority': authority,
                'prior_entries_relabeled': [sample.stem for sample in labeled_prior],
                'prior_label_variants': sorted({str(tuple(sample.boxes)) for sample in rows if sample.source_kind == 'prior_v9'}),
                'reviewed_box_count': len(reviewed_boxes),
            })

        if split == 'train':
            output.extend(labeled_multi)
            output.extend(labeled_prior)
        else:
            output.append(labeled_multi[0])
            dropped = labeled_prior + labeled_multi[1:]
            if dropped:
                validation_deduplicated.append({
                    'image_sha256': digest,
                    'retained': labeled_multi[0].stem,
                    'retained_source_kind': labeled_multi[0].source_kind,
                    'retained_label_authority': authority,
                    'dropped_entries': [
                        {'source_kind': sample.source_kind, 'stem': sample.stem}
                        for sample in sorted(dropped, key=lambda sample: (sample.source_kind, sample.stem))
                    ],
                })

    final_val_hashes = [sample.image_sha256 for sample in output if sample.split == 'val']
    duplicate_val = [digest for digest, count in Counter(final_val_hashes).items() if count > 1]
    if duplicate_val:
        raise BuildError(f'重叠标签统一后验证集仍有重复：{duplicate_val[:10]}')
    report = {
        'policy': 'v4 human-reviewed boxes override v9 labels for exact-pixel overlaps; validation is one representative per image',
        'overlap_hashes_all_splits': sum(
            bool(any(sample.source_kind == 'prior_v9' for sample in rows))
            and bool(any(sample.source_kind == 'multiobject_v4' for sample in rows))
            for rows in by_hash.values()
        ),
        'prior_entries_relabeled': sum(len(item['prior_entries_relabeled']) for item in authority_changes),
        'reviewed_label_authority_groups': len(authority_changes),
        'validation_images_deduplicated': len(validation_deduplicated),
        'validation_entries_dropped': sum(len(item['dropped_entries']) for item in validation_deduplicated),
        'authority_changes': authority_changes,
        'validation_deduplication': validation_deduplicated,
    }
    return output, report


def copy_source(sample: SourceSample, target_image: Path, target_label: Path) -> str:
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    source_resolved = sample.image.resolve(strict=True)
    try:
        os.link(source_resolved, target_image)
    except OSError:
        shutil.copy2(source_resolved, target_image)
    target_label.write_text(yolo_text(sample.boxes), encoding='utf-8')
    copied_hash = sha256_file(target_image)
    if copied_hash != sample.image_sha256:
        raise BuildError(f'复制后图片哈希不一致：{sample.image} -> {target_image}')
    return copied_hash


def normalized_boxes_to_xyxy(boxes: Iterable[Box], width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for box in boxes:
        x1 = (box.x - box.width / 2.0) * width
        y1 = (box.y - box.height / 2.0) * height
        x2 = (box.x + box.width / 2.0) * width
        y2 = (box.y + box.height / 2.0) * height
        rows.append((box.class_index, x1, y1, x2, y2))
    return rows


def flip_xyxy_boxes(rows: Iterable[tuple[int, float, float, float, float]], width: int) -> list[tuple[int, float, float, float, float]]:
    return [
        (class_index, width - x2, y1, width - x1, y2)
        for class_index, x1, y1, x2, y2 in rows
    ]


def transform_xyxy_boxes(
    rows: Iterable[tuple[int, float, float, float, float]],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[tuple[int, float, float, float, float]], int]:
    transformed: list[tuple[int, float, float, float, float]] = []
    dropped = 0
    for class_index, x1, y1, x2, y2 in rows:
        original_area = max(1.0, (x2 - x1) * (y2 - y1))
        corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
        points = cv2.transform(corners, matrix)[0]
        nx1 = max(0.0, min(float(width - 1), float(points[:, 0].min())))
        ny1 = max(0.0, min(float(height - 1), float(points[:, 1].min())))
        nx2 = max(0.0, min(float(width - 1), float(points[:, 0].max())))
        ny2 = max(0.0, min(float(height - 1), float(points[:, 1].max())))
        clipped_area = max(0.0, nx2 - nx1) * max(0.0, ny2 - ny1)
        if nx2 - nx1 < 2.0 or ny2 - ny1 < 2.0 or clipped_area < max(12.0, original_area * 0.20):
            dropped += 1
            continue
        transformed.append((class_index, nx1, ny1, nx2, ny2))
    return transformed, dropped


def xyxy_boxes_to_normalized(
    rows: Iterable[tuple[int, float, float, float, float]],
    width: int,
    height: int,
) -> tuple[Box, ...]:
    output: list[Box] = []
    for class_index, x1, y1, x2, y2 in rows:
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 1.0 or box_height <= 1.0:
            continue
        normalized = Box(
            class_index=class_index,
            x=((x1 + x2) / 2.0) / width,
            y=((y1 + y2) / 2.0) / height,
            width=box_width / width,
            height=box_height / height,
        )
        if normalized.width >= 0.98 and normalized.height >= 0.98:
            continue
        if normalized.x - normalized.width / 2 < -1e-6 or normalized.x + normalized.width / 2 > 1 + 1e-6:
            continue
        if normalized.y - normalized.height / 2 < -1e-6 or normalized.y + normalized.height / 2 > 1 + 1e-6:
            continue
        output.append(normalized)
    return tuple(output)


def yolo_text(boxes: Iterable[Box]) -> str:
    lines = []
    for box in boxes:
        lines.append(
            f'{box.class_index} {box.x:.6f} {box.y:.6f} '
            f'{box.width:.6f} {box.height:.6f}'
        )
    return ('\n'.join(lines) + '\n') if lines else ''


def photometric_transform(image: np.ndarray, rng: random.Random, profile: int) -> tuple[np.ndarray, dict[str, float]]:
    alpha = rng.uniform(0.86, 1.16)
    beta = rng.uniform(-16.0, 16.0)
    rgb = image.astype(np.float32) * alpha + beta
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hue_delta = rng.randint(-6, 6)
    saturation_scale = rng.uniform(0.82, 1.22)
    value_scale = rng.uniform(0.88, 1.12)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_delta) % 180).astype(np.uint8)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * saturation_scale, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * value_scale, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gamma = 1.0
    if profile == 2:
        gamma = rng.uniform(0.88, 1.14)
        lut = np.array([((value / 255.0) ** gamma) * 255.0 for value in range(256)], dtype=np.uint8)
        rgb = cv2.LUT(rgb, lut)
    blur_kernel = 0
    if profile == 1 and rng.random() < 0.28:
        blur_kernel = rng.choice((3, 5))
        rgb = cv2.GaussianBlur(rgb, (blur_kernel, blur_kernel), 0)
    noise_sigma = 0.0
    if profile == 2 and rng.random() < 0.22:
        noise_sigma = rng.uniform(1.5, 5.0)
        noise = np.random.default_rng(rng.randrange(0, 2**32)).normal(0.0, noise_sigma, rgb.shape)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return rgb, {
        'brightness_contrast_alpha': round(alpha, 6),
        'brightness_contrast_beta': round(beta, 6),
        'hue_delta': float(hue_delta),
        'saturation_scale': round(saturation_scale, 6),
        'value_scale': round(value_scale, 6),
        'gamma': round(gamma, 6),
        'blur_kernel': float(blur_kernel),
        'noise_sigma': round(noise_sigma, 6),
    }


def augment_image(
    image: np.ndarray,
    boxes: tuple[Box, ...],
    rng: random.Random,
    profile: int,
) -> tuple[np.ndarray, tuple[Box, ...], dict[str, Any]]:
    height, width = image.shape[:2]
    if width < 32 or height < 32:
        raise BuildError(f'图片尺寸异常：{width}x{height}')
    rows = normalized_boxes_to_xyxy(boxes, width, height)
    profiles = (
        {'max_angle': 7.0, 'scale_lo': 0.92, 'scale_hi': 1.08, 'translate': 0.045, 'force_flip': False},
        {'max_angle': 3.0, 'scale_lo': 0.965, 'scale_hi': 1.035, 'translate': 0.025, 'force_flip': False},
        {'max_angle': 10.0, 'scale_lo': 0.88, 'scale_hi': 1.12, 'translate': 0.055, 'force_flip': True},
    )
    policy = profiles[profile]
    do_flip = bool(policy['force_flip'] or rng.random() < 0.52)
    if do_flip:
        image = cv2.flip(image, 1)
        rows = flip_xyxy_boxes(rows, width)
    angle = rng.uniform(-float(policy['max_angle']), float(policy['max_angle']))
    scale = rng.uniform(float(policy['scale_lo']), float(policy['scale_hi']))
    translate_x = rng.uniform(-float(policy['translate']), float(policy['translate'])) * width
    translate_y = rng.uniform(-float(policy['translate']), float(policy['translate'])) * height
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y
    transformed_rows, dropped = transform_xyxy_boxes(rows, matrix, width, height)
    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    image, photo_record = photometric_transform(image, rng, profile)
    transformed_boxes = xyxy_boxes_to_normalized(transformed_rows, width, height)
    if boxes and not transformed_boxes:
        raise BuildError('增强后所有目标框都被裁掉')
    record = {
        'profile': profile + 1,
        'horizontal_flip': do_flip,
        'rotation_degrees': round(angle, 6),
        'scale': round(scale, 6),
        'translate_pixels': [round(translate_x, 4), round(translate_y, 4)],
        'input_box_count': len(boxes),
        'output_box_count': len(transformed_boxes),
        'dropped_box_count': dropped,
        **photo_record,
    }
    return image, transformed_boxes, record


def write_augmented_sample(
    sample: SourceSample,
    output_image: Path,
    output_label: Path,
    variant: int,
    seed: int,
) -> tuple[Box, ...]:
    image = cv2.imread(str(sample.image.resolve(strict=True)), cv2.IMREAD_COLOR)
    if image is None:
        raise BuildError(f'无法读取图片：{sample.image}')
    rng = random.Random(seed)
    augmented, boxes, _ = augment_image(image, sample.boxes, rng, variant)
    if np.array_equal(augmented, image):
        raise BuildError(f'增强结果与原图完全相同：{sample.image}')
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(output_image),
        augmented,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    if not ok or not output_image.is_file():
        raise BuildError(f'写增强图片失败：{output_image}')
    output_label.write_text(yolo_text(boxes), encoding='utf-8')
    return boxes


def draw_contact_sheet(
    records: list[dict[str, Any]],
    names: list[str],
    output: Path,
    *,
    data_root: Path,
) -> None:
    if not records:
        raise BuildError('没有可用于预览的增强记录')
    tile_width, tile_height = 320, 240
    columns, rows = 4, 6
    sheet = np.full((tile_height * rows, tile_width * columns, 3), 245, dtype=np.uint8)
    selected = records[: columns * rows]
    for index, record in enumerate(selected):
        image_path = Path(record['output_image'])
        label_path = Path(record['output_label'])
        if not image_path.is_absolute():
            image_path = data_root / image_path
        if not label_path.is_absolute():
            label_path = data_root / label_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise BuildError(f'预览图片无法读取：{image_path}')
        source_height, source_width = image.shape[:2]
        scale = min(tile_width / source_width, tile_height / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        tile = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
        offset_x = (tile_width - resized_width) // 2
        offset_y = (tile_height - resized_height) // 2
        tile[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
        for box in parse_label(label_path, context=str(label_path)):
            x1 = int(round((box.x - box.width / 2) * resized_width)) + offset_x
            y1 = int(round((box.y - box.height / 2) * resized_height)) + offset_y
            x2 = int(round((box.x + box.width / 2) * resized_width)) + offset_x
            y2 = int(round((box.y + box.height / 2) * resized_height)) + offset_y
            color = ((37 * (box.class_index + 1)) % 210 + 30,
                     (83 * (box.class_index + 1)) % 210 + 30,
                     (131 * (box.class_index + 1)) % 210 + 30)
            cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2)
            label = f'{box.class_index}:{names[box.class_index].split("-", 1)[-1]}'
            cv2.putText(tile, label, (max(2, x1), max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(tile, label, (max(2, x1), max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        row = index // columns
        column = index % columns
        sheet[row * tile_height:(row + 1) * tile_height, column * tile_width:(column + 1) * tile_width] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise BuildError(f'写预览图失败：{output}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='合并全部 40 类数据并生成训练集离线增强 v10')
    parser.add_argument('--prior-root', type=Path, default=DEFAULT_PRIOR)
    parser.add_argument('--multi-root', type=Path, default=DEFAULT_MULTI)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--augmentations-per-train', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prior_root = args.prior_root.resolve()
    multi_root = args.multi_root.resolve()
    output_root = args.output_root.resolve()
    if args.augmentations_per_train < 1 or args.augmentations_per_train > 8:
        raise BuildError('--augmentations-per-train 必须在 1..8')
    prior_audit = load_json(prior_root / 'audit_summary.json')
    multi_audit = load_json(multi_root / 'audit_summary.json')
    if prior_audit.get('training_ready') is not True:
        raise BuildError('基础 v9 数据集未通过 training_ready 审计')
    if multi_audit.get('complete_review') is not True or multi_audit.get('review_export_valid') is not True:
        raise BuildError('多物品 v4 导出不是完整有效的人工复核结果')
    if int(multi_audit.get('item_count', -1)) != 321 or int(multi_audit.get('box_count', -1)) != 1646:
        raise BuildError('多物品 v4 导出数量与已完成复核不一致')

    prior_names, prior_samples = discover_source(prior_root, 'prior_v9')
    multi_names, multi_samples = discover_source(multi_root, 'multiobject_v4')
    if prior_names != multi_names:
        raise BuildError('两个来源的 40 类名称或顺序不一致')
    names = prior_names
    original_counts_by_source = {
        ('prior_v9', 'train'): sum(sample.split == 'train' for sample in prior_samples),
        ('prior_v9', 'val'): sum(sample.split == 'val' for sample in prior_samples),
        ('multiobject_v4', 'train'): sum(sample.split == 'train' for sample in multi_samples),
        ('multiobject_v4', 'val'): sum(sample.split == 'val' for sample in multi_samples),
    }
    expected_original_counts = {
        ('prior_v9', 'train'): 812,
        ('prior_v9', 'val'): 121,
        ('multiobject_v4', 'train'): 246,
        ('multiobject_v4', 'val'): 75,
    }
    if original_counts_by_source != expected_original_counts:
        raise BuildError(
            f'来源图片数量不符合预期：实际={original_counts_by_source} 预期={expected_original_counts}'
        )
    if sum(len(sample.boxes) for sample in multi_samples) != 1646:
        raise BuildError('多物品标签框总数不是 1646')

    multi_samples, split_reconciliation = reconcile_multiobject_splits(prior_samples, multi_samples, multi_root)
    reconciled_samples = prior_samples + multi_samples
    reconciled_counts = Counter((sample.source_kind, sample.split) for sample in reconciled_samples)
    if reconciled_counts[('prior_v9', 'train')] != 812 or reconciled_counts[('prior_v9', 'val')] != 121:
        raise BuildError('v9 基础划分在合并时被意外改变')
    if reconciled_counts[('multiobject_v4', 'train')] + reconciled_counts[('multiobject_v4', 'val')] != 321:
        raise BuildError('多物品照片在划分校正时数量改变')

    samples, reviewed_overlap_labeling = apply_reviewed_overlap_labels(prior_samples, multi_samples)
    counts = Counter((sample.source_kind, sample.split) for sample in samples)
    if counts[('multiobject_v4', 'train')] + counts[('multiobject_v4', 'val')] != 321:
        raise BuildError('重叠标签统一时丢失了多物品人工复核照片')
    dropped_prior_validation = sum(
        1
        for item in reviewed_overlap_labeling['validation_deduplication']
        for dropped in item['dropped_entries']
        if dropped['source_kind'] == 'prior_v9'
    )
    if counts[('prior_v9', 'train')] != 812 or counts[('prior_v9', 'val')] != 121 - dropped_prior_validation:
        raise BuildError('v9 基础数据在重叠去重时出现非预期条目变化')

    group_count, prior_source_count, group_splits = audit_group_splits(prior_root, multi_root, samples)
    train_hashes = {sample.image_sha256 for sample in samples if sample.split == 'train'}
    val_hashes = {sample.image_sha256 for sample in samples if sample.split == 'val'}
    if train_hashes & val_hashes:
        raise BuildError(f'训练/验证存在相同图片内容：{sorted(train_hashes & val_hashes)[:5]}')
    val_hash_counts = Counter(sample.image_sha256 for sample in samples if sample.split == 'val')
    duplicate_val = [digest for digest, count in val_hash_counts.items() if count > 1]
    if duplicate_val:
        raise BuildError(f'验证集存在重复图片：{duplicate_val[:5]}')

    original_counts = {
        'train': sum(sample.split == 'train' for sample in samples),
        'val': sum(sample.split == 'val' for sample in samples),
    }
    original_boxes = {
        split: sum(len(sample.boxes) for sample in samples if sample.split == split)
        for split in SPLITS
    }
    plan = {
        'prior_root': str(prior_root),
        'multi_root': str(multi_root),
        'output_root': str(output_root),
        'classes': len(names),
        'source_images': original_counts,
        'source_boxes': original_boxes,
        'source_split_reconciliation': split_reconciliation,
        'reviewed_overlap_labeling': reviewed_overlap_labeling,
        'unique_source_hashes': {'train': len(train_hashes), 'val': len(val_hashes)},
        'duplicate_train_entries': original_counts['train'] - len(train_hashes),
        'augmentations_per_train': args.augmentations_per_train,
        'planned_augmented_train': original_counts['train'] * args.augmentations_per_train,
        'planned_output_images': {
            'train': original_counts['train'] * (args.augmentations_per_train + 1),
            'val': original_counts['val'],
        },
        'physical_groups': group_count,
        'prior_unique_resolved_sources': prior_source_count,
    }
    if args.dry_run:
        print(json.dumps({'ok': True, 'dry_run': True, 'plan': plan}, ensure_ascii=False, indent=2))
        return 0

    if output_root.exists() and any(output_root.iterdir()):
        raise BuildError(f'输出目录已存在且非空，拒绝覆盖：{output_root}')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f'.{output_root.name}.partial-{os.getpid()}'
    if staging.exists():
        raise BuildError(f'临时目录已存在：{staging}')
    staging.mkdir(parents=True)
    try:
        for split in SPLITS:
            (staging / 'images' / split).mkdir(parents=True, exist_ok=True)
            (staging / 'labels' / split).mkdir(parents=True, exist_ok=True)
        source_manifest_path = staging / 'source_manifest.jsonl'
        augmentation_manifest_path = staging / 'augmentation_manifest.jsonl'
        copied_by_source: dict[tuple[str, str, str], CopiedSample] = {}
        original_hashes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
        with source_manifest_path.open('w', encoding='utf-8') as source_manifest:
            for sample in samples:
                prefix = 'prior' if sample.source_kind == 'prior_v9' else 'multiobj'
                output_stem = f'{prefix}__{safe_component(sample.stem)}'
                if any(
                    existing.source.split == sample.split and existing.output_stem == output_stem
                    for existing in copied_by_source.values()
                ):
                    raise BuildError(f'输出文件名冲突：{sample.split}/{output_stem}')
                output_image = staging / 'images' / sample.split / f'{output_stem}{sample.image.suffix.lower()}'
                output_label = staging / 'labels' / sample.split / f'{output_stem}.txt'
                image_hash = copy_source(sample, output_image, output_label)
                original_hashes_by_split[sample.split].add(image_hash)
                copied = CopiedSample(sample, output_stem, output_image, output_label, image_hash)
                copied_by_source[(sample.source_kind, sample.split, sample.stem)] = copied
                source_manifest.write(json.dumps({
                    'source_kind': sample.source_kind,
                    'source_root': str(sample.source_root),
                    'split': sample.split,
                    'source_image_entry': str(sample.image_entry),
                    'source_label_entry': str(sample.label_entry),
                    'source_image': str(sample.image),
                    'source_label': str(sample.label),
                    'label_authority': sample.label_authority,
                    'output_image': str(output_image.relative_to(staging)),
                    'output_label': str(output_label.relative_to(staging)),
                    'image_sha256': image_hash,
                    'box_count': len(sample.boxes),
                    'boxes': [
                        {
                            'class_index': box.class_index,
                            'x': box.x,
                            'y': box.y,
                            'width': box.width,
                            'height': box.height,
                        }
                        for box in sample.boxes
                    ],
                }, ensure_ascii=False) + '\n')
        if original_hashes_by_split['train'] & original_hashes_by_split['val']:
            raise BuildError('写入后训练/验证原图哈希发生交叉')

        augmentation_records: list[dict[str, Any]] = []
        train_sources = sorted(
            (sample for sample in samples if sample.split == 'train'),
            key=lambda sample: (sample.source_kind, sample.stem),
        )
        with augmentation_manifest_path.open('w', encoding='utf-8') as augmentation_manifest:
            for sample_index, sample in enumerate(train_sources):
                copied = copied_by_source[(sample.source_kind, sample.split, sample.stem)]
                for variant in range(args.augmentations_per_train):
                    seed = int(args.seed) + sample_index * 1009 + variant * 7919
                    output_image = staging / 'images/train' / f'aug__v{variant + 1:02d}__{copied.output_stem}.jpg'
                    output_label = staging / 'labels/train' / f'aug__v{variant + 1:02d}__{copied.output_stem}.txt'
                    output_boxes = write_augmented_sample(sample, output_image, output_label, variant, seed)
                    record = {
                        'source_image': str(sample.image.resolve()),
                        'label_authority': sample.label_authority,
                        'source_output_image': str(copied.output_image.relative_to(staging)),
                        'output_image': str(output_image.relative_to(staging)),
                        'output_label': str(output_label.relative_to(staging)),
                        'split': 'train',
                        'variant': variant + 1,
                        'seed': seed,
                        'source_box_count': len(sample.boxes),
                        'output_box_count': len(output_boxes),
                        'output_sha256': sha256_file(output_image),
                    }
                    augmentation_records.append(record)
                    augmentation_manifest.write(json.dumps(record, ensure_ascii=False) + '\n')

        expected_augmented = original_counts['train'] * args.augmentations_per_train
        if len(augmentation_records) != expected_augmented:
            raise BuildError(f'增强数量不符：{len(augmentation_records)} != {expected_augmented}')
        augmented_hashes = [record['output_sha256'] for record in augmentation_records]
        if len(set(augmented_hashes)) != len(augmented_hashes):
            raise BuildError('增强图片出现重复内容')
        if set(augmented_hashes) & original_hashes_by_split['val']:
            raise BuildError('增强图片与验证集图片内容相同')

        output_counts: dict[str, int] = {}
        output_boxes: dict[str, int] = {}
        class_image_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
        class_box_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
        empty_label_counts: dict[str, int] = {split: 0 for split in SPLITS}
        for split in SPLITS:
            image_dir = staging / 'images' / split
            label_dir = staging / 'labels' / split
            images = {
                path.stem: path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            }
            labels = {
                path.stem: path
                for path in label_dir.iterdir()
                if path.is_file() and path.suffix.lower() == '.txt'
            }
            if set(images) != set(labels):
                raise BuildError(f'输出 {split} 图片标签不配对')
            output_counts[split] = len(images)
            output_boxes[split] = 0
            for stem in sorted(images):
                image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
                if image is None or image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2:
                    raise BuildError(f'输出图片损坏：{images[stem]}')
                boxes = parse_label(labels[stem], context=f'output/{split}/{stem}')
                if not boxes:
                    empty_label_counts[split] += 1
                for box in boxes:
                    class_box_counts[split][box.class_index] += 1
                    class_image_counts[split][box.class_index] += 1
                output_boxes[split] += len(boxes)

        expected_output_counts = {
            'train': original_counts['train'] * (args.augmentations_per_train + 1),
            'val': original_counts['val'],
        }
        if output_counts != expected_output_counts:
            raise BuildError(f'输出数量不符：{output_counts} != {expected_output_counts}')
        all_40_train_and_val = all(
            class_box_counts[split][class_index] > 0
            for split in SPLITS
            for class_index in range(40)
        )
        if not all_40_train_and_val:
            missing = {
                split: [index for index in range(40) if class_box_counts[split][index] == 0]
                for split in SPLITS
            }
            raise BuildError(f'输出数据集不是 40 类全覆盖：{missing}')

        val_source_by_stem = {
            sample.stem: sample
            for sample in samples
            if sample.split == 'val'
        }
        for sample in samples:
            if sample.split != 'val':
                continue
            prefix = 'prior' if sample.source_kind == 'prior_v9' else 'multiobj'
            output_stem = f'{prefix}__{safe_component(sample.stem)}'
            candidates = list((staging / 'images/val').glob(f'{output_stem}.*'))
            if len(candidates) != 1:
                raise BuildError(f'验证集图片缺失或重复：{output_stem}')
            if sha256_file(candidates[0]) != sample.image_sha256:
                raise BuildError(f'验证集图片被改动：{candidates[0]}')
        if any('aug__' in path.name for path in (staging / 'images/val').iterdir()):
            raise BuildError('验证集出现增强图片')

        dataset_payload = {
            'path': str(output_root),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 40,
            'names': names,
        }
        (staging / 'dataset.yaml').write_text(
            yaml.safe_dump(dataset_payload, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )

        augmentation_policy = {
            'schema_version': 1,
            'mode': 'offline_train_only',
            'variants_per_train_source': args.augmentations_per_train,
            'seed': args.seed,
            'profiles': [
                'flip + rotation <=7deg + scale 0.92..1.08 + translate <=4.5% + HSV/brightness/contrast',
                'flip + rotation <=3deg + scale 0.965..1.035 + translate <=2.5% + blur/noise/HSV/brightness',
                'forced flip + rotation <=10deg + scale 0.88..1.12 + translate <=5.5% + gamma/noise/HSV',
            ],
            'validation_policy': 'byte-identical originals, no augmentation',
            'box_policy': 'axis-aligned boxes transformed with image, clipped, boxes with <20% visible area dropped',
            'border_policy': 'constant black border; no reflected duplicate objects',
        }
        (staging / 'augmentation_policy.yaml').write_text(
            yaml.safe_dump(augmentation_policy, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )

        with (staging / 'class_counts.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['split', 'class_index', 'official_class_id', 'class_name', 'images', 'boxes'])
            for split in SPLITS:
                for class_index, name in enumerate(names):
                    writer.writerow([
                        split,
                        class_index,
                        class_index + 1,
                        name,
                        class_image_counts[split][class_index],
                        class_box_counts[split][class_index],
                    ])

        preview_records = augmentation_records[::max(1, len(augmentation_records) // 24)][:24]
        draw_contact_sheet(
            preview_records,
            names,
            staging / 'review/augmentation_preview.jpg',
            data_root=staging,
        )

        checks = {
            'prior_v9_training_ready': True,
            'multiobject_v4_complete_review': True,
            'class_catalog_exactly_40_and_identical': prior_names == multi_names,
            'all_verified_source_entries_accounted_for': (
                sum(original_counts.values()) + reviewed_overlap_labeling['validation_entries_dropped'] == 1254
            ),
            'all_reviewed_multiobject_images_imported': counts[('multiobject_v4', 'train')] + counts[('multiobject_v4', 'val')] == 321,
            'all_reviewed_multiobject_boxes_imported': sum(len(sample.boxes) for sample in multi_samples) == 1646,
            'reviewed_overlap_labels_applied': (
                reviewed_overlap_labeling['overlap_hashes_all_splits'] == 99
                and reviewed_overlap_labeling['reviewed_label_authority_groups'] == 99
                and reviewed_overlap_labeling['prior_entries_relabeled'] == 166
                and all(
                    (authority_rows := [
                        sample for sample in samples
                        if sample.image_sha256 == item['image_sha256']
                    ])
                    and len({
                        tuple(sample.boxes)
                        for sample in authority_rows
                        if sample.source_kind == 'multiobject_v4'
                    }) == 1
                    and all(
                        sample.label_authority == item['label_authority']
                        and sample.boxes == next(
                            reviewed_sample.boxes
                            for reviewed_sample in authority_rows
                            if reviewed_sample.source_kind == 'multiobject_v4'
                        )
                        for sample in authority_rows
                    )
                    for item in reviewed_overlap_labeling['authority_changes']
                )
            ),
            'image_label_pairing_complete': True,
            'all_labels_valid_yolo': True,
            'all_40_classes_have_train_and_val': all_40_train_and_val,
            'no_class_99': all(99 not in counter for counter in class_box_counts.values()),
            'train_val_source_hash_disjoint': not bool(original_hashes_by_split['train'] & original_hashes_by_split['val']),
            'train_val_exact_image_disjoint': not bool(original_hashes_by_split['train'] & original_hashes_by_split['val']),
            'validation_exact_image_unique': len(original_hashes_by_split['val']) == original_counts['val'],
            'source_physical_groups_do_not_cross_split': not any(len(splits) > 1 for splits in group_splits.values()),
            'reviewed_bursts_do_not_cross_split_after_reconciliation': split_reconciliation['final_group_cross_split_count'] == 0,
            'validation_one_to_one_unchanged': True,
            'validation_not_augmented': True,
            'augmentation_train_only': True,
            'augmented_images_unique': len(set(augmented_hashes)) == len(augmented_hashes),
            'augmented_images_do_not_duplicate_validation': not bool(set(augmented_hashes) & original_hashes_by_split['val']),
            'source_photos_read_only': True,
            'reviews_preserved': True,
            'prior_datasets_preserved': True,
            'model_not_modified': True,
            'task_a_b_not_modified': True,
        }
        if not all(checks.values()):
            raise BuildError(f'最终审计失败：{checks}')
        audit = {
            'schema_version': 1,
            'created_at': now_iso(),
            'dataset_root': str(output_root),
            'dataset_yaml': str(output_root / 'dataset.yaml'),
            'materialization': 'hardlinks_or_copies_for_originals; offline_jpeg_for_augmentations',
            'sources': {
                'prior_v9': {
                    'root': str(prior_root),
                    'audit_sha256': sha256_file(prior_root / 'audit_summary.json'),
                    'images': {'train': counts[('prior_v9', 'train')], 'val': counts[('prior_v9', 'val')]},
                },
                'multiobject_v4': {
                    'root': str(multi_root),
                    'audit_sha256': sha256_file(multi_root / 'audit_summary.json'),
                    'images': {'train': counts[('multiobject_v4', 'train')], 'val': counts[('multiobject_v4', 'val')]},
                    'reviewed_boxes': 1646,
                },
            },
            'classes': {'count': 40, 'names': names},
            'source_images': original_counts,
            'source_boxes': original_boxes,
            'unique_source_image_hashes': {
                'train': len(original_hashes_by_split['train']),
                'val': len(original_hashes_by_split['val']),
            },
            'intentional_duplicate_train_entries': original_counts['train'] - len(original_hashes_by_split['train']),
            'augmentation': {
                'offline_images': len(augmentation_records),
                'variants_per_train_entry': args.augmentations_per_train,
                'seed': args.seed,
                'validation_untouched': True,
                'policy_file': str(output_root / 'augmentation_policy.yaml'),
            },
            'images': output_counts,
            'boxes': output_boxes,
            'empty_labels': empty_label_counts,
            'class_boxes': {
                split: {str(index): class_box_counts[split][index] for index in range(40)}
                for split in SPLITS
            },
            'class_images': {
                split: {str(index): class_image_counts[split][index] for index in range(40)}
                for split in SPLITS
            },
            'physical_groups_checked': group_count,
            'source_split_reconciliation': split_reconciliation,
            'reviewed_overlap_labeling': reviewed_overlap_labeling,
            'notes': [
                '所有 321 张已人工复核的多物品照片及 1646 个框均已并入。',
                f'验证集 {output_counts["val"]} 张保持原图字节不变且不做增强；'
                f'完全重复像素只保留一张验证代表图，共去掉 '
                f'{reviewed_overlap_labeling["validation_entries_dropped"]} 个重复条目。',
                f'训练集 {original_counts["train"]} 个来源条目额外生成 '
                f'{args.augmentations_per_train} 个离线增强版本，共 '
                f'{output_counts["train"]} 张训练图片。',
                f'依据物理采集组和 v9 审定划分，调整了 '
                f'{split_reconciliation["entries_changed"]} 个多物品复核条目的 split。',
                '未复核的原始 burst 帧没有冒充有标签数据；无可信框的照片不会生成训练标签。',
                '本次只构建数据集，没有训练模型，也没有修改 Task A/B。',
            ],
            'checks': checks,
            'training_ready': True,
            'training_started': False,
            'model_modified': False,
            'task_a_b_modified': False,
        }
        (staging / 'audit_summary.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        readme = (
            '# 赛道一 v10 全量多物品增强训练集\n\n'
            f'- 类别：40 类\n'
            f'- 原始/既有数据条目：train {original_counts["train"]}，val {original_counts["val"]}\n'
            f'- 多物品人工复核：321 张，1646 个框\n'
            f'- 训练集增强：每个来源条目额外 {args.augmentations_per_train} 张，仅训练集\n'
            f'- 最终图片：train {output_counts["train"]}，val {output_counts["val"]}\n'
            f'- 最终框：train {output_boxes["train"]}，val {output_boxes["val"]}\n'
            f'- 重叠图复核：{reviewed_overlap_labeling["overlap_hashes_all_splits"]} 个图像哈希采用 v4 人工框，'
            f'覆盖 {reviewed_overlap_labeling["prior_entries_relabeled"]} 条 v9 训练副本\n'
            f'- 验证集：原图字节不变、不增强；去掉 {reviewed_overlap_labeling["validation_entries_dropped"]} 个完全相同像素重复条目\n'
            f'- 物理划分：按 v9 审定结果调整 {split_reconciliation["entries_changed"]} 个多物品复核条目\n'
            f'- 数据配置：`dataset.yaml`\n'
            f'- 增强策略：`augmentation_policy.yaml`\n'
            f'- 审计：`audit_summary.json`\n'
            f'- 增强预览：`review/augmentation_preview.jpg`\n\n'
            '未复核、无可信框的原始照片没有作为标注图混入训练集；本次未训练模型，未修改 Task A/B。\n'
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
        'images': output_counts,
        'boxes': output_boxes,
        'augmented_train_images': len(augmentation_records),
        'unique_source_hashes': {
            'train': len(original_hashes_by_split['train']),
            'val': len(original_hashes_by_split['val']),
        },
        'training_ready': True,
        'training_started': False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f'错误：{exc}', file=os.sys.stderr)
        raise SystemExit(1)
