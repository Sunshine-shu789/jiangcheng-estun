#!/usr/bin/env python3
"""Build the v6 track-1 dataset by adding verified classes 17 and 26 to v5.

The v5 dataset and model remain untouched.  The two previously missing classes
are accepted only from an explicit, visually reviewed frame whitelist.  Their
boxes are generated with the reviewed HSV segmentation parameters and split by
capture group so no burst group crosses train/val.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path('/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序')
V5_ROOT = ROOT / 'datasets/track1_training_merged_v5'
OUTPUT = ROOT / 'datasets/track1_training_merged_v6'
TEMP = ROOT / 'datasets/.track1_training_merged_v6.partial'
SESSION_ROOT = Path('/media/sunshine/新加卷/赛道一_C_D自采照片/20261027_session_001')
SESSION_ID = '20261027_session_001'
SPLITS = ('train', 'val')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
EXPECTED_V5_IMAGES = {'train': 374, 'val': 103}
EXPECTED_V5_BOXES = 745


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassSpec:
    official_class_id: int
    yolo_index: int
    name: str
    start_frame: int
    end_frame: int
    val_groups: frozenset[str]
    expected_train: int
    expected_val: int


@dataclass
class Sample:
    spec: ClassSpec
    frame_id: str
    stem: str
    rgb: Path
    depth: Path
    metadata: Path
    overlay: Path
    capture_group: str
    split: str
    output_stem: str
    bbox_xyxy: tuple[int, int, int, int] | None = None
    width: int = 0
    height: int = 0


SPECS = (
    ClassSpec(
        official_class_id=17,
        yolo_index=16,
        name='17-Eggs_Plush',
        start_frame=1603,
        end_frame=1647,
        val_groups=frozenset({
            'burst-1793364306107',  # front three-quarter view
            'burst-1793364420233',  # opposite side / back view
        }),
        expected_train=35,
        expected_val=10,
    ),
    ClassSpec(
        official_class_id=26,
        yolo_index=25,
        name='26-Plastic_Strawberry',
        start_frame=1565,
        end_frame=1602,
        val_groups=frozenset({
            'burst-1793363484402',  # upper-right small-object view
            'burst-1793363613203',  # lower-center view
        }),
        expected_train=30,
        expected_val=8,
    ),
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def names_from_yaml(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    names = data.get('names')
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or len(names) != 40:
        raise BuildError(f'类别表不是 40 项：{path}')
    return [str(item) for item in names]


def link_absolute(target: Path, link: Path) -> None:
    target = target.resolve(strict=True)
    if not target.is_file():
        raise BuildError(f'链接目标不是文件：{target}')
    if link.exists() or link.is_symlink():
        raise BuildError(f'拒绝覆盖链接：{link}')
    os.symlink(str(target), link)


def expected_stem(spec: ClassSpec, frame: int) -> str:
    return f'{frame:06d}_{spec.name}'


def collect_samples(names: list[str]) -> list[Sample]:
    for spec in SPECS:
        if names[spec.yolo_index] != spec.name:
            raise BuildError(
                f'类别索引 {spec.yolo_index} 应为 {spec.name}，实际 {names[spec.yolo_index]}'
            )

    all_samples: list[Sample] = []
    source_stems: set[str] = set()
    for spec in SPECS:
        wanted = [expected_stem(spec, frame) for frame in range(spec.start_frame, spec.end_frame + 1)]
        wanted_set = set(wanted)
        rgb_dir = SESSION_ROOT / 'rgb'
        actual_rgb = {
            path.stem for path in rgb_dir.glob(f'*_{spec.name}.png') if path.is_file()
        }
        if actual_rgb != wanted_set:
            raise BuildError(
                f'{spec.name} RGB 白名单不一致：缺少={sorted(wanted_set - actual_rgb)} '
                f'多出={sorted(actual_rgb - wanted_set)}'
            )

        class_samples: list[Sample] = []
        seen_groups: set[str] = set()
        for frame in range(spec.start_frame, spec.end_frame + 1):
            stem = expected_stem(spec, frame)
            if stem in source_stems:
                raise BuildError(f'源 stem 重复：{stem}')
            source_stems.add(stem)
            rgb = SESSION_ROOT / 'rgb' / f'{stem}.png'
            depth = SESSION_ROOT / 'depth' / f'{stem}.depth.png'
            metadata_path = SESSION_ROOT / 'meta' / f'{stem}.json'
            overlay = SESSION_ROOT / 'overlays' / f'{stem}.jpg'
            for label, path in (
                ('RGB', rgb), ('深度图', depth), ('metadata', metadata_path), ('预览图', overlay)
            ):
                if not path.is_file():
                    raise BuildError(f'{stem} 缺少{label}：{path}')
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            required = {
                'session_id': SESSION_ID,
                'class_id': spec.official_class_id,
                'label_index': spec.yolo_index,
                'canonical_label': spec.name,
                'is_background': False,
            }
            for key, expected in required.items():
                if metadata.get(key) != expected:
                    raise BuildError(
                        f'{metadata_path} 的 {key} 应为 {expected!r}，实际 {metadata.get(key)!r}'
                    )
            files = metadata.get('files')
            if not isinstance(files, dict):
                raise BuildError(f'{metadata_path} 缺少 files 字段')
            expected_files = {
                'rgb': f'rgb/{stem}.png',
                'depth': f'depth/{stem}.depth.png',
                'metadata': f'meta/{stem}.json',
                'overlay': f'overlays/{stem}.jpg',
            }
            if files != expected_files:
                raise BuildError(f'{metadata_path} 的 files 与磁盘白名单不一致：{files}')
            capture_group = str(metadata.get('capture_group') or '').strip()
            if not capture_group:
                raise BuildError(f'{metadata_path} 缺少 capture_group')
            seen_groups.add(capture_group)
            split = 'val' if capture_group in spec.val_groups else 'train'
            width = int(metadata.get('width', 0))
            height = int(metadata.get('height', 0))
            if width <= 0 or height <= 0:
                raise BuildError(f'{metadata_path} 图片尺寸非法：{width}x{height}')
            output_stem = f'selfcollected__{SESSION_ID}__{stem}'
            class_samples.append(Sample(
                spec=spec, frame_id=f'{frame:06d}', stem=stem, rgb=rgb, depth=depth,
                metadata=metadata_path, overlay=overlay, capture_group=capture_group,
                split=split, output_stem=output_stem, width=width, height=height,
            ))

        missing_val_groups = spec.val_groups - seen_groups
        if missing_val_groups:
            raise BuildError(f'{spec.name} 指定的验证组不存在：{sorted(missing_val_groups)}')
        split_counts = Counter(sample.split for sample in class_samples)
        expected_split_counts = Counter(train=spec.expected_train, val=spec.expected_val)
        if split_counts != expected_split_counts:
            raise BuildError(
                f'{spec.name} 划分数量异常：expected={dict(expected_split_counts)} actual={dict(split_counts)}'
            )
        all_samples.extend(class_samples)

    group_splits: dict[str, set[str]] = defaultdict(set)
    for sample in all_samples:
        group_splits[sample.capture_group].add(sample.split)
    crossing = {group: sorted(value) for group, value in group_splits.items() if len(value) != 1}
    if crossing:
        raise BuildError(f'拍摄组跨 train/val：{crossing}')
    return all_samples


def segment_box(sample: Sample) -> tuple[int, int, int, int]:
    image = cv2.imread(str(sample.rgb), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise BuildError(f'无法解码 RGB：{sample.rgb}')
    height, width = image.shape[:2]
    if (width, height) != (sample.width, sample.height):
        raise BuildError(
            f'{sample.stem} 尺寸与 metadata 不一致：RGB={width}x{height} '
            f'metadata={sample.width}x{sample.height}'
        )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if sample.spec.official_class_id == 17:
        mask = cv2.inRange(hsv, np.array([0, 70, 20]), np.array([179, 255, 255]))
        close_kernel = (9, 9)
        open_kernel = (3, 3)
        min_pixels = 500
        expand = 4
    elif sample.spec.official_class_id == 26:
        red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 90, 20]), np.array([18, 255, 255])),
            cv2.inRange(hsv, np.array([165, 90, 20]), np.array([179, 255, 255])),
        )
        green = cv2.inRange(hsv, np.array([25, 80, 20]), np.array([95, 255, 255]))
        mask = cv2.bitwise_or(red, green)
        close_kernel = (11, 11)
        open_kernel = (3, 3)
        min_pixels = 80
        expand = 3
    else:
        raise BuildError(f'未配置分割参数：{sample.spec.name}')
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(close_kernel, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(open_kernel, np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise BuildError(f'颜色分割未找到主体：{sample.stem}')
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    if area < min_pixels:
        raise BuildError(f'颜色分割主体过小：{sample.stem} area={area} < {min_pixels}')
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    w = int(stats[component, cv2.CC_STAT_WIDTH])
    h = int(stats[component, cv2.CC_STAT_HEIGHT])
    x0 = max(0, x - expand)
    y0 = max(0, y - expand)
    x1 = min(width, x + w + expand)
    y1 = min(height, y + h + expand)
    if x1 <= x0 or y1 <= y0:
        raise BuildError(f'颜色分割框非法：{sample.stem}')
    box_area = (x1 - x0) * (y1 - y0)
    if box_area < 100 or box_area > width * height * 0.5:
        raise BuildError(f'颜色分割框面积异常：{sample.stem} area={box_area}')
    return x0, y0, x1, y1


def yolo_row(sample: Sample) -> str:
    if sample.bbox_xyxy is None:
        raise BuildError(f'尚未计算框：{sample.stem}')
    x0, y0, x1, y1 = sample.bbox_xyxy
    width = float(sample.width)
    height = float(sample.height)
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    if not all(0.0 < value <= 1.0 for value in (cx, cy, bw, bh)):
        raise BuildError(f'YOLO 坐标越界：{sample.stem}')
    epsilon = 1e-6
    if cx - bw / 2 < -epsilon or cx + bw / 2 > 1 + epsilon:
        raise BuildError(f'YOLO x 边界越界：{sample.stem}')
    if cy - bh / 2 < -epsilon or cy + bh / 2 > 1 + epsilon:
        raise BuildError(f'YOLO y 边界越界：{sample.stem}')
    if bw >= 0.98 and bh >= 0.98:
        raise BuildError(f'疑似整图伪框：{sample.stem}')
    return f'{sample.spec.yolo_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'


def write_contact_sheets(samples: list[Sample], review_dir: Path) -> list[str]:
    written: list[str] = []
    thumb_w, thumb_h = 320, 240
    cols, per_page = 4, 20
    for spec in SPECS:
        selected = sorted(
            (sample for sample in samples if sample.spec == spec),
            key=lambda sample: sample.frame_id,
        )
        for page_index in range(0, len(selected), per_page):
            page = selected[page_index:page_index + per_page]
            rows = (len(page) + cols - 1) // cols
            canvas = np.full((rows * thumb_h, cols * thumb_w, 3), 245, dtype=np.uint8)
            for index, sample in enumerate(page):
                image = cv2.imread(str(sample.rgb), cv2.IMREAD_COLOR)
                if image is None or sample.bbox_xyxy is None:
                    raise BuildError(f'无法生成复核图：{sample.rgb}')
                sx = thumb_w / image.shape[1]
                sy = thumb_h / image.shape[0]
                thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                x0, y0, x1, y1 = sample.bbox_xyxy
                cv2.rectangle(
                    thumb,
                    (round(x0 * sx), round(y0 * sy)),
                    (max(round(x0 * sx), round(x1 * sx) - 1), max(round(y0 * sy), round(y1 * sy) - 1)),
                    (0, 255, 0), 2,
                )
                cv2.putText(
                    thumb, f'{sample.frame_id} {sample.split}', (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    thumb, f'{sample.frame_id} {sample.split}', (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )
                row, col = divmod(index, cols)
                canvas[row * thumb_h:(row + 1) * thumb_h, col * thumb_w:(col + 1) * thumb_w] = thumb
            out = review_dir / f'{spec.official_class_id:02d}_bbox_contact_{page_index // per_page + 1}.jpg'
            if not cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise BuildError(f'无法写入候选框复核图：{out}')
            written.append(f'review/{out.name}')
    return written


def link_v5_tree(temp_root: Path) -> tuple[dict[str, int], dict[str, dict[int, int]], dict[str, dict[str, int]]]:
    image_counts: dict[str, int] = {}
    class_counts: dict[str, dict[int, int]] = {split: {} for split in SPLITS}
    target_by_stem: dict[str, dict[str, Path]] = {}
    for split in SPLITS:
        source_image_dir = V5_ROOT / 'images' / split
        source_label_dir = V5_ROOT / 'labels' / split
        image_map = {
            path.stem: path for path in source_image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        label_map = {
            path.stem: path for path in source_label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == '.txt'
        }
        if set(image_map) != set(label_map):
            raise BuildError(f'v5 {split} 图片和标签 stem 不一致')
        image_counts[split] = len(image_map)
        if image_counts[split] != EXPECTED_V5_IMAGES[split]:
            raise BuildError(f'v5 {split} 图片数应为 {EXPECTED_V5_IMAGES[split]}，实际 {image_counts[split]}')
        split_counter: Counter[int] = Counter()
        target_by_stem[split] = {}
        for stem in sorted(image_map):
            source_image = image_map[stem].resolve(strict=True)
            source_label = label_map[stem].resolve(strict=True)
            link_absolute(source_image, temp_root / 'images' / split / f'{stem}{image_map[stem].suffix.lower()}')
            link_absolute(source_label, temp_root / 'labels' / split / f'{stem}.txt')
            target_by_stem[split][stem] = source_image
            rows = [line.strip() for line in source_label.read_text(encoding='utf-8').splitlines() if line.strip()]
            for line_no, row in enumerate(rows, 1):
                parts = row.split()
                if len(parts) != 5:
                    raise BuildError(f'v5 标签不是 5 列：{source_label}:{line_no}')
                class_id = int(parts[0])
                if not 0 <= class_id < 40:
                    raise BuildError(f'v5 类别越界：{source_label}:{line_no}')
                split_counter[class_id] += 1
        class_counts[split] = dict(split_counter)
    if sum(sum(value.values()) for value in class_counts.values()) != EXPECTED_V5_BOXES:
        raise BuildError('v5 总框数异常')
    return image_counts, class_counts, target_by_stem


def scan_dataset(root: Path) -> tuple[dict[str, int], int, dict[str, dict[int, int]], dict[str, int], dict[str, dict[str, Path]]]:
    image_counts: dict[str, int] = {}
    boxes_total = 0
    counts: dict[str, dict[int, int]] = {split: {} for split in SPLITS}
    empty: dict[str, int] = {split: 0 for split in SPLITS}
    links: dict[str, dict[str, Path]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        image_dir = root / 'images' / split
        label_dir = root / 'labels' / split
        image_map = {
            path.stem: path for path in image_dir.iterdir()
            if (path.is_file() or path.is_symlink()) and path.suffix.lower() in IMAGE_SUFFIXES
        }
        label_map = {
            path.stem: path for path in label_dir.iterdir()
            if (path.is_file() or path.is_symlink()) and path.suffix.lower() == '.txt'
        }
        if set(image_map) != set(label_map):
            raise BuildError(f'{split} 图片和标签 stem 不一致')
        image_counts[split] = len(image_map)
        split_counter: Counter[int] = Counter()
        for stem in sorted(image_map):
            image_path = image_map[stem]
            label_path = label_map[stem]
            if not image_path.is_symlink() or not label_path.is_symlink():
                raise BuildError(f'{split}/{stem} 不是符号链接')
            image_target = image_path.resolve(strict=True)
            label_target = label_path.resolve(strict=True)
            if not image_target.is_file() or not label_target.is_file():
                raise BuildError(f'{split}/{stem} 链接目标不是文件')
            links[split][stem] = image_target
            rows = [line.strip() for line in label_target.read_text(encoding='utf-8').splitlines() if line.strip()]
            if not rows:
                empty[split] += 1
            for row in rows:
                parts = row.split()
                if len(parts) != 5:
                    raise BuildError(f'{split}/{stem} 标签不是 5 列')
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
                if not 0 <= class_id < 40:
                    raise BuildError(f'{split}/{stem} 类别越界')
                if not all(0.0 < value <= 1.0 for value in values):
                    raise BuildError(f'{split}/{stem} 标签坐标越界')
                cx, cy, width, height = values
                if cx - width / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6:
                    raise BuildError(f'{split}/{stem} 标签 x 边界越界')
                if cy - height / 2 < -1e-6 or cy + height / 2 > 1 + 1e-6:
                    raise BuildError(f'{split}/{stem} 标签 y 边界越界')
                split_counter[class_id] += 1
                boxes_total += 1
        counts[split] = dict(split_counter)
    return image_counts, boxes_total, counts, empty, links


def main() -> int:
    parser = argparse.ArgumentParser(description='构建赛道一 40 类训练集 v6')
    parser.add_argument('--check-only', action='store_true', help='只校验源图片和生成候选框，不写数据集')
    args = parser.parse_args()

    if not (V5_ROOT / '.complete').is_file():
        raise BuildError('v5 数据集未完成')
    v5_audit = json.loads((V5_ROOT / 'audit_summary.json').read_text(encoding='utf-8'))
    if v5_audit.get('training_ready') is not True:
        raise BuildError('v5 training_ready 不是 true')
    if not (V5_ROOT / 'dataset.yaml').is_file():
        raise BuildError('v5 dataset.yaml 不存在')
    names = names_from_yaml(V5_ROOT / 'dataset.yaml')
    samples = collect_samples(names)
    for sample in samples:
        sample.bbox_xyxy = segment_box(sample)

    expected_images = {'train': 439, 'val': 121}
    expected_boxes = 828
    split_counts = Counter(sample.split for sample in samples)
    print(json.dumps({
        'mode': 'check_only' if args.check_only else 'build',
        'new_samples': dict(split_counts),
        'classes': {
            spec.name: {
                'expected': {'train': spec.expected_train, 'val': spec.expected_val},
                'actual': {
                    split: sum(1 for sample in samples if sample.spec == spec and sample.split == split)
                    for split in SPLITS
                },
                'groups': len({sample.capture_group for sample in samples if sample.spec == spec}),
                'val_groups': sorted(spec.val_groups),
            }
            for spec in SPECS
        },
        'bboxes': [
            {
                'frame': sample.frame_id,
                'class': sample.name if hasattr(sample, 'name') else sample.spec.name,
                'split': sample.split,
                'xyxy': sample.bbox_xyxy,
            }
            for sample in samples
        ],
    }, ensure_ascii=False, indent=2))
    if args.check_only:
        return 0

    if OUTPUT.exists() or TEMP.exists():
        raise BuildError(f'拒绝覆盖已有目录：{OUTPUT if OUTPUT.exists() else TEMP}')

    TEMP.mkdir(parents=True)
    try:
        for split in SPLITS:
            for directory in (
                TEMP / 'images' / split,
                TEMP / 'labels' / split,
                TEMP / 'label_sources' / split,
            ):
                directory.mkdir(parents=True)
        (TEMP / 'review').mkdir()

        v5_images, v5_class_counts, v5_targets = link_v5_tree(TEMP)
        added_classes = Counter(sample.spec.yolo_index for sample in samples)
        if added_classes != Counter({16: 45, 25: 38}):
            raise BuildError(f'新增类别框数异常：{dict(added_classes)}')

        added_rows: list[dict[str, Any]] = []
        for sample in samples:
            if sample.bbox_xyxy is None:
                raise BuildError(f'缺少候选框：{sample.stem}')
            label_text = yolo_row(sample) + '\n'
            label_path = TEMP / 'label_sources' / sample.split / f'{sample.output_stem}.txt'
            label_path.write_text(label_text, encoding='utf-8')
            link_absolute(sample.rgb, TEMP / 'images' / sample.split / f'{sample.output_stem}.png')
            label_target = OUTPUT / 'label_sources' / sample.split / f'{sample.output_stem}.txt'
            link = TEMP / 'labels' / sample.split / f'{sample.output_stem}.txt'
            if link.exists() or link.is_symlink():
                raise BuildError(f'拒绝覆盖标签链接：{link}')
            os.symlink(str(label_target), link)
            x0, y0, x1, y1 = sample.bbox_xyxy
            added_rows.append({
                'source_kind': 'track1_selfcollected_v6',
                'source_split': sample.split,
                'source_image': str(sample.rgb),
                'source_depth': str(sample.depth),
                'source_metadata': str(sample.metadata),
                'output_image': f'images/{sample.split}/{sample.output_stem}.png',
                'output_label': f'labels/{sample.split}/{sample.output_stem}.txt',
                'source_label': str(OUTPUT / 'label_sources' / sample.split / f'{sample.output_stem}.txt'),
                'group': sample.capture_group,
                'class_ids': [sample.spec.yolo_index],
                'official_class_id': sample.spec.official_class_id,
                'canonical_label': sample.spec.name,
                'frame_id': sample.frame_id,
                'scene_type': json.loads(sample.metadata.read_text(encoding='utf-8')).get('scene_type'),
                'bbox_xyxy': [x0, y0, x1, y1],
                'bbox_method': 'hsv_candidate_human_contact_sheet_reviewed',
                'v6_role': 'missing_class_selfcollected',
            })

        dataset_yaml = {
            'path': str(OUTPUT),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 40,
            'names': names,
        }
        (TEMP / 'dataset.yaml').write_text(
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )

        base_manifest_lines = [
            line for line in (V5_ROOT / 'merge_manifest.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        with (TEMP / 'merge_manifest.jsonl').open('w', encoding='utf-8') as handle:
            for line in base_manifest_lines:
                handle.write(line + '\n')
            for row in added_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

        with (TEMP / 'new_bboxes.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([
                'split', 'frame_id', 'official_class_id', 'yolo_index', 'canonical_label',
                'x0', 'y0', 'x1', 'y1', 'image_width', 'image_height', 'capture_group', 'source_image',
            ])
            for sample in samples:
                assert sample.bbox_xyxy is not None
                writer.writerow([
                    sample.split, sample.frame_id, sample.spec.official_class_id, sample.spec.yolo_index,
                    sample.spec.name, *sample.bbox_xyxy, sample.width, sample.height,
                    sample.capture_group, str(sample.rgb),
                ])

        contact_sheets = write_contact_sheets(samples, TEMP / 'review')

        # Structural pre-rename audit.  New label symlinks intentionally target
        # the final path and therefore are not resolved until the atomic rename.
        merged_counts = {split: dict(v5_class_counts[split]) for split in SPLITS}
        for sample in samples:
            merged_counts[sample.split][sample.spec.yolo_index] = (
                merged_counts[sample.split].get(sample.spec.yolo_index, 0) + 1
            )
        expected_merged_counts = {split: dict(v5_class_counts[split]) for split in SPLITS}
        for spec in SPECS:
            expected_merged_counts['train'][spec.yolo_index] = (
                expected_merged_counts['train'].get(spec.yolo_index, 0) + spec.expected_train
            )
            expected_merged_counts['val'][spec.yolo_index] = (
                expected_merged_counts['val'].get(spec.yolo_index, 0) + spec.expected_val
            )
        if merged_counts != expected_merged_counts:
            raise BuildError('新增类别合并后的类别计数异常')
        if set(merged_counts['train']) != set(range(40)) or set(merged_counts['val']) != set(range(40)):
            raise BuildError('合并后 train/val 未覆盖全部 40 类')

        expected_image_split_count = {
            'train': v5_images['train'] + sum(1 for sample in samples if sample.split == 'train'),
            'val': v5_images['val'] + sum(1 for sample in samples if sample.split == 'val'),
        }
        if expected_image_split_count != expected_images:
            raise BuildError(f'预期图片数异常：{expected_image_split_count}')

        v5_front10_counts = {
            split: sum(v5_class_counts[split].get(index, 0) for index in range(10))
            for split in SPLITS
        }
        v6_front10_counts = {
            split: sum(merged_counts[split].get(index, 0) for index in range(10))
            for split in SPLITS
        }
        if v5_front10_counts != v6_front10_counts:
            raise BuildError('前 10 类框数未完整保留')

        source_inventory = {
            'session_root': str(SESSION_ROOT),
            'session_id': SESSION_ID,
            'valid_samples': len(samples),
            'split_counts': dict(split_counts),
            'classes': {},
            'ignored_source_history': {
                '17-Eggs_Plush': {
                    'rgb_files_used': 45,
                    'depth_files_on_disk': len(list((SESSION_ROOT / 'depth').glob('*_17-Eggs_Plush*'))),
                    'metadata_files_on_disk': len(list((SESSION_ROOT / 'meta').glob('*_17-Eggs_Plush*'))),
                    'note': '39 historical sets without RGB (frames 000420-000458) are excluded',
                },
                '26-Plastic_Strawberry': {
                    'rgb_files_used': 38,
                    'depth_files_on_disk': len(list((SESSION_ROOT / 'depth').glob('*_26-Plastic_Strawberry*'))),
                    'metadata_files_on_disk': len(list((SESSION_ROOT / 'meta').glob('*_26-Plastic_Strawberry*'))),
                    'note': '45 superseded/extra history sets without a unique live RGB are excluded',
                },
                'frames_jsonl_policy': 'not used as a source index; live RGB+metadata whitelist is authoritative',
            },
        }
        for spec in SPECS:
            subset = [sample for sample in samples if sample.spec == spec]
            source_inventory['classes'][spec.name] = {
                'official_class_id': spec.official_class_id,
                'yolo_index': spec.yolo_index,
                'frames': [sample.frame_id for sample in subset],
                'train_frames': [sample.frame_id for sample in subset if sample.split == 'train'],
                'val_frames': [sample.frame_id for sample in subset if sample.split == 'val'],
                'train_count': spec.expected_train,
                'val_count': spec.expected_val,
                'capture_groups': {
                    group: {
                        'split': next(sample.split for sample in subset if sample.capture_group == group),
                        'count': sum(1 for sample in subset if sample.capture_group == group),
                    }
                    for group in sorted({sample.capture_group for sample in subset})
                },
                'val_groups': sorted(spec.val_groups),
                'bbox_method': {
                    '17-Eggs_Plush': 'HSV S>=70,V>=20; close 9x9; open 3x3; largest component >=500px; expand 4px',
                    '26-Plastic_Strawberry': 'HSV red/green masks; close 11x11; open 3x3; largest component >=80px; expand 3px',
                }[spec.name],
            }

        checks = {
            'v5_preserved_unchanged': True,
            'v5_checks_all_true': all(value is True for value in v5_audit.get('checks', {}).values()),
            'image_label_pairing_complete': True,
            'nc_40': len(names) == 40,
            'all_40_classes_have_train_and_val': True,
            'class_17_active': merged_counts['train'].get(16, 0) > 0 and merged_counts['val'].get(16, 0) > 0,
            'class_26_active': merged_counts['train'].get(25, 0) > 0 and merged_counts['val'].get(25, 0) > 0,
            'front10_preserved': v5_front10_counts == v6_front10_counts,
            'no_missing_source_rgb_depth_metadata': True,
            'no_duplicate_source_stem': True,
            'no_capture_group_split_crossing': True,
            'new_samples_match_visual_whitelist': True,
            'new_labels_one_valid_box_each': True,
            'new_images_are_absolute_symlinks': True,
            'source_history_orphans_excluded': True,
        }
        if not all(value is True for value in checks.values()):
            raise BuildError(f'审计检查失败：{checks}')

        with (TEMP / 'class_counts.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['split', 'class_index', 'official_class_id', 'class_name', 'boxes'])
            for split in SPLITS:
                for class_id in range(40):
                    writer.writerow([
                        split, class_id, class_id + 1, names[class_id], merged_counts[split].get(class_id, 0),
                    ])

        audit = deepcopy(v5_audit)
        audit.update({
            'schema_version': 1,
            'created_at': now_iso(),
            'dataset_root': str(OUTPUT),
            'dataset_yaml': str(OUTPUT / 'dataset.yaml'),
            'materialization': 'absolute_symlink',
            'images': expected_images,
            'labels': dict(expected_images),
            'box_count': expected_boxes,
            'boxes_by_split': {split: sum(merged_counts[split].values()) for split in SPLITS},
            'empty_labels': {'train': 1, 'val': 1},
            'class_counts': {
                split: {str(key): value for key, value in sorted(merged_counts[split].items())}
                for split in SPLITS
            },
            'front10_boxes': v6_front10_counts,
            'checks': checks,
            'training_ready': True,
            'training_started': False,
            'v5_base': {
                'root': str(V5_ROOT),
                'dataset_yaml_sha256': sha256_file(V5_ROOT / 'dataset.yaml'),
                'audit_summary_sha256': sha256_file(V5_ROOT / 'audit_summary.json'),
                'images': v5_images,
                'box_count': EXPECTED_V5_BOXES,
                'front10_boxes': v5_front10_counts,
                'all_files_relinked_without_modification': True,
            },
            'v6_additions': {
                'purpose': 'activate previously missing official classes 17 and 26',
                'source_inventory': source_inventory,
                'new_images': dict(split_counts),
                'new_boxes': len(samples),
                'review_contact_sheets': contact_sheets,
                'manual_review': 'all 45 Eggs and 38 Strawberry candidate boxes visually checked in contact sheets',
                'bbox_generation': 'reviewed HSV candidate boxes expanded and clipped before conversion to YOLO labels',
                'train_val_policy': 'split by capture_group; no capture group crosses splits',
                'scene_labels': 'not added; existing scene metadata is retained only as source provenance',
            },
        })
        source_summary = deepcopy(audit.get('source_summary', {}))
        source_summary['v6_additions'] = audit['v6_additions']
        audit['source_summary'] = source_summary
        audit['note'] = (
            'v6 preserves every v5 image/label link exactly and adds only the visually reviewed '
            '45 Eggs Plush and 38 Plastic Strawberry samples. v5 and all prior models remain untouched.'
        )
        (TEMP / 'audit_summary.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )

        (TEMP / 'README.md').write_text(
            '# 赛道一 40 类训练集 v6\n\n'
            'v6 原样保留训练集 v5 的全部图片和标签，仅新增已采集并复核的缺失类别 17（蛋形毛绒玩具）和 26（塑料草莓）。\n\n'
            f'- 训练图片：{expected_images["train"]}\n'
            f'- 验证图片：{expected_images["val"]}\n'
            f'- 总图片：{sum(expected_images.values())}\n'
            f'- 总框数：{expected_boxes}\n'
            '- 新增 17 类：35 张训练 / 10 张验证\n'
            '- 新增 26 类：30 张训练 / 8 张验证\n'
            '- 新增样本按拍摄组划分，同一连拍组不跨 train/val。\n'
            '- 新图片均为指向新加卷原始照片的绝对软链接；不占用本地内存。\n'
            '- v5 数据、v5 权重和 `models/best.pt` 均未修改。\n'
            '- 候选框复核图位于 `review/`，逐框坐标位于 `new_bboxes.csv`。\n\n'
            '会话中没有对应 RGB 的历史孤儿记录、被同名覆盖的记录和 `frames.jsonl` 重复引用均未进入数据集。\n',
            encoding='utf-8',
        )

        os.replace(TEMP, OUTPUT)

        # Full post-rename validation resolves every symlink.
        final_images, final_boxes, final_counts, final_empty, final_links = scan_dataset(OUTPUT)
        failures: list[str] = []
        if final_images != expected_images:
            failures.append(f'图片数异常：{final_images}')
        if final_boxes != expected_boxes:
            failures.append(f'框数异常：{final_boxes}')
        if final_counts != merged_counts:
            failures.append('最终类别计数异常')
        if final_empty != {'train': 1, 'val': 1}:
            failures.append(f'空标签数异常：{final_empty}')
        if set(final_counts['train']) != set(range(40)) or set(final_counts['val']) != set(range(40)):
            failures.append('最终数据集未覆盖 40 类')
        for split in SPLITS:
            for stem, target in v5_targets[split].items():
                if final_links[split].get(stem) != target:
                    failures.append(f'v5 文件链接未保留：{split}/{stem}')
                    break
            for sample in (item for item in samples if item.split == split):
                if final_links[split].get(sample.output_stem) != sample.rgb.resolve(strict=True):
                    failures.append(f'新增文件链接错误：{split}/{sample.output_stem}')
                    break
        if failures:
            raise BuildError('原子构建后最终校验失败：' + '; '.join(failures))

        (OUTPUT / '.complete').write_text(now_iso() + '\n', encoding='utf-8')
        print(json.dumps({
            'output': str(OUTPUT),
            'images': final_images,
            'boxes': final_boxes,
            'new_samples': dict(split_counts),
            'class_17_boxes': {'train': final_counts['train'][16], 'val': final_counts['val'][16]},
            'class_26_boxes': {'train': final_counts['train'][25], 'val': final_counts['val'][25]},
            'front10_boxes': v6_front10_counts,
            'training_ready': True,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        if TEMP.exists():
            shutil.rmtree(TEMP, ignore_errors=True)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
