#!/usr/bin/env python3
"""Track-1 v9 diagnostics for Eggs/Strawberry hard negatives (read-only).

This tool only reads the static training/validation dataset and writes reports
under the data volume. It never opens a camera, imports ROS, or controls the
robot/gripper. Validation frames are displayed for diagnosis only; the v9
builder must never copy them into train.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

DATA_ROOT = Path('/media/sunshine/新加卷').resolve()
DEFAULT_DATASET = DATA_ROOT / '埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v7_oversample_new2/dataset.yaml'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MEAT_CAN = 9       # official 10-meat_can
EGGS = 16          # official 17-Eggs_Plush
STRAWBERRY = 25    # official 26-Plastic_Strawberry
EGG_CONFUSION_CLASSES = (5, 12, 18, 28)  # Foam_Brick, Bowl, Elmers_Glue, Paper_Towels


@dataclass
class Frame:
    split: str
    dataset_stem: str
    image_path: Path
    source_path: Path
    copies: int
    boxes: list[tuple[int, float, float, float, float]]
    egg_green_fraction: float | None
    egg_area: float | None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def names_from_yaml(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        return [str(raw[index]) for index in range(len(raw))]
    raise ValueError('dataset names 必须是列表或字典')


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.is_file():
        return boxes
    for line_no, raw in enumerate(label_path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{label_path}:{line_no}: 不是五列 YOLO 标签')
        class_id = int(parts[0])
        x, y, width, height = (float(value) for value in parts[1:])
        if not 0 <= class_id < 40:
            raise ValueError(f'{label_path}:{line_no}: 类别越界 {class_id}')
        boxes.append((class_id, x, y, width, height))
    return boxes


def image_map(root: Path, split: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((root / 'images' / split).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            result[path.stem] = path
    return result


def collect_frames(root: Path, split: str) -> list[Frame]:
    images = image_map(root, split)
    labels = sorted((root / 'labels' / split).glob('*.txt'))
    grouped: dict[Path, list[tuple[str, Path, list[tuple[int, float, float, float, float]]]]] = defaultdict(list)
    for label_path in labels:
        if label_path.stem not in images:
            raise FileNotFoundError(f'标签没有对应图片: {label_path}')
        image_path = images[label_path.stem]
        source_path = image_path.resolve(strict=True)
        if not is_within(source_path, DATA_ROOT):
            raise ValueError(f'图片解析后不在数据盘: {image_path} -> {source_path}')
        grouped[source_path].append((label_path.stem, image_path, read_boxes(label_path)))

    frames: list[Frame] = []
    for source_path, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        members.sort(key=lambda item: item[0])
        dataset_stem, image_path, boxes = members[0]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'无法读取图片: {image_path}')
        height, width = image.shape[:2]
        egg_fraction = None
        egg_area = None
        egg_boxes = [box for box in boxes if box[0] == EGGS]
        if egg_boxes:
            _, x, y, box_width, box_height = max(egg_boxes, key=lambda box: box[3] * box[4])
            x1 = max(0, min(width, int((x - box_width / 2.0) * width)))
            x2 = max(0, min(width, int((x + box_width / 2.0) * width)))
            y1 = max(0, min(height, int((y - box_height / 2.0) * height)))
            y2 = max(0, min(height, int((y + box_height / 2.0) * height)))
            crop = image[y1:y2, x1:x2]
            if crop.size:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                green_mask = (
                    (hsv[:, :, 0] >= 30)
                    & (hsv[:, :, 0] <= 100)
                    & (hsv[:, :, 1] >= 50)
                    & (hsv[:, :, 2] >= 40)
                )
                egg_fraction = float(green_mask.mean())
                egg_area = float(box_width * box_height)
        frames.append(Frame(
            split=split,
            dataset_stem=dataset_stem,
            image_path=image_path,
            source_path=source_path,
            copies=len(members),
            boxes=boxes,
            egg_green_fraction=egg_fraction,
            egg_area=egg_area,
        ))
    return frames


def fit_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.full((height, width, 3), 28, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return tile


def annotated_tile(frame: Frame, names: list[str], title: str, width: int = 400, height: int = 300) -> np.ndarray:
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise ValueError(f'无法读取图片: {frame.image_path}')
    h, w = image.shape[:2]
    for class_id, x, y, box_width, box_height in frame.boxes:
        x1 = max(0, min(w - 1, int((x - box_width / 2.0) * w)))
        x2 = max(0, min(w - 1, int((x + box_width / 2.0) * w)))
        y1 = max(0, min(h - 1, int((y - box_height / 2.0) * h)))
        y2 = max(0, min(h - 1, int((y + box_height / 2.0) * h)))
        color = (60, 220, 60) if class_id in (EGGS, STRAWBERRY) else (0, 150, 255)
        if class_id == MEAT_CAN:
            color = (255, 80, 40)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        caption = f'{class_id}:{names[class_id]}'
        text_y = max(16, y1 - 5)
        cv2.putText(image, caption, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    tile = fit_tile(image, width, height)
    footer = np.full((56, width, 3), 18, dtype=np.uint8)
    cv2.putText(tile, title[:58], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    detail = f'{frame.split.upper()} copies={frame.copies} classes={[box[0] for box in frame.boxes]}'[:60]
    cv2.putText(tile, detail, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 210, 230), 1, cv2.LINE_AA)
    return np.vstack([tile, footer])


def make_sheet(frames: list[Frame], names: list[str], path: Path, title: str, columns: int = 4) -> None:
    if not frames:
        return
    tiles = []
    for index, frame in enumerate(frames, 1):
        egg_detail = ''
        if frame.egg_green_fraction is not None:
            egg_detail = f' green={frame.egg_green_fraction:.3f} area={frame.egg_area:.4f}'
        caption = f'{index:02d} {frame.dataset_stem}{egg_detail}'
        tiles.append(annotated_tile(frame, names, caption))
    rows = math.ceil(len(tiles) / columns)
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.full((rows * tile_h + 52, columns * tile_w, 3), 10, dtype=np.uint8)
    cv2.putText(canvas, title[:110], (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[52 + row * tile_h:52 + (row + 1) * tile_h, column * tile_w:(column + 1) * tile_w] = tile
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f'无法写接触表: {path}')


def serializable_frame(frame: Frame) -> dict[str, Any]:
    return {
        'split': frame.split,
        'dataset_stem': frame.dataset_stem,
        'source_path': str(frame.source_path),
        'dataset_copies': frame.copies,
        'class_ids': [box[0] for box in frame.boxes],
        'box_count': len(frame.boxes),
        'egg_green_fraction': frame.egg_green_fraction,
        'egg_bbox_area': frame.egg_area,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    dataset_yaml = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not dataset_yaml.is_file():
        raise SystemExit(f'dataset.yaml 不存在: {dataset_yaml}')
    if not is_within(output, DATA_ROOT):
        raise SystemExit(f'输出目录必须位于数据盘: {output}')
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'拒绝覆盖非空诊断目录: {output}')
    output.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    names = names_from_yaml(raw.get('names'))
    if len(names) != 40:
        raise SystemExit(f'应读取 40 类，实际 {len(names)} 类')
    root = dataset_yaml.parent
    train_frames = collect_frames(root, 'train')
    val_frames = collect_frames(root, 'val')

    meat_train = [frame for frame in train_frames if any(box[0] == MEAT_CAN for box in frame.boxes)]
    green_eggs_train = [
        frame for frame in train_frames
        if frame.egg_green_fraction is not None and frame.egg_green_fraction >= 0.30
    ]
    orange_eggs_train = [
        frame for frame in train_frames
        if frame.egg_green_fraction is not None and frame.egg_green_fraction < 0.30
    ]
    green_eggs_val = [
        frame for frame in val_frames
        if frame.egg_green_fraction is not None and frame.egg_green_fraction >= 0.30
    ]
    strawberry_train = [frame for frame in train_frames if any(box[0] == STRAWBERRY for box in frame.boxes)]

    make_sheet(meat_train, names, output / 'meat_can_train_unique_frames.jpg', 'Train-only 10-meat_can hard negatives (official class 10)')
    make_sheet(green_eggs_train, names, output / 'eggs_train_green_front_unique_frames.jpg', 'Train-only green-front 17-Eggs_Plush frames')
    make_sheet(orange_eggs_train, names, output / 'eggs_train_orange_back_unique_frames.jpg', 'Train-only orange-back 17-Eggs_Plush frames')
    make_sheet(green_eggs_val, names, output / 'eggs_val_green_front_diagnostic_only.jpg', 'Validation diagnostic ONLY - never copy to train')
    make_sheet(strawberry_train, names, output / 'strawberry_train_unique_frames.jpg', 'Train-only 26-Plastic_Strawberry frames')
    for class_id in EGG_CONFUSION_CLASSES:
        frames = [frame for frame in train_frames if any(box[0] == class_id for box in frame.boxes)]
        make_sheet(frames, names, output / f'egg_confusion_train_{class_id:02d}_{names[class_id]}.jpg', f'Train-only Eggs hard-negative context: {class_id} {names[class_id]}')

    csv_path = output / 'unique_frame_inventory.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'split', 'dataset_stem', 'source_path', 'dataset_copies', 'class_ids',
            'box_count', 'egg_green_fraction', 'egg_bbox_area', 'green_front_egg',
        ])
        for frame in train_frames + val_frames:
            writer.writerow([
                frame.split,
                frame.dataset_stem,
                str(frame.source_path),
                frame.copies,
                '|'.join(str(box[0]) for box in frame.boxes),
                len(frame.boxes),
                '' if frame.egg_green_fraction is None else f'{frame.egg_green_fraction:.6f}',
                '' if frame.egg_area is None else f'{frame.egg_area:.8f}',
                bool(frame.egg_green_fraction is not None and frame.egg_green_fraction >= 0.30),
            ])

    summary = {
        'schema_version': 1,
        'created_at': now_iso(),
        'dataset_yaml': str(dataset_yaml),
        'mode': 'read_only_static_diagnostics',
        'opens_camera': False,
        'imports_ros': False,
        'robot_control': False,
        'gripper_control': False,
        'class_map': {
            '10-meat_can': MEAT_CAN,
            '17-Eggs_Plush': EGGS,
            '26-Plastic_Strawberry': STRAWBERRY,
        },
        'unique_frames': {
            'train': len(train_frames),
            'val': len(val_frames),
            'train_meat_can': len(meat_train),
            'train_eggs_green_front': len(green_eggs_train),
            'train_eggs_orange_back': len(orange_eggs_train),
            'train_strawberry': len(strawberry_train),
            'val_eggs_green_front_diagnostic_only': len(green_eggs_val),
        },
        'train_meat_can_frames': [serializable_frame(frame) for frame in meat_train],
        'train_green_egg_frames': [serializable_frame(frame) for frame in green_eggs_train],
        'train_orange_egg_frames': [serializable_frame(frame) for frame in orange_eggs_train],
        'val_green_egg_frames_diagnostic_only': [serializable_frame(frame) for frame in green_eggs_val],
        'hard_negative_plan': {
            'meat_can_class_index': MEAT_CAN,
            'meat_can_unique_train_sources': len(meat_train),
            'meat_can_dataset_copies_before_v9': sum(frame.copies for frame in meat_train),
            'green_egg_unique_train_sources': len(green_eggs_train),
            'green_egg_dataset_copies_before_v9': sum(frame.copies for frame in green_eggs_train),
            'orange_egg_unique_train_sources': len(orange_eggs_train),
            'orange_egg_dataset_copies_before_v9': sum(frame.copies for frame in orange_eggs_train),
            'validation_must_remain_one_to_one': True,
            'do_not_copy_validation_to_train': True,
        },
    }
    (output / 'diagnostics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# 赛道一 v9 Eggs 困难负样本诊断',
        '',
        f'- 生成时间：{summary["created_at"]}',
        f'- 数据集：`{dataset_yaml}`',
        '- 模式：只读静态图片；不打开相机、不导入 ROS、不控制机械臂或夹爪。',
        '- 验证集图片只用于失败诊断，v9 禁止复制到 train。',
        '',
        '## 数量',
        '',
        f'- train 唯一原始帧：{len(train_frames)}',
        f'- val 唯一原始帧：{len(val_frames)}',
        f'- train 10-meat_can 唯一帧：{len(meat_train)}，现有数据集副本 {sum(frame.copies for frame in meat_train)}',
        f'- train 绿色正面 Eggs 唯一帧：{len(green_eggs_train)}，现有副本 {sum(frame.copies for frame in green_eggs_train)}',
        f'- train 橙色背面 Eggs 唯一帧：{len(orange_eggs_train)}，现有副本 {sum(frame.copies for frame in orange_eggs_train)}',
        f'- val 绿色正面 Eggs 仅诊断：{len(green_eggs_val)}',
        '',
        '## 判断',
        '',
        '- v8 将绿色肉罐头以高置信度同时报为 Eggs，必须用训练集肉罐头困难负样本压制。',
        '- v8 漏检的 5 张 val 图均为绿色正面 Eggs；训练集存在绿色正面帧，但 v9 应提高这类帧的重复次数。',
        '- 不能删除、过滤或降低阈值来掩盖同框误报。',
        '',
        '## 产物',
        '',
        '- `meat_can_train_unique_frames.jpg`',
        '- `eggs_train_green_front_unique_frames.jpg`',
        '- `eggs_train_orange_back_unique_frames.jpg`',
        '- `eggs_val_green_front_diagnostic_only.jpg`',
        '- `strawberry_train_unique_frames.jpg`',
        '- `unique_frame_inventory.csv`',
        '- `diagnostics.json`',
        '',
    ]
    (output / 'diagnostics.md').write_text('\n'.join(lines), encoding='utf-8')

    print(json.dumps({
        'output': str(output),
        'train_unique_frames': len(train_frames),
        'val_unique_frames': len(val_frames),
        'train_meat_can_frames': len(meat_train),
        'train_green_egg_frames': len(green_eggs_train),
        'train_orange_egg_frames': len(orange_eggs_train),
        'val_green_egg_frames_diagnostic_only': len(green_eggs_val),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
