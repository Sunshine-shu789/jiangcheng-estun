#!/usr/bin/env python3
"""Create a standalone 40-class Track-1 initialization weight from YOLO11s COCO.

The backbone/neck and shape-compatible detection parameters are inherited from
the generic pretrained checkpoint. Every YOLO classification output row is
newly initialized for the 40 Track-1 classes; COCO class semantics are never
copied into the new class rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import __version__ as ultralytics_version
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel

DATA_ROOT = Path('/media/sunshine/新加卷').resolve()
PROJECT_ROOT = DATA_ROOT / '埃斯顿比赛_2026/赛道一比赛程序'
DEFAULT_DATASET = PROJECT_ROOT / 'datasets/track1_training_merged_v10_all_multiclass_augmented_20261031/dataset.yaml'
DEFAULT_SOURCE = DATA_ROOT / 'taskA/yolo11s.pt'
DEFAULT_OUTPUT = PROJECT_ROOT / 'models/yolo11s_backbone_40class_init_v10aug.pt'
DEFAULT_REPORT = PROJECT_ROOT / 'reports/track1_v11_init_weight_latest.json'


class InitError(RuntimeError):
    pass


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


def load_names(dataset_yaml: Path) -> list[str]:
    payload = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise InitError(f'dataset.yaml 顶层不是对象: {dataset_yaml}')
    names = payload.get('names')
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or len(names) != 40:
        raise InitError(f'dataset.yaml 必须包含 40 个类别，实际: {names!r}')
    if int(payload.get('nc', -1)) != 40:
        raise InitError(f'dataset.yaml nc 不是 40: {payload.get("nc")}')
    return [str(name) for name in names]


def detection_class_output_keys(model: DetectionModel) -> set[str]:
    keys: set[str] = set()
    final_layer = model.model[-1]
    for attr in ('cv3', 'one2one_cv3'):
        blocks = getattr(final_layer, attr, None)
        if not blocks:
            continue
        for branch in blocks:
            if not isinstance(branch, torch.nn.Sequential) or len(branch) == 0:
                continue
            conv = branch[-1]
            if getattr(conv, 'out_channels', None) != model.nc:
                continue
            for name, parameter in model.named_parameters():
                if parameter is conv.weight or parameter is conv.bias:
                    keys.add(name)
    return keys


def load_source_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise InitError('COCO 权重不是 Ultralytics checkpoint 字典')
    candidate = checkpoint.get('ema') or checkpoint.get('model')
    if not isinstance(candidate, torch.nn.Module):
        raise InitError('COCO checkpoint 中找不到模型对象')
    return checkpoint


def verify_checkpoint(path: Path, expected_names: list[str]) -> dict[str, Any]:
    from ultralytics import YOLO

    wrapper = YOLO(str(path), task='detect')
    model = wrapper.model.float().eval()
    actual_names = model.names
    if isinstance(actual_names, dict):
        actual_list = [str(actual_names[index]) for index in range(len(actual_names))]
    elif isinstance(actual_names, list):
        actual_list = [str(name) for name in actual_names]
    else:
        raise InitError(f'新权重类别类型非法: {type(actual_names).__name__}')
    if int(model.nc) != 40 or actual_list != expected_names:
        raise InitError('新权重加载后不是预期的 40 类顺序')
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64, dtype=torch.float32))
    if output is None:
        raise InitError('新权重前向传播无输出')
    return {
        'nc': int(model.nc),
        'names': actual_list,
        'forward_ok': True,
        'loaded_class': type(model).__name__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='生成赛道一 40 类 YOLO11s 新初始化权重')
    parser.add_argument('--dataset-yaml', default=str(DEFAULT_DATASET))
    parser.add_argument('--source-weights', default=str(DEFAULT_SOURCE))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--report', default=str(DEFAULT_REPORT))
    parser.add_argument('--force', action='store_true', help='仅用于重新生成尚不存在的新版权重')
    args = parser.parse_args()

    dataset_yaml = Path(args.dataset_yaml).expanduser().resolve()
    source_weights = Path(args.source_weights).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if not dataset_yaml.is_file():
        raise InitError(f'数据集配置不存在: {dataset_yaml}')
    if not source_weights.is_file():
        raise InitError(f'通用预训练权重不存在: {source_weights}')
    if not is_within(source_weights, DATA_ROOT) or not is_within(output, PROJECT_ROOT) or not is_within(report_path, PROJECT_ROOT):
        raise InitError('源权重必须位于数据盘，输出和报告必须位于正式项目目录')
    expected_names = load_names(dataset_yaml)
    if output.exists() and not args.force:
        raise InitError(f'新初始化权重已存在，拒绝覆盖: {output}')

    source_sha = sha256_file(source_weights)
    source_checkpoint = load_source_checkpoint(source_weights)
    source_model = source_checkpoint.get('ema') or source_checkpoint['model']
    if int(getattr(source_model, 'nc', -1)) != 80:
        raise InitError(f'通用预训练权重应为 COCO 80 类，实际 nc={getattr(source_model, "nc", None)}')

    model_cfg = deepcopy(getattr(source_model, 'yaml', None))
    if not isinstance(model_cfg, dict):
        raise InitError('通用权重缺少可复用的模型 YAML')
    model_cfg['nc'] = 40
    target = DetectionModel(model_cfg, nc=40, verbose=True)
    target.nc = 40
    target.names = {index: name for index, name in enumerate(expected_names)}
    target.load(source_checkpoint, verbose=True)

    target_state = target.state_dict()
    source_state = source_model.float().state_dict()
    compatible_keys = {
        key for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    class_output_keys = detection_class_output_keys(target)
    if not class_output_keys:
        raise InitError('未识别到 40 类检测头分类输出参数')
    reused_class_outputs = class_output_keys & compatible_keys
    if reused_class_outputs:
        raise InitError(f'检测头分类输出不应从 COCO 复制: {sorted(reused_class_outputs)}')
    backbone_reused = sum(1 for key in compatible_keys if not key.startswith('model.23.'))
    if backbone_reused < 300:
        raise InitError(f'继承的骨干/颈部参数过少: {backbone_reused}')

    target.args = {
        **DEFAULT_CFG_DICT,
        'task': 'detect',
        'mode': 'train',
        'model': str(output),
        'data': str(dataset_yaml),
        'imgsz': 640,
        'nc': 40,
    }
    target.task = 'detect'
    target.pt_path = str(output)
    target.yaml = deepcopy(model_cfg)
    target.yaml['nc'] = 40
    init_audit = {
        'schema_version': 1,
        'created_at': now_iso(),
        'purpose': 'YOLO11s COCO backbone/neck -> fresh 40-class Track-1 detection head',
        'dataset_yaml': str(dataset_yaml),
        'dataset_yaml_sha256': sha256_file(dataset_yaml),
        'source_weights': str(source_weights),
        'source_weights_sha256': source_sha,
        'source_nc': 80,
        'target_nc': 40,
        'target_names': expected_names,
        'source_state_tensors': len(source_state),
        'target_state_tensors': len(target_state),
        'shape_compatible_tensors_reused': len(compatible_keys),
        'non_head_shape_compatible_tensors_reused': backbone_reused,
        'fresh_class_output_tensors': sorted(class_output_keys),
        'coco_class_outputs_reused': False,
        'python_package_version': package_version('ultralytics'),
        'ultralytics_version': ultralytics_version,
        'torch_version': torch.__version__,
    }
    checkpoint = {
        'model': target,
        'ema': None,
        'optimizer': None,
        'train_args': {},
        'train_metrics': {},
        'train_results': {},
        'date': now_iso(),
        'version': ultralytics_version,
        'init_audit': init_audit,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.tmp.{os.getpid()}')
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    verify = verify_checkpoint(output, expected_names)
    output_sha = sha256_file(output)
    report = {
        **init_audit,
        'output_weights': str(output),
        'output_weights_sha256': output_sha,
        'output_size_bytes': output.stat().st_size,
        'verify': verify,
        'passed': True,
        'old_models_modified': False,
        'task_a_b_modified': False,
    }
    report_tmp = report_path.with_name(report_path.name + f'.tmp.{os.getpid()}')
    report_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(report_tmp, report_path)
    print(json.dumps({
        'ok': True,
        'output': str(output),
        'sha256': output_sha,
        'reused_tensors': len(compatible_keys),
        'fresh_class_output_tensors': len(class_output_keys),
        'report': str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
