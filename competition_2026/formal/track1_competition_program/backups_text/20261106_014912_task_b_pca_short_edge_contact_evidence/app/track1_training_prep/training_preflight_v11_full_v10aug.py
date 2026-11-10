#!/usr/bin/env python3
"""Guarded full retraining for Track-1 v11 on the audited v10 augmented dataset."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import yaml

DATA_ROOT = Path('/media/sunshine/新加卷').resolve()
PROJECT_ROOT = DATA_ROOT / '埃斯顿比赛_2026/赛道一比赛程序'
DEFAULT_CONFIG = PROJECT_ROOT / 'config/track1_train_40class_v11_full_v10aug_from_yolo11s.yaml'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = ('train', 'val')
REQUIRED_CONFIG = (
    'dataset_yaml', 'dataset_audit_report', 'dataset_audit_summary', 'base_weights',
    'base_weights_report', 'output_root', 'preflight_report', 'post_train_report',
    'run_name_prefix', 'expected', 'train', 'post_train',
)


class PreflightError(RuntimeError):
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


def resolve_path(value: Any, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f'训练配置不存在: {path}')
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise PreflightError('训练配置顶层必须是字典')
    missing = [key for key in REQUIRED_CONFIG if key not in payload]
    if missing:
        raise PreflightError(f'训练配置缺少字段: {missing}')
    if not isinstance(payload['expected'], dict) or not isinstance(payload['train'], dict):
        raise PreflightError('expected 和 train 必须是字典')
    return payload


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise PreflightError(f'JSON 顶层不是对象: {path}')
    return payload


def names_to_list(value: Any) -> list[str]:
    if isinstance(value, dict):
        try:
            return [str(value[index]) for index in range(len(value))]
        except KeyError as exc:
            raise PreflightError(f'类别映射不是连续索引: {exc}') from exc
    if isinstance(value, list):
        return [str(item) for item in value]
    raise PreflightError(f'类别类型非法: {type(value).__name__}')


def add_check(checks: dict[str, Any], name: str, ok: bool, detail: Any) -> None:
    checks[name] = {'ok': bool(ok), 'detail': detail}


def configure_caches() -> dict[str, Path]:
    root = PROJECT_ROOT / 'cache/track1_training'
    paths = {
        'tmp': root / 'tmp', 'matplotlib': root / 'matplotlib', 'numba': root / 'numba',
        'cuda': root / 'cuda', 'torch': root / 'torch', 'xdg': root / 'xdg',
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    alias = Path('/tmp') / f'track1_training_tmp_{os.getuid()}'
    target = paths['tmp'].resolve()
    if alias.is_symlink():
        if alias.resolve() != target:
            raise PreflightError(f'临时目录别名指向错误: {alias}')
    elif alias.exists():
        raise PreflightError(f'临时目录别名被占用: {alias}')
    else:
        alias.symlink_to(target, target_is_directory=True)
    os.environ.update({
        'TMPDIR': str(alias), 'TEMP': str(alias), 'TMP': str(alias),
        'MPLCONFIGDIR': str(paths['matplotlib']), 'NUMBA_CACHE_DIR': str(paths['numba']),
        'CUDA_CACHE_PATH': str(paths['cuda']), 'TORCH_HOME': str(paths['torch']),
        'XDG_CACHE_HOME': str(paths['xdg']), 'YOLO_CONFIG_DIR': str(PROJECT_ROOT / 'config/ultralytics'),
        'WANDB_MODE': 'disabled', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    })
    return paths


def check_dataset(
    dataset_yaml: Path, audit_report_path: Path, audit_summary_path: Path,
    expected: dict[str, Any], checks: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    for path, label in (
        (dataset_yaml, 'dataset_yaml'), (audit_report_path, 'audit_report'),
        (audit_summary_path, 'audit_summary'),
    ):
        add_check(checks, f'{label}_exists', path.is_file(), str(path))
        if not path.is_file():
            errors.append(f'{label} 不存在: {path}')
    if errors:
        return {'ready': False, 'errors': errors}
    dataset = dataset_yaml.parent
    add_check(checks, 'dataset_on_data_volume', is_within(dataset, DATA_ROOT), str(dataset))
    add_check(checks, 'complete_marker', (dataset / '.complete').is_file(), str(dataset / '.complete'))
    add_check(checks, 'augmentation_policy', (dataset / 'augmentation_policy.yaml').is_file(), str(dataset / 'augmentation_policy.yaml'))

    payload = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise PreflightError('dataset.yaml 顶层不是字典')
    names = names_to_list(payload.get('names'))
    expected_names = [str(value) for value in expected.get('classes', [])]
    add_check(checks, 'class_count', int(payload.get('nc', -1)) == int(expected.get('nc', 40)) == 40 and len(names) == 40, {'nc': payload.get('nc'), 'names': len(names)})
    add_check(checks, 'class_catalog_exact', names == expected_names, {'actual_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest()})
    add_check(checks, 'dataset_yaml_root', Path(str(payload.get('path', ''))).resolve() == dataset.resolve(), payload.get('path'))
    add_check(checks, 'dataset_yaml_split_paths', payload.get('train') == 'images/train' and payload.get('val') == 'images/val', {'train': payload.get('train'), 'val': payload.get('val')})

    audit = read_json(audit_report_path)
    reported_checks = audit.get('checks', {})
    failed_checks = [name for name, item in reported_checks.items() if not isinstance(item, dict) or item.get('ok') is not True]
    add_check(checks, 'independent_audit_passed', audit.get('passed') is True and audit.get('training_ready') is True, {'passed': audit.get('passed'), 'training_ready': audit.get('training_ready')})
    add_check(checks, 'independent_audit_all_checks', bool(reported_checks) and not failed_checks, failed_checks)
    add_check(checks, 'dataset_yaml_sha256', audit.get('dataset_yaml_sha256') == sha256_file(dataset_yaml), audit.get('dataset_yaml_sha256'))
    add_check(checks, 'audit_summary_sha256', audit.get('audit_summary_sha256') == sha256_file(audit_summary_path), audit.get('audit_summary_sha256'))
    expected_summary = {
        'images': {split: int(expected['images'][split]) for split in SPLITS},
        'boxes': {split: int(expected['boxes'][split]) for split in SPLITS},
        'source_images': {split: int(expected['source_images'][split]) for split in SPLITS},
        'augmented_images': int(expected['augmented_images']),
    }
    actual_summary = audit.get('summary', {})
    add_check(checks, 'audit_counts', all(actual_summary.get(key) == value for key, value in expected_summary.items()), {'expected': expected_summary, 'actual': {key: actual_summary.get(key) for key in expected_summary}})
    empty_expected = {split: int(expected['empty_labels'][split]) for split in SPLITS}
    add_check(checks, 'intentional_background_negatives', actual_summary.get('empty_labels') == empty_expected, {'expected': empty_expected, 'actual': actual_summary.get('empty_labels')})
    v4_images_check = reported_checks.get('source_v4_unique_images_321', {})
    v4_boxes_check = reported_checks.get('source_v4_boxes_1646', {})
    add_check(checks, 'multiobject_review_included', v4_images_check.get('detail') == int(expected['source_multiobject_images']) and v4_boxes_check.get('detail') == int(expected['source_multiobject_boxes']), {'images': v4_images_check.get('detail'), 'boxes': v4_boxes_check.get('detail')})

    builder = read_json(audit_summary_path)
    builder_names = [str(value) for value in builder.get('classes', {}).get('names', [])]
    add_check(checks, 'builder_class_catalog_exact', builder.get('classes', {}).get('count') == 40 and builder_names == expected_names, builder.get('classes', {}).get('count'))
    add_check(checks, 'builder_training_ready', builder.get('training_ready') is True and builder.get('training_started') is False, {'training_ready': builder.get('training_ready'), 'training_started': builder.get('training_started')})

    filesystem_counts: dict[str, Any] = {}
    for split in SPLITS:
        image_dir = dataset / 'images' / split
        label_dir = dataset / 'labels' / split
        images = {path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
        labels = {path.stem for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == '.txt'}
        augmented = sum(1 for stem in images if stem.startswith('aug__'))
        filesystem_counts[split] = {
            'images': len(images), 'labels': len(labels),
            'missing_labels': len(images - labels), 'extra_labels': len(labels - images),
            'augmented': augmented,
        }
        add_check(checks, f'{split}_filesystem_pairing', images == labels and len(images) == int(expected['images'][split]), filesystem_counts[split])
        add_check(checks, f'{split}_augmentation_policy', augmented == (int(expected['augmented_images']) if split == 'train' else 0), augmented)
    if any(item['ok'] is False for item in checks.values()):
        errors.extend(name for name, item in checks.items() if item['ok'] is False)
    return {'ready': not errors, 'errors': errors, 'classes': names, 'filesystem': filesystem_counts}


def check_weights(
    weights_path: Path, report_path: Path, expected: dict[str, Any],
    expected_names: list[str], checks: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    add_check(checks, 'base_weights_exists', weights_path.is_file(), str(weights_path))
    add_check(checks, 'base_weights_report_exists', report_path.is_file(), str(report_path))
    if not weights_path.is_file() or not report_path.is_file():
        return {'ready': False, 'errors': ['初始化权重或报告不存在']}
    report = read_json(report_path)
    weights_sha = sha256_file(weights_path)
    add_check(checks, 'init_report_passed', report.get('passed') is True, report.get('passed'))
    add_check(checks, 'init_weights_sha256', report.get('output_weights_sha256') == weights_sha, weights_sha)
    add_check(checks, 'init_source_nc_80', report.get('source_nc') == int(expected['init_source_nc']), report.get('source_nc'))
    add_check(checks, 'init_target_nc_40', report.get('target_nc') == 40 and report.get('verify', {}).get('nc') == 40, report.get('target_nc'))
    add_check(checks, 'init_classes_exact', report.get('target_names') == expected_names, len(report.get('target_names', [])))
    fresh_keys = report.get('fresh_class_output_tensors', [])
    add_check(checks, 'init_detection_head_fresh', len(fresh_keys) == int(expected['init_fresh_class_output_tensors']) and report.get('coco_class_outputs_reused') is False, {'fresh_tensors': len(fresh_keys), 'coco_reused': report.get('coco_class_outputs_reused')})
    add_check(checks, 'init_report_forward_ok', report.get('verify', {}).get('forward_ok') is True, report.get('verify', {}).get('forward_ok'))

    old_model = PROJECT_ROOT / 'models/best.pt'
    old_hash_ok = old_model.is_file() and sha256_file(old_model) == str(expected['old_model_sha256'])
    add_check(checks, 'old_model_unchanged', old_hash_ok, sha256_file(old_model) if old_model.is_file() else None)
    add_check(checks, 'new_weight_not_old_model', weights_sha != str(expected['old_model_sha256']), weights_sha)

    from ultralytics import YOLO
    wrapper = YOLO(str(weights_path), task='detect')
    model_names = names_to_list(wrapper.model.names)
    add_check(checks, 'init_load_40_classes', int(wrapper.model.nc) == 40 and model_names == expected_names, {'nc': int(wrapper.model.nc), 'names': len(model_names)})
    if any(item['ok'] is False for item in checks.values()):
        errors.extend(name for name, item in checks.items() if item['ok'] is False)
    return {'ready': not errors, 'errors': errors, 'sha256': weights_sha}


def check_runtime(expected: dict[str, Any], train_config: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
    add_check(checks, 'free_space', free_gb >= float(expected['min_free_gb']), {'free_gb': round(free_gb, 2), 'minimum_gb': expected['min_free_gb']})
    try:
        import torch
        add_check(checks, 'torch_version', True, torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        add_check(checks, 'cuda_available', cuda_available, cuda_available)
        if cuda_available:
            index = int(str(train_config.get('device', '0')).split(',')[0])
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            memory = {
                'name': props.name, 'capability': f'{props.major}.{props.minor}',
                'total_gb': round(total_bytes / (1024 ** 3), 2),
                'free_gb': round(free_bytes / (1024 ** 3), 2),
            }
            add_check(checks, 'gpu_memory', free_bytes / (1024 ** 3) >= float(expected['min_gpu_memory_gb']), memory)
            probe = torch.zeros(1, device=f'cuda:{index}')
            torch.cuda.synchronize(index)
            del probe
            add_check(checks, 'cuda_compute_probe', True, memory)
        add_check(checks, 'amp_requested', bool(train_config.get('amp')), train_config.get('amp'))
    except Exception as exc:
        add_check(checks, 'runtime', False, f'{type(exc).__name__}: {exc}')
    add_check(checks, 'ultralytics_version', True, package_version('ultralytics'))
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], capture_output=True, text=True, timeout=10, check=False)
        add_check(checks, 'nvidia_smi', result.returncode == 0, result.stdout.strip() or result.stderr.strip())
    except Exception as exc:
        add_check(checks, 'nvidia_smi', False, str(exc))
    if any(item['ok'] is False for item in checks.values()):
        errors.extend(name for name, item in checks.items() if item['ok'] is False)
    return {'ready': not errors, 'errors': errors}


def check_output_and_safety(
    output_root: Path, output_report: Path, weights_path: Path,
    train_config: dict[str, Any], checks: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    add_check(checks, 'output_on_data_volume', is_within(output_root, PROJECT_ROOT), str(output_root))
    add_check(checks, 'post_report_on_data_volume', is_within(output_report, PROJECT_ROOT), str(output_report))
    add_check(checks, 'fresh_training_only', train_config.get('resume') is False, {'resume': train_config.get('resume')})
    add_check(checks, 'weights_not_old_model', weights_path.resolve() != (PROJECT_ROOT / 'models/best.pt').resolve(), str(weights_path))
    existing = sorted(path.name for path in output_root.iterdir() if path.is_dir()) if output_root.is_dir() else []
    add_check(checks, 'output_root_namespace', output_root != PROJECT_ROOT / 'runs/track1_train_40class_v10_targetfix_from_v7', str(output_root))
    if any(item['ok'] is False for item in checks.values()):
        errors.extend(name for name, item in checks.items() if item['ok'] is False)
    return {'ready': not errors, 'errors': errors, 'existing_run_dirs': existing}


def build_preflight(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    configure_caches()
    checks: dict[str, Any] = {}
    dataset_yaml = resolve_path(config['dataset_yaml'])
    audit_report = resolve_path(config['dataset_audit_report'])
    audit_summary = resolve_path(config['dataset_audit_summary'])
    dataset_result = check_dataset(dataset_yaml, audit_report, audit_summary, config['expected'], checks)
    names = dataset_result.get('classes', [str(value) for value in config['expected'].get('classes', [])])
    weights_path = resolve_path(config['base_weights'])
    weights_result = check_weights(weights_path, resolve_path(config['base_weights_report']), config['expected'], names, checks)
    runtime_result = check_runtime(config['expected'], config['train'], checks)
    output_root = resolve_path(config['output_root'])
    output_report = resolve_path(config['post_train_report'])
    safety_result = check_output_and_safety(output_root, output_report, weights_path, config['train'], checks)
    blockers = [
        *dataset_result.get('errors', []), *weights_result.get('errors', []),
        *runtime_result.get('errors', []), *safety_result.get('errors', []),
    ]
    report = {
        'schema_version': 1,
        'created_at': now_iso(),
        'config': str(config_path),
        'dataset_yaml': str(dataset_yaml),
        'base_weights': str(weights_path),
        'base_weights_sha256': weights_result.get('sha256'),
        'output_root': str(output_root),
        'checks': checks,
        'blockers': blockers,
        'ready': not blockers,
        'training_started': False,
        'old_models_modified': False,
        'task_a_b_modified': False,
    }
    report_path = resolve_path(config['preflight_report'])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, report_path)
    return report


def metrics_to_dict(metrics: Any, names: list[str]) -> dict[str, Any]:
    box = metrics.box
    if box is None:
        return {'per_class': {}, 'class_count': 0}
    class_indices = [int(value) for value in box.ap_class_index.tolist()]
    ap_values = [float(value) for value in box.ap.tolist()]
    ap50_values = [float(value) for value in box.ap50.tolist()]
    precision_values = [float(value) for value in box.p.tolist()]
    recall_values = [float(value) for value in box.r.tolist()]
    per_class = {}
    for class_id, ap, ap50, precision, recall in zip(class_indices, ap_values, ap50_values, precision_values, recall_values):
        per_class[str(class_id)] = {
            'official_id': class_id + 1, 'name': names[class_id],
            'precision': precision, 'recall': recall, 'ap50': ap50, 'ap50_95': ap,
        }
    return {
        'class_count': len(class_indices), 'classes': class_indices,
        'map50': float(box.map50), 'map50_95': float(box.map),
        'per_class': per_class,
    }


def make_multiobject_val(dataset_yaml: Path, run_dir: Path, prefix: str) -> tuple[Path, int]:
    dataset = dataset_yaml.parent
    root = run_dir / 'post_train_audit/multiobject_val_data'
    image_dir = root / 'images/val'
    label_dir = root / 'labels/val'
    image_dir.mkdir(parents=True, exist_ok=False)
    label_dir.mkdir(parents=True, exist_ok=False)
    images = sorted(
        path for path in (dataset / 'images/val').iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem.startswith(prefix)
    )
    for image in images:
        label = dataset / 'labels/val' / f'{image.stem}.txt'
        if not label.is_file():
            raise PreflightError(f'多物品验证图片缺标签: {image}')
        (image_dir / image.name).symlink_to(image.resolve())
        (label_dir / label.name).symlink_to(label.resolve())
    payload = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))
    payload['path'] = str(root)
    payload['train'] = 'images/val'
    payload['val'] = 'images/val'
    yaml_path = root / 'dataset.yaml'
    yaml_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding='utf-8')
    if not images:
        raise PreflightError('验证集没有多物品图片')
    return yaml_path, len(images)


def acquire_training_lock():
    lock_path = PROJECT_ROOT / 'state/track1_v11_training.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open('w')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        print(f'已有 v11 训练进程持有锁: {lock_path}', file=sys.stderr)
        return None
    return lock_handle


def finalize_training(
    config_path: Path, config: dict[str, Any], report: dict[str, Any], save_dir: Path,
    old_model_sha256_before: str, resume_used: bool,
    resumed_from_epoch: int | None = None,
) -> int:
    from ultralytics import YOLO

    save_dir = save_dir.resolve()
    best_weights = save_dir / 'weights/best.pt'
    last_weights = save_dir / 'weights/last.pt'
    if not best_weights.is_file() or not last_weights.is_file():
        raise PreflightError(f'训练结束但权重不完整: {save_dir}')

    names = [str(value) for value in config['expected']['classes']]
    final_model = YOLO(str(best_weights), task='detect')
    final_names = names_to_list(final_model.model.names)
    if int(final_model.model.nc) != 40 or final_names != names:
        raise PreflightError('best.pt 类别槽或类别顺序异常')
    validation = final_model.val(
        data=str(resolve_path(config['dataset_yaml'])), split='val',
        imgsz=int(config['train']['imgsz']), batch=int(config['post_train']['final_validation_batch']),
        device=config['train']['device'], workers=int(config['train']['workers']),
        amp=bool(config['train']['amp']), conf=0.001, iou=0.6,
        plots=True, project=str(save_dir / 'post_train_validation'), name='all_val', exist_ok=True,
    )
    all_metrics = metrics_to_dict(validation, names)

    multi_yaml, multi_count = make_multiobject_val(
        resolve_path(config['dataset_yaml']), save_dir,
        str(config['post_train']['multiobject_val_prefix']),
    )
    multi_validation = YOLO(str(best_weights), task='detect').val(
        data=str(multi_yaml), split='val',
        imgsz=int(config['train']['imgsz']), batch=int(config['post_train']['final_validation_batch']),
        device=config['train']['device'], workers=int(config['train']['workers']),
        amp=bool(config['train']['amp']), conf=0.001, iou=0.6,
        plots=True, project=str(save_dir / 'post_train_audit'), name='multiobject_val_results', exist_ok=True,
    )
    multi_metrics = metrics_to_dict(multi_validation, names)
    old_hash_after = sha256_file(PROJECT_ROOT / 'models/best.pt')
    post_report = {
        'schema_version': 1,
        'created_at': now_iso(),
        'training_config': str(config_path),
        'run_dir': str(save_dir),
        'best_weights': str(best_weights),
        'best_weights_sha256': sha256_file(best_weights),
        'last_weights': str(last_weights),
        'last_weights_sha256': sha256_file(last_weights),
        'base_weights': str(resolve_path(config['base_weights'])),
        'base_weights_sha256': report['base_weights_sha256'],
        'dataset_yaml': str(resolve_path(config['dataset_yaml'])),
        'all_val': all_metrics,
        'multiobject_val': {'images': multi_count, **multi_metrics},
        'class_coverage_40': all_metrics.get('class_count') == 40 and multi_metrics.get('class_count') == 40,
        'old_model_sha256_before': old_model_sha256_before,
        'old_model_sha256_after': old_hash_after,
        'old_model_unchanged': old_hash_after == old_model_sha256_before,
        'resume_used': resume_used,
        'resumed_from_epoch': resumed_from_epoch,
        'task_a_b_modified': False,
        'training_environment': {
            'python_package_version': package_version('ultralytics'),
            'torch_version': __import__('torch').__version__,
        },
    }
    post_path = save_dir / 'post_train_audit.json'
    post_path.write_text(json.dumps(post_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    latest = resolve_path(config['post_train_report'])
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(post_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'training_complete', 'run_dir': str(save_dir),
        'best_weights': str(best_weights), 'best_sha256': post_report['best_weights_sha256'],
        'all_val_map50': all_metrics.get('map50'), 'all_val_map50_95': all_metrics.get('map50_95'),
        'multiobject_val_map50': multi_metrics.get('map50'), 'multiobject_val_map50_95': multi_metrics.get('map50_95'),
        'class_coverage_40': post_report['class_coverage_40'],
        'old_model_unchanged': post_report['old_model_unchanged'],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if post_report['class_coverage_40'] and post_report['old_model_unchanged'] else 5


def execute_training(config_path: Path, config: dict[str, Any], report: dict[str, Any]) -> int:
    if not report['ready']:
        print('预检未通过，禁止训练：', json.dumps(report['blockers'], ensure_ascii=False), file=sys.stderr)
        return 2
    lock_handle = acquire_training_lock()
    if lock_handle is None:
        return 3

    from ultralytics import YOLO

    output_root = resolve_path(config['output_root'])
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = f'{config["run_name_prefix"]}_{datetime.now().astimezone():%Y%m%d_%H%M%S}'
    run_dir = output_root / run_name
    if run_dir.exists():
        print(f'训练目录已存在，拒绝覆盖: {run_dir}', file=sys.stderr)
        return 4
    old_model_sha256_before = sha256_file(PROJECT_ROOT / 'models/best.pt')
    start_payload = {
        'schema_version': 1,
        'started_at': now_iso(),
        'config': str(config_path),
        'dataset_yaml': str(resolve_path(config['dataset_yaml'])),
        'base_weights': str(resolve_path(config['base_weights'])),
        'base_weights_sha256': report['base_weights_sha256'],
        'run_dir': str(run_dir),
        'resume': False,
        'pretrained': True,
        'training_args': config['train'],
        'old_model_sha256': old_model_sha256_before,
        'task_a_b_modified': False,
    }
    start_path = output_root / 'training_start.json'
    start_path.write_text(json.dumps(start_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'training_start', **start_payload}, ensure_ascii=False), flush=True)

    model = YOLO(str(resolve_path(config['base_weights'])), task='detect')
    train_args = dict(config['train'])
    train_args['project'] = str(output_root)
    train_args['name'] = run_name
    train_args['exist_ok'] = False
    train_args['resume'] = False
    model.train(data=str(resolve_path(config['dataset_yaml'])), **train_args)
    save_dir = Path(model.trainer.save_dir).resolve()
    if save_dir != run_dir.resolve():
        raise PreflightError(f'训练输出目录异常: expected={run_dir}, actual={save_dir}')
    return finalize_training(
        config_path, config, report, save_dir, old_model_sha256_before,
        resume_used=False,
    )


def resume_training(
    config_path: Path, config: dict[str, Any], report: dict[str, Any], resume_run: Path,
) -> int:
    if not report['ready']:
        print('预检未通过，禁止续训：', json.dumps(report['blockers'], ensure_ascii=False), file=sys.stderr)
        return 2
    output_root = resolve_path(config['output_root'])
    run_dir = resume_run.expanduser().resolve()
    if run_dir == output_root or not is_within(run_dir, output_root) or not run_dir.is_dir():
        print(f'续训目录不合法或不存在: {run_dir}', file=sys.stderr)
        return 4
    last_weights = run_dir / 'weights/last.pt'
    best_weights = run_dir / 'weights/best.pt'
    start_path = output_root / 'training_start.json'
    if not last_weights.is_file() or not best_weights.is_file() or not start_path.is_file():
        print(f'续训所需检查点或原始训练记录不完整: {run_dir}', file=sys.stderr)
        return 4

    start_payload = read_json(start_path)
    if Path(str(start_payload.get('run_dir', ''))).resolve() != run_dir:
        print('training_start.json 与续训目录不一致，拒绝续训', file=sys.stderr)
        return 4
    expected_old_sha = str(config['expected']['old_model_sha256'])
    old_model_sha256_before = str(start_payload.get('old_model_sha256', ''))
    old_model_now = sha256_file(PROJECT_ROOT / 'models/best.pt')
    if old_model_sha256_before != expected_old_sha or old_model_now != expected_old_sha:
        print('旧模型哈希不符合原始训练记录，拒绝续训', file=sys.stderr)
        return 4

    lock_handle = acquire_training_lock()
    if lock_handle is None:
        return 3

    from ultralytics import YOLO

    model = YOLO(str(last_weights), task='detect')
    if not model.ckpt or model.ckpt.get('optimizer') is None or int(model.ckpt.get('epoch', -1)) < 0:
        print('last.pt 不包含可恢复的优化器状态，拒绝伪续训', file=sys.stderr)
        return 4
    names = [str(value) for value in config['expected']['classes']]
    if int(model.model.nc) != 40 or names_to_list(model.model.names) != names:
        print('last.pt 类别槽或顺序异常，拒绝续训', file=sys.stderr)
        return 4

    checkpoint_epoch = int(model.ckpt['epoch'])
    resumed_from_epoch = checkpoint_epoch + 2
    resume_payload = {
        'schema_version': 1,
        'resumed_at': now_iso(),
        'config': str(config_path),
        'run_dir': str(run_dir),
        'last_weights': str(last_weights),
        'last_weights_sha256': sha256_file(last_weights),
        'checkpoint_epoch_zero_based': checkpoint_epoch,
        'resume_next_epoch_one_based': resumed_from_epoch,
        'reason': '训练终端会话达到运行时限，原 run 检查点完整；仅续跑同一 run',
        'old_model_sha256': old_model_sha256_before,
        'task_a_b_modified': False,
    }
    resume_path = run_dir / 'training_resume.json'
    resume_path.write_text(json.dumps(resume_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output_root / 'training_resume_history.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(resume_payload, ensure_ascii=False) + '\n')
    print(json.dumps({'event': 'training_resume', **resume_payload}, ensure_ascii=False), flush=True)

    # Ultralytics restores the complete checkpoint args, optimizer, scheduler, EMA,
    # epoch, best fitness, and original run directory from last.pt.
    model.train(resume=True)
    save_dir = Path(model.trainer.save_dir).resolve()
    if save_dir != run_dir:
        raise PreflightError(f'续训输出目录异常: expected={run_dir}, actual={save_dir}')
    return finalize_training(
        config_path, config, report, save_dir, old_model_sha256_before,
        resume_used=True, resumed_from_epoch=resumed_from_epoch,
    )


def finalize_existing_training(
    config_path: Path, config: dict[str, Any], report: dict[str, Any], run_dir_arg: Path,
) -> int:
    if not report['ready']:
        print('预检未通过，禁止训练后评估：', json.dumps(report['blockers'], ensure_ascii=False), file=sys.stderr)
        return 2
    output_root = resolve_path(config['output_root'])
    run_dir = run_dir_arg.expanduser().resolve()
    if run_dir == output_root or not is_within(run_dir, output_root) or not run_dir.is_dir():
        print(f'评估目录不合法或不存在: {run_dir}', file=sys.stderr)
        return 4
    start_path = output_root / 'training_start.json'
    resume_path = run_dir / 'training_resume.json'
    if not start_path.is_file() or not resume_path.is_file():
        print('缺少原始训练或续训记录，拒绝训练后评估', file=sys.stderr)
        return 4
    start_payload = read_json(start_path)
    resume_payload = read_json(resume_path)
    if Path(str(start_payload.get('run_dir', ''))).resolve() != run_dir:
        print('training_start.json 与评估目录不一致，拒绝评估', file=sys.stderr)
        return 4
    expected_old_sha = str(config['expected']['old_model_sha256'])
    old_model_sha256_before = str(start_payload.get('old_model_sha256', ''))
    old_model_now = sha256_file(PROJECT_ROOT / 'models/best.pt')
    if old_model_sha256_before != expected_old_sha or old_model_now != expected_old_sha:
        print('旧模型哈希不符合原始训练记录，拒绝评估', file=sys.stderr)
        return 4
    lock_handle = acquire_training_lock()
    if lock_handle is None:
        return 3
    return finalize_training(
        config_path, config, report, run_dir, old_model_sha256_before,
        resume_used=bool(resume_payload.get('schema_version')),
        resumed_from_epoch=resume_payload.get('resume_next_epoch_one_based'),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='赛道一 v11 40 类全量训练预检；只有 --execute 才训练')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--execute', action='store_true', help='明确启动训练')
    parser.add_argument('--resume-run', type=Path, help='仅续跑指定 v11 run 的 last.pt；必须同时给出 --execute')
    parser.add_argument('--finalize-run', type=Path, help='仅对已完成的指定 v11 run 做训练后评估；必须同时给出 --execute')
    args = parser.parse_args(argv)
    if args.resume_run is not None and args.finalize_run is not None:
        parser.error('--resume-run 与 --finalize-run 不能同时使用')
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    report = build_preflight(config_path, config)
    print(json.dumps({
        'ready': report['ready'], 'blockers': report['blockers'],
        'preflight_report': str(resolve_path(config['preflight_report'])),
        'base_weights': report['base_weights'], 'base_weights_sha256': report['base_weights_sha256'],
        'output_root': report['output_root'],
    }, ensure_ascii=False, indent=2))
    if not report['ready'] or not args.execute:
        return 0 if report['ready'] else 1
    if args.finalize_run is not None:
        return finalize_existing_training(config_path, config, report, args.finalize_run)
    if args.resume_run is not None:
        return resume_training(config_path, config, report, args.resume_run)
    return execute_training(config_path, config, report)


if __name__ == '__main__':
    raise SystemExit(main())
