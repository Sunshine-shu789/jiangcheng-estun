#!/usr/bin/env python3
"""Preflight and train the isolated 3-class can detector."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
DEFAULT_CONFIG = ROOT / "config/track1_train_can_secondary_v1.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError("training config is not a mapping")
    return cfg


def check_dataset_yaml(dataset_yaml: Path, expected: dict) -> dict:
    with dataset_yaml.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = dataset_yaml.parent
    if int(data.get("nc", -1)) != int(expected["nc"]):
        raise RuntimeError(f"nc mismatch: {data.get('nc')} != {expected['nc']}")
    if data.get("names") != expected["names"]:
        raise RuntimeError(f"names mismatch: {data.get('names')} != {expected['names']}")

    result = {"classes": {}, "images": {}, "labels": {}, "empty_labels": {}, "bad_rows": []}
    for split in ("train", "val"):
        image_dir = root / data[split]
        label_dir = root / "labels" / split
        images = sorted(p for p in image_dir.iterdir() if p.is_file())
        labels = sorted(p for p in label_dir.glob("*.txt") if p.is_file())
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        if image_stems != label_stems:
            raise RuntimeError(
                f"{split} image/label stems differ: missing_labels={len(image_stems-label_stems)} "
                f"missing_images={len(label_stems-image_stems)}"
            )
        counts = Counter()
        empty = 0
        for label in labels:
            text = label.read_text(encoding="utf-8")
            if not text.strip():
                empty += 1
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    result["bad_rows"].append(f"{label}:{line_no}")
                    continue
                cls = int(float(fields[0]))
                if cls not in (0, 1, 2):
                    result["bad_rows"].append(f"{label}:{line_no}:class={cls}")
                vals = [float(v) for v in fields[1:]]
                if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    result["bad_rows"].append(f"{label}:{line_no}:bbox={vals}")
                counts[cls] += 1
        result["images"][split] = len(images)
        result["labels"][split] = len(labels)
        result["empty_labels"][split] = empty
        result["classes"][split] = {str(k): v for k, v in sorted(counts.items())}
        if len(images) != expected["images"][split]:
            raise RuntimeError(f"image count mismatch {split}: {len(images)} != {expected['images'][split]}")
        if counts != Counter(expected["instances"][split]):
            raise RuntimeError(f"instance count mismatch {split}: {dict(counts)} != {expected['instances'][split]}")
    if result["bad_rows"]:
        raise RuntimeError(f"bad label rows: {result['bad_rows'][:10]}")

    symlinks = []
    for p in root.rglob("*"):
        if p.is_symlink():
            symlinks.append(str(p))
            if len(symlinks) >= 10:
                break
    if symlinks:
        raise RuntimeError(f"dataset contains symlinks: {symlinks}")
    return result


def check_audit(dataset_yaml: Path, expected: dict) -> dict:
    audit_path = dataset_yaml.parent / "audit_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("materialization") != expected["materialization"]:
        raise RuntimeError("dataset materialization policy mismatch")
    if audit.get("source_policy", {}).get("official_competition_dataset_used") is not False:
        raise RuntimeError("official competition dataset flag is not false")
    if audit.get("source_policy", {}).get("same_physical_group_across_train_val") is not False:
        raise RuntimeError("physical group split is not isolated")
    return {"path": str(audit_path), "sha256_checked_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def check_holdout(cfg: dict) -> dict:
    path = Path(cfg["holdout_yaml"])
    if not path.is_file():
        raise RuntimeError(f"holdout yaml missing: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = path.parent
    images = sorted((root / data["val"]).glob("*"))
    labels = sorted((root / "labels/val").glob("*.txt"))
    if len(images) != 1 or len(labels) != 1 or images[0].stem != labels[0].stem:
        raise RuntimeError(f"holdout expected exactly one labeled image: images={len(images)} labels={len(labels)}")
    classes = Counter()
    for line in labels[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            classes[int(float(line.split()[0]))] += 1
    if classes != Counter({0: 1, 1: 1, 2: 1}):
        raise RuntimeError(f"holdout must contain exactly one of each can: {dict(classes)}")
    return {"image": str(images[0]), "label": str(labels[0]), "classes": dict(classes)}


def preflight(cfg: dict) -> dict:
    root = Path(cfg["dataset_yaml"]).parent
    expected = cfg["expected"]
    dataset = check_dataset_yaml(Path(cfg["dataset_yaml"]), expected)
    audit = check_audit(Path(cfg["dataset_yaml"]), expected)
    holdout = check_holdout(cfg)

    base = Path(cfg["base_weights"])
    if not base.is_file() or base.stat().st_size < 1_000_000:
        raise RuntimeError(f"base weights missing or invalid: {base}")

    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    if free_gb < expected["min_free_gb"]:
        raise RuntimeError(f"free space too low: {free_gb:.2f} GiB")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gpu_gb = free_bytes / (1024**3)
    if free_gpu_gb < expected["min_gpu_free_gb"]:
        raise RuntimeError(f"GPU free memory too low: {free_gpu_gb:.2f} GiB")
    torch.cuda.reset_peak_memory_stats(0)
    torch.empty(1, device="cuda")

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(DEFAULT_CONFIG),
        "dataset": dataset,
        "audit": audit,
        "holdout": holdout,
        "base_weights": {"path": str(base), "size": base.stat().st_size},
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_free_gb": round(free_gpu_gb, 3),
            "gpu_total_gb": round(total_bytes / (1024**3), 3),
        },
        "disk": {"free_gb": round(free_gb, 3)},
        "ok": True,
    }
    report_path = Path(cfg["preflight_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run_training(cfg: dict) -> Path:
    from ultralytics import YOLO

    train_cfg = dict(cfg["train"])
    output_root = Path(cfg["output_root"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"can3_{stamp}"
    params = dict(train_cfg)
    params.update({
        "data": str(Path(cfg["dataset_yaml"])),
        "project": str(output_root),
        "name": name,
        "exist_ok": False,
    })
    model = YOLO(str(Path(cfg["base_weights"])))
    print(f"starting isolated can3 training: {output_root / name}", flush=True)
    model.train(**params)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"training finished without best.pt: {best}")

    metrics = model.val(
        data=str(Path(cfg["dataset_yaml"])),
        split="val",
        imgsz=int(train_cfg["imgsz"]),
        batch=int(train_cfg["batch"]),
        device=train_cfg["device"],
        conf=cfg["holdout_eval"]["conf"],
        iou=cfg["holdout_eval"]["iou"],
        plots=True,
        project=str(output_root),
        name=f"val_{stamp}",
    )
    names = {i: n for i, n in enumerate(cfg["expected"]["names"])}
    per_class = {}
    if metrics.box is not None:
        maps = metrics.box.maps.tolist()
        p = metrics.box.p.tolist()
        r = metrics.box.r.tolist()
        ap50 = metrics.box.ap50.tolist()
        ap = metrics.box.ap.tolist()
        for i, name_i in names.items():
            if i < len(maps):
                per_class[name_i] = {
                    "precision": float(p[i]),
                    "recall": float(r[i]),
                    "ap50": float(ap50[i]),
                    "ap50_95": float(ap[i]),
                    "map": float(maps[i]),
                }

    holdout_model = YOLO(str(best))
    holdout_eval = cfg["holdout_eval"]
    holdout_dir = output_root / f"holdout_{stamp}"
    predictions = holdout_model.predict(
        source=str(Path(cfg["holdout_yaml"]).parent / "images/val"),
        imgsz=int(train_cfg["imgsz"]),
        conf=float(holdout_eval["conf"]),
        iou=float(holdout_eval["iou"]),
        max_det=int(holdout_eval["max_det"]),
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_root),
        name=f"holdout_predict_{stamp}",
        verbose=True,
    )
    pred_rows = []
    for result in predictions:
        if result.boxes is None:
            continue
        for box in result.boxes:
            pred_rows.append({
                "class_id": int(box.cls.item()),
                "class_name": names[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "xyxy": [float(x) for x in box.xyxy[0].tolist()],
            })

    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "save_dir": str(save_dir),
        "best_weights": str(best),
        "best_weights_sha256": __import__("hashlib").sha256(best.read_bytes()).hexdigest(),
        "metrics": {
            "fitness": float(metrics.fitness) if getattr(metrics, "fitness", None) is not None else None,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "per_class": per_class,
        },
        "holdout": {
            "conf": holdout_eval["conf"],
            "iou": holdout_eval["iou"],
            "predictions": pred_rows,
            "prediction_dir": str(holdout_dir),
        },
    }
    (save_dir / "post_train_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "latest_can_secondary_training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    preflight(cfg)
    if not args.execute:
        print("preflight only; add --execute to start training")
        return 0
    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
