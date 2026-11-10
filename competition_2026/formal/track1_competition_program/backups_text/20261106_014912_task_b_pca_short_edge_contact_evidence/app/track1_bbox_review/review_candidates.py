#!/usr/bin/env python3
"""Generate human-reviewable bbox candidates for offline Track-1 review."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

CAPTURE_APP = Path(
    "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_capture_web"
)
if str(CAPTURE_APP) not in sys.path:
    sys.path.insert(0, str(CAPTURE_APP))

from capture_store import load_class_catalog  # noqa: E402
from depth_plane_segment import (  # noqa: E402
    Intrinsics,
    clean_mask,
    component_summary,
    fit_dominant_plane,
    unproject,
)
from review_core import (  # noqa: E402
    atomic_write_json,
    load_review_index,
    load_yaml,
    now_iso,
)


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0,
        float(box[3]) - float(box[1]),
    )


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _valid_box(
    box: Sequence[float],
    *,
    width: int,
    height: int,
    is_depth: bool,
) -> bool:
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return False
    area_fraction = _box_area(box) / (width * height)
    if area_fraction >= 0.65:
        return False
    edge_hits = sum(
        (
            x1 <= 3.0,
            y1 <= 3.0,
            x2 >= width - 4.0,
            y2 >= height - 4.0,
        )
    )
    if not is_depth and edge_hits >= 2:
        return False
    return True


def generate_depth_candidates(
    *,
    image: np.ndarray,
    depth_raw: np.ndarray,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    height, width = image.shape[:2]
    raw_intr = metadata["intrinsics"]
    intrinsics = Intrinsics(
        width=int(raw_intr["width"]),
        height=int(raw_intr["height"]),
        fx=float(raw_intr["fx"]),
        fy=float(raw_intr["fy"]),
        ppx=float(raw_intr.get("ppx", raw_intr.get("cx"))),
        ppy=float(raw_intr.get("ppy", raw_intr.get("cy"))),
    )
    if (intrinsics.width, intrinsics.height) != (width, height):
        raise ValueError("深度/彩色图像尺寸与内参不一致")
    scale = float(metadata["depth_scale_m_per_unit"])
    depth_m = depth_raw.astype(np.float32) * scale
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= float(config["min_depth_m"]))
        & (depth_m <= float(config["max_depth_m"]))
    )
    ys, xs = np.nonzero(valid)
    if len(ys) < 50:
        raise ValueError("有效深度点不足")
    points = unproject(depth_m, valid, intrinsics)
    sample_step = max(1, int(config["depth_sample_step"]))
    plane, _ = fit_dominant_plane(
        points[::sample_step],
        float(config["plane_threshold_m"]),
        int(config["ransac_iterations"]),
        seed=int(config.get("ransac_seed", 20261028)),
    )
    distances = points @ plane.normal + plane.offset
    height_map = np.full(valid.shape, np.nan, dtype=np.float32)
    height_map[ys, xs] = distances
    table_mask = np.zeros(valid.shape, dtype=bool)
    table_mask[ys, xs] = np.abs(distances) <= float(config["plane_threshold_m"])
    object_mask = np.zeros(valid.shape, dtype=bool)
    object_mask[ys, xs] = (
        (distances >= float(config["object_min_height_m"]))
        & (distances <= float(config["object_max_height_m"]))
    )
    object_mask = clean_mask(
        object_mask,
        int(config["morph_kernel"]),
        int(config["min_component_pixels"]),
        bool(config.get("keep_border_components", True)),
    )
    _, object_pixels, _, components = component_summary(object_mask, height_map)
    candidates: list[dict[str, Any]] = []
    for component in components:
        x, y, box_width, box_height = (
            int(value) for value in component["bbox_xywh"]
        )
        box = [float(x), float(y), float(x + box_width), float(y + box_height)]
        if not _valid_box(
            box,
            width=width,
            height=height,
            is_depth=True,
        ):
            continue
        candidates.append(
            {
                "candidate_id": f"D{len(candidates) + 1:02d}",
                "source": "depth",
                "bbox_xyxy": [round(value, 2) for value in box],
                "area_pixels": int(component["area_pixels"]),
                "height_median_m": component.get("height_median_m"),
                "height_p95_m": component.get("height_p95_m"),
            }
        )
    candidates.sort(key=lambda item: -int(item["area_pixels"]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"D{index:02d}"
    stats = {
        "valid_depth_fraction": float(valid.mean()),
        "table_fraction_of_valid": float(table_mask.sum() / max(valid.sum(), 1)),
        "object_pixels": int(object_pixels),
        "depth_candidate_count": len(candidates),
        "plane_normal_camera_xyz": [float(value) for value in plane.normal],
        "camera_plane_distance_m": float(plane.offset),
    }
    return candidates, stats


class YoloWorldCandidates:
    def __init__(
        self,
        *,
        weights: Path,
        prompts: Mapping[int, str],
        excluded_class_ids: Sequence[int] = (17, 26),
    ) -> None:
        from ultralytics import YOLOWorld

        if not weights.is_file():
            raise ValueError(f"YOLO-World 权重不存在：{weights}")
        missing = sorted(set(range(1, 41)) - set(prompts))
        if missing:
            raise ValueError(f"YOLO-World 提示词缺少类别：{missing}")
        self.model = YOLOWorld(str(weights))
        self.class_ids = list(range(1, 41))
        self.excluded_class_ids = {int(value) for value in excluded_class_ids}
        self.names = [str(prompts[class_id]) for class_id in self.class_ids]
        self.model.set_classes(self.names)

    def infer(
        self,
        image: np.ndarray,
        *,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        max_det: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        height, width = image.shape[:2]
        result = self.model.predict(
            image,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=device,
            verbose=False,
            max_det=int(max_det),
        )[0]
        raw: list[dict[str, Any]] = []
        warnings: list[str] = []
        for box in result.boxes:
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf.item())
            model_index = int(box.cls.item())
            if not 0 <= model_index < len(self.class_ids):
                continue
            if not _valid_box(
                xyxy,
                width=width,
                height=height,
                is_depth=False,
            ):
                continue
            class_id = self.class_ids[model_index]
            if class_id in self.excluded_class_ids:
                continue
            raw.append(
                {
                    "source": "yolo_world",
                    "class_id": class_id,
                    "prompt": self.names[model_index],
                    "confidence": confidence,
                    "bbox_xyxy": [round(value, 2) for value in xyxy],
                }
            )
        raw.sort(key=lambda item: -float(item["confidence"]))
        kept: list[dict[str, Any]] = []
        for candidate in raw:
            duplicate = any(
                int(existing["class_id"]) == int(candidate["class_id"])
                and _box_iou(
                    existing["bbox_xyxy"],
                    candidate["bbox_xyxy"],
                )
                >= 0.85
                for existing in kept
            )
            if duplicate:
                continue
            candidate["candidate_id"] = f"W{len(kept) + 1:02d}"
            kept.append(candidate)
            if len(kept) >= 80:
                warnings.append("YOLO-World 候选超过 80 个，已截断")
                break
        return kept, warnings


def _resolve_optional(root: Path, value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def generate_candidate_file(
    *,
    item: Mapping[str, Any],
    source_root: Path,
    config: Mapping[str, Any],
    output_path: Path,
    world: YoloWorldCandidates | None,
) -> dict[str, Any]:
    image_path = source_root / str(item["source_rgb"])
    depth_path = source_root / str(item["source_depth"])
    meta_path = source_root / str(item["source_meta"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取 RGB：{image_path}")
    if depth_raw is None or depth_raw.dtype != np.uint16:
        raise ValueError(f"无法读取 uint16 深度图：{depth_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    warnings: list[str] = []
    try:
        depth_candidates, depth_stats = generate_depth_candidates(
            image=image,
            depth_raw=depth_raw,
            metadata=metadata,
            config=config,
        )
    except Exception as error:
        depth_candidates = []
        depth_stats = {}
        warnings.append(f"深度候选失败：{type(error).__name__}: {error}")

    world_candidates: list[dict[str, Any]] = []
    if world is not None:
        try:
            world_candidates, world_warnings = world.infer(
                image,
                conf=float(config["world_conf"]),
                iou=float(config["world_iou"]),
                imgsz=int(config["world_imgsz"]),
                device=str(config["world_device"]),
                max_det=int(config["world_max_det"]),
            )
            warnings.extend(world_warnings)
        except Exception as error:
            warnings.append(f"YOLO-World 候选失败：{type(error).__name__}: {error}")

    payload = {
        "schema_version": 1,
        "review_id": item["review_id"],
        "source_frame_id": item["selected_frame_id"],
        "generated_at": now_iso(),
        "capture_hint_class_id": item["hint_class_id"],
        "important_note": (
            "所有候选都只是提示，不是训练标签；必须人工确认，"
            "并补齐同图其他类别和同类其他实例。"
        ),
        "depth_stats": depth_stats,
        "candidates": [*depth_candidates, *world_candidates],
        "warnings": warnings,
    }
    atomic_write_json(output_path, payload)
    return payload


def prepare_all_candidates(
    *,
    config_path: Path,
    force: bool = False,
    limit: int | None = None,
    with_world: bool = True,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    paths = config["paths"]
    project_root = Path(str(paths["project_root"])).resolve()
    source_root = _resolve_optional(project_root, paths["source_root"])
    review_root = _resolve_optional(project_root, paths["review_root"])
    class_map = _resolve_optional(project_root, paths["class_map"])
    if source_root is None or review_root is None or class_map is None:
        raise ValueError("配置缺少 source_root、review_root 或 class_map")
    index = load_review_index(review_root)
    catalog = load_class_catalog(class_map)

    world: YoloWorldCandidates | None = None
    if with_world:
        weights = _resolve_optional(project_root, paths.get("world_weights"))
        if weights is None:
            raise ValueError("启用 YOLO-World 时缺少 world_weights")
        prompts = {
            int(key): str(value)
            for key, value in dict(config.get("world_prompts") or {}).items()
        }
        # Validate all catalog IDs are represented before paying model init cost.
        if set(prompts) != {entry.class_id for entry in catalog.entries}:
            raise ValueError("world_prompts 必须完整覆盖官方类别 1-40")
        policy = dict(config.get("review_policy") or {})
        excluded_class_ids = [
            int(value) for value in policy.get("missing_spots_class_ids", [17, 26])
        ]
        world = YoloWorldCandidates(
            weights=weights,
            prompts=prompts,
            excluded_class_ids=excluded_class_ids,
        )

    items = list(index["items"])
    if limit is not None:
        items = items[: max(0, int(limit))]
    cache_root = review_root / "candidates"
    cache_root.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    source_counter: Counter[str] = Counter()
    for position, item in enumerate(items, start=1):
        output_path = cache_root / f"{item['review_id']}.json"
        if output_path.is_file() and not force:
            skipped += 1
            continue
        try:
            payload = generate_candidate_file(
                item=item,
                source_root=source_root,
                config=dict(config["candidate_generation"]),
                output_path=output_path,
                world=world,
            )
            generated += 1
            for candidate in payload.get("candidates", []):
                source_counter[str(candidate["source"])] += 1
        except Exception as error:
            failures.append(
                {
                    "review_id": str(item["review_id"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if position % 25 == 0 or position == len(items):
            print(
                f"候选预生成 {position}/{len(items)}；新生成={generated}，"
                f"跳过={skipped}，失败={len(failures)}",
                flush=True,
            )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(config_path.resolve()),
        "source_root": str(source_root),
        "review_root": str(review_root),
        "with_world": with_world,
        "requested_items": len(items),
        "generated_files": generated,
        "skipped_existing": skipped,
        "failure_count": len(failures),
        "failures": failures,
        "candidate_source_counts": dict(source_counter),
        "disclaimer": "候选框不能直接作为训练标签，必须人工多物品复核。",
    }
    atomic_write_json(review_root / "candidate_generation_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="预生成 Track-1 多物品 bbox 候选")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-world", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_all_candidates(
        config_path=Path(args.config),
        force=bool(args.force),
        limit=args.limit,
        with_world=not bool(args.no_world),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
