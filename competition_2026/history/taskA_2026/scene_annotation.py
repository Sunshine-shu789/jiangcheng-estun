#!/usr/bin/env python3
"""Helpers for freezing, reviewing, and saving random RGB-D scenes."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from depth_plane_segment import overlay_result


def _box_iou(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float]:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0,
        float(first[3]) - float(first[1]),
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0,
        float(second[3]) - float(second[1]),
    )
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0 else 0.0
    cover = intersection / second_area if second_area > 0 else 0.0
    return iou, cover


def _candidate_bbox_xyxy(candidate: Mapping[str, Any]) -> list[float]:
    bbox_xyxy = candidate.get("bbox_xyxy")
    if bbox_xyxy is not None:
        return [float(value) for value in bbox_xyxy]
    x, y, width, height = (
        float(value) for value in candidate["bbox_xywh"]
    )
    return [x, y, x + width, y + height]


def add_detection_suggestions(
    candidates: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach conservative class suggestions to depth candidates."""
    results = [dict(candidate) for candidate in candidates]
    for candidate in results:
        bbox = _candidate_bbox_xyxy(candidate)
        candidate_area = max(1.0, float(candidate["area_pixels"]))
        x1, y1, x2, y2 = (float(value) for value in bbox)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        best: tuple[float, Mapping[str, Any]] | None = None
        for detection in detections:
            detection_bbox = detection["bbox"]
            detection_area = max(
                1.0,
                (float(detection_bbox[2]) - float(detection_bbox[0]))
                * (float(detection_bbox[3]) - float(detection_bbox[1])),
            )
            iou, cover = _box_iou(detection_bbox, bbox)
            center_inside = (
                float(detection_bbox[0]) <= center_x <= float(detection_bbox[2])
                and float(detection_bbox[1])
                <= center_y
                <= float(detection_bbox[3])
            )
            if not center_inside or detection_area > candidate_area * 5.0:
                continue
            if iou < 0.10 and cover < 0.35:
                continue
            score = float(detection["confidence"]) + 0.30 * iou + 0.10 * cover
            if best is None or score > best[0]:
                best = (score, detection)
        if best is not None:
            detection = best[1]
            candidate["suggested_class"] = str(detection["class"])
            candidate["suggested_confidence"] = float(
                detection["confidence"]
            )
    return results


def render_scene_preview(
    frame: np.ndarray,
    table_mask: np.ndarray,
    object_mask: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Render numbered depth candidates for interactive review."""
    preview = overlay_result(frame, table_mask, object_mask)
    height, width = preview.shape[:2]
    colors = (
        (65, 188, 255),
        (80, 220, 120),
        (255, 170, 70),
        (220, 110, 240),
        (90, 220, 230),
        (255, 220, 80),
    )
    for index, candidate in enumerate(candidates):
        x, y, box_width, box_height = (
            int(value) for value in candidate["bbox_xywh"]
        )
        x1 = max(0, min(x, width - 1))
        y1 = max(0, min(y, height - 1))
        x2 = max(0, min(x + box_width - 1, width - 1))
        y2 = max(0, min(y + box_height - 1, height - 1))
        color = colors[index % len(colors)]
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

        label = f"#{index + 1}"
        suggested_class = candidate.get("suggested_class")
        if suggested_class:
            confidence = float(candidate.get("suggested_confidence") or 0.0)
            label += f" {suggested_class} {confidence * 100:.0f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )
        label_top = y1 - text_height - baseline - 5
        if label_top < 0:
            label_top = y1 + 2
        label_right = min(x1 + text_width + 8, width - 1)
        label_bottom = min(
            label_top + text_height + baseline + 5,
            height - 1,
        )
        cv2.rectangle(
            preview,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )
        cv2.putText(
            preview,
            label,
            (x1 + 4, label_bottom - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (16, 20, 24),
            1,
            cv2.LINE_AA,
        )
    return preview


def save_scene(
    scene_root: Path,
    frame: np.ndarray,
    depth_m: np.ndarray,
    depth_scale_m_per_unit: float,
    intrinsics: Any,
    candidates: Sequence[Mapping[str, Any]],
    assignments: Mapping[int | str, int],
    class_names: Sequence[str],
    box_padding_px: int,
    manual_boxes: Sequence[Mapping[str, Any]] = (),
    preview: np.ndarray | None = None,
) -> dict[str, Any]:
    """Save one reviewed scene as YOLO images, labels, and depth."""
    if not assignments and not manual_boxes:
        raise ValueError("至少需要标注一个物体")

    height, width = frame.shape[:2]
    label_lines: list[str] = []
    saved_objects: list[dict[str, Any]] = []
    for raw_candidate_id, raw_class_id in assignments.items():
        candidate_id = int(raw_candidate_id)
        class_id = int(raw_class_id)
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"类别编号无效：{class_id}")
        if not 1 <= candidate_id <= len(candidates):
            raise ValueError(f"候选编号无效：{candidate_id}")

        candidate = candidates[candidate_id - 1]
        x, y, box_width, box_height = (
            float(value) for value in candidate["bbox_xywh"]
        )
        x1 = max(0.0, x - box_padding_px)
        y1 = max(0.0, y - box_padding_px)
        x2 = min(float(width), x + box_width + box_padding_px)
        y2 = min(float(height), y + box_height + box_padding_px)
        if x2 <= x1 or y2 <= y1:
            continue

        center_x = (x1 + x2) * 0.5 / width
        center_y = (y1 + y2) * 0.5 / height
        normalized_width = (x2 - x1) / width
        normalized_height = (y2 - y1) / height
        label_lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )
        saved_objects.append(
            {
                "candidate_id": candidate_id,
                "source": "depth_candidate",
                "class_id": class_id,
                "class_name": class_names[class_id],
                "bbox_xyxy": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
                "area_pixels": int(candidate["area_pixels"]),
                "height_median_m": candidate.get("height_median_m"),
                "height_p95_m": candidate.get("height_p95_m"),
            }
        )

    for index, manual_box in enumerate(manual_boxes, start=1):
        class_id = int(manual_box["class_id"])
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"手动框类别编号无效：{class_id}")
        raw_bbox = manual_box.get("bbox_xyxy")
        if not isinstance(raw_bbox, Sequence) or len(raw_bbox) != 4:
            raise ValueError("手动框必须包含 bbox_xyxy")
        x1 = max(0.0, min(float(raw_bbox[0]), float(width)))
        y1 = max(0.0, min(float(raw_bbox[1]), float(height)))
        x2 = max(0.0, min(float(raw_bbox[2]), float(width)))
        y2 = max(0.0, min(float(raw_bbox[3]), float(height)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"手动框 #{index} 无效")

        center_x = (x1 + x2) * 0.5 / width
        center_y = (y1 + y2) * 0.5 / height
        normalized_width = (x2 - x1) / width
        normalized_height = (y2 - y1) / height
        label_lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )
        saved_objects.append(
            {
                "source": "manual",
                "manual_index": index,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "bbox_xyxy": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
            }
        )

    if not label_lines:
        raise ValueError("没有可保存的有效标注框")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"scene_{stamp}_{uuid.uuid4().hex[:6]}"
    image_dir = scene_root / "images"
    label_dir = scene_root / "labels"
    depth_dir = scene_root / "depth"
    metadata_dir = scene_root / "metadata"
    for directory in (image_dir, label_dir, depth_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"{stem}.jpg"
    label_path = label_dir / f"{stem}.txt"
    depth_path = depth_dir / f"{stem}.png"
    overlay_path = metadata_dir / f"{stem}_overlay.jpg"
    if not cv2.imwrite(
        str(image_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise RuntimeError(f"无法写入图片：{image_path}")
    depth_units = np.clip(
        np.rint(depth_m / depth_scale_m_per_unit),
        0,
        65535,
    ).astype(np.uint16)
    if not cv2.imwrite(str(depth_path), depth_units):
        raise RuntimeError(f"无法写入深度图：{depth_path}")
    if preview is not None:
        reviewed_preview = preview.copy()
        for index, manual_box in enumerate(manual_boxes, start=1):
            x1, y1, x2, y2 = (
                int(round(float(value)))
                for value in manual_box["bbox_xyxy"]
            )
            cv2.rectangle(
                reviewed_preview,
                (x1, y1),
                (x2, y2),
                (240, 110, 220),
                2,
            )
            cv2.putText(
                reviewed_preview,
                f"M{index}",
                (x1 + 3, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 110, 220),
                1,
                cv2.LINE_AA,
            )
        if not cv2.imwrite(
            str(overlay_path),
            reviewed_preview,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise RuntimeError(f"无法写入候选预览图：{overlay_path}")
    label_path.write_text(
        "\n".join(label_lines) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "scene": stem,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": str(image_path),
        "label": str(label_path),
        "depth": str(depth_path),
        "depth_scale_m_per_unit": float(depth_scale_m_per_unit),
        "color_intrinsics": {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "ppx": float(intrinsics.ppx),
            "ppy": float(intrinsics.ppy),
        },
        "box_padding_px": int(box_padding_px),
        "objects": saved_objects,
    }
    metadata_path = metadata_dir / f"{stem}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (scene_root / "scenes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    return {
        "scene": stem,
        "image": str(image_path),
        "label": str(label_path),
        "depth": str(depth_path),
        "overlay": str(overlay_path),
        "objects": len(saved_objects),
    }
