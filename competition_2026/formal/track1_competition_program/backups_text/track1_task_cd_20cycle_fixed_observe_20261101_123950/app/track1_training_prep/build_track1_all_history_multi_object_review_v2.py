#!/usr/bin/env python3
"""Build the all-history Track-1 multi-object human-review package.

Scope:
  * discover every session below the self-collected photo root;
  * select all candidate assets using capture hints, scene hints, prior
    human-reviewed multi-box records, model multi-class hints, or explicit
    per-frame multi-item notes;
  * expand candidate capture groups, merge overlapping groups through shared
    assets, and preserve each group in exactly one train/val split;
  * copy complete RGB-D four-piece assets and keep incomplete frames
    (especially missing-RGB frames) in a separate human-review area.

This program is read-only with respect to all capture sessions.  It never
trains a model, changes a model, opens a camera, imports ROS, or controls a
robot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
DEFAULT_CAPTURE_ROOT = Path("/media/sunshine/新加卷/赛道一_C_D自采照片")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets/track1_all_history_multi_object_review_20261031_v2"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "datasets/track1_selfcollected_v1/manifests/class_map.yaml"
DEFAULT_OLD_REVIEWS = (
    PROJECT_ROOT / "datasets/track1_postclasses_bbox_review_v1",
    PROJECT_ROOT / "datasets/track1_front10_bbox_review_v1",
)

ASSET_KINDS = ("rgb", "depth", "metadata", "overlay")
OUTPUT_DIRS = {
    "rgb": "rgb",
    "depth": "depth",
    "metadata": "meta",
    "overlay": "overlays",
}
GENERIC_NOTE = "多物品混放"
CLASS_99 = 99
REASON_LABELS = {
    "class_99": "采集类别 99-Multi_Object",
    "clutter_scene": "场景 box_11l_clutter",
    "model_multi_class": "模型辅助检测至少 2 个官方类别",
    "old_manual_multi": "旧人工复核至少 2 个 bbox",
    "specific_note": "逐帧明确多物品备注",
}


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def natural_key(value: str | int) -> tuple[int, int | str, str]:
    text = str(value)
    if text.isdigit():
        return (0, int(text), text)
    # Keep non-numeric names deterministic and after numeric names.
    return (1, text, text)


def path_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def csv_text(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    atomic_text(path, csv_text(rows, fields))


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点不是对象：{path}")
    return payload


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", value.strip())
    return value.strip("._") or "session"


def note_is_specific(note: str) -> bool:
    """Accept only notes that explicitly enumerate items, not generic hints."""
    value = " ".join(str(note or "").split())
    if not value or value == GENERIC_NOTE:
        return False
    tail = value.split("：", 1)[-1].split(":", 1)[-1]
    return any(separator in tail for separator in ("、", ",", "，", ";", "；"))


def discover_sessions(root: Path) -> list[Path]:
    if (root / "frames.jsonl").is_file():
        return [root.resolve()]
    sessions: list[Path] = []
    if not root.is_dir():
        raise ValueError(f"采集根目录不存在：{root}")
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.is_dir() and (child / "frames.jsonl").is_file():
            sessions.append(child.resolve())
    return sessions


def parse_sessions(sessions: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    duplicate_lines: list[dict[str, Any]] = []
    for session in sessions:
        session_name = sanitize_name(session.name)
        line_counts: Counter[str] = Counter()
        with (session / "frames.jsonl").open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                line_counts[text] += 1
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as error:
                    errors.append(
                        {
                            "session": session_name,
                            "line": line_number,
                            "error": f"JSON解析失败：{error}",
                        }
                    )
                    continue
                if not isinstance(payload, dict):
                    errors.append(
                        {
                            "session": session_name,
                            "line": line_number,
                            "error": "JSONL 行根节点不是对象",
                        }
                    )
                    continue
                files = payload.get("files")
                if not isinstance(files, dict):
                    errors.append(
                        {
                            "session": session_name,
                            "line": line_number,
                            "error": "缺少 files 对象",
                        }
                    )
                    continue
                frame_id = str(payload.get("frame_id") or "")
                metadata_value = str(files.get("metadata") or "")
                if not frame_id or not metadata_value:
                    errors.append(
                        {
                            "session": session_name,
                            "line": line_number,
                            "error": "缺少 frame_id 或 files.metadata",
                        }
                    )
                    continue
                defaults = {
                    "rgb": f"rgb/{frame_id}_{payload.get('class_id', '')}_{payload.get('canonical_label', '')}.png",
                    "depth": f"depth/{frame_id}_{payload.get('class_id', '')}_{payload.get('canonical_label', '')}.depth.png",
                    "metadata": f"meta/{frame_id}_{payload.get('class_id', '')}_{payload.get('canonical_label', '')}.json",
                    "overlay": f"overlays/{frame_id}_{payload.get('class_id', '')}_{payload.get('canonical_label', '')}.jpg",
                }
                parsed_paths: dict[str, Path] = {}
                bad_path = False
                for kind in ASSET_KINDS:
                    raw_value = str(files.get(kind) or defaults[kind])
                    candidate = (session / raw_value).resolve()
                    if not path_within(session, candidate):
                        errors.append(
                            {
                                "session": session_name,
                                "line": line_number,
                                "error": f"{kind} 路径越界：{raw_value}",
                            }
                        )
                        bad_path = True
                        break
                    parsed_paths[kind] = candidate
                if bad_path:
                    continue
                row = dict(payload)
                row["_session"] = session_name
                row["_session_root"] = session
                row["_line_number"] = line_number
                row["_paths"] = parsed_paths
                row["_asset_key"] = str(parsed_paths["metadata"])
                rows.append(row)
        for text, count in line_counts.items():
            if count > 1:
                try:
                    frame_id = str(json.loads(text).get("frame_id") or "")
                except json.JSONDecodeError:
                    frame_id = ""
                duplicate_lines.append(
                    {
                        "session": session_name,
                        "frame_id": frame_id,
                        "duplicate_count": count,
                        "line_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                )
    return rows, errors, duplicate_lines


def raw_group_key(row: Mapping[str, Any]) -> str:
    class_id = int(row.get("class_id") or -1)
    capture_mode = str(row.get("capture_mode") or "")
    capture_group = str(row.get("capture_group") or row.get("frame_id") or "")
    return f"{row.get('_session') or ''}|{class_id}|{capture_mode}:{capture_group}"


def group_short(value: str) -> str:
    return value[-30:]


def old_review_records(
    roots: Sequence[Path],
    *,
    allowed_class_ids: set[int],
) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[dict[str, Any]]]:
    by_basename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for root in roots:
        index_path = root / "index.json"
        reviews_dir = root / "reviews"
        if not index_path.is_file():
            warnings.append(f"旧复核索引不存在：{index_path}")
            continue
        try:
            index = read_json(index_path)
            items = {
                str(item.get("review_id")): item
                for item in index.get("items", [])
                if isinstance(item, dict) and item.get("review_id")
            }
        except Exception as error:  # noqa: BLE001 - source is reported as audit warning
            warnings.append(f"旧复核索引读取失败：{index_path}: {error}")
            continue
        if not reviews_dir.is_dir():
            warnings.append(f"旧复核 reviews 目录不存在：{reviews_dir}")
            continue
        for review_path in sorted(reviews_dir.glob("*.json")):
            review_id = review_path.stem
            item = items.get(review_id)
            if item is None:
                warnings.append(f"旧复核文件没有对应 index item：{review_path}")
                continue
            try:
                review = read_json(review_path)
            except Exception as error:  # noqa: BLE001
                warnings.append(f"旧复核文件读取失败：{review_path}: {error}")
                continue
            boxes = review.get("boxes")
            if not isinstance(boxes, list) or len(boxes) < 2:
                continue
            source_rgb = Path(str(item.get("source_rgb") or "")).name
            if not source_rgb:
                continue
            class_ids = sorted(
                {
                    int(box.get("class_id"))
                    for box in boxes
                    if isinstance(box, dict)
                    and box.get("class_id") is not None
                    and int(box.get("class_id")) in allowed_class_ids
                }
            )
            record = {
                "source": root.name,
                "review_id": review_id,
                "source_rgb": source_rgb,
                "split": str(item.get("split") or ""),
                "box_count": len(boxes),
                "class_ids": class_ids,
                "class_count": len(class_ids),
                "status": str(review.get("status") or ""),
            }
            rows.append(record)
            by_basename[source_rgb].append(record)
    return by_basename, warnings, rows


def load_class_info(path: Path) -> dict[int, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = payload.get("classes") if isinstance(payload, dict) else None
    if not isinstance(classes, dict):
        raise ValueError(f"类别表格式错误：{path}")
    result: dict[int, dict[str, Any]] = {}
    for key, raw in classes.items():
        if not isinstance(raw, dict):
            continue
        class_id = int(raw.get("class_id", key))
        result[class_id] = {
            "class_id": class_id,
            "canonical_label": str(raw.get("canonical_label") or f"{class_id}-unknown"),
            "english_name": str(raw.get("english_name") or ""),
            "display_name": str(raw.get("display_name") or ""),
        }
    return result


def detection_class_ids(row: Mapping[str, Any], allowed_class_ids: set[int]) -> set[int]:
    prediction = row.get("model_prediction") or {}
    detections = prediction.get("detections") or []
    result: set[int] = set()
    if not isinstance(detections, list):
        return result
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        raw = detection.get("catalog_class_id", detection.get("class_id"))
        try:
            class_id = int(raw)
        except (TypeError, ValueError):
            continue
        if class_id in allowed_class_ids:
            result.add(class_id)
    return result


def candidate_reasons_for_row(
    row: Mapping[str, Any],
    *,
    old_by_basename: Mapping[str, list[dict[str, Any]]],
    allowed_class_ids: set[int],
) -> set[str]:
    reasons: set[str] = set()
    try:
        class_id = int(row.get("class_id") or -1)
    except (TypeError, ValueError):
        class_id = -1
    if class_id == CLASS_99:
        reasons.add("class_99")
    if str(row.get("scene_type") or "") == "box_11l_clutter":
        reasons.add("clutter_scene")
    if len(detection_class_ids(row, allowed_class_ids)) >= 2:
        reasons.add("model_multi_class")
    rgb_basename = Path(str((row.get("_paths") or {}).get("rgb") or "")).name
    if rgb_basename and rgb_basename in old_by_basename:
        reasons.add("old_manual_multi")
    if note_is_specific(str(row.get("notes") or "")):
        reasons.add("specific_note")
    return reasons


def asset_completeness(asset: Mapping[str, Any]) -> tuple[bool, list[str], dict[int, int]]:
    paths = asset["paths"]
    missing: list[str] = []
    for kind in ASSET_KINDS:
        path = Path(paths[kind])
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(kind)
    width = 0
    height = 0
    try:
        metadata = read_json(Path(paths["metadata"]))
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        if width <= 0 or height <= 0:
            missing.append("metadata_dimensions")
    except Exception:
        missing.append("metadata_parse")
        metadata = {}
    return len(missing) == 0, sorted(set(missing)), {
        "width": width,
        "height": height,
        **({"metadata": metadata} if isinstance(metadata, dict) else {}),
    }


def image_sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return -math.inf
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    brightness = float(image.mean())
    penalty = 0.0
    if brightness < 35:
        penalty += (35.0 - brightness) * 5.0
    elif brightness > 225:
        penalty += (brightness - 225.0) * 5.0
    return sharpness - penalty


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size:
        raise OSError(f"复制后大小校验失败：{source} -> {destination}")


def draw_label(image: np.ndarray, text: str, height: int = 34) -> np.ndarray:
    label = np.full((height, image.shape[1], 3), (20, 28, 35), dtype=np.uint8)
    cv2.putText(
        label,
        text[:86],
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (235, 242, 246),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((image, label))


def make_contact_sheets(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    title: str,
    prefix: str,
    thumbnail: tuple[int, int] = (300, 225),
    columns: int = 6,
    rows_per_sheet: int = 4,
) -> list[str]:
    width, height = thumbnail
    label_height = 34
    header_height = 42
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for start in range(0, len(rows), columns * rows_per_sheet):
        chunk = list(rows[start : start + columns * rows_per_sheet])
        cells: list[np.ndarray] = []
        for offset, row in enumerate(chunk):
            image_path = Path(str(row.get("contact_image") or ""))
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                image = np.full((height, width, 3), (55, 45, 45), dtype=np.uint8)
                cv2.putText(
                    image,
                    "UNREADABLE",
                    (18, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (220, 220, 220),
                    2,
                )
            else:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            label = f"{start + offset + 1:04d} | {row.get('frame_id', '')} | {row.get('group_short', '')}"
            if row.get("incomplete"):
                label += " | NO-RGB"
            cells.append(draw_label(image, label, label_height))
        blank = np.full((height + label_height, width, 3), (12, 18, 24), dtype=np.uint8)
        while len(cells) < columns * rows_per_sheet:
            cells.append(blank.copy())
        grid_rows = [
            np.hstack(cells[index : index + columns])
            for index in range(0, len(cells), columns)
        ]
        grid = np.vstack(grid_rows)
        canvas = np.full(
            (header_height + grid.shape[0], grid.shape[1], 3),
            (12, 18, 24),
            dtype=np.uint8,
        )
        canvas[header_height:, :, :] = grid
        cv2.putText(
            canvas,
            f"{title} | sheet {len(generated) + 1} | items {start + 1}-{start + len(chunk)}/{len(rows)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 245, 248),
            1,
            cv2.LINE_AA,
        )
        filename = f"contact_{len(generated) + 1:03d}.jpg"
        target = output_dir / filename
        if not cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"无法写入缩略图：{target}")
        generated.append(f"{prefix}/{filename}")
    return generated


def gallery(title: str, description: str, cards: Sequence[str]) -> str:
    if not cards:
        return ""
    content = "\n".join(
        f'<a class="sheet" href="{html.escape(name)}"><img loading="lazy" src="{html.escape(name)}" alt="{html.escape(name)}"><span>{html.escape(name)}</span></a>'
        for name in cards
    )
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<p>{html.escape(description)}</p><div class=\"grid\">{content}</div></section>"
    )


def write_overview_html(
    path: Path,
    *,
    audit: Mapping[str, Any],
    burst_cards: Sequence[str],
    all_cards: Sequence[str],
    incomplete_cards: Sequence[str],
) -> None:
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>赛道一全历史多物品复核（当前照片）</title>
<style>
:root{{color-scheme:dark;--bg:#0b1117;--panel:#121c25;--line:#2a3b49;--text:#eef4f7;--muted:#9eb0bd;--accent:#48c7a2}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 6px;font-size:27px}} h2{{margin:0 0 5px;font-size:19px}} p{{color:var(--muted);margin:0 0 14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:18px 0 24px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}} .card strong{{display:block;font-size:26px;color:var(--accent)}}
section{{margin:28px 0}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}} .sheet{{display:block;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;color:var(--text);text-decoration:none}} .sheet img{{width:100%;display:block}} .sheet span{{display:block;padding:8px 10px;color:var(--muted)}}
.links a{{color:#7fe6c4;margin-right:18px}} code{{background:#17232d;padding:2px 5px;border-radius:4px;color:#d7f9ef;word-break:break-all}}
</style></head><body><main>
<h1>赛道一 · 全历史多物品照片复核（当前照片）</h1>
<p>来源为全部历史自采会话；采集类别只是提示，不是唯一标签。原图未移动、未改名、未删除。模型结果只作辅助提示，绝不直接当作训练标签。</p>
<div class="cards">
<div class="card"><strong>{audit['candidate_asset_count']}</strong>候选唯一素材</div>
<div class="card"><strong>{audit['final_unit_count']}</strong>合并后复核组</div>
<div class="card"><strong>{audit['complete_asset_count']}</strong>完整四件套</div>
<div class="card"><strong>{audit['incomplete_asset_count']}</strong>不完整素材（含缺RGB）</div>
<div class="card"><strong>{audit['incomplete_rgb_count']}</strong>缺 RGB 帧</div>
<div class="card"><strong>{audit['source_record_count']}</strong>源会话记录</div>
</div>
<div class="links"><a href="{Path(audit["output_root"]).joinpath("review_photos").as_uri()}">打开全部候选 RGB 目录</a><a href="{Path(audit["output_root"]).joinpath("manifests").as_uri()}">打开清单目录</a><a href="http://127.0.0.1:9131/" target="_blank">打开逐框复核网页（需先启动）</a></div>
{gallery('按连拍组去重后的代表帧', '先看这一组；每组通常 3 帧，共享素材造成的重叠已合并，不会把同一张图重复算成两组。', burst_cards)}
{gallery('全部完整候选帧', '全部完整 RGB-D 四件套都列出，便于核对同组相似帧、遮挡、透明箱反光和模型漏检。', all_cards)}
{gallery('缺 RGB / 不完整帧', '这些帧不能当作完整四件套；展示已有 overlay，原始 depth/meta 也保存在 incomplete/ 中，供人工复核。', incomplete_cards)}
</main></body></html>
"""
    atomic_text(path, page)


def build_audit_readme(audit: Mapping[str, Any]) -> str:
    return f"""# 赛道一全历史多物品复核（当前照片）

## 结论

- 自采会话：{', '.join(audit['sessions']) or '无'}
- 源会话记录：{audit['source_record_count']} 条
- 候选记录：{audit['candidate_record_count']} 条
- 候选唯一素材：{audit['candidate_asset_count']} 个
- 原始候选连拍组：{audit['candidate_raw_group_count']} 个
- 合并共享素材后的复核组：{audit['final_unit_count']} 个
- 展开后唯一素材：{audit['expanded_asset_count']} 个
- 完整四件套：{audit['complete_asset_count']} 个
- 不完整素材：{audit['incomplete_asset_count']} 个
- 缺 RGB：{audit['incomplete_rgb_count']} 个
- 共享素材：{audit['shared_asset_count']} 个
- 最终索引可复核组：{audit['review_item_count']} 个（没有可用 RGB 代表帧的组只进入 incomplete 清单）

## 筛选口径

候选按唯一 metadata 素材取并集，命中以下任一条即纳入：

1. `class_id == 99`
2. `scene_type == box_11l_clutter`
3. 模型辅助检测到至少 2 个官方类别
4. 旧人工复核记录至少 2 个 bbox
5. 逐帧备注明确列出多个物品

所有模型结果均为辅助提示，`selected_label_confirmed_by_model` 不是人工真值；本包没有训练，也没有修改任何模型。

## 复核规则

- 一张图允许多个官方类别。
- 同类有多个实例时，每个实例分别画框。
- 99 只作为采集提示，不会出现在逐框类别下拉框，也不会成为 YOLO 第 41 类。
- 官方 17、26 类可选，不按“空槽”隐藏。
- 空背景保存为空 `.txt`，不创建空背景类别。
- 看不清或无法确认的物品不要乱标，可在逐框复核页备注，交给人工复核。

## 数据位置

- 完整四件套源：`{audit['output_root']}`
- 全部候选 RGB：`{audit['output_root']}/review_photos`
- 复核组代表缩略图：`{audit['output_root']}/contact_sheets/index.html`
- 完整帧清单：`{audit['output_root']}/manifests/split_manifest.csv`
- 缺 RGB 清单：`{audit['output_root']}/manifests/missing_rgb_candidates.csv`
- 审计 JSON：`{audit['output_root']}/audit/audit.json`

## 原始数据保护

程序对源会话只读，复制到新目录；没有移动、删除、改名或覆盖原图。失败或缺失文件都写入清单，不静默丢弃。
"""


def build_index_item(
    unit: Mapping[str, Any],
    *,
    staging: Path,
    class_info: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any] | None:
    complete_assets = list(unit["complete_assets"])
    if not complete_assets:
        return None
    ordered = sorted(complete_assets, key=lambda asset: natural_key(asset["frame_id"]))
    middle = len(ordered) // 2
    representative = max(
        ordered,
        key=lambda asset: (
            image_sharpness(staging / asset["output_paths"]["rgb"]),
            -abs(ordered.index(asset) - middle),
        ),
    )
    hint_id = int(unit["hint_class_id"])
    if hint_id == CLASS_99:
        hint_label = "99-Multi_Object"
        hint_display = "多物品混放"
    else:
        info = class_info.get(hint_id, {})
        hint_label = str(info.get("canonical_label") or f"{hint_id}-unknown")
        hint_display = str(info.get("display_name") or hint_label)
    metadata = representative["metadata"]
    return {
        "review_id": str(unit["unit_id"]),
        "split_unit": str(unit["unit_id"]),
        "split": str(unit["split"]),
        "hint_class_id": hint_id,
        "hint_class_label": hint_label,
        "hint_display_name": hint_display,
        "selected_frame_id": str(representative["frame_id"]),
        "source_meta": str(representative["output_paths"]["metadata"]),
        "source_rgb": str(representative["output_paths"]["rgb"]),
        "source_depth": str(representative["output_paths"]["depth"]),
        "width": int(metadata.get("width") or 0),
        "height": int(metadata.get("height") or 0),
        "frame_count": len(complete_assets),
        "frame_ids": [str(asset["frame_id"]) for asset in ordered],
        "complete_frame_count": len(complete_assets),
        "incomplete_frame_ids": [str(asset["frame_id"]) for asset in sorted(unit["incomplete_assets"], key=lambda item: natural_key(item["frame_id"]))],
        "candidate_reasons": list(unit["candidate_reasons"]),
        "priority_reasons": list(unit["priority_reasons"]),
        "raw_group_keys": list(unit["raw_group_keys"]),
        "historical_splits": list(unit["historical_splits"]),
    }


def candidate_payload_for_asset(asset: Mapping[str, Any], review_id: str) -> dict[str, Any]:
    metadata = asset.get("metadata") or {}
    prediction = metadata.get("model_prediction") or {}
    detections = prediction.get("detections") or []
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    if not isinstance(detections, list):
        detections = []
    for index, detection in enumerate(detections, start=1):
        if not isinstance(detection, dict):
            continue
        try:
            class_id = int(detection.get("catalog_class_id", detection.get("class_id")))
            confidence = float(detection.get("confidence") or 0.0)
            bbox = [float(value) for value in detection.get("bbox_xyxy", [])]
        except (TypeError, ValueError):
            continue
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            continue
        if class_id < 1 or class_id > 40:
            continue
        x1, y1, x2, y2 = bbox
        if width > 0 and height > 0:
            x1 = min(max(x1, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            x2 = min(max(x2, 0.0), float(width))
            y2 = min(max(y2, 0.0), float(height))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue
        signature = (class_id, round(x1), round(y1), round(x2), round(y2))
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(
            {
                "candidate_id": f"M{len(candidates) + 1:02d}",
                "source": "model_hint",
                "class_id": class_id,
                "class_label": str(detection.get("catalog_label") or ""),
                "prompt": str(detection.get("model_class") or ""),
                "confidence": confidence,
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )
    return {
        "schema_version": 1,
        "review_id": review_id,
        "source_frame_id": asset["frame_id"],
        "capture_hint_class_id": int(asset.get("class_id") or -1),
        "important_note": (
            "模型候选只作辅助提示，绝不是训练标签；必须人工确认并补齐同图其他类别和实例。"
        ),
        "model_mode": str(prediction.get("mode") or ""),
        "selected_label_confirmed_by_model": prediction.get("selected_label_confirmed_by_model"),
        "candidates": candidates,
    }


def write_audit_files(staging: Path, audit: Mapping[str, Any]) -> None:
    atomic_json(staging / "audit/audit.json", audit)
    atomic_text(staging / "audit/audit.md", build_audit_readme(audit))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成赛道一全历史多物品复核 v2 包")
    parser.add_argument("--capture-root", default=str(DEFAULT_CAPTURE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP))
    parser.add_argument("--old-review-root", action="append", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--limit-units", type=int, default=0, help="仅用于小样测试")
    args = parser.parse_args(argv)

    capture_root = Path(args.capture_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    class_map = Path(args.class_map).expanduser().resolve()
    old_roots = tuple(
        Path(value).expanduser().resolve()
        for value in (args.old_review_root or list(DEFAULT_OLD_REVIEWS))
    )
    if not class_map.is_file():
        raise ValueError(f"类别表不存在：{class_map}")
    class_info = load_class_info(class_map)
    allowed_class_ids = set(class_info)
    sessions = discover_sessions(capture_root)
    if not sessions:
        raise ValueError(f"没有发现包含 frames.jsonl 的会话：{capture_root}")

    raw_rows, parse_errors, duplicate_lines = parse_sessions(sessions)
    if not raw_rows:
        raise ValueError("没有可解析的 frames.jsonl 记录")

    asset_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        asset_rows[row["_asset_key"]].append(row)

    old_by_basename, old_warnings, old_rows = old_review_records(
        old_roots,
        allowed_class_ids=allowed_class_ids,
    )

    assets: dict[str, dict[str, Any]] = {}
    for key, rows in asset_rows.items():
        canonical = rows[0]
        reasons: set[str] = set()
        for row in rows:
            reasons.update(
                candidate_reasons_for_row(
                    row,
                    old_by_basename=old_by_basename,
                    allowed_class_ids=allowed_class_ids,
                )
            )
        paths = dict(canonical["_paths"])
        complete, missing, metadata_details = asset_completeness(
            {"paths": paths, "metadata": {}}
        )
        metadata = metadata_details.pop("metadata", {}) if isinstance(metadata_details, dict) else {}
        assets[key] = {
            "asset_key": key,
            "session": canonical["_session"],
            "session_root": canonical["_session_root"],
            "frame_id": str(canonical.get("frame_id") or ""),
            "class_id": int(canonical.get("class_id") or -1),
            "capture_mode": str(canonical.get("capture_mode") or ""),
            "capture_group": str(canonical.get("capture_group") or ""),
            "paths": paths,
            "rows": rows,
            "reasons": reasons,
            "complete": complete,
            "missing_kinds": missing,
            "metadata": metadata,
            "width": int(metadata_details.get("width") or 0),
            "height": int(metadata_details.get("height") or 0),
        }

    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        group_rows[raw_group_key(row)].append(row)
    group_assets: dict[str, set[str]] = {
        key: {row["_asset_key"] for row in rows}
        for key, rows in group_rows.items()
    }
    candidate_assets = {key for key, asset in assets.items() if asset["reasons"]}
    candidate_groups = {
        group
        for group, keys in group_assets.items()
        if keys & candidate_assets
    }

    # Merge groups that share a physical asset.  The asset is copied once and
    # the merged review unit gets both original group keys in its audit trail.
    parent: dict[str, str] = {group: group for group in candidate_groups}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(first: str, second: str) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[b] = a

    asset_to_groups: dict[str, set[str]] = defaultdict(set)
    for group in candidate_groups:
        for asset_key in group_assets[group]:
            asset_to_groups[asset_key].add(group)
    for groups in asset_to_groups.values():
        candidate_members = sorted(groups & candidate_groups)
        for group in candidate_members[1:]:
            union(candidate_members[0], group)
    components: dict[str, set[str]] = defaultdict(set)
    for group in candidate_groups:
        components[find(group)].add(group)

    component_rows: list[dict[str, Any]] = []
    for groups in components.values():
        members: set[str] = set()
        for group in groups:
            members.update(group_assets[group])
        component_rows.append(
            {
                "groups": sorted(groups),
                "member_assets": sorted(members),
                "sort_key": min(
                    (
                        natural_key(assets[key]["frame_id"]),
                        assets[key]["session"],
                        key,
                    )
                    for key in members
                ),
            }
        )
    component_rows.sort(key=lambda item: item["sort_key"])
    if args.limit_units and args.limit_units > 0:
        component_rows = component_rows[: args.limit_units]

    units: list[dict[str, Any]] = []
    for unit_index, component in enumerate(component_rows, start=1):
        member_assets = [assets[key] for key in component["member_assets"]]
        complete_assets = [asset for asset in member_assets if asset["complete"]]
        incomplete_assets = [asset for asset in member_assets if not asset["complete"]]
        capture_class_ids = {
            int(asset["class_id"])
            for asset in member_assets
            if int(asset["class_id"]) in {CLASS_99, *allowed_class_ids}
        }
        if CLASS_99 in capture_class_ids:
            hint_class_id = CLASS_99
        else:
            hint_class_id = min(capture_class_ids) if capture_class_ids else CLASS_99
        reasons = sorted({reason for asset in member_assets for reason in asset["reasons"]})

        historical_splits: set[str] = set()
        for asset in member_assets:
            rgb_name = Path(asset["paths"]["rgb"]).name
            for record in old_by_basename.get(rgb_name, []):
                split = str(record.get("split") or "")
                if split in {"train", "val"}:
                    historical_splits.add(split)
        if len(historical_splits) == 1:
            split = next(iter(historical_splits))
        elif historical_splits:
            # Extremely defensive policy: never let a group cross train/val.
            split = "val" if "val" in historical_splits else "train"
        else:
            stable = hashlib.sha256(
                "|".join(component["groups"]).encode("utf-8")
            ).digest()
            split = "val" if int.from_bytes(stable[:8], "big") % 5 == 0 else "train"

        priority_reasons: set[str] = set()
        for asset in member_assets:
            metadata = asset.get("metadata") or {}
            detections = (metadata.get("model_prediction") or {}).get("detections") or []
            if not detections:
                priority_reasons.add("no_model_detection")
            elif len(detection_class_ids(metadata, allowed_class_ids)) >= 2:
                priority_reasons.add("model_multi_class")
            if asset["reasons"]:
                priority_reasons.update(asset["reasons"])
        unit = {
            "unit_id": f"all-history-multi-{unit_index:04d}",
            "raw_group_keys": component["groups"],
            "member_assets": member_assets,
            "complete_assets": complete_assets,
            "incomplete_assets": incomplete_assets,
            "candidate_reasons": reasons,
            "priority_reasons": sorted(priority_reasons),
            "hint_class_id": hint_class_id,
            "split": split,
            "historical_splits": sorted(historical_splits),
        }
        units.append(unit)

    # Count physical JSONL rows, not unique metadata paths, so duplicate lines
    # remain visible in the audit trail.
    candidate_record_count = sum(
        1 for row in raw_rows if row["_asset_key"] in candidate_assets
    )
    expanded_asset_keys = {asset["asset_key"] for unit in units for asset in unit["member_assets"]}
    shared_assets = [
        {
            "asset_key": key,
            "frame_id": assets[key]["frame_id"],
            "class_id": assets[key]["class_id"],
            "group_keys": sorted(asset_to_groups.get(key, set())),
        }
        for key in sorted(expanded_asset_keys)
        if len(asset_to_groups.get(key, set())) > 1
    ]

    audit_before_copy = {
        "schema_version": 2,
        "created_at": iso_now(),
        "mode": "audit_only" if args.audit_only else "materialize",
        "capture_root": str(capture_root),
        "output_root": str(output_root),
        "class_map": str(class_map),
        "sessions": [session.name for session in sessions],
        "source_record_count": len(raw_rows),
        "source_unique_asset_count": len(assets),
        "source_parse_error_count": len(parse_errors),
        "duplicate_jsonl_line_count": len(duplicate_lines),
        "candidate_record_count": candidate_record_count,
        "candidate_asset_count": len(candidate_assets),
        "candidate_raw_group_count": len(candidate_groups),
        "final_unit_count": len(units),
        "expanded_asset_count": len(expanded_asset_keys),
        "complete_asset_count": sum(1 for key in expanded_asset_keys if assets[key]["complete"]),
        "incomplete_asset_count": sum(1 for key in expanded_asset_keys if not assets[key]["complete"]),
        "incomplete_rgb_count": sum(1 for key in expanded_asset_keys if not (assets[key]["paths"]["rgb"]).is_file()),
        "shared_asset_count": len(shared_assets),
        "review_item_count": sum(1 for unit in units if unit["complete_assets"]),
        "historical_multi_review_record_count": len(old_rows),
        "historical_multi_asset_count": len(
            {record["source_rgb"] for record in old_rows}
        ),
        "old_review_warnings": old_warnings,
        "old_review_sources": [str(root) for root in old_roots],
        "candidate_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for asset in assets.values()
                    for reason in asset["reasons"]
                ).items()
            )
        ),
        "candidate_assets_missing_rgb": sorted(
            [
                {
                    "frame_id": assets[key]["frame_id"],
                    "class_id": assets[key]["class_id"],
                    "session": assets[key]["session"],
                    "metadata": str(assets[key]["paths"]["metadata"]),
                    "overlay": str(assets[key]["paths"]["overlay"]),
                }
                for key in expanded_asset_keys
                if not assets[key]["paths"]["rgb"].is_file()
            ],
            key=lambda item: natural_key(item["frame_id"]),
        ),
        "shared_assets": shared_assets,
        "parse_errors": parse_errors,
        "duplicate_jsonl_lines": duplicate_lines,
        "source_read_only": True,
        "training_started": False,
        "model_modified": False,
    }

    if args.audit_only:
        print(json.dumps(audit_before_copy, ensure_ascii=False, indent=2))
        return 0

    if output_root.exists():
        raise ValueError(
            f"输出目录已存在，不覆盖：{output_root}；如需重做请换新的 v2 目录"
        )
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        raise ValueError(f"临时输出目录已存在，不删除：{staging}")
    staging.mkdir(parents=True)
    for directory in ("rgb", "depth", "meta", "overlays", "review_photos", "incomplete", "manifests", "candidates", "contact_sheets", "audit"):
        (staging / directory).mkdir(parents=True, exist_ok=True)

    # Assign output-relative paths once, detect collisions, and copy every
    # candidate asset exactly once.  Incomplete assets are kept under a
    # separate tree so their missing RGB cannot be mistaken for a complete
    # four-piece training sample.
    used_destinations: set[str] = set()
    for asset in assets.values():
        if asset["asset_key"] not in expanded_asset_keys:
            continue
        session = sanitize_name(asset["session"])
        asset["output_paths"] = {}
        for kind in ASSET_KINDS:
            source = Path(asset["paths"][kind])
            base_name = source.name
            if asset["complete"]:
                rel = Path(OUTPUT_DIRS[kind]) / session / base_name
            else:
                rel = Path("incomplete") / session / OUTPUT_DIRS[kind] / base_name
            rel_text = str(rel)
            if rel_text in used_destinations:
                raise ValueError(f"输出路径冲突：{rel_text}")
            used_destinations.add(rel_text)
            asset["output_paths"][kind] = rel
            if source.is_file() and source.stat().st_size > 0:
                copy_file(source, staging / rel)
        if asset["complete"]:
            rel_review = Path("review_photos") / session / Path(asset["paths"]["rgb"]).name
            rel_review_text = str(rel_review)
            if rel_review_text in used_destinations:
                raise ValueError(f"复核 RGB 输出路径冲突：{rel_review_text}")
            used_destinations.add(rel_review_text)
            copy_file(Path(asset["paths"]["rgb"]), staging / rel_review)
            asset["review_photo"] = rel_review

    # Build manifests and index entries.  Only complete assets enter
    # split_manifest.csv; every incomplete asset stays separately visible.
    split_rows: list[dict[str, Any]] = []
    index_items: list[dict[str, Any]] = []
    group_manifest_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    candidate_asset_rows: list[dict[str, Any]] = []
    for unit in units:
        index_item = build_index_item(unit, staging=staging, class_info=class_info)
        if index_item is not None:
            index_items.append(index_item)
        complete_assets = list(unit["complete_assets"])
        incomplete_assets = list(unit["incomplete_assets"])
        representative = None
        if complete_assets:
            representative = min(
                complete_assets,
                key=lambda asset: (
                    asset["frame_id"],
                    asset["asset_key"],
                ),
            )
        elif incomplete_assets:
            representative = min(incomplete_assets, key=lambda asset: (asset["frame_id"], asset["asset_key"]))
        if representative is not None:
            contact = (
                staging / representative["output_paths"]["rgb"]
                if representative["complete"]
                else staging / representative["output_paths"]["overlay"]
            )
            group_manifest_rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "split": unit["split"],
                    "hint_class_id": unit["hint_class_id"],
                    "raw_group_keys": ";".join(unit["raw_group_keys"]),
                    "frame_ids": ",".join(sorted({str(asset["frame_id"]) for asset in unit["member_assets"]}, key=natural_key)),
                    "complete_frame_ids": ",".join(sorted({str(asset["frame_id"]) for asset in complete_assets}, key=natural_key)),
                    "incomplete_frame_ids": ",".join(sorted({str(asset["frame_id"]) for asset in incomplete_assets}, key=natural_key)),
                    "complete_frame_count": len(complete_assets),
                    "incomplete_frame_count": len(incomplete_assets),
                    "candidate_reasons": ";".join(unit["candidate_reasons"]),
                    "priority_reasons": ";".join(unit["priority_reasons"]),
                    "representative_frame_id": representative["frame_id"],
                    "contact_image": str(contact),
                    "group_short": group_short(unit["unit_id"]),
                }
            )
        for asset in complete_assets:
            reasons = ";".join(sorted(asset["reasons"]))
            split_rows.append(
                {
                    "source_meta": str(asset["output_paths"]["metadata"]),
                    "frame_id": str(asset["frame_id"]),
                    "class_id": str(unit["hint_class_id"]),
                    "source_capture_class_id": str(asset["class_id"]),
                    "capture_role": "candidate_complete",
                    "effective_capture_group": ";".join(unit["raw_group_keys"]),
                    "split_unit": unit["unit_id"],
                    "split": unit["split"],
                    "rgb": str(asset["output_paths"]["rgb"]),
                    "depth": str(asset["output_paths"]["depth"]),
                    "overlay": str(asset["output_paths"]["overlay"]),
                    "session": asset["session"],
                    "candidate_reasons": reasons,
                    "historical_splits": ";".join(unit["historical_splits"]),
                }
            )
        for asset in incomplete_assets:
            incomplete_rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "session": asset["session"],
                    "frame_id": asset["frame_id"],
                    "source_capture_class_id": asset["class_id"],
                    "hint_class_id": unit["hint_class_id"],
                    "split": unit["split"],
                    "missing_kinds": ";".join(asset["missing_kinds"]),
                    "rgb": str(asset["output_paths"]["rgb"]) if (staging / asset["output_paths"]["rgb"]).is_file() else "",
                    "depth": str(asset["output_paths"]["depth"]) if (staging / asset["output_paths"]["depth"]).is_file() else "",
                    "metadata": str(asset["output_paths"]["metadata"]) if (staging / asset["output_paths"]["metadata"]).is_file() else "",
                    "overlay": str(asset["output_paths"]["overlay"]) if (staging / asset["output_paths"]["overlay"]).is_file() else "",
                    "contact_image": str(staging / asset["output_paths"]["overlay"]) if (staging / asset["output_paths"]["overlay"]).is_file() else "",
                }
            )
        for asset in unit["member_assets"]:
            candidate_asset_rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "session": asset["session"],
                    "frame_id": asset["frame_id"],
                    "class_id": asset["class_id"],
                    "complete": str(asset["complete"]).lower(),
                    "missing_kinds": ";".join(asset["missing_kinds"]),
                    "candidate_reasons": ";".join(sorted(asset["reasons"])),
                    "rgb": str(asset["output_paths"]["rgb"]),
                    "depth": str(asset["output_paths"]["depth"]),
                    "metadata": str(asset["output_paths"]["metadata"]),
                    "overlay": str(asset["output_paths"]["overlay"]),
                    "review_photo": str(asset.get("review_photo") or ""),
                }
            )

    # Write the split manifest required by the existing reviewer.
    write_csv(
        staging / "manifests/split_manifest.csv",
        split_rows,
        [
            "source_meta", "frame_id", "class_id", "source_capture_class_id",
            "capture_role", "effective_capture_group", "split_unit", "split",
            "rgb", "depth", "overlay", "session", "candidate_reasons",
            "historical_splits",
        ],
    )
    write_csv(
        staging / "manifests/candidate_assets.csv",
        candidate_asset_rows,
        [
            "unit_id", "session", "frame_id", "class_id", "complete",
            "missing_kinds", "candidate_reasons", "rgb", "depth", "metadata",
            "overlay", "review_photo",
        ],
    )
    write_csv(
        staging / "manifests/review_groups.csv",
        group_manifest_rows,
        [
            "unit_id", "split", "hint_class_id", "raw_group_keys", "frame_ids",
            "complete_frame_ids", "incomplete_frame_ids", "complete_frame_count",
            "incomplete_frame_count", "candidate_reasons", "priority_reasons",
            "representative_frame_id", "contact_image", "group_short",
        ],
    )
    write_csv(
        staging / "manifests/incomplete_frames.csv",
        incomplete_rows,
        [
            "unit_id", "session", "frame_id", "source_capture_class_id",
            "hint_class_id", "split", "missing_kinds", "rgb", "depth",
            "metadata", "overlay", "contact_image",
        ],
    )
    write_csv(
        staging / "manifests/missing_rgb_candidates.csv",
        [row for row in incomplete_rows if "rgb" in row["missing_kinds"].split(";")],
        [
            "unit_id", "session", "frame_id", "source_capture_class_id",
            "hint_class_id", "split", "missing_kinds", "depth", "metadata",
            "overlay", "contact_image",
        ],
    )
    write_csv(
        staging / "manifests/old_manual_multi_frames.csv",
        old_rows,
        ["source", "review_id", "source_rgb", "split", "box_count", "class_ids", "class_count", "status"],
    )
    write_csv(
        staging / "manifests/shared_assets.csv",
        shared_assets,
        ["asset_key", "frame_id", "class_id", "group_keys"],
    )
    write_csv(
        staging / "manifests/duplicate_jsonl_lines.csv",
        duplicate_lines,
        ["session", "frame_id", "duplicate_count", "line_sha256"],
    )
    write_csv(
        staging / "manifests/parse_errors.csv",
        parse_errors,
        ["session", "line", "error"],
    )

    # Candidate cards for the review UI.  They are deliberately auxiliary.
    for unit in units:
        if not unit["complete_assets"]:
            continue
        representative = min(
            unit["complete_assets"],
            key=lambda asset: (asset["frame_id"], asset["asset_key"]),
        )
        payload = candidate_payload_for_asset(representative, unit["unit_id"])
        atomic_json(staging / "candidates" / f"{unit['unit_id']}.json", payload)

    # Contact sheets: complete groups, all complete assets, and incomplete
    # frames (overlay previews) are all available for human review.
    burst_rows = [
        {
            **row,
            "contact_image": row["contact_image"],
            "incomplete": row["incomplete_frame_count"] > 0,
        }
        for row in group_manifest_rows
    ]
    burst_cards = make_contact_sheets(
        burst_rows,
        staging / "contact_sheets/burst_representatives",
        title="BURST REPRESENTATIVES",
        prefix="burst_representatives",
    )
    all_complete_rows = []
    for asset in sorted(
        (asset for key, asset in assets.items() if key in expanded_asset_keys and asset["complete"]),
        key=lambda item: (natural_key(item["frame_id"]), item["session"], item["asset_key"]),
    ):
        all_complete_rows.append(
            {
                "frame_id": asset["frame_id"],
                "group_short": group_short(asset["session"]),
                "contact_image": str(staging / asset["output_paths"]["rgb"]),
            }
        )
    all_cards = make_contact_sheets(
        all_complete_rows,
        staging / "contact_sheets/all_complete",
        title="ALL COMPLETE CANDIDATE FRAMES",
        prefix="all_complete",
    )
    incomplete_cards = make_contact_sheets(
        [
            {
                **row,
                "group_short": group_short(row["unit_id"]),
                "incomplete": True,
            }
            for row in incomplete_rows
        ],
        staging / "contact_sheets/incomplete",
        title="INCOMPLETE / MISSING RGB",
        prefix="incomplete",
    )
    write_overview_html(
        staging / "contact_sheets/index.html",
        audit=audit_before_copy,
        burst_cards=burst_cards,
        all_cards=all_cards,
        incomplete_cards=incomplete_cards,
    )

    # The index is generated after copying so all paths are stable and relative
    # to the final review source root.
    index = {
        "schema_version": 1,
        "created_at": iso_now(),
        "source_root": str(output_root),
        "class_map": str(class_map),
        "review_root": str(output_root),
        "selection_policy": (
            "每个合并后的 capture group 选 1 张清晰度最高的完整 RGB 代表帧；"
            "同一组绝不跨 train/val。历史复核存在的组优先保留旧 split，新组使用稳定哈希。"
        ),
        "multi_object_policy": (
            "采集类别/文件名只是提示；一张图必须标出全部可辨认的官方 1-40 类物品，"
            "同类多实例分别标框。99 仅作采集提示，不进入逐框类别。"
        ),
        "item_count": len(index_items),
        "split_counts": dict(sorted(Counter(item["split"] for item in index_items).items())),
        "hint_class_counts": dict(
            sorted(
                Counter(str(item["hint_class_id"]) for item in index_items).items(),
                key=lambda item: int(item[0]),
            )
        ),
        "items": index_items,
    }
    atomic_json(staging / "index.json", index)

    audit = dict(audit_before_copy)
    audit.update(
        {
            "review_item_count": len(index_items),
            "split_manifest_row_count": len(split_rows),
            "incomplete_row_count": len(incomplete_rows),
            "review_groups_row_count": len(group_manifest_rows),
            "candidate_json_count": len(index_items),
            "contact_sheet_counts": {
                "burst_representatives": len(burst_cards),
                "all_complete": len(all_cards),
                "incomplete": len(incomplete_cards),
            },
            "review_photos_directory": str(output_root / "review_photos"),
            "contact_sheets_directory": str(output_root / "contact_sheets"),
            "class_ids_available_for_bbox": sorted(allowed_class_ids),
            "class_99_in_bbox_catalog": False,
            "unknown_or_uncertain_policy": (
                "无法确认物品的帧仍全部进入候选/不完整清单；模型候选不自动保存为标签。"
            ),
        }
    )
    write_audit_files(staging, audit)
    atomic_text(staging / "README.md", build_audit_readme(audit))

    staging.rename(output_root)
    print(
        json.dumps(
            {
                "ok": True,
                "output_root": str(output_root),
                "source_record_count": len(raw_rows),
                "candidate_asset_count": len(candidate_assets),
                "candidate_raw_group_count": len(candidate_groups),
                "final_unit_count": len(units),
                "review_item_count": len(index_items),
                "complete_asset_count": audit["complete_asset_count"],
                "incomplete_asset_count": audit["incomplete_asset_count"],
                "incomplete_rgb_count": audit["incomplete_rgb_count"],
                "review_photos": str(output_root / "review_photos"),
                "contact_sheets": str(output_root / "contact_sheets/index.html"),
                "audit": str(output_root / "audit/audit.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(2)
