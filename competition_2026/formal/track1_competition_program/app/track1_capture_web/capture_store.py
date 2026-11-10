#!/usr/bin/env python3
"""RGB-D capture storage for the Track 1 self-collected dataset.

This module only writes files below the configured photos root. It does not
import ROS, connect to the robot, or issue gripper/motion commands.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml


class CaptureError(RuntimeError):
    """Raised when a capture request or filesystem operation is invalid."""


@dataclass(frozen=True)
class CatalogEntry:
    class_id: int
    canonical_label: str
    english_name: str
    display_name: str
    subset: int | None
    model_class_names: tuple[str, ...]
    is_background: bool = False
    is_collection_hint: bool = False
    capture_hint_kind: str | None = None

    @property
    def label_index(self) -> int | None:
        if self.is_background or self.is_collection_hint:
            return None
        return self.class_id - 1

    def to_api(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "canonical_label": self.canonical_label,
            "english_name": self.english_name,
            "display_name": self.display_name,
            "subset": self.subset,
            "model_class_names": list(self.model_class_names),
            "is_background": self.is_background,
            "is_collection_hint": self.is_collection_hint,
            "capture_hint_kind": self.capture_hint_kind,
            "requires_per_object_bbox_review": bool(
                self.is_collection_hint
                and self.capture_hint_kind == "multi_object"
            ),
            "label_index": self.label_index,
        }


@dataclass(frozen=True)
class ClassCatalog:
    dataset_name: str
    entries: tuple[CatalogEntry, ...]
    background: CatalogEntry
    model_support: Mapping[str, Any]
    source: Mapping[str, Any]

    def by_id(self, class_id: int) -> CatalogEntry:
        if class_id == self.background.class_id:
            return self.background
        for entry in self.entries:
            if entry.class_id == class_id:
                return entry
        raise CaptureError(f"未知类别编号：{class_id}")

    def by_canonical_label(self, label: str) -> CatalogEntry:
        if label == self.background.canonical_label:
            return self.background
        for entry in self.entries:
            if entry.canonical_label == label:
                return entry
        raise CaptureError(f"未知类别标签：{label}")

    def model_name_to_entry(self) -> dict[str, CatalogEntry]:
        result: dict[str, CatalogEntry] = {}
        for entry in (self.background, *self.entries):
            for model_name in entry.model_class_names:
                if model_name in result:
                    raise CaptureError(f"模型类别名重复：{model_name}")
                result[model_name] = entry
        return result

    def to_api(self) -> dict[str, Any]:
        supported = {
            int(value)
            for value in self.model_support.get("supported_class_ids", [])
        }
        return {
            "dataset_name": self.dataset_name,
            "background": self.background.to_api(),
            "classes": [entry.to_api() for entry in self.entries],
            "model_support": {
                **dict(self.model_support),
                "supported_class_ids": sorted(supported),
            },
            "source": dict(self.source),
            "class_count": len(self.entries),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_api(),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureError(f"类别表字段 {field} 必须是非空字符串")
    return value.strip()


def load_class_catalog(path: str | Path) -> ClassCatalog:
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        raise CaptureError(f"类别表不存在：{catalog_path}")
    try:
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CaptureError(f"无法读取类别表：{catalog_path}：{error}") from error
    if not isinstance(payload, dict):
        raise CaptureError("类别表根节点必须是对象")

    dataset_name = _required_text(payload.get("dataset_name"), "dataset_name")
    raw_background = payload.get("background")
    raw_classes = payload.get("classes")
    if not isinstance(raw_background, dict) or not isinstance(raw_classes, dict):
        raise CaptureError("类别表必须包含 background 和 classes 对象")

    def build_entry(raw: Mapping[str, Any], *, background: bool) -> CatalogEntry:
        class_id = int(raw.get("class_id", 0 if background else -1))
        if background and class_id != 0:
            raise CaptureError("空背景 class_id 必须为 0")
        model_names = raw.get("model_class_names", [])
        if not isinstance(model_names, list):
            raise CaptureError(f"类别 {class_id} 的 model_class_names 必须是列表")
        subset_raw = raw.get("subset")
        return CatalogEntry(
            class_id=class_id,
            canonical_label=_required_text(
                raw.get("canonical_label"),
                f"class[{class_id}].canonical_label",
            ),
            english_name=_required_text(
                raw.get("english_name"),
                f"class[{class_id}].english_name",
            ),
            display_name=_required_text(
                raw.get("display_name"),
                f"class[{class_id}].display_name",
            ),
            subset=int(subset_raw) if subset_raw is not None else None,
            model_class_names=tuple(str(item) for item in model_names),
            is_background=background,
        )

    background = build_entry(raw_background, background=True)
    entries: list[CatalogEntry] = []
    for raw_id, raw_entry in raw_classes.items():
        if not isinstance(raw_entry, dict):
            raise CaptureError(f"classes[{raw_id}] 必须是对象")
        entry = build_entry(raw_entry, background=False)
        if int(raw_id) != entry.class_id:
            raise CaptureError(
                f"classes 键 {raw_id} 与内部 class_id {entry.class_id} 不一致"
            )
        entries.append(entry)
    entries.sort(key=lambda item: item.class_id)

    expected_ids = list(range(1, 41))
    actual_ids = [entry.class_id for entry in entries]
    if actual_ids != expected_ids:
        raise CaptureError(
            f"类别编号必须连续为 1-40；当前为 {actual_ids[:5]}...{actual_ids[-5:]}"
        )
    labels = [entry.canonical_label for entry in entries]
    if len(labels) != len(set(labels)):
        raise CaptureError("canonical_label 存在重复")
    if background.canonical_label in labels:
        raise CaptureError("空背景 canonical_label 与物品类别重复")

    source = dict(payload.get("source") or {})
    source.update(
        {
            "class_map_path": str(catalog_path),
            "class_map_sha256": sha256_file(catalog_path),
        }
    )
    catalog = ClassCatalog(
        dataset_name=dataset_name,
        entries=tuple(entries),
        background=background,
        model_support=dict(payload.get("model_support") or {}),
        source=source,
    )
    catalog.model_name_to_entry()
    return catalog


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _encode_image(image: np.ndarray, suffix: str, params: Sequence[int]) -> bytes:
    ok, encoded = cv2.imencode(suffix, image, list(params))
    if not ok:
        raise CaptureError(f"无法编码 {suffix} 图像")
    return encoded.tobytes()


def _intrinsics_mapping(intrinsics: Any) -> dict[str, float | int]:
    def read(name: str) -> float:
        if isinstance(intrinsics, Mapping):
            value = intrinsics.get(name)
            if value is None and name == "ppx":
                value = intrinsics.get("cx")
            if value is None and name == "ppy":
                value = intrinsics.get("cy")
        else:
            value = getattr(intrinsics, name, None)
            if value is None and name == "ppx":
                value = getattr(intrinsics, "cx", None)
            if value is None and name == "ppy":
                value = getattr(intrinsics, "cy", None)
        if value is None:
            raise CaptureError(f"内参缺少字段：{name}")
        return float(value)

    width = int(read("width"))
    height = int(read("height"))
    fx = read("fx")
    fy = read("fy")
    ppx = read("ppx")
    ppy = read("ppy")
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise CaptureError("RGB-D 内参无效")
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "ppx": ppx,
        "ppy": ppy,
        "cx": ppx,
        "cy": ppy,
    }


class CaptureStore:
    """Create immutable frame files inside one photos root."""

    def __init__(
        self,
        photos_root: str | Path,
        catalog: ClassCatalog,
        *,
        scene_types: Mapping[str, Mapping[str, Any]] | None = None,
        session_id_regex: str = r"^[0-9A-Za-z][0-9A-Za-z._-]{0,63}$",
        default_lighting_id: str = "site_A",
        default_scene_type: str = "box_11l",
        require_positive_instance_id: bool = False,
        require_notes_scene_types: Sequence[str] = (),
        enforce_label_scene_consistency: bool = False,
        burst_default_count: int = 5,
        burst_default_interval_ms: int = 700,
        burst_max_count: int = 10,
        multi_object_hint: Mapping[str, Any] | None = None,
    ) -> None:
        self.photos_root = Path(photos_root).expanduser().resolve()
        self.catalog = catalog
        self.scene_types = dict(scene_types or {})
        self.session_id_pattern = re.compile(session_id_regex)
        self.default_lighting_id = default_lighting_id
        self.default_scene_type = str(default_scene_type).strip()
        self.require_positive_instance_id = bool(require_positive_instance_id)
        self.require_notes_scene_types = tuple(
            str(item).strip() for item in require_notes_scene_types if str(item).strip()
        )
        self.enforce_label_scene_consistency = bool(enforce_label_scene_consistency)
        self.burst_default_count = int(burst_default_count)
        self.burst_default_interval_ms = int(burst_default_interval_ms)
        self.burst_max_count = int(burst_max_count)
        self.multi_object_hint = self._build_multi_object_hint(multi_object_hint)
        unknown_note_scenes = set(self.require_notes_scene_types) - set(self.scene_types)
        if unknown_note_scenes:
            raise CaptureError(
                "困难场景备注规则含未知场景："
                + ", ".join(sorted(unknown_note_scenes))
            )
        if self.default_scene_type not in self.scene_types:
            raise CaptureError(f"默认采集场景不存在：{self.default_scene_type}")
        if not 2 <= self.burst_default_count <= self.burst_max_count <= 10:
            raise CaptureError("连拍默认张数和上限必须满足 2 <= 默认值 <= 上限 <= 10")
        if not 100 <= self.burst_default_interval_ms <= 5000:
            raise CaptureError("连拍默认间隔必须在 100 到 5000 毫秒之间")
        self._lock = threading.RLock()
        self.photos_root.mkdir(parents=True, exist_ok=True)
        self.ensure_writable()

    def _build_multi_object_hint(
        self,
        raw: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise CaptureError("multi_object_hint 必须是对象")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise CaptureError("multi_object_hint.enabled 必须是布尔值")
        if not enabled:
            return None

        try:
            class_id = int(raw.get("class_id", 99))
        except (TypeError, ValueError) as error:
            raise CaptureError("multi_object_hint.class_id 必须是整数") from error
        official_ids = {entry.class_id for entry in self.catalog.entries}
        if class_id in official_ids or class_id == self.catalog.background.class_id:
            raise CaptureError(
                f"采集提示类别 {class_id} 不能与官方 1-40 类或空背景 0 重叠"
            )

        canonical_label = str(raw.get("canonical_label") or "").strip()
        english_name = str(raw.get("english_name") or "").strip()
        display_name = str(raw.get("display_name") or "").strip()
        default_scene_type = str(raw.get("default_scene_type") or "").strip()
        default_notes = str(raw.get("default_notes") or "").strip()
        require_instance_id = raw.get("require_instance_id", False)
        if not canonical_label or not english_name or not display_name:
            raise CaptureError("multi_object_hint 的类别名称不能为空")
        if not default_scene_type or default_scene_type not in self.scene_types:
            raise CaptureError(
                f"multi_object_hint.default_scene_type 不存在：{default_scene_type}"
            )
        if not default_notes or len(default_notes) > 1000:
            raise CaptureError("multi_object_hint.default_notes 必须为 1-1000 个字符")
        if not isinstance(require_instance_id, bool):
            raise CaptureError("multi_object_hint.require_instance_id 必须是布尔值")
        if require_instance_id:
            raise CaptureError("多物品采集提示类不能要求实例 ID")

        entry = CatalogEntry(
            class_id=class_id,
            canonical_label=canonical_label,
            english_name=english_name,
            display_name=display_name,
            subset=None,
            model_class_names=(),
            is_background=False,
            is_collection_hint=True,
            capture_hint_kind="multi_object",
        )
        return {
            "entry": entry,
            "enabled": True,
            "default_scene_type": default_scene_type,
            "default_notes": default_notes,
            "require_instance_id": False,
        }

    def resolve_capture_entry(self, class_id: int) -> CatalogEntry:
        class_id = int(class_id)
        if self.multi_object_hint is not None:
            hint_entry = self.multi_object_hint["entry"]
            if class_id == hint_entry.class_id:
                return hint_entry
        return self.catalog.by_id(class_id)

    def capture_policy_api(self) -> dict[str, Any]:
        """Return capture safeguards for the web UI without exposing paths."""
        multi_object_hint_api = None
        if self.multi_object_hint is not None:
            hint = self.multi_object_hint
            multi_object_hint_api = {
                **hint["entry"].to_api(),
                "enabled": True,
                "default_scene_type": hint["default_scene_type"],
                "default_notes": hint["default_notes"],
                "require_instance_id": False,
            }
        return {
            "default_scene_type": self.default_scene_type,
            "require_scene_selection": True,
            "require_positive_instance_id": self.require_positive_instance_id,
            "require_notes_scene_types": list(self.require_notes_scene_types),
            "enforce_label_scene_consistency": self.enforce_label_scene_consistency,
            "burst_default_count": self.burst_default_count,
            "burst_default_interval_ms": self.burst_default_interval_ms,
            "burst_max_count": self.burst_max_count,
            "multi_object_hint": multi_object_hint_api,
        }

    def ensure_writable(self) -> None:
        probe = self.photos_root / f".write-test-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            probe.write_text("ok\n", encoding="utf-8")
        except OSError as error:
            raise CaptureError(f"采集目录不可写：{self.photos_root}：{error}") from error
        finally:
            try:
                probe.unlink()
            except FileNotFoundError:
                pass

    def validate_session_id(self, value: str) -> str:
        session_id = value.strip()
        if not self.session_id_pattern.fullmatch(session_id):
            raise CaptureError(
                "会话 ID 只能使用英文字母、数字、点、下划线和连字符，"
                "必须以字母或数字开头，长度不超过 64"
            )
        if session_id in {".", ".."}:
            raise CaptureError("会话 ID 无效")
        return session_id

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        if not self.photos_root.is_dir():
            return sessions
        for path in self.photos_root.iterdir():
            if not path.is_dir() or not (path / "session.yaml").is_file():
                continue
            try:
                metadata = self._read_yaml(path / "session.yaml")
                sessions.append(
                    {
                        "session_id": str(metadata.get("session_id") or path.name),
                        "total_frames": int(metadata.get("total_frames") or 0),
                        "updated_at": metadata.get("updated_at"),
                        "path": str(path),
                    }
                )
            except (OSError, ValueError, CaptureError):
                continue
        sessions.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return sessions[:limit]

    def next_session_id(self, now: datetime | None = None) -> str:
        current = now or datetime.now().astimezone()
        prefix = current.strftime("%Y%m%d_session_")
        highest = 0
        used: set[str] = set()
        if self.photos_root.is_dir():
            for path in self.photos_root.iterdir():
                if not path.is_dir():
                    continue
                used.add(path.name)
                match = re.fullmatch(
                    re.escape(prefix) + r"([0-9]{3,})",
                    path.name,
                )
                if match:
                    highest = max(highest, int(match.group(1)))
        index = max(1, highest + 1)
        while f"{prefix}{index:03d}" in used:
            index += 1
        return f"{prefix}{index:03d}"

    def summary(self, session_id: str | None = None, recent_limit: int = 8) -> dict[str, Any]:
        sessions = self.list_sessions()
        selected = (
            self.validate_session_id(session_id)
            if isinstance(session_id, str) and session_id.strip()
            else None
        )
        if selected is None:
            selected = sessions[0]["session_id"] if sessions else self.next_session_id()
        session_dir = self.photos_root / selected
        metadata: dict[str, Any] = {}
        if (session_dir / "session.yaml").is_file():
            try:
                metadata = self._read_yaml(session_dir / "session.yaml")
            except (OSError, CaptureError, ValueError):
                metadata = {}
        return {
            "photosRoot": str(self.photos_root),
            "sessionId": selected,
            "sessionPath": str(session_dir),
            "sessions": sessions,
            "session": metadata,
            "recent": self._recent_records(session_dir, recent_limit),
        }

    def capture(
        self,
        *,
        session_id: str,
        class_id: int,
        frame: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale_m_per_unit: float,
        intrinsics: Any,
        camera_serial: str,
        rotate_180: bool,
        scene_type: str,
        instance_id: str = "",
        lighting_id: str = "",
        notes: str = "",
        capture_mode: str = "single",
        capture_group: str = "",
        model_predictions: Sequence[Mapping[str, Any]] = (),
        overlay: np.ndarray | None = None,
        save_overlay_jpeg: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            return self._capture_locked(
                session_id=session_id,
                class_id=class_id,
                frame=frame,
                depth_raw=depth_raw,
                depth_scale_m_per_unit=depth_scale_m_per_unit,
                intrinsics=intrinsics,
                camera_serial=camera_serial,
                rotate_180=rotate_180,
                scene_type=scene_type,
                instance_id=instance_id,
                lighting_id=lighting_id,
                notes=notes,
                capture_mode=capture_mode,
                capture_group=capture_group,
                model_predictions=model_predictions,
                overlay=overlay,
                save_overlay_jpeg=save_overlay_jpeg,
            )

    def _capture_locked(
        self,
        *,
        session_id: str,
        class_id: int,
        frame: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale_m_per_unit: float,
        intrinsics: Any,
        camera_serial: str,
        rotate_180: bool,
        scene_type: str,
        instance_id: str,
        lighting_id: str,
        notes: str,
        capture_mode: str,
        capture_group: str,
        model_predictions: Sequence[Mapping[str, Any]],
        overlay: np.ndarray | None,
        save_overlay_jpeg: bool,
    ) -> dict[str, Any]:
        session_id = self.validate_session_id(session_id)
        entry = self.resolve_capture_entry(int(class_id))
        if entry.is_collection_hint:
            hint = self.multi_object_hint
            if hint is None or entry != hint["entry"]:
                raise CaptureError("采集提示类别配置不可用")
            scene_type = str(hint["default_scene_type"])
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise CaptureError("彩色帧必须是 uint8 ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise CaptureError("彩色帧必须是 BGR 三通道图像")
        if not isinstance(depth_raw, np.ndarray) or depth_raw.dtype != np.uint16:
            raise CaptureError("深度帧必须是 uint16 ndarray")
        if depth_raw.ndim != 2 or depth_raw.shape != frame.shape[:2]:
            raise CaptureError("深度图与彩色图尺寸不一致")
        if not np.isfinite(depth_scale_m_per_unit) or depth_scale_m_per_unit <= 0:
            raise CaptureError("深度尺度必须大于 0")
        if not camera_serial.strip():
            raise CaptureError("没有读到 D405 相机序列号")
        if scene_type not in self.scene_types:
            raise CaptureError(f"未知采集场景：{scene_type}")
        scene_meta = self.scene_types[scene_type]
        box_type = str(scene_meta.get("box_type") or "none")
        if entry.is_collection_hint:
            hint = self.multi_object_hint
            if hint is None:
                raise CaptureError("采集提示类别配置不可用")
            instance_id = ""
            if not notes.strip():
                notes = str(hint["default_notes"])
        instance = instance_id.strip()
        if len(instance) > 100 or any(char in instance for char in "/\\\0"):
            raise CaptureError("实例 ID 长度不能超过 100，且不能包含路径分隔符")
        lighting = (lighting_id.strip() or self.default_lighting_id)
        if len(lighting) > 80 or any(char in lighting for char in "/\\\0"):
            raise CaptureError("光照/场地 ID 无效")
        note_text = notes.strip()
        if len(note_text) > 1000:
            raise CaptureError("备注不能超过 1000 个字符")
        if self.enforce_label_scene_consistency:
            if entry.is_background and scene_type != "box_11l_empty":
                raise CaptureError("空背景/无目标样本必须选择“空 11L 收纳盒”场景")
            if not entry.is_background and scene_type == "box_11l_empty":
                raise CaptureError("“空 11L 收纳盒”场景不能选择正类别")
        if (
            self.require_positive_instance_id
            and not entry.is_background
            and not entry.is_collection_hint
            and not instance
        ):
            raise CaptureError("正类样本必须填写实例 ID，例如 11-Padlock#01")
        if scene_type in self.require_notes_scene_types and not note_text:
            raise CaptureError(
                f"场景“{scene_meta.get('label') or scene_type}”必须填写物品布局、遮挡或视角备注"
            )
        mode = capture_mode.strip() or "single"
        if mode not in {"single", "burst"}:
            raise CaptureError("capture_mode 必须是 single 或 burst")
        group = capture_group.strip()
        if len(group) > 100 or any(char in group for char in "/\\\0"):
            raise CaptureError("capture_group 无效")

        now = datetime.now().astimezone()
        timestamp = now.isoformat(timespec="milliseconds")
        session_dir = self.photos_root / session_id
        directories = {
            "rgb": session_dir / "rgb",
            "depth": session_dir / "depth",
            "meta": session_dir / "meta",
            "overlays": session_dir / "overlays",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        session_path = session_dir / "session.yaml"
        session = (
            self._read_yaml(session_path)
            if session_path.is_file()
            else {
                "schema_version": 1,
                "session_id": session_id,
                "dataset_name": self.catalog.dataset_name,
                "created_at": timestamp,
                "status": "active",
                "total_frames": 0,
                "counts_by_class": {},
                "counts_by_scene": {},
                "notes": [],
            }
        )
        previous_serial = str(session.get("camera_serial") or "")
        if previous_serial and previous_serial != camera_serial:
            raise CaptureError(
                "同一会话检测到不同相机序列号："
                f"{previous_serial} -> {camera_serial}；请新建会话"
            )

        next_index = self._next_frame_index(directories["rgb"])
        frame_id = f"{next_index:06d}"
        stem = f"{frame_id}_{entry.canonical_label}"
        rgb_path = directories["rgb"] / f"{stem}.png"
        depth_path = directories["depth"] / f"{stem}.depth.png"
        metadata_path = directories["meta"] / f"{stem}.json"
        overlay_path = directories["overlays"] / f"{stem}.jpg"

        rgb_bytes = _encode_image(frame, ".png", [cv2.IMWRITE_PNG_COMPRESSION, 3])
        depth_bytes = _encode_image(
            depth_raw,
            ".png",
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if overlay is None:
            overlay = frame.copy()
        if overlay.shape != frame.shape:
            raise CaptureError("预览图尺寸与彩色帧不一致")
        overlay_copy = overlay.copy()
        label_text = f"{entry.class_id:02d} {entry.display_name} | {entry.canonical_label}"
        cv2.rectangle(overlay_copy, (0, 0), (overlay_copy.shape[1], 34), (18, 24, 30), -1)
        cv2.putText(
            overlay_copy,
            label_text[:100],
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 248, 252),
            1,
            cv2.LINE_AA,
        )
        overlay_bytes = _encode_image(
            overlay_copy,
            ".jpg",
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        rel_rgb = rgb_path.relative_to(session_dir).as_posix()
        rel_depth = depth_path.relative_to(session_dir).as_posix()
        rel_meta = metadata_path.relative_to(session_dir).as_posix()
        rel_overlay = overlay_path.relative_to(session_dir).as_posix()
        intrinsic_payload = _intrinsics_mapping(intrinsics)
        if (
            intrinsic_payload["width"] != frame.shape[1]
            or intrinsic_payload["height"] != frame.shape[0]
        ):
            raise CaptureError("内参尺寸与最终 RGB-D 图像尺寸不一致")

        normalized_predictions = self._normalize_model_predictions(
            model_predictions,
            entry.class_id,
        )
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "dataset_name": self.catalog.dataset_name,
            "class_map_sha256": self.catalog.source.get(
                "class_map_sha256"
            ),
            "class_catalog_fingerprint": self.catalog.fingerprint(),
            "session_id": session_id,
            "scene_id": f"{session_id}-{frame_id}",
            "frame_id": frame_id,
            "captured_at": timestamp,
            "class_id": entry.class_id,
            "label_index": entry.label_index,
            "canonical_label": entry.canonical_label,
            "english_name": entry.english_name,
            "display_name": entry.display_name,
            "is_background": entry.is_background,
            "is_collection_hint": entry.is_collection_hint,
            "capture_hint_kind": entry.capture_hint_kind,
            "requires_per_object_bbox_review": bool(
                entry.is_collection_hint
                and entry.capture_hint_kind == "multi_object"
            ),
            "instance_id": instance,
            "scene_type": scene_type,
            "scene_label": str(scene_meta.get("label") or scene_type),
            "box_type": box_type,
            "lighting_id": lighting,
            "capture_mode": mode,
            "capture_group": group,
            "notes": note_text,
            "camera_serial": camera_serial,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "rgb_encoding": "PNG/BGR8",
            "depth_encoding": "PNG/uint16",
            "depth_scale_m": float(depth_scale_m_per_unit),
            "depth_scale_m_per_unit": float(depth_scale_m_per_unit),
            "intrinsics": intrinsic_payload,
            "rotate_180": bool(rotate_180),
            "alignment": "depth_to_color",
            "model_prediction": normalized_predictions,
            "model_support": {
                "current_model_supported": entry.class_id in {
                    int(value)
                    for value in self.catalog.model_support.get(
                        "supported_class_ids", []
                    )
                },
                "purpose": "auxiliary_only_not_dataset_label",
            },
            "split": None,
            "files": {
                "rgb": rel_rgb,
                "depth": rel_depth,
                "metadata": rel_meta,
                "overlay": rel_overlay if save_overlay_jpeg else None,
            },
        }

        _atomic_write_bytes(rgb_path, rgb_bytes)
        _atomic_write_bytes(depth_path, depth_bytes)
        _atomic_write_text(
            metadata_path,
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )
        if save_overlay_jpeg:
            _atomic_write_bytes(overlay_path, overlay_bytes)

        frames_path = session_dir / "frames.jsonl"
        with frames_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        counts_by_class = dict(session.get("counts_by_class") or {})
        counts_by_class[entry.canonical_label] = int(
            counts_by_class.get(entry.canonical_label, 0)
        ) + 1
        counts_by_scene = dict(session.get("counts_by_scene") or {})
        counts_by_scene[scene_type] = int(counts_by_scene.get(scene_type, 0)) + 1
        session.update(
            {
                "updated_at": timestamp,
                "camera_serial": camera_serial,
                "camera_model": "Intel RealSense D405",
                "camera_width": int(frame.shape[1]),
                "camera_height": int(frame.shape[0]),
                "rotate_180": bool(rotate_180),
                "depth_scale_m_per_unit": float(depth_scale_m_per_unit),
                "intrinsics": intrinsic_payload,
                "class_map": str(
                    Path(str(self.catalog.source.get("class_map_path", "")))
                    if self.catalog.source.get("class_map_path")
                    else ""
                ),
                "class_map_sha256": self.catalog.source.get(
                    "class_map_sha256"
                ),
                "total_frames": int(session.get("total_frames", 0)) + 1,
                "last_frame_id": frame_id,
                "counts_by_class": counts_by_class,
                "counts_by_scene": counts_by_scene,
                "last_lighting_id": lighting,
                "last_scene_type": scene_type,
            }
        )
        note_history = list(session.get("session_notes") or [])
        if note_text and (not note_history or note_history[-1] != note_text):
            note_history.append(note_text)
            session["session_notes"] = note_history[-50:]
        _atomic_write_text(
            session_path,
            yaml.safe_dump(
                session,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
        )
        manifest_path = self._update_checksums(session_dir)
        recent = self._recent_records(session_dir, 8)
        return {
            "ok": True,
            "session_id": session_id,
            "frame_id": frame_id,
            "scene_id": metadata["scene_id"],
            "class_id": entry.class_id,
            "class_name": entry.canonical_label,
            "display_name": entry.display_name,
            "session_path": str(session_dir),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "metadata_path": str(metadata_path),
            "overlay_url": (
                f"/api/capture/overlay.jpg?session_id={session_id}&frame_id={frame_id}"
                if save_overlay_jpeg
                else None
            ),
            "total_frames": int(session["total_frames"]),
            "counts_by_class": counts_by_class,
            "counts_by_scene": counts_by_scene,
            "recent": recent,
            "checksums_path": str(manifest_path),
        }

    def overlay_path(self, session_id: str, frame_id: str) -> Path:
        session = self.validate_session_id(session_id)
        if not re.fullmatch(r"[0-9]{6,20}", frame_id):
            raise CaptureError("frame_id 无效")
        overlay_dir = self.photos_root / session / "overlays"
        matches = sorted(overlay_dir.glob(f"{frame_id}_*.jpg"))
        if not matches:
            raise FileNotFoundError("找不到该帧预览图")
        candidate = matches[0].resolve()
        if candidate.parent != overlay_dir.resolve():
            raise CaptureError("预览图路径越界")
        return candidate

    def _next_frame_index(self, rgb_dir: Path) -> int:
        highest = 0
        for path in rgb_dir.glob("*.png"):
            match = re.match(r"([0-9]{6,20})_", path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _normalize_model_predictions(
        self,
        predictions: Sequence[Mapping[str, Any]],
        selected_class_id: int,
    ) -> dict[str, Any]:
        name_to_entry = self.catalog.model_name_to_entry()
        normalized: list[dict[str, Any]] = []
        for prediction in predictions:
            model_class = str(prediction.get("class") or "")
            catalog_entry = name_to_entry.get(model_class)
            normalized.append(
                {
                    "model_class": model_class,
                    "catalog_class_id": (
                        catalog_entry.class_id if catalog_entry else None
                    ),
                    "catalog_label": (
                        catalog_entry.canonical_label if catalog_entry else None
                    ),
                    "confidence": round(
                        float(prediction.get("stableConfidence", prediction.get("confidence", 0.0))),
                        4,
                    ),
                    "bbox_xyxy": [
                        round(float(value), 1)
                        for value in prediction.get("bbox", [])
                    ],
                }
            )
        top = max(normalized, key=lambda item: item["confidence"], default=None)
        return {
            "mode": "auxiliary_only",
            "selected_label_confirmed_by_model": bool(
                top and top.get("catalog_class_id") == selected_class_id
            ),
            "top": top,
            "detections": normalized,
        }

    def _recent_records(self, session_dir: Path, limit: int) -> list[dict[str, Any]]:
        frames_path = session_dir / "frames.jsonl"
        if not frames_path.is_file():
            return []
        records: list[dict[str, Any]] = []
        try:
            with frames_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(
                        {
                            "frame_id": item.get("frame_id"),
                            "class_id": item.get("class_id"),
                            "canonical_label": item.get("canonical_label"),
                            "display_name": item.get("display_name"),
                            "scene_type": item.get("scene_type"),
                            "captured_at": item.get("captured_at"),
                            "overlay_url": (
                                f"/api/capture/overlay.jpg?session_id={session_dir.name}"
                                f"&frame_id={item.get('frame_id')}"
                                if item.get("files", {}).get("overlay")
                                else None
                            ),
                        }
                    )
        except OSError:
            return []
        return records[-max(1, limit):][::-1]

    def _update_checksums(self, session_dir: Path) -> Path:
        manifest = session_dir / "SHA256SUMS"
        existing: dict[str, str] = {}
        if manifest.is_file():
            try:
                for line in manifest.read_text(encoding="utf-8").splitlines():
                    digest, separator, relative = line.partition("  ")
                    if (
                        separator
                        and len(digest) == 64
                        and all(character in "0123456789abcdef" for character in digest)
                    ):
                        existing[relative] = digest
            except OSError:
                existing = {}
        files = sorted(
            path
            for path in session_dir.rglob("*")
            if path.is_file() and path != manifest
        )
        # Image/depth/overlay files are immutable after creation.  Only the
        # growing indexes and session metadata need to be hashed again.
        changing_names = {"frames.jsonl", "session.yaml"}
        lines: list[str] = []
        for path in files:
            relative = path.relative_to(session_dir).as_posix()
            digest = existing.get(relative)
            if digest is None or path.name in changing_names:
                digest = sha256_file(path)
            lines.append(f"{digest}  {relative}")
        _atomic_write_text(manifest, "\n".join(lines) + "\n")
        return manifest

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise CaptureError(f"YAML 格式错误：{path}：{error}") from error
        if not isinstance(payload, dict):
            raise CaptureError(f"YAML 根节点必须是对象：{path}")
        return payload
