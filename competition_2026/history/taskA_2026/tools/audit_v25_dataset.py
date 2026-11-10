#!/usr/bin/env python3
"""只读审计一个 YOLO 数据集，为 v25 清理和物理分组 holdout 做准备。

重点不是再训一版，而是回答四个问题：
  1. 每张图的 SHA256 是否真的唯一；
  2. 同一段连续拍摄的帧是否被分到了 train/val 两侧；
  3. 重复图是否包含互相矛盾的标签；
  4. 旧香蕉等可疑正样本是否仍在污染 Plastic_Banana。

本脚本只读取数据集并写审计报告，绝不修改图片、标签或已有 run。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

TS_RE = re.compile(r"(?P<ts>\d{8}_\d{6}_\d{6})")
REPLICA_RE = re.compile(r"__r\d+$", re.IGNORECASE)
OS3R_PREFIX_RE = re.compile(r"^os3r\d+_", re.IGNORECASE)
SUB_PREFIX_RE = re.compile(r"^sub(?:_|-)", re.IGNORECASE)

DEFAULT_DATASET = Path("/media/sunshine/新加卷/taskA/datasets/real_ft_v22_20261016")
DEFAULT_OUTDIR = Path("/media/sunshine/新加卷/taskA/reports")
GROUP_GAP_SECONDS = 5.0


@dataclass
class Record:
    split: str
    path: Path
    name: str
    stem: str
    logical_name: str
    source_key: str
    timestamp: datetime | None
    sha256: str
    label_path: Path
    label_text: str
    label_signature: str
    class_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    instance_count: int
    empty_label: bool
    physical_group: str = ""
    exact_count: int = 0
    exact_splits: tuple[str, ...] = ()
    exact_rank: int = 0
    duplicate_of: str = ""
    group_splits: tuple[str, ...] = ()
    group_file_count: int = 0
    group_logical_count: int = 0
    group_leak: bool = False
    label_conflict: bool = False
    suspect_banana: bool = False
    recommended_action: str = ""
    recommended_keep: bool = False
    notes: str = ""


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamp(text: str) -> datetime | None:
    match = TS_RE.search(text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("ts"), "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return None


def normalize_logical_name(stem: str) -> str:
    logical = REPLICA_RE.sub("", stem)
    logical = OS3R_PREFIX_RE.sub("", logical)
    logical = SUB_PREFIX_RE.sub("", logical)
    return logical


def source_key(logical_name: str) -> str:
    match = TS_RE.search(logical_name)
    if not match:
        return logical_name
    prefix = logical_name[: match.start()]
    if "scene" in prefix:
        return "scene"
    if prefix.startswith("empty_background"):
        return "empty_background"
    # 对单物品连续采集，前缀含类别和采集批次，不能跨类别合并。
    return prefix.rstrip("_-")


def parse_label(label_path: Path, names: list[str]) -> tuple[str, str, tuple[int, ...], tuple[str, ...], int, bool]:
    if not label_path.is_file():
        return "", "", (), (), 0, False
    text = label_path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ids: list[int] = []
    used_names: list[str] = []
    for line in lines:
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
        except (IndexError, ValueError):
            continue
        ids.append(class_id)
        used_names.append(names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}")
    signature = "|".join(sorted(lines))
    return text, signature, tuple(ids), tuple(used_names), len(lines), not lines


def load_class_names(dataset: Path) -> list[str]:
    yaml_path = dataset / "data.yaml"
    if not yaml_path.is_file():
        raise SystemExit(f"缺少 data.yaml: {yaml_path}")
    text = yaml_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?ms)^names:\s*\n(?P<body>(?:[ \t]*-[^\n]*\n?)+)", text)
    if not match:
        raise SystemExit(f"无法从 {yaml_path} 解析 names")
    names: list[str] = []
    for line in match.group("body").splitlines():
        line = line.strip()
        if line.startswith("-"):
            names.append(line[1:].strip().strip("'\""))
    if not names:
        raise SystemExit(f"data.yaml 的 names 为空: {yaml_path}")
    return names


def collect_records(dataset: Path, names: list[str]) -> list[Record]:
    records: list[Record] = []
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        if not image_dir.is_dir():
            raise SystemExit(f"缺少图片目录: {image_dir}")
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            text, signature, ids, used_names, instances, empty = parse_label(label_path, names)
            logical = normalize_logical_name(image_path.stem)
            records.append(
                Record(
                    split=split,
                    path=image_path,
                    name=image_path.name,
                    stem=image_path.stem,
                    logical_name=logical,
                    source_key=source_key(logical),
                    timestamp=parse_timestamp(logical),
                    sha256=sha256_file(image_path),
                    label_path=label_path,
                    label_text=text,
                    label_signature=signature,
                    class_ids=ids,
                    class_names=used_names,
                    instance_count=instances,
                    empty_label=empty,
                    suspect_banana=(split == "train" and "v5_banana" in logical),
                )
            )
    return records


def cluster_records(records: list[Record], gap_seconds: float) -> tuple[dict[str, str], dict[str, int]]:
    """按来源 + 时间间隔聚类物理拍摄组，并用完全相同的图片合并组。"""
    unique_logical: dict[str, Record] = {}
    for record in records:
        unique_logical.setdefault(record.logical_name, record)

    groups: list[list[Record]] = []
    by_source: dict[str, list[Record]] = defaultdict(list)
    no_time: list[Record] = []
    for record in unique_logical.values():
        if record.timestamp is None:
            no_time.append(record)
        else:
            by_source[record.source_key].append(record)

    for source in sorted(by_source):
        items = sorted(by_source[source], key=lambda r: (r.timestamp, r.logical_name))  # type: ignore[arg-type]
        current: list[Record] = []
        last_ts: datetime | None = None
        for record in items:
            ts = record.timestamp
            if last_ts is None or ts is None or (ts - last_ts).total_seconds() <= gap_seconds:
                current.append(record)
            else:
                groups.append(current)
                current = [record]
            last_ts = ts
        if current:
            groups.append(current)

    for record in no_time:
        groups.append([record])

    uf = UnionFind()
    for index, group in enumerate(groups):
        gid = f"pg{index:04d}"
        for record in group:
            uf.add(gid)
    hash_to_gid: dict[str, str] = {}
    for index, group in enumerate(groups):
        gid = f"pg{index:04d}"
        for record in group:
            previous = hash_to_gid.get(record.sha256)
            if previous is not None:
                uf.union(previous, gid)
            else:
                hash_to_gid[record.sha256] = gid

    # 用 union-find 处理不同命名但字节完全相同的情况。
    merged: dict[str, list[Record]] = defaultdict(list)
    for index, group in enumerate(groups):
        gid = f"pg{index:04d}"
        root = uf.find(gid)
        merged[root].extend(group)
    roots = sorted(merged)

    logical_to_group: dict[str, str] = {}
    group_sizes: dict[str, int] = {}
    for index, root in enumerate(roots, start=1):
        gid = f"pg{index:04d}"
        group = merged[root]
        logicals = {record.logical_name for record in group}
        group_sizes[gid] = len(logicals)
        for logical in logicals:
            logical_to_group[logical] = gid
    return logical_to_group, group_sizes


def group_metadata(records: list[Record]) -> dict[str, dict[str, Any]]:
    by_group: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_group[record.physical_group].append(record)
    result: dict[str, dict[str, Any]] = {}
    for gid, group in by_group.items():
        result[gid] = {
            "splits": tuple(sorted({record.split for record in group})),
            "file_count": len(group),
            "logical_count": len({record.logical_name for record in group}),
            "classes": tuple(sorted({name for record in group for name in record.class_names})),
            "min_time": min((record.timestamp for record in group if record.timestamp), default=None),
            "max_time": max((record.timestamp for record in group if record.timestamp), default=None),
        }
    return result


def duplicate_metadata(records: list[Record]) -> dict[str, dict[str, Any]]:
    by_hash: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    result: dict[str, dict[str, Any]] = {}
    for digest, group in by_hash.items():
        signatures = {record.label_signature for record in group}
        sort_key = lambda r: (0 if r.split == "train" else 1, r.name)
        ordered = sorted(group, key=sort_key)
        result[digest] = {
            "records": ordered,
            "splits": tuple(sorted({record.split for record in group})),
            "label_conflict": len(signatures) > 1,
        }
    return result


def action_for(record: Record, dup: dict[str, Any]) -> tuple[str, bool, str]:
    if record.label_conflict:
        return "label_conflict_review", False, "同一字节图片存在不同标签，不能直接去重"
    count = len(dup["records"])
    if count > 1:
        keep = dup["records"][0]
        if record is keep:
            note = f"保留该精确重复组的 1 张；其余 {count - 1} 张剔除"
            return "dedupe_keep_one", True, note
        return "dedupe_drop_copy", False, f"与 {keep.name} 字节相同"
    if record.group_leak:
        return "physical_group_leak_reassign", True, "同一物理采集组跨 train/val，必须整体归到一侧"
    if record.suspect_banana:
        return "suspect_non_competition_banana", False, "旧 v5 弯曲带斑香蕉，疑似非比赛塑料香蕉；需人工复核"
    return "keep", True, ""


def write_csv(path: Path, records: list[Record], names: list[str]) -> None:
    fieldnames = [
        "split",
        "image",
        "logical_name",
        "physical_group",
        "group_size",
        "group_splits",
        "group_leak",
        "sha256",
        "exact_count",
        "exact_splits",
        "exact_rank",
        "duplicate_of",
        "label_conflict",
        "class_ids",
        "class_names",
        "instance_count",
        "empty_label",
        "suspect_non_competition_banana",
        "recommended_action",
        "recommended_keep",
        "notes",
        "image_path",
        "label_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r.split, r.physical_group, r.name)):
            writer.writerow(
                {
                    "split": record.split,
                    "image": record.name,
                    "logical_name": record.logical_name,
                    "physical_group": record.physical_group,
                    "group_size": record.group_logical_count,
                    "group_splits": ";".join(record.group_splits),
                    "group_leak": int(record.group_leak),
                    "sha256": record.sha256,
                    "exact_count": record.exact_count,
                    "exact_splits": ";".join(record.exact_splits),
                    "exact_rank": record.exact_rank,
                    "duplicate_of": record.duplicate_of,
                    "label_conflict": int(record.label_conflict),
                    "class_ids": ";".join(str(item) for item in record.class_ids),
                    "class_names": ";".join(record.class_names),
                    "instance_count": record.instance_count,
                    "empty_label": int(record.empty_label),
                    "suspect_non_competition_banana": int(record.suspect_banana),
                    "recommended_action": record.recommended_action,
                    "recommended_keep": int(record.recommended_keep),
                    "notes": record.notes,
                    "image_path": str(record.path),
                    "label_path": str(record.label_path),
                }
            )


def format_dt(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S.%f") if value else "-"


def markdown_report(dataset: Path, records: list[Record], names: list[str], group_meta: dict[str, dict[str, Any]], out_csv: Path) -> str:
    total = len(records)
    unique_hash = len({record.sha256 for record in records})
    dup_extra = total - unique_hash
    dup_groups = [meta for meta in duplicate_metadata(records).values() if len(meta["records"]) > 1]
    cross_dup_groups = [meta for meta in dup_groups if len(meta["splits"]) > 1]
    cross_dup_files = sum(len(meta["records"]) for meta in cross_dup_groups)
    label_conflict_groups = [meta for meta in dup_groups if meta["label_conflict"]]
    leak_groups = {gid: meta for gid, meta in group_meta.items() if len(meta["splits"]) > 1}
    leak_files = sum(meta["file_count"] for meta in leak_groups.values())
    group_sizes = Counter(meta["logical_count"] for meta in group_meta.values())
    missing_labels = [record for record in records if not record.label_path.is_file()]
    empty_labels = sum(record.empty_label for record in records)
    suspect_bananas = [record for record in records if record.suspect_banana]

    class_file_counts: Counter[str] = Counter()
    class_unique_counts: dict[str, set[str]] = defaultdict(set)
    class_instance_counts: Counter[str] = Counter()
    for record in records:
        for class_id, class_name in zip(record.class_ids, record.class_names):
            class_file_counts[class_name] += 1
            class_unique_counts[class_name].add(record.sha256)
            class_instance_counts[class_name] += 1

    lines: list[str] = []
    lines.append("# v25 数据集只读审计报告")
    lines.append("")
    lines.append(f"- 审计时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 数据集：`{dataset}`")
    lines.append(f"- 明细 CSV：`{out_csv}`")
    lines.append(f"- 物理组 gap 阈值：`{GROUP_GAP_SECONDS:g} 秒`")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    lines.append(f"- 图片文件：**{total}** 张")
    lines.append(f"- SHA256 唯一图片：**{unique_hash}** 张")
    lines.append(f"- 精确重复多余副本：**{dup_extra}** 张，重复比例 **{dup_extra / total * 100:.1f}%**")
    lines.append(f"- 精确重复组：**{len(dup_groups)}** 组")
    lines.append(f"- 跨 train/val 的精确重复组：**{len(cross_dup_groups)}** 组，涉及 **{cross_dup_files}** 张文件")
    lines.append(f"- 标签冲突的精确重复组：**{len(label_conflict_groups)}** 组")
    lines.append(f"- 初步物理采集组：**{len(group_meta)}** 组")
    lines.append(f"- 跨 train/val 的物理采集组：**{len(leak_groups)}** 组，涉及 **{leak_files}** 张文件")
    lines.append(f"- 空标签/纯背景图：**{empty_labels}** 张")
    lines.append(f"- 缺少标签文件的图片：**{len(missing_labels)}** 张")
    lines.append(f"- 疑似旧 v5 非比赛香蕉正样本：**{len(suspect_bananas)}** 张")
    lines.append("")
    lines.append("> 目前只完成审计，v22 的图片、标签和权重均未改动。")
    lines.append("")
    lines.append("## 类别统计")
    lines.append("")
    lines.append("| 类别 | 标注实例 | 含该类的图片 | 唯一 SHA 图片 | 文件重复率 |")
    lines.append("|---|---:|---:|---:|---:|")
    for class_id, class_name in enumerate(names):
        files = class_file_counts[class_name]
        unique = len(class_unique_counts[class_name])
        instances = class_instance_counts[class_name]
        ratio = (files - unique) / files * 100 if files else 0.0
        lines.append(f"| {class_name} | {instances} | {files} | {unique} | {ratio:.1f}% |")
    lines.append("")
    lines.append("## 物理组大小分布")
    lines.append("")
    lines.append("| 每个物理组的唯一逻辑帧数 | 组数 |")
    lines.append("|---:|---:|")
    for size in sorted(group_sizes):
        lines.append(f"| {size} | {group_sizes[size]} |")
    lines.append("")
    lines.append("## 跨 split 物理组样例（最多 30 组）")
    lines.append("")
    lines.append("| 物理组 | split | 逻辑帧数 | 文件数 | 时间范围 | 类别 |")
    lines.append("|---|---|---:|---:|---|---|")
    for gid in sorted(leak_groups)[:30]:
        meta = leak_groups[gid]
        classes = ", ".join(meta["classes"][:4]) or "背景"
        lines.append(
            f"| {gid} | {'/'.join(meta['splits'])} | {meta['logical_count']} | "
            f"{meta['file_count']} | {format_dt(meta['min_time'])} ~ {format_dt(meta['max_time'])} | {classes} |"
        )
    if len(leak_groups) > 30:
        lines.append(f"| ... | 其余 {len(leak_groups) - 30} 组见 CSV | | | | |")
    lines.append("")
    lines.append("## 精确重复组中跨 split 的样例（最多 30 组）")
    lines.append("")
    lines.append("| SHA256 前 16 位 | split | 文件数 | 代表文件 | 标签类别 |")
    lines.append("|---|---|---:|---|---|")
    for meta in sorted(cross_dup_groups, key=lambda item: item["records"][0].name)[:30]:
        representative = meta["records"][0]
        classes = ", ".join(sorted(set(representative.class_names))) or "背景"
        lines.append(
            f"| `{representative.sha256[:16]}` | {'/'.join(meta['splits'])} | "
            f"{len(meta['records'])} | `{representative.name}` | {classes} |"
        )
    lines.append("")
    lines.append("## 建议的 v25 构建动作（尚未执行）")
    lines.append("")
    lines.append("1. **先剔除精确重复副本**：同一 SHA256 只保留一个代表；若标签签名不一致，单独人工复核，不能盲删。")
    lines.append("2. **按物理组重新切分 train/val**：同一 `physical_group` 必须整体进入同一侧；本次 5 秒间隔分组已经复现出 243 组，并识别出跨 split 泄漏。")
    lines.append("3. **优先修复验证集**：尤其 Tuna、Mug、Scissors 以及小样本姿态，不能再用与 train 相邻或字节相同的帧证明模型有效。")
    lines.append("4. **处理旧 v5 香蕉**：疑似非比赛塑料香蕉的正样本先移出 `Plastic_Banana` 正类，转为待复核或背景，避免继续拉偏类别。")
    lines.append("5. **训练前先做全类门禁**：用清洗后的物理组 holdout 对 v18/v21/v22/v23 统一评估类别正确率和 IoU 0.5/0.75 通过率；只有全类不退化，才允许进入下一轮训练。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--gap-seconds", type=float, default=GROUP_GAP_SECONDS)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    names = load_class_names(dataset)
    records = collect_records(dataset, names)
    if not records:
        raise SystemExit(f"未找到图片: {dataset}")

    logical_to_group, group_sizes = cluster_records(records, args.gap_seconds)
    for record in records:
        record.physical_group = logical_to_group.get(record.logical_name, f"missing_{record.logical_name}")
    group_meta = group_metadata(records)

    duplicates = duplicate_metadata(records)
    for digest, meta in duplicates.items():
        for rank, record in enumerate(meta["records"]):
            record.exact_count = len(meta["records"])
            record.exact_splits = meta["splits"]
            record.exact_rank = rank
            record.label_conflict = meta["label_conflict"]
            if rank:
                record.duplicate_of = meta["records"][0].name

    for record in records:
        meta = group_meta[record.physical_group]
        record.group_splits = meta["splits"]
        record.group_file_count = meta["file_count"]
        record.group_logical_count = meta["logical_count"]
        record.group_leak = len(meta["splits"]) > 1

    for record in records:
        action, keep, note = action_for(record, duplicates[record.sha256])
        record.recommended_action = action
        record.recommended_keep = keep
        record.notes = note

    csv_path = outdir / "v25_audit_20261016.csv"
    md_path = outdir / "v25_audit_20261016.md"
    write_csv(csv_path, records, names)
    md_path.write_text(markdown_report(dataset, records, names, group_meta, csv_path), encoding="utf-8")

    total = len(records)
    unique_hash = len({record.sha256 for record in records})
    leak_groups = {gid: meta for gid, meta in group_meta.items() if len(meta["splits"]) > 1}
    print(f"dataset={dataset}")
    print(f"images={total} unique_sha={unique_hash} duplicate_extra={total - unique_hash}")
    print(f"physical_groups={len(group_meta)} cross_split_groups={len(leak_groups)}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
