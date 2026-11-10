#!/usr/bin/env python3
"""Build a real/simulation fine-tune dataset with session-grouped validation.

Real frames captured as a burst are highly correlated, so splitting them by
frame index leaks nearly identical views into validation. This builder groups
frames by source and timestamp gaps, then holds out whole capture groups where
possible. Auxiliary real_val and sim_val directories are generated for clean
model comparison.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6}_\d{6})$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class RealSample:
    source: str
    class_id: int
    class_name: str
    image_name: str
    label_name: str
    image_path: Path
    label_path: Path
    timestamp: datetime
    capture_group: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", required=True)
    parser.add_argument("--sim-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument("--group-gap", type=float, default=2.0)
    parser.add_argument("--group-val-count", type=int, default=1)
    parser.add_argument(
        "--holdout-source",
        action="append",
        default=None,
        help="Use every frame from this capture source as validation.",
    )
    parser.add_argument("--tail-val-fraction", type=float, default=0.33)
    parser.add_argument("--sim-val", type=int, default=60)
    return parser.parse_args()


def parse_timestamp(image_name: str) -> datetime:
    match = TIMESTAMP_RE.search(Path(image_name).stem)
    if not match:
        raise ValueError(f"cannot parse capture timestamp from {image_name}")
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S_%f")


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def link(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"missing source file: {src}")
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"refusing to replace existing link: {dst}")
    os.symlink(src.resolve(), dst)


def load_real_samples(real_data: Path) -> list[RealSample]:
    manifest_path = real_data / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    samples: list[RealSample] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_id = int(row["class_id"])
            source = row["source"]
            image_name = row["image"]
            stem = Path(image_name).stem
            suffix = Path(image_name).suffix.lower()
            staged_image_name = f"real_c{class_id}_{source}_{stem}{suffix}"
            staged_label_name = f"real_c{class_id}_{source}_{stem}.txt"
            image_path = real_data / "images" / staged_image_name
            label_path = real_data / "labels" / staged_label_name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not label_path.is_file():
                raise FileNotFoundError(label_path)

            label_rows = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(label_rows) != 1:
                raise ValueError(
                    f"{label_path}: expected one box, found {len(label_rows)}"
                )
            fields = label_rows[0].split()
            if len(fields) != 5 or int(float(fields[0])) != class_id:
                raise ValueError(f"{label_path}: invalid label or class mismatch")

            samples.append(
                RealSample(
                    source=source,
                    class_id=class_id,
                    class_name=row["class_name"],
                    image_name=staged_image_name,
                    label_name=staged_label_name,
                    image_path=image_path,
                    label_path=label_path,
                    timestamp=parse_timestamp(image_name),
                )
            )
    if not samples:
        raise ValueError(f"no real samples found in {manifest_path}")
    return samples


def assign_capture_groups(
    samples: list[RealSample], gap_seconds: float
) -> dict[str, list[list[RealSample]]]:
    by_source: dict[str, list[RealSample]] = defaultdict(list)
    for sample in samples:
        by_source[sample.source].append(sample)

    grouped: dict[str, list[list[RealSample]]] = {}
    for source, source_samples in by_source.items():
        ordered = sorted(source_samples, key=lambda item: item.timestamp)
        groups: list[list[RealSample]] = [[ordered[0]]]
        for sample in ordered[1:]:
            gap = (sample.timestamp - groups[-1][-1].timestamp).total_seconds()
            if gap > gap_seconds:
                groups.append([])
            groups[-1].append(sample)
        grouped[source] = groups
    return grouped


def choose_split(
    source: str,
    groups: list[list[RealSample]],
    group_val_count: int,
    holdout_sources: set[str],
    tail_val_fraction: float,
    keep_source_in_train: bool,
) -> tuple[list[RealSample], list[RealSample], str]:
    if source in holdout_sources:
        return [], [sample for group in groups for sample in group], "source_holdout"
    if keep_source_in_train:
        return [sample for group in groups for sample in group], [], "alternate_source_holdout"

    if len(groups) >= 3:
        val_group_count = min(max(1, group_val_count), len(groups) - 1)
        train = [sample for group in groups[:-val_group_count] for sample in group]
        val = [sample for group in groups[-val_group_count:] for sample in group]
        return train, val, "whole_group_holdout"

    ordered = [sample for group in groups for sample in group]
    if len(ordered) < 2:
        raise ValueError(f"{source}: not enough samples to create train/val split")
    val_count = max(1, min(len(ordered) - 1, round(len(ordered) * tail_val_fraction)))
    return ordered[:-val_count], ordered[-val_count:], "single_session_tail"


def sample_evenly(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or not paths:
        return []
    count = min(count, len(paths))
    indexes = [round(index * (len(paths) - 1) / max(1, count - 1)) for index in range(count)]
    return [paths[index] for index in sorted(set(indexes))][:count]


def write_data_yaml(out: Path, class_names: list[str]) -> None:
    (out / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_names)}",
                "names:",
                *[f"- {name}" for name in class_names],
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_link(
    image_path: Path,
    label_path: Path,
    split: str,
    prefix: str,
    out: Path,
    extra_split: str | None = None,
) -> None:
    destination_image = f"{prefix}_{image_path.name}"
    destination_label = f"{prefix}_{label_path.stem}.txt"
    link(image_path, out / "images" / split / destination_image)
    link(label_path, out / "labels" / split / destination_label)
    if extra_split is not None:
        link(image_path, out / "images" / extra_split / destination_image)
        link(label_path, out / "labels" / extra_split / destination_label)


def main() -> int:
    args = parse_args()
    real_data = Path(args.real_data).resolve()
    sim_data = Path(args.sim_data).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {out}")
    if args.ratio < 0:
        raise SystemExit("--ratio must be non-negative")

    holdout_sources = set(args.holdout_source or ["banana_closeup"])
    samples = load_real_samples(real_data)
    grouped = assign_capture_groups(samples, args.group_gap)
    holdout_class_ids = {
        sample.class_id
        for source in holdout_sources
        for sample in (
            sample
            for group in grouped.get(source, [])
            for sample in group
        )
    }

    real_train: list[RealSample] = []
    real_val: list[RealSample] = []
    split_methods: dict[str, str] = {}
    group_rows: list[dict[str, str | int]] = []
    for source in sorted(grouped):
        groups = grouped[source]
        keep_source_in_train = (
            source not in holdout_sources
            and any(sample.class_id in holdout_class_ids for group in groups for sample in group)
        )
        train, val, method = choose_split(
            source,
            groups,
            args.group_val_count,
            holdout_sources,
            args.tail_val_fraction,
            keep_source_in_train,
        )
        real_train.extend(train)
        real_val.extend(val)
        split_methods[source] = method
        for group_index, group in enumerate(groups, start=1):
            split = "val" if any(sample in val for sample in group) else "train"
            for sample in group:
                group_rows.append(
                    {
                        "split": split,
                        "source": source,
                        "capture_group": group_index,
                        "timestamp": sample.timestamp.isoformat(timespec="milliseconds"),
                        "image": sample.image_name,
                        "class_id": sample.class_id,
                        "class_name": sample.class_name,
                        "selection": method,
                    }
                )

    if not real_train or not real_val:
        raise SystemExit("both real train and real val must be non-empty")

    for split in ("train", "val", "real_val", "sim_val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sample in real_train:
        write_link(
            sample.image_path,
            sample.label_path,
            "train",
            "real",
            out,
        )
    for sample in real_val:
        write_link(
            sample.image_path,
            sample.label_path,
            "val",
            "real",
            out,
            extra_split="real_val",
        )

    sim_train_pool = image_files(sim_data / "images" / "train")
    sim_count = min(
        len(sim_train_pool),
        int(round(args.ratio * len(real_train))),
    )
    sim_train = sample_evenly(sim_train_pool, sim_count)
    for image_path in sim_train:
        label_path = sim_data / "labels" / "train" / f"{image_path.stem}.txt"
        write_link(image_path, label_path, "train", "sim", out)

    sim_val_pool = image_files(sim_data / "images" / "val")
    sim_val = sample_evenly(sim_val_pool, args.sim_val)
    for image_path in sim_val:
        label_path = sim_data / "labels" / "val" / f"{image_path.stem}.txt"
        write_link(
            image_path,
            label_path,
            "val",
            "sim",
            out,
            extra_split="sim_val",
        )

    classes_file = real_data / "classes.txt"
    class_names = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (out / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )
    write_data_yaml(out, class_names)

    with (out / "grouped_split.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "source",
            "capture_group",
            "timestamp",
            "image",
            "class_id",
            "class_name",
            "selection",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(group_rows)

    train_classes = Counter(sample.class_id for sample in real_train)
    val_classes = Counter(sample.class_id for sample in real_val)
    print(f"grouped fine-tune dataset -> {out}")
    print(
        f"  train: real={len(real_train)} sim={len(sim_train)} "
        f"total={len(real_train) + len(sim_train)}"
    )
    print(
        f"  val:   real={len(real_val)} sim={len(sim_val)} "
        f"total={len(real_val) + len(sim_val)}"
    )
    print(
        "  real train classes: "
        + ", ".join(f"{class_id}:{count}" for class_id, count in sorted(train_classes.items()))
    )
    print(
        "  real val classes:   "
        + ", ".join(f"{class_id}:{count}" for class_id, count in sorted(val_classes.items()))
    )
    for source in sorted(split_methods):
        group_count = len(grouped[source])
        print(
            f"  {source}: groups={group_count} method={split_methods[source]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
