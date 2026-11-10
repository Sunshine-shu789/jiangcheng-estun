#!/usr/bin/env python3
"""Synchronize one metadata file from its validated label and scene index."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, only validate and report.",
    )
    return parser.parse_args()


def load_label_objects(root: Path, scene: str) -> list[dict[str, Any]]:
    label_path = root / "labels" / f"{scene}.txt"
    return [
        {
            "class_id": int(row.split()[0]),
            "bbox": [float(value) for value in row.split()[1:]],
        }
        for row in label_path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]


def load_scene_row(root: Path, scene: str) -> dict[str, Any]:
    matches = [
        json.loads(line)
        for line in (root / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("scene") == scene
    ]
    if len(matches) != 1:
        raise ValueError(f"{scene}: found {len(matches)} index rows")
    return matches[0]


def expected_label_object(
    obj: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = obj["bbox_xyxy"]
    return {
        "class_id": int(obj["class_id"]),
        "bbox": [
            (x1 + x2) / 2 / width,
            (y1 + y2) / 2 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ],
    }


def validate_index_against_label(
    root: Path,
    scene: str,
    row: dict[str, Any],
    width: int,
    height: int,
) -> None:
    labels = load_label_objects(root, scene)
    objects = row.get("objects")
    if not isinstance(objects, list) or len(labels) != len(objects):
        raise ValueError(f"{scene}: label/index object count mismatch")
    for index, (label, obj) in enumerate(zip(labels, objects), start=1):
        expected = expected_label_object(obj, width, height)
        if label["class_id"] != expected["class_id"] or any(
            abs(actual - wanted) > 1e-6
            for actual, wanted in zip(label["bbox"], expected["bbox"])
        ):
            raise ValueError(f"{scene}: label/index mismatch at object {index}")


def write_atomic(path: Path, content: str) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    metadata_path = root / "metadata" / f"{args.scene}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("scene") != args.scene:
        raise ValueError(f"{metadata_path}: scene id mismatch")
    width = int(metadata["color_intrinsics"]["width"])
    height = int(metadata["color_intrinsics"]["height"])

    row = load_scene_row(root, args.scene)
    validate_index_against_label(root, args.scene, row, width, height)
    index_objects = row.get("objects", [])
    metadata_objects = metadata.get("objects", [])
    if len(index_objects) != len(metadata_objects):
        raise ValueError(f"{args.scene}: metadata/index object count mismatch")

    changes: list[dict[str, Any]] = []
    for index, (source, target) in enumerate(
        zip(index_objects, metadata_objects),
        start=1,
    ):
        if (
            int(source["class_id"]) != int(target["class_id"])
            or source.get("class_name") != target.get("class_name")
        ):
            raise ValueError(f"{args.scene}: class mismatch at object {index}")
        if source["bbox_xyxy"] != target["bbox_xyxy"]:
            changes.append(
                {
                    "object": index,
                    "old": target["bbox_xyxy"],
                    "new": source["bbox_xyxy"],
                }
            )
            target["bbox_xyxy"] = json.loads(json.dumps(source["bbox_xyxy"]))

    if not changes:
        print(f"{args.scene}: metadata already matches the label and index.")
        return 0
    print(f"Scene: {args.scene}")
    print(json.dumps(changes, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run only; pass --apply to write changes.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root.parent / f"{root.name}_metadata_fix_{timestamp}"
    backup_root.mkdir(parents=True)
    shutil.copy2(metadata_path, backup_root / f"{args.scene}.metadata.json")
    shutil.copy2(
        root / "labels" / f"{args.scene}.txt",
        backup_root / f"{args.scene}.label.txt",
    )
    shutil.copy2(root / "scenes.jsonl", backup_root / "scenes.jsonl")
    write_atomic(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"Backup: {backup_root}")
    print("Applied metadata correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
