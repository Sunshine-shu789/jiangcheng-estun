#!/usr/bin/env python3
"""Correct a class id in selected single-object scene annotations."""

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
    parser.add_argument("--old-class-id", type=int, required=True)
    parser.add_argument("--new-class-id", type=int, required=True)
    parser.add_argument("--old-class-name", required=True)
    parser.add_argument("--new-class-name", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, only validate and report.",
    )
    return parser.parse_args()


def target_scenes(
    root: Path,
    old_class_id: int,
    old_class_name: str,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    targets: dict[str, tuple[Path, dict[str, Any]]] = {}
    for metadata_path in sorted((root / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        objects = metadata.get("objects", [])
        if len(objects) != 1:
            continue
        obj = objects[0]
        if (
            obj.get("class_id") == old_class_id
            and obj.get("class_name") == old_class_name
        ):
            scene = str(metadata["scene"])
            if scene in targets:
                raise ValueError(f"Duplicate scene id: {scene}")
            targets[scene] = (metadata_path, metadata)
    return targets


def updated_label(path: Path, old_class_id: int, new_class_id: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError(f"{path}: expected one label line, found {len(lines)}")
    fields = lines[0].split()
    if len(fields) != 5:
        raise ValueError(f"{path}: expected five YOLO fields, found {len(fields)}")
    if fields[0] != str(old_class_id):
        raise ValueError(
            f"{path}: expected class {old_class_id}, found {fields[0]}"
        )
    fields[0] = str(new_class_id)
    return " ".join(fields) + "\n"


def updated_metadata(
    original: dict[str, Any],
    old_class_id: int,
    new_class_id: int,
    new_class_name: str,
) -> tuple[dict[str, Any], str]:
    metadata = json.loads(json.dumps(original))
    objects = metadata.get("objects", [])
    if len(objects) != 1 or objects[0].get("class_id") != old_class_id:
        raise ValueError(f"{metadata.get('scene')}: target object changed")
    objects[0]["class_id"] = new_class_id
    objects[0]["class_name"] = new_class_name
    return metadata, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"


def updated_scenes(
    path: Path,
    scene_ids: set[str],
    old_class_id: int,
    new_class_id: int,
    new_class_name: str,
) -> tuple[str, int]:
    output: list[str] = []
    changed = 0
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        scene = json.loads(line)
        scene_id = str(scene["scene"])
        if scene_id in scene_ids:
            objects = scene.get("objects", [])
            if len(objects) != 1 or objects[0].get("class_id") != old_class_id:
                raise ValueError(
                    f"{path}:{line_number}: target object changed for {scene_id}"
                )
            objects[0]["class_id"] = new_class_id
            objects[0]["class_name"] = new_class_name
            changed += 1
            seen.add(scene_id)
        output.append(json.dumps(scene, ensure_ascii=False))
    missing = scene_ids - seen
    if missing:
        raise ValueError(
            f"{path}: missing target scenes: {', '.join(sorted(missing))}"
        )
    if changed != len(scene_ids):
        raise ValueError(f"{path}: changed {changed}, expected {len(scene_ids)}")
    return "\n".join(output) + "\n", changed


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
    targets = target_scenes(
        root,
        args.old_class_id,
        args.old_class_name,
    )
    if args.expected_count is not None and len(targets) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} target scenes, found {len(targets)}"
        )
    if not targets:
        raise SystemExit("No target scenes found")

    label_changes: dict[Path, str] = {}
    metadata_changes: dict[Path, str] = {}
    for scene_id, (metadata_path, metadata) in targets.items():
        label_path = root / "labels" / f"{scene_id}.txt"
        label_changes[label_path] = updated_label(
            label_path,
            args.old_class_id,
            args.new_class_id,
        )
        _, metadata_text = updated_metadata(
            metadata,
            args.old_class_id,
            args.new_class_id,
            args.new_class_name,
        )
        metadata_changes[metadata_path] = metadata_text

    scenes_path = root / "scenes.jsonl"
    scenes_text, scenes_changed = updated_scenes(
        scenes_path,
        set(targets),
        args.old_class_id,
        args.new_class_id,
        args.new_class_name,
    )

    print(
        f"Validated {len(targets)} scenes: "
        f"{len(label_changes)} labels, "
        f"{len(metadata_changes)} metadata files, "
        f"{scenes_changed} index rows"
    )
    print(
        f"Class change: {args.old_class_id}/{args.old_class_name} -> "
        f"{args.new_class_id}/{args.new_class_name}"
    )
    if not args.apply:
        print("Dry run only; pass --apply to write changes.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root.parent / f"{root.name}_backup_{timestamp}"
    (backup_root / "labels").mkdir(parents=True)
    (backup_root / "metadata").mkdir(parents=True)
    for path in label_changes:
        shutil.copy2(path, backup_root / "labels" / path.name)
    for path in metadata_changes:
        shutil.copy2(path, backup_root / "metadata" / path.name)
    shutil.copy2(scenes_path, backup_root / "scenes.jsonl")
    manifest = {
        "source_root": str(root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "old_class_id": args.old_class_id,
        "old_class_name": args.old_class_name,
        "new_class_id": args.new_class_id,
        "new_class_name": args.new_class_name,
        "scenes": sorted(targets),
    }
    (backup_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for path, content in label_changes.items():
        write_atomic(path, content)
    for path, content in metadata_changes.items():
        write_atomic(path, content)
    write_atomic(scenes_path, scenes_text)

    print(f"Backup: {backup_root}")
    print(f"Applied {len(targets)} scene corrections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
