#!/usr/bin/env python3
"""Export reviewed RGB-D scenes as a scene-isolated YOLO dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


CLASS_NAMES = [
    "1-Cheez-it",
    "2-Starkist_Tuna",
    "3-Scissors",
    "4-Frenchs_Mustard",
    "5-Tomato_Soup",
    "6-Foam_Brick",
    "7-Clamp",
    "8-Plastic_Banana",
    "9-Mug",
    "10-meat_can",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_scenes(path: Path) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        scenes.append(payload)
    if not scenes:
        raise ValueError(f"{path}: no scenes found")
    scene_ids = [str(scene.get("scene", "")) for scene in scenes]
    if not all(scene_ids):
        raise ValueError(f"{path}: scene id is missing")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"{path}: duplicate scene id")
    return scenes


def validate_label(path: Path, expected_classes: list[int]) -> None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) != len(expected_classes):
        raise ValueError(
            f"{path}: expected {len(expected_classes)} boxes, found {len(lines)}"
        )

    for index, (line, expected_class) in enumerate(
        zip(lines, expected_classes, strict=True),
        start=1,
    ):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}: box {index} does not have 5 fields")
        try:
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}: box {index} is not numeric") from exc
        if class_id != expected_class:
            raise ValueError(
                f"{path}: box {index} class {class_id} != metadata {expected_class}"
            )
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{path}: box {index} class id is out of range")
        if not all(math.isfinite(value) for value in (cx, cy, width, height)):
            raise ValueError(f"{path}: box {index} has a non-finite value")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise ValueError(f"{path}: box {index} center is outside the image")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{path}: box {index} size is outside (0, 1]")
        if cx - width / 2 < -1e-6 or cx + width / 2 > 1.0 + 1e-6:
            raise ValueError(f"{path}: box {index} crosses the horizontal edge")
        if cy - height / 2 < -1e-6 or cy + height / 2 > 1.0 + 1e-6:
            raise ValueError(f"{path}: box {index} crosses the vertical edge")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")


def symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    os.symlink(source.resolve(), destination)


def validate_split(
    scenes: list[dict[str, Any]],
    split_manifest: dict[str, Any],
) -> tuple[set[str], set[str]]:
    all_scenes = {str(scene["scene"]) for scene in scenes}
    val_scenes = {str(scene) for scene in split_manifest["val"]["scenes"]}
    train_scenes = {str(scene) for scene in split_manifest["train"]["scenes"]}
    if val_scenes & train_scenes:
        raise ValueError("train and val scene sets overlap")
    if val_scenes | train_scenes != all_scenes:
        missing = sorted(all_scenes - (val_scenes | train_scenes))
        extra = sorted((val_scenes | train_scenes) - all_scenes)
        raise ValueError(f"split does not cover all scenes; missing={missing}, extra={extra}")
    return train_scenes, val_scenes


def class_counts(
    scenes: list[dict[str, Any]],
    selected: set[str],
) -> Counter[int]:
    return Counter(
        int(obj["class_id"])
        for scene in scenes
        if str(scene["scene"]) in selected
        for obj in scene["objects"]
    )


def write_data_yaml(out: Path) -> None:
    lines = [
        f"path: {out}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
        *[f"- {name}" for name in CLASS_NAMES],
        "",
    ]
    (out / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    scene_root = Path(args.scene_root).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")

    scenes = load_scenes(scene_root / "scenes.jsonl")
    split_manifest = load_json(Path(args.split_manifest).resolve())
    train_scenes, val_scenes = validate_split(scenes, split_manifest)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for scene in scenes:
        scene_id = str(scene["scene"])
        split = "val" if scene_id in val_scenes else "train"
        image_path = Path(str(scene["image"]))
        label_path = Path(str(scene["label"]))
        depth_path = Path(str(scene["depth"]))
        metadata_path = scene_root / "metadata" / f"{scene_id}.json"
        overlay_path = scene_root / "metadata" / f"{scene_id}_overlay.jpg"

        require_file(image_path, "image")
        require_file(label_path, "label")
        require_file(depth_path, "depth")
        require_file(metadata_path, "metadata")
        require_file(overlay_path, "overlay")
        expected_classes = [int(obj["class_id"]) for obj in scene["objects"]]
        validate_label(label_path, expected_classes)

        symlink(image_path, out / "images" / split / image_path.name)
        symlink(label_path, out / "labels" / split / label_path.name)
        rows.append(
            {
                "split": split,
                "scene": scene_id,
                "image": image_path.name,
                "label": label_path.name,
                "instances": len(expected_classes),
                "classes": " ".join(
                    CLASS_NAMES[class_id] for class_id in expected_classes
                ),
            }
        )

    (out / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "scene",
                "image",
                "label",
                "instances",
                "classes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_data_yaml(out)

    train_counts = class_counts(scenes, train_scenes)
    val_counts = class_counts(scenes, val_scenes)
    report = [
        "# Random real scene export",
        "",
        f"- Source: `{scene_root}`",
        f"- Train: {len(train_scenes)} scenes / {sum(train_counts.values())} instances",
        f"- Val: {len(val_scenes)} scenes / {sum(val_counts.values())} instances",
        "- Split policy: source scene holdout",
        "",
        "## Class counts",
        "",
        "| class_id | class_name | train | val |",
        "| ---: | --- | ---: | ---: |",
    ]
    for class_id, class_name in enumerate(CLASS_NAMES):
        report.append(
            f"| {class_id} | {class_name} | {train_counts[class_id]} | "
            f"{val_counts[class_id]} |"
        )
    report.append("")
    (out / "export_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"exported scene-isolated YOLO dataset -> {out}")
    print(
        f"  train: {len(train_scenes)} scenes / {sum(train_counts.values())} instances"
    )
    print(f"  val: {len(val_scenes)} scenes / {sum(val_counts.values())} instances")
    for class_id, class_name in enumerate(CLASS_NAMES):
        if train_counts[class_id] or val_counts[class_id]:
            print(
                f"  class {class_id} {class_name}: "
                f"train={train_counts[class_id]} val={val_counts[class_id]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
