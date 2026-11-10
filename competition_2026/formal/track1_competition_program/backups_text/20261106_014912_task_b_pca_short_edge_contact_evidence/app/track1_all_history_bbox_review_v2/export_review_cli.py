#!/usr/bin/env python3
"""CLI export for reviewed Track-1 multi-object bbox labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from capture_store import load_class_catalog
from review_core import export_reviews, load_review_index, load_yaml


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出赛道一全历史多物品复核 v2 YOLO 标签")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    paths = dict(config["paths"])
    project_root = _resolve(Path.cwd(), paths["project_root"])
    source_root = _resolve(project_root, paths["source_root"])
    review_root = _resolve(project_root, paths["review_root"])
    export_root = _resolve(project_root, paths["export_root"])
    class_map = _resolve(project_root, paths["class_map"])
    index = load_review_index(review_root)
    catalog = load_class_catalog(class_map)
    policy = dict(config.get("review_policy") or {})
    missing_spot_class_ids = sorted(
        {int(value) for value in policy.get("missing_spots_class_ids", [])}
    )
    expected_known_class_count = int(
        policy.get("expected_known_class_count", 40)
    )
    result = export_reviews(
        source_root=source_root,
        review_root=review_root,
        export_root=export_root,
        index=index,
        catalog=catalog,
        missing_spot_class_ids=missing_spot_class_ids,
        expected_known_class_count=expected_known_class_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
