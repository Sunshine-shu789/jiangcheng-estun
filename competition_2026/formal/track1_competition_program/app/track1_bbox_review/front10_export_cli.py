#!/usr/bin/env python3
"""CLI export for the strictly front-10 self-collected review set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from capture_store import load_class_catalog
from front10_review_core import (
    FRONT10_IDS,
    TRIPLE_CAN_IDS,
    export_front10_reviews,
)
from review_core import load_review_index, load_yaml


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出赛道一前10类自采 YOLO 标签")
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
    allowed = tuple(
        int(value)
        for value in policy.get("front10_only_positive_classes", FRONT10_IDS)
    )
    triple_ids = tuple(
        int(value)
        for value in policy.get("triple_can_required_classes", TRIPLE_CAN_IDS)
    )
    if allowed != FRONT10_IDS:
        raise ValueError(f"前10类配置无效：{allowed}")
    if triple_ids != TRIPLE_CAN_IDS:
        raise ValueError(f"三罐头类别配置无效：{triple_ids}")
    expected_triple_splits = {
        str(key): str(value)
        for key, value in dict(policy.get("triple_can_layout_splits") or {}).items()
    }
    result = export_front10_reviews(
        source_root=source_root,
        review_root=review_root,
        export_root=export_root,
        index=index,
        catalog=catalog,
        path_catalog_id="front10",
        expected_triple_splits=expected_triple_splits,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
