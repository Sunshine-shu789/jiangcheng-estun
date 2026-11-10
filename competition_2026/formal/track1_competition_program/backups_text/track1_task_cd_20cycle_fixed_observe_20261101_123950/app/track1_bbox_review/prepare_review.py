#!/usr/bin/env python3
"""Prepare one representative frame per burst and precompute bbox candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from review_candidates import prepare_all_candidates
from review_core import build_review_index, load_yaml


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成赛道一已有照片复核索引和候选框"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="覆盖已有候选缓存")
    parser.add_argument("--limit", type=int, help="只处理前 N 张，用于试跑")
    parser.add_argument("--no-world", action="store_true", help="只生成深度候选")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    paths = dict(config["paths"])
    project_root = _resolve(Path.cwd(), paths["project_root"])
    source_root = _resolve(project_root, paths["source_root"])
    review_root = _resolve(project_root, paths["review_root"])
    class_map = _resolve(project_root, paths["class_map"])
    index = build_review_index(
        source_root=source_root,
        class_map_path=class_map,
        review_root=review_root,
    )
    print(
        f"索引完成：{index['item_count']} 个 split_unit，"
        f"train={index['split_counts'].get('train', 0)}，"
        f"val={index['split_counts'].get('val', 0)}"
    )
    if args.index_only:
        print(json.dumps(index, ensure_ascii=False, indent=2))
        return 0
    report = prepare_all_candidates(
        config_path=config_path,
        force=bool(args.force),
        limit=args.limit,
        with_world=not bool(args.no_world),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
