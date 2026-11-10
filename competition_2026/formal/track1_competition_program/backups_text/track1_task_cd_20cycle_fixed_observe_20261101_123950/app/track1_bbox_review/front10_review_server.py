#!/usr/bin/env python3
"""Dedicated front-10 review server with a filtered class catalog."""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from review_server import ReviewRequestHandler, ReviewService
from front10_review_core import (
    FRONT10_IDS,
    TRIPLE_CAN_IDS,
    export_front10_reviews,
    filter_front10_catalog,
    load_review_plan,
)


PROJECT_ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
DEFAULT_CONFIG = PROJECT_ROOT / "config/track1_front10_bbox_review_v1.yaml"


class Front10ReviewService(ReviewService):
    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path)
        policy = dict(self.config.get("review_policy") or {})
        allowed = tuple(
            int(value)
            for value in policy.get("front10_only_positive_classes", FRONT10_IDS)
        )
        self.catalog = filter_front10_catalog(self.catalog, allowed)
        self.triple_class_ids = tuple(
            int(value)
            for value in policy.get("triple_can_required_classes", TRIPLE_CAN_IDS)
        )
        self.triple_splits = {
            str(key): str(value)
            for key, value in dict(policy.get("triple_can_layout_splits") or {}).items()
        }
        plan = load_review_plan(self.source_root)
        self.triple_by_id = {
            str(item["split_unit"]): item
            for item in list(plan.get("triple_can_units") or [])
        }
        for item in self.index["items"]:
            hint_class_id = int(item["hint_class_id"])
            if hint_class_id not in FRONT10_IDS:
                raise ValueError(
                    f"前10类复核索引混入官方类别 {hint_class_id}：{item['review_id']}"
                )

    def item_payload(self, review_id: str) -> dict[str, Any]:
        payload = super().item_payload(review_id)
        triple = self.triple_by_id.get(review_id)
        if triple is not None:
            payload["review_requirements"] = {
                "multi_instance": True,
                "required_class_ids": list(self.triple_class_ids),
                "note": "三个罐头分别标 2、5、10；每个实例一个框，不能画整组大框。",
            }
        return payload

    def export(self) -> dict[str, Any]:
        return export_front10_reviews(
            source_root=self.source_root,
            review_root=self.review_root,
            export_root=self.export_root,
            index=self.index,
            catalog=self.catalog,
            path_catalog_id="front10",
            expected_triple_splits=self.triple_splits,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="赛道一前10类专用多物品 bbox 复核网页（不使用相机、不连接机器人）"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9121)
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        service = Front10ReviewService(Path(args.config))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"前10类复核服务配置错误：{error}", file=sys.stderr)
        return 2
    ReviewRequestHandler.service = service
    try:
        server = ThreadingHTTPServer((args.host, args.port), ReviewRequestHandler)
    except OSError as error:
        print(f"端口启动失败：{args.host}:{args.port}：{error}", file=sys.stderr)
        return 2
    url = f"http://{args.host}:{args.port}/"
    print(f"Track 1 front-10 bbox review: {url}", file=sys.stderr)
    print(f"source: {service.source_root}", file=sys.stderr)
    print(f"review: {service.review_root}", file=sys.stderr)
    print("camera: closed; robot commands: disabled", file=sys.stderr)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
