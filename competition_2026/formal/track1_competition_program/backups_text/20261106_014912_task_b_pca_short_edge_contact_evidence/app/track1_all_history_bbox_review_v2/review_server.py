#!/usr/bin/env python3
"""Standalone offline web reviewer for Track-1 multi-object bbox labels.

This service never opens the camera and never imports ROS.  It only reads the
derived v3 dataset and writes review files below the configured review root.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

CAPTURE_APP = Path(
    "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_capture_web"
)
if str(CAPTURE_APP) not in sys.path:
    sys.path.insert(0, str(CAPTURE_APP))

from capture_store import load_class_catalog  # noqa: E402
from review_core import (  # noqa: E402
    build_review_index,
    export_reviews,
    load_review,
    load_review_index,
    load_yaml,
    review_summary,
    save_review,
)


PROJECT_ROOT = Path(
    "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序"
)
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "track1_all_history_multi_object_review_20261031_v2.yaml"
INDEX_HTML = Path(__file__).with_name("index.html")


def _resolve(root: Path, value: str | Path | None) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("配置路径不能为空")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


class ReviewService:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_yaml(self.config_path)
        paths = dict(self.config.get("paths") or {})
        self.project_root = _resolve(Path.cwd(), paths.get("project_root", PROJECT_ROOT))
        self.source_root = _resolve(self.project_root, paths.get("source_root"))
        self.review_root = _resolve(self.project_root, paths.get("review_root"))
        self.export_root = _resolve(self.project_root, paths.get("export_root"))
        self.class_map = _resolve(self.project_root, paths.get("class_map"))
        self.catalog = load_class_catalog(self.class_map)
        policy = dict(self.config.get("review_policy") or {})
        self.missing_spot_class_ids = sorted(
            {int(value) for value in policy.get("missing_spots_class_ids", [17, 26])}
        )
        self.review_root.mkdir(parents=True, exist_ok=True)
        if not (self.review_root / "index.json").is_file():
            build_review_index(
                source_root=self.source_root,
                class_map_path=self.class_map,
                review_root=self.review_root,
            )
        self.index = load_review_index(self.review_root)
        self.by_id = {
            str(item["review_id"]): item for item in self.index["items"]
        }
        if not INDEX_HTML.is_file():
            raise ValueError(f"复核网页不存在：{INDEX_HTML}")

    def item_payload(self, review_id: str) -> dict[str, Any]:
        item = self.by_id.get(review_id)
        if item is None:
            raise KeyError(f"复核项不存在：{review_id}")
        review = load_review(self.review_root, review_id)
        candidate_path = self.review_root / "candidates" / f"{review_id}.json"
        candidate_payload: dict[str, Any] | None = None
        if candidate_path.is_file():
            try:
                loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    candidate_payload = loaded
            except (OSError, json.JSONDecodeError):
                candidate_payload = None
        return {
            "ok": True,
            "item": item,
            "review": review,
            "status": str(review.get("status")) if review else "pending",
            "image_url": f"/api/image?id={review_id}",
            "overlay_url": (
                f"/api/review-overlay?id={review_id}"
                if (self.review_root / "overlays" / f"{review_id}.jpg").is_file()
                else None
            ),
            "candidates": candidate_payload,
            "candidate_file_exists": candidate_payload is not None,
        }

    def bootstrap(self) -> dict[str, Any]:
        summary = review_summary(
            review_root=self.review_root,
            index=self.index,
        )
        return {
            "ok": True,
            "title": self.config.get("title", "赛道一多物品 bbox 复核"),
            "catalog": self.catalog.to_api(),
            "summary": summary,
            "index": {
                "item_count": self.index["item_count"],
                "split_counts": self.index["split_counts"],
                "selection_policy": self.index["selection_policy"],
                "multi_object_policy": self.index["multi_object_policy"],
                "source_root": self.index["source_root"],
            },
            "paths": {
                "source_root": str(self.source_root),
                "review_root": str(self.review_root),
                "export_root": str(self.export_root),
            },
        }

    def export(self) -> dict[str, Any]:
        return export_reviews(
            source_root=self.source_root,
            review_root=self.review_root,
            export_root=self.export_root,
            index=self.index,
            catalog=self.catalog,
            missing_spot_class_ids=self.missing_spot_class_ids,
        )


class ReviewRequestHandler(BaseHTTPRequestHandler):
    service: ReviewService

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.client_address[0], self.log_date_time_string(), fmt % args)
        )

    def send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        status: int = HTTPStatus.OK,
        cache: str = "no-store",
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Content-Length 无效") from error
        if length <= 0 or length > 2 * 1024 * 1024:
            raise ValueError("请求体为空或过大")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON 根节点必须是对象")
        return payload

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        try:
            if parsed.path in {"", "/", "/index.html"}:
                body = INDEX_HTML.read_bytes()
                self.send_bytes(
                    body,
                    content_type="text/html; charset=utf-8",
                    cache="no-store",
                )
                return
            if parsed.path == "/status.json":
                summary = review_summary(
                    review_root=self.service.review_root,
                    index=self.service.index,
                )
                self.send_json(
                    {
                        "ok": True,
                        "total": summary["total"],
                        "reviewed": summary["reviewed"],
                        "pending": summary["pending"],
                        "completion": summary["completion"],
                    }
                )
                return
            if parsed.path == "/api/bootstrap":
                self.send_json(self.service.bootstrap())
                return
            if parsed.path == "/api/summary":
                self.send_json(
                    {
                        "ok": True,
                        "summary": review_summary(
                            review_root=self.service.review_root,
                            index=self.service.index,
                        ),
                    }
                )
                return
            if parsed.path == "/api/item":
                values = parse_qs(parsed.query).get("id", [])
                if not values:
                    raise ValueError("缺少 id")
                self.send_json(self.service.item_payload(values[0]))
                return
            if parsed.path == "/api/image":
                values = parse_qs(parsed.query).get("id", [])
                if not values:
                    raise ValueError("缺少 id")
                item = self.service.by_id.get(values[0])
                if item is None:
                    raise KeyError(f"复核项不存在：{values[0]}")
                image_path = self.service.source_root / str(item["source_rgb"])
                if not image_path.is_file():
                    raise KeyError(f"图片不存在：{image_path}")
                content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                self.send_bytes(
                    image_path.read_bytes(),
                    content_type=content_type,
                    cache="private, max-age=3600",
                )
                return
            if parsed.path == "/api/review-overlay":
                values = parse_qs(parsed.query).get("id", [])
                if not values:
                    raise ValueError("缺少 id")
                overlay = (
                    self.service.review_root
                    / "overlays"
                    / f"{values[0]}.jpg"
                )
                if not overlay.is_file():
                    raise KeyError("尚无复核预览")
                self.send_bytes(
                    overlay.read_bytes(),
                    content_type="image/jpeg",
                    cache="no-store",
                )
                return
            raise KeyError(f"未知路径：{parsed.path}")
        except KeyError as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.NOT_FOUND,
            )
        except (ValueError, OSError, json.JSONDecodeError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as error:  # Keep the review UI alive on unexpected data.
            traceback.print_exc()
            self.send_json(
                {"ok": False, "error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/save":
                review_id = str(payload.get("review_id") or "")
                item = self.service.by_id.get(review_id)
                if item is None:
                    raise KeyError(f"复核项不存在：{review_id}")
                review = save_review(
                    review_root=self.service.review_root,
                    source_root=self.service.source_root,
                    item=item,
                    payload=payload,
                    catalog=self.service.catalog,
                )
                summary = review_summary(
                    review_root=self.service.review_root,
                    index=self.service.index,
                )
                self.send_json(
                    {
                        "ok": True,
                        "review": review,
                        "summary": summary,
                    }
                )
                return
            if parsed.path == "/api/export":
                self.send_json(self.service.export())
                return
            raise KeyError(f"未知路径：{parsed.path}")
        except KeyError as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.NOT_FOUND,
            )
        except (ValueError, OSError, json.JSONDecodeError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as error:
            traceback.print_exc()
            self.send_json(
                {"ok": False, "error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="赛道一全历史多物品 bbox 复核网页 v2（不使用相机、不连接机器人）"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9131)
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        service = ReviewService(Path(args.config))
    except (ValueError, OSError) as error:
        print(f"多物品复核服务配置错误：{error}", file=sys.stderr)
        return 2
    ReviewRequestHandler.service = service
    try:
        server = ThreadingHTTPServer(
            (args.host, args.port),
            ReviewRequestHandler,
        )
    except OSError as error:
        print(f"端口启动失败：{args.host}:{args.port}：{error}", file=sys.stderr)
        return 2
    url = f"http://{args.host}:{args.port}/"
    print(f"Track 1 multi-object bbox review: {url}", file=sys.stderr)
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
