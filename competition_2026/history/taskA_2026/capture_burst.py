#!/usr/bin/env python3
"""Request a burst of raw captures from the running camera preview.

--class-id 5        采集 6-Foam_Brick（服务端类别编号 5）
--class-id null     采集空背景（保存到 empty_background/）
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


BACKGROUND_TOKENS = {"none", "null", "empty", "background", "-1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9100/api/capture")
    parser.add_argument(
        "--class-id",
        required=True,
        help="类别编号；空背景传 null/none/background",
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.35)
    parser.add_argument("--start-delay", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count 必须大于 0")
    if args.interval < 0 or args.start_delay < 0:
        raise SystemExit("--interval 和 --start-delay 不能为负数")

    token = str(args.class_id).strip().lower()
    if token in BACKGROUND_TOKENS:
        class_id: int | None = None
    else:
        try:
            class_id = int(token)
        except ValueError:
            raise SystemExit(f"无法解析 --class-id：{args.class_id}") from None

    if args.start_delay:
        time.sleep(args.start_delay)

    for index in range(args.count):
        body = json.dumps({"class_id": class_id}).encode("utf-8")
        request = urllib.request.Request(
            args.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "采图失败"))
        print(
            f"{index + 1}/{args.count}  {result.get('class_name', 'empty_background')}  "
            f"{result['path']}",
            flush=True,
        )
        if index + 1 < args.count:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
