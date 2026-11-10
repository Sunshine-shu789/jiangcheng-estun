#!/usr/bin/env python3
"""画面变化触发式采集：只在画面明显变化时按快门。

用途：人手慢慢转动/移动物品，脚本自动把不同姿态存下来，避免连拍出一堆重复帧。

    auto_capture.py --class-id 0 --duration 120

依赖正在运行的 live_camera_preview.py（默认 127.0.0.1:9100）。
只采图，不连接、不控制任何机械臂。
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request

import numpy as np
from PIL import Image


def fetch(url: str, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def signature(jpeg: bytes, top_crop: int = 26, size: tuple[int, int] = (96, 72)) -> np.ndarray:
    image = Image.open(io.BytesIO(jpeg)).convert("L")
    if top_crop:
        image = image.crop((0, top_crop, image.width, image.height))
    image = image.resize(size, Image.BILINEAR)
    return np.asarray(image).astype(np.float32)


BACKGROUND_TOKENS = {"none", "null", "empty", "background", "-1"}


def parse_class_id(token: str) -> int | None:
    if str(token).strip().lower() in BACKGROUND_TOKENS:
        return None
    return int(token)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:9100")
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--poll", type=float, default=0.4)
    parser.add_argument("--min-diff", type=float, default=7.0, help="触发阈值（灰度均值差）")
    parser.add_argument("--min-gap", type=float, default=1.0, help="两次快门最小间隔（秒）")
    parser.add_argument("--max-shots", type=int, default=60)
    args = parser.parse_args()

    class_id = parse_class_id(args.class_id)
    base = args.base.rstrip("/")
    snap_url = f"{base}/snapshot.jpg"
    cap_url = f"{base}/api/capture"

    deadline = time.time() + args.duration
    last_saved_at = 0.0
    reference: np.ndarray | None = None
    saved = 0
    triggers = 0

    while time.time() < deadline and saved < args.max_shots:
        try:
            jpeg = fetch(snap_url)
        except Exception as error:  # noqa: BLE001
            print(f"取画面失败：{error}", flush=True)
            time.sleep(1.0)
            continue

        current = signature(jpeg)
        now = time.time()

        if reference is None:
            reference = current
            time.sleep(args.poll)
            continue

        diff = float(np.abs(current - reference).mean())
        fresh = (now - last_saved_at) >= args.min_gap

        if diff >= args.min_diff and fresh:
            triggers += 1
            body = json.dumps({"class_id": class_id}).encode("utf-8")
            request = urllib.request.Request(
                cap_url, data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=6) as response:
                    result = json.load(response)
            except Exception as error:  # noqa: BLE001
                print(f"快门失败：{error}", flush=True)
                time.sleep(0.5)
                continue
            if result.get("ok"):
                saved += 1
                last_saved_at = now
                reference = current
                print(
                    f"[{saved:3d}] diff={diff:5.1f}  {result.get('class_name')}  "
                    f"{result['path'].rsplit('/', 1)[-1]}",
                    flush=True,
                )
            else:
                print(f"快门被拒：{result.get('error')}", flush=True)
        else:
            # 慢慢把参考帧跟上，避免缓慢漂移一直不触发
            reference = 0.7 * reference + 0.3 * current

        time.sleep(args.poll)

    print(f"\n完成：触发 {triggers} 次，实际保存 {saved} 张", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
