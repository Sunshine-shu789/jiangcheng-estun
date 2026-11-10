#!/usr/bin/env python3
"""真机采图：从若干 ROS2 彩色图像话题各抓 N 张，存成 PNG + 元数据 JSON。

典型用法（三台相机同时采）：
    python real_capture.py \
        --topic /cam_top/color/image_raw --topic /cam_front/color/image_raw \
        --topic /cam_side/color/image_raw \
        --out /media/sunshine/新加卷/taskA/real_raw --count 60 --interval 2.0

要点：
- 只订阅彩色流（RGB），不使用深度/红外；采图前请在相机驱动里关掉 IR 投射器。
- 每个话题抓 `--count` 张，拍完把物体换姿态/换光照再抓下一轮。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def decode(msg) -> "np.ndarray":
    """把 sensor_msgs/Image 解成 HxWx3 的 RGB/RGBA 数组（不依赖 cv_bridge）。"""
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, w, 3)
        return img[:, :, ::-1] if enc == "bgr8" else img
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(h, w, 4)[:, :, :3]
        return img[:, :, ::-1] if enc == "bgra8" else img
    raise ValueError(f"暂不支持的编码：{msg.encoding}（请把相机设成 rgb8/bgr8）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", action="append", required=True, help="可重复，按顺序对应 --tag")
    parser.add_argument("--tag", action="append", default=None, help="相机短名，如 top/front/side")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--interval", type=float, default=2.0, help="同一相机两次抓拍间隔（秒）")
    parser.add_argument("--timeout", type=float, default=10.0, help="等首帧超时（秒）")
    args = parser.parse_args()

    import cv2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    tags = args.tag or [f"cam{i}" for i in range(len(args.topic))]
    if len(tags) != len(args.topic):
        raise SystemExit("--tag 数量必须和 --topic 相同")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Node("taskA_capture")
    latest: dict[str, object] = {}
    for topic in args.topic:
        node.create_subscription(
            Image, topic, lambda msg, t=topic: latest.__setitem__(t, msg), 10
        )

    kept: dict[str, int] = {t: 0 for t in args.topic}
    last_save: dict[str, float] = {t: 0.0 for t in args.topic}
    meta = []
    deadline = time.time() + args.timeout
    while any(v < args.count for v in kept.values()):
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.time()
        if not latest and now > deadline:
            raise SystemExit(f"等待图像超时，收到 0 帧。检查话题名：{args.topic}")
        for topic, tag in zip(args.topic, tags):
            msg = latest.get(topic)
            if msg is None or kept[topic] >= args.count:
                continue
            if now - last_save[topic] < args.interval:
                continue
            img = decode(msg)
            stamp = f"{now:.3f}"
            name = f"{tag}_{stamp}_{kept[topic]:03d}.png"
            cv2.imwrite(str(out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            meta.append(
                {
                    "file": name,
                    "camera": tag,
                    "topic": topic,
                    "width": msg.width,
                    "height": msg.height,
                    "encoding": msg.encoding,
                    "ros_stamp": f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
                }
            )
            kept[topic] += 1
            last_save[topic] = now
            print(f"[{tag}] {kept[topic]}/{args.count} -> {name}", flush=True)

    (out / "capture_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    node.destroy_node()
    rclpy.shutdown()
    print(f"完成：{sum(kept.values())} 张 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
