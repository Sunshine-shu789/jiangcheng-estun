#!/usr/bin/env python3
"""Capture one aligned D405 frame without touching the robot.

The saved ``*_app`` image/intrinsics match the 180-degree view rotation used by
the competition vision stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--serial", default="")
    args = parser.parse_args()

    import pyrealsense2 as rs

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.depth,
        args.width,
        args.height,
        rs.format.z16,
        args.fps,
    )
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
    except RuntimeError:
        depth_scale = 0.001
    align = rs.align(rs.stream.color)
    try:
        for _ in range(max(0, args.warmup)):
            pipeline.wait_for_frames(3000)
        frames = pipeline.wait_for_frames(5000)
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense returned an incomplete frame set")

        color = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data()).copy()
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        raw_k = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "depth_scale_m": depth_scale,
        }
        app_k = dict(raw_k)
        app_k["cx"] = float((intr.width - 1) - intr.ppx)
        app_k["cy"] = float((intr.height - 1) - intr.ppy)

        color_app = cv2.rotate(color, cv2.ROTATE_180)
        depth_app = cv2.rotate(depth_raw, cv2.ROTATE_180)
        depth_m = depth_raw.astype(np.float32) * depth_scale

        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = out / f"board_scene_{stamp}"
        cv2.imwrite(str(prefix.with_name(prefix.name + "_raw.png")), color)
        cv2.imwrite(str(prefix.with_name(prefix.name + "_app.png")), color_app)
        np.save(prefix.with_name(prefix.name + "_depth_raw.npy"), depth_raw)
        np.save(prefix.with_name(prefix.name + "_depth_m.npy"), depth_m)
        payload = {
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "serial": args.serial,
            "raw_intrinsics": raw_k,
            "app_intrinsics": app_k,
            "raw_image": prefix.with_name(prefix.name + "_raw.png").name,
            "app_image": prefix.with_name(prefix.name + "_app.png").name,
            "depth_raw": prefix.with_name(prefix.name + "_depth_raw.npy").name,
            "depth_m": prefix.with_name(prefix.name + "_depth_m.npy").name,
            "rotation_deg": 180,
        }
        with open(prefix.with_name(prefix.name + "_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(prefix.with_name(prefix.name + "_app.png"))
        print(prefix.with_name(prefix.name + "_meta.json"))
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
