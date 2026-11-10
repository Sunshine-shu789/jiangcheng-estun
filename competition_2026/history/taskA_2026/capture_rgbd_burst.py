#!/usr/bin/env python3
"""Capture aligned RealSense RGB-D frames for Task A.

Each frame is stored as an 8-bit BGR JPEG plus a 16-bit depth PNG in depth
units.  Camera intrinsics and the depth scale are written once per session.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def intrinsics_to_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.interval < 0:
        raise SystemExit("--interval cannot be negative")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    output = Path(args.out).resolve()
    rgb_dir = output / "rgb"
    depth_dir = output / "depth"
    depth_vis_dir = output / "depth_vis"
    for directory in (rgb_dir, depth_dir, depth_vis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    config.enable_stream(
        rs.stream.depth,
        args.width,
        args.height,
        rs.format.z16,
        args.fps,
    )

    pipeline = rs.pipeline()
    align = rs.align(rs.stream.color)
    metadata_rows: list[dict[str, Any]] = []
    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        session = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device_name": device.get_info(rs.camera_info.name),
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "firmware_version": device.get_info(rs.camera_info.firmware_version),
            "depth_scale_m_per_unit": depth_scale,
            "rotation_degrees": args.rotate,
            "color_intrinsics": intrinsics_to_dict(color_profile.get_intrinsics()),
            "depth_intrinsics": intrinsics_to_dict(depth_profile.get_intrinsics()),
        }
        (output / "session.json").write_text(
            json.dumps(session, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        for _ in range(args.warmup):
            pipeline.wait_for_frames(timeout_ms=5000)

        for index in range(args.count):
            frames = align.process(
                pipeline.wait_for_frames(timeout_ms=10000)
            )
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("aligned color/depth frame is unavailable")

            color = rotate_frame(
                np.asanyarray(color_frame.get_data()),
                args.rotate,
            )
            depth = rotate_frame(
                np.asanyarray(depth_frame.get_data()),
                args.rotate,
            )
            valid_depth = depth[depth > 0]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stem = f"rgbd_{stamp}"
            rgb_path = rgb_dir / f"{stem}.jpg"
            depth_path = depth_dir / f"{stem}.png"
            depth_vis_path = depth_vis_dir / f"{stem}.jpg"

            if not cv2.imwrite(
                str(rgb_path),
                color,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            ):
                raise RuntimeError(f"cannot write {rgb_path}")
            if not cv2.imwrite(str(depth_path), depth):
                raise RuntimeError(f"cannot write {depth_path}")

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03),
                cv2.COLORMAP_TURBO,
            )
            cv2.imwrite(
                str(depth_vis_path),
                depth_vis,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )

            metadata_rows.append(
                {
                    "stem": stem,
                    "rgb": str(rgb_path),
                    "depth": str(depth_path),
                    "depth_visualization": str(depth_vis_path),
                    "depth_scale_m_per_unit": depth_scale,
                    "valid_depth_pixels": int(valid_depth.size),
                    "depth_min_m": (
                        float(valid_depth.min() * depth_scale)
                        if valid_depth.size
                        else None
                    ),
                    "depth_max_m": (
                        float(valid_depth.max() * depth_scale)
                        if valid_depth.size
                        else None
                    ),
                }
            )
            print(f"{index + 1}/{args.count} {rgb_path}", flush=True)
            if index + 1 < args.count:
                time.sleep(args.interval)
    finally:
        pipeline.stop()

    with (output / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
