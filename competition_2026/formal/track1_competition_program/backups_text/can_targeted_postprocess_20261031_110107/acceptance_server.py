#!/usr/bin/env python3
"""Track 1 final 40-class visual acceptance service.

This process only reads the D405 camera and runs YOLO. It never imports ROS,
moves the robot, or controls the gripper.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import hashlib
import json
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from ultralytics import YOLO


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "track1_acceptance_40class.yaml"
INDEX_HTML = (APP_ROOT / "index.html").read_text(encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("验收配置文件不是 YAML 对象")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def class_color(class_id: int) -> tuple[int, int, int]:
    hue = int((class_id * 37) % 180)
    hsv = np.uint8([[[hue, 205, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class ModelInfo:
    def __init__(
        self,
        weights: Path,
        names: list[str],
        expected_names: list[str],
        expected_class_ids: list[int],
        reserved_empty_class_ids: list[int],
        sha256: str,
    ) -> None:
        self.weights = weights
        self.names = names
        self.expected_names = expected_names
        self.expected_class_ids = expected_class_ids
        self.reserved_empty_class_ids = reserved_empty_class_ids
        self.sha256 = sha256

    def to_api(self) -> dict[str, Any]:
        classes = []
        expected = set(self.expected_class_ids)
        reserved = set(self.reserved_empty_class_ids)
        for class_id, name in enumerate(self.names):
            classes.append(
                {
                    "id": class_id,
                    "official_id": class_id + 1,
                    "name": name,
                    "expected": class_id in expected,
                    "reserved_empty": class_id in reserved,
                    "front_ten": class_id < 10,
                }
            )
        return {
            "weights": str(self.weights),
            "sha256": self.sha256,
            "nc": len(self.names),
            "classes": classes,
            "expected_class_ids": sorted(expected),
            "reserved_empty_class_ids": sorted(reserved),
        }


class AcceptanceState:
    def __init__(self, model_info: ModelInfo, conf: float, iou: float) -> None:
        self.model_info = model_info
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.settings_lock = threading.Lock()
        self.conf = float(conf)
        self.iou = float(iou)
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.status: dict[str, Any] = {
            "ok": True,
            "phase": "starting",
            "error": "",
            "cameraSerial": "",
            "cameraModel": "",
            "frameWidth": 0,
            "frameHeight": 0,
            "frameSequence": 0,
            "fps": 0.0,
            "inferenceMs": 0.0,
            "inferenceAgeMs": 0.0,
            "conf": self.conf,
            "iou": self.iou,
            "detections": [],
            "model": {
                "weights": str(model_info.weights),
                "sha256": model_info.sha256,
                "nc": len(model_info.names),
            },
            "startedAt": self.started_at,
            "robotControl": False,
            "gripperControl": False,
        }

    def get_settings(self) -> tuple[float, float]:
        with self.settings_lock:
            return self.conf, self.iou

    def set_settings(self, conf: float | None = None) -> tuple[float, float]:
        with self.settings_lock:
            if conf is not None:
                if not 0.01 <= conf <= 0.95:
                    raise ValueError("置信度阈值必须在 0.01 到 0.95 之间")
                self.conf = float(conf)
            return self.conf, self.iou

    def set_phase(self, phase: str) -> None:
        with self.condition:
            self.status["phase"] = phase
            self.condition.notify_all()

    def set_camera(self, serial: str, model: str) -> None:
        with self.condition:
            self.status["cameraSerial"] = serial
            self.status["cameraModel"] = model
            self.condition.notify_all()

    def update_frame(
        self,
        jpeg: bytes,
        width: int,
        height: int,
        fps: float,
        detections: list[dict[str, Any]],
        inference_ms: float,
        inference_at: float,
        conf: float,
    ) -> None:
        with self.condition:
            self.sequence += 1
            self.jpeg = jpeg
            self.status.update(
                {
                    "ok": True,
                    "phase": "running",
                    "error": "",
                    "frameWidth": width,
                    "frameHeight": height,
                    "frameSequence": self.sequence,
                    "fps": round(fps, 1),
                    "inferenceMs": round(inference_ms, 1),
                    "inferenceAgeMs": round(
                        max(0.0, (time.monotonic() - inference_at) * 1000.0),
                        0,
                    ),
                    "conf": round(conf, 3),
                    "detections": detections,
                }
            )
            self.condition.notify_all()

    def set_error(self, message: str) -> None:
        with self.condition:
            self.status["ok"] = False
            self.status["phase"] = "error"
            self.status["error"] = message
            self.condition.notify_all()

    def snapshot(self) -> tuple[dict[str, Any], bytes | None]:
        with self.condition:
            return json.loads(json.dumps(self.status)), self.jpeg

    def wait_for_jpeg(self, after_sequence: int, timeout: float = 2.0) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= after_sequence or self.jpeg is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(timeout=min(remaining, 0.25))
            return self.sequence, self.jpeg


class AcceptanceWorker(threading.Thread):
    def __init__(
        self,
        args: argparse.Namespace,
        model_info: ModelInfo,
        state: AcceptanceState,
    ) -> None:
        super().__init__(name="track1-40class-acceptance", daemon=True)
        self.args = args
        self.model_info = model_info
        self.state = state
        self.stop_event = threading.Event()
        self.pipeline: rs.pipeline | None = None
        self.model = YOLO(str(model_info.weights))

    def stop(self) -> None:
        self.stop_event.set()

    def _open_camera(self) -> None:
        config = rs.config()
        if self.args.serial:
            config.enable_device(self.args.serial)
        config.enable_stream(
            rs.stream.color,
            self.args.camera_width,
            self.args.camera_height,
            rs.format.bgr8,
            self.args.camera_fps,
        )
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)
        device = profile.get_device()
        try:
            serial = str(device.get_info(rs.camera_info.serial_number)).strip()
            model_name = str(device.get_info(rs.camera_info.name)).strip()
        except RuntimeError as error:
            raise RuntimeError("无法读取 D405 相机信息") from error
        if not serial:
            raise RuntimeError("D405 相机序列号为空")
        self.state.set_camera(serial, model_name)
        # 本程序只做彩色视觉识别，不启用深度流，避免 D405 双流启动超时。
        for warmup_index in range(self.args.warmup_frames):
            try:
                self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                if warmup_index >= 2:
                    raise
                time.sleep(0.2)

    def _read_frame(self) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("D405 尚未打开")
        last_error: RuntimeError | None = None
        for attempt in range(3):
            if self.stop_event.is_set():
                raise RuntimeError("验收服务正在停止")
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("D405 未返回彩色帧")
                color = np.asanyarray(color_frame.get_data())
                if color is None or color.size == 0:
                    raise RuntimeError("D405 彩色帧为空")
                return rotate_frame(color, self.args.rotate)
            except RuntimeError as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.2)
        raise RuntimeError(f"D405 连续 3 次未返回彩色帧：{last_error}")

    def _infer(self, frame: np.ndarray, conf: float, iou: float) -> tuple[list[dict[str, Any]], float]:
        started = time.monotonic()
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=self.args.imgsz,
            device=self.args.device,
            max_det=self.args.max_det,
            verbose=False,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        detections: list[dict[str, Any]] = []
        if not results:
            return detections, elapsed_ms
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections, elapsed_ms
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        for bbox, score, class_id in zip(xyxy, confidences, classes):
            if not 0 <= int(class_id) < len(self.model_info.names):
                continue
            detections.append(
                {
                    "id": int(class_id),
                    "officialId": int(class_id) + 1,
                    "name": self.model_info.names[int(class_id)],
                    "confidence": round(float(score), 4),
                    "bbox": [round(float(value), 1) for value in bbox],
                }
            )
        detections.sort(key=lambda item: -float(item["confidence"]))
        return detections, elapsed_ms

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        serial: str,
        conf: float,
    ) -> np.ndarray:
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        for item in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in item["bbox"])
            x1 = min(max(x1, 0), width - 1)
            y1 = min(max(y1, 0), height - 1)
            x2 = min(max(x2, 0), width - 1)
            y2 = min(max(y2, 0), height - 1)
            color = class_color(int(item["id"]))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID {item['id']:02d} | {item['name']} | {item['confidence'] * 100:.1f}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            top = y1 - text_height - baseline - 8
            if top < 28:
                top = min(y1 + 2, height - text_height - baseline - 8)
            bottom = top + text_height + baseline + 8
            right = min(x1 + text_width + 10, width - 1)
            cv2.rectangle(annotated, (x1, top), (right, bottom), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 5, top + text_height + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        header = f"FINAL 40-CLASS | conf {conf:.2f} | boxes {len(detections)} | D405 {serial}"
        cv2.rectangle(annotated, (0, 0), (width, 27), (16, 24, 32), -1)
        cv2.putText(
            annotated,
            header,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (225, 245, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def run(self) -> None:
        last_inference_at = 0.0
        last_inference_ms = 0.0
        latest_detections: list[dict[str, Any]] = []
        fps_counter_started = time.monotonic()
        fps_frames = 0
        fps = 0.0
        try:
            self.state.set_phase("loading_camera")
            self._open_camera()
            self.state.set_phase("running")
            while not self.stop_event.is_set():
                frame = self._read_frame()
                now = time.monotonic()
                conf, iou = self.state.get_settings()
                if now - last_inference_at >= self.args.detect_interval:
                    latest_detections, last_inference_ms = self._infer(frame, conf, iou)
                    last_inference_at = now
                with self.state.condition:
                    serial = str(self.state.status.get("cameraSerial", ""))
                annotated = self._annotate(frame, latest_detections, serial, conf)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)],
                )
                if not ok:
                    raise RuntimeError("OpenCV 编码 JPEG 失败")
                fps_frames += 1
                elapsed = time.monotonic() - fps_counter_started
                if elapsed >= 1.0:
                    fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_counter_started = time.monotonic()
                self.state.update_frame(
                    jpeg=encoded.tobytes(),
                    width=frame.shape[1],
                    height=frame.shape[0],
                    fps=fps,
                    detections=latest_detections,
                    inference_ms=last_inference_ms,
                    inference_at=last_inference_at,
                    conf=conf,
                )
        except Exception as error:
            self.state.set_error(f"{type(error).__name__}: {error}")
        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    pass


class AcceptanceRequestHandler(BaseHTTPRequestHandler):
    state: AcceptanceState
    model_info: ModelInfo
    snapshot_dir: Path
    latest_snapshot: Path | None = None
    latest_snapshot_lock = threading.Lock()
    server_version = "Track1Acceptance40/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_bytes(body, "application/json; charset=utf-8", status)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length < 0 or length > 65536:
            raise ValueError("请求体大小无效")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return value

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            status, _ = self.state.snapshot()
            self.send_json(status)
            return
        if path == "/status.json":
            status, _ = self.state.snapshot()
            status["config"] = self.model_info.to_api()
            self.send_json(status)
            return
        if path == "/api/model":
            self.send_json(self.model_info.to_api())
            return
        if path == "/snapshot.jpg":
            _, jpeg = self.state.snapshot()
            if jpeg is None:
                self.send_json({"ok": False, "error": "相机画面尚未就绪"}, HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                self.send_bytes(jpeg, "image/jpeg")
            return
        if path == "/api/snapshot/latest.jpg":
            with self.latest_snapshot_lock:
                latest = self.latest_snapshot
            if latest is None or not latest.is_file():
                self.send_json({"ok": False, "error": "尚无验收截图"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(latest.read_bytes(), "image/jpeg")
            return
        if path == "/stream.mjpg":
            self.stream_frames()
            return
        self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/api/settings":
                payload = self.read_json()
                conf = float(payload["conf"]) if "conf" in payload else None
                current_conf, current_iou = self.state.set_settings(conf)
                self.send_json({"ok": True, "conf": current_conf, "iou": current_iou})
                return
            if path == "/api/snapshot":
                payload = self.read_json()
                note = str(payload.get("note", "")).strip()[:200]
                status, jpeg = self.state.snapshot()
                if jpeg is None:
                    self.send_json({"ok": False, "error": "相机画面尚未就绪"}, HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
                image_path = self.snapshot_dir / f"acceptance_{stamp}.jpg"
                json_path = self.snapshot_dir / f"acceptance_{stamp}.json"
                image_path.write_bytes(jpeg)
                with self.latest_snapshot_lock:
                    self.latest_snapshot = image_path
                sidecar = dict(status)
                sidecar["snapshot"] = str(image_path)
                sidecar["note"] = note
                sidecar["savedAt"] = datetime.now().astimezone().isoformat(timespec="seconds")
                json_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.send_json({"ok": True, "image": str(image_path), "metadata": str(json_path)})
                return
            if path == "/api/stop":
                self.read_json()
                self.send_json({"ok": True, "stopping": True})
                threading.Thread(target=self.server.shutdown, name="acceptance-stop", daemon=True).start()
                return
            self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            self.send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)

    def stream_frames(self) -> None:
        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        sequence = -1
        try:
            while True:
                result = self.state.wait_for_jpeg(sequence, timeout=0.5)
                if result is None:
                    continue
                sequence, jpeg = result
                self.wfile.write(
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def validate_model(config: dict[str, Any], model_path: Path) -> tuple[YOLO, ModelInfo]:
    if not model_path.is_file():
        raise ValueError(f"最终权重不存在：{model_path}")
    model_section = config["model"]
    expected_names = [str(value) for value in model_section["class_names"]]
    expected_count = int(model_section["expected_class_count"])
    expected_sha = str(model_section["sha256"]).strip().lower()
    actual_sha = sha256_file(model_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"权重 SHA-256 不一致：实际 {actual_sha}，配置 {expected_sha}")
    model = YOLO(str(model_path))
    names = [str(model.names[index]) for index in sorted(model.names)]
    if int(model.model.nc) != expected_count or len(names) != expected_count:
        raise ValueError(f"模型类别数错误：nc={model.model.nc}, names={len(names)}, expected={expected_count}")
    if names != expected_names:
        mismatch = next(
            (index for index, pair in enumerate(zip(names, expected_names)) if pair[0] != pair[1]),
            min(len(names), len(expected_names)),
        )
        raise ValueError(f"模型类别表不一致，首个差异在索引 {mismatch}")
    expected_ids = [int(value) for value in model_section["expected_class_ids"]]
    reserved_ids = [int(value) for value in model_section["reserved_empty_class_ids"]]
    if set(expected_ids) & set(reserved_ids):
        raise ValueError("expected_class_ids 与 reserved_empty_class_ids 重叠")
    if set(expected_ids) | set(reserved_ids) != set(range(expected_count)):
        raise ValueError("期望类别与保留空槽没有完整覆盖 0-39")
    info = ModelInfo(
        weights=model_path,
        names=names,
        expected_names=expected_names,
        expected_class_ids=expected_ids,
        reserved_empty_class_ids=reserved_ids,
        sha256=actual_sha,
    )
    return model, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="赛道一最终 40 类模型视觉验收；不控制机械臂或夹爪")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=0)
    parser.add_argument("--camera-fps", type=int, default=0)
    parser.add_argument("--serial", default="")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=-1)
    parser.add_argument("--warmup-frames", type=int, default=-1)
    parser.add_argument("--device", default="")
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--conf", type=float, default=-1.0)
    parser.add_argument("--iou", type=float, default=-1.0)
    parser.add_argument("--max-det", type=int, default=0)
    parser.add_argument("--detect-interval", type=float, default=-1.0)
    parser.add_argument("--jpeg-quality", type=int, default=0)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def apply_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    camera = config["camera"]
    model = config["model"]
    runtime = config["runtime"]
    server = config["server"]
    args.host = args.host or str(server["host"])
    args.port = args.port or int(server["port"])
    args.camera_width = args.camera_width or int(camera["width"])
    args.camera_height = args.camera_height or int(camera["height"])
    args.camera_fps = args.camera_fps or int(camera["fps"])
    args.rotate = args.rotate if args.rotate >= 0 else int(camera["rotate_degrees"])
    args.warmup_frames = args.warmup_frames if args.warmup_frames >= 0 else int(camera["warmup_frames"])
    args.device = args.device or str(model["device"])
    args.imgsz = args.imgsz or int(model["imgsz"])
    args.conf = args.conf if args.conf >= 0 else float(model["default_conf"])
    args.iou = args.iou if args.iou >= 0 else float(model["iou"])
    args.max_det = args.max_det or int(model["max_det"])
    args.detect_interval = args.detect_interval if args.detect_interval >= 0 else float(model["detect_interval_s"])
    args.jpeg_quality = args.jpeg_quality or int(runtime["jpeg_quality"])
    if not 0 < args.port < 65536:
        raise ValueError("端口必须在 1-65535")
    if not 0.01 <= args.conf <= 0.95:
        raise ValueError("conf 必须在 0.01-0.95")
    if not 0 < args.iou <= 1:
        raise ValueError("iou 必须在 (0,1]")
    if args.camera_fps <= 0 or args.warmup_frames < 0:
        raise ValueError("相机参数无效")
    if args.detect_interval < 0 or args.jpeg_quality not in range(1, 101):
        raise ValueError("检测间隔或 JPEG 质量无效")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve()
        config = load_config(config_path)
        apply_config(args, config)
        weights = Path(str(config["model"]["weights"])).expanduser().resolve()
        _, model_info = validate_model(config, weights)
    except Exception as error:
        print(f"验收配置失败：{error}", flush=True)
        return 2

    print("FINAL 40-CLASS ACCEPTANCE", flush=True)
    print(f"weights={model_info.weights}", flush=True)
    print(f"sha256={model_info.sha256}", flush=True)
    print(f"nc={len(model_info.names)} expected={len(model_info.expected_class_ids)} reserved={model_info.reserved_empty_class_ids}", flush=True)
    if args.self_test:
        print("self_test=PASS", flush=True)
        return 0

    state = AcceptanceState(model_info, conf=args.conf, iou=args.iou)
    worker = AcceptanceWorker(args, model_info, state)
    AcceptanceRequestHandler.state = state
    AcceptanceRequestHandler.model_info = model_info
    AcceptanceRequestHandler.snapshot_dir = Path(str(config["paths"]["snapshot_dir"])).resolve()
    try:
        server = ThreadingHTTPServer((args.host, args.port), AcceptanceRequestHandler)
    except OSError as error:
        print(f"网页端口启动失败：{args.host}:{args.port}: {error}", flush=True)
        return 2

    url = f"http://{args.host}:{args.port}/"
    print(f"url={url}", flush=True)
    print("robot_control=disabled gripper_control=disabled", flush=True)
    worker.start()
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        server.server_close()
        worker.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
