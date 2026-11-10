#!/usr/bin/env python3
"""One-click launcher for the Task A live camera detector."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
PREVIEW_SCRIPT = TASK_ROOT / "live_camera_preview.py"
PYTHON_EXECUTABLE = Path("/usr/bin/python3")
VENV_SITE_PACKAGES = Path(
    "/media/sunshine/新加卷/taskA/venv/lib/python3.10/site-packages"
)
PREFERRED_WEIGHTS = Path(
    "/media/sunshine/新加卷/taskA/runs/"
    "yolo11s_real_ft_v23_hardneg_20261016/weights/epoch3.pt"
)
FUSION_CALIBRATION_DIR = Path(
    "/media/sunshine/新加卷/taskA/reports/camera_eval_20261016/"
    "box_calibration_v3_20261016"
)
RUNS_ROOT = Path("/media/sunshine/新加卷/taskA/runs")
STATE_DIR = Path.home() / ".local/state/task-a-camera-recognition"
PID_FILE = STATE_DIR / "preview.pid"
LOG_FILE = STATE_DIR / "preview.log"
DEFAULT_WAIT_SECONDS = 180.0
MAX_PREVIEW_START_ATTEMPTS = 3
RETRYABLE_START_FAILURE = "getCount is non-monotonic."
DEFAULT_SCENE_ROOT = Path(
    os.environ.get(
        "TASK_A_SCENE_ROOT",
        "/media/sunshine/新加卷/taskA/eval_scenes_20261016",
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动、检查或停止 Task A 相机实时识别窗口。"
    )
    parser.add_argument("--weights", help="YOLO 权重路径；默认自动选择 v12b")
    parser.add_argument("--camera", help="V4L2 相机路径；默认自动查找 RealSense")
    parser.add_argument("--device", default="0", help="YOLO 推理设备")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO 推理分辨率；训练尺度为 640",
    )
    parser.add_argument(
        "--detection-scale",
        default="1.0,0.75,0.65",
        help="检测缩放：单值或逗号分隔多尺度（vote3 默认 1.0,0.75,0.65）",
    )
    parser.add_argument(
        "--fusion",
        choices=("legacy", "vote3"),
        default="vote3",
        help="后处理：vote3=推荐融合档；legacy=旧的多尺度合并",
    )
    parser.add_argument(
        "--fusion-min-views",
        type=int,
        default=3,
        help="vote3 下同一框至少出现的视角数",
    )
    parser.add_argument(
        "--fusion-tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vote3 下每个尺度启用 TTA",
    )
    parser.add_argument(
        "--fusion-arbitration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vote3 下启用类别仲裁",
    )
    parser.add_argument(
        "--fusion-calibration-dir",
        default=str(FUSION_CALIBRATION_DIR),
        help="vote3 的框校准目录",
    )
    parser.add_argument(
        "--stable-confidence-alpha",
        type=float,
        default=0.10,
        help="界面稳定置信度的 EMA 系数；1.0 表示不平滑",
    )
    parser.add_argument("--port", type=int, default=9100, help="本地网页端口")
    parser.add_argument(
        "--scene-root",
        default=str(DEFAULT_SCENE_ROOT),
        help="冻结分割后的图像、标签和深度保存目录",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        help="等待模型就绪秒数",
    )
    parser.add_argument(
        "--rgbd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 RealSense RGB-D（冻结并分割需要）；用 --no-rgbd 关闭",
    )
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--status", action="store_true", help="只显示当前状态")
    parser.add_argument("--stop", action="store_true", help="停止识别服务")
    return parser


def normalize_detection_scales(value: Any) -> tuple[float, ...]:
    """把 "0.65,0.75" / 0.65 / None 统一成尺度元组，便于比较与校验。"""
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    scales: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        scales.append(float(part))
    return tuple(scales)


def resolve_weights(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            return path
        raise RuntimeError(f"权重文件不存在：{path}")

    candidates = [PREFERRED_WEIGHTS]
    candidates.extend(
        Path(item)
        for item in sorted(
            glob.glob(
                str(RUNS_ROOT / "yolo11s_random_real_v8*/weights/best.pt")
            ),
            reverse=True,
        )
    )
    candidates.extend(
        Path(item)
        for item in sorted(
            glob.glob(
                str(RUNS_ROOT / "yolo11s_random_real_v7*/weights/best.pt")
            ),
            reverse=True,
        )
    )
    candidates.extend(
        Path(item)
        for item in sorted(
            glob.glob(
                str(RUNS_ROOT / "yolo11s_real_ft_v5*/weights/best.pt")
            ),
            reverse=True,
        )
    )
    candidates.extend(
        Path(item)
        for item in sorted(
            glob.glob(
                str(RUNS_ROOT / "yolo11s_real_ft_v4*/weights/best.pt")
            ),
            reverse=True,
        )
    )
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("没有找到可用的 v12b、v8、v7、v5 或 v4 权重")


def resolve_camera(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return path
        raise RuntimeError(f"相机设备不存在：{path}")

    patterns = [
        "/dev/v4l/by-id/*RealSense*video-index4",
        "/dev/v4l/by-id/*video-index4",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[0])
    raise RuntimeError(
        "没有找到 RealSense 彩色相机。请检查 USB 连接，并确认相机已插入。"
    )


def fetch_status(port: int, timeout: float = 0.8) -> dict[str, Any] | None:
    url = f"http://127.0.0.1:{port}/status.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ):
        return None
    return payload if isinstance(payload, dict) else None


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def read_pid() -> int | None:
    try:
        value = int(PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None
    return value if value > 1 else None


def pid_command(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def is_preview_process(pid: int, port: int | None = None) -> bool:
    command = pid_command(pid)
    if "live_camera_preview.py" not in command:
        return False
    return port is None or f"--port {port}" in command


def find_preview_pids(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "live_camera_preview.py"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []

    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid != os.getpid() and is_preview_process(pid, port):
            pids.append(pid)
    return pids


def stop_service(port: int) -> bool:
    pids = find_preview_pids(port)
    saved_pid = read_pid()
    if saved_pid is not None and is_preview_process(saved_pid, port):
        pids.append(saved_pid)
    pids = sorted(set(pids))

    if not pids:
        return True

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as error:
            raise RuntimeError(f"没有权限停止进程 {pid}") from error

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if (
            not any(is_preview_process(pid, port) for pid in pids)
            and not port_is_open(port)
        ):
            break
        time.sleep(0.1)

    remaining = [pid for pid in pids if is_preview_process(pid, port)]
    if remaining:
        raise RuntimeError(f"识别服务未能正常停止，进程：{remaining}")
    if port_is_open(port):
        raise RuntimeError(f"识别服务已退出，但端口 {port} 未释放")

    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass
    return True


def log_tail(max_lines: int = 20) -> str:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return "没有生成启动日志"
    return "\n".join(lines[-max_lines:])


def launch_preview(
    weights: Path,
    camera: Path,
    device: str,
    imgsz: int,
    detection_scale: str,
    stable_confidence_alpha: float,
    port: int,
    scene_root: Path,
    fusion: str = "legacy",
    fusion_min_views: int = 3,
    fusion_tta: bool = False,
    fusion_arbitration: bool = False,
    fusion_calibration_dir: str = "",
    rgbd: bool = True,
) -> int:
    if not PYTHON_EXECUTABLE.is_file():
        raise RuntimeError(f"Python 解释器不存在：{PYTHON_EXECUTABLE}")
    if not VENV_SITE_PACKAGES.is_dir():
        raise RuntimeError(f"Python 依赖目录不存在：{VENV_SITE_PACKAGES}")
    if not PREVIEW_SCRIPT.is_file():
        raise RuntimeError(f"实时识别程序不存在：{PREVIEW_SCRIPT}")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON_EXECUTABLE),
        str(PREVIEW_SCRIPT),
        "--weights",
        str(weights),
        "--camera",
        str(camera),
        "--device",
        device,
        "--imgsz",
        str(imgsz),
        "--stable-confidence-alpha",
        str(stable_confidence_alpha),
        "--rotate",
        "180",
        "--detection-scale",
        str(detection_scale),
        "--scene-root",
        str(scene_root),
    ]
    if fusion != "legacy":
        command += ["--fusion", fusion, "--fusion-min-views", str(fusion_min_views)]
        if fusion_tta:
            command.append("--fusion-tta")
        if fusion_arbitration:
            command.append("--fusion-arbitration")
        if fusion_calibration_dir:
            command += ["--fusion-calibration-dir", fusion_calibration_dir]
    if rgbd:
        command.append("--rgbd")
    command += [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    python_paths = [str(VENV_SITE_PACKAGES)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    with LOG_FILE.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=TASK_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_FILE.write_text(f"{process.pid}\n", encoding="utf-8")
    return process.pid


def wait_until_ready(
    process_pid: int,
    port: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(timeout, 1.0)
    while time.monotonic() < deadline:
        status = fetch_status(port)
        if status and status.get("ready"):
            return status
        if not is_preview_process(process_pid, port):
            raise RuntimeError(
                "识别服务启动后异常退出。\n\n"
                f"日志：{LOG_FILE}\n\n{log_tail()}"
            )
        time.sleep(0.25)
    raise RuntimeError(
        f"等待识别服务超时（{timeout:g} 秒）。\n\n"
        f"日志：{LOG_FILE}\n\n{log_tail()}"
    )


def log_size() -> int:
    try:
        return LOG_FILE.stat().st_size
    except OSError:
        return 0


def log_since(offset: int) -> str:
    try:
        with LOG_FILE.open("rb") as log:
            log.seek(offset)
            return log.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def is_retryable_start_failure(log_start: int) -> bool:
    return RETRYABLE_START_FAILURE in log_since(log_start)


def start_preview_with_retry(
    weights: Path,
    camera: Path,
    device: str,
    imgsz: int,
    detection_scale: str,
    stable_confidence_alpha: float,
    port: int,
    scene_root: Path,
    timeout: float,
    fusion: str = "legacy",
    fusion_min_views: int = 3,
    fusion_tta: bool = False,
    fusion_arbitration: bool = False,
    fusion_calibration_dir: str = "",
    rgbd: bool = True,
) -> tuple[int, dict[str, Any]]:
    last_error: RuntimeError | None = None
    for attempt in range(1, MAX_PREVIEW_START_ATTEMPTS + 1):
        log_start = log_size()
        process_pid = launch_preview(
            weights=weights,
            camera=camera,
            device=device,
            imgsz=imgsz,
            detection_scale=detection_scale,
            stable_confidence_alpha=stable_confidence_alpha,
            port=port,
            scene_root=scene_root,
            fusion=fusion,
            fusion_min_views=fusion_min_views,
            fusion_tta=fusion_tta,
            fusion_arbitration=fusion_arbitration,
            fusion_calibration_dir=fusion_calibration_dir,
            rgbd=rgbd,
        )
        try:
            return process_pid, wait_until_ready(process_pid, port, timeout)
        except RuntimeError as error:
            if not is_retryable_start_failure(log_start):
                raise
            last_error = error
            if attempt >= MAX_PREVIEW_START_ATTEMPTS:
                break
            print(
                "检测到 PyTorch 时钟初始化偶发崩溃，"
                f"正在重试（{attempt + 1}/{MAX_PREVIEW_START_ATTEMPTS}）...",
                file=sys.stderr,
            )
            time.sleep(0.5)

    raise RuntimeError(
        f"PyTorch 时钟初始化连续失败 {MAX_PREVIEW_START_ATTEMPTS} 次。\n\n"
        f"{last_error}"
    ) from last_error


def open_url(url: str) -> None:
    xdg_open = shutil.which("xdg-open")
    if xdg_open:
        subprocess.Popen(
            [xdg_open, url],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return
    webbrowser.open(url)


def notify(title: str, message: str) -> None:
    executable = shutil.which("notify-send")
    if not executable:
        return
    subprocess.run(
        [executable, title, message],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def show_error(message: str) -> None:
    notify("相机识别启动失败", message)
    zenity = shutil.which("zenity")
    if zenity:
        subprocess.run(
            [
                zenity,
                "--error",
                "--title=相机识别启动失败",
                f"--text={message}",
                "--width=620",
            ],
            check=False,
        )
    print(message, file=sys.stderr)


def status_text(port: int) -> str:
    status = fetch_status(port)
    if not status:
        return json.dumps(
            {"running": False, "port": port},
            ensure_ascii=False,
            indent=2,
        )
    return json.dumps(
        {
            "running": bool(status.get("ready")),
            "port": port,
            "frameSequence": status.get("frameSequence"),
            "fps": status.get("fps"),
            "inferenceMs": status.get("inferenceMs"),
            "imgsz": status.get("imgsz"),
            "detectionScale": status.get("detectionScale"),
            "fusion": status.get("fusion"),
            "fusionMinViews": status.get("fusionMinViews"),
            "fusionTta": status.get("fusionTta"),
            "fusionArbitration": status.get("fusionArbitration"),
            "stableConfidenceAlpha": status.get("stableConfidenceAlpha"),
            "weights": status.get("weights"),
            "sceneRoot": status.get("sceneRoot"),
            "classNames": status.get("classNames", []),
            "detections": status.get("detections", []),
        },
        ensure_ascii=False,
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.port < 65536:
        show_error("端口必须在 1 到 65535 之间。")
        return 2
    if not 64 <= args.imgsz <= 4096:
        show_error("推理分辨率必须在 64 到 4096 之间。")
        return 2
    if not 0.0 < args.stable_confidence_alpha <= 1.0:
        show_error("稳定置信度系数必须在 0 到 1 之间。")
        return 2
    detection_scales = normalize_detection_scales(args.detection_scale)
    if not detection_scales or len(detection_scales) > 4 or any(
        not 0.1 <= scale <= 1.0 for scale in detection_scales
    ):
        show_error("检测缩放必须在 0.1 到 1.0 之间，最多 4 个尺度。")
        return 2

    try:
        if args.stop:
            stop_service(args.port)
            notify("相机识别", "实时识别服务已停止。")
            print("实时识别服务已停止。")
            return 0

        if args.status:
            print(status_text(args.port))
            return 0

        desired_weights = resolve_weights(args.weights)
        scene_root = Path(args.scene_root).expanduser().resolve()
        scene_root.mkdir(parents=True, exist_ok=True)
        writable_probe = scene_root / ".task-a-write-test"
        try:
            writable_probe.write_text("ok\n", encoding="utf-8")
        finally:
            try:
                writable_probe.unlink()
            except FileNotFoundError:
                pass
        current = fetch_status(args.port)
        if current and current.get("ready"):
            current_weights = current.get("weights")
            current_scene_root = current.get("sceneRoot")
            if (
                not current_weights
                or Path(current_weights).resolve() != desired_weights
                or not current_scene_root
                or Path(current_scene_root).resolve() != scene_root
                or current.get("imgsz") != args.imgsz
                or normalize_detection_scales(current.get("detectionScale"))
                != detection_scales
                or current.get("fusion") != args.fusion
                or int(current.get("fusionMinViews") or 0) != args.fusion_min_views
                or bool(current.get("fusionTta")) != args.fusion_tta
                or bool(current.get("fusionArbitration")) != args.fusion_arbitration
                or current.get("stableConfidenceAlpha")
                != args.stable_confidence_alpha
                or (args.rgbd and current.get("depth") is None)
            ):
                stop_service(args.port)
                current = None

        if current and current.get("ready"):
            url = f"http://127.0.0.1:{args.port}/"
            if not args.no_browser:
                open_url(url)
            print(f"实时识别服务已经运行：{url}")
            return 0

        if port_is_open(args.port):
            existing_pids = find_preview_pids(args.port)
            if not existing_pids:
                raise RuntimeError(
                    f"端口 {args.port} 已被其他程序占用，请使用 --port 指定其他端口。"
                )
            status = wait_until_ready(
                existing_pids[0],
                args.port,
                args.wait,
            )
            url = f"http://127.0.0.1:{args.port}/"
            if not args.no_browser:
                open_url(url)
            notify("相机识别", "实时识别窗口已启动。")
            print(
                f"实时识别已启动：{url}\n"
                f"状态：{status.get('fps', 0):.1f} FPS"
            )
            return 0

        camera = resolve_camera(args.camera)
        _, status = start_preview_with_retry(
            weights=desired_weights,
            camera=camera,
            device=args.device,
            imgsz=args.imgsz,
            detection_scale=args.detection_scale,
            stable_confidence_alpha=args.stable_confidence_alpha,
            port=args.port,
            scene_root=scene_root,
            timeout=args.wait,
            fusion=args.fusion,
            fusion_min_views=args.fusion_min_views,
            fusion_tta=args.fusion_tta,
            fusion_arbitration=args.fusion_arbitration,
            fusion_calibration_dir=str(Path(args.fusion_calibration_dir).expanduser().resolve()),
            rgbd=args.rgbd,
        )
        url = f"http://127.0.0.1:{args.port}/"
        if not args.no_browser:
            open_url(url)
        notify("相机识别", "实时识别窗口已启动。")
        print(
            f"实时识别已启动：{url}\n"
            f"权重：{desired_weights}\n"
            f"相机：{camera}\n"
            f"标注保存目录：{scene_root}\n"
            f"状态：{status.get('fps', 0):.1f} FPS"
        )
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        try:
            stop_service(args.port)
        except RuntimeError:
            pass
        show_error(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
