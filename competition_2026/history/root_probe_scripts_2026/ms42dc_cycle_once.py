#!/usr/bin/env python3
"""MS42DC one-shot bench test: query status, close once, then open once."""

from __future__ import annotations

import glob
import os
import stat
import sys
import time
from datetime import datetime

import ms42dc_gripper as mg

LOG_PATH = "/home/sunshine/ms42dc_cycle_once.log"
ADDRESS = 1
SUBDIVIDE = 32
ANGLE_DEG = 1872.0
SPEED_RAD_S = 20.0
MOVE_WAIT_S = 7.0


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def port_candidates() -> list[str]:
    configured = os.environ.get("MS42DC_PORT", "").strip()
    values = [configured] if configured else []
    values.extend(
        [
            "/dev/motor_serial",
            "/dev/ttyACM0",
            "/dev/ttyCH343USB0",
            "/dev/ttyUSB0",
        ]
    )
    values.extend(sorted(glob.glob("/dev/ttyACM*")))
    values.extend(sorted(glob.glob("/dev/ttyCH343USB*")))
    values.extend(sorted(glob.glob("/dev/ttyUSB*")))
    values.extend(sorted(glob.glob("/dev/serial/by-id/*")))

    result: list[str] = []
    for path in values:
        if path in result:
            continue
        try:
            mode = os.stat(path).st_mode
        except OSError:
            continue
        if stat.S_ISCHR(mode):
            result.append(path)
    return result


def open_first_available():
    candidates = port_candidates()
    if not candidates:
        print("\n没有找到夹爪串口。")
        print("请检查 Type-C 数据线、12V 供电和 1 号 OFF / 2 号 ON 的拨码状态。")
        return None, None

    print("\n检测到串口：" + "，".join(candidates))
    errors: list[str] = []
    for path in candidates:
        try:
            ser = mg.open_serial(path)
            print(f"已打开：{path}（115200 8N1，DTR/RTS 保持无效）")
            return path, ser
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    print("\n串口打开失败：")
    for error in errors:
        print("  " + error)
    print("请关闭其它串口软件；必要时重新登录，使 dialout 组权限生效。")
    return None, None


def query_status(ser) -> mg.GripperStatus | None:
    try:
        status = mg.request_status(ser, ADDRESS, timeout=0.5)
        print("状态查询成功：" + status.describe())
        return status
    except Exception as exc:
        print(f"状态查询失败：{exc}")
        return None


def send_move_and_wait(ser, direction: int, name: str) -> bool:
    frame = mg.position_frame(
        address=ADDRESS,
        direction=direction,
        divides=SUBDIVIDE,
        angle_deg=ANGLE_DEG,
        speed_rad_s=SPEED_RAD_S,
    )
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    print(f"\n已发送{name}帧：{frame.hex(' ')}")
    print(f"{name}中……请观察夹爪。")

    deadline = time.monotonic() + MOVE_WAIT_S
    last_status = None
    while time.monotonic() < deadline:
        time.sleep(0.18)
        try:
            last_status = mg.request_status(ser, ADDRESS, timeout=0.3)
        except Exception:
            continue
        if last_status.reached:
            print(f"{name}完成：" + last_status.describe())
            return True

    if last_status is None:
        print(f"{name}等待结束，但没有收到状态回包；请以肉眼观察为准。")
    else:
        print(f"{name}等待结束，最后状态：" + last_status.describe())
    return False


def countdown(seconds: int = 5) -> None:
    print("\n安全提示：夹爪两指之间必须清空，12V 开关放在手边。")
    print("如果方向异常或出现卡死，立即关闭 12V。")
    for remaining in range(seconds, 0, -1):
        print(f"{remaining} 秒后开始……", end="\r", flush=True)
        time.sleep(1)
    print("开始动作。          ")


def pause() -> None:
    try:
        input("\n按回车关闭窗口……")
    except (EOFError, KeyboardInterrupt):
        pass


def main() -> int:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print(f"\n[{stamp}] MS42DC 单次开合测试")
        print("模式要求：1 号拨码 OFF，2 号拨码 ON；改变拨码前必须先断电。")
        print(f"日志：{LOG_PATH}")

        path, ser = open_first_available()
        if ser is None:
            return 1

        try:
            initial = query_status(ser)
            if initial is None:
                print("\n没有收到状态回包。仍只发送一轮开合命令，请以肉眼观察。")
                print("如果完全不动，请停止重复发送并检查 2 号拨码是否在 ON 侧。")

            countdown()

            close_seen = send_move_and_wait(ser, mg.DIR_CLOSE, "闭合")
            time.sleep(1.0)
            open_seen = send_move_and_wait(ser, mg.DIR_OPEN, "张开")
            query_status(ser)

            if close_seen or open_seen:
                print("\n结论：串口状态反馈和动作链路至少已有一部分工作。")
            else:
                print("\n结论：本轮没有收到到位反馈。")
                print("若肉眼也没有动作，请不要继续重复发送，先检查拨码、驱动和供电。")
            print(f"使用串口：{path}")
            return 0
        finally:
            try:
                ser.close()
            except Exception:
                pass
    except KeyboardInterrupt:
        print("\n已取消，没有继续发送后续动作。")
        return 130
    finally:
        print("\n测试程序结束。")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        pause()


if __name__ == "__main__":
    raise SystemExit(main())
