#!/usr/bin/env python3
"""MS42DC 夹爪实机测试：一次只发送一帧，并保持串口读取反馈。"""

from __future__ import annotations

import glob
import os
import stat
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    print("缺少 pyserial，请先安装：sudo apt install python3-serial")
    input("按回车退出...")
    raise SystemExit(1)

HEAD = 0x7B
TAIL = 0x7D
BAUDRATE = 115200
LOG_PATH = "/home/sunshine/ms42dc_gripper_test.log"


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


def make_frame(
    mode: int,
    direction: int,
    subdivide: int,
    angle_deg: float,
    speed_rad_s: float,
) -> bytes:
    angle = round(angle_deg * 10)
    speed = round(speed_rad_s * 10)
    body = bytes(
        [
            HEAD,
            0x01,
            mode,
            direction,
            subdivide,
            (angle >> 8) & 0xFF,
            angle & 0xFF,
            (speed >> 8) & 0xFF,
            speed & 0xFF,
        ]
    )
    check = 0
    for byte in body:
        check ^= byte
    return body + bytes([check, TAIL])


# 厂家手册标准帧：1800°、30 rad/s、32 细分、设备地址 1。
OPEN_FRAME = make_frame(2, 0, 32, 1800.0, 30.0)
CLOSE_FRAME = make_frame(2, 1, 32, 1800.0, 30.0)
STATUS_FRAME = make_frame(0, 0, 0, 0.0, 0.0)


def port_candidates() -> list[str]:
    values: list[str] = []
    configured = os.environ.get("MS42DC_PORT", "").strip()
    if configured:
        values.append(configured)

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


def open_gripper_serial():
    candidates = port_candidates()
    if not candidates:
        print("\n没有找到可用的串口设备。")
        print("请确认：USB-C 已插好、12V 已上电、没有同时打开 WHEELTEC 串口助手。")
        print("正常设备应为 /dev/motor_serial 或 /dev/ttyACM0。")
        return None

    print("\n检测到的串口：" + "，".join(candidates))
    errors: list[str] = []
    for path in candidates:
        try:
            ser = serial.Serial(
                port=path,
                baudrate=BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.25,
                write_timeout=1.0,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                exclusive=True,
            )
            ser.reset_input_buffer()
            print(f"已打开：{path}（115200 8N1，无流控）")
            return ser
        except (OSError, serial.SerialException) as exc:
            errors.append(f"{path}: {exc}")

    print("\n串口打开失败：")
    for error in errors:
        print("  " + error)
    print("请关闭其它串口软件，并确认当前用户属于 dialout 组。")
    return None


def read_feedback(ser: serial.Serial, seconds: float) -> bytes:
    deadline = time.monotonic() + seconds
    received = bytearray()
    print(f"保持串口读取 {seconds:.0f} 秒，等待夹爪回包……")
    while time.monotonic() < deadline:
        try:
            waiting = ser.in_waiting
            chunk = ser.read(waiting if waiting else 1)
        except (OSError, serial.SerialException) as exc:
            print(f"读取串口失败：{exc}")
            break
        if chunk:
            received.extend(chunk)
            print(f"收到 {len(chunk)} 字节：{chunk.hex(' ')}")
    if received:
        print(f"合计收到 {len(received)} 字节：{received.hex(' ')}")
    else:
        print("未收到任何回包（这不代表运动命令没有生效，以肉眼观察为准）。")
    return bytes(received)


def decode_status(data: bytes) -> None:
    if len(data) < 9:
        return
    raw = data[:9]
    check = 0
    for byte in raw[:8]:
        check ^= byte
    if check != raw[8]:
        print("收到的 9 字节数据不是有效状态帧（校验不通过）。")
        return
    arrived = "已到位" if raw[1] else "未到位/运动中"
    speed = ((raw[2] << 8) | raw[3]) / 10.0
    angle = int.from_bytes(raw[4:8], "big") / 10.0
    print(f"状态解析：地址={raw[0]}，{arrived}，速度={speed:.1f} rad/s，角度={angle:.1f}°")


def send_one_frame(ser: serial.Serial, name: str, frame: bytes, read_seconds: float) -> None:
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    print(f"\n已发送{name}帧：{frame.hex(' ')}")
    data = read_feedback(ser, read_seconds)
    decode_status(data)


def pause() -> None:
    try:
        input("\n按回车返回菜单……")
    except (EOFError, KeyboardInterrupt):
        pass


def show_menu() -> None:
    print("\n" + "=" * 64)
    print("MS42DC 夹爪实机测试")
    print("=" * 64)
    print("安全提示：夹爪周围必须清空；12V 电源开关放在手边。")
    print("先关闭 WHEELTEC 上位机，避免它占用串口。")
    print("1. 只读状态查询（不驱动电机）")
    print("2. 张开（第一次请选这个）")
    print("3. 闭合")
    print("4. 退出")
    print("=" * 64)


def main() -> int:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print(f"\n[{stamp}] 启动夹爪实机测试")
        print(f"张开帧：{OPEN_FRAME.hex(' ')}")
        print(f"闭合帧：{CLOSE_FRAME.hex(' ')}")
        print(f"日志：{LOG_PATH}")

        ser = open_gripper_serial()
        if ser is None:
            input("\n按回车退出……")
            return 1

        try:
            while True:
                show_menu()
                try:
                    choice = input("请选择 [1-4]：").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n退出。")
                    return 0

                if choice == "1":
                    send_one_frame(ser, "状态查询", STATUS_FRAME, 2.0)
                    pause()
                elif choice == "2":
                    send_one_frame(ser, "张开", OPEN_FRAME, 7.0)
                    print("\n请观察夹爪是否张开。")
                    print("如果方向相反或没有动作，立即断 12V，不要连续重复发送。")
                    pause()
                elif choice == "3":
                    send_one_frame(ser, "闭合", CLOSE_FRAME, 7.0)
                    print("\n请观察夹爪是否闭合。")
                    print("如果方向相反或没有动作，立即断 12V，不要连续重复发送。")
                    pause()
                elif choice == "4":
                    print("退出。")
                    return 0
                else:
                    print("无效选择，请输入 1、2、3 或 4。")
        finally:
            try:
                ser.close()
            except Exception:
                pass
    finally:
        print("\n测试程序已结束。")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
