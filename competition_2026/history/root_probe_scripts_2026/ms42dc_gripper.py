#!/usr/bin/env python3
"""MS42DC 驱控一体步进电机柔性机械爪串口控制器。

资料版本: V3.0 (2026-03-24 用户手册)
通信参数: USB CDC 串口, 115200 8N1
USB 串口控制时拨码开关 2 必须拨到 ON。

常用命令:
    python3 ms42dc_gripper.py status
    python3 ms42dc_gripper.py open
    python3 ms42dc_gripper.py close
    python3 ms42dc_gripper.py move close --angle-deg 900 --speed 20

角度参数表示本次朝指定方向转动的角度，不是绝对角度。
默认每次转动 3000°，以确保常见行程走满；夹爪到位后由堵转保护停止。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

try:
    import serial
except ImportError:
    sys.exit("缺少 pyserial，请先安装：python3 -m pip install pyserial")

HEADER = 0x7B
TAIL = 0x7D
MODE_POSITION = 0x02
DIR_OPEN = 0x00   # 逆时针
DIR_CLOSE = 0x01  # 顺时针
DEFAULT_PORT = os.environ.get("MS42DC_PORT", "/dev/ttyACM0")
BAUDRATE = 115200


def checksum(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return value


def position_frame(
    address: int,
    direction: int,
    divides: int,
    angle_deg: float,
    speed_rad_s: float,
) -> bytes:
    if not 0 <= address <= 0xFF:
        raise ValueError("设备地址必须在 0..255")
    if direction not in (DIR_OPEN, DIR_CLOSE):
        raise ValueError("方向必须是 open(0) 或 close(1)")
    if divides not in (2, 4, 8, 16, 32):
        raise ValueError("细分必须是 2、4、8、16 或 32")
    if not 0 < angle_deg <= 6553.5:
        raise ValueError("单次角度必须大于 0 且不超过 6553.5°")
    if not 0 < speed_rad_s <= 42.0:
        raise ValueError("速度必须大于 0 且不超过 42 rad/s")

    angle_raw = round(angle_deg * 10)
    speed_raw = round(speed_rad_s * 10)
    if angle_raw > 0xFFFF or speed_raw > 0xFFFF:
        raise ValueError("角度或速度超出协议 16 位范围")

    frame = bytearray(
        [
            HEADER,
            address,
            MODE_POSITION,
            direction,
            divides,
            (angle_raw >> 8) & 0xFF,
            angle_raw & 0xFF,
            (speed_raw >> 8) & 0xFF,
            speed_raw & 0xFF,
        ]
    )
    frame.append(checksum(frame))
    frame.append(TAIL)
    return bytes(frame)


def status_request_frame(address: int) -> bytes:
    if not 0 <= address <= 0xFF:
        raise ValueError("设备地址必须在 0..255")
    frame = bytearray([HEADER, address, 0, 0, 0, 0, 0, 0, 0])
    frame.append(checksum(frame))
    frame.append(TAIL)
    return bytes(frame)


@dataclass
class GripperStatus:
    address: int
    reached: bool
    speed_rad_s: float
    angle_deg: float

    @classmethod
    def parse(cls, data: bytes) -> "GripperStatus":
        if len(data) != 9:
            raise ValueError(f"状态帧长度应为 9，实际为 {len(data)}")
        if checksum(data[:8]) != data[8]:
            raise ValueError(
                f"状态帧 BCC 校验失败: {data.hex(' ')}，期望 {checksum(data[:8]):02x}"
            )
        return cls(
            address=data[0],
            reached=bool(data[1]),
            speed_rad_s=int.from_bytes(data[2:4], "big") / 10.0,
            angle_deg=int.from_bytes(data[4:8], "big") / 10.0,
        )

    def describe(self) -> str:
        state = "已到达目标位置" if self.reached else "未到达目标位置/正在运动"
        return (
            f"地址={self.address}，{state}，"
            f"速度={self.speed_rad_s:.1f} rad/s，角度={self.angle_deg:.1f}°"
        )


def open_serial(port: str) -> serial.Serial:
    try:
        # Create first and clear DTR/RTS before open. Some USB CDC boards reset
        # or ignore UART traffic if pyserial asserts these modem-control lines.
        ser = serial.Serial(
            port=None,
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
        ser.port = port
        ser.dtr = False
        ser.rts = False
        ser.open()
        return ser
    except (OSError, serial.SerialException) as exc:
        raise RuntimeError(f"无法打开串口 {port}: {exc}") from exc


def request_status(port_file: serial.Serial, address: int, timeout: float) -> GripperStatus:
    request = status_request_frame(address)
    port_file.reset_input_buffer()
    port_file.write(request)
    port_file.flush()

    deadline = time.monotonic() + timeout
    buffer = bytearray()
    while time.monotonic() < deadline:
        chunk = port_file.read(9 - len(buffer))
        if chunk:
            buffer.extend(chunk)
            if len(buffer) == 9:
                return GripperStatus.parse(bytes(buffer))
    received = bytes(buffer)
    raise RuntimeError(
        "未收到完整状态帧。请确认拨码开关 2 为 ON、供电正常，且设备地址正确。"
        + (f" 已收到: {received.hex(' ')}" if received else "")
    )


def send_position(
    port: str,
    frame: bytes,
    wait: bool,
    address: int,
    angle_deg: float,
    speed_rad_s: float,
    timeout: float,
) -> int:
    with open_serial(port) as ser:
        ser.reset_input_buffer()
        ser.write(frame)
        ser.flush()
        print(f"已发送: {frame.hex(' ')}")

        if not wait:
            return 0

        movement_seconds = angle_deg / max(speed_rad_s * 57.295779513, 1.0)
        deadline = time.monotonic() + min(timeout, movement_seconds + 8.0)
        last_status = None
        while time.monotonic() < deadline:
            time.sleep(0.12)
            try:
                last_status = request_status(ser, address, timeout=0.3)
            except (RuntimeError, ValueError):
                continue
            if last_status.reached:
                print(last_status.describe())
                return 0

        if last_status is not None:
            print(f"等待超时，最后状态：{last_status.describe()}")
        else:
            print("等待超时，未获取到有效状态。")
        return 2


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"串口，默认 {DEFAULT_PORT}")
    parser.add_argument("--address", type=lambda x: int(x, 0), default=1, help="设备地址，默认 1")
    parser.add_argument("--subdivide", type=int, default=32, choices=(2, 4, 8, 16, 32),
                        help="细分值，默认 32")
    parser.add_argument("--dry-run", action="store_true", help="只打印协议帧，不打开串口")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="控制 MS42DC 步进电机柔性机械爪",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  %(prog)s status\n"
            "  %(prog)s open\n"
            "  %(prog)s close\n"
            "  %(prog)s move open --angle-deg 900 --speed 20 --wait\n\n"
            "注意：USB 串口控制时拨码开关 2 必须为 ON；上下电前务必断开电源，不支持热插拔。"
        ),
    )
    add_connection_args(parser)
    sub = parser.add_subparsers(dest="command")

    status_parser = sub.add_parser("status", help="请求并显示当前速度、角度和到位状态")
    status_parser.add_argument("--timeout", type=float, default=1.0, help="状态读取超时，默认 1 秒")

    for name, help_text in (("open", "张开"), ("close", "闭合")):
        move_parser = sub.add_parser(name, help=help_text)
        move_parser.add_argument("--angle-deg", type=float, default=3000.0,
                                 help="本次转动角度，默认 3000°，可按实际行程调整")
        move_parser.add_argument("--speed", type=float, default=30.0,
                                 help="速度 rad/s，默认 30")
        move_parser.add_argument("--wait", action="store_true", help="等待状态帧报告到位")
        move_parser.add_argument("--timeout", type=float, default=30.0, help="最长等待秒数，默认 30")

    move_parser = sub.add_parser("move", help="按指定方向和角度移动")
    move_parser.add_argument("direction", choices=("open", "close"), help="open=张开，close=闭合")
    move_parser.add_argument("--angle-deg", type=float, required=True, help="本次转动角度")
    move_parser.add_argument("--speed", type=float, default=30.0, help="速度 rad/s，默认 30")
    move_parser.add_argument("--wait", action="store_true", help="等待状态帧报告到位")
    move_parser.add_argument("--timeout", type=float, default=30.0, help="最长等待秒数，默认 30")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args.command = "status"
        args.timeout = 1.0

    try:
        if args.command == "status":
            request = status_request_frame(args.address)
            if args.dry_run:
                print(f"状态请求帧: {request.hex(' ')}")
                return 0
            with open_serial(args.port) as ser:
                status = request_status(ser, args.address, args.timeout)
            print(status.describe())
            return 0

        direction = DIR_CLOSE if (args.command == "close" or getattr(args, "direction", None) == "close") else DIR_OPEN
        direction_name = "闭合" if direction == DIR_CLOSE else "张开"
        frame = position_frame(
            address=args.address,
            direction=direction,
            divides=args.subdivide,
            angle_deg=args.angle_deg,
            speed_rad_s=args.speed,
        )
        if args.dry_run:
            print(f"{direction_name}帧: {frame.hex(' ')}")
            return 0
        print(
            f"夹爪{direction_name}: 角度 {args.angle_deg:.1f}°，"
            f"速度 {args.speed:.1f} rad/s，串口 {args.port}"
        )
        return send_position(
            port=args.port,
            frame=frame,
            wait=args.wait,
            address=args.address,
            angle_deg=args.angle_deg,
            speed_rad_s=args.speed,
            timeout=args.timeout,
        )
    except (ValueError, RuntimeError, serial.SerialException) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
