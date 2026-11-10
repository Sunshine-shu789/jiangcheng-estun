#!/usr/bin/env python3
"""MS42DC 只读地址扫描：只请求状态，不发送任何运动命令。"""

from __future__ import annotations

import glob
import os
import stat
import sys
import time
from datetime import datetime

import ms42dc_gripper as mg

LOG_PATH = "/home/sunshine/ms42dc_scan_status.log"
SCAN_TIMEOUT_S = 0.08
LISTEN_BEFORE_SCAN_S = 1.0


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


def candidates() -> list[str]:
    configured = os.environ.get("MS42DC_PORT", "").strip()
    values = [configured] if configured else []
    values += ["/dev/motor_serial", "/dev/ttyACM0", "/dev/ttyCH343USB0", "/dev/ttyUSB0"]
    values += sorted(glob.glob("/dev/serial/by-id/*"))
    values += sorted(glob.glob("/dev/ttyACM*"))
    values += sorted(glob.glob("/dev/ttyCH343USB*"))
    values += sorted(glob.glob("/dev/ttyUSB*"))

    found: list[str] = []
    for path in values:
        if path and path not in found:
            try:
                if stat.S_ISCHR(os.stat(path).st_mode):
                    found.append(path)
            except OSError:
                pass
    return found


def open_port():
    paths = candidates()
    if not paths:
        print("没有找到串口。请确认 Type-C 数据线已接好，并重新插拔 USB。")
        return None, None

    print("检测到串口：" + "，".join(paths))
    for path in paths:
        try:
            ser = mg.open_serial(path)
            print(f"已打开：{path}（115200 8N1，只读扫描）")
            return path, ser
        except Exception as exc:
            print(f"打开失败：{path}: {exc}")
    return None, None


def parse_any_status(data: bytes):
    """在一段字节流中寻找 BCC 正确的 9 字节状态帧。"""
    for start in range(max(0, len(data) - 8)):
        frame = data[start:start + 9]
        if len(frame) != 9:
            continue
        if mg.checksum(frame[:8]) == frame[8]:
            try:
                return mg.GripperStatus.parse(frame)
            except ValueError:
                pass
    return None


def main() -> int:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"\n[{stamp}] MS42DC 只读状态/地址扫描")
    print("不会发送开合命令。拨码要求：1 号往下，2 号往上（串口模式）。")

    path, ser = open_port()
    if ser is None:
        return 1

    found = []
    try:
        # 如果固件配置成 20 Hz 主动上报，先监听，不发送任何字节。
        print(f"先监听 {LISTEN_BEFORE_SCAN_S:.0f} 秒主动状态回包……")
        ser.timeout = 0.1
        deadline = time.monotonic() + LISTEN_BEFORE_SCAN_S
        stream = bytearray()
        while time.monotonic() < deadline:
            chunk = ser.read(64)
            if chunk:
                stream.extend(chunk)
        if stream:
            print("监听期间收到原始数据：" + stream.hex(" "))
            status = parse_any_status(bytes(stream))
            if status:
                print("识别到状态：" + status.describe())
                found.append(status.address)
        else:
            print("监听期间没有主动回包。")

        print("开始扫描地址 0..255，每个地址只发状态请求……")
        for address in range(256):
            request = mg.status_request_frame(address)
            ser.reset_input_buffer()
            ser.write(request)
            ser.flush()

            deadline = time.monotonic() + SCAN_TIMEOUT_S
            response = bytearray()
            while time.monotonic() < deadline:
                chunk = ser.read(64)
                if chunk:
                    response.extend(chunk)
                    if len(response) >= 9:
                        break

            status = parse_any_status(bytes(response)) if response else None
            if status:
                print(
                    f"收到回包：请求地址={address}，"
                    f"回包地址={status.address}，{status.describe()}"
                )
                if status.address not in found:
                    found.append(status.address)
            elif response:
                print(
                    f"地址 {address} 收到无法校验的数据：{response.hex(' ')}"
                )

            if address % 32 == 31:
                print(f"已扫描到地址 {address}，继续……")

    except KeyboardInterrupt:
        print("\n用户中止扫描。")
    except Exception as exc:
        print(f"\n扫描错误：{exc}")
    finally:
        try:
            ser.close()
        except Exception:
            pass

    print("\n扫描结果：")
    if found:
        print("可用设备地址：" + "，".join(str(x) for x in found))
        print("下一步先用该地址做一次状态查询，再考虑发送运动命令。")
        return 0

    print("0..255 全部没有状态回包。")
    print("请检查：")
    print("1. 12V 是否真正加到接口一的 4 脚（正极）和 1 脚（负极），范围 10–25V；")
    print("2. 2 号拨码是否确实朝上（串口模式），1 号是否确实朝下；")
    print("3. 断电、重新上电后再运行本扫描；")
    print("4. 如果仍然没有回包，改用厂家 Windows 上位机验证板子是否进入上位机模式。")
    print(f"使用串口：{path}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
