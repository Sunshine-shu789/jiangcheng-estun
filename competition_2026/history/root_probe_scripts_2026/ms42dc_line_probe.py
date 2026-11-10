#!/usr/bin/env python3
"""MS42DC read-only serial probe across DTR/RTS idle states.

Sends only 11-byte status requests. No motion command is sent.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime

import serial

PORT = "/dev/ttyACM0"
LOG = "/home/sunshine/ms42dc_line_probe.log"


def frame(address: int) -> bytes:
    body = bytes([0x7B, address, 0, 0, 0, 0, 0, 0, 0])
    checksum = 0
    for b in body:
        checksum ^= b
    return body + bytes([checksum, 0x7D])


def describe(data: bytes) -> str:
    if len(data) >= 9:
        for i in range(len(data) - 8):
            f = data[i:i + 9]
            if f[0] <= 0xFF and f[8] == 0x7D:
                bcc = 0
                for b in f[:8]:
                    bcc ^= b
                if bcc == f[8]:
                    return "valid-like"
    return "raw"


def drain(ser: serial.Serial, seconds: float) -> bytes:
    out = bytearray()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chunk = ser.read(128)
        if chunk:
            out.extend(chunk)
    return bytes(out)


def open_port() -> serial.Serial:
    ser = serial.Serial()
    ser.port = PORT
    ser.baudrate = 115200
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = 0.05
    ser.write_timeout = 1.0
    ser.xonxoff = False
    ser.rtscts = False
    ser.dsrdtr = False
    ser.exclusive = True
    ser.open()
    return ser


def probe_state(ser: serial.Serial, dtr: bool, rts: bool) -> list[tuple[int, bytes]]:
    ser.dtr = dtr
    ser.rts = rts
    time.sleep(0.8)
    print(f"\nDTR={dtr} RTS={rts} CTS={ser.cts} DSR={ser.dsr} CD={ser.cd}")
    ser.reset_input_buffer()
    passive = drain(ser, 0.7)
    if passive:
        print("  被动数据:", passive.hex(" "))

    hits: list[tuple[int, bytes]] = []
    # Address 1 first because it is the documented default; then scan all.
    order = [1] + [a for a in range(256) if a != 1]
    for address in order:
        req = frame(address)
        ser.reset_input_buffer()
        ser.write(req)
        ser.flush()
        response = drain(ser, 0.06)
        if response:
            hits.append((address, response))
            print(f"  地址 {address:3d} 收到: {response.hex(' ')}")
            if address == 1 and len(response) >= 9:
                break
        if address % 64 == 0 and address:
            print(f"  已扫描到地址 {address}")
    return hits


def main() -> int:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] MS42DC read-only line probe")
    print("只发送状态请求，不会发送运动帧。")
    ser = open_port()
    print(f"串口已打开: {PORT} @115200 8N1")
    hits: list[tuple[int, bytes]] = []
    try:
        for dtr, rts in [(True, True), (True, False), (False, True), (False, False)]:
            hits += probe_state(ser, dtr, rts)
    finally:
        ser.close()

    if hits:
        print("\n结论：至少在一个 DTR/RTS 状态下收到数据。")
        return 0
    print("\n结论：所有 DTR/RTS 状态和 0..255 地址均无任何回包。")
    print("这已排除单纯的 DTR/RTS 线状态问题，下一步应核对 2 号拨码到底是 ON 还是 OFF。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
