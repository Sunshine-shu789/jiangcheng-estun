#!/usr/bin/env python3
"""Read-only MS42DC probe across common UART baud rates."""
from __future__ import annotations

import time
from datetime import datetime

import serial

PORT = "/dev/ttyACM0"
LOG_PATH = "/home/sunshine/ms42dc_baud_probe.log"
BAUDS = (115200, 9600, 19200, 38400, 57600, 230400, 460800, 921600, 1000000)


def request(address: int) -> bytes:
    body = bytes([0x7B, address, 0, 0, 0, 0, 0, 0, 0])
    bcc = 0
    for b in body:
        bcc ^= b
    return body + bytes([bcc, 0x7D])


def valid_status(data: bytes) -> bool:
    if len(data) < 9:
        return False
    for i in range(len(data) - 8):
        frame = data[i:i + 9]
        bcc = 0
        for b in frame[:8]:
            bcc ^= b
        if bcc == frame[8]:
            return True
    return False


def read_for(ser: serial.Serial, seconds: float) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chunk = ser.read(64)
        if chunk:
            data.extend(chunk)
    return bytes(data)


def probe(baud: int) -> list[bytes]:
    ser = serial.Serial()
    ser.port = PORT
    ser.baudrate = baud
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = 0.02
    ser.write_timeout = 1.0
    ser.xonxoff = False
    ser.rtscts = False
    ser.dsrdtr = False
    ser.exclusive = True
    ser.open()
    hits: list[bytes] = []
    try:
        ser.dtr = True
        ser.rts = True
        time.sleep(0.25)
        ser.reset_input_buffer()
        passive = read_for(ser, 0.18)
        if passive:
            hits.append(passive)
            print(f"  passive: {passive.hex(' ')}")

        for address in [1] + [a for a in range(256) if a != 1]:
            ser.reset_input_buffer()
            ser.write(request(address))
            ser.flush()
            data = read_for(ser, 0.035)
            if data:
                hits.append(data)
                print(f"  address {address:3d}: {data.hex(' ')}")
                if valid_status(data):
                    print(f"  VALID status response for address {address}")
                    return hits
    finally:
        ser.close()
    return hits


def main() -> int:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] read-only baud scan")
    print("Only status requests; no motion frames.")
    all_hits: list[tuple[int, bytes]] = []
    for baud in BAUDS:
        print(f"\nTrying {baud} 8N1")
        try:
            hits = probe(baud)
        except Exception as exc:
            print(f"  open/probe error: {exc}")
            continue
        if hits:
            all_hits.extend((baud, hit) for hit in hits)
            if any(valid_status(hit) for hit in hits):
                print(f"\nSUCCESS at {baud} baud")
                return 0
        else:
            print("  no bytes")
    if all_hits:
        print("\nData received, but no valid status frame. See raw bytes above.")
        return 1
    print("\nNo bytes at any baud rate or address. USB UART-to-MCU link is not responding.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
