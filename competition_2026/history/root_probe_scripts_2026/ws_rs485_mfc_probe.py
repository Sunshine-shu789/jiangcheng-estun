#!/usr/bin/env python3
"""Send one gripper status query and poll the MFC receive nodes.

This only requests status. It never sends a motion/open/close frame.
"""

import argparse
import base64
import json
import os
import socket
import struct
import time

from ws_rs485_probe import recv_frame, request, send_frame


STATUS_QUERY = bytes([0xEB, 0x90, 0x01, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00])
MFC_PATHS = [
    "EtherCAT/Slave/MFC/receiverStatus",
    "EtherCAT/Slave/MFC/receiverLength",
    "EtherCAT/Slave/MFC/receiverIndex",
    "EtherCAT/Slave/MFC/receiverData",
    "EtherCAT/Slave/MFC/transferEnable",
    "EtherCAT/Slave/MFC/transferIndex",
    "EtherCAT/Slave/MFC/transferData",
]


def make_call(tag, plugin, action, data=None):
    payload = {
        "id": f"{tag}-{time.time():.3f}",
        "time": 0,
        "token": "user:admin",
        "type": plugin,
        "action": action,
    }
    if data is not None:
        payload["data"] = data
    return payload


def get_mfc_values(sock, reply_seconds):
    payload = {
        "id": f"mfc-read-{time.time():.3f}",
        "time": 0,
        "token": "user:admin",
        "type": "common",
        "action": "getparam",
        "data": MFC_PATHS,
    }
    reply = request(sock, payload, reply_seconds)
    if not reply or reply.get("code") != 200:
        return None
    data = reply.get("data")
    if not isinstance(data, dict):
        return None
    values = data.get("data")
    if not isinstance(values, dict):
        return None
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--reply-seconds", type=float, default=4.0)
    parser.add_argument("--poll-seconds", type=float, default=4.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--data-bit", type=int, default=8)
    parser.add_argument("--parity", type=int, default=0)
    parser.add_argument("--stop-bit", type=int, default=0)
    parser.add_argument("--frame", default=STATUS_QUERY.hex(" "))
    args = parser.parse_args()

    frame = bytes(int(token, 16) for token in args.frame.replace(",", " ").split())
    print("status query:", frame.hex(" "))

    key = base64.b64encode(os.urandom(16)).decode()
    headers = (
        f"GET / HTTP/1.1\r\nHost: {args.host}:{args.port}\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )

    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        sock.sendall(headers.encode())
        response = bytearray()
        while b"\r\n\r\n" not in response:
            response.extend(sock.recv(4096))
        if not bytes(response).startswith(b"HTTP/1.1 101"):
            raise SystemExit("WebSocket handshake failed")

        try:
            sock.settimeout(1.0)
            _, welcome = recv_frame(sock)
            print("WELCOME", welcome.decode(errors="replace"))
        except socket.timeout:
            pass

        before = get_mfc_values(sock, args.reply_seconds)
        print("BEFORE", json.dumps(before, ensure_ascii=False))

        for tag, action, data in (
            ("flush", "flushReadBuffer", None),
            (
                "init",
                "init",
                {
                    "baudrate": args.baudrate,
                    "stopBit": args.stop_bit,
                    "parity": args.parity,
                    "dataBit": args.data_bit,
                },
            ),
            ("write", "write", list(frame)),
        ):
            reply = request(sock, make_call(tag, "EC2RS485", action, data),
                            args.reply_seconds)
            print(tag.upper(), json.dumps(reply, ensure_ascii=False))

        deadline = time.monotonic() + args.poll_seconds
        previous = None
        samples = []
        while time.monotonic() < deadline:
            values = get_mfc_values(sock, args.reply_seconds)
            if values is not None:
                samples.append(values)
                if values != previous:
                    print("MFC", json.dumps(values, ensure_ascii=False))
                    previous = values
            time.sleep(max(0.0, args.poll_interval))

        if samples:
            nonzero_lengths = [v["EtherCAT/Slave/MFC/receiverLength"]
                               for v in samples
                               if v["EtherCAT/Slave/MFC/receiverLength"]]
            nonzero_status = [v["EtherCAT/Slave/MFC/receiverStatus"]
                              for v in samples
                              if v["EtherCAT/Slave/MFC/receiverStatus"]]
            print("SUMMARY", json.dumps({
                "samples": len(samples),
                "receiverLength_nonzero": nonzero_lengths,
                "receiverStatus_nonzero": nonzero_status,
                "final": samples[-1],
            }, ensure_ascii=False))

        send_frame(sock, 0x8, struct.pack("!H", 1000))


if __name__ == "__main__":
    main()
