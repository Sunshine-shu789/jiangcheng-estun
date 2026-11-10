#!/usr/bin/env python3
"""Call one EtherCAT plugin action over the controller WebSocket.

This helper is intended for read-only diagnostics. It does not send motion
commands or change the robot state.
"""

import argparse
import base64
import json
import os
import socket
import struct
import time

from ws_rs485_probe import recv_frame, request, send_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--type", default="EtherCAT")
    parser.add_argument("--action", required=True)
    parser.add_argument("--data-json", default="null")
    parser.add_argument("--reply-seconds", type=float, default=5.0)
    args = parser.parse_args()

    payload = {
        "id": f"ethercat-{time.time_ns()}",
        "time": int(time.time() * 1000),
        "token": "user:admin",
        "type": args.type,
        "action": args.action,
        "data": json.loads(args.data_json),
    }

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

        reply = request(sock, payload, args.reply_seconds)
        print("REPLY", json.dumps(reply, ensure_ascii=False))
        send_frame(sock, 0x8, struct.pack("!H", 1000))


if __name__ == "__main__":
    main()
