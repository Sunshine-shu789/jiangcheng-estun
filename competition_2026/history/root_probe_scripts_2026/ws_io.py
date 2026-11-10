#!/usr/bin/env python3
"""Drive controller IO ports over WebSocket and print each response."""

import argparse
import base64
import json
import os
import socket
import struct
import time


def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data.extend(chunk)
    return bytes(data)


def recv_frame(sock):
    fragments = []
    message_opcode = None
    while True:
        first, second = recv_exact(sock, 2)
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", recv_exact(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", recv_exact(sock, 8))[0]
        mask = recv_exact(sock, 4) if masked else None
        payload = bytearray(recv_exact(sock, length))
        if mask:
            for i in range(length):
                payload[i] ^= mask[i % 4]

        if opcode in (0x1, 0x2):
            message_opcode = opcode
            fragments = [bytes(payload)]
        elif opcode == 0x0 and message_opcode is not None:
            fragments.append(bytes(payload))
        else:
            return opcode, bytes(payload)

        if fin:
            return message_opcode, b"".join(fragments)


def send_frame(sock, opcode, payload):
    if isinstance(payload, str):
        payload = payload.encode()
    mask = os.urandom(4)
    length = len(payload)
    header = bytearray([0x80 | opcode])
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    header.extend(mask)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    sock.sendall(header + masked)


def request(sock, payload, reply_seconds=2.0):
    wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    send_frame(sock, 0x1, wire)
    sent = time.monotonic()
    while time.monotonic() - sent < reply_seconds:
        sock.settimeout(max(0.1, reply_seconds - (time.monotonic() - sent)))
        try:
            opcode, data = recv_frame(sock)
        except socket.timeout:
            return None
        if opcode == 0x9:
            send_frame(sock, 0xA, data)
            continue
        if opcode in (0x1, 0x2):
            return json.loads(data.decode())
    return None


def show(tag, reply):
    if reply is None:
        print(f"{tag}: 无响应")
        return
    body = reply.get("data", {})
    print(f"{tag}: code={reply.get('code')} msg={reply.get('msg')!r} data={body}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "steps",
        nargs="+",
        help=(
            '"force:P[:F]" 设置强制标志, "set:P=V" 写入端口值, '
            '"get:P[,P...]" 回读端口值; 冒号后的数字是等待秒数'
        ),
    )
    args = parser.parse_args()

    key = base64.b64encode(os.urandom(16)).decode()
    head = (
        f"GET / HTTP/1.1\r\nHost: {args.host}:{args.port}\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )

    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        sock.sendall(head.encode())
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            buf.extend(sock.recv(4096))
        if not bytes(buf).startswith(b"HTTP/1.1 101"):
            raise SystemExit("WebSocket handshake failed")
        # drain the welcome frame
        try:
            sock.settimeout(1.5)
            recv_frame(sock)
        except socket.timeout:
            pass

        for index, step in enumerate(args.steps):
            if step.startswith("force:"):
                _, port, value, *rest = step.split(":")
                delay = float(rest[0]) if rest else 1.5
                payload = {
                    "id": f"force-{index}",
                    "time": 0,
                    "token": "user:admin",
                    "type": "IOManager",
                    "action": "SetIOForcedFlag",
                    "data": [{"port": int(port), "forced": int(value)}],
                }
            elif step.startswith("set:"):
                _, assign, *rest = step.split(":")
                delay = float(rest[0]) if rest else 1.5
                port, value = assign.split("=", 1)
                payload = {
                    "id": f"set-{index}",
                    "time": 0,
                    "token": "user:admin",
                    "type": "IOManager",
                    "action": "SetIOValue",
                    "data": [{"port": int(port), "value": int(value)}],
                }
            else:
                _, ports, *rest = step.split(":")
                delay = float(rest[0]) if rest else 0.5
                payload = {
                    "id": f"get-{index}",
                    "time": 0,
                    "token": "user:admin",
                    "type": "IOManager",
                    "action": "GetIOValue",
                    "data": [
                        {"port": int(port)}
                        for port in ports.split(",")
                    ],
                }
            print(f"STEP {index}: {step}")
            show("  reply", request(sock, payload))
            time.sleep(delay)

        send_frame(sock, 0x8, struct.pack("!H", 1000))


if __name__ == "__main__":
    main()
