#!/usr/bin/env python3
"""Small dependency-free WebSocket probe for the robot controller."""

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
    return fin, opcode, bytes(payload)


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


def receive_until(sock, deadline):
    while time.monotonic() < deadline:
        timeout = max(0.1, min(0.5, deadline - time.monotonic()))
        sock.settimeout(timeout)
        try:
            fin, opcode, payload = recv_frame(sock)
        except socket.timeout:
            continue
        if opcode == 0x8:
            print("CLOSE", payload.decode(errors="replace"))
            return False
        if opcode == 0x9:
            send_frame(sock, 0xA, payload)
            print("PING", payload.decode(errors="replace"))
            continue
        if opcode == 0xA:
            print("PONG", payload.decode(errors="replace"))
            continue
        print("FRAME", "fin" if fin else "more", hex(opcode), payload.decode(errors="replace"))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--path", default="/")
    parser.add_argument("--send")
    parser.add_argument("--welcome-seconds", type=float, default=2.0)
    parser.add_argument("--reply-seconds", type=float, default=5.0)
    args = parser.parse_args()

    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET {args.path} HTTP/1.1\r\n"
        f"Host: {args.host}:{args.port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )

    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        sock.sendall(request.encode())
        response = bytearray()
        while b"\r\n\r\n" not in response:
            response.extend(sock.recv(4096))
        head, _, remainder = bytes(response).partition(b"\r\n\r\n")
        print(head.decode(errors="replace"))
        if not head.startswith(b"HTTP/1.1 101"):
            raise SystemExit("WebSocket handshake failed")

        deadline = time.monotonic() + args.welcome_seconds
        if not receive_until(sock, deadline):
            return

        if args.send is not None:
            payload = json.loads(args.send)
            wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            print("SEND", wire)
            send_frame(sock, 0x1, wire)
            receive_until(sock, time.monotonic() + args.reply_seconds)

        send_frame(sock, 0x8, struct.pack("!H", 1000))


if __name__ == "__main__":
    main()
