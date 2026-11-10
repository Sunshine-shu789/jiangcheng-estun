#!/usr/bin/env python3
"""Exercise the controller's RS485Flange CallFunction test interface.

The sequence only initializes the port, sends a gripper status query, and
reads. It never sends a gripper motion command.
"""

import argparse
import base64
import json
import os
import socket
import struct
import time


STATUS_QUERY = "EB 90 01 07 00 00 00 00 00"


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
            for index in range(length):
                payload[index] ^= mask[index % 4]

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
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    sock.sendall(header + masked)


def request(sock, payload, reply_seconds):
    wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    print("SEND", wire)
    send_frame(sock, 0x1, wire)
    started = time.monotonic()
    deadline = started + reply_seconds
    while time.monotonic() < deadline:
        sock.settimeout(max(0.1, deadline - time.monotonic()))
        try:
            opcode, data = recv_frame(sock)
        except socket.timeout:
            return None
        if opcode == 0x9:
            send_frame(sock, 0xA, data)
            continue
        if opcode == 0x8:
            return {"code": "closed", "msg": data.decode(errors="replace")}
        if opcode in (0x1, 0x2):
            text = data.decode(errors="replace")
            print(f"RECV ({time.monotonic() - started:.3f}s)", text)
            try:
                return json.loads(text)
            except ValueError:
                return {"code": "raw", "msg": text}
    return None


def call_function(sock, instance, method, param, reply_seconds):
    payload = {
        "id": f"{method}-{time.time():.3f}",
        "time": 0,
        "token": "user:admin",
        "type": "System",
        "action": "CallFunction",
        "data": {
            "function": [instance, method],
            "async": False,
            "param": param,
        },
    }
    print(f"--- {instance}/{method} param={param!r}")
    reply = request(sock, payload, reply_seconds)
    if reply is None:
        print("RESULT: no reply")
    else:
        print(
            "RESULT:",
            f"code={reply.get('code')!r}",
            f"msg={reply.get('msg')!r}",
            f"data={reply.get('data')!r}",
        )
    return reply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--instance", default="RS485Flange")
    parser.add_argument("--reply-seconds", type=float, default=6.0)
    parser.add_argument("--init-param", default="[2,0,0,8]")
    parser.add_argument("--frame", default=STATUS_QUERY)
    parser.add_argument("--gap", type=float, default=0.3)
    parser.add_argument("--skip-clear", action="store_true")
    parser.add_argument("--skip-init", action="store_true")
    parser.add_argument("--read-attempts", type=int, default=3)
    args = parser.parse_args()

    init_param = json.loads(args.init_param)
    frame = " ".join(args.frame.replace(",", " ").split())

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
            sock.settimeout(1.5)
            _, welcome = recv_frame(sock)
            print("WELCOME", welcome.decode(errors="replace"))
        except socket.timeout:
            pass

        if not args.skip_clear:
            call_function(
                sock, args.instance, "testClear", ["this is testing"],
                args.reply_seconds,
            )
            time.sleep(args.gap)

        if not args.skip_init:
            call_function(
                sock, args.instance, "testInit", init_param, args.reply_seconds,
            )
            time.sleep(args.gap)

        call_function(
            sock, args.instance, "testWrite", frame, args.reply_seconds,
        )
        time.sleep(args.gap)

        for attempt in range(args.read_attempts):
            call_function(
                sock, args.instance, "testRead", ["this is testing"],
                args.reply_seconds,
            )
            if attempt + 1 < args.read_attempts:
                time.sleep(args.gap)

        send_frame(sock, 0x8, struct.pack("!H", 1000))


if __name__ == "__main__":
    main()
