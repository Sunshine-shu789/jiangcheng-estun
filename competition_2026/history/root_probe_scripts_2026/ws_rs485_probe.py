#!/usr/bin/env python3
"""Probe a controller RS485 plugin in one WebSocket session.

Sequence: flushReadBuffer -> init -> write(gripper status query) -> read.
Only reads status; never commands the gripper to move.
"""

import argparse
import base64
import json
import os
import socket
import struct
import time


STATUS_QUERY = bytes([0xEB, 0x90, 0x01, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00])


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
    return opcode, bytes(payload)


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


def show(tag, reply):
    if reply is None:
        print(f"{tag}: no reply")
        return
    body = reply.get("data")
    print(f"{tag}: code={reply.get('code')} msg={reply.get('msg')!r} data={body}")
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        raw = bytes(int(v) & 0xFF for v in body["data"])
        print(f"{tag}: bytes={raw.hex(' ')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--plugin", default="EC2RS485",
                       choices=("EC2RS485", "RS485"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--data-bit", type=int, default=8)
    parser.add_argument("--parity", type=int, default=0)
    parser.add_argument("--stop-bit", type=int, default=0)
    parser.add_argument("--reply-seconds", type=float, default=6.0)
    parser.add_argument("--gap", type=float, default=0.15)
    parser.add_argument("--read-lengths", default="9,32,1")
    parser.add_argument("--timeout-us", type=int, default=None,
                       help="plugins reads with timeout in microseconds; omit for the 3s default")
    parser.add_argument("--read-attempts", type=int, default=3)
    parser.add_argument("--frame", default=STATUS_QUERY.hex(" "))
    parser.add_argument("--skip-flush", action="store_true")
    args = parser.parse_args()

    frame = bytes(int(tok, 16) for tok in args.frame.replace(",", " ").split())
    print("frame:", frame.hex(" "))

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
        try:
            sock.settimeout(1.5)
            opcode, data = recv_frame(sock)
            print("WELCOME", data.decode(errors="replace"))
        except socket.timeout:
            pass

        def call(tag, action, data=None):
            payload = {
                "id": f"{tag}-{time.time():.3f}",
                "time": 0,
                "token": "user:admin",
                "type": args.plugin,
                "action": action,
            }
            if data is not None:
                payload["data"] = data
            print(f"--- {tag}")
            reply = request(sock, payload, args.reply_seconds)
            show(tag, reply)
            return reply

        if not args.skip_flush:
            call("flushReadBuffer", "flushReadBuffer", None)

        if args.plugin == "EC2RS485":
            call("init", "init", {
                "baudrate": args.baudrate,
                "stopBit": args.stop_bit,
                "parity": args.parity,
                "dataBit": args.data_bit,
            })
            call("write", "write", list(frame))
        else:
            call("init", "init", {
                "baudrate": args.baudrate,
                "stopBit": args.stop_bit,
                "parity": args.parity,
                "dataBit": args.data_bit,
            })
            call("write", "write", list(frame))

        time.sleep(args.gap)

        index = 0
        for text in args.read_lengths.split(","):
            length = int(text)
            if args.plugin == "EC2RS485":
                data = {"length": length}
                if args.timeout_us is not None:
                    data["timeout"] = args.timeout_us
            else:
                data = {"length": length}
                if args.timeout_us is not None:
                    data["timeout"] = args.timeout_us
            for attempt in range(args.read_attempts):
                reply = call(f"read-{index}-len{length}-try{attempt}", "read", data)
                index += 1
                body = (reply or {}).get("data")
                if isinstance(body, dict) and body.get("data"):
                    send_frame(sock, 0x8, struct.pack("!H", 1000))
                    return

        send_frame(sock, 0x8, struct.pack("!H", 1000))


if __name__ == "__main__":
    main()
