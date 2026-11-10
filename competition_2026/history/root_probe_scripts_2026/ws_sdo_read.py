#!/usr/bin/env python3
"""Read controller/RS485 diagnostic values without changing robot motion state.

The only "setparam" call mirrors the Web UI's SDO read button: it writes the
requested SDO name to Robot/Control/readSDO, which triggers a read. All other
calls are common/getparam reads.
"""

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
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(header + masked)


def request(sock, payload, timeout):
    wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    print("SEND", wire)
    send_frame(sock, 0x1, wire)
    deadline = time.monotonic() + timeout
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
            print("RECV", text)
            try:
                return json.loads(text)
            except ValueError:
                return {"code": "raw", "msg": text}
    return None


def make_message(action, data):
    return {
        "id": f"sdo-read-{time.time_ns()}",
        "time": int(time.time() * 1000),
        "token": "user:admin",
        "type": "common",
        "action": action,
        "data": data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--sdo", default="CmdInterpCnt")
    parser.add_argument("--reply-seconds", type=float, default=8.0)
    parser.add_argument("--result-delay", type=float, default=1.0)
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="additional common/getparam path; may be repeated",
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

        try:
            sock.settimeout(1.5)
            opcode, data = recv_frame(sock)
            print("WELCOME", opcode, data.decode(errors="replace"))
        except socket.timeout:
            pass

        trigger = make_message(
            "setparam",
            [{"path": "Robot/Control/readSDO", "value": args.sdo}],
        )
        trigger_reply = request(sock, trigger, args.reply_seconds)
        print("TRIGGER", trigger_reply)

        time.sleep(args.result_delay)

        result_path = f"Robot/Info/SDO/{args.sdo}"
        paths = [result_path] + args.extra_path
        result = request(sock, make_message("getparam", paths), args.reply_seconds)
        print("RESULT", json.dumps(result, ensure_ascii=False, indent=2))

        send_frame(sock, 0x8, struct.pack("!H", 1000))


if __name__ == "__main__":
    main()
