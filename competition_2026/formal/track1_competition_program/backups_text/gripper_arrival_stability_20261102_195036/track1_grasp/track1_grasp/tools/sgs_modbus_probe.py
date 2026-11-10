"""Read-only ESTUN SGS Modbus/TCP probe.

This command has no write flags and constructs a transport whose
``allow_writes`` value is permanently False.  It is therefore safe to use for
the first field connection, but it still does not prove that the HMI service
port exposes LAN3 or that a motion-control command path is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from track1_grasp.sgs_modbus import (
    ModbusError,
    SgsModbusClient,
    register_map_as_dict,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read ESTUN SGS holding registers with FC3 only. "
            "This program cannot write coils or registers."
        )
    )
    parser.add_argument(
        "--host",
        help="Controller IP. Required for a live probe; omitted for offline modes.",
    )
    parser.add_argument("--port", type=int, default=502, help="Modbus TCP port")
    parser.add_argument("--unit-id", type=int, default=1, help="Modbus unit id")
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Connect/response timeout in seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one machine-readable JSON document",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print the built-in register map and exit",
    )
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="Run against an in-process fake SGS; opens only a local loopback socket",
    )
    return parser


def _print_human(payload: dict[str, object]) -> None:
    firmware = payload["firmware"]
    core = payload["core"]
    status = core["status"]
    io = payload["io"]
    commands = payload["do_commands"]
    print("[read-only] ESTUN SGS Modbus TCP")
    print(f"firmware: {firmware}")
    print(
        "core: "
        f"move_rate={core['move_rate']} robot_mode={core['robot_mode']} "
        f"safety_mode={core['safety_mode']}"
    )
    print(
        "status: "
        f"raw=0x{status['raw']:04x} "
        f"program={status['program_state']} moving={status['moving']} "
        f"powered={status['powered']} fault={status['fault']} "
        f"estop={status['emergency_stop']} unknown_mask=0x"
        f"{status['unknown_mask']:04x}"
    )
    print(
        "io: "
        f"board_do=0x{io['board_do']:04x} tool_do=0x{io['tool_do']:04x} "
        f"board_cmd=0x{commands['board_value']:04x} "
        f"board_mask=0x{commands['board_mask']:04x} "
        f"tool_cmd=0x{commands['tool_value']:04x} "
        f"tool_mask=0x{commands['tool_mask']:04x}"
    )
    if status["emergency_stop"]:
        print("warning: emergency-stop bit is active", file=sys.stderr)
    if status["fault"]:
        print("warning: controller fault bit is active", file=sys.stderr)
    if status["unknown_mask"]:
        print(
            "warning: status word contains undefined bits; update the schema "
            "before using them",
            file=sys.stderr,
        )


def _offline_snapshot() -> dict[str, object]:
    from track1_grasp.sgs_modbus import (
        REG_BOARD_DO,
        REG_BOARD_DO_COMMAND,
        REG_BOARD_DO_MASK,
        REG_MAJOR_VERSION,
        REG_MINOR_VERSION,
        REG_MOVE_RATE,
        REG_ROBOT_COMMAND,
        REG_ROBOT_MODE,
        REG_ROBOT_SAFETY_MODE,
        REG_ROBOT_STATUS,
        REG_TOOL_DO,
        REG_TOOL_DO_COMMAND,
        REG_TOOL_DO_MASK,
    )
    from track1_grasp.sgs_modbus_fake import FakeSgsModbusServer

    registers = {
        REG_MAJOR_VERSION: 1,
        REG_MINOR_VERSION: 6,
        REG_MOVE_RATE: 100,
        REG_ROBOT_MODE: 1,
        REG_ROBOT_SAFETY_MODE: 1,
        REG_ROBOT_STATUS: (1 << 1) | (1 << 10),
        REG_ROBOT_COMMAND: 0,
        REG_BOARD_DO: 0x0001,
        REG_TOOL_DO: 0x0000,
        REG_BOARD_DO_COMMAND: 0x0001,
        REG_BOARD_DO_MASK: 0x0001,
        REG_TOOL_DO_COMMAND: 0x0000,
        REG_TOOL_DO_MASK: 0x0000,
    }
    with FakeSgsModbusServer(registers=registers) as fake:
        with SgsModbusClient.from_host("127.0.0.1", port=fake.port) as reader:
            return reader.read_snapshot().as_dict()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.schema:
        payload: dict[str, object] = {
            "mode": "schema",
            "writes_enabled": False,
            "registers": register_map_as_dict(),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print("[read-only] SGS register map")
            for name, item in register_map_as_dict().items():
                print(
                    f"{name:24s} {item['address_hex']} "
                    f"count={item['count']} access={item['access']}"
                )
        return 0

    try:
        if args.offline_smoke:
            snapshot = _offline_snapshot()
            payload = {"mode": "offline_smoke", "writes_enabled": False, **snapshot}
        else:
            if not args.host:
                raise ValueError("--host is required for a live read-only probe")
            with SgsModbusClient.from_host(
                args.host,
                port=args.port,
                unit_id=args.unit_id,
                timeout_s=args.timeout,
            ) as reader:
                payload = {
                    "mode": "live_read_only",
                    "writes_enabled": False,
                    **reader.read_snapshot().as_dict(),
                }
    except (ValueError, ModbusError) as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
