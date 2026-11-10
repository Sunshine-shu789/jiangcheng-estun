#!/usr/bin/env python3
"""Hold one cabinet DO bit and verify that its state remains stable."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TASK_A_SRC = Path(
    "/home/sunshine/codex/easton/ros2_ws/src/taskA_grasp"
)
sys.path.insert(0, str(TASK_A_SRC))

from taskA_grasp.modbus import ModbusTcpClient  # noqa: E402


STATE = 0x1002
COMMAND = 0x1300
MASK = 0x1301


def apply(client: ModbusTcpClient, mask: int, value: int) -> None:
    client.write_register(COMMAND, value & mask)
    client.write_register(MASK, mask)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, choices=(16, 17))
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--period", type=float, default=0.25)
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--unit", type=int, default=1)
    args = parser.parse_args()

    bit = 1 << (args.port - 16)
    with ModbusTcpClient(args.host, 502, args.unit) as client:
        apply(client, 0x0003, 0)
        time.sleep(0.2)
        print(f"baseline=0x{client.read_holding_registers(STATE, 1)[0]:04X}")
        apply(client, bit, bit)
        end = time.monotonic() + args.seconds
        try:
            while time.monotonic() < end:
                state = client.read_holding_registers(STATE, 1)[0]
                print(
                    f"DO{args.port}: state=0x{state:04X} "
                    f"bit={'1' if state & bit else '0'}"
                )
                time.sleep(args.period)
        finally:
            apply(client, 0x0003, 0)
            time.sleep(0.2)
            state = client.read_holding_registers(STATE, 1)[0]
            print(f"released=0x{state:04X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
