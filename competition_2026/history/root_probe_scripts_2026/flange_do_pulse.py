#!/usr/bin/env python3
"""Drive Codroid cabinet or flange DO0/DO1 with readback.

Usage:
    python3 flange_do_pulse.py sweep --target board --hold 1.0
    python3 flange_do_pulse.py sweep --target tool --hold 1.0
    python3 flange_do_pulse.py open --hold 1.0
    python3 flange_do_pulse.py close --hold 1.0

``board`` maps to ports 16/17 (0x1002, bits 0/1); ``tool`` maps to
flange ports 46/47 (0x1003, bits 14/15).
"""

import argparse
import sys
import time

sys.path.insert(0, "/home/sunshine/codex/easton/ros2_ws/src/taskA_grasp")

from taskA_grasp.modbus import ModbusTcpClient  # noqa: E402

TARGETS = {
    "board": {
        "state": 0x1002,
        "command": 0x1300,
        "mask": 0x1301,
        "do0": 1 << 0,
        "do1": 1 << 1,
        "do0_name": "DO0/port16",
        "do1_name": "DO1/port17",
    },
    "tool": {
        "state": 0x1003,
        "command": 0x1302,
        "mask": 0x1303,
        "do0": 1 << 14,
        "do1": 1 << 15,
        "do0_name": "flangeDO0/port46",
        "do1_name": "flangeDO1/port47",
    },
}


def state(client, target):
    return client.read_holding_registers(target["state"], 1)[0]


def apply(client, target, mask, value):
    client.write_register(target["command"], value & mask)
    client.write_register(target["mask"], mask)
    time.sleep(0.25)
    return state(client, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("open", "close", "sweep", "release"))
    parser.add_argument("--target", choices=tuple(TARGETS), default="board")
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--host", default="192.168.101.100")
    parser.add_argument("--port", type=int, default=502)
    args = parser.parse_args()
    target = TARGETS[args.target]
    both = target["do0"] | target["do1"]

    with ModbusTcpClient(args.host, args.port, 1, timeout_s=3.0) as client:
        print(
            f"target={args.target} start 0x{target['state']:04X}="
            f"0x{state(client, target):04X}"
        )
        print(f"both low -> 0x{apply(client, target, both, 0):04X}")

        if args.action == "sweep":
            lines = (
                (target["do0_name"], target["do0"]),
                (target["do1_name"], target["do1"]),
            )
        elif args.action == "release":
            lines = ()
        else:
            bit = target["do0"] if args.action == "open" else target["do1"]
            lines = ((args.action, bit),)

        for name, bit in lines:
            print(f"{name} HIGH -> 0x{apply(client, target, bit, bit):04X}")
            time.sleep(args.hold)
            print(f"{name} low  -> 0x{apply(client, target, bit, 0):04X}")
            time.sleep(args.hold)

        print(
            f"end 0x{target['state']:04X}="
            f"0x{apply(client, target, both, 0):04X}"
        )


if __name__ == "__main__":
    main()
