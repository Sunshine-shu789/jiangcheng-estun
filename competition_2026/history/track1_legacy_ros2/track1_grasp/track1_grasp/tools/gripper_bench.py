"""Bench bring-up tool for the electric gripper, safe by default.

``status`` is read-only: it sends the controller's status request and prints
the decoded reply.  ``open``/``close``/``cycle`` move the fingers and are
therefore gated behind ``--really``, matching the safety convention already
used by ``taskA_grasp.estun_rs485``.

Nothing in this tool touches the robot arm.  Use it on the bench, with the
gripper off the flange, before it is mounted.
"""

from __future__ import annotations

import argparse
import sys
import time

from ..config import load_config
from ..gripper import build_gripper


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gripper_bench",
        description="Read or exercise the electric gripper without moving the arm.",
    )
    parser.add_argument(
        "action",
        choices=("status", "open", "close", "cycle", "measure"),
        help="status is read-only; the other actions need --really",
    )
    parser.add_argument("--config", required=True, help="track1 YAML config path")
    parser.add_argument(
        "--really",
        action="store_true",
        help="confirm the fingers are clear and actually send motion frames",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="number of open/close cycles for the cycle action (default: 1)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="override speed_rad_s for this run",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="override the travel angle in degrees for this run",
    )
    parser.add_argument(
        "--probe-angle",
        type=float,
        default=3600.0,
        help="measure: angle used to drive each end to its hard stop (default: 3600)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    overrides: dict[str, object] = {}
    if args.speed is not None:
        overrides["speed_rad_s"] = args.speed
    if args.angle is not None:
        overrides["open_angle_deg"] = args.angle
        overrides["close_angle_deg"] = args.angle
    if overrides:
        from dataclasses import replace

        config = replace(config, gripper=replace(config.gripper, **overrides))

    if args.action != "status" and not args.really:
        print(
            "refusing to move the fingers without --really\n"
            "confirm the jaws are clear, nothing is between them and nothing\n"
            "can fall when they open, then re-run with --really",
            file=sys.stderr,
        )
        return 2

    log = lambda message: print(message, flush=True)  # noqa: E731
    gripper = build_gripper(config.gripper, log=log)

    try:
        if args.action == "status":
            return _status(gripper)
        if args.action == "open":
            gripper.open()
            print("[bench] jaws commanded open")
            return _status(gripper)
        if args.action == "close":
            gripper.close()
            print("[bench] jaws commanded closed")
            return _status(gripper)
        if args.action == "measure":
            return _measure(gripper, args.probe_angle)
        return _cycle(gripper, args.cycles)
    finally:
        gripper.close_backend()


def _status(gripper) -> int:
    read = getattr(gripper, "read_status", None)
    if not callable(read):
        print(
            f"[bench] {type(gripper).__name__} has no feedback channel; "
            "cannot report jaw state",
            file=sys.stderr,
        )
        return 1
    status = read()
    if status is None:
        print(
            "[bench] no valid status reply: check power, wiring, baud rate,\n"
            "and that the controller's DIP switch selects the right transport",
            file=sys.stderr,
        )
        return 1
    print(
        f"[bench] id={status.device_id} arrived={status.arrived} "
        f"angle={status.angle_deg:.1f} deg speed={status.speed_rad_s:.1f} rad/s"
    )
    return 0


def _measure(gripper, probe_angle: float) -> int:
    """Drive both ends to their hard stops and report the real travel.

    ``probe_angle`` is deliberately larger than any known travel (the largest
    shipped firmware is 9.5 turns = 3420 deg) so both ends are reached; the
    controller's own 堵转保护 stops the motor there.  The difference between
    the two reported angles is the travel the flashed firmware actually uses,
    which is also the value that belongs in ``open_angle_deg`` /
    ``close_angle_deg``.

    Requires ``read_status`` - without feedback there is nothing to measure.
    """

    if not callable(getattr(gripper, "read_status", None)):
        print(
            f"[bench] {type(gripper).__name__} has no feedback channel; "
            "cannot measure travel",
            file=sys.stderr,
        )
        return 1

    print(
        f"[bench] driving to each hard stop with a {probe_angle:.0f} deg probe;\n"
        "        the 堵转保护 will stop the motor, expect a short stall at each end"
    )

    from dataclasses import replace as _replace

    probe = _replace(
        gripper.config,
        open_angle_deg=probe_angle,
        close_angle_deg=probe_angle,
        motion_mode=2,  # relative, so each probe is a full sweep from wherever it is
    )
    original = gripper.config
    gripper.config = probe
    try:
        gripper.open()
        after_open = gripper.read_status()
        gripper.close()
        after_close = gripper.read_status()
    finally:
        gripper.config = original

    if after_open is None or after_close is None:
        print(
            "[bench] no valid status reply during the sweep; cannot measure",
            file=sys.stderr,
        )
        return 1

    # The motor keeps turning the same way while closing, so the two reported
    # angles are comparable directly.  Absolute value guards a reversed sign.
    travel = abs(after_close.angle_deg - after_open.angle_deg)
    turns = travel / 360.0
    print()
    print(f"[bench] open  hard stop: angle={after_open.angle_deg:.1f} deg")
    print(f"[bench] close hard stop: angle={after_close.angle_deg:.1f} deg")
    print(f"[bench] measured travel = {travel:.1f} deg = {turns:.2f} turns")
    print()
    print("        nearest shipped firmware:")
    for candidate in (4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.8, 6.1, 7.0, 7.5, 8.5, 9.5):
        if abs(candidate - turns) <= 0.15:
            print(f"          {candidate} turns ({candidate * 360:.0f} deg)")
    print()
    print("        put the measured value in the gripper config:")
    print(f"          open_angle_deg: {travel:.1f}")
    print(f"          close_angle_deg: {travel:.1f}")
    print(
        "        then re-run `close --really` on an empty gripper and use that\n"
        "        angle for hold_min_angle_deg (slightly below it)"
    )
    return 0


def _cycle(gripper, cycles: int) -> int:
    for index in range(max(1, cycles)):
        print(f"[bench] cycle {index + 1}/{max(1, cycles)}: opening")
        gripper.open()
        time.sleep(0.5)
        print(f"[bench] cycle {index + 1}/{max(1, cycles)}: closing")
        gripper.close()
        time.sleep(0.5)
    print("[bench] cycle finished; watch that open really opens and close really closes")
    return _status(gripper)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
