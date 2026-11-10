#!/usr/bin/env python3
"""Backwards-compatible entry point for the pick tasks (B/C/D/E).

Equivalent to ``track1_node``; kept because the C/D/E scaffolding and the
existing taskA workflow scripts already reference ``pick_node``.
"""

from __future__ import annotations

from .track1_node import main as track1_main


def main(argv=None) -> int:
    return track1_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
