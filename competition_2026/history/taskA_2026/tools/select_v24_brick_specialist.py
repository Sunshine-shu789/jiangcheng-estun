#!/usr/bin/env python3
"""按固定门禁从 expert checkpoint 扫描表中选出可上线权重。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def failures(row: dict[str, str]) -> list[str]:
    checks = [
        (as_int(row, "new_hits") == as_int(row, "new_total"), "新姿态未全中"),
        (as_int(row, "old_hits") == as_int(row, "old_total"), "旧泡沫砖未全中"),
        (as_int(row, "old_val_hits") == as_int(row, "old_val_total"), "旧泡沫砖留出未全中"),
        (as_int(row, "empty_hold_fp") == 0, "空桌/杂物留出有误报"),
        (as_int(row, "empty_fp") == 0, "空桌/杂物全集有误报"),
        (as_int(row, "other_fp") == 0, "其它物品被误报为泡沫砖"),
    ]
    return [message for ok, message in checks if not ok]


def rank_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        -as_float(row, "new_min_conf"),
        -as_float(row, "new_max_conf"),
        as_float(row, "worst_empty_conf"),
        as_float(row, "other_worst_conf"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_csv")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    with Path(args.scan_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("empty scan csv")

    qualified = []
    report = []
    for row in rows:
        failed = failures(row)
        item = {
            "checkpoint": row["checkpoint"],
            "qualified": not failed,
            "failures": failed,
            "new_min_conf": as_float(row, "new_min_conf"),
            "new_max_conf": as_float(row, "new_max_conf"),
            "worst_empty_conf": as_float(row, "worst_empty_conf"),
            "other_worst_conf": as_float(row, "other_worst_conf"),
        }
        report.append(item)
        if not failed:
            qualified.append(row)

    qualified.sort(key=rank_key)
    if args.out_json:
        output = {
            "selected": qualified[0]["checkpoint"] if qualified else None,
            "threshold": {
                "new_hits": "all",
                "old_hits": "all",
                "old_val_hits": "all",
                "empty_fp": 0,
                "empty_hold_fp": 0,
                "other_fp": 0,
            },
            "rows": report,
        }
        Path(args.out_json).write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if not qualified:
        for item in report:
            print(f"[拒绝] {item['checkpoint']}: {', '.join(item['failures'])}")
        return 2

    best = qualified[0]
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"[选中] {best['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
