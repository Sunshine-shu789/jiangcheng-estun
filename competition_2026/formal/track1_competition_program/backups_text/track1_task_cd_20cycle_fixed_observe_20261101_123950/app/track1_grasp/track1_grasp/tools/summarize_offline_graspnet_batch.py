#!/usr/bin/env python3
"""Summarize a batch of offline GraspNet candidate runs.

This tool only reads existing offline reports. It does not connect to hardware,
send motion commands, or alter candidates and safety thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


FRAME_DIR_RE = re.compile(r"^frame_(\d+)$")


def _number(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _frame_sort_key(path: Path) -> tuple[int, str]:
    match = FRAME_DIR_RE.match(path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _candidate_payload(record: dict[str, Any]) -> dict[str, Any]:
    score = record.get("score_camera") or {}
    world = record.get("world_clearance") or {}
    graspnet = record.get("graspnet") or {}
    ik = record.get("ik") or {}
    tcp = world.get("tcp_base_m") or [None, None, None]
    nms_kept = bool(record.get("nms_kept"))
    if not nms_kept:
        ik_status = "NMS suppressed"
    elif ik.get("skipped"):
        ik_status = "skipped"
    elif ik.get("success"):
        ik_status = "success"
    elif ik:
        ik_status = "failed"
    else:
        ik_status = "not evaluated"
    return {
        "candidate_id": record.get("candidate_id"),
        "quality": _number(score.get("quality")),
        "graspnet_reference_width_m": _number(graspnet.get("width_m")),
        "selected_opening_m": _number(score.get("selected_opening_m")),
        "measured_width_m": _number(score.get("measured_width_m")),
        "tcp_base_m": list(tcp),
        "support_minimum_clearance_m": _number(
            world.get("support_minimum_clearance_m")
        ),
        "tcp_workspace_margin_m": _number(world.get("tcp_workspace_margin_m")),
        "nms_kept": nms_kept,
        "ik_status": ik_status,
        "ik_success": bool(ik.get("success")),
        "ik_position_error_m": _number(ik.get("position_error_m")),
        "ik_orientation_error_rad": _number(ik.get("orientation_error_rad")),
        "ik_at_joint_limit": bool(ik.get("at_joint_limit")),
        "ik_max_joint_delta_deg": _number(ik.get("max_joint_delta_deg")),
    }


def _read_frame(frame_dir: Path) -> dict[str, Any]:
    payload_path = frame_dir / "offline_candidates.json"
    match = FRAME_DIR_RE.match(frame_dir.name)
    frame = int(match.group(1)) if match else None
    result: dict[str, Any] = {
        "frame": frame,
        "frame_dir": str(frame_dir),
        "status": "missing",
        "error": None,
    }
    if not payload_path.is_file():
        result["error"] = "offline_candidates.json is missing"
        return result

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = "invalid"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    summary = payload.get("summary") or {}
    support = payload.get("support_plane") or {}
    candidates = payload.get("candidates") or []
    final_candidates = [
        _candidate_payload(record)
        for record in candidates
        if bool(record.get("final_feasible"))
    ]
    final_count = int(summary.get("final_feasible_count", 0) or 0)
    nms_survivor_count = int(summary.get("nms_survivor_count", 0) or 0)
    ik_evaluated = int(summary.get("ik_evaluated_count", 0) or 0)
    ik_success = int(summary.get("ik_success_count", 0) or 0)
    offline_only = bool(payload.get("offline_only"))
    motion_attempted = bool(payload.get("motion_attempted"))
    hardware_connected = bool(payload.get("hardware_connected"))
    support_plane_valid = bool(summary.get("support_plane_valid"))

    checks = {
        "offline_only": offline_only,
        "motion_not_attempted": not motion_attempted,
        "hardware_not_connected": not hardware_connected,
        "support_plane_valid": support_plane_valid,
        # NMS is applied before IK.  The number of IK evaluations must match
        # the NMS survivors, not the pre-NMS final-feasible candidate count.
        "ik_complete": nms_survivor_count == ik_evaluated,
        "has_ik_success": ik_success > 0,
        "has_final_candidate": final_count > 0,
    }
    result.update(
        {
            "status": "complete",
            "offline_only": offline_only,
            "motion_attempted": motion_attempted,
            "quality_validated": bool(payload.get("quality_validated")),
            "motion_authorized": bool(payload.get("motion_authorized")),
            "hardware_connected": hardware_connected,
            "support_plane_valid": support_plane_valid,
            "candidate_count": int(summary.get("candidate_count", 0) or 0),
            "scoring_error_count": int(summary.get("scoring_error_count", 0) or 0),
            "graspnet_reference_score_feasible_count": int(
                summary.get("graspnet_reference_score_feasible_count", 0) or 0
            ),
            "aperture_search_trial_count": int(
                summary.get("aperture_search_trial_count", 0) or 0
            ),
            "score_feasible_count": int(summary.get("score_feasible_count", 0) or 0),
            "world_clearance_feasible_count": int(
                summary.get("world_clearance_feasible_count", 0) or 0
            ),
            "final_feasible_count": final_count,
            "nms_survivor_count": nms_survivor_count,
            "ik_evaluated_count": ik_evaluated,
            "ik_success_count": ik_success,
            "support_plane_inlier_ratio": _number(support.get("inlier_ratio")),
            "support_plane_tilt_deg": _number(support.get("tilt_deg")),
            "checks": checks,
            "final_candidates": final_candidates,
        }
    )
    return result


def summarize_batch(batch_root: Path) -> dict[str, Any]:
    batch_root = Path(batch_root)
    frame_dirs = sorted(
        (
            path
            for path in batch_root.iterdir()
            if path.is_dir() and FRAME_DIR_RE.match(path.name)
        ),
        key=_frame_sort_key,
    )
    frames = [_read_frame(path) for path in frame_dirs]
    completed = [item for item in frames if item["status"] == "complete"]
    safety_ok = [
        item
        for item in completed
        if item["checks"]["offline_only"]
        and item["checks"]["motion_not_attempted"]
        and item["checks"]["hardware_not_connected"]
    ]
    support_plane_ok = [item for item in completed if item["support_plane_valid"]]
    frames_with_final = [
        item for item in completed if item["final_feasible_count"] > 0
    ]
    frames_with_ik = [item for item in completed if item["ik_success_count"] > 0]
    final_candidates = [
        {"frame": item["frame"], **candidate}
        for item in completed
        for candidate in item["final_candidates"]
    ]

    def total(key: str) -> int:
        return sum(int(item.get(key, 0) or 0) for item in completed)

    return {
        "schema_version": 1,
        "batch_root": str(batch_root),
        "frames_discovered": len(frames),
        "frames_completed": len(completed),
        "frames_safety_ok": len(safety_ok),
        "frames_support_plane_valid": len(support_plane_ok),
        "frames_with_final_candidate": len(frames_with_final),
        "frames_with_ik_success": len(frames_with_ik),
        "grasp_frame_coverage": (
            len(frames_with_final) / len(completed) if completed else 0.0
        ),
        "candidate_count": total("candidate_count"),
        "scoring_error_count": total("scoring_error_count"),
        "graspnet_reference_score_feasible_count": total(
            "graspnet_reference_score_feasible_count"
        ),
        "aperture_search_trial_count": total("aperture_search_trial_count"),
        "score_feasible_count": total("score_feasible_count"),
        "world_clearance_feasible_count": total("world_clearance_feasible_count"),
        "final_feasible_count": total("final_feasible_count"),
        "nms_survivor_count": total("nms_survivor_count"),
        "ik_evaluated_count": total("ik_evaluated_count"),
        "ik_success_count": total("ik_success_count"),
        "ik_success_rate": (
            total("ik_success_count") / total("ik_evaluated_count")
            if total("ik_evaluated_count")
            else 0.0
        ),
        "final_candidates": final_candidates,
        "frames": frames,
        "caveats": [
            "This is an offline diagnostic and does not authorize robot motion.",
            "A frame without a feasible candidate is not automatically a software failure; thresholds were not relaxed.",
            "IK success does not include full whole-arm motion planning or real collision validation.",
            "The support plane and camera pose still require site calibration before live use.",
        ],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Offline GraspNet Aperture Multi-frame Summary",
        "",
        f"- batch root: `{summary['batch_root']}`",
        f"- frames discovered/completed: `{summary['frames_discovered']}` / `{summary['frames_completed']}`",
        f"- frames with final candidate: `{summary['frames_with_final_candidate']}`",
        f"- frames with IK success: `{summary['frames_with_ik_success']}`",
        f"- grasp-frame coverage: `{summary['grasp_frame_coverage']:.3%}`",
        f"- final feasible candidates: `{summary['final_feasible_count']}`",
        f"- NMS survivors: `{summary['nms_survivor_count']}`",
        f"- IK evaluated/success: `{summary['ik_evaluated_count']}` / `{summary['ik_success_count']}`",
        f"- aggregate IK success rate: `{summary['ik_success_rate']:.3%}`",
        f"- safety-boundary clean frames: `{summary['frames_safety_ok']}`",
        f"- valid support-plane frames: `{summary['frames_support_plane_valid']}`",
        "",
        "## Per-frame results",
        "",
        "| frame | candidates | score feasible | clearance feasible | final | NMS | IK evaluated | IK success | support plane | status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in summary["frames"]:
        if item["status"] != "complete":
            lines.append(
                f"| {item['frame']} | n/a | n/a | n/a | n/a | n/a | n/a | {item['status']} |"
            )
            continue
        checks = item["checks"]
        if not checks["has_final_candidate"]:
            status = "no final candidate"
        elif not checks["ik_complete"]:
            status = "IK incomplete"
        elif not checks["has_ik_success"]:
            status = "no IK success"
        else:
            status = "ok"
        lines.append(
            "| {frame} | {candidate_count} | {score_feasible_count} | "
            "{world_clearance_feasible_count} | {final_feasible_count} | "
            "{nms_survivor_count} | {ik_evaluated_count} | {ik_success_count} | "
            "{support} | {status} |".format(
                frame=item["frame"],
                candidate_count=item["candidate_count"],
                score_feasible_count=item["score_feasible_count"],
                world_clearance_feasible_count=item[
                    "world_clearance_feasible_count"
                ],
                final_feasible_count=item["final_feasible_count"],
                nms_survivor_count=item["nms_survivor_count"],
                ik_evaluated_count=item["ik_evaluated_count"],
                ik_success_count=item["ik_success_count"],
                support="yes" if item["support_plane_valid"] else "no",
                status=status,
            )
        )

    lines.extend(
        [
            "",
            "## Final candidates",
            "",
        ]
    )
    if not summary["final_candidates"]:
        lines.append("**No final feasible candidates.**")
    else:
        lines.extend(
            [
                "| frame | candidate | quality | ref width mm | opening mm | TCP base m | clearance mm | TCP z margin mm | NMS/IK |",
                "|---:|---|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for item in summary["final_candidates"]:
            tcp = ", ".join(
                "n/a" if value is None else f"{float(value):.4f}"
                for value in item["tcp_base_m"]
            )
            lines.append(
                "| {frame} | {candidate_id} | {quality:.5f} | {ref:.3f} | "
                "{opening:.3f} | {tcp} | {clear:.3f} | {margin:.3f} | {ik} |".format(
                    frame=item["frame"],
                    candidate_id=item["candidate_id"],
                    quality=item["quality"],
                    ref=1000.0 * item["graspnet_reference_width_m"],
                    opening=1000.0 * item["selected_opening_m"],
                    tcp=tcp,
                    clear=1000.0 * item["support_minimum_clearance_m"],
                    margin=1000.0 * item["tcp_workspace_margin_m"],
                    ik=item["ik_status"],
                )
            )

    lines.extend(["", "## Safety boundary", ""])
    lines.extend(f"- {caveat}" for caveat in summary["caveats"])
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, frames: Iterable[dict[str, Any]]) -> None:
    columns = [
        "frame",
        "status",
        "support_plane_valid",
        "candidate_count",
        "scoring_error_count",
        "graspnet_reference_score_feasible_count",
        "aperture_search_trial_count",
        "score_feasible_count",
        "world_clearance_feasible_count",
        "final_feasible_count",
        "nms_survivor_count",
        "ik_evaluated_count",
        "ik_success_count",
        "support_plane_inlier_ratio",
        "support_plane_tilt_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in frames:
            writer.writerow({key: item.get(key) for key in columns})


def write_summary(batch_root: Path) -> dict[str, Any]:
    summary = summarize_batch(batch_root)
    batch_root = Path(batch_root)
    (batch_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (batch_root / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    _write_csv(batch_root / "summary.csv", summary["frames"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    args = parser.parse_args(argv)
    if not args.batch_root.is_dir():
        parser.error(f"batch root is not a directory: {args.batch_root}")
    summary = write_summary(args.batch_root)
    print(
        json.dumps(
            {
                "summary_json": str(args.batch_root / "summary.json"),
                "frames_completed": summary["frames_completed"],
                "frames_with_final_candidate": summary[
                    "frames_with_final_candidate"
                ],
                "final_feasible_count": summary["final_feasible_count"],
                "ik_success_count": summary["ik_success_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
