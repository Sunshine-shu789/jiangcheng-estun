#!/usr/bin/env python3
"""Operator GUI for bounded, per-task arm speed scaling.

This tool never imports ROS, starts a task, takes ``task.lock`` or commands an
arm.  It only writes a small scale file read by the next pre-checked motion
segment of an already running task.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..speed_limit import (
    MIN_SPEED_SCALE,
    PROFILE_ORDER,
    PROFILE_SPECS,
    SpeedCapabilities,
    SpeedLimitStore,
    load_speed_capabilities,
)


def format_speed(value: float) -> str:
    return f"{value:.3f} m/s"


def format_joint_speed(value: float) -> str:
    return f"{value:.3f} rad/s"


class ArmSpeedControlApp:
    """Small Tkinter UI with a hard 10--100 percent range."""

    def __init__(self, root, *, store: SpeedLimitStore | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.root = root
        self.store = store or SpeedLimitStore()
        self.capabilities: dict[str, SpeedCapabilities] = {
            key: load_speed_capabilities(key) for key in PROFILE_ORDER
        }
        self.display_to_profile = {
            f"{key}  {spec.label.split(' ', 1)[-1]}": key
            for key, spec in PROFILE_SPECS.items()
        }

        root.title("机械臂安全调速")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        frame = ttk.Frame(root, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="选择任务").grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=6
        )
        self.profile_var = tk.StringVar(value=next(iter(self.display_to_profile)))
        self.profile_combo = ttk.Combobox(
            frame,
            textvariable=self.profile_var,
            values=tuple(self.display_to_profile),
            state="readonly",
            width=32,
        )
        self.profile_combo.grid(row=0, column=1, sticky="ew", pady=6)
        self.profile_combo.bind("<<ComboboxSelected>>", self._profile_changed)

        ttk.Label(frame, text="速度比例").grid(
            row=1, column=0, sticky="nw", padx=(0, 12), pady=(14, 6)
        )
        scale_frame = ttk.Frame(frame)
        scale_frame.grid(row=1, column=1, sticky="ew", pady=(14, 6))
        scale_frame.columnconfigure(0, weight=1)
        self.percent_var = tk.IntVar(value=100)
        self.scale_slider = tk.Scale(
            scale_frame,
            from_=round(MIN_SPEED_SCALE * 100),
            to=100,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.percent_var,
            command=lambda _value: self._update_readout(),
            length=360,
            showvalue=True,
        )
        self.scale_slider.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Button(
            scale_frame,
            text="恢复 100%",
            command=self._restore_full_speed,
        ).grid(row=1, column=1, sticky="e", pady=(8, 0))

        self.speed_var = tk.StringVar()
        self.joint_var = tk.StringVar()
        self.limit_var = tk.StringVar()
        ttk.Label(frame, text="当前主运动速度").grid(
            row=2, column=0, sticky="w", padx=(0, 12), pady=4
        )
        ttk.Label(frame, textvariable=self.speed_var).grid(
            row=2, column=1, sticky="w", pady=4
        )
        ttk.Label(frame, text="当前关节速度上限").grid(
            row=3, column=0, sticky="w", padx=(0, 12), pady=4
        )
        ttk.Label(frame, textvariable=self.joint_var).grid(
            row=3, column=1, sticky="w", pady=4
        )
        ttk.Label(frame, text="100% 安全上限").grid(
            row=4, column=0, sticky="w", padx=(0, 12), pady=4
        )
        ttk.Label(frame, textvariable=self.limit_var).grid(
            row=4, column=1, sticky="w", pady=4
        )

        ttk.Separator(frame, orient="horizontal").grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=12
        )
        self.safety_var = tk.StringVar()
        ttk.Label(
            frame,
            textvariable=self.safety_var,
            justify="left",
            wraplength=470,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 10))

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=7, column=0, columnspan=2, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="保存后从下一段运动生效")
        ttk.Label(button_frame, textvariable=self.status_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(
            button_frame,
            text="保存并应用",
            command=self._save,
        ).grid(row=0, column=1, padx=(12, 8))
        ttk.Button(button_frame, text="关闭", command=root.destroy).grid(
            row=0, column=2
        )

        self._profile_changed()
        root.update_idletasks()
        width = root.winfo_reqwidth()
        height = root.winfo_reqheight()
        root.geometry(f"{width}x{height}")

    @property
    def selected_profile(self) -> str:
        return self.display_to_profile[self.profile_var.get()]

    def _profile_changed(self, _event=None) -> None:
        capabilities = self.capabilities[self.selected_profile]
        scale = self.store.scale_for(self.selected_profile)
        self.percent_var.set(int(round(scale * 100)))
        self._update_readout()
        self.status_var.set(
            f"已载入 {self.percent_var.get()}%；保存后从下一段运动生效"
        )

    def _update_readout(self) -> None:
        if not hasattr(self, "capabilities"):
            return
        capabilities = self.capabilities[self.selected_profile]
        scale = max(
            round(MIN_SPEED_SCALE * 100),
            min(100, int(self.percent_var.get())),
        ) / 100.0
        self.speed_var.set(format_speed(capabilities.speed_mps * scale))
        self.joint_var.set(
            format_joint_speed(capabilities.safe_joint_speed_rad_s * scale)
        )
        self.limit_var.set(
            f"{format_speed(capabilities.speed_mps)} / "
            f"{format_joint_speed(capabilities.safe_joint_speed_rad_s)}"
        )
        safety = (
            "100% 仅等于当前正式配置中已验证的安全速度，不能超过。"
            "最低为 10%，不会保存 0%。"
        )
        if capabilities.dry_run:
            safety += " 本任务当前为 dry-run，不会因调速而启用真实运动。"
        if capabilities.gripper_backend == "mock":
            safety += " 夹爪当前为 mock。"
        safety += " 硬件急停仍是最终保护。"
        self.safety_var.set(safety)

    def _save(self) -> None:
        percent = max(
            round(MIN_SPEED_SCALE * 100),
            min(100, int(self.percent_var.get())),
        )
        try:
            saved = self.store.save_scale(self.selected_profile, percent / 100.0)
        except (OSError, ValueError) as exc:
            self.status_var.set(f"保存失败：{exc}")
            return
        self.status_var.set(
            f"已保存 {int(round(saved * 100))}%；下一段运动生效"
        )

    def _restore_full_speed(self) -> None:
        self.percent_var.set(100)
        self._update_readout()
        self._save()


def _check() -> int:
    store = SpeedLimitStore()
    for key in PROFILE_ORDER:
        capabilities = load_speed_capabilities(key)
        scale = store.scale_for(key)
        print(
            f"{key}: {int(round(scale * 100))}%  "
            f"cartesian={format_speed(capabilities.speed_mps * scale)}  "
            f"joint={format_joint_speed(capabilities.safe_joint_speed_rad_s * scale)}  "
            f"limit={format_speed(capabilities.speed_mps)}/"
            f"{format_joint_speed(capabilities.safe_joint_speed_rad_s)}  "
            f"dry_run={capabilities.dry_run}  "
            f"gripper={capabilities.gripper_backend}  "
            f"config={Path(capabilities.config_path).name}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate all five formal profiles and print current scales without opening a GUI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return _check()
    try:
        import tkinter as tk
    except ImportError as exc:
        print(f"无法加载 Tkinter：{exc}", file=sys.stderr)
        return 2
    root = tk.Tk()
    ArmSpeedControlApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
