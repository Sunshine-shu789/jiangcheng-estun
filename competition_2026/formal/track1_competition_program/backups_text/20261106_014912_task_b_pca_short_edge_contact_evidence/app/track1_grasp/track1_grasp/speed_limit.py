"""Operator-selectable arm speed profiles with hard configuration caps.

The UI stores only a 0.10--1.00 scale for each task.  The actual Cartesian
and joint speeds always remain bounded by the validated values in the formal
task YAML, so an edited or damaged state file can never raise a speed above
the currently accepted safety limit.  A valid positive scale is also required
so no consumer can accidentally turn an operator setting into a zero-speed
move that never completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from numbers import Real
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping


MIN_SPEED_SCALE = 0.10
MAX_SPEED_SCALE = 1.00
DEFAULT_SPEED_SCALE = 1.00
STATE_VERSION = 1


@dataclass(frozen=True)
class SpeedProfileSpec:
    """One operator profile and the formal configuration that caps it."""

    key: str
    label: str
    config_filename: str

    @property
    def config_path(self) -> Path:
        return project_root() / "config" / self.config_filename


PROFILE_SPECS: dict[str, SpeedProfileSpec] = {
    "A": SpeedProfileSpec(
        key="A",
        label="任务A 十点位巡检",
        config_filename="track1_task_a_legacy.yaml",
    ),
    "B": SpeedProfileSpec(
        key="B",
        label="任务B 五点位拾放",
        config_filename="track1_task_b_five_point_adaptive_safe_20261102.yaml",
    ),
    "C": SpeedProfileSpec(
        key="C",
        label="任务C 无序抓取",
        config_filename="track1_task_c_v11_official_20261102.yaml",
    ),
    "D": SpeedProfileSpec(
        key="D",
        label="任务D 未知物体抓取",
        config_filename="track1_task_d_v11_official_20261102.yaml",
    ),
    "E": SpeedProfileSpec(
        key="E",
        label="任务E 顺序抓取",
        config_filename="track1_live.yaml",
    ),
}
PROFILE_ORDER = tuple(PROFILE_SPECS)


@dataclass(frozen=True)
class SpeedCapabilities:
    """Validated speed bounds shown by the operator UI."""

    profile: str
    label: str
    config_path: Path
    speed_mps: float
    joint_speed_rad_s: float
    joint_speed_safety_limit_rad_s: float
    dry_run: bool
    gripper_backend: str

    @property
    def safe_joint_speed_rad_s(self) -> float:
        return min(self.joint_speed_rad_s, self.joint_speed_safety_limit_rad_s)


def project_root() -> Path:
    """Return the formal competition-program root."""

    return Path(__file__).resolve().parents[3]


def default_state_path() -> Path:
    return project_root() / "state" / "arm_speed_overrides.json"


def normalize_profile(profile: str) -> str:
    key = str(profile).strip().upper()
    if key not in PROFILE_SPECS:
        raise ValueError(
            f"unknown arm speed profile {profile!r}; expected one of "
            f"{', '.join(PROFILE_ORDER)}"
        )
    return key


def normalize_speed_scale(
    value: Any,
    *,
    clamp_above_max: bool = False,
) -> float:
    """Validate a UI/file scale without ever accepting a zero speed."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("speed scale must be a real number")
    scale = float(value)
    if not math.isfinite(scale):
        raise ValueError("speed scale must be finite")
    if scale <= 0.0:
        raise ValueError("speed scale must be greater than zero")
    if scale > MAX_SPEED_SCALE:
        if not clamp_above_max:
            raise ValueError("speed scale must not exceed 1.00 (100%)")
        return MAX_SPEED_SCALE
    if scale < MIN_SPEED_SCALE:
        raise ValueError("speed scale must be at least 0.10 (10%)")
    return scale


def normalize_backend_speed_scale(value: Any) -> float:
    """Defensively interpret a scale used by the motion backend.

    Values above 100% are reduced to the configuration cap.  Missing, zero,
    negative, non-finite or below-minimum values raise so callers can retain a
    previously valid cached value (or the 100% startup default).
    """

    return normalize_speed_scale(value, clamp_above_max=True)


def bounded_cartesian_speed_mps(
    requested_speed_mps: float,
    configured_speed_mps: float,
    scale: float,
) -> float:
    """Return ``min(requested, configured) * safe_scale``."""

    requested = float(requested_speed_mps)
    configured = float(configured_speed_mps)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested Cartesian speed must be positive and finite")
    if not math.isfinite(configured) or configured <= 0.0:
        raise ValueError("configured Cartesian speed must be positive and finite")
    safe_scale = normalize_backend_speed_scale(scale)
    return min(requested, configured) * safe_scale


def bounded_joint_speed_rad_s(
    configured_joint_speed_rad_s: float,
    safety_limit_rad_s: float,
    scale: float,
) -> float:
    """Return the configured joint cap multiplied by a safe scale."""

    configured = float(configured_joint_speed_rad_s)
    safety_limit = float(safety_limit_rad_s)
    if not math.isfinite(configured) or configured <= 0.0:
        raise ValueError("configured joint speed must be positive and finite")
    if not math.isfinite(safety_limit) or safety_limit <= 0.0:
        raise ValueError("joint speed safety limit must be positive and finite")
    safe_scale = normalize_backend_speed_scale(scale)
    return min(configured, safety_limit) * safe_scale


class SpeedLimitStore:
    """Read and atomically update independent task speed scales.

    Readers intentionally do not take an inter-process or GUI lock.  Atomic
    replacement means each read observes a complete old or new JSON document.
    Only the tiny in-process cache update is serialized, and a damaged or
    partially updated file falls back to that cache without blocking motion.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        default_scale: float = DEFAULT_SPEED_SCALE,
    ) -> None:
        self.path = Path(path).expanduser() if path is not None else default_state_path()
        self.default_scale = normalize_speed_scale(default_scale)
        self._lock = threading.RLock()
        self._last_valid: dict[str, float] = {}

    def scale_for(self, profile: str) -> float:
        """Return a fresh non-blocking snapshot or the last safe value."""

        key = normalize_profile(profile)
        try:
            profiles = self._read_profiles()
            if key not in profiles:
                raise KeyError(key)
            entry = profiles[key]
            # The writer stores ``{"scale": ..., "updated_at": ...}``.  Accept
            # a direct number as a legacy/read-only form as well.
            scale_value = entry.get("scale") if isinstance(entry, Mapping) else entry
            scale = normalize_backend_speed_scale(scale_value)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            with self._lock:
                return self._last_valid.get(key, self.default_scale)
        with self._lock:
            self._last_valid[key] = scale
        return scale

    read_scale = scale_for

    def save_scale(self, profile: str, scale: float) -> float:
        """Validate and atomically persist one profile's percentage."""

        key = normalize_profile(profile)
        safe_scale = normalize_speed_scale(scale)
        with self._lock:
            profiles: dict[str, float] = {}
            try:
                for existing_key, existing_value in self._read_profiles().items():
                    try:
                        normalized_key = normalize_profile(existing_key)
                        scale_value = (
                            existing_value.get("scale")
                            if isinstance(existing_value, Mapping)
                            else existing_value
                        )
                        profiles[normalized_key] = normalize_backend_speed_scale(
                            scale_value
                        )
                    except (ValueError, TypeError, KeyError):
                        continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                profiles = {}

            profiles[key] = safe_scale
            payload = {
                "version": STATE_VERSION,
                "profiles": {
                    profile_key: {
                        "scale": profiles[profile_key],
                        "updated_at": datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                    }
                    for profile_key in sorted(profiles)
                },
            }
            self._atomic_write(payload)
            self._last_valid[key] = safe_scale
        return safe_scale

    def _read_profiles(self) -> Mapping[str, Any]:
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("arm speed state must be a JSON object")
        if payload.get("version") != STATE_VERSION:
            raise ValueError(
                f"unsupported arm speed state version {payload.get('version')!r}"
            )
        profiles = payload.get("profiles")
        if not isinstance(profiles, Mapping):
            raise ValueError("arm speed state profiles must be a JSON object")
        return profiles

    def _atomic_write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            self._fsync_directory(self.path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(directory, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)


def load_speed_capabilities(profile: str) -> SpeedCapabilities:
    """Load the validated formal configuration for one operator profile."""

    key = normalize_profile(profile)
    spec = PROFILE_SPECS[key]
    from .config import load_config

    config = load_config(spec.config_path)
    return SpeedCapabilities(
        profile=key,
        label=spec.label,
        config_path=spec.config_path,
        speed_mps=float(config.motion.speed_mps),
        joint_speed_rad_s=float(config.motion.joint_speed_rad_s),
        joint_speed_safety_limit_rad_s=float(
            config.motion.joint_speed_safety_limit_rad_s
        ),
        dry_run=bool(config.motion.dry_run),
        gripper_backend=str(config.gripper.backend),
    )


__all__ = [
    "DEFAULT_SPEED_SCALE",
    "MAX_SPEED_SCALE",
    "MIN_SPEED_SCALE",
    "PROFILE_ORDER",
    "PROFILE_SPECS",
    "STATE_VERSION",
    "SpeedCapabilities",
    "SpeedLimitStore",
    "SpeedProfileSpec",
    "bounded_cartesian_speed_mps",
    "bounded_joint_speed_rad_s",
    "default_state_path",
    "load_speed_capabilities",
    "normalize_backend_speed_scale",
    "normalize_profile",
    "normalize_speed_scale",
    "project_root",
]
