"""Search a safe CAD-consistent opening for a GraspNet grasp pose.

GraspNet's 17-D ``width`` output is treated as a reference/upper-bound
opening.  It is not assumed to equal the opening at which the physical jaws
make contact with the visible object.  For each predicted pose this module
samples candidate openings between the CAD limit and the network reference,
then applies the full jaw contact/collision scorer to every trial.

The search never relaxes the scorer.  It only changes the opening supplied to
the CAD geometry checks, while retaining the original GraspNet width as a hard
upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from track1_grasp.jaw_grasp_scoring import (
    JawAwareGraspScorer,
    JawGraspCandidate,
    JawGraspScore,
)
from track1_grasp.jaw_profile import JawProfile


@dataclass(frozen=True)
class JawOpeningSearchConfig:
    """Deterministic aperture-search bounds and resource limit."""

    enabled: bool = True
    step_m: float = 0.0005
    max_trials: int = 96
    minimum_extra_opening_m: float = 0.0005
    maximum_measured_delta_m: float | None = None

    def __post_init__(self) -> None:
        step = float(self.step_m)
        extra = float(self.minimum_extra_opening_m)
        maximum_delta = (
            None
            if self.maximum_measured_delta_m is None
            else float(self.maximum_measured_delta_m)
        )
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step_m must be positive and finite")
        if not math.isfinite(extra) or extra < 0.0:
            raise ValueError("minimum_extra_opening_m must be finite and non-negative")
        if maximum_delta is not None and (
            not math.isfinite(maximum_delta) or maximum_delta < extra
        ):
            raise ValueError(
                "maximum_measured_delta_m must be finite and >= minimum_extra_opening_m"
            )
        if int(self.max_trials) < 3:
            raise ValueError("max_trials must be at least three")
        object.__setattr__(self, "step_m", step)
        object.__setattr__(self, "minimum_extra_opening_m", extra)
        object.__setattr__(self, "maximum_measured_delta_m", maximum_delta)
        object.__setattr__(self, "max_trials", int(self.max_trials))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JawOpeningSearchConfig":
        data = payload.get("aperture_search", payload)
        if not isinstance(data, Mapping):
            raise ValueError("aperture_search must be a mapping")
        defaults = cls()
        values: dict[str, Any] = {}
        for name in defaults.__dataclass_fields__:
            if name in data:
                values[name] = data[name]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class JawOpeningTrial:
    opening_m: float
    candidate: JawGraspCandidate
    score: JawGraspScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "opening_m": float(self.opening_m),
            "feasible": bool(self.score.feasible),
            "quality": float(self.score.quality),
            "reason_codes": list(self.score.reason_codes),
            "measured_width_m": self.score.measured_width_m,
            "contact_left_points": int(self.score.contact_left_points),
            "contact_right_points": int(self.score.contact_right_points),
            "target_penetration_points": int(self.score.target_penetration_points),
            "obstacle_finger_points": int(self.score.obstacle_finger_points),
            "obstacle_cavity_points": int(self.score.obstacle_cavity_points),
            "obstacle_clearance_m": float(self.score.obstacle_clearance_m),
        }


@dataclass(frozen=True)
class JawOpeningSearchResult:
    reference_candidate: JawGraspCandidate
    reference_score: JawGraspScore
    trials: tuple[JawOpeningTrial, ...]
    feasible_trials: tuple[JawOpeningTrial, ...]
    lower_opening_m: float | None
    upper_opening_m: float | None
    status: str
    reason_codes: tuple[str, ...]
    config: JawOpeningSearchConfig

    @property
    def best_feasible(self) -> JawOpeningTrial | None:
        return self.feasible_trials[0] if self.feasible_trials else None

    @property
    def best_diagnostic(self) -> JawOpeningTrial | None:
        if not self.trials:
            return None
        return max(
            self.trials,
            key=lambda trial: (
                bool(trial.score.feasible),
                float(trial.score.quality),
                -len(trial.score.reason_codes),
                -abs(float(trial.opening_m)),
            ),
        )

    @property
    def selected_trial(self) -> JawOpeningTrial | None:
        return self.best_feasible or self.best_diagnostic

    def to_dict(self, *, include_trials: bool = True) -> dict[str, Any]:
        selected = self.selected_trial
        payload: dict[str, Any] = {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "graspnet_reference_width_m": float(self.reference_candidate.width_m),
            "measured_width_m": self.reference_score.measured_width_m,
            "lower_opening_m": self.lower_opening_m,
            "upper_opening_m": self.upper_opening_m,
            "trial_count": len(self.trials),
            "local_feasible_count": len(self.feasible_trials),
            "feasible_openings_m": [float(trial.opening_m) for trial in self.feasible_trials],
            "selected_opening_m": None if selected is None else float(selected.opening_m),
            "selected_quality": None if selected is None else float(selected.score.quality),
        }
        if include_trials:
            payload["trials"] = [trial.to_dict() for trial in self.trials]
        return payload


def _opening_grid(
    lower_m: float,
    upper_m: float,
    *,
    step_m: float,
    max_trials: int,
) -> tuple[float, ...]:
    """Return an inclusive deterministic grid, coarsened only at the cap."""

    span = max(0.0, float(upper_m) - float(lower_m))
    nominal_count = int(math.floor(span / float(step_m) + 1e-9)) + 1
    if nominal_count <= max_trials:
        values = [float(lower_m) + index * float(step_m) for index in range(nominal_count)]
        if not values or values[-1] < float(upper_m) - 1e-9:
            values.append(float(upper_m))
        return tuple(values)
    values = np.linspace(float(lower_m), float(upper_m), int(max_trials))
    return tuple(float(value) for value in values)


def search_jaw_opening(
    profile: JawProfile,
    scorer: JawAwareGraspScorer,
    candidate: JawGraspCandidate,
    target_points: Sequence[Sequence[float]] | np.ndarray | None,
    obstacle_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    config: JawOpeningSearchConfig | None = None,
) -> JawOpeningSearchResult:
    """Find CAD-feasible openings without exceeding GraspNet's reference width."""

    config = config or JawOpeningSearchConfig()
    if not config.enabled:
        reference_score = scorer.score(candidate, target_points, obstacle_points)
        return JawOpeningSearchResult(
            reference_candidate=candidate,
            reference_score=reference_score,
            trials=(),
            feasible_trials=(),
            lower_opening_m=None,
            upper_opening_m=None,
            status="disabled",
            reason_codes=("opening_search_disabled",),
            config=config,
        )

    # The reference score establishes the visible width and provides a useful
    # fail-closed diagnostic even if no alternate opening can be evaluated.
    reference_score = scorer.score(candidate, target_points, obstacle_points)
    measured_width = reference_score.measured_width_m
    if measured_width is None:
        return JawOpeningSearchResult(
            reference_candidate=candidate,
            reference_score=reference_score,
            trials=(),
            feasible_trials=(),
            lower_opening_m=None,
            upper_opening_m=None,
            status="width_unmeasured",
            reason_codes=tuple(dict.fromkeys((*reference_score.reason_codes, "width_unmeasured"))),
            config=config,
        )

    contact_x = profile.contact_plane_x_m(candidate.depth_m)
    profile_limits = profile.section(contact_x, 0.0)
    cad_lower = (
        profile_limits.closed_opening_m
        + scorer.config.minimum_closure_margin_m
    )
    cad_upper = (
        profile_limits.open_opening_m
        - scorer.config.opening_reserve_m
    )
    measured_delta = (
        scorer.config.width_tolerance_m
        if config.maximum_measured_delta_m is None
        else config.maximum_measured_delta_m
    )
    lower = max(cad_lower, float(measured_width) + config.minimum_extra_opening_m)
    upper = min(
        float(candidate.width_m),
        cad_upper,
        float(measured_width) + float(measured_delta),
    )
    if upper < lower - 1e-9:
        return JawOpeningSearchResult(
            reference_candidate=candidate,
            reference_score=reference_score,
            trials=(),
            feasible_trials=(),
            lower_opening_m=float(lower),
            upper_opening_m=float(upper),
            status="no_opening_bounds",
            reason_codes=("no_opening_bounds",),
            config=config,
        )

    openings = _opening_grid(
        lower,
        upper,
        step_m=config.step_m,
        max_trials=config.max_trials,
    )
    trials: list[JawOpeningTrial] = []
    for opening in openings:
        trial_candidate = replace(candidate, selected_opening_m=opening)
        trial_score = scorer.score(trial_candidate, target_points, obstacle_points)
        trials.append(
            JawOpeningTrial(
                opening_m=float(opening),
                candidate=trial_candidate,
                score=trial_score,
            )
        )

    feasible = tuple(
        sorted(
            (trial for trial in trials if trial.score.feasible),
            key=lambda trial: (-float(trial.score.quality), float(trial.opening_m)),
        )
    )
    if feasible:
        status = "selected"
        reasons: tuple[str, ...] = ()
    else:
        status = "no_feasible_opening"
        diagnostic = max(
            trials,
            key=lambda trial: (
                float(trial.score.quality),
                -len(trial.score.reason_codes),
            ),
        )
        reasons = tuple(
            dict.fromkeys(("no_feasible_selected_opening", *diagnostic.score.reason_codes))
        )
    return JawOpeningSearchResult(
        reference_candidate=candidate,
        reference_score=reference_score,
        trials=tuple(trials),
        feasible_trials=feasible,
        lower_opening_m=float(lower),
        upper_opening_m=float(upper),
        status=status,
        reason_codes=reasons,
        config=config,
    )


__all__ = [
    "JawOpeningSearchConfig",
    "JawOpeningSearchResult",
    "JawOpeningTrial",
    "search_jaw_opening",
]
