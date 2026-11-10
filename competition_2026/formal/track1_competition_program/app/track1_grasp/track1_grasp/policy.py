"""Task C/D/E target-selection policies.

Selection is deliberately separated from detection.  A detector answers
"what is visible"; this module answers "is this object a legal next target for
the current competition task".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import TargetConfig
from .domain import Detection, Selection


def _matches_order_entry(detection: Detection, entry: str) -> bool:
    if "#" not in entry:
        return detection.label == entry
    label, instance = entry.split("#", 1)
    return detection.label == label and detection.instance_id == instance


@dataclass
class OrderedCursor:
    entries: tuple[str, ...]
    index: int = 0
    attempts: int = 0

    @property
    def complete(self) -> bool:
        return self.index >= len(self.entries)

    @property
    def current(self) -> str | None:
        if self.complete:
            return None
        return self.entries[self.index]

    def success(self, max_attempts: int) -> None:
        self.index += 1
        self.attempts = 0

    def failure(self, max_attempts: int) -> bool:
        """Return True while the same ordered target may be retried."""
        self.attempts += 1
        if self.attempts >= max_attempts:
            self.index += 1
            self.attempts = 0
            return False
        return True


@dataclass
class TargetPolicy:
    config: TargetConfig
    order: tuple[str, ...] = ()
    ordered: OrderedCursor | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.config.policy == "ordered":
            self.ordered = OrderedCursor(self.order or self.config.order)

    @property
    def complete(self) -> bool:
        return self.ordered is not None and self.ordered.complete

    def select(self, detections: list[Detection]) -> Selection:
        if self.config.policy == "any":
            return self._best(detections, "accepted by unordered policy")
        if self.config.policy == "known":
            allowed = set(self.config.known_classes)
            candidates = [d for d in detections if d.label in allowed]
            return self._best(candidates, "accepted known class")
        if self.config.policy == "known_and_unseen":
            known = set(self.config.known_classes)
            unknown = set(self.config.unknown_labels)
            candidates: list[Detection] = []
            for detection in detections:
                if known and detection.label in known:
                    candidates.append(detection)
                    continue
                if detection.label in unknown and (
                    detection.graspability >= self.config.unknown_min_graspability
                ):
                    candidates.append(detection)
            return self._best(candidates, "accepted known/unseen object")
        if self.config.policy == "ordered":
            return self._ordered_select(detections)
        raise ValueError(f"unsupported policy: {self.config.policy!r}")

    def commit_success(self, selection: Selection) -> None:
        if self.ordered is not None and selection.accepted:
            self.ordered.success(self.config.max_attempts_per_target)

    def commit_failure(self) -> None:
        if self.ordered is not None:
            self.ordered.failure(self.config.max_attempts_per_target)

    @staticmethod
    def _best(candidates: list[Detection], reason: str) -> Selection:
        if not candidates:
            return Selection(None, False, "no detection satisfies target policy")
        best = max(candidates, key=lambda detection: detection.score())
        return Selection(best, True, reason)

    def _ordered_select(self, detections: list[Detection]) -> Selection:
        assert self.ordered is not None
        entry = self.ordered.current
        if entry is None:
            return Selection(None, False, "ordered target list is complete")
        candidates = [d for d in detections if _matches_order_entry(d, entry)]
        if not candidates:
            return Selection(
                None,
                False,
                f"ordered target {entry!r} is not visible",
                self.ordered.index,
            )
        best = max(candidates, key=lambda detection: detection.score())
        return Selection(
            best,
            True,
            f"ordered target {entry!r}",
            self.ordered.index,
        )
