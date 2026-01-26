"""Clock quality metrics for timekeeping evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class HoldoverStats:
    """Holdover behavior summary."""

    max_offset: float
    rms_offset: float
    duration: float


def tdev(offsets: Sequence[float], tau: int) -> float:
    """Compute a simple TDEV estimate for a given tau (samples)."""

    if tau <= 0:
        raise ValueError("tau must be positive")
    if len(offsets) < 3 * tau:
        raise ValueError("Not enough samples for TDEV")
    diffs = []
    for i in range(len(offsets) - 3 * tau + 1):
        avg1 = sum(offsets[i : i + tau]) / tau
        avg2 = sum(offsets[i + tau : i + 2 * tau]) / tau
        avg3 = sum(offsets[i + 2 * tau : i + 3 * tau]) / tau
        diffs.append(avg1 - 2 * avg2 + avg3)
    variance = sum(diff * diff for diff in diffs) / (2.0 * len(diffs))
    return variance**0.5


def mtie(offsets: Sequence[float], window: int) -> float:
    """Compute MTIE for a sliding window of length `window`."""

    if window <= 0:
        raise ValueError("window must be positive")
    if len(offsets) < window:
        raise ValueError("Not enough samples for MTIE")
    max_tie = 0.0
    for i in range(len(offsets) - window + 1):
        segment = offsets[i : i + window]
        tie = max(segment) - min(segment)
        if tie > max_tie:
            max_tie = tie
    return max_tie


def holdover_stats(offsets: Iterable[float], sample_interval: float) -> HoldoverStats:
    """Compute holdover stats over a sequence of offsets."""

    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    values = list(offsets)
    if not values:
        raise ValueError("offsets must be non-empty")
    max_offset = max(abs(value) for value in values)
    rms_offset = (sum(value * value for value in values) / len(values)) ** 0.5
    duration = sample_interval * (len(values) - 1)
    return HoldoverStats(max_offset=max_offset, rms_offset=rms_offset, duration=duration)
