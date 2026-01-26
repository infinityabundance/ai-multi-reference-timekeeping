"""Sensor signal characterization helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunningStats:
    """Online mean/variance estimator."""

    mean: float = 0.0
    m2: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return self.variance**0.5


class SensorCharacterization:
    """Track reference residuals to characterize sensor noise."""

    def __init__(self) -> None:
        self._stats: dict[str, RunningStats] = {}

    def update(self, name: str, residual: float) -> None:
        stats = self._stats.setdefault(name, RunningStats())
        stats.update(residual)

    def z_score(self, name: str, residual: float) -> float:
        stats = self._stats.get(name)
        if stats is None or stats.count < 2 or stats.stddev == 0:
            return 0.0
        return (residual - stats.mean) / stats.stddev
