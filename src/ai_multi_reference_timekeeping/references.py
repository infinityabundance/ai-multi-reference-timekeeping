"""Reference source simulators for testing and experimentation.

These classes provide a controlled way to generate offset measurements with
known noise characteristics, which is useful for unit tests and notebook demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import random

from .fusion import Measurement


@dataclass
class ReferenceSource:
    """A simulated timing reference that produces noisy offset measurements.

    Attributes:
        name: Identifier for the reference source.
        noise_std: Standard deviation of measurement noise in seconds.
        bias: Optional constant bias added to measurements.
        drift_bias: Optional drift bias applied over time (seconds/second).
    """

    name: str
    noise_std: float
    bias: float = 0.0
    drift_bias: float = 0.0

    def measure(self, true_offset: float, elapsed: float) -> Measurement:
        """Generate a noisy measurement for the given true offset.

        Args:
            true_offset: The simulated true offset between the clock and reference.
            elapsed: Elapsed time in seconds (used for drift bias).

        Returns:
            A Measurement including the noise variance.
        """

        if self.noise_std <= 0:
            raise ValueError("noise_std must be positive")

        noise = random.gauss(0.0, self.noise_std)
        biased_offset = true_offset + self.bias + self.drift_bias * elapsed + noise
        return Measurement(
            name=self.name,
            offset=biased_offset,
            variance=self.noise_std**2,
        )


class ReferenceEnsemble:
    """Collection of reference sources that generate measurements together."""

    def __init__(self, *sources: ReferenceSource) -> None:
        if not sources:
            raise ValueError("At least one reference source is required")
        self._sources = sources

    def snapshot(self, true_offset: float, elapsed: float) -> list[Measurement]:
        """Generate measurements from all sources for the current step."""

        return [source.measure(true_offset, elapsed) for source in self._sources]


class DeterministicReference:
    """Reference source driven by a deterministic callback.

    The deterministic path is useful for tests, because it avoids reliance on
    random number generation while still exercising the fusion logic.
    """

    def __init__(self, name: str, variance: float, generator: Callable[[float], float]) -> None:
        if variance <= 0:
            raise ValueError("variance must be positive")
        self._name = name
        self._variance = variance
        self._generator = generator

    def measure(self, elapsed: float) -> Measurement:
        """Generate a deterministic measurement for a given elapsed time."""

        return Measurement(
            name=self._name,
            offset=self._generator(elapsed),
            variance=self._variance,
        )
