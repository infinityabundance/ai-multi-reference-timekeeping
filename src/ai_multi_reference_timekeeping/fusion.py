"""Reference fusion logic for building a virtual clock.

The fusion logic keeps measurements explicit and transparent so that the
implementation can be audited and extended for research workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .kalman import ClockKalmanFilter, ClockState


@dataclass(frozen=True)
class Measurement:
    """Represents a single timing reference measurement.

    Attributes:
        name: Identifier for the reference source.
        offset: Measured offset in seconds (positive means reference is ahead).
        variance: Variance of the measurement noise.
    """

    name: str
    offset: float
    variance: float


@dataclass(frozen=True)
class QualityMeasurement(Measurement):
    """Measurement with a quality score (0..1)."""

    quality: float = 1.0


@dataclass
class ClockUpdate:
    """Container for a virtual clock update step."""

    fused_offset: float
    fused_variance: float
    state: ClockState
    reference_weights: Dict[str, float]


class ReferenceFusion:
    """Fuse multiple measurements into a single offset estimate.

    The fusion strategy is inverse-variance weighting, which yields the
    minimum-variance unbiased estimator when measurement errors are independent
    and Gaussian.
    """

    def fuse(self, measurements: Iterable[Measurement]) -> Tuple[float, float, Dict[str, float]]:
        """Compute the fused offset and variance.

        Args:
            measurements: Iterable of Measurement objects.

        Returns:
            A tuple of (fused_offset, fused_variance, normalized_weights).
        """

        # Raw inverse-variance weights before normalization.
        weights: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

        measurement_list = list(measurements)
        if not measurement_list:
            raise ValueError("At least one measurement is required for fusion")

        for measurement in measurement_list:
            if measurement.variance <= 0:
                raise ValueError(f"Measurement variance must be positive for {measurement.name}")
            # Inverse variance weighting (minimum-variance unbiased estimator).
            weight = 1.0 / measurement.variance
            weights[measurement.name] = weight
            weight_total += weight
            weighted_sum += weight * measurement.offset

        fused_offset = weighted_sum / weight_total
        fused_variance = 1.0 / weight_total

        # Normalize weights to sum to 1.0 for reporting.
        normalized_weights = {name: weight / weight_total for name, weight in weights.items()}
        return fused_offset, fused_variance, normalized_weights


class HeuristicFusion(ReferenceFusion):
    """Fuse measurements with an optional quality multiplier.

    The quality score (0..1) down-weights references when their contextual
    reliability is lower. If a measurement lacks a quality attribute, 1.0 is
    assumed.
    """

    def fuse(self, measurements: Iterable[Measurement]) -> Tuple[float, float, Dict[str, float]]:
        # Quality-adjusted inverse-variance weights.
        weights: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

        measurement_list = list(measurements)
        if not measurement_list:
            raise ValueError("At least one measurement is required for fusion")

        for measurement in measurement_list:
            if measurement.variance <= 0:
                raise ValueError(f"Measurement variance must be positive for {measurement.name}")
            # Clamp quality to [0,1] to avoid negative or exaggerated weights.
            quality = max(0.0, min(1.0, getattr(measurement, "quality", 1.0)))
            if quality <= 0:
                continue
            weight = (1.0 / measurement.variance) * quality
            weights[measurement.name] = weight
            weight_total += weight
            weighted_sum += weight * measurement.offset

        if weight_total <= 0:
            raise ValueError("No valid measurements available after applying quality scores")

        fused_offset = weighted_sum / weight_total
        fused_variance = 1.0 / weight_total
        normalized_weights = {name: weight / weight_total for name, weight in weights.items()}
        return fused_offset, fused_variance, normalized_weights


class VirtualClock:
    """Virtual clock estimator that fuses references and runs a Kalman filter."""

    def __init__(
        self,
        kalman_filter: ClockKalmanFilter,
        fusion: ReferenceFusion | None = None,
    ) -> None:
        """Create the virtual clock.

        Args:
            kalman_filter: Kalman filter instance that models the clock.
            fusion: Optional fusion strategy (defaults to inverse variance).
        """

        self._kalman = kalman_filter
        self._fusion = fusion or ReferenceFusion()

    @property
    def state(self) -> ClockState:
        """Expose the current clock state estimate."""

        return self._kalman.state

    def step(self, dt: float, measurements: Iterable[Measurement]) -> ClockUpdate:
        """Advance the virtual clock by one time step.

        The step performs a predict/update cycle:
          1. Predict the clock state forward by dt.
          2. Fuse measurements to obtain a single offset observation.
          3. Update the Kalman filter with the fused offset.

        Args:
            dt: Time step in seconds.
            measurements: Measurements from timing references.

        Returns:
            ClockUpdate with fused data and the updated state.
        """

        self._kalman.predict(dt)
        fused_offset, fused_variance, weights = self._fusion.fuse(measurements)
        self._kalman.update(fused_offset, fused_variance)

        return ClockUpdate(
            fused_offset=fused_offset,
            fused_variance=fused_variance,
            state=self._kalman.state,
            reference_weights=weights,
        )

    def history(self) -> List[Tuple[float, float]]:
        """Return a one-entry history for compatibility with notebooks.

        This method is intentionally simple; it exists so notebooks can call
        virtual_clock.history() and receive an iterable, even if the internal
        implementation is stateful. For multi-step histories, keep track of
        ClockUpdate objects externally.
        """

        return [(self._kalman.state.offset, self._kalman.state.drift)]
