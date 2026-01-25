"""Kalman filter utilities for a two-state clock model.

The model is intentionally minimal so it can be reused across simulations and
reference fusion experiments without specialized dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ClockState:
    """Represents the estimated clock state.

    Attributes:
        offset: Estimated time offset in seconds.
        drift: Estimated fractional frequency error (seconds/second).
    """

    offset: float
    drift: float


@dataclass
class ClockCovariance:
    """Represents the 2x2 covariance matrix for the clock state.

    The covariance is stored as explicit elements to keep the implementation
    lightweight and explicit (avoiding external linear algebra libraries).
    """

    p00: float
    p01: float
    p10: float
    p11: float

    def as_matrix(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the covariance as a nested tuple for debugging or logging."""

        return ((self.p00, self.p01), (self.p10, self.p11))


class ClockKalmanFilter:
    """A tiny Kalman filter for offset + drift clock tracking.

    State vector:
        x = [offset, drift]

    State transition:
        offset_{k+1} = offset_k + drift_k * dt
        drift_{k+1}  = drift_k

    Measurement:
        z = offset + measurement noise
    """

    def __init__(
        self,
        state: ClockState,
        covariance: ClockCovariance,
        process_noise_offset: float,
        process_noise_drift: float,
    ) -> None:
        """Initialize the filter with model noise settings.

        Args:
            state: Initial clock state estimate.
            covariance: Initial covariance for the state estimate.
            process_noise_offset: Variance of the offset random walk per second.
            process_noise_drift: Variance of the drift random walk per second.
        """

        self._state = state
        self._cov = covariance
        self._q_offset = process_noise_offset
        self._q_drift = process_noise_drift

    @property
    def state(self) -> ClockState:
        """Return the current state estimate."""

        return self._state

    @property
    def covariance(self) -> ClockCovariance:
        """Return the current covariance estimate."""

        return self._cov

    def predict(self, dt: float) -> None:
        """Advance the state estimate without incorporating measurements.

        Args:
            dt: Time step in seconds.
        """

        if dt <= 0:
            raise ValueError("dt must be positive")

        # State prediction: x = F x where F = [[1, dt], [0, 1]]
        new_offset = self._state.offset + self._state.drift * dt
        new_drift = self._state.drift
        self._state = ClockState(offset=new_offset, drift=new_drift)

        # Covariance prediction: P = F P F^T + Q
        # Explicit form keeps intent readable and avoids matrix libraries.
        p00 = self._cov.p00 + dt * (self._cov.p10 + self._cov.p01) + dt**2 * self._cov.p11
        p01 = self._cov.p01 + dt * self._cov.p11
        p10 = self._cov.p10 + dt * self._cov.p11
        p11 = self._cov.p11

        # Add process noise scaled by dt to represent continuous-time noise.
        p00 += self._q_offset * dt
        p11 += self._q_drift * dt

        self._cov = ClockCovariance(p00=p00, p01=p01, p10=p10, p11=p11)

    def update(self, measured_offset: float, measurement_variance: float) -> None:
        """Update the filter with an offset measurement.

        Args:
            measured_offset: Observed offset measurement in seconds.
            measurement_variance: Variance of the measurement noise.
        """

        if measurement_variance <= 0:
            raise ValueError("measurement_variance must be positive")

        # Measurement model: z = H x + v, H = [1, 0]
        # Innovation covariance: S = H P H^T + R = P00 + R
        innovation_cov = self._cov.p00 + measurement_variance
        kalman_gain_offset = self._cov.p00 / innovation_cov
        kalman_gain_drift = self._cov.p10 / innovation_cov

        # Innovation: residual between measured offset and predicted offset.
        residual = measured_offset - self._state.offset

        # State update: x = x + K * residual
        updated_offset = self._state.offset + kalman_gain_offset * residual
        updated_drift = self._state.drift + kalman_gain_drift * residual
        self._state = ClockState(offset=updated_offset, drift=updated_drift)

        # Covariance update: P = (I - K H) P
        p00 = (1 - kalman_gain_offset) * self._cov.p00
        p01 = (1 - kalman_gain_offset) * self._cov.p01
        p10 = self._cov.p10 - kalman_gain_drift * self._cov.p00
        p11 = self._cov.p11 - kalman_gain_drift * self._cov.p01
        self._cov = ClockCovariance(p00=p00, p01=p01, p10=p10, p11=p11)
