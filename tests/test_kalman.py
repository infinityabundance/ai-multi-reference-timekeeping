import math

import pytest

from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState


def test_kalman_predict_and_update():
    state = ClockState(offset=0.0, drift=1e-6)
    covariance = ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0)
    kalman = ClockKalmanFilter(
        state=state,
        covariance=covariance,
        process_noise_offset=0.0,
        process_noise_drift=0.0,
    )

    kalman.predict(dt=10.0)

    assert math.isclose(kalman.state.offset, 1e-5, rel_tol=1e-9)
    assert math.isclose(kalman.state.drift, 1e-6, rel_tol=1e-9)

    kalman.update(measured_offset=2e-5, measurement_variance=1.0)

    # The offset should move toward the measurement.
    assert kalman.state.offset > 1e-5
    # The drift should adjust upward because offset residuals correlate with drift.
    assert kalman.state.drift > 1e-6


def test_kalman_rejects_nonpositive_variance():
    state = ClockState(offset=0.0, drift=0.0)
    covariance = ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0)
    kalman = ClockKalmanFilter(
        state=state,
        covariance=covariance,
        process_noise_offset=0.0,
        process_noise_drift=0.0,
    )

    with pytest.raises(ValueError):
        kalman.update(measured_offset=0.0, measurement_variance=0.0)
