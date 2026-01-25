import math

from ai_multi_reference_timekeeping.fusion import Measurement, ReferenceFusion, VirtualClock
from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState


def test_virtual_clock_tracks_fused_measurement():
    kalman = ClockKalmanFilter(
        state=ClockState(offset=0.0, drift=0.0),
        covariance=ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
        process_noise_offset=1e-6,
        process_noise_drift=1e-8,
    )
    clock = VirtualClock(kalman_filter=kalman, fusion=ReferenceFusion())

    measurements = [
        Measurement(name="gnss", offset=0.002, variance=1e-6),
        Measurement(name="ptp", offset=0.001, variance=1e-7),
    ]

    update = clock.step(dt=1.0, measurements=measurements)

    assert update.fused_offset > 0.001
    assert update.fused_offset < 0.002
    assert math.isclose(update.reference_weights["gnss"] + update.reference_weights["ptp"], 1.0, rel_tol=1e-9)
    assert update.state.offset > 0.0
