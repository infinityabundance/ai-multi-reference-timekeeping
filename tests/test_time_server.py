from __future__ import annotations

from ai_multi_reference_timekeeping.fusion import Measurement, ReferenceFusion, VirtualClock
from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState
from ai_multi_reference_timekeeping.time_server import (
    AudioFeatureSensor,
    LinearInferenceModel,
    LightweightInferenceModel,
    SensorAggregator,
    SensorFrame,
    SlewDriftDetector,
    TimeServer,
)


class StaticReference:
    def __init__(self, name: str, offset: float, variance: float) -> None:
        self._name = name
        self._offset = offset
        self._variance = variance

    def sample(self, frame: SensorFrame) -> Measurement:
        del frame
        return Measurement(name=self._name, offset=self._offset, variance=self._variance)


def _make_clock() -> VirtualClock:
    kalman = ClockKalmanFilter(
        state=ClockState(offset=0.0, drift=0.0),
        covariance=ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
        process_noise_offset=1e-4,
        process_noise_drift=1e-6,
    )
    return VirtualClock(kalman_filter=kalman, fusion=ReferenceFusion())


def test_lightweight_inference_model_adjusts_variance() -> None:
    model = LightweightInferenceModel(nominal_hum_hz=60.0)
    frame = SensorFrame(timestamp=0.0, ac_hum_hz=55.0, gps_lock=False)
    variance = model.adjusted_variance(1.0, frame)
    assert variance > 1.0


def test_slew_drift_detector_estimates_linear_drift() -> None:
    detector = SlewDriftDetector(window=5)
    for step in range(5):
        estimate = detector.update(float(step), 0.1 * step)
    assert estimate.samples == 5
    assert 0.09 < estimate.drift < 0.11


def test_time_server_step_reports_drift_hint() -> None:
    clock = _make_clock()
    reference = StaticReference(name="ref", offset=0.2, variance=0.5)

    class TempSensor:
        def sample(self) -> dict[str, float]:
            return {"temperature_c": 25.0}

    server = TimeServer(clock, references=[reference], sensors=SensorAggregator(TempSensor()))
    update, frame, drift_estimate, drift_hint = server.step(1.0)

    assert frame.temperature_c == 25.0
    assert update.fused_offset == 0.2
    assert drift_estimate.samples == 1
    assert drift_hint == 0.0


def test_linear_inference_model_scales_variance() -> None:
    model = LinearInferenceModel(feature_weights={"temperature_c": 0.5}, reference_bias={"ref": 0.1})
    frame = SensorFrame(timestamp=0.0, temperature_c=10.0)
    scaled = model.adjusted_variance(1.0, frame, "ref")
    assert scaled > 1.0


def test_audio_feature_sensor_extracts_activity() -> None:
    class ToneSource:
        def sample(self) -> tuple[list[float], int]:
            sample_rate = 8000
            samples = [0.5] * 128
            return samples, sample_rate

    sensor = AudioFeatureSensor(ToneSource())
    features = sensor.sample()
    assert features["ambient_audio_db"] is not None
