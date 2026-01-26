from __future__ import annotations

import pytest

from ai_multi_reference_timekeeping.fusion import Measurement, ReferenceFusion, VirtualClock
from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState
from ai_multi_reference_timekeeping.time_server import (
    AudioFeatureSensor,
    LinearInferenceModel,
    LightweightInferenceModel,
    MlVarianceModel,
    EnvironmentalSensor,
    I2CEnvironmentalSensor,
    SerialReference,
    SensorAggregator,
    SensorFrame,
    SlewDriftDetector,
    SecurityMonitor,
    SensorValidator,
    TimeServer,
)
from ai_multi_reference_timekeeping.models import SensorFrameModel


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
    assert update.fused_offset == pytest.approx(0.2)
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
    assert features["ac_hum_phase_rad"] is not None
    assert features["ac_hum_uncertainty"] is not None


def test_sensor_validator_blocks_out_of_range() -> None:
    validator = SensorValidator(max_samples_per_sec=100.0)
    result = validator.validate({"temperature_c": 200.0})
    assert result == {}


def test_sensor_frame_model_validation() -> None:
    model = SensorFrameModel(temperature_c=25.0, humidity_pct=50.0)
    assert model.temperature_c == 25.0


def test_security_monitor_divergence_flag() -> None:
    monitor = SecurityMonitor(divergence_threshold=0.01, max_measurement_rate=0.0)
    frame = SensorFrame(timestamp=0.0, ac_hum_hz=60.0)
    measurement_a = Measurement(name="ptp", offset=0.0, variance=0.1)
    measurement_b = Measurement(name="gps", offset=0.02, variance=0.1)
    monitor.evaluate_measurement(measurement_a, frame)
    _, alerts = monitor.evaluate_measurement(measurement_b, frame)
    assert any(alert.code == "reference_divergence" for alert in alerts)


def test_ml_variance_model_updates_bias() -> None:
    model = MlVarianceModel(feature_weights={"temperature_c": 0.1}, learning_rate=0.1)
    frame = SensorFrame(timestamp=0.0, temperature_c=20.0)
    measurement = Measurement(name="ref", offset=2.0, variance=0.1)
    variance_before = model.adjusted_variance(0.1, frame, "ref")
    model.update(frame, [measurement], fused_offset=0.0, ground_truth_offset=1.0)
    variance_after = model.adjusted_variance(0.1, frame, "ref")
    assert variance_after != variance_before


def test_environmental_sensor_adapter() -> None:
    sensor = EnvironmentalSensor(lambda: (22.0, 45.0, 1012.0))
    frame = sensor.sample()
    assert frame["temperature_c"] == 22.0
    assert frame["humidity_pct"] == 45.0
    assert frame["pressure_hpa"] == 1012.0


def test_i2c_environmental_sensor_adapter() -> None:
    def reader(bus: int, address: int) -> tuple[float, float, float]:
        assert bus == 1
        assert address == 0x76
        return 21.0, 50.0, 1008.0

    sensor = I2CEnvironmentalSensor(reader, bus=1, address=0x76)
    frame = sensor.sample()
    assert frame["temperature_c"] == 21.0
    assert frame["humidity_pct"] == 50.0
    assert frame["pressure_hpa"] == 1008.0


def test_serial_reference_adapter() -> None:
    def parser(line: str) -> Measurement:
        return Measurement(name="serial", offset=float(line), variance=0.5)

    source = iter(["0.25"]).__next__
    ref = SerialReference(parser, source)
    measurement = ref.sample(SensorFrame(timestamp=0.0))
    assert measurement.offset == 0.25
