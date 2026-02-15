"""Integration tests for the fixed stubs and missing components."""

import pytest

from ai_multi_reference_timekeeping.api import build_time_server
from ai_multi_reference_timekeeping.fusion import Measurement
from ai_multi_reference_timekeeping.ml_model import LinearVarianceModel, TrainingSample
from ai_multi_reference_timekeeping.time_server import (
    InferenceModel,
    LightweightInferenceModel,
    SensorFrame,
    SensorInput,
    TimeServer,
    _feature_value,
)

# Test constants
DUMMY_REFERENCE_OFFSET = 0.001
DUMMY_REFERENCE_VARIANCE = 1e-6


def test_feature_value_extraction() -> None:
    """Test that _feature_value extracts sensor features correctly."""
    frame = SensorFrame(
        timestamp=1000.0,
        temperature_c=25.0,
        humidity_pct=50.0,
        gps_lock=True,
    )
    assert _feature_value(frame, "temperature_c") == 25.0
    assert _feature_value(frame, "humidity_pct") == 50.0
    assert _feature_value(frame, "gps_lock") == 1.0
    assert _feature_value(frame, "missing_field") == 0.0


def test_linear_variance_model_with_feature_value() -> None:
    """Test that LinearVarianceModel uses _feature_value correctly."""
    frame = SensorFrame(
        timestamp=1000.0,
        temperature_c=20.0,
        humidity_pct=40.0,
    )
    model = LinearVarianceModel(
        weights={"temperature_c": 0.1, "humidity_pct": 0.05},
        bias=0.5,
    )
    scale = model.predict_scale(frame)
    # score = 0.5 + 0.1*20 + 0.05*40 = 0.5 + 2.0 + 2.0 = 4.5
    # scale = 1.0 + score = 1.0 + 4.5 = 5.5
    assert scale == 5.5


def test_linear_variance_model_training() -> None:
    """Test that LinearVarianceModel can be trained."""
    frame1 = SensorFrame(timestamp=1000.0, temperature_c=20.0)
    frame2 = SensorFrame(timestamp=1001.0, temperature_c=25.0)

    samples = [
        TrainingSample(frame=frame1, target_scale=1.5),
        TrainingSample(frame=frame2, target_scale=2.0),
    ]

    model = LinearVarianceModel(weights={"temperature_c": 0.0}, bias=1.0)
    model.train(samples, learning_rate=1e-3, epochs=10)
    # Just verify it doesn't crash and weights change
    assert model.weights["temperature_c"] != 0.0


def test_inference_model_protocol() -> None:
    """Test that InferenceModel protocol is satisfied."""
    frame = SensorFrame(timestamp=1000.0, temperature_c=22.0)
    inference: InferenceModel = LightweightInferenceModel()

    # Test protocol methods
    variance = inference.adjusted_variance(1.0, frame)
    assert isinstance(variance, float)
    assert variance > 0

    drift = inference.drift_hint(frame)
    assert isinstance(drift, float)

    inference.record_frame(frame)


def test_time_server_with_extended_signature() -> None:
    """Test that TimeServer accepts the extended signature from api.py."""

    class DummySensor:
        def sample(self):
            return {"temperature_c": 22.0}

    class DummyReference:
        def sample(self, frame):
            return Measurement(name="dummy", offset=DUMMY_REFERENCE_OFFSET, variance=DUMMY_REFERENCE_VARIANCE)

    from ai_multi_reference_timekeeping.fusion import HeuristicFusion, VirtualClock
    from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState
    from ai_multi_reference_timekeeping.partitioning import PartitionSupervisor
    from ai_multi_reference_timekeeping.safety import SafetyCase
    from ai_multi_reference_timekeeping.time_server import SensorAggregator

    import logging

    kalman = ClockKalmanFilter(
        state=ClockState(offset=0.0, drift=0.0),
        covariance=ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
        process_noise_offset=1e-4,
        process_noise_drift=1e-6,
    )
    clock = VirtualClock(kalman_filter=kalman, fusion=HeuristicFusion())

    # Test with all optional parameters that api.py passes
    server = TimeServer(
        clock=clock,
        references=[DummyReference()],
        sensors=SensorAggregator(DummySensor()),
        inference=LightweightInferenceModel(),
        safety_case=SafetyCase(),
        logger=logging.getLogger("test"),
        partition_supervisor=PartitionSupervisor(),
    )

    # Verify it works
    update, frame, drift, hint = server.step(dt=1.0)
    assert update.fused_offset == pytest.approx(DUMMY_REFERENCE_OFFSET, abs=1e-9)


def test_build_time_server_integration() -> None:
    """Test that build_time_server creates a working runtime."""

    class DummySensor:
        def sample(self):
            return {"temperature_c": 22.0}

    class DummyReference:
        def sample(self, frame):
            return Measurement(name="dummy", offset=DUMMY_REFERENCE_OFFSET, variance=DUMMY_REFERENCE_VARIANCE)

    runtime = build_time_server(
        references=[DummyReference()],
        sensors=[DummySensor()],
        inference=None,  # Will use default
    )

    assert runtime.server is not None
    assert runtime.health is not None
    assert runtime.metrics is not None
    assert runtime.exporter is not None

    # Test a step
    update, frame, drift, hint = runtime.server.step(dt=1.0)
    assert update.fused_offset == pytest.approx(DUMMY_REFERENCE_OFFSET, abs=1e-9)
