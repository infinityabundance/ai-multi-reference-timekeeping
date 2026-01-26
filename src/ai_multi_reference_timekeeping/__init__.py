"""Top-level package for AI-assisted multi-reference timekeeping utilities."""

from .fusion import ClockUpdate, Measurement, ReferenceFusion, VirtualClock
from .kalman import ClockKalmanFilter
from .time_server import (
    GpioPulseSensor,
    LightweightInferenceModel,
    NmeaGpsReference,
    NtpReference,
    RtcReference,
    SensorAggregator,
    SensorFrame,
    SerialLineSensor,
    SlewDriftDetector,
    TimeServer,
    open_line_source,
)

__all__ = [
    "ClockUpdate",
    "ClockKalmanFilter",
    "Measurement",
    "ReferenceFusion",
    "VirtualClock",
    "LightweightInferenceModel",
    "GpioPulseSensor",
    "NmeaGpsReference",
    "NtpReference",
    "RtcReference",
    "SensorAggregator",
    "SensorFrame",
    "SerialLineSensor",
    "SlewDriftDetector",
    "TimeServer",
    "open_line_source",
]
