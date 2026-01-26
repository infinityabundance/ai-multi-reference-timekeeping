"""Top-level package for AI-assisted multi-reference timekeeping utilities."""

from .characterization import RunningStats, SensorCharacterization
from .api import TimeServerRuntime, build_time_server
from .daemon import DaemonConfig, TimeServerDaemon, run_daemon
from .chrony import ChronyShmSample, ChronyShmWriter
from .config import LoggingConfig, MetricsConfig, SecurityConfig, TimeServerSettings
from .fusion import ClockUpdate, HeuristicFusion, Measurement, QualityMeasurement, ReferenceFusion, VirtualClock
from .kalman import ClockKalmanFilter
from .metrics import HoldoverStats, holdover_stats, mtie, tdev
from .ml_model import LinearVarianceModel, TrainingSample
from .models import SensorFrameModel
from .observability import HealthMonitor, HealthStatus, MetricsExporter, MetricsTracker
from .partitioning import PartitionState, PartitionSupervisor
from .safety import Hazard, HazardOccurrence, RiskMatrix, SafetyCase
from .logging_utils import JsonFormatter, configure_logging
from .time_server import (
    AudioFeatureSensor,
    EnvironmentalSensor,
    I2CEnvironmentalSensor,
    GeigerCounterSensor,
    GpsLockSensor,
    GpioPulseSensor,
    LinearInferenceModel,
    LightweightInferenceModel,
    MlVarianceModel,
    NmeaGpsReference,
    NtpReference,
    RtcReference,
    SerialReference,
    SensorAggregator,
    SensorFrame,
    SerialLineSensor,
    SlewDriftDetector,
    TimeServer,
    TimeServerStep,
    open_line_source,
    RadioSnrSensor,
    SecurityAlert,
    SecurityMonitor,
    SensorValidator,
)

__all__ = [
    "ClockUpdate",
    "ClockKalmanFilter",
    "Measurement",
    "QualityMeasurement",
    "ReferenceFusion",
    "HeuristicFusion",
    "VirtualClock",
    "ChronyShmSample",
    "ChronyShmWriter",
    "TimeServerRuntime",
    "build_time_server",
    "DaemonConfig",
    "TimeServerDaemon",
    "run_daemon",
    "LoggingConfig",
    "MetricsConfig",
    "SecurityConfig",
    "TimeServerSettings",
    "RunningStats",
    "SensorCharacterization",
    "HoldoverStats",
    "holdover_stats",
    "mtie",
    "tdev",
    "LinearVarianceModel",
    "TrainingSample",
    "HealthMonitor",
    "HealthStatus",
    "MetricsExporter",
    "MetricsTracker",
    "SensorFrameModel",
    "JsonFormatter",
    "configure_logging",
    "PartitionState",
    "PartitionSupervisor",
    "Hazard",
    "HazardOccurrence",
    "RiskMatrix",
    "SafetyCase",
    "LightweightInferenceModel",
    "LinearInferenceModel",
    "MlVarianceModel",
    "SecurityAlert",
    "SecurityMonitor",
    "SensorValidator",
    "AudioFeatureSensor",
    "EnvironmentalSensor",
    "I2CEnvironmentalSensor",
    "GeigerCounterSensor",
    "GpsLockSensor",
    "GpioPulseSensor",
    "NmeaGpsReference",
    "NtpReference",
    "RtcReference",
    "SerialReference",
    "RadioSnrSensor",
    "SensorAggregator",
    "SensorFrame",
    "SerialLineSensor",
    "SlewDriftDetector",
    "TimeServer",
    "TimeServerStep",
    "open_line_source",
]
