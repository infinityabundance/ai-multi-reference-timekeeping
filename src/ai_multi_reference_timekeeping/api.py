"""Public API facade for the timekeeping toolkit.

This module provides a single, discoverable entry point that assembles the
core components needed to run the time server in a safe, auditable way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import logging

from .config import TimeServerSettings
from .fusion import HeuristicFusion, VirtualClock
from .kalman import ClockCovariance, ClockKalmanFilter, ClockState
from .logging_utils import configure_logging
from .observability import HealthMonitor, MetricsExporter, MetricsTracker
from .partitioning import PartitionSupervisor
from .safety import SafetyCase
from .time_server import (
    InferenceModel,
    ReferenceInput,
    SensorAggregator,
    SensorInput,
    TimeServer,
)


@dataclass
class TimeServerRuntime:
    """Structured runtime handles for operators."""

    server: TimeServer
    health: HealthMonitor
    metrics: MetricsTracker
    exporter: MetricsExporter


def build_time_server(
    references: Iterable[ReferenceInput],
    sensors: Iterable[SensorInput],
    inference: InferenceModel | None = None,
    settings: TimeServerSettings | None = None,
    logger: logging.Logger | None = None,
    safety_case: SafetyCase | None = None,
) -> TimeServerRuntime:
    """Create a TimeServer with sensible defaults and observability."""

    settings = settings or TimeServerSettings()
    configure_logging(settings.logging)

    logger = logger or logging.getLogger(__name__)
    kalman = ClockKalmanFilter(
        state=ClockState(offset=0.0, drift=0.0),
        covariance=ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
        process_noise_offset=1e-4,
        process_noise_drift=1e-6,
    )
    clock = VirtualClock(kalman_filter=kalman, fusion=HeuristicFusion())
    partition_supervisor = PartitionSupervisor()
    aggregator = SensorAggregator(*sensors, logger=logger, partitions=partition_supervisor)
    server = TimeServer(
        clock=clock,
        references=references,
        sensors=aggregator,
        inference=inference,
        safety_case=safety_case,
        logger=logger,
        partition_supervisor=partition_supervisor,
    )

    health = HealthMonitor()
    metrics = MetricsTracker(window_size=settings.metrics.window_size)
    exporter = MetricsExporter(metrics, health)
    if settings.metrics.enabled:
        exporter.start(settings.metrics.host, settings.metrics.port)

    return TimeServerRuntime(server=server, health=health, metrics=metrics, exporter=exporter)
