"""Daemon service for running the TimeServer in a long-lived process."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .api import TimeServerRuntime, build_time_server
from .chrony import ChronyShmSample, ChronyShmWriter
from .config import TimeServerSettings


@dataclass
class DaemonConfig:
    """Configuration for the daemon runtime loop."""

    step_interval_s: float = 1.0
    chrony_enabled: bool = True


class TimeServerDaemon:
    """Daemon runner that periodically steps the TimeServer."""

    def __init__(self, runtime: TimeServerRuntime, config: DaemonConfig) -> None:
        self._runtime = runtime
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._chrony = ChronyShmWriter() if config.chrony_enabled else None
        self._running = False

    def run(self) -> None:
        """Run the daemon loop until interrupted."""

        self._running = True
        self._logger.info("daemon_started", extra={"interval_s": self._config.step_interval_s})
        while self._running:
            update, frame, drift, drift_hint = self._runtime.server.step(self._config.step_interval_s)
            self._runtime.metrics.update(update.fused_offset)
            self._runtime.health.mark_update()
            if self._chrony:
                self._chrony.write(ChronyShmSample(offset=update.fused_offset, delay=0.0))
                self._runtime.health.mark_shm_write()
            time.sleep(self._config.step_interval_s)

    def stop(self) -> None:
        """Stop the daemon loop."""

        self._running = False


def run_daemon(settings: TimeServerSettings | None = None) -> None:
    """Entry point for a basic daemon execution."""

    settings = settings or TimeServerSettings()
    runtime = build_time_server(references=[], sensors=[], settings=settings)
    daemon = TimeServerDaemon(runtime, DaemonConfig())
    daemon.run()
