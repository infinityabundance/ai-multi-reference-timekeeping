"""Metrics export and health checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable

from .metrics import mtie, tdev


@dataclass
class HealthStatus:
    """Health status snapshot."""

    last_update: float
    shm_last_write: float | None
    ok: bool


class HealthMonitor:
    """Track freshness of TimeServer updates and SHM writes."""

    def __init__(self, freshness_window: float = 10.0) -> None:
        self._freshness_window = freshness_window
        self._last_update: float | None = None
        self._shm_last_write: float | None = None

    def mark_update(self) -> None:
        self._last_update = time.time()

    def mark_shm_write(self) -> None:
        self._shm_last_write = time.time()

    def status(self) -> HealthStatus:
        now = time.time()
        last_update = self._last_update or 0.0
        ok = now - last_update <= self._freshness_window if last_update else False
        return HealthStatus(last_update=last_update, shm_last_write=self._shm_last_write, ok=ok)


class MetricsTracker:
    """Maintain a rolling buffer of offsets for metrics export."""

    def __init__(self, window_size: int = 60) -> None:
        self._window_size = window_size
        self._offsets: list[float] = []

    def update(self, offset: float) -> None:
        self._offsets.append(offset)
        if len(self._offsets) > self._window_size:
            self._offsets.pop(0)

    def metrics(self) -> dict[str, float]:
        if len(self._offsets) < 3:
            return {}
        return {
            "tdev_tau1": tdev(self._offsets, tau=1),
            "mtie_window2": mtie(self._offsets, window=2),
        }


class MetricsExporter:
    """HTTP server exposing metrics and health endpoints."""

    def __init__(self, tracker: MetricsTracker, health: HealthMonitor) -> None:
        self._tracker = tracker
        self._health = health
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, host: str, port: int) -> None:
        tracker = self._tracker
        health = self._health

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/metrics":
                    body = []
                    for key, value in tracker.metrics().items():
                        body.append(f"{key} {value}")
                    response = "\n".join(body).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(response)
                elif self.path == "/health":
                    status = health.status()
                    response = json.dumps(status.__dict__).encode()
                    self.send_response(200 if status.ok else 503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(response)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Iterable[object]) -> None:  # noqa: A003
                return

        self._server = HTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=1.0)
