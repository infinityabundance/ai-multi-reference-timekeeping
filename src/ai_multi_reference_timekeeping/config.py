"""Configuration management for timekeeping services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import tomllib


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    # Logging severity threshold (INFO/DEBUG/etc.).
    level: str = Field(default="INFO", description="Logging level")
    # Emit JSON if True; otherwise emit a human-readable format.
    json: bool = Field(default=True, description="Emit JSON logs")
    # Optional file path for log output; if None, logs go to stdout.
    log_file: str | None = Field(default=None, description="Optional log file path")
    # Maximum size (bytes) before log rotation.
    max_bytes: int = Field(default=1_000_000, description="Max log file size before rotation")
    # Number of backup files to retain.
    backup_count: int = Field(default=3, description="Number of rotated log files to keep")


class MetricsConfig(BaseModel):
    """Metrics/health endpoint configuration."""

    # Enable HTTP exporter when True.
    enabled: bool = Field(default=True, description="Enable metrics exporter")
    # Host for HTTP server.
    host: str = Field(default="0.0.0.0", description="Metrics host")
    # TCP port for HTTP server.
    port: int = Field(default=8000, description="Metrics port")
    # Rolling window length for TDEV/MTIE calculations.
    window_size: int = Field(default=60, description="Offset window size for metrics")


class SecurityConfig(BaseModel):
    """Security configuration settings."""

    # Maximum acceptable reference divergence in seconds.
    divergence_threshold: float = Field(default=0.005, description="Reference divergence threshold (seconds)")
    # Expected grid frequency (50/60 Hz) or None to disable AC-hum checks.
    grid_frequency: float | None = Field(default=60.0, description="Expected grid frequency")
    # Allowed deviation from grid frequency in Hz.
    grid_tolerance_hz: float = Field(default=0.5, description="Grid frequency tolerance (Hz)")
    # Per-reference maximum measurement rate (samples/sec).
    max_measurement_rate: float = Field(default=5.0, description="Max measurements/sec per reference")


class TimeServerSettings(BaseSettings):
    """Configuration settings loaded from env or optional TOML."""

    # Environment keys use AIMRT_ prefix and "__" nesting.
    model_config = SettingsConfigDict(env_prefix="AIMRT_", env_nested_delimiter="__", extra="ignore")

    # Nested configs provide defaults for each subsystem.
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    # Chrony shared memory path for SHM refclock.
    chrony_shm_path: str = Field(default="/dev/shm/chrony_shm0", description="Chrony SHM path")

    @classmethod
    def from_toml(cls, path: str | Path) -> "TimeServerSettings":
        data = tomllib.loads(Path(path).read_text())
        return cls.model_validate(data)
