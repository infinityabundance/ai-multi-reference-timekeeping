"""Configuration management for timekeeping services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import tomllib


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    level: str = Field(default="INFO", description="Logging level")
    json: bool = Field(default=True, description="Emit JSON logs")
    log_file: str | None = Field(default=None, description="Optional log file path")
    max_bytes: int = Field(default=1_000_000, description="Max log file size before rotation")
    backup_count: int = Field(default=3, description="Number of rotated log files to keep")


class MetricsConfig(BaseModel):
    """Metrics/health endpoint configuration."""

    enabled: bool = Field(default=True, description="Enable metrics exporter")
    host: str = Field(default="0.0.0.0", description="Metrics host")
    port: int = Field(default=8000, description="Metrics port")
    window_size: int = Field(default=60, description="Offset window size for metrics")


class SecurityConfig(BaseModel):
    """Security configuration settings."""

    divergence_threshold: float = Field(default=0.005, description="Reference divergence threshold (seconds)")
    grid_frequency: float | None = Field(default=60.0, description="Expected grid frequency")
    grid_tolerance_hz: float = Field(default=0.5, description="Grid frequency tolerance (Hz)")
    max_measurement_rate: float = Field(default=5.0, description="Max measurements/sec per reference")


class TimeServerSettings(BaseSettings):
    """Configuration settings loaded from env or optional TOML."""

    model_config = SettingsConfigDict(env_prefix="AIMRT_", env_nested_delimiter="__", extra="ignore")

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    chrony_shm_path: str = Field(default="/dev/shm/chrony_shm0", description="Chrony SHM path")

    @classmethod
    def from_toml(cls, path: str | Path) -> "TimeServerSettings":
        data = tomllib.loads(Path(path).read_text())
        return cls.model_validate(data)
