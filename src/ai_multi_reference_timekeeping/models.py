"""Pydantic models for validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class SensorFrameModel(BaseModel):
    """Validated sensor frame input."""

    model_config = ConfigDict(extra="ignore")

    temperature_c: float | None = Field(default=None, ge=-50.0, le=100.0)
    humidity_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    pressure_hpa: float | None = Field(default=None, ge=800.0, le=1200.0)
    ac_hum_hz: float | None = Field(default=None, ge=40.0, le=70.0)
    ac_hum_phase_rad: float | None = Field(default=None)
    ac_hum_uncertainty: float | None = Field(default=None, ge=0.0)
    radio_snr_db: float | None = Field(default=None, ge=0.0, le=80.0)
    geiger_cpm: float | None = Field(default=None, ge=0.0, le=20000.0)
    ambient_audio_db: float | None = Field(default=None, ge=-140.0, le=10.0)
    bird_activity: float | None = Field(default=None, ge=0.0)
    traffic_activity: float | None = Field(default=None, ge=0.0)
    gps_lock: bool | None = Field(default=None)
