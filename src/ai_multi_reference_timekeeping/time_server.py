"""Time server components for multi-reference timekeeping.

The time server is intentionally modular: each sensor can provide contextual
features, each reference can provide offset measurements, and a lightweight
inference model tunes reference trustworthiness and drift estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping, MutableMapping, Protocol

import socket
import struct
import subprocess
import time
import shutil

from .fusion import ClockUpdate, Measurement, VirtualClock

NTP_EPOCH = 2208988800  # seconds between 1900-01-01 and 1970-01-01


@dataclass
class SensorFrame:
    """Contextual sensor features that help tune reference weighting."""

    timestamp: float
    temperature_c: float | None = None
    humidity_pct: float | None = None
    pressure_hpa: float | None = None
    ac_hum_hz: float | None = None
    radio_snr_db: float | None = None
    geiger_cpm: float | None = None
    ambient_audio_db: float | None = None
    bird_activity: float | None = None
    traffic_activity: float | None = None
    gps_lock: bool | None = None


class SensorInput(Protocol):
    """Protocol for a sensor feature provider."""

    def sample(self) -> Mapping[str, float | bool | None]:
        """Return a mapping of sensor fields to update."""


class SensorAggregator:
    """Aggregate sensor inputs into a single frame."""

    def __init__(self, *inputs: SensorInput) -> None:
        self._inputs = inputs

    def sample(self) -> SensorFrame:
        updates: MutableMapping[str, float | bool | None] = {}
        for sensor in self._inputs:
            updates.update(sensor.sample())
        return SensorFrame(timestamp=time.time(), **updates)


class ReferenceInput(Protocol):
    """Protocol for timing reference sources."""

    def sample(self, frame: SensorFrame) -> Measurement:
        """Return a timing offset measurement."""


@dataclass
class ReferenceReading:
    """Reference reading plus auxiliary metadata."""

    measurement: Measurement
    metadata: Mapping[str, float | str] = field(default_factory=dict)


class LightweightInferenceModel:
    """Lightweight heuristic model to tune reference variance and drift hints."""

    def __init__(
        self,
        nominal_hum_hz: float = 60.0,
        temperature_coeff: float = 0.01,
        humidity_coeff: float = 0.005,
        pressure_coeff: float = 0.002,
        ac_hum_coeff: float = 0.1,
        audio_coeff: float = 0.01,
        radiation_coeff: float = 0.003,
        radio_coeff: float = 0.004,
        gps_lock_penalty: float = 2.0,
    ) -> None:
        self._nominal_hum_hz = nominal_hum_hz
        self._temperature_coeff = temperature_coeff
        self._humidity_coeff = humidity_coeff
        self._pressure_coeff = pressure_coeff
        self._ac_hum_coeff = ac_hum_coeff
        self._audio_coeff = audio_coeff
        self._radiation_coeff = radiation_coeff
        self._radio_coeff = radio_coeff
        self._gps_lock_penalty = gps_lock_penalty
        self._last_frame: SensorFrame | None = None

    def adjusted_variance(self, base_variance: float, frame: SensorFrame) -> float:
        """Return a variance adjusted by the contextual sensor conditions."""

        if base_variance <= 0:
            raise ValueError("base_variance must be positive")

        previous = self._last_frame
        penalty = 1.0
        if frame.ac_hum_hz is not None:
            deviation = abs(frame.ac_hum_hz - self._nominal_hum_hz) / self._nominal_hum_hz
            penalty += deviation * self._ac_hum_coeff
        if frame.temperature_c is not None and previous is not None:
            penalty += abs(frame.temperature_c - (previous.temperature_c or frame.temperature_c)) * (
                self._temperature_coeff
            )
        if frame.humidity_pct is not None:
            penalty += abs(frame.humidity_pct) * self._humidity_coeff / 100.0
        if frame.pressure_hpa is not None:
            penalty += abs(frame.pressure_hpa - 1013.25) * self._pressure_coeff / 100.0
        if frame.ambient_audio_db is not None:
            penalty += frame.ambient_audio_db * self._audio_coeff / 100.0
        if frame.geiger_cpm is not None:
            penalty += frame.geiger_cpm * self._radiation_coeff / 100.0
        if frame.radio_snr_db is not None:
            penalty += max(0.0, 30.0 - frame.radio_snr_db) * self._radio_coeff / 30.0
        if frame.gps_lock is False:
            penalty *= self._gps_lock_penalty

        return base_variance * penalty

    def drift_hint(self, frame: SensorFrame) -> float:
        """Compute a heuristic drift hint based on environmental sensors."""

        hint = 0.0
        previous = self._last_frame
        if frame.temperature_c is not None and previous is not None:
            hint += (frame.temperature_c - (previous.temperature_c or frame.temperature_c)) * 1e-9
        if frame.humidity_pct is not None:
            hint += (frame.humidity_pct - 50.0) * 5e-10
        if frame.pressure_hpa is not None:
            hint += (frame.pressure_hpa - 1013.25) * 1e-10
        return hint

    def record_frame(self, frame: SensorFrame) -> None:
        """Persist the latest sensor frame for future deltas."""

        self._last_frame = frame


@dataclass
class SlewDriftEstimate:
    """Estimated slew and drift from recent offsets."""

    slew: float
    drift: float
    samples: int


class SlewDriftDetector:
    """Track recent offsets to estimate drift and slew."""

    def __init__(self, window: int = 10) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = window
        self._history: list[tuple[float, float]] = []

    def update(self, timestamp: float, offset: float) -> SlewDriftEstimate:
        self._history.append((timestamp, offset))
        if len(self._history) > self._window:
            self._history.pop(0)

        if len(self._history) < 2:
            return SlewDriftEstimate(slew=0.0, drift=0.0, samples=len(self._history))

        times, offsets = zip(*self._history)
        t_mean = sum(times) / len(times)
        o_mean = sum(offsets) / len(offsets)
        numerator = sum((t - t_mean) * (o - o_mean) for t, o in self._history)
        denominator = sum((t - t_mean) ** 2 for t in times) or 1.0
        drift = numerator / denominator
        slew = offsets[-1] - offsets[0]
        return SlewDriftEstimate(slew=slew, drift=drift, samples=len(self._history))


class NtpReference:
    """SNTP-based reference using an external time server (e.g., NIST)."""

    def __init__(
        self,
        name: str,
        host: str = "time.nist.gov",
        port: int = 123,
        base_variance: float = 1e-3,
        timeout: float = 1.0,
    ) -> None:
        self._name = name
        self._host = host
        self._port = port
        self._base_variance = base_variance
        self._timeout = timeout

    def sample(self, frame: SensorFrame) -> Measurement:
        del frame
        t1 = time.time()
        packet = b"\x1b" + 47 * b"\0"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(self._timeout)
            sock.sendto(packet, (self._host, self._port))
            data, _ = sock.recvfrom(48)
        t4 = time.time()
        if len(data) < 48:
            raise ValueError("Invalid NTP response length")

        t2 = _ntp_to_unix(data[32:40])
        t3 = _ntp_to_unix(data[40:48])
        offset = ((t2 - t1) + (t3 - t4)) / 2.0
        return Measurement(name=self._name, offset=offset, variance=self._base_variance)


class RtcReference:
    """Reference based on the motherboard real-time clock via hwclock."""

    def __init__(self, name: str, base_variance: float = 5e-3) -> None:
        self._name = name
        self._base_variance = base_variance
        self._hwclock = _locate_hwclock()

    def sample(self, frame: SensorFrame) -> Measurement:
        del frame
        output = subprocess.check_output([self._hwclock, "--get", "--utc", "--format", "%Y-%m-%d %H:%M:%S"])
        rtc_time = datetime.strptime(output.decode().strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        rtc_timestamp = rtc_time.timestamp()
        system_timestamp = time.time()
        return Measurement(
            name=self._name,
            offset=rtc_timestamp - system_timestamp,
            variance=self._base_variance,
        )


class NmeaGpsReference:
    """GPS reference that parses NMEA sentences for UTC time."""

    def __init__(
        self,
        name: str,
        line_source: Callable[[], str],
        base_variance: float = 1e-4,
    ) -> None:
        self._name = name
        self._line_source = line_source
        self._base_variance = base_variance

    def sample(self, frame: SensorFrame) -> Measurement:
        del frame
        for _ in range(5):
            line = self._line_source().strip()
            if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                return _parse_rmc(self._name, line, self._base_variance)
        raise ValueError("No valid GPS NMEA sentences found")


class SerialLineSensor:
    """Sensor input built on a line source (serial/USB)."""

    def __init__(self, parser: Callable[[str], Mapping[str, float | bool | None]], line_source: Callable[[], str]) -> None:
        self._parser = parser
        self._line_source = line_source

    def sample(self) -> Mapping[str, float | bool | None]:
        return self._parser(self._line_source())


class GpioPulseSensor:
    """GPIO pulse sensor that estimates frequency from edge timing."""

    def __init__(self, read_level: Callable[[], int], field_name: str, min_period: float = 0.01) -> None:
        self._read_level = read_level
        self._field_name = field_name
        self._min_period = min_period
        self._last_level: int | None = None
        self._last_edge: float | None = None

    def sample(self) -> Mapping[str, float | None]:
        level = self._read_level()
        now = time.time()
        frequency: float | None = None
        if self._last_level is not None and level != self._last_level:
            if self._last_edge is not None:
                period = now - self._last_edge
                if period >= self._min_period:
                    frequency = 1.0 / period
            self._last_edge = now
        self._last_level = level
        return {self._field_name: frequency}


class TimeServer:
    """Coordinate sensors, references, and the virtual clock."""

    def __init__(
        self,
        clock: VirtualClock,
        references: Iterable[ReferenceInput],
        sensors: SensorAggregator | None = None,
        inference: LightweightInferenceModel | None = None,
        drift_detector: SlewDriftDetector | None = None,
    ) -> None:
        self._clock = clock
        self._references = list(references)
        self._sensors = sensors or SensorAggregator()
        self._inference = inference or LightweightInferenceModel()
        self._drift_detector = drift_detector or SlewDriftDetector()

    def step(self, dt: float) -> tuple[ClockUpdate, SensorFrame, SlewDriftEstimate, float]:
        frame = self._sensors.sample()
        measurements = []
        for reference in self._references:
            measurement = reference.sample(frame)
            adjusted_variance = self._inference.adjusted_variance(measurement.variance, frame)
            measurements.append(
                Measurement(
                    name=measurement.name,
                    offset=measurement.offset,
                    variance=adjusted_variance,
                )
            )
        update = self._clock.step(dt, measurements)
        drift_estimate = self._drift_detector.update(frame.timestamp, update.state.offset)
        drift_hint = self._inference.drift_hint(frame)
        self._inference.record_frame(frame)
        return update, frame, drift_estimate, drift_hint


def _ntp_to_unix(data: bytes) -> float:
    seconds, fraction = struct.unpack("!II", data)
    return seconds - NTP_EPOCH + fraction / 2**32


def _parse_rmc(name: str, line: str, variance: float) -> Measurement:
    fields = line.split(",")
    if len(fields) < 10 or fields[2] != "A":
        raise ValueError("Invalid RMC data")
    time_str = fields[1]
    date_str = fields[9]
    if len(time_str) < 6 or len(date_str) != 6:
        raise ValueError("Invalid RMC time/date")
    hh = int(time_str[0:2])
    mm = int(time_str[2:4])
    ss = int(time_str[4:6])
    day = int(date_str[0:2])
    month = int(date_str[2:4])
    year = int(date_str[4:6]) + 2000
    gps_time = datetime(year, month, day, hh, mm, ss, tzinfo=timezone.utc)
    offset = gps_time.timestamp() - time.time()
    return Measurement(name=name, offset=offset, variance=variance)


def _locate_hwclock() -> str:
    path = shutil.which("hwclock") or ""
    if not path:
        raise RuntimeError("hwclock not available for RTC reference")
    return path


def open_line_source(path: str) -> Callable[[], str]:
    """Create a line source callable for serial or USB devices."""

    stream = open(path, "r", encoding="ascii", errors="ignore", buffering=1)
    return stream.readline
