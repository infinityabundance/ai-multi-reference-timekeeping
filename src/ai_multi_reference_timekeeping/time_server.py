"""Time server components for multi-reference timekeeping.

The time server is intentionally modular: each sensor can provide contextual
features, each reference can provide offset measurements, and a lightweight
inference model tunes reference trustworthiness and drift estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping, MutableMapping, Protocol, Sequence

import logging
import math
import threading
import socket
import struct
import subprocess
import time
import shutil

from .characterization import SensorCharacterization
from .models import SensorFrameModel
from .partitioning import PartitionSupervisor
from .safety import SafetyCase
from .fusion import ClockUpdate, Measurement, QualityMeasurement, VirtualClock

NTP_EPOCH = 2208988800  # seconds between 1900-01-01 and 1970-01-01


@dataclass
class SensorFrame:
    """Contextual sensor features that help tune reference weighting."""

    timestamp: float
    temperature_c: float | None = None
    humidity_pct: float | None = None
    pressure_hpa: float | None = None
    ac_hum_hz: float | None = None
    ac_hum_phase_rad: float | None = None
    ac_hum_uncertainty: float | None = None
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


class AudioSampleSource(Protocol):
    """Protocol for microphone/line-in audio samples."""

    def sample(self) -> tuple[Sequence[float], int]:
        """Return a buffer of audio samples and its sample rate (Hz)."""


class SensorAggregator:
    """Aggregate sensor inputs into a single frame.

    This aggregator coordinates multiple sensor providers, validates inputs,
    and emits a unified SensorFrame for downstream fusion and inference.
    """

    def __init__(
        self,
        *inputs: SensorInput,
        validator: "SensorValidator" | None = None,
        logger: logging.Logger | None = None,
        partitions: PartitionSupervisor | None = None,
    ) -> None:
        self._inputs = inputs
        self._validator = validator or SensorValidator(logger=logger)
        self._logger = logger or logging.getLogger(__name__)
        self._partitions = partitions or PartitionSupervisor()

    def sample(self) -> SensorFrame:
        updates: MutableMapping[str, float | bool | None] = {}
        for sensor in self._inputs:
            partition_name = sensor.__class__.__name__
            try:
                if self._partitions.should_reboot(partition_name):
                    self._partitions.reboot(partition_name)
                    self._logger.info("partition_rebooted", extra={"partition": partition_name})
                updates.update(self._validator.validate(sensor.sample()))
                self._partitions.record_success(partition_name)
            except Exception as exc:
                self._logger.error(
                    "sensor_sample_failed",
                    extra={"sensor": sensor.__class__.__name__, "error": str(exc)},
                )
                state = self._partitions.record_failure(partition_name)
                self._logger.warning(
                    "partition_fault",
                    extra={"partition": partition_name, "failures": state.failures, "healthy": state.healthy},
                )
        return SensorFrame(timestamp=time.time(), **updates)


class ReferenceInput(Protocol):
    """Protocol for timing reference sources."""

    def sample(self, frame: SensorFrame) -> Measurement:
        """Return a timing offset measurement."""


class InferenceModel(Protocol):
    """Protocol for inference models that tune reference weighting."""

    def adjusted_variance(self, base_variance: float, frame: SensorFrame, reference_name: str) -> float:
        """Return a variance adjusted by context."""

    def drift_hint(self, frame: SensorFrame) -> float:
        """Return a drift hint based on the latest sensor frame."""

    def record_frame(self, frame: SensorFrame) -> None:
        """Persist the latest sensor frame."""

    def update(
        self,
        frame: SensorFrame,
        measurements: Sequence[Measurement],
        fused_offset: float,
        ground_truth_offset: float | None = None,
    ) -> None:
        """Optionally update model parameters after fusion."""


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

    def adjusted_variance(self, base_variance: float, frame: SensorFrame, reference_name: str = "") -> float:
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
        if reference_name:
            penalty *= 1.0

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

    def update(
        self,
        frame: SensorFrame,
        measurements: Sequence[Measurement],
        fused_offset: float,
        ground_truth_offset: float | None = None,
    ) -> None:
        del frame, measurements, fused_offset, ground_truth_offset


class LinearInferenceModel:
    """Tiny linear model for variance scaling and drift hints."""

    def __init__(
        self,
        feature_weights: Mapping[str, float] | None = None,
        drift_weights: Mapping[str, float] | None = None,
        reference_bias: Mapping[str, float] | None = None,
        bias: float = 0.0,
    ) -> None:
        self._feature_weights = dict(feature_weights or {})
        self._drift_weights = dict(drift_weights or {})
        self._reference_bias = dict(reference_bias or {})
        self._bias = bias
        self._last_frame: SensorFrame | None = None

    def adjusted_variance(self, base_variance: float, frame: SensorFrame, reference_name: str = "") -> float:
        if base_variance <= 0:
            raise ValueError("base_variance must be positive")
        score = self._bias + self._reference_bias.get(reference_name, 0.0)
        for feature, weight in self._feature_weights.items():
            score += weight * _feature_value(frame, feature)
        scale = 1.0 + 1.0 / (1.0 + math.exp(-score))
        return base_variance * scale

    def drift_hint(self, frame: SensorFrame) -> float:
        hint = 0.0
        for feature, weight in self._drift_weights.items():
            hint += weight * _feature_value(frame, feature)
        return hint

    def record_frame(self, frame: SensorFrame) -> None:
        self._last_frame = frame

    def update(
        self,
        frame: SensorFrame,
        measurements: Sequence[Measurement],
        fused_offset: float,
        ground_truth_offset: float | None = None,
    ) -> None:
        del frame, measurements, fused_offset, ground_truth_offset


class MlVarianceModel:
    """Small online model that adapts variance scaling from residuals."""

    def __init__(
        self,
        feature_weights: Mapping[str, float] | None = None,
        reference_bias: Mapping[str, float] | None = None,
        bias: float = 0.0,
        learning_rate: float = 1e-3,
        min_scale: float = 0.5,
        max_scale: float = 5.0,
        characterization: SensorCharacterization | None = None,
    ) -> None:
        self._feature_weights = dict(feature_weights or {})
        self._reference_bias = dict(reference_bias or {})
        self._bias = bias
        self._learning_rate = learning_rate
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._last_frame: SensorFrame | None = None
        self._characterization = characterization or SensorCharacterization()

    def adjusted_variance(self, base_variance: float, frame: SensorFrame, reference_name: str = "") -> float:
        if base_variance <= 0:
            raise ValueError("base_variance must be positive")
        score = self._bias + self._reference_bias.get(reference_name, 0.0)
        for feature, weight in self._feature_weights.items():
            score += weight * _feature_value(frame, feature)
        scale = 1.0 + math.tanh(score)
        scale = min(self._max_scale, max(self._min_scale, scale))
        return base_variance * scale

    def drift_hint(self, frame: SensorFrame) -> float:
        hint = 0.0
        if self._last_frame and frame.temperature_c is not None:
            hint += (frame.temperature_c - (self._last_frame.temperature_c or frame.temperature_c)) * 5e-10
        return hint

    def record_frame(self, frame: SensorFrame) -> None:
        self._last_frame = frame

    def update(
        self,
        frame: SensorFrame,
        measurements: Sequence[Measurement],
        fused_offset: float,
        ground_truth_offset: float | None = None,
    ) -> None:
        del frame
        target_offset = ground_truth_offset if ground_truth_offset is not None else fused_offset
        for measurement in measurements:
            residual = measurement.offset - target_offset
            expected = math.sqrt(max(measurement.variance, 1e-12))
            error = (abs(residual) - expected) / max(expected, 1e-6)
            self._characterization.update(measurement.name, residual)
            if abs(error) > 3.0 or abs(self._characterization.z_score(measurement.name, residual)) > 3.0:
                continue
            step = max(-self._learning_rate, min(self._learning_rate, self._learning_rate * error))
            self._bias += step
            self._reference_bias[measurement.name] = self._reference_bias.get(measurement.name, 0.0) + step


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


class EnvironmentalSensor:
    """Adapter for temperature/humidity/pressure sensors."""

    def __init__(self, reader: Callable[[], tuple[float | None, float | None, float | None]]) -> None:
        self._reader = reader

    def sample(self) -> Mapping[str, float | None]:
        temperature_c, humidity_pct, pressure_hpa = self._reader()
        return {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "pressure_hpa": pressure_hpa,
        }


class I2CEnvironmentalSensor:
    """Adapter for I2C temperature/pressure sensors using a callable reader."""

    def __init__(
        self,
        reader: Callable[[int, int], tuple[float | None, float | None, float | None]],
        bus: int,
        address: int,
    ) -> None:
        self._reader = reader
        self._bus = bus
        self._address = address

    def sample(self) -> Mapping[str, float | None]:
        temperature_c, humidity_pct, pressure_hpa = self._reader(self._bus, self._address)
        return {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "pressure_hpa": pressure_hpa,
        }


class GeigerCounterSensor:
    """Adapter for Geiger counter readings (counts per minute)."""

    def __init__(self, reader: Callable[[], float | None]) -> None:
        self._reader = reader

    def sample(self) -> Mapping[str, float | None]:
        return {"geiger_cpm": self._reader()}


class RadioSnrSensor:
    """Adapter for SDR radio SNR measurements."""

    def __init__(self, reader: Callable[[], float | None]) -> None:
        self._reader = reader

    def sample(self) -> Mapping[str, float | None]:
        return {"radio_snr_db": self._reader()}


class GpsLockSensor:
    """Adapter reporting GPS lock state."""

    def __init__(self, reader: Callable[[], bool | None]) -> None:
        self._reader = reader

    def sample(self) -> Mapping[str, bool | None]:
        return {"gps_lock": self._reader()}


class SerialReference:
    """Reference source backed by a serial or USB line source."""

    def __init__(self, parser: Callable[[str], Measurement], line_source: Callable[[], str]) -> None:
        self._parser = parser
        self._line_source = line_source

    def sample(self, frame: SensorFrame) -> Measurement:
        del frame
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


class AudioFeatureSensor:
    """Extracts AC hum, ambient audio, and activity bands from audio samples."""

    def __init__(
        self,
        sample_source: AudioSampleSource,
        hum_candidates: Sequence[float] = (50.0, 60.0),
        bird_band: tuple[float, float] = (2000.0, 8000.0),
        traffic_band: tuple[float, float] = (50.0, 300.0),
    ) -> None:
        self._source = sample_source
        self._hum_candidates = hum_candidates
        self._bird_band = bird_band
        self._traffic_band = traffic_band

    def sample(self) -> Mapping[str, float | None]:
        samples, sample_rate = self._source.sample()
        if not samples:
            return {
                "ambient_audio_db": None,
                "ac_hum_hz": None,
                "ac_hum_phase_rad": None,
                "ac_hum_uncertainty": None,
                "bird_activity": None,
                "traffic_activity": None,
            }

        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        ambient_db = 20.0 * math.log10(max(rms, 1e-12))

        ac_hum_hz, ac_hum_phase, ac_hum_uncertainty = _estimate_ac_hum(samples, sample_rate, self._hum_candidates)

        bird_energy = _band_energy(samples, sample_rate, self._bird_band)
        traffic_energy = _band_energy(samples, sample_rate, self._traffic_band)

        return {
            "ambient_audio_db": ambient_db,
            "ac_hum_hz": ac_hum_hz,
            "ac_hum_phase_rad": ac_hum_phase,
            "ac_hum_uncertainty": ac_hum_uncertainty,
            "bird_activity": bird_energy,
            "traffic_activity": traffic_energy,
        }


@dataclass
class TimeServerStep:
    """Full step output for the time server."""

    update: ClockUpdate
    frame: SensorFrame
    drift: SlewDriftEstimate
    drift_hint: float
    measurements: Sequence[Measurement]
    safety_case: SafetyCase | None = None


class TimeServer:
    """Coordinate sensors, references, and the virtual clock."""

    def __init__(
        self,
        clock: VirtualClock,
        references: Iterable[ReferenceInput],
        sensors: SensorAggregator | None = None,
        inference: InferenceModel | None = None,
        drift_detector: SlewDriftDetector | None = None,
        ground_truth_reference: str | None = None,
        security_monitor: "SecurityMonitor" | None = None,
        logger: logging.Logger | None = None,
        safety_case: SafetyCase | None = None,
        partition_supervisor: PartitionSupervisor | None = None,
    ) -> None:
        self._clock = clock
        self._references = list(references)
        self._sensors = sensors or SensorAggregator(partitions=partition_supervisor, logger=logger)
        self._inference = inference or LightweightInferenceModel()
        self._drift_detector = drift_detector or SlewDriftDetector()
        self._ground_truth_reference = ground_truth_reference
        self._safety = safety_case
        self._security = security_monitor or SecurityMonitor(safety_case=self._safety)
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

    def step(self, dt: float) -> tuple[ClockUpdate, SensorFrame, SlewDriftEstimate, float]:
        result = self.step_with_context(dt)
        return result.update, result.frame, result.drift, result.drift_hint

    def step_with_context(self, dt: float) -> TimeServerStep:
        with self._lock:
            frame = self._sensors.sample()
            measurements: list[Measurement] = []
            ground_truth_offset: float | None = None
            for alert in self._security.evaluate_frame(frame):
                self._logger.warning("security_alert", extra={"code": alert.code, "detail": alert.detail})
            for reference in self._references:
                try:
                    measurement = reference.sample(frame)
                except Exception as exc:
                    self._logger.error(
                        "reference_sample_failed",
                        extra={"reference": reference.__class__.__name__, "error": str(exc)},
                    )
                    continue
                measurement, alerts = self._security.evaluate_measurement(measurement, frame)
                for alert in alerts:
                    self._logger.warning("security_alert", extra={"code": alert.code, "detail": alert.detail})
                adjusted_variance = self._inference.adjusted_variance(measurement.variance, frame, measurement.name)
                quality = 1.0 / (1.0 + adjusted_variance)
                if self._ground_truth_reference and measurement.name == self._ground_truth_reference:
                    ground_truth_offset = measurement.offset
                measurements.append(
                    QualityMeasurement(
                        name=measurement.name,
                        offset=measurement.offset,
                        variance=adjusted_variance,
                        quality=quality,
                    )
                )
            if not measurements:
                raise ValueError("No valid measurements available after sampling references")
            update = self._clock.step(dt, measurements)
            drift_estimate = self._drift_detector.update(frame.timestamp, update.state.offset)
            drift_hint = self._inference.drift_hint(frame)
            self._inference.record_frame(frame)
            self._inference.update(frame, measurements, update.fused_offset, ground_truth_offset=ground_truth_offset)
            return TimeServerStep(
                update=update,
                frame=frame,
                drift=drift_estimate,
                drift_hint=drift_hint,
                measurements=measurements,
                safety_case=self._safety,
            )


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


def _feature_value(frame: SensorFrame, name: str) -> float:
    value = getattr(frame, name, None)
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _goertzel(samples: Sequence[float], sample_rate: int, frequency: float) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frequency <= 0:
        return 0.0
    k = int(0.5 + (len(samples) * frequency) / sample_rate)
    omega = (2.0 * math.pi * k) / len(samples)
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return power


def _goertzel_complex(samples: Sequence[float], sample_rate: int, frequency: float) -> complex:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frequency <= 0:
        return complex(0.0, 0.0)
    k = int(0.5 + (len(samples) * frequency) / sample_rate)
    omega = (2.0 * math.pi * k) / len(samples)
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    real = s_prev - s_prev2 * math.cos(omega)
    imag = s_prev2 * math.sin(omega)
    return complex(real, imag)


def _band_energy(samples: Sequence[float], sample_rate: int, band: tuple[float, float]) -> float:
    low, high = band
    if low <= 0 or high <= low:
        return 0.0
    step = max(1.0, (high - low) / 5.0)
    energy = 0.0
    freq = low
    while freq <= high:
        energy += _goertzel(samples, sample_rate, freq)
        freq += step
    return energy


def _estimate_ac_hum(
    samples: Sequence[float],
    sample_rate: int,
    candidates: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    if not samples or sample_rate <= 0:
        return None, None, None
    best_frequency = None
    best_power = 0.0
    total_power = sum(sample * sample for sample in samples) / len(samples)
    search_frequencies = []
    for candidate in candidates:
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
            search_frequencies.append(candidate + offset)
    for frequency in search_frequencies:
        power = _goertzel(samples, sample_rate, frequency)
        if power > best_power:
            best_power = power
            best_frequency = frequency
    if best_frequency is None:
        return None, None, None
    phase = _goertzel_complex(samples, sample_rate, best_frequency)
    phase_rad = math.atan2(phase.imag, phase.real)
    snr = best_power / max(total_power, 1e-12)
    uncertainty = 1.0 / (1.0 + snr)
    return best_frequency, phase_rad, uncertainty


@dataclass
class SecurityAlert:
    """Represents a security or anomaly alert."""

    code: str
    detail: str
    severity: str = "warn"


class SensorValidator:
    """Validate and rate limit sensor inputs.

    The validator applies range checking, rate limiting, and model validation
    to ensure incoming sensor values are safe and well-formed.
    """

    def __init__(
        self,
        max_samples_per_sec: float = 5.0,
        ranges: Mapping[str, tuple[float, float]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._max_samples_per_sec = max_samples_per_sec
        self._ranges = ranges or {
            "temperature_c": (-40.0, 85.0),
            "humidity_pct": (0.0, 100.0),
            "pressure_hpa": (900.0, 1100.0),
            "ac_hum_hz": (49.5, 60.5),
            "radio_snr_db": (0.0, 60.0),
            "geiger_cpm": (0.0, 20000.0),
            "ambient_audio_db": (-120.0, 0.0),
            "bird_activity": (0.0, 1e9),
            "traffic_activity": (0.0, 1e9),
        }
        self._last_sample: float | None = None
        self._logger = logger or logging.getLogger(__name__)

    def validate(self, values: Mapping[str, float | bool | None]) -> Mapping[str, float | bool | None]:
        now = time.time()
        if self._max_samples_per_sec > 0:
            if self._last_sample is not None and now - self._last_sample < 1.0 / self._max_samples_per_sec:
                self._logger.warning("sensor_rate_limited", extra={"max_samples_per_sec": self._max_samples_per_sec})
                return {}
            self._last_sample = now
        else:
            self._last_sample = now
        filtered: dict[str, float | bool | None] = {}
        for key, value in values.items():
            if value is None:
                filtered[key] = value
                continue
            if isinstance(value, bool):
                filtered[key] = value
                continue
            range_limit = self._ranges.get(key)
            if range_limit and (value < range_limit[0] or value > range_limit[1]):
                self._logger.warning(
                    "sensor_value_out_of_range",
                    extra={"field": key, "value": value, "min": range_limit[0], "max": range_limit[1]},
                )
                continue
            filtered[key] = value
        try:
            SensorFrameModel.model_validate(filtered)
        except Exception:
            self._logger.error("sensor_validation_failed", extra={"values": filtered})
            return {}
        return filtered


class RateLimiter:
    """Simple rate limiter for measurement sampling."""

    def __init__(self, max_samples_per_sec: float) -> None:
        self._max_samples_per_sec = max_samples_per_sec
        self._last_sample: float | None = None

    def allow(self) -> bool:
        now = time.time()
        if self._max_samples_per_sec <= 0:
            self._last_sample = now
            return True
        if self._last_sample is not None and now - self._last_sample < 1.0 / self._max_samples_per_sec:
            return False
        self._last_sample = now
        return True


class SecurityMonitor:
    """Detect anomalies across references and sensor inputs."""

    def __init__(
        self,
        divergence_threshold: float = 0.005,
        grid_frequency: float | None = 60.0,
        grid_tolerance_hz: float = 0.5,
        max_measurement_rate: float = 5.0,
        logger: logging.Logger | None = None,
        safety_case: SafetyCase | None = None,
    ) -> None:
        self._divergence_threshold = divergence_threshold
        self._grid_frequency = grid_frequency
        self._grid_tolerance_hz = grid_tolerance_hz
        self._rate_limit = RateLimiter(max_samples_per_sec=max_measurement_rate)
        self._last_measurements: dict[str, float] = {}
        self._logger = logger or logging.getLogger(__name__)
        self._safety = safety_case

    def evaluate_frame(self, frame: SensorFrame) -> list[SecurityAlert]:
        alerts: list[SecurityAlert] = []
        if frame.ac_hum_hz is not None and self._grid_frequency is not None:
            if abs(frame.ac_hum_hz - self._grid_frequency) > self._grid_tolerance_hz:
                alert = SecurityAlert(
                    code="grid_frequency_out_of_range",
                    detail=f"AC hum {frame.ac_hum_hz:.2f} Hz outside expected grid",
                    severity="warn",
                )
                self._logger.warning("security_alert", extra={"code": alert.code, "detail": alert.detail})
                if self._safety:
                    self._safety.record("AC_HUM_INJECTION", alert.detail, time.time())
                alerts.append(
                    SecurityAlert(
                        code="grid_frequency_out_of_range",
                        detail=f"AC hum {frame.ac_hum_hz:.2f} Hz outside expected grid",
                        severity="warn",
                    )
                )
        return alerts

    def evaluate_measurement(
        self, measurement: Measurement, frame: SensorFrame
    ) -> tuple[Measurement, list[SecurityAlert]]:
        alerts: list[SecurityAlert] = []
        if not self._rate_limit.allow():
            alert = SecurityAlert(code="rate_limited", detail=f"{measurement.name} rate limited")
            self._logger.warning("security_alert", extra={"code": alert.code, "detail": alert.detail})
            if self._safety:
                self._safety.record("SENSOR_FLOODING", alert.detail, time.time())
            alerts.append(alert)
            return Measurement(name=measurement.name, offset=measurement.offset, variance=measurement.variance * 10.0), alerts

        for name, offset in self._last_measurements.items():
            if name == measurement.name:
                continue
            if abs(offset - measurement.offset) > self._divergence_threshold:
                alert = SecurityAlert(
                    code="reference_divergence",
                    detail=f"{measurement.name} diverged from {name}",
                    severity="warn",
                )
                self._logger.warning("security_alert", extra={"code": alert.code, "detail": alert.detail})
                if self._safety:
                    self._safety.record("REFERENCE_DIVERGENCE", alert.detail, time.time())
                alerts.append(
                    SecurityAlert(
                        code="reference_divergence",
                        detail=f"{measurement.name} diverged from {name}",
                        severity="warn",
                    )
                )
        if measurement.name.lower().startswith("gps") and frame.ac_hum_hz is not None:
            if abs(measurement.offset) > self._divergence_threshold:
                alert = SecurityAlert(code="gps_spoofing_suspected", detail="GPS offset spike detected")
                self._logger.warning("security_alert", extra={"code": alert.code, "detail": alert.detail})
                if self._safety:
                    self._safety.record("GPS_SPOOFING", alert.detail, time.time())
                alerts.append(alert)

        self._last_measurements[measurement.name] = measurement.offset
        return measurement, alerts
