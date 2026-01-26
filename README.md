# 🕒 ai-multi-reference-timekeeping
**AI-Assisted Multi-Reference Timekeeping for Commodity Networks**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Zenodo DOI](https://img.shields.io/badge/DOI-Zenodo-inactive.svg)](https://zenodo.org/)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-orange.svg)](https://colab.research.google.com/)
[![IEEE 1588](https://img.shields.io/badge/IEEE-1588%20PTP-lightgrey.svg)](https://standards.ieee.org/standard/1588-2008.html)

This repository contains the reference implementation and reproducibility artifacts for the paper:

**_An AI-Assisted Multi-Reference Timekeeping Architecture for Commodity Networks_**  
👤 Riaan de Beer  
📄 Zenodo DOI: [https://zenodo.org/records/18366358](https://zenodo.org/records/18366358)

This project explores a low-cost, AI-assisted approach to time synchronization that synthesizes a **virtual master clock** from multiple imperfect timing references (e.g., GNSS 🌍, PTP 🔗, NTP 🌐) using classical estimation techniques augmented with lightweight machine learning 🤖.

The goal is to improve **practical packet-level synchronization** on commodity hardware — without requiring atomic clocks ⚛️ or specialized time cards.

---

## 🎯 Motivation

High-precision time synchronization is increasingly important for distributed systems, including:

- ⏱️ Time-sensitive networking (TSN)
- 💾 Coordinated I/O and storage pipelines
- 📦 Packet scheduling and timestamping
- 🧪 Experimental distributed systems research

Commercial solutions typically rely on **atomic oscillators and dedicated PCIe time cards**, which remain costly and inaccessible to many researchers and open-source projects.

This work investigates whether **intelligent multi-reference fusion**, combined with lightweight local learning, can narrow the gap for practical synchronization tasks using commodity hardware.

---

## 🧠 What This Project Does

- 🔀 Fuses multiple heterogeneous timing references into a **single virtual clock**
- 📐 Combines a **state-space clock model** with a **lightweight neural network**
- 📊 Adapts reference weighting based on observed jitter, stability, and context
- 🔌 Exposes time via standard mechanisms (PTP, PHC, NTP)
- ♻️ Targets reproducibility using open-source tools and Google Colab notebooks

---

## 🛰️ Time Server Scaffold (Sensors + AI Weighting)

The repository now includes a **time server scaffold** in `src/ai_multi_reference_timekeeping/time_server.py`
that lets you:

- 🧩 Plug in sensor inputs (temperature, humidity, pressure, AC hum, SDR SNR, Geiger CPM, audio activity)
- 📡 Collect references over NTP, GPS NMEA, or the hardware RTC (via `hwclock`)
- 🔌 Listen from GPIO/USB/serial by wiring sensors with `GpioPulseSensor`, `SerialLineSensor`, or `open_line_source`
- 🎙️ Extract AC hum, ambient audio, bird, and traffic activity features with `AudioFeatureSensor`
- 🧠 Adjust reference variance using heuristic, linear, or online ML models
- 📉 Estimate drift and slew from recent offsets
- ⚖️ Support heuristic fusion via `HeuristicFusion` when quality scores are available
- 📏 Provide TDEV/MTIE/holdover metrics and Chrony SHM integration helpers
- 🛡️ Include sensor validation, rate limiting, and anomaly detection for spoofing and flooding mitigation
- 🧪 Support sensor characterization and I2C environmental adapters for temperature/pressure tracking
- ⚙️ Provide Pydantic settings, structured logging, and metrics/health endpoints
- ✅ Safety case tracking aligned with MIL-STD-882E / DO-178C / NASA NPR 7150.2D
- 🧭 Partition supervision and fault containment inspired by STANAG 4626

Example usage:

```python
from ai_multi_reference_timekeeping.fusion import HeuristicFusion, VirtualClock
from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState
from ai_multi_reference_timekeeping.time_server import (
    AudioFeatureSensor,
    LinearInferenceModel,
    LightweightInferenceModel,
    MlVarianceModel,
    NtpReference,
    EnvironmentalSensor,
    I2CEnvironmentalSensor,
    SensorAggregator,
    TimeServer,
)

kalman = ClockKalmanFilter(
    state=ClockState(offset=0.0, drift=0.0),
    covariance=ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
    process_noise_offset=1e-4,
    process_noise_drift=1e-6,
)
clock = VirtualClock(kalman_filter=kalman, fusion=HeuristicFusion())

class EnvSensor:
    def sample(self) -> dict[str, float]:
        return {"temperature_c": 27.0, "humidity_pct": 40.0}

class AudioSource:
    def sample(self) -> tuple[list[float], int]:
        return [0.0] * 128, 8000

server = TimeServer(
    clock=clock,
    references=[NtpReference(name="nist")],
    sensors=SensorAggregator(
        EnvSensor(),
        AudioFeatureSensor(AudioSource()),
        EnvironmentalSensor(lambda: (27.0, 40.0, 1010.0)),
        I2CEnvironmentalSensor(lambda bus, address: (27.1, 41.0, 1009.5), bus=1, address=0x76),
    ),
    inference=MlVarianceModel(feature_weights={"temperature_c": 0.02, "humidity_pct": 0.01}),
)

update, frame, drift_estimate, drift_hint = server.step(dt=1.0)
print(update.fused_offset, drift_estimate.drift, drift_hint)
```

## ✅ Quickstart API (recommended)

Use the `build_time_server` API to assemble a ready-to-run server with
observability and partition supervision configured:

```python
from ai_multi_reference_timekeeping.api import build_time_server
from ai_multi_reference_timekeeping.time_server import NtpReference, SensorAggregator, SensorInput

class EnvSensor:
    def sample(self) -> dict[str, float]:
        return {"temperature_c": 27.0, "humidity_pct": 40.0}

runtime = build_time_server(
    references=[NtpReference(name="nist")],
    sensors=[EnvSensor()],
)
server = runtime.server
update, frame, drift, drift_hint = server.step(1.0)
```

Chrony integration and metrics utilities:

```python
from ai_multi_reference_timekeeping.chrony import ChronyShmSample, ChronyShmWriter
from ai_multi_reference_timekeeping.metrics import holdover_stats, mtie, tdev

writer = ChronyShmWriter()
writer.write(ChronyShmSample(offset=0.001, delay=0.0001))

offsets = [0.0, 0.0005, 0.001]
print(tdev(offsets, tau=1))
print(mtie(offsets, window=2))
print(holdover_stats(offsets, sample_interval=1.0))
```

Security and anomaly mitigation example:

```python
from ai_multi_reference_timekeeping.time_server import SecurityMonitor, SensorValidator

validator = SensorValidator(max_samples_per_sec=2.0)
monitor = SecurityMonitor(divergence_threshold=0.01, grid_frequency=60.0)
print(validator.validate({"temperature_c": 20.0, "humidity_pct": 45.0}))
print(monitor.evaluate_frame(frame))
```

Sensor characterization example:

```python
from ai_multi_reference_timekeeping.characterization import SensorCharacterization

characterization = SensorCharacterization()
characterization.update("gps", 0.0002)
print(characterization.z_score("gps", 0.0005))
```

Configuration + structured logging example:

```python
from ai_multi_reference_timekeeping.config import TimeServerSettings
from ai_multi_reference_timekeeping.logging_utils import configure_logging

settings = TimeServerSettings()
configure_logging(settings.logging)
```

Metrics/health exporter example:

```python
from ai_multi_reference_timekeeping.observability import HealthMonitor, MetricsExporter, MetricsTracker

tracker = MetricsTracker(window_size=60)
health = HealthMonitor(freshness_window=10.0)
exporter = MetricsExporter(tracker, health)
exporter.start(host="0.0.0.0", port=8000)
```

Safety case example:

```python
from ai_multi_reference_timekeeping.safety import Hazard, SafetyCase

safety = SafetyCase()
safety.register(
    Hazard(
        code="GPS_SPOOFING",
        description="Spoofing detected",
        severity=2,
        likelihood="C",
        mitigation="Cross-check GNSS/PTP/AC-hum",
    )
)
```

Partition supervision example:

```python
from ai_multi_reference_timekeeping.partitioning import PartitionSupervisor

supervisor = PartitionSupervisor(max_failures=2, reboot_delay=2.0)
```

---

## 🧠 ML Logic (How the AI weighting works)

The ML pipeline is intentionally lightweight and auditable:

1. **Feature extraction**  
   Sensor inputs are normalized through `SensorValidator` and mapped into a `SensorFrame`. Each field (e.g., temperature, humidity, AC hum frequency/phase) represents a context signal that can impact reference stability.  
2. **Variance scaling (adaptive weighting)**  
   - `LightweightInferenceModel` uses heuristic penalties (temperature drift, AC hum deviation, audio activity) to scale variance.  
   - `LinearInferenceModel` applies a logistic transform to feature-weighted scores.  
   - `MlVarianceModel` applies bounded online updates and residual characterization to adjust reference bias.  
3. **Fusion and update**  
   Weighted measurements are fused via `HeuristicFusion`, and the Kalman filter updates the virtual clock.  
4. **Safety hooks**  
   Security alerts can record hazards via `SafetyCase`, and partition supervision isolates failing sensors.

This design emphasizes **explicit, bounded, and logged** updates to support traceability and operational safety.

---

## ⚠️ Known limitations and roadmap

The following items are acknowledged gaps; they are explicitly tracked and should be addressed before claiming production readiness:

1. No long-duration integration/performance tests or 24h memory-leak checks.  
2. Limited thread-safety verification beyond a coarse lock in `TimeServer`.  
3. No property-based or fuzz testing for edge cases.  
4. Sensor aggregator logging is not thread-local.  
5. SHM write atomicity vs Chrony read race not validated.  
6. Sensor I/O blocking could stall fusion loops (no async I/O).  
7. `ValueError` used for “no measurements” control flow.  
8. Sensor failures are logged but may accumulate without operator intervention.  
9. `MlVarianceModel` rejection thresholds are heuristic (3σ).  
10. Goertzel phase unwrapping not formally verified.  
11. Feature scaling/normalization not standardized.  
12. CRC32 collision probability not quantified in docs.  
13. Rate limiting is basic and not token-bucket based.  
14. `/health` and `/metrics` are unauthenticated.  
15. Secrets handling not implemented in config.  
16. Pydantic and TOML are shimmed for local use; a full dependency strategy is needed.  
17. No schema validation beyond runtime checks.  
18. Structured logging coverage is partial; tracing not implemented.  
19. Prometheus/OpenTelemetry exporters are not implemented.  
20. No profiling hooks or performance counters.  
21. Model selection criteria (ML vs linear vs heuristic) not automated.  
22. Online learning convergence is not guaranteed.  
23. Model versioning and rollback are not implemented.  
24. Training data quality validation is minimal.  
25. Hazard tracking is not coupled to automated mitigation.  
26. Risk matrix scoring is not calibrated to operational data.  
27. No FMEA artifacts.  
28. Hardware validation not performed.  

These are explicit non-goals in the current iteration, but they are documented here for transparency and future planning.

---

## ✅ Test coverage and validation notes

The test suite includes:

- **Unit tests** for fusion, Kalman filter, metrics, safety case, partitioning, and Chrony SHM CRC.
- **Smoke tests** for configuration defaults, observability tracker metrics, and sensor validation.

The following test categories are planned but not fully implemented:

- Thread-safety stress tests (multi-threaded `TimeServer.step`).
- Integration tests with real Chrony and hardware sensors.
- Performance benchmarks for high-rate sensor streams.
- Fault injection for sensor outages and degraded timing references.

These items are noted to align with the requested engineering rigor.

---

## 🚫 Non-goals

This project explicitly does **not** aim to:

- ❌ Replace atomic clocks or high-stability oscillators
- ❌ Provide nanosecond-level absolute UTC accuracy under all conditions
- ❌ Serve as a primary time standard
- ❌ Offer cryptographic guarantees against fully adversarial time manipulation

The system prioritizes **robustness, accessibility, and cost-effectiveness** for packet-level synchronization in experimental and operational environments.

---

## 🗂️ Repository Structure

    ai-multi-reference-timekeeping/
    ├── paper/              # 📄 LaTeX source and figures for the paper
    ├── notebooks/          # 📓 Google Colab–friendly notebooks
    ├── src/                # 🧩 Fusion, ML, and evaluation code
    ├── data/               # 🗃️ Example and processed datasets
    ├── models/             # 🧠 Trained and baseline models
    ├── configs/            # ⚙️ Configuration files
    ├── scripts/            # 🛠️ CLI utilities
    ├── reproducibility/    # 🔁 Experimental protocols and hardware notes
    ├── environment/        # 📦 Dependency specifications
    ├── LICENSE
    └── README.md

---

## ☁️ Google Colab Reproducibility

The notebooks in `notebooks/` are designed to run directly in **Google Colab** — no specialized hardware required.

🔰 **Recommended entry point**:
- `notebooks/00_overview.ipynb` — overview of the architecture and experiments  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-org/ai-multi-reference-timekeeping/blob/main/notebooks/00_overview.ipynb)

✅ **Notebook test runs**:
- `notebooks/10_test_fusion.ipynb` — validates fusion and quality weighting  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-org/ai-multi-reference-timekeeping/blob/main/notebooks/10_test_fusion.ipynb)
- `notebooks/11_test_time_server.ipynb` — validates time server + ML variance model  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-org/ai-multi-reference-timekeeping/blob/main/notebooks/11_test_time_server.ipynb)

Each notebook includes an **Open in Colab** link and installs dependencies automatically.

---

## 📈 Evaluation Metrics

The evaluation framework focuses on standard timing metrics, including:

- ⏲️ Time Deviation (TDEV)
- 📏 Maximum Time Interval Error (MTIE)
- 🔄 PTP offset stability
- 🕰️ Holdover behavior during reference loss

Absolute UTC ground truth is not required for most experiments.

---

## 🔐 Security and Threat Model

The system is designed to tolerate noisy, intermittent, and partially unreliable timing sources.  
It does **not** assume a fully adversarial environment.

Considered threats include:

- 📡 GNSS degradation, multipath, and interference
- 🔀 Network-induced delay asymmetry
- ⚠️ Transient reference instability

Coordinated compromise of all timing references is considered out of scope.

---

## 📜 License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this work, please cite:

    @misc{debeer2026aimrt,
      title  = {An AI-Assisted Multi-Reference Timekeeping Architecture for Commodity Networks},
      author = {de Beer, Riaan},
      year   = {2026},
      doi    = {10.5281/zenodo.XXXXXXX}
    }

---

## 🚧 Status

This repository accompanies a research paper and is intended to evolve.  
Contributions, discussion, and replication studies are welcome 🤝.

---

## 🙏 Acknowledgments

This work builds on established research in time metrology, clock ensembles,
and IEEE 1588 Precision Time Protocol, and aims to make these ideas more
accessible to open-source and experimental systems communities.

## 🚧 Status

This repository accompanies a research paper and is intended to evolve.  
Contributions, discussion, and replication studies are welcome 🤝.

---

## 🙏 Acknowledgments

This work builds on established research in time metrology, clock ensembles,
and IEEE 1588 Precision Time Protocol, and aims to make these ideas more
accessible to open-source and experimental systems communities.
