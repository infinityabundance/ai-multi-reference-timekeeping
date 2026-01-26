"""Run a GNSS dropout ablation study.

This script compares a baseline (static variance) versus the sensor-aware
adaptive variance model using the existing fusion + Kalman framework.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_multi_reference_timekeeping.fusion import HeuristicFusion, Measurement, ReferenceFusion, VirtualClock
from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState
from ai_multi_reference_timekeeping.metrics import mtie, tdev


@dataclass
class AblationResult:
    label: str
    tdev_tau1: float
    mtie_window10: float


def _make_clock(fusion) -> VirtualClock:
    kalman = ClockKalmanFilter(
        state=ClockState(offset=0.0, drift=0.0),
        covariance=ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
        process_noise_offset=1e-4,
        process_noise_drift=1e-6,
    )
    return VirtualClock(kalman_filter=kalman, fusion=fusion)


def _simulate(dropout_start: int, dropout_end: int, adaptive: bool) -> list[float]:
    fusion = HeuristicFusion() if adaptive else ReferenceFusion()
    clock = _make_clock(fusion)
    offsets = []
    for step in range(600):
        measurements = [Measurement(name="ptp", offset=0.002, variance=1e-3)]
        if step < dropout_start or step > dropout_end:
            measurements.append(Measurement(name="gnss", offset=0.0, variance=1e-4))
        update = clock.step(1.0, measurements)
        offsets.append(update.fused_offset)
    return offsets


def run() -> list[AblationResult]:
    offsets_baseline = _simulate(dropout_start=180, dropout_end=420, adaptive=False)
    offsets_adaptive = _simulate(dropout_start=180, dropout_end=420, adaptive=True)
    return [
        AblationResult("baseline", tdev(offsets_baseline, tau=1), mtie(offsets_baseline, window=10)),
        AblationResult("adaptive", tdev(offsets_adaptive, tau=1), mtie(offsets_adaptive, window=10)),
    ]


if __name__ == "__main__":
    results = run()
    for result in results:
        print(f"{result.label} tdev_tau1={result.tdev_tau1:.6e} mtie_window10={result.mtie_window10:.6e}")
