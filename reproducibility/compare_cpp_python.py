"""Compare C++ and Python fusion results for a deterministic scenario."""

from __future__ import annotations

import sys

from ai_multi_reference_timekeeping.fusion import HeuristicFusion, VirtualClock
from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState
from ai_multi_reference_timekeeping.time_server import PythonTimeServer, SensorAggregator

try:
    import aimrt_python as cpp
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "C++ bindings not available. Build with -DAIMRT_BUILD_PYTHON=ON before running this script."
    ) from exc


class PyReference:
    def __init__(self, name: str, variance: float, slope: float) -> None:
        self._name = name
        self._variance = variance
        self._slope = slope
        self._step = 0

    def sample(self, frame):
        from ai_multi_reference_timekeeping.fusion import Measurement

        offset = self._slope * self._step
        self._step += 1
        return Measurement(name=self._name, offset=offset, variance=self._variance)


class CppReference(cpp.ReferenceInput):
    def __init__(self, name: str, variance: float, slope: float) -> None:
        super().__init__()
        self._name = name
        self._variance = variance
        self._slope = slope
        self._step = 0

    def sample(self, frame: cpp.SensorFrame) -> cpp.Measurement:
        measurement = cpp.Measurement()
        measurement.name = self._name
        measurement.offset = self._slope * self._step
        measurement.variance = self._variance
        self._step += 1
        return measurement


def run_python(steps: int) -> list[float]:
    kalman = ClockKalmanFilter(
        state=ClockState(offset=0.0, drift=0.0),
        covariance=ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
        process_noise_offset=1e-4,
        process_noise_drift=1e-6,
    )
    clock = VirtualClock(kalman_filter=kalman, fusion=HeuristicFusion())
    references = [
        PyReference("ref_a", variance=1e-6, slope=1e-6),
        PyReference("ref_b", variance=2e-6, slope=-2e-6),
        PyReference("ref_c", variance=5e-7, slope=5e-7),
    ]
    server = PythonTimeServer(clock=clock, references=references, sensors=SensorAggregator())

    offsets: list[float] = []
    for _ in range(steps):
        update, _frame, _drift, _drift_hint = server.step(1.0)
        offsets.append(update.fused_offset)
    return offsets


def run_cpp(steps: int) -> list[float]:
    references = [
        CppReference("ref_a", variance=1e-6, slope=1e-6),
        CppReference("ref_b", variance=2e-6, slope=-2e-6),
        CppReference("ref_c", variance=5e-7, slope=5e-7),
    ]
    runtime = cpp.build_time_server_py(references, [])

    offsets: list[float] = []
    for _ in range(steps):
        update, _frame, _drift, _drift_hint = runtime.server.step(1.0)
        offsets.append(update.fused_offset)
    return offsets


def main() -> int:
    steps = 10_000
    py_offsets = run_python(steps)
    cpp_offsets = run_cpp(steps)

    max_diff = max(abs(a - b) for a, b in zip(py_offsets, cpp_offsets))
    print(f"Max absolute difference: {max_diff:.3e} s")
    if max_diff >= 1e-9:
        raise AssertionError(f"Max difference {max_diff:.3e} exceeds threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
