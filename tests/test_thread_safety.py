import threading

from ai_multi_reference_timekeeping.fusion import Measurement, ReferenceFusion, VirtualClock
from ai_multi_reference_timekeeping.kalman import ClockCovariance, ClockKalmanFilter, ClockState
from ai_multi_reference_timekeeping.time_server import SensorAggregator, SensorFrame, TimeServer


class StaticReference:
    def __init__(self, name: str, offset: float, variance: float) -> None:
        self._name = name
        self._offset = offset
        self._variance = variance

    def sample(self, frame: SensorFrame) -> Measurement:
        del frame
        return Measurement(name=self._name, offset=self._offset, variance=self._variance)


def _make_server() -> TimeServer:
    kalman = ClockKalmanFilter(
        state=ClockState(offset=0.0, drift=0.0),
        covariance=ClockCovariance(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
        process_noise_offset=1e-4,
        process_noise_drift=1e-6,
    )
    clock = VirtualClock(kalman_filter=kalman, fusion=ReferenceFusion())
    return TimeServer(clock, references=[StaticReference("ref", 0.1, 0.5)], sensors=SensorAggregator())


def test_time_server_thread_safety_smoke() -> None:
    server = _make_server()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(200):
                server.step(1.0)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    # Basic invariant: state is finite after concurrent updates.
    assert abs(server.state.offset) < 10
