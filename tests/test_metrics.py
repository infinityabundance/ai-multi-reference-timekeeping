from pathlib import Path

import pytest

from ai_multi_reference_timekeeping.metrics import holdover_stats, mtie, tdev
from ai_multi_reference_timekeeping.observability import MetricsTracker


def test_tdev_requires_samples() -> None:
    with pytest.raises(ValueError):
        tdev([0.0, 0.1], tau=2)


def test_mtie_computes_peak_interval_error() -> None:
    offsets = [0.0, 0.1, -0.05, 0.2]
    assert mtie(offsets, window=2) == pytest.approx(0.25)


def test_holdover_stats_summary() -> None:
    stats = holdover_stats([0.0, -0.1, 0.1], sample_interval=1.0)
    assert stats.max_offset == pytest.approx(0.1)
    assert stats.duration == pytest.approx(2.0)


def test_metrics_from_sample_data() -> None:
    data_path = Path(__file__).parent / "data" / "offsets.csv"
    offsets = []
    for line in data_path.read_text().splitlines()[1:]:
        offsets.append(float(line))
    assert tdev(offsets, tau=1) > 0.0
    assert mtie(offsets, window=2) > 0.0


def test_metrics_tracker_exports_metrics() -> None:
    tracker = MetricsTracker(window_size=5)
    for offset in [0.0, 0.0002, 0.0001, -0.0001, 0.0003]:
        tracker.update(offset)
    metrics = tracker.metrics()
    assert "tdev_tau1" in metrics
    assert "mtie_window2" in metrics
