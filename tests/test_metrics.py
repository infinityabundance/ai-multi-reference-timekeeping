import pytest

from ai_multi_reference_timekeeping.metrics import holdover_stats, mtie, tdev


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
