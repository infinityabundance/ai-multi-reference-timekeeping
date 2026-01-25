import math

import pytest

from ai_multi_reference_timekeeping.fusion import Measurement, ReferenceFusion


def test_reference_fusion_weights_average():
    fusion = ReferenceFusion()
    measurements = [
        Measurement(name="gnss", offset=1.0, variance=4.0),
        Measurement(name="ptp", offset=3.0, variance=1.0),
    ]

    fused_offset, fused_variance, weights = fusion.fuse(measurements)

    # Expected weighted mean: (1/4 * 1.0 + 1/1 * 3.0) / (1/4 + 1)
    expected_offset = (0.25 * 1.0 + 1.0 * 3.0) / 1.25
    expected_variance = 1.0 / 1.25

    assert math.isclose(fused_offset, expected_offset, rel_tol=1e-9)
    assert math.isclose(fused_variance, expected_variance, rel_tol=1e-9)
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
    assert weights["ptp"] > weights["gnss"]


def test_reference_fusion_requires_measurements():
    fusion = ReferenceFusion()
    with pytest.raises(ValueError):
        fusion.fuse([])
