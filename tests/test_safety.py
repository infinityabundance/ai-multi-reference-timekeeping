from ai_multi_reference_timekeeping.safety import Hazard, RiskMatrix, SafetyCase


def test_risk_matrix_scoring() -> None:
    assert RiskMatrix.score(2, "C") == 9
    assert RiskMatrix.acceptable(4, "E")


def test_safety_case_records_occurrence() -> None:
    safety = SafetyCase()
    safety.register(
        Hazard(
            code="GPS_SPOOFING",
            description="GPS spoofing detected",
            severity=2,
            likelihood="C",
            mitigation="Cross-check references",
        )
    )
    safety.record("GPS_SPOOFING", "offset spike", 1.0)
    assert safety.occurrences
