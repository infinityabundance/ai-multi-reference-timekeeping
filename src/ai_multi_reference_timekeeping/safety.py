"""Safety case artifacts inspired by military and aerospace standards.

This module provides lightweight hazard tracking and risk scoring aligned with
MIL-STD-882E, DO-178C, NASA NPR 7150.2D, MIL-STD-498, STANAG 4626, IEC 61508,
and NASA Power of 10 principles. It is intentionally simple and explicit to
remain analyzable and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Hazard:
    """Hazard definition with severity and likelihood.

    Severity and likelihood follow MIL-STD-882E qualitative levels:
    - severity: 1 (catastrophic) .. 4 (negligible)
    - likelihood: A (frequent) .. E (improbable)
    """

    code: str
    description: str
    severity: int
    likelihood: str
    mitigation: str


@dataclass
class HazardOccurrence:
    """Record of a hazard occurrence."""

    code: str
    detail: str
    timestamp: float


class RiskMatrix:
    """Simple MIL-STD-882E risk matrix."""

    _order = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

    @classmethod
    def score(cls, severity: int, likelihood: str) -> int:
        # Higher score indicates higher risk.
        if severity < 1 or severity > 4:
            raise ValueError("severity must be 1..4")
        if likelihood not in cls._order:
            raise ValueError("likelihood must be A..E")
        return (5 - severity) * cls._order[likelihood]

    @classmethod
    def acceptable(cls, severity: int, likelihood: str, threshold: int = 6) -> bool:
        # Acceptable risk threshold is configurable.
        return cls.score(severity, likelihood) <= threshold


@dataclass
class SafetyCase:
    """Safety case state with hazard registry and occurrence log."""

    hazards: dict[str, Hazard] = field(default_factory=dict)
    occurrences: list[HazardOccurrence] = field(default_factory=list)

    def register(self, hazard: Hazard) -> None:
        # Register hazards before recording occurrences.
        self.hazards[hazard.code] = hazard

    def record(self, code: str, detail: str, timestamp: float) -> None:
        # Record hazard occurrence with timestamp and detail.
        if code not in self.hazards:
            raise ValueError(f"Unknown hazard code {code}")
        self.occurrences.append(HazardOccurrence(code=code, detail=detail, timestamp=timestamp))

    def risk_summary(self) -> dict[str, int]:
        # Provide a risk score per hazard for reporting.
        return {
            code: RiskMatrix.score(hazard.severity, hazard.likelihood)
            for code, hazard in self.hazards.items()
        }

    def unacceptable_hazards(self, threshold: int = 6) -> list[Hazard]:
        # Filter hazards that exceed acceptable risk.
        return [
            hazard
            for hazard in self.hazards.values()
            if not RiskMatrix.acceptable(hazard.severity, hazard.likelihood, threshold=threshold)
        ]
