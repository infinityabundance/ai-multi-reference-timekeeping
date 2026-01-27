"""Trainable ML model for variance scaling.

This module defines a minimal, auditable model that learns a linear mapping
from sensor features to a variance scale. The implementation is intentionally
simple to remain analyzable under safety-oriented standards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .time_server import SensorFrame, _feature_value


@dataclass
class TrainingSample:
    """One training example linking sensor features to an ideal scale."""

    frame: SensorFrame
    target_scale: float


@dataclass
class LinearVarianceModel:
    """A trainable linear model for variance scaling.

    This is the "core" ML component: it learns weights from data.
    """

    weights: dict[str, float]
    bias: float = 0.0

    def predict_scale(self, frame: SensorFrame) -> float:
        score = self.bias
        for feature, weight in self.weights.items():
            score += weight * _feature_value(frame, feature)
        # Clamp scale to a safe positive range.
        return max(0.1, min(10.0, 1.0 + score))

    def train(self, samples: Iterable[TrainingSample], learning_rate: float = 1e-3, epochs: int = 100) -> None:
        """Train weights using a simple gradient descent."""

        for _ in range(epochs):
            for sample in samples:
                pred = self.predict_scale(sample.frame)
                error = pred - sample.target_scale
                # Gradient update for bias.
                self.bias -= learning_rate * error
                for feature in self.weights:
                    grad = error * _feature_value(sample.frame, feature)
                    self.weights[feature] -= learning_rate * grad
