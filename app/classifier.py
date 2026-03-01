from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ClassificationResult:
    intent: str
    confidence: float


class IntentClassifier:
    """Stub classifier interface for future ML implementation."""

    def __init__(self) -> None:
        # TODO: Initialize vectorizer/model artifacts.
        pass

    def predict(self, text: str) -> ClassificationResult:
        # TODO: Replace stub intent prediction with ML-based classification.
        _ = text
        return ClassificationResult(intent="General Question", confidence=0.0)
