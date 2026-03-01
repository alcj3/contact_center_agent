from __future__ import annotations

from app.models import MetricsResponse


def get_metrics_snapshot() -> MetricsResponse:
    # TODO: Compute metrics from in-memory conversation data.
    return MetricsResponse(
        total_conversations=0,
        average_messages_per_conversation=0.0,
        escalation_rate=0.0,
        average_classifier_confidence=0.0,
        intent_distribution={},
    )
