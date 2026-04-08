from __future__ import annotations

from app.models import MetricsResponse
from app.state_manager import conversation_store


def get_metrics_snapshot() -> MetricsResponse:
    conversations = list(conversation_store.values())

    if not conversations:
        return MetricsResponse(
            total_conversations=0,
            average_messages_per_conversation=0.0,
            average_sentiment_score=0.0,
            escalation_rate=0.0,
            intent_distribution={},
        )

    total = len(conversations)

    average_messages = sum(len(s.messages) for s in conversations) / total

    average_sentiment = sum(s.sentiment_score for s in conversations) / total

    escalated_count = sum(1 for s in conversations if s.escalated)
    escalation_rate = escalated_count / total

    intent_distribution: dict[str, int] = {}
    for state in conversations:
        for intent in state.intent_history:
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1

    return MetricsResponse(
        total_conversations=total,
        average_messages_per_conversation=average_messages,
        average_sentiment_score=average_sentiment,
        escalation_rate=escalation_rate,
        intent_distribution=intent_distribution,
    )
