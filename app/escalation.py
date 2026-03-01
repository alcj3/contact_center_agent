from __future__ import annotations


def should_escalate(
    user_text: str,
    confidence: float,
    sentiment_score: float,
    repeated_intent_count: int,
) -> bool:
    # TODO: Implement deterministic escalation rules.
    _ = (user_text, confidence, sentiment_score, repeated_intent_count)
    return False
