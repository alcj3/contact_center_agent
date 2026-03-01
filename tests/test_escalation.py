from app.escalation import should_escalate


def test_escalation_stub_returns_false() -> None:
    result = should_escalate(
        user_text="agent please",
        confidence=0.2,
        sentiment_score=-0.8,
        repeated_intent_count=3,
    )
    assert result is False
