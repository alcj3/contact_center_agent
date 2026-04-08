from __future__ import annotations

from app.escalation import should_escalate
from app.state_manager import ConversationState


def make_state(
    sentiment_score: float = 0.0,
    failed_attempts: int = 0,
) -> ConversationState:
    state = ConversationState()
    state.sentiment_score = sentiment_score
    state.failed_attempts = failed_attempts
    return state


# ---------------------------------------------------------------------------
# 1. Normal conditions — no escalation
# ---------------------------------------------------------------------------

def test_no_escalation_under_normal_conditions() -> None:
    state = make_state(sentiment_score=0.0, failed_attempts=0)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is False
    assert reason == ""


# ---------------------------------------------------------------------------
# 2. Low confidence triggers escalation
# ---------------------------------------------------------------------------

def test_low_confidence_triggers_escalation() -> None:
    state = make_state()
    escalate, reason = should_escalate(state, confidence=0.39)

    assert escalate is True


def test_confidence_at_threshold_does_not_escalate() -> None:
    state = make_state()
    escalate, reason = should_escalate(state, confidence=0.4)

    assert escalate is False


# ---------------------------------------------------------------------------
# 3. Negative sentiment triggers escalation
# ---------------------------------------------------------------------------

def test_negative_sentiment_triggers_escalation() -> None:
    state = make_state(sentiment_score=-0.51)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is True


def test_sentiment_at_threshold_does_not_escalate() -> None:
    state = make_state(sentiment_score=-0.5)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is False


# ---------------------------------------------------------------------------
# 4. Too many failed attempts triggers escalation
# ---------------------------------------------------------------------------

def test_failed_attempts_at_limit_triggers_escalation() -> None:
    state = make_state(failed_attempts=3)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is True


def test_failed_attempts_below_limit_does_not_escalate() -> None:
    state = make_state(failed_attempts=2)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is False


# ---------------------------------------------------------------------------
# 5. Reason string is non-empty and descriptive when escalating
# ---------------------------------------------------------------------------

def test_low_confidence_reason_is_descriptive() -> None:
    state = make_state()
    escalate, reason = should_escalate(state, confidence=0.1)

    assert escalate is True
    assert len(reason) > 0
    assert "confidence" in reason.lower()


def test_negative_sentiment_reason_is_descriptive() -> None:
    state = make_state(sentiment_score=-0.9)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is True
    assert len(reason) > 0
    assert "sentiment" in reason.lower()


def test_failed_attempts_reason_is_descriptive() -> None:
    state = make_state(failed_attempts=5)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is True
    assert len(reason) > 0
    assert "attempt" in reason.lower()


# ---------------------------------------------------------------------------
# 6. Each condition independently triggers escalation
# ---------------------------------------------------------------------------

def test_only_low_confidence_triggers_escalation() -> None:
    state = make_state(sentiment_score=0.0, failed_attempts=0)
    escalate, reason = should_escalate(state, confidence=0.1)

    assert escalate is True


def test_only_negative_sentiment_triggers_escalation() -> None:
    state = make_state(sentiment_score=-0.9, failed_attempts=0)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is True


def test_only_failed_attempts_triggers_escalation() -> None:
    state = make_state(sentiment_score=0.0, failed_attempts=3)
    escalate, reason = should_escalate(state, confidence=0.8)

    assert escalate is True
