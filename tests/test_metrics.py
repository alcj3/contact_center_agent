from __future__ import annotations

import pytest

from app.metrics import get_metrics_snapshot
from app.state_manager import ConversationState, conversation_store


def setup_function() -> None:
    """Clear the store before each test so tests don't bleed into each other."""
    conversation_store.clear()


# ---------------------------------------------------------------------------
# 1. Empty store returns zero values
# ---------------------------------------------------------------------------

def test_empty_store_returns_zero_values() -> None:
    result = get_metrics_snapshot()

    assert result.total_conversations == 0
    assert result.average_messages_per_conversation == 0.0
    assert result.average_sentiment_score == 0.0
    assert result.escalation_rate == 0.0
    assert result.intent_distribution == {}


# ---------------------------------------------------------------------------
# 2. total_conversations
# ---------------------------------------------------------------------------

def test_total_conversations_counts_all_entries() -> None:
    conversation_store["conv-1"] = ConversationState()
    conversation_store["conv-2"] = ConversationState()
    conversation_store["conv-3"] = ConversationState()

    result = get_metrics_snapshot()

    assert result.total_conversations == 3


def test_total_conversations_is_one_when_one_entry() -> None:
    conversation_store["conv-1"] = ConversationState()

    result = get_metrics_snapshot()

    assert result.total_conversations == 1


# ---------------------------------------------------------------------------
# 3. average_sentiment_score
# ---------------------------------------------------------------------------

def test_average_sentiment_score_single_conversation() -> None:
    state = ConversationState()
    state.sentiment_score = -0.6
    conversation_store["conv-1"] = state

    result = get_metrics_snapshot()

    assert result.average_sentiment_score == pytest.approx(-0.6)


def test_average_sentiment_score_multiple_conversations() -> None:
    state_a = ConversationState()
    state_a.sentiment_score = 0.4

    state_b = ConversationState()
    state_b.sentiment_score = -0.2

    conversation_store["conv-1"] = state_a
    conversation_store["conv-2"] = state_b

    result = get_metrics_snapshot()

    assert result.average_sentiment_score == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 4. escalation_rate
# ---------------------------------------------------------------------------

def test_escalation_rate_none_escalated() -> None:
    conversation_store["conv-1"] = ConversationState()
    conversation_store["conv-2"] = ConversationState()

    result = get_metrics_snapshot()

    assert result.escalation_rate == pytest.approx(0.0)


def test_escalation_rate_all_escalated() -> None:
    state_a = ConversationState()
    state_a.escalated = True

    state_b = ConversationState()
    state_b.escalated = True

    conversation_store["conv-1"] = state_a
    conversation_store["conv-2"] = state_b

    result = get_metrics_snapshot()

    assert result.escalation_rate == pytest.approx(1.0)


def test_escalation_rate_half_escalated() -> None:
    state_a = ConversationState()
    state_a.escalated = True

    state_b = ConversationState()
    state_b.escalated = False

    conversation_store["conv-1"] = state_a
    conversation_store["conv-2"] = state_b

    result = get_metrics_snapshot()

    assert result.escalation_rate == pytest.approx(0.5)


def test_escalation_rate_is_between_zero_and_one() -> None:
    state = ConversationState()
    state.escalated = True
    conversation_store["conv-1"] = state

    result = get_metrics_snapshot()

    assert 0.0 <= result.escalation_rate <= 1.0


# ---------------------------------------------------------------------------
# 5. intent_distribution
# ---------------------------------------------------------------------------

def test_intent_distribution_single_intent() -> None:
    state = ConversationState()
    state.intent_history = ["Billing", "Billing", "Billing"]
    conversation_store["conv-1"] = state

    result = get_metrics_snapshot()

    assert result.intent_distribution == {"Billing": 3}


def test_intent_distribution_multiple_intents() -> None:
    state_a = ConversationState()
    state_a.intent_history = ["Billing", "Technical"]

    state_b = ConversationState()
    state_b.intent_history = ["Billing", "Cancellation"]

    conversation_store["conv-1"] = state_a
    conversation_store["conv-2"] = state_b

    result = get_metrics_snapshot()

    assert result.intent_distribution == {
        "Billing": 2,
        "Technical": 1,
        "Cancellation": 1,
    }


def test_intent_distribution_empty_histories() -> None:
    conversation_store["conv-1"] = ConversationState()
    conversation_store["conv-2"] = ConversationState()

    result = get_metrics_snapshot()

    assert result.intent_distribution == {}
