from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.responder import ResponseEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(reply: str = "How can I help with your bill?") -> MagicMock:
    """Return a mock Anthropic client whose .messages.create() returns `reply`."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=reply)]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    return mock_client


# ---------------------------------------------------------------------------
# (1) Returns a non-empty string response
# ---------------------------------------------------------------------------

def test_generate_returns_non_empty_string() -> None:
    engine = ResponseEngine(client=_make_client("You're all set!"))
    result = engine.generate(intent="General Question", user_text="hello", history=[])

    assert isinstance(result, str)
    assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# (2) Response varies based on intent
# ---------------------------------------------------------------------------

def test_generate_billing_intent() -> None:
    engine = ResponseEngine(client=_make_client("Here is your billing summary."))
    engine.generate(intent="Billing", user_text="I was charged twice", history=[])

    call_args = engine.client.messages.create.call_args
    system_prompt = call_args.kwargs.get("system", "")
    assert "billing" in system_prompt.lower()


def test_generate_technical_intent() -> None:
    engine = ResponseEngine(client=_make_client("Let's troubleshoot your issue."))
    engine.generate(intent="Technical", user_text="my app keeps crashing", history=[])

    call_args = engine.client.messages.create.call_args
    system_prompt = call_args.kwargs.get("system", "")
    assert "technical" in system_prompt.lower()


def test_generate_cancellation_intent() -> None:
    engine = ResponseEngine(client=_make_client("I'm sorry to hear you want to cancel."))
    engine.generate(intent="Cancellation", user_text="cancel my account", history=[])

    call_args = engine.client.messages.create.call_args
    system_prompt = call_args.kwargs.get("system", "")
    assert "cancellation" in system_prompt.lower()


def test_generate_different_intents_produce_different_system_prompts() -> None:
    billing_engine = ResponseEngine(client=_make_client())
    technical_engine = ResponseEngine(client=_make_client())

    billing_engine.generate(intent="Billing", user_text="overcharged", history=[])
    technical_engine.generate(intent="Technical", user_text="app broken", history=[])

    billing_prompt = billing_engine.client.messages.create.call_args.kwargs.get("system", "")
    technical_prompt = technical_engine.client.messages.create.call_args.kwargs.get("system", "")

    assert billing_prompt != technical_prompt


# ---------------------------------------------------------------------------
# (3) Conversation history is passed to the API
# ---------------------------------------------------------------------------

def test_generate_includes_history_in_messages() -> None:
    history = ["I need help with my bill.", "My account number is 1234."]
    engine = ResponseEngine(client=_make_client())
    engine.generate(intent="Billing", user_text="why was I charged?", history=history)

    call_args = engine.client.messages.create.call_args
    messages_sent = call_args.kwargs.get("messages", [])
    message_texts = [m["content"] for m in messages_sent if isinstance(m.get("content"), str)]

    for prior_msg in history:
        assert any(prior_msg in text for text in message_texts)


def test_generate_empty_history_still_works() -> None:
    engine = ResponseEngine(client=_make_client("Sure!"))
    engine.generate(intent="General Question", user_text="hello", history=[])

    call_args = engine.client.messages.create.call_args
    messages_sent = call_args.kwargs.get("messages", [])
    # Only the current user_text message should be present
    assert len(messages_sent) == 1


# ---------------------------------------------------------------------------
# (4) Handles API errors gracefully
# ---------------------------------------------------------------------------

def test_generate_returns_fallback_on_api_error() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API timeout")

    engine = ResponseEngine(client=mock_client)
    result = engine.generate(intent="Billing", user_text="help", history=[])

    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_generate_fallback_does_not_raise() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("connection refused")

    engine = ResponseEngine(client=mock_client)
    try:
        engine.generate(intent="Technical", user_text="broken", history=[])
    except Exception:  # noqa: BLE001
        pytest.fail("generate() raised an exception on API error instead of returning a fallback")
