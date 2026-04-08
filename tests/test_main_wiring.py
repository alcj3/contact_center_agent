from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.classifier import ClassificationResult
from app.main import app
from app.state_manager import conversation_store

client = TestClient(app)

CONVERSATION_ID = "test-conv"


def setup_function() -> None:
    conversation_store.clear()


def make_payload(text: str = "hello") -> dict:
    return {
        "conversation_id": CONVERSATION_ID,
        "message": {"role": "customer", "text": text},
    }


def enter_standard_patches(stack: ExitStack, intent: str, confidence: float, escalate: bool) -> None:
    """Enter patches for all three collaborators onto an ExitStack."""
    escalate_return = (True, "reason") if escalate else (False, "")
    stack.enter_context(patch("app.main.classifier.predict",
                              return_value=ClassificationResult(intent=intent, confidence=confidence)))
    stack.enter_context(patch("app.main.should_escalate", return_value=escalate_return))
    stack.enter_context(patch("app.main.responder.generate", return_value="mocked response"))


# ---------------------------------------------------------------------------
# 1. state.confidence reflects the classifier's confidence value
# ---------------------------------------------------------------------------

def test_state_confidence_is_written_after_chat() -> None:
    with ExitStack() as stack:
        enter_standard_patches(stack, intent="Billing", confidence=0.85, escalate=False)
        client.post("/chat", json=make_payload())

    state = conversation_store[CONVERSATION_ID]
    assert state.confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# 2. state.intent_history has the intent appended
# ---------------------------------------------------------------------------

def test_intent_is_appended_to_intent_history_after_chat() -> None:
    with ExitStack() as stack:
        enter_standard_patches(stack, intent="Billing", confidence=0.9, escalate=False)
        client.post("/chat", json=make_payload())

    state = conversation_store[CONVERSATION_ID]
    assert state.intent_history == ["Billing"]


# ---------------------------------------------------------------------------
# 3. state.escalated is True when should_escalate() returns True
# ---------------------------------------------------------------------------

def test_state_escalated_is_true_when_escalation_triggered() -> None:
    with ExitStack() as stack:
        enter_standard_patches(stack, intent="Billing", confidence=0.1, escalate=True)
        client.post("/chat", json=make_payload())

    state = conversation_store[CONVERSATION_ID]
    assert state.escalated is True


# ---------------------------------------------------------------------------
# 4. state.escalated remains False when should_escalate() returns False
# ---------------------------------------------------------------------------

def test_state_escalated_is_false_when_no_escalation() -> None:
    with ExitStack() as stack:
        enter_standard_patches(stack, intent="Billing", confidence=0.9, escalate=False)
        client.post("/chat", json=make_payload())

    state = conversation_store[CONVERSATION_ID]
    assert state.escalated is False


# ---------------------------------------------------------------------------
# 5. Multiple turns append to intent_history, not overwrite
# ---------------------------------------------------------------------------

def test_multiple_turns_append_to_intent_history() -> None:
    with ExitStack() as stack:
        stack.enter_context(patch("app.main.should_escalate", return_value=(False, "")))
        stack.enter_context(patch("app.main.responder.generate", return_value="mocked response"))
        mock_predict = stack.enter_context(patch("app.main.classifier.predict"))

        mock_predict.return_value = ClassificationResult(intent="Billing", confidence=0.9)
        client.post("/chat", json=make_payload("billing question"))

        mock_predict.return_value = ClassificationResult(intent="Technical", confidence=0.8)
        client.post("/chat", json=make_payload("app is broken"))

        mock_predict.return_value = ClassificationResult(intent="Cancellation", confidence=0.7)
        client.post("/chat", json=make_payload("want to cancel"))

    state = conversation_store[CONVERSATION_ID]
    assert state.intent_history == ["Billing", "Technical", "Cancellation"]
