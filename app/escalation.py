from __future__ import annotations

from app.state_manager import ConversationState

CONFIDENCE_THRESHOLD = 0.4
SENTIMENT_THRESHOLD = -0.5
FAILED_ATTEMPTS_LIMIT = 3


def should_escalate(state: ConversationState, confidence: float) -> tuple[bool, str]:
    if confidence < CONFIDENCE_THRESHOLD:
        return True, f"Low confidence score ({confidence:.2f}) — agent could not understand the request."

    if state.sentiment_score < SENTIMENT_THRESHOLD:
        return True, f"Negative sentiment detected ({state.sentiment_score:.2f}) — customer may be frustrated."

    if state.failed_attempts >= FAILED_ATTEMPTS_LIMIT:
        return True, f"Too many failed attempts ({state.failed_attempts}) — customer needs direct assistance."

    return False, ""
