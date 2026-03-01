from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ConversationState:
    messages: list[str] = field(default_factory=list)
    last_intent: str | None = None
    failed_attempts: int = 0
    sentiment_score: float = 0.0


conversation_store: dict[str, ConversationState] = {}


def get_or_create_state(conversation_id: str) -> ConversationState:
    # TODO: Add persistence backing store when needed.
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = ConversationState()
    return conversation_store[conversation_id]
