from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["customer", "agent"]
    text: str = Field(min_length=1)


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    message: ChatMessage


class ChatResponse(BaseModel):
    conversation_id: str
    response_text: str
    escalated: bool = False


class HealthResponse(BaseModel):
    status: str


class MetricsResponse(BaseModel):
    total_conversations: int
    average_messages_per_conversation: float
    escalation_rate: float
    average_sentiment_score: float
    intent_distribution: dict[str, int]
