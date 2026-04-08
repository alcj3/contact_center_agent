from __future__ import annotations

from fastapi import FastAPI

from app.classifier import IntentClassifier
from app.escalation import should_escalate
from app.metrics import get_metrics_snapshot
from app.models import ChatRequest, ChatResponse, HealthResponse, MetricsResponse
from app.responder import ResponseEngine
from app.state_manager import get_or_create_state

app = FastAPI(title="AI Contact Center Agent", version="0.1.0")

classifier = IntentClassifier()
responder = ResponseEngine()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """Stub chat endpoint for future orchestration logic."""
    state = get_or_create_state(payload.conversation_id)
    state.messages.append(payload.message.text)

    classification = classifier.predict(payload.message.text)
    escalate = should_escalate(
        user_text=payload.message.text,
        confidence=classification.confidence,
        sentiment_score=state.sentiment_score,
        repeated_intent_count=1,
    )

    response_text = responder.generate(classification.intent, payload.message.text, history=state.messages[:-1])

    # TODO: Record classifier confidence, intent counts, and escalation events in metrics.
    return ChatResponse(
        conversation_id=payload.conversation_id,
        response_text=response_text,
        escalated=escalate,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Stub metrics endpoint."""
    return get_metrics_snapshot()
