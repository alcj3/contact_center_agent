from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_stub() -> None:
    payload = {
        "conversation_id": "conv-1",
        "message": {"role": "customer", "text": "I need help"},
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["conversation_id"] == "conv-1"
    assert "response_text" in body
    assert isinstance(body["escalated"], bool)


def test_metrics_endpoint_returns_valid_response() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["total_conversations"], int)
    assert isinstance(body["escalation_rate"], float)
    assert isinstance(body["intent_distribution"], dict)
