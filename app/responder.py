from __future__ import annotations

from typing import Any

_SYSTEM_PROMPTS: dict[str, str] = {
    "Billing": (
        "You are a helpful billing support agent. "
        "Assist the customer with billing inquiries, charges, and payment issues."
    ),
    "Technical": (
        "You are a technical support agent. "
        "Help the customer troubleshoot technical issues with their product or service."
    ),
    "Cancellation": (
        "You are a retention support agent. "
        "Address the customer's cancellation request with empathy and explore alternatives."
    ),
}

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful customer support agent. Assist the customer with their inquiry."
)

_FALLBACK_RESPONSE = (
    "I'm sorry, I'm having trouble responding right now. "
    "Please try again or contact support directly."
)


class ResponseEngine:
    def __init__(self, client: Any = None) -> None:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def generate(self, intent: str, user_text: str, history: list[str]) -> str:
        system_prompt = _SYSTEM_PROMPTS.get(intent, _DEFAULT_SYSTEM_PROMPT)

        messages = [{"role": "user", "content": msg} for msg in history]
        messages.append({"role": "user", "content": user_text})

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text
        except Exception:
            return _FALLBACK_RESPONSE
