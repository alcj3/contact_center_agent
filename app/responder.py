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
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.client = client

    def generate(self, intent: str, user_text: str, history: list[str]) -> str:
        system_prompt = _SYSTEM_PROMPTS.get(intent, _DEFAULT_SYSTEM_PROMPT)

        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": "user", "content": msg} for msg in history]
        messages.append({"role": "user", "content": user_text})

        try:
            response = self.client.chat.completions.create(
                model="llama3.2",
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception:
            return _FALLBACK_RESPONSE
