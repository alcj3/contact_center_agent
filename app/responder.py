from __future__ import annotations


class ResponseEngine:
    """Stub response engine for template-based replies."""

    def generate(self, intent: str, user_text: str) -> str:
        # TODO: Add deterministic response templates per intent.
        _ = (intent, user_text)
        return "Thanks for contacting support."
