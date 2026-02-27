from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal

app = FastAPI()

class Message(BaseModel):
    speaker: Literal["customer", "agent"]
    text: str

class Transcript(BaseModel):
    transcript_id: str
    channel: Literal["call", "chat", "email"]
    messages: List[Message]

@app.post("/analyze")
def analyze(transcript: Transcript):
    text = " ".join([m.text.lower() for m in transcript.messages])

    intent = "other"
    if "refund" in text:
        intent = "Refund"
    elif "bill" in text:
        intent = "Billing Issue"
    elif "cancel" in text:
        intent = "Cancellation"

    return {
        "transcript_id": transcript.transcript_id,
        "intent": intent,
        "message_count": len(transcript.messages),
    }