# AI Contact Center Agent (Scaffold)

Minimal FastAPI monolith scaffold for an AI contact center agent.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Server

```bash
uvicorn app.main:app --reload
```

Server endpoints:
- `GET /health`
- `POST /chat` (stub)
- `GET /metrics` (stub)

## Run Tests

```bash
pytest
```

## Run Evaluation Script

```bash
python3 -m app.evaluation
```

## Notes

- ML classifier logic is intentionally not implemented yet.
- Escalation logic and metrics calculations are stubs with TODO markers.
