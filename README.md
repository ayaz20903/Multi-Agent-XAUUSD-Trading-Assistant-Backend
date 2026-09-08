# Gold AI

XAUUSD multi-agent trading assistant with LangGraph and FastAPI.

## Install

```bash
uv sync
cp .env.example .env  # add API keys
```

## Start API

```bash
uv run uvicorn gold_ai.api.app:app --reload
```

## Health

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status": "ok"}
```

## Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current Strategy 2 signal?"}'
```

Response:

```json
{
  "response": "Strategy 2 signal: WAIT. Price is currently outside the defined trading zones...",
  "agents": ["technical"]
}
```

## Tests

```bash
uv run pytest
```
