# Multi-Agent XAUUSD Trading Assistant Backend

XAUUSD multi-agent trading assistant with LangGraph and FastAPI.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM |
| `TAVILY_API_KEY` | Yes | Tavily API key for news search |
| `GOOGLE_API_KEY` | No | Google API key (optional) |
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default: `http://localhost:3000`) |

## Local Setup

```bash
uv sync
cp .env.example .env  # add API keys
```

## Start API (development)

```bash
uv run uvicorn gold_ai.api.app:app --reload --port 8000
```

The vector store is automatically created from `data/` on first startup if missing.

## Start API (production)

```bash
uv run gunicorn gold_ai.api.app:app \
  -k uvicorn.workers.UvicornWorker \
  -w 2 \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

## Docker

```bash
docker build -t gold-ai-backend .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e TAVILY_API_KEY=your_key \
  -e CORS_ORIGINS=https://your-frontend.vercel.app \
  gold-ai-backend
```

## Health

```bash
curl http://localhost:8000/health
```

## Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current Strategy 2 signal?"}'
```

## Market Analysis

```bash
curl http://localhost:8000/api/market-analysis
```

## Strategy

```bash
curl http://localhost:8000/api/strategy
```

## Tests

```bash
uv run pytest
```

## Deploy to Render

1. Push to GitHub
2. Create a new **Web Service** on Render
3. Connect your GitHub repository
4. Select **Docker** as the runtime environment
5. Render auto-detects the Dockerfile and builds
6. Add environment variables in Render dashboard:
   - `GROQ_API_KEY` (required)
   - `TAVILY_API_KEY` (required)
   - `CORS_ORIGINS` (optional, e.g. `https://my-frontend.onrender.com`)
7. Set the health check path to `/health`
8. Render provides `$PORT` automatically — the Dockerfile uses it
9. The vector store is created automatically on first startup from `data/`
