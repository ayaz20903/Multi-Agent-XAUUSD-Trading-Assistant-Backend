FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY data/ data/

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["sh", "-c", ".venv/bin/python -m gunicorn gold_ai.api.app:app -k uvicorn.workers.UvicornWorker -w 1 --bind 0.0.0.0:${PORT:-8000} --timeout 120"]