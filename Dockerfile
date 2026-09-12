FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY data/ data/

RUN uv sync --frozen --no-dev \
    --no-install-package torch \
    --no-install-package nvidia-cublas \
    --no-install-package nvidia-cuda-cupti \
    --no-install-package nvidia-cuda-nvrtc \
    --no-install-package nvidia-cuda-runtime \
    --no-install-package nvidia-cudnn-cu13 \
    --no-install-package nvidia-cufft \
    --no-install-package nvidia-cufile \
    --no-install-package nvidia-curand \
    --no-install-package nvidia-cusolver \
    --no-install-package nvidia-cusparse \
    --no-install-package nvidia-cusparselt-cu13 \
    --no-install-package nvidia-nccl-cu13 \
    --no-install-package nvidia-nvjitlink \
    --no-install-package nvidia-nvshmem-cu13 \
    --no-install-package nvidia-nvtx \
    --no-install-package triton \
    --no-install-package cuda-bindings \
    --no-install-package cuda-pathfinder \
    --no-install-package cuda-toolkit

RUN uv pip install --python .venv --no-deps torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["sh", "-c", ".venv/bin/python -m gunicorn gold_ai.api.app:app -k uvicorn.workers.UvicornWorker -w 1 --bind 0.0.0.0:${PORT:-8000} --timeout 120"]