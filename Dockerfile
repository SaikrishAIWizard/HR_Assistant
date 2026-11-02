FROM ghcr.io/astral-sh/uv:0.12.21 AS uv

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    CORPUS_DIR=/app/data/corpus \
    MANIFEST_PATH=/app/data/manifest.json \
    DB_PATH=/app/data/veltrix.db \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends -y tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 8000

CMD ["veltrix-api"]
