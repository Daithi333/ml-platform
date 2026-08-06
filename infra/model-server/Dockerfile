FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
    uv sync --frozen --no-dev

COPY src /app/src


FROM python:3.12-slim AS dev

EXPOSE 8001

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=base /app /app
COPY --from=base /usr/local/bin/uv /usr/local/bin/uv
COPY --from=base /usr/local/bin/uvx /usr/local/bin/uvx

ENV PATH="/app/.venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["uvicorn", "src.model_server.app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]


FROM python:3.12-slim AS prod

EXPOSE 8001

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=base /app /app

ENV PATH="/app/.venv/bin:$PATH"

# Single worker — model lives in process memory, scale horizontally via replicas
CMD ["uvicorn", "src.model_server.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
