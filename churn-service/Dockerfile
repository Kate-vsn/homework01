FROM python:3.11-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
RUN uv sync --frozen --no-dev
EXPOSE 8000
CMD [".venv/bin/uvicorn", "churn.service.app:app", "--host", "0.0.0.0", "--port", "8000"]