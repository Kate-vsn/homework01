FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 GIT_PYTHON_REFRESH=quiet

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev

COPY .dvc/config .dvc/config
COPY src/churn/data/*.dvc src/churn/data/
RUN uv run --no-sync dvc config --local core.no_scm true

EXPOSE 8000
CMD [".venv/bin/uvicorn", "churn.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
