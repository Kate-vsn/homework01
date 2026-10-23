Production-ready сервис машинного обучения, построенный с использованием **FastAPI**, **Docker**, **PostgreSQL** и **Kubernetes (Kind)**.

## Структура проекта
- `src/` — FastAPI приложение, логика модели и логирование в базу данных.
- `k8s/` — Манифесты Kubernetes (`deployment.yaml`, `service.yaml`).
- `tests/` — Модульные и интеграционные тесты (`pytest`).
- `Dockerfile` & `compose.yaml` — Конфигурации для контейнеризации.

---

## Быстрый старт (3 основные команды)

### 1. Запуск тестов
Проверка логики кода, контрактов API и инференса модели с помощью `pytest`:

```bash
uv run pytest
```

### 2. Запуск через Docker Compose
Сборка и запуск FastAPI-сервиса в связке с PostgreSQL для персистентного логирования запросов:

```bash
docker compose up -d --build
```

### 3. Деплой в Kubernetes (Kind)
Применение манифестов для развертывания сервиса в локальном кластере Kubernetes-in-Docker:

```bash
kubectl apply -f k8s/
```
