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

```bash
kind create cluster --name churn-cluster
docker build -t churn-service:latest .
kind load docker-image churn-service:latest --name churn-cluster
```
И далее в новом терминале:
```bash
kubectl apply -f k8s/
kubectl rollout status deployment/churn-api-deployment
kubectl port-forward svc/churn-api-service 8000:80
```
И для проверки в ещё одном терминале можно задать:

```bash
curl -X GET http://localhost:8000/health

```
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 3.5,
    "HouseAge": 25.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1200.0,
    "AveOccup": 3.0,
    "Latitude": 35.6,
    "Longitude": -119.5
  }'
```