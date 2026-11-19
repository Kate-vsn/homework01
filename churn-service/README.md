# California Housing ML service

Сервис предсказывает медианную стоимость дома в районе Калифорнии (в сотнях тысяч долларов, признак
`MedHouseVal` датасета California Housing). Модель — `RandomForestRegressor` (100 деревьев, глубина 20),
обёрнутый в `Pipeline` со `StandardScaler`; признаки: доход, возраст домов, комнаты, спальни, население,
занятость, широта и долгота. На отложенной выборке R² около 0.65. Сервис на FastAPI: `POST /v1/predict`
отдаёт предсказание, версию модели и `request_id`, каждый запрос пишется в Postgres.

Жизненный цикл: данные лежат в DVC (S3 в кластере), обучение и проверка схемы идут в подах из того же
образа, что и сервис, результат — версия в MLflow Model registry с алиасами `champion` и `challenger`.
Airflow раз в сутки прогоняет `validate → train → promoted → rollout`; новая версия становится
`champion`, только если её R² лучше текущей на `MIN_GAIN`, и тогда DAG перезапускает деплоймент: поды
подхватывают нового `champion` из реестра, образ при этом не пересобирается. Метрики сервиса и модели
видны в Prometheus и Grafana.

## Структура проекта

- `src/churn/service/app.py` — FastAPI приложение, метрики Prometheus, логирование в базу.
- `src/churn/train.py` — обучение, регистрация в MLflow, гейт champion/challenger.
- `src/churn/data.py` — схемы pandera на очищенный слой данных, отдельный шаг конвейера.
- `src/churn/data/housing.csv.dvc` — указатель DVC (самих данных в git и в образе нет).
- `dags/ml_retrain.py` — DAG переобучения в Airflow.
- `k8s/` — манифесты сервиса, базы и мониторинга; `platform/` — вся платформа одной командой.
- `tests/` — модульные и интеграционные тесты (`pytest`).

---

## Quickstart

Всё выполняется из корня репозитория. Команды — из Git Bash, macOS или Linux; в PowerShell bash
запускать так: `& "C:\Program Files\Git\bin\bash.exe" platform/up.sh`.

### 0. Что нужно поставить заранее

Docker Desktop, `uv`, `kind`, `kubectl`, `helm`, Git Bash. Образы платформы скачиваются заранее —
`up.sh` кладёт их в узел kind из локального Docker:

```bash
docker pull apache/airflow:3.3.2
docker pull ghcr.io/mlflow/mlflow:v3.16.1
docker pull rustfs/rustfs:1.0.0
```

### 1. Клон и зависимости

```bash
git clone https://github.com/Kate-vsn/homework01
cd homework01
uv sync --all-groups --directory churn-service
```

### 2. Платформа одной командой

```bash
bash platform/up.sh          # Git Bash, macOS, Linux
```

`platform/up.sh` поднимает kind с портом 80 (в узел 30080), Traefik, MLflow, Airflow, S3 (RustFS),
Prometheus и Grafana и создаёт бакет `dvc`. Повторный запуск безопасен. С нуля на чистой машине — 
`kind delete cluster --name kind`, затем снова `bash platform/up.sh`.

Адреса после запуска: `http://mlflow.localhost`, `http://airflow.localhost` (без пароля),
`http://grafana.localhost` (`admin` / `admin`), `http://s3-console.localhost` (`dvcadmin` / `dvc-secret-123`).

**Windows: имена `*.localhost`.** Браузер и `curl` сами отправляют их на 127.0.0.1, а Python спрашивает
DNS, и тот может их не знать — тогда MLflow и DVC падают с `getaddrinfo failed`. Проверка:

```bash
uv run python -c "import socket; print(socket.gethostbyname('s3.localhost'))"
```

Если это не `127.0.0.1`, добавьте в `C:\Windows\System32\drivers\etc\hosts` строку:

```
127.0.0.1 mlflow.localhost airflow.localhost grafana.localhost s3.localhost s3-console.localhost ml.localhost
```

### 3. Данные из DVC

Ключи S3 лежат только в локальном конфиге DVC (в git их нет, в образ они тоже не едут):

```bash
uv run dvc remote modify --local laptop access_key_id dvcadmin
uv run dvc remote modify --local laptop secret_access_key dvc-secret-123
uv run dvc pull
```

### 4. Первый champion

```bash
uv run python -m churn.data      # проверка схемы pandera, прогон validate в MLflow
uv run python -m churn.train     # обучение: R², data_md5, регистрация версии и алиас champion
uv run dvc push                  # данные в S3 кластера (сначала dvc push, потом git push)
```

`churn.train` пишет прогон в MLflow и создаёт версию в реестре `california_housing_rf`: первая версия
сразу получает алиас `champion`.

### 5. Регистрация runner и первый деплой по кнопке

В настройках репозитория добавьте секрет `DB_PASSWORD` (например, `password`) и запустите runner
с меткой `kind`:

```bash
mkdir -p actions-runner && cd actions-runner
curl -o actions-runner.tar.gz -L https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz
tar xzf actions-runner.tar.gz && ./config.sh --url https://github.com/Kate-vsn/homework01 --token <TOKEN> --labels kind
./run.sh
```

Токен берётся в Settings → Actions → Runners → New self-hosted runner. Деплой запускается кнопкой:
Actions → `ci` → Run workflow (или пушем в `main`). Job `deploy` собирает образ, заливает его в kind,
создаёт секрет `ml-secrets`, деплоит базу и сервис, применяет ServiceMonitor, дашборд и ConfigMap с DAG,
ставит переменную `ml_image` в Airflow и проходит smoke через Ingress.

### 6. Ответ сервиса через Ingress

```bash
curl -s http://ml.localhost/health
curl -s -X POST http://ml.localhost/v1/predict \
  -H "Content-Type: application/json" -d @churn-service/good.json
```

Ответ:

```json
{"prediction": 2.71, "version": "california_housing_rf-v1", "request_id": "...", "latency_ms": 55.2}
```

## Переобучение руками

DAG можно прогнать из Airflow: `http://airflow.localhost` → DAG `ml_retrain` → Trigger.
Локально без кластера шаги конвейера запускаются так:

```bash
uv run python -m churn.data            # validate: схема pandera, тег valid и failure_cases.csv в MLflow
uv run python -m churn.train           # train: новая версия, алиас challenger, гейт по R²
kubectl rollout restart deploy/ml-api-deployment   # rollout: поды загрузят нового champion
```

Проверить, что версия в подах сменилась:

```bash
curl -s http://ml.localhost/health
curl -s http://ml.localhost/metrics | grep ml_model_info
```

Плохая выгрузка и откат:

```bash
uv run python scripts/break_export.py   # портит значения: файл читается, типы те же, медиана не та
uv run dvc push && git commit -am "bad data" && git push    # validate красный, дальше задачи не идут
git revert --no-edit HEAD && uv run dvc checkout --force && uv run dvc push && git push
```

## Проверки

```bash
uv run pytest -q          # тесты без датасета
uv run ruff check .
bash platform/up.sh       # поднять/обновить платформу
```

## Интерфейсы

| Что | Адрес |
|---|---|
| Сервис (Ingress) | http://ml.localhost |
| MLflow | http://mlflow.localhost |
| Airflow | http://airflow.localhost |
| Grafana (`admin`/`admin`) | http://grafana.localhost |
| S3 консоль (`dvcadmin`/`dvc-secret-123`) | http://s3-console.localhost |
| Prometheus | `kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090` |
