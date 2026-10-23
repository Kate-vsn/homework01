import os
import time
import uuid
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, ConfigDict
import psycopg2
from psycopg2.extras import Json

ml_bundle = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.abspath(os.path.join(base_dir, "..", "artifacts"))

    try:
        ml_bundle["model"] = joblib.load(
            os.path.join(artifacts_dir, "housing_model.pkl")
        )
        ml_bundle["scaler"] = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
        ml_bundle["features"] = joblib.load(
            os.path.join(artifacts_dir, "feature_names.pkl")
        )
        ml_bundle["version"] = "1.0.0"
    except Exception as e:
        print(f"Ошибка загрузки артефактов модели: {e}")

    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    request_id VARCHAR(64),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version VARCHAR(32),
                    features JSONB,
                    prediction FLOAT,
                    latency_ms FLOAT,
                    status_code INT
                )
            """)
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Не удалось подключиться к базе данных: {e}")

    yield
    ml_bundle.clear()


app = FastAPI(title="California Housing ML Service", lifespan=lifespan)


class HousingFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    MedInc: float = Field(..., description="Медианный доход")
    HouseAge: float = Field(..., description="Возраст домов")
    AveRooms: float = Field(..., description="Среднее кол-во комнат")
    AveBedrms: float = Field(..., description="Среднее кол-во спален")
    Population: float = Field(..., description="Население")
    AveOccup: float = Field(..., description="Среднее число жильцов")
    Latitude: float = Field(..., description="Широта")
    Longitude: float = Field(..., description="Долгота")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if "model" in ml_bundle and "scaler" in ml_bundle and "features" in ml_bundle:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model not loaded")


@app.post("/v1/predict")
def predict(payload: HousingFeatures, request: Request, response: Response):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    version = ml_bundle.get("version", "1.0.0")

    features_dict = payload.model_dump()

    try:
        input_df = pd.DataFrame([features_dict])[ml_bundle["features"]]
        input_scaled = ml_bundle["scaler"].transform(input_df)
        pred_value = ml_bundle["model"].predict(input_scaled)[0]
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        log_to_db(request_id, version, features_dict, None, latency_ms, 500)
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start_time) * 1000
    status_code = 200

    log_to_db(
        request_id,
        version,
        features_dict,
        float(pred_value),
        latency_ms,
        status_code,
    )

    return {
        "prediction": float(pred_value),
        "version": version,
        "request_id": request_id,
        "latency_ms": round(latency_ms, 2),
    }


def log_to_db(request_id, version, features, prediction, latency_ms, status_code):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO prediction_logs (request_id, version, features, prediction, latency_ms, status_code)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (request_id, version, Json(features), prediction, latency_ms, status_code),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Ошибка логирования в БД: {e}")
