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
    bundle_path = os.path.join(artifacts_dir, "model_bundle.joblib")

    try:
        loaded_bundle = joblib.load(bundle_path)
        ml_bundle["pipeline"] = loaded_bundle["pipeline"]

        passport = loaded_bundle.get("passport", {})
        ml_bundle["version"] = passport.get("version", "1.0.0")
        ml_bundle["features"] = passport.get("feature_names", [])
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

    MedInc: float = Field(..., gt=0.0, le=20.0, description="Медианный доход")
    HouseAge: float = Field(..., ge=1.0, le=100.0, description="Возраст домов")
    AveRooms: float = Field(..., gt=0.0, le=200.0, description="Среднее кол-во комнат")
    AveBedrms: float = Field(..., gt=0.0, le=50.0, description="Среднее число спален")
    Population: float = Field(..., ge=1.0, le=100000.0, description="Население")
    AveOccup: float = Field(..., gt=0.0, le=2000.0, description="Среднее число жильцов")
    Latitude: float = Field(
        ..., ge=32.0, le=43.0, description="Широта (география Калифорнии)"
    )
    Longitude: float = Field(
        ..., ge=-125.0, le=-113.0, description="Долгота (география Калифорнии)"
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if "pipeline" in ml_bundle and "version" in ml_bundle:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model not loaded")


@app.post("/v1/predict")
def predict(payload: HousingFeatures, request: Request, response: Response):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    version = ml_bundle.get("version", "1.0.0")

    features_dict = payload.model_dump()

    try:
        if ml_bundle.get("features"):
            input_df = pd.DataFrame([features_dict])[ml_bundle["features"]]
        else:
            input_df = pd.DataFrame([features_dict])

        pred_value = ml_bundle["pipeline"].predict(input_df)[0]
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
