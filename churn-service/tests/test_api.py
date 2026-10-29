import os

import psycopg2
import pytest
from fastapi.testclient import TestClient

from churn.service.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "MedInc": 3.5,
    "HouseAge": 25.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1200.0,
    "AveOccup": 3.0,
    "Latitude": 35.6,
    "Longitude": -119.5,
}


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 500
    assert response.json()["status"] == "ok"


def test_ready_check(client):
    response = client.get("/ready")
    assert response.status_code == 200


def test_extra_field_is_422(client):
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["unknown_feature"] = 999.0
    response = client.post("/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_missing_field_is_422(client):
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["MedInc"]
    response = client.post("/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_smoke_prediction(client):
    response = client.post("/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] > 0


def test_determinism(client):
    r1 = client.post("/v1/predict", json=VALID_PAYLOAD)
    r2 = client.post("/v1/predict", json=VALID_PAYLOAD)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["prediction"] == pytest.approx(r2.json()["prediction"], abs=1e-6)


def get_latest_db_log():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        "SELECT request_id, status_code, prediction, features FROM prediction_logs ORDER BY created_at DESC LIMIT 1;"
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def test_integration_db_logging_success(client):
    db_url = os.getenv("DATABASE_URL")
    response = client.post("/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    if db_url:
        row = get_latest_db_log()
        assert row is not None
        assert row[0] == request_id
        assert row[1] == 200
        assert row[2] is not None


def test_integration_db_logging_422(client):
    db_url = os.getenv("DATABASE_URL")
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["unknown_feature"] = 999.0
    response = client.post("/v1/predict", json=invalid_payload)
    assert response.status_code == 422

    if db_url:
        row = get_latest_db_log()
        assert row is not None
        assert row[1] == 422
        assert row[2] is None
