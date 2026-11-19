import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from mlflow.client import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churn.preprocess import CaliforniaHousingPreprocessor  # noqa: E402

REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "california_housing_rf")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "california_housing_v2")
MIN_GAIN = float(os.getenv("GATE_MIN_GAIN", "0.01"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "100"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "20"))
DATA_PATH = Path(
    os.getenv("DATA_PATH", Path(__file__).resolve().parent / "data" / "housing.csv")
)


def champion_r2(client: MlflowClient) -> tuple[str | None, float | None]:
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    except MlflowException:
        return None, None
    return mv.version, client.get_run(mv.run_id).data.metrics.get("r2")


def train_pipeline() -> dict:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.localhost"))
    mlflow.set_experiment(EXPERIMENT)

    data_md5 = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()

    preprocessor = CaliforniaHousingPreprocessor(
        data_path=DATA_PATH, test_size=0.2, random_state=42
    )
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = (
        preprocessor.load_and_preprocess_data()
    )

    client = MlflowClient()
    with mlflow.start_run() as run:
        mlflow.log_param("data_md5", data_md5)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        r2 = float(r2_score(y_test, predictions))
        mse = float(mean_squared_error(y_test, predictions))

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mse", mse)

        pipeline = Pipeline([("scaler", scaler), ("model", model)])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(feature_names, model.feature_importances_)
        ax.set_title("Важность признаков (Feature Importances)")
        ax.set_xlabel("Вес признака")
        mlflow.log_figure(fig, "feature_importances.png")
        plt.close(fig)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        model_version = model_info.registered_model_version

    old_version, old_r2 = champion_r2(client)
    promoted = old_r2 is None or r2 >= old_r2 + MIN_GAIN
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "challenger", model_version
    )
    if promoted:
        print(
            f"R2: {r2:.4f}, champion до этого {old_r2}: новая версия становится champion"
        )
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "champion", model_version
        )
    else:
        print(
            f"R2: {r2:.4f} < {old_r2:.4f} + {MIN_GAIN}: гейт не пустил, версия только challenger"
        )

    result = {
        "run_id": run.info.run_id,
        "version": str(model_version),
        "r2": round(r2, 4),
        "champion_before": old_version,
        "champion_r2_before": old_r2,
        "promoted": promoted,
        "data_md5": data_md5,
    }
    print(json.dumps(result, ensure_ascii=False))
    xcom = Path("/airflow/xcom")
    if xcom.is_dir():
        (xcom / "return.json").write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    train_pipeline()
