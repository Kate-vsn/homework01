import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import RestException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from churn.preprocess import CaliforniaHousingPreprocessor

REGISTERED_MODEL_NAME = "california_housing_rf"
MIN_GAIN = 0.01


def train_pipeline():
    mlflow.set_tracking_uri("http://mlflow.localhost")

    exp_name = "california_housing_v2"
    try:
        mlflow.create_experiment(exp_name, artifact_location="mlflow-artifacts:/")
    except RestException:
        pass  # Эксперимент уже существует

    mlflow.set_experiment(exp_name)

    DATA_PATH = Path(__file__).resolve().parent / "data" / "housing.csv"
    data_md5 = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()

    preprocessor = CaliforniaHousingPreprocessor(
        data_path=DATA_PATH, test_size=0.2, random_state=42
    )
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = (
        preprocessor.load_and_preprocess_data()
    )

    with mlflow.start_run() as run:
        mlflow.log_param("data_md5", data_md5)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 20)

        model = RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        r2 = float(r2_score(y_test, predictions))
        mse = float(mean_squared_error(y_test, predictions))

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mse", mse)

        pipeline = Pipeline([("scaler", scaler), ("model", model)])

        fig, ax = plt.subplots(figsize=(10, 8))
        importances = model.feature_importances_
        ax.barh(feature_names, importances)
        ax.set_title("Важность признаков (Feature Importances)")
        ax.set_xlabel("Вес признака")
        mlflow.log_figure(fig, "feature_importances.png")
        plt.close(fig)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        client = MlflowClient()
        model_version = model_info.registered_model_version

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "challenger", model_version
        )

        try:
            champion_version = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME, "champion"
            )
            champion_run = client.get_run(champion_version.run_id)
            champion_r2 = champion_run.data.metrics.get("r2", -float("inf"))

            if r2 >= (champion_r2 + MIN_GAIN):
                print(f"R2: {r2:.4f} >= {champion_r2:.4f} + {MIN_GAIN}")
                client.set_registered_model_alias(
                    REGISTERED_MODEL_NAME, "champion", model_version
                )
            else:
                print(f"R2: {r2:.4f} < {champion_r2:.4f} + {MIN_GAIN}")

        except mlflow.exceptions.RestException:
            print("Первый запуск: назначаем текущую модель как champion.")
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, "champion", model_version
            )


if __name__ == "__main__":
    train_pipeline()
