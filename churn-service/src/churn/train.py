from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from churn.preprocess import CaliforniaHousingPreprocessor


def train_pipeline():
    preprocessor = CaliforniaHousingPreprocessor(test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = (
        preprocessor.load_and_preprocess_data()
    )

    model = RandomForestRegressor(
        n_estimators=50, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    bundle = {
        "pipeline": pipeline,
        "passport": {
            "version": "1.0.0",
            "model_type": "RandomForestRegressor",
            "feature_names": feature_names,
            "metrics": {
                "r2": round(float(r2), 4),
                "mse": round(float(mse), 4),
            },
        },
    }

    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = artifacts_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=("zlib", 3))


if __name__ == "__main__":
    train_pipeline()
