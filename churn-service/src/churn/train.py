import joblib
from churn.preprocess import CaliforniaHousingPreprocessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from pathlib import Path


def train_pipeline():
    print("1. Инициализация препроцессора и загрузка данных...")
    preprocessor = CaliforniaHousingPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test, scaler, feature_names = (
        preprocessor.load_and_preprocess_data()
    )

    print("2. Обучение модели (RandomForest)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Оценка качества модели
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f"Качество модели:")
    print(f"  - R2 Score: {r2:.4f}")
    print(f"  - MSE: {mse:.4f}")

    print("3. Сохранение артефактов...")
    joblib.dump(model, artifacts_dir / "housing_model.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(feature_names, artifacts_dir / "feature_names.pkl")
    print("Готово! Все артефакты успешно сохранены.")


if __name__ == "__main__":
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_pipeline()
