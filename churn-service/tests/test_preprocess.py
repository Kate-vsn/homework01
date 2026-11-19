import numpy as np
import pandas as pd

from churn.preprocess import CaliforniaHousingPreprocessor

COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]


def test_california_preprocessor(tmp_path):
    """Датасет в CI не лежит (он в DVC), поэтому проверяем на маленьком образце из случайных чисел."""
    rng = np.random.default_rng(0)
    path = tmp_path / "housing.csv"
    pd.DataFrame(rng.uniform(1, 10, size=(50, len(COLUMNS))), columns=COLUMNS).to_csv(
        path, index=False
    )

    preprocessor = CaliforniaHousingPreprocessor(
        data_path=path, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, _y_test, _scaler, feature_names = (
        preprocessor.load_and_preprocess_data()
    )

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(feature_names) == 8
    assert y_train.shape[0] == X_train.shape[0]
