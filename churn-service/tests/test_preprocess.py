from churn.preprocess import CaliforniaHousingPreprocessor


def test_california_preprocessor():
    preprocessor = CaliforniaHousingPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, _y_test, _scaler, feature_names = (
        preprocessor.load_and_preprocess_data()
    )

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(feature_names) == 8
    assert y_train.shape[0] == X_train.shape[0]
