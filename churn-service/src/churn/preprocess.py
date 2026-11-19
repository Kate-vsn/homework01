import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CaliforniaHousingPreprocessor:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)

        X = df.drop(columns=["MedHouseVal"])
        y = df["MedHouseVal"]

        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            self.scaler,
            self.feature_names,
        )
