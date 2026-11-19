"""Проверка данных схемой pandera"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

RAW_PATH = Path(
    os.getenv("DATA_PATH", Path(__file__).resolve().parent / "data" / "housing.csv")
)
EXPERIMENT = os.getenv("DATA_EXPERIMENT", "california_housing_data")

REGIONS = ["south", "central", "north"]
LAT_BINS = [-90.0, 35.0, 38.0, 90.0]

ROWS = pa.DataFrameSchema(
    {
        "MedInc": pa.Column(float, pa.Check.in_range(0.0, 15.0001, include_min=False)),
        "HouseAge": pa.Column(float, pa.Check.in_range(1.0, 52.0)),
        "AveRooms": pa.Column(float, pa.Check.in_range(0.0, 200.0, include_min=False)),
        "AveBedrms": pa.Column(float, pa.Check.in_range(0.0, 50.0, include_min=False)),
        "Population": pa.Column(float, pa.Check.in_range(1.0, 40_000.0)),
        "AveOccup": pa.Column(
            float, pa.Check.in_range(0.0, 2_000.0, include_min=False)
        ),
        "Latitude": pa.Column(float, pa.Check.in_range(32.0, 42.5)),
        "Longitude": pa.Column(float, pa.Check.in_range(-124.5, -114.0)),
        "MedHouseVal": pa.Column(
            float, pa.Check.in_range(0.0, 5.0001, include_min=False)
        ),  # сотни тысяч долларов, потолок 5
        "Region": pa.Column(str, pa.Check.isin(REGIONS)),
    },
    strict=True,
)


EXPORT = pa.DataFrameSchema(
    ROWS.columns,
    checks=[
        pa.Check(
            lambda df: len(df) >= 15_000,
            error="меньше 15 тысяч строк: выгрузка неполная",
        ),
        pa.Check(
            lambda df: 1.2 <= df["MedHouseVal"].median() <= 2.5,
            error="медиана цены вне 1.2..2.5: сменились единицы?",
        ),
        pa.Check(
            lambda df: 2.5 <= df["MedInc"].median() <= 5.0,
            error="медиана дохода вне 2.5..5: сменились единицы?",
        ),
        pa.Check(
            lambda df: df["Region"]
            .value_counts(normalize=True)
            .reindex(REGIONS)
            .fillna(0)
            .min()
            >= 0.05,
            error="в одном из регионов меньше 5% строк: сдвиг категории",
        ),
    ],
    strict=True,
)


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna().reset_index(drop=True)
    region = pd.cut(df["Latitude"], bins=LAT_BINS, labels=REGIONS, right=False).astype(
        str
    )
    return df.assign(Region=region)


def main() -> int:
    import mlflow

    raw = load_raw()
    df = clean(raw)
    md5 = file_md5(RAW_PATH)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="validate"):
        mlflow.log_params(
            {"step": "validate", "data": str(RAW_PATH.name), "data_md5": md5}
        )
        mlflow.log_metrics({"raw_rows": len(raw), "clean_rows": len(df)})
        try:
            EXPORT.validate(df, lazy=True)
        except pa.errors.SchemaErrors as e:
            fc = e.failure_cases
            summary = (
                fc.groupby(["column", "check"], dropna=False)
                .size()
                .rename("cases")
                .reset_index()
            )
            with tempfile.TemporaryDirectory() as tmp:
                fc.head(1000).to_csv(Path(tmp) / "failure_cases.csv", index=False)
                mlflow.log_artifact(str(Path(tmp) / "failure_cases.csv"))
            mlflow.log_metric("failed_checks", len(summary))
            mlflow.set_tag("valid", "false")
            print("данные не прошли проверку, отчёт failure_cases.csv в MLflow:")
            print(summary.to_string(index=False))
            return 1
        mlflow.set_tag("valid", "true")
    result = {
        "valid": True,
        "data_md5": md5,
        "clean_rows": len(df),
        "median_price": round(float(df["MedHouseVal"].median()), 3),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
