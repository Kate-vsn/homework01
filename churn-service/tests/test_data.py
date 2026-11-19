import pandas as pd
import pandera.pandas as pa
import pytest

from churn.data import EXPORT, ROWS, clean


@pytest.fixture()
def raw():
    return pd.DataFrame(
        {
            "MedInc": [8.3252, 3.5348, 5.6431, 2.0, 4.1, None],
            "HouseAge": [41.0, 29.0, 52.0, 10.0, 25.0, 30.0],
            "AveRooms": [6.98, 5.23, 5.8, 4.5, 6.0, 5.0],
            "AveBedrms": [1.02, 1.05, 1.07, 1.1, 1.0, 1.0],
            "Population": [322.0, 1166.0, 558.0, 900.0, 1500.0, 800.0],
            "AveOccup": [2.55, 2.82, 2.55, 3.0, 2.7, 3.1],
            "Latitude": [37.88, 34.26, 39.5, 33.9, 36.0, 34.0],
            "Longitude": [-122.23, -118.49, -121.9, -117.9, -120.0, -118.0],
            "MedHouseVal": [4.526, 1.797, 3.422, 1.2, 2.0, 1.5],
        }
    )


def test_good_rows_pass(raw):
    df = clean(raw)
    assert len(df) == 5
    assert set(df["Region"]) == {"south", "central", "north"}
    ROWS.validate(df)


def test_spoiled_value_is_caught(raw):
    bad = clean(raw).assign(MedInc=lambda d: d["MedInc"] * 10)
    with pytest.raises(pa.errors.SchemaErrors) as e:
        ROWS.validate(bad, lazy=True)
    assert set(e.value.failure_cases["column"]) == {"MedInc"}


def test_shifted_category_is_caught(raw):
    bad = clean(raw).assign(Region="west")
    with pytest.raises(pa.errors.SchemaErrors) as e:
        ROWS.validate(bad, lazy=True)
    assert set(e.value.failure_cases["column"]) == {"Region"}


def test_small_export_is_rejected(raw):
    with pytest.raises(pa.errors.SchemaErrors) as e:
        EXPORT.validate(clean(raw), lazy=True)
    assert "меньше 15 тысяч строк: выгрузка неполная" in set(
        e.value.failure_cases["check"]
    )
