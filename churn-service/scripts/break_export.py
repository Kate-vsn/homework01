"""Испорченная выгрузка для шага validate: источник тихо перешёл с десятков тысяч долларов на тысячи.

  uv run python scripts/break_export.py

Перезаписывает housing.csv, ни одной ошибки при этом не возникает: файл читается, типы те же,
строк столько же, диапазоны строк тоже проходят. Ловит только проверка распределения:
медиана MedInc 0.35 вместо 3.53. Вернуть как было:

  git checkout -- src/churn/data/housing.csv.dvc && uv run dvc checkout --force
"""
from pathlib import Path

import pandas as pd

PATH = Path("src/churn/data/housing.csv")
df = pd.read_csv(PATH)
df["MedInc"] = df["MedInc"] / 10
df.to_csv(PATH, index=False)
print(f"{PATH}: доход в тысячах, медиана {df['MedInc'].median():.2f}, строк {len(df)}")
