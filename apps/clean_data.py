from pathlib import Path

import pandas as pd

BASE_DIR = Path.cwd()
file_path = BASE_DIR / "data" / "raw" / "student_lifestyle_dataset.csv"

df = pd.read_csv(file_path)

print(f"Before: {len(df)} rows, {df.isna().sum().sum()} missing values")

df_clean = df.dropna()

print(f"After: {len(df_clean)} rows, {df_clean.isna().sum().sum()} missing values")

df_clean.to_csv(file_path, index=False)
