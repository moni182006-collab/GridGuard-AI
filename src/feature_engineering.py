from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

DATA_CONFIG = CONFIG["data"]

PROCESSED_DIR = BASE_DIR / DATA_CONFIG["processed_path"]

INPUT_PATH = PROCESSED_DIR / "clean_data.csv"
OUTPUT_PATH = PROCESSED_DIR / "features.csv"

ID_COLUMN = DATA_CONFIG["id_column"]
TARGET_COLUMN = DATA_CONFIG["target_column"]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("GRIDGUARD AI — FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Input columns: {len(df.columns):,}"
)


# ============================================================
# REQUIRED BASE FEATURES
# ============================================================

required_features = [
    "avg_consumption",
    "median_consumption",
    "std_consumption",
    "min_consumption",
    "max_consumption",
    "zero_days",
    "missing_days",
    "zero_ratio",
    "missing_ratio",
]

missing_features = [
    feature
    for feature in required_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(numerator, denominator):

    denominator = denominator.replace(
        0,
        np.nan
    )

    result = numerator / denominator

    return result.replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["consumption_range"] = (
    df["max_consumption"]
    - df["min_consumption"]
)

df["peak_to_average"] = safe_divide(
    df["max_consumption"],
    df["avg_consumption"]
)

df["coefficient_variation"] = safe_divide(
    df["std_consumption"],
    df["avg_consumption"]
)

df["median_to_average"] = safe_divide(
    df["median_consumption"],
    df["avg_consumption"]
)

df["min_to_average"] = safe_divide(
    df["min_consumption"],
    df["avg_consumption"]
)

df["max_to_median"] = safe_divide(
    df["max_consumption"],
    df["median_consumption"]
)

df["zero_percentage"] = (
    df["zero_ratio"] * 100
)

df["missing_percentage"] = (
    df["missing_ratio"] * 100
)

# Higher stability means lower relative variation.
df["stability_score"] = (
    1 /
    (
        1 +
        df["coefficient_variation"].abs()
    )
)

df["consumption_intensity"] = (
    df["avg_consumption"]
    * (
        1 -
        df["missing_ratio"]
    )
)


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    df[column] = df[column].replace(
        [np.inf, -np.inf],
        np.nan
    )

    if column not in [TARGET_COLUMN]:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nFeature engineering completed.")

print(
    f"Final rows: {len(df):,}"
)

print(
    f"Final columns: {len(df.columns):,}"
)

print(
    f"Missing values: "
    f"{df.isna().sum().sum():,}"
)

print(
    f"\nSaved:\n{OUTPUT_PATH}"
)