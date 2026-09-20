from pathlib import Path
import sys

import pandas as pd
import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_CONFIG = CONFIG["data"]

RAW_PATH = BASE_DIR / DATA_CONFIG["raw_path"]
PROCESSED_DIR = BASE_DIR / DATA_CONFIG["processed_path"]

ID_COLUMN = DATA_CONFIG["id_column"]
TARGET_COLUMN = DATA_CONFIG["target_column"]

OUTPUT_PATH = PROCESSED_DIR / "clean_data.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GRIDGUARD AI — PREPROCESSING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(RAW_PATH)

print(f"Rows detected: {len(df):,}")
print(f"Columns detected: {len(df.columns):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [ID_COLUMN, TARGET_COLUMN]

missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise ValueError(
        f"Required columns missing: {missing_required}"
    )


# ============================================================
# DETECT CONSUMPTION COLUMNS DYNAMICALLY
# ============================================================

consumption_columns = [
    column
    for column in df.columns
    if column not in required_columns
]

if not consumption_columns:
    raise ValueError(
        "No consumption columns were detected."
    )

print(
    f"Consumption columns detected: "
    f"{len(consumption_columns):,}"
)


# ============================================================
# CONVERT CONSUMPTION DATA TO NUMERIC
# ============================================================

consumption_data = df[consumption_columns].apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# ENGINEER BASIC HOUSEHOLD STATISTICS
# ============================================================

df["avg_consumption"] = consumption_data.mean(axis=1)

df["median_consumption"] = consumption_data.median(axis=1)

df["std_consumption"] = consumption_data.std(
    axis=1
)

df["min_consumption"] = consumption_data.min(
    axis=1
)

df["max_consumption"] = consumption_data.max(
    axis=1
)

df["zero_days"] = (
    consumption_data.eq(0)
    .sum(axis=1)
)

df["missing_days"] = (
    consumption_data.isna()
    .sum(axis=1)
)

total_consumption_columns = len(
    consumption_columns
)

df["zero_ratio"] = (
    df["zero_days"] /
    total_consumption_columns
)

df["missing_ratio"] = (
    df["missing_days"] /
    total_consumption_columns
)


# ============================================================
# CLEAN ENGINEERED FEATURES
# ============================================================

engineered_columns = [
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

for column in engineered_columns:

    median_value = df[column].median()

    df[column] = df[column].fillna(
        median_value
    )


# ============================================================
# FINAL DATASET
# ============================================================

final_columns = [
    ID_COLUMN,
    TARGET_COLUMN,
    *engineered_columns,
]

clean_df = df[final_columns].copy()


# ============================================================
# SAVE
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

clean_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nPreprocessing completed.")

print(
    f"Final rows: {len(clean_df):,}"
)

print(
    f"Final columns: {len(clean_df.columns):,}"
)

print(
    f"Missing values: "
    f"{clean_df.isna().sum().sum():,}"
)

print(
    f"\nSaved:\n{OUTPUT_PATH}"
)