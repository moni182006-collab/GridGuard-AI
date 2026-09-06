import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# GRIDGUARD AI
# PHASE 3 — DATA PREPROCESSING
# ============================================================

RAW_PATH = Path("../data/raw/sgcc_raw.csv")
OUTPUT_PATH = Path("../data/processed/clean_data.csv")


ID_COLUMN = "CONS_NO"
TARGET_COLUMN = "FLAG"


def load_data():
    """Load the raw SGCC dataset."""

    print("Loading dataset...")

    df = pd.read_csv(RAW_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


def identify_consumption_columns(df):
    """Identify daily consumption columns."""

    consumption_columns = [
        col for col in df.columns
        if col not in [ID_COLUMN, TARGET_COLUMN]
    ]

    print(
        f"Consumption columns found: "
        f"{len(consumption_columns)}"
    )

    return consumption_columns


def convert_consumption_to_numeric(
    df,
    consumption_columns
):
    """Convert consumption values to numeric."""

    print("Converting consumption values...")

    df[consumption_columns] = df[
        consumption_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return df


def create_features(
    df,
    consumption_columns
):
    """Create household-level statistical features."""

    print("Creating features...")

    consumption = df[consumption_columns]

    df["avg_consumption"] = consumption.mean(axis=1)

    df["median_consumption"] = consumption.median(axis=1)

    df["std_consumption"] = consumption.std(axis=1)

    df["min_consumption"] = consumption.min(axis=1)

    df["max_consumption"] = consumption.max(axis=1)

    df["zero_days"] = (
        consumption == 0
    ).sum(axis=1)

    df["missing_days"] = (
        consumption.isna()
    ).sum(axis=1)

    df["zero_ratio"] = (
        df["zero_days"] /
        len(consumption_columns)
    )

    df["missing_ratio"] = (
        df["missing_days"] /
        len(consumption_columns)
    )

    return df


def handle_missing_values(df):
    """
    Handle missing values in engineered features.

    We do NOT replace genuine zero consumption.
    """

    print("Handling missing feature values...")

    feature_columns = [
        "avg_consumption",
        "median_consumption",
        "std_consumption",
        "min_consumption",
        "max_consumption",
    ]

    for column in feature_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )

    return df


def select_final_features(df):
    """Select compact ML-ready features."""

    feature_columns = [
        ID_COLUMN,
        TARGET_COLUMN,

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

    return df[feature_columns].copy()


def save_data(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Processed dataset saved to: "
        f"{OUTPUT_PATH}"
    )


def main():

    print("=" * 60)
    print("GRIDGUARD AI — PHASE 3")
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Load
    df = load_data()

    # Identify consumption columns
    consumption_columns = (
        identify_consumption_columns(df)
    )

    # Convert values
    df = convert_consumption_to_numeric(
        df,
        consumption_columns
    )

    # Create features
    df = create_features(
        df,
        consumption_columns
    )

    # Handle missing feature values
    df = handle_missing_values(df)

    # Select final features
    clean_df = select_final_features(df)

    # Save
    save_data(clean_df)

    # Final information
    print("\nFinal dataset shape:")
    print(clean_df.shape)

    print("\nFinal columns:")
    print(clean_df.columns.tolist())

    print("\nTarget distribution:")
    print(clean_df[TARGET_COLUMN].value_counts())

    print("\nMissing values:")
    print(clean_df.isnull().sum())

    print("\n" + "=" * 60)
    print("PHASE 3 PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()