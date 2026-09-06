import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# GRIDGUARD AI
# PHASE 4 — FEATURE ENGINEERING
# ============================================================

INPUT_PATH = Path("../data/processed/clean_data.csv")
OUTPUT_PATH = Path("../data/processed/features.csv")

ID_COLUMN = "CONS_NO"
TARGET_COLUMN = "FLAG"


def load_data():
    print("Loading processed dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


def create_advanced_features(df):

    print("Creating advanced features...")

    # --------------------------------------------------------
    # 1. Consumption range
    # --------------------------------------------------------

    df["consumption_range"] = (
        df["max_consumption"]
        - df["min_consumption"]
    )


    # --------------------------------------------------------
    # 2. Peak to average ratio
    # --------------------------------------------------------

    df["peak_to_average"] = (
        df["max_consumption"]
        / (df["avg_consumption"] + 1e-8)
    )


    # --------------------------------------------------------
    # 3. Coefficient of variation
    # --------------------------------------------------------

    df["coefficient_variation"] = (
        df["std_consumption"]
        / (df["avg_consumption"] + 1e-8)
    )


    # --------------------------------------------------------
    # 4. Median to average ratio
    # --------------------------------------------------------

    df["median_to_average"] = (
        df["median_consumption"]
        / (df["avg_consumption"] + 1e-8)
    )


    # --------------------------------------------------------
    # 5. Minimum to average ratio
    # --------------------------------------------------------

    df["min_to_average"] = (
        df["min_consumption"]
        / (df["avg_consumption"] + 1e-8)
    )


    # --------------------------------------------------------
    # 6. Maximum to median ratio
    # --------------------------------------------------------

    df["max_to_median"] = (
        df["max_consumption"]
        / (df["median_consumption"] + 1e-8)
    )


    # --------------------------------------------------------
    # 7. Zero-day percentage
    # --------------------------------------------------------

    df["zero_percentage"] = (
        df["zero_ratio"] * 100
    )


    # --------------------------------------------------------
    # 8. Missing-day percentage
    # --------------------------------------------------------

    df["missing_percentage"] = (
        df["missing_ratio"] * 100
    )


    # --------------------------------------------------------
    # 9. Consumption stability score
    # --------------------------------------------------------

    df["stability_score"] = (
        1 /
        (1 + df["coefficient_variation"])
    )


    # --------------------------------------------------------
    # 10. Consumption intensity
    # --------------------------------------------------------

    df["consumption_intensity"] = (
        df["avg_consumption"]
        * (1 - df["missing_ratio"])
    )


    return df


def clean_features(df):

    print("Cleaning engineered features...")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    # Replace infinite values

    df[numeric_columns] = df[
        numeric_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill any unexpected missing values

    for column in numeric_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    return df


def save_features(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Features saved to: {OUTPUT_PATH}"
    )


def main():

    print("=" * 60)
    print("GRIDGUARD AI — PHASE 4")
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Load

    df = load_data()

    # Create advanced features

    df = create_advanced_features(df)

    # Clean

    df = clean_features(df)

    # Save

    save_features(df)

    # Display information

    print("\nFinal feature shape:")
    print(df.shape)

    print("\nFinal columns:")

    for column in df.columns:
        print(" -", column)

    print("\nMissing values:")

    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("PHASE 4 FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()