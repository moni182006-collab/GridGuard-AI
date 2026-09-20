from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "sgcc_raw.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "clean_data.csv"


# ============================================================
# DATASET COLUMN DETECTION
# ============================================================

def detect_id_column(df: pd.DataFrame) -> str:
    """Detect the household/customer identifier column dynamically."""

    preferred_names = [
        "CONS_NO",
        "customer_id",
        "customer",
        "consumer_id",
        "id",
    ]

    for column in preferred_names:
        if column in df.columns:
            return column

    # Fall back to a column that is unique for every row.
    unique_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) == len(df)
    ]

    if unique_columns:
        return unique_columns[0]

    raise ValueError(
        "Could not automatically identify a household/customer ID column."
    )


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect the binary target column dynamically."""

    preferred_names = [
        "FLAG",
        "flag",
        "target",
        "label",
        "class",
    ]

    for column in preferred_names:
        if column in df.columns:
            return column

    # Look for a binary column.
    binary_columns = []

    for column in df.columns:
        values = df[column].dropna().unique()

        if len(values) == 2:
            binary_columns.append(column)

    if len(binary_columns) == 1:
        return binary_columns[0]

    raise ValueError(
        "Could not automatically identify a unique binary target column."
    )


def detect_consumption_columns(
    df: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> list[str]:
    """
    Detect numeric consumption/time-series columns dynamically.

    Excludes:
      - household ID
      - target
      - other obvious metadata columns

    Numeric columns are selected because the consumption measurements
    are expected to be numerical.
    """

    excluded = {
        id_column,
        target_column,
    }

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    consumption_columns = [
        column
        for column in numeric_columns
        if column not in excluded
    ]

    if not consumption_columns:
        raise ValueError(
            "No numeric consumption columns were detected."
        )

    return consumption_columns


# ============================================================
# FEATURE CALCULATION
# ============================================================

def build_aggregate_features(
    consumption_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate household-level consumption statistics.

    All statistics are derived from the detected consumption
    columns. No dataset-specific number of days is assumed.
    """

    numeric = consumption_data.apply(
        pd.to_numeric,
        errors="coerce",
    )

    total_measurements = numeric.shape[1]

    features = pd.DataFrame(
        {
            "avg_consumption": numeric.mean(axis=1),
            "median_consumption": numeric.median(axis=1),
            "std_consumption": numeric.std(axis=1),
            "min_consumption": numeric.min(axis=1),
            "max_consumption": numeric.max(axis=1),
            "zero_days": numeric.eq(0).sum(axis=1),
            "missing_days": numeric.isna().sum(axis=1),
        },
        index=numeric.index,
    )

    # Derive ratios from the actual number of measurements.
    features["zero_ratio"] = (
        features["zero_days"] / total_measurements
    )

    features["missing_ratio"] = (
        features["missing_days"] / total_measurements
    )

    return features


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("GRIDGUARD AI — PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(RAW_PATH)

    print(f"Rows detected: {len(df):,}")
    print(f"Columns detected: {len(df.columns):,}")

    # --------------------------------------------------------
    # Detect important columns
    # --------------------------------------------------------

    id_column = detect_id_column(df)
    target_column = detect_target_column(df)

    consumption_columns = detect_consumption_columns(
        df,
        id_column,
        target_column,
    )

    print(f"ID column detected: {id_column}")
    print(f"Target column detected: {target_column}")
    print(
        f"Consumption columns detected: "
        f"{len(consumption_columns):,}"
    )

    # --------------------------------------------------------
    # Convert target
    # --------------------------------------------------------

    target_values = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    if target_values.isna().any():
        raise ValueError(
            "Target column contains values that cannot be "
            "converted to numeric."
        )

    unique_targets = sorted(
        target_values.unique().tolist()
    )

    if len(unique_targets) != 2:
        raise ValueError(
            "Expected a binary target, but detected "
            f"{len(unique_targets)} unique values: "
            f"{unique_targets}"
        )

    # --------------------------------------------------------
    # Extract consumption data
    # --------------------------------------------------------

    consumption_data = df[consumption_columns].copy()

    consumption_data = consumption_data.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Build features in a separate DataFrame
    # --------------------------------------------------------

    print("\nCalculating household statistics...")

    aggregate_features = build_aggregate_features(
        consumption_data
    )

    # --------------------------------------------------------
    # Handle invalid values
    # --------------------------------------------------------

    aggregate_features = aggregate_features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Fill feature missing values using each feature's
    # dataset-derived median.
    feature_medians = aggregate_features.median(
        numeric_only=True
    )

    aggregate_features = aggregate_features.fillna(
        feature_medians
    )

    # --------------------------------------------------------
    # Construct final dataset ONCE
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        {
            id_column: df[id_column].values,
            target_column: target_values.values,
        },
        index=df.index,
    )

    cleaned = pd.concat(
        [
            metadata,
            aggregate_features,
        ],
        axis=1,
    )

    # Final safety check.
    cleaned = cleaned.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    cleaned = cleaned.fillna(
        cleaned.median(numeric_only=True)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if cleaned[id_column].duplicated().any():
        raise ValueError(
            "Duplicate household IDs detected."
        )

    if cleaned[target_column].isna().any():
        raise ValueError(
            "Missing target values remain after preprocessing."
        )

    numeric_missing = (
        cleaned.select_dtypes(include=[np.number])
        .isna()
        .sum()
        .sum()
    )

    if numeric_missing > 0:
        raise ValueError(
            f"{numeric_missing} numeric missing values remain."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cleaned.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nPreprocessing completed.")
    print(f"Final rows: {len(cleaned):,}")
    print(f"Final columns: {len(cleaned.columns):,}")
    print(
        f"Feature columns: "
        f"{len(cleaned.columns) - 2:,}"
    )
    print(
        f"Missing values: "
        f"{int(cleaned.isna().sum().sum()):,}"
    )

    print("\nTarget distribution:")

    print(
        cleaned[target_column]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()