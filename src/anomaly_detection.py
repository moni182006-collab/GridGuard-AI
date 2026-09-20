from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# GRIDGUARD AI — PHASE 6
# ISOLATION FOREST ANOMALY DETECTION
# ============================================================

print("=" * 70)
print("GRIDGUARD AI — PHASE 6")
print("ISOLATION FOREST ANOMALY DETECTION")
print("=" * 70)


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found:\n{CONFIG_PATH}"
    )

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:
    CONFIG = yaml.safe_load(file)


# ============================================================
# DATA CONFIGURATION
# ============================================================

DATA_CONFIG = CONFIG["data"]

DATA_DIR = (
    BASE_DIR
    / DATA_CONFIG["processed_path"]
)

INPUT_PATH = (
    DATA_DIR
    / "features.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

ID_COLUMN = DATA_CONFIG["id_column"]

TARGET_COLUMN = DATA_CONFIG["target_column"]


# ============================================================
# ANOMALY CONFIGURATION
# ============================================================

ANOMALY_CONFIG = CONFIG[
    "anomaly_detection"
]

RANDOM_STATE = CONFIG[
    "project"
]["random_state"]


# ============================================================
# VALIDATE INPUT
# ============================================================

print("\nProject root:")
print(BASE_DIR)

print("\nFeature dataset:")
print(INPUT_PATH)

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        "\nfeatures.csv was not found.\n\n"
        "Expected location:\n"
        f"{INPUT_PATH}\n\n"
        "Run Phase 4 first:\n"
        "python src\\feature_engineering.py"
    )


# ============================================================
# LOAD FEATURE DATASET
# ============================================================

print("\nLoading feature dataset...")

df = pd.read_csv(
    INPUT_PATH
)

print(
    f"Rows detected: {len(df):,}"
)

print(
    f"Columns detected: {len(df.columns):,}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    ID_COLUMN,
    TARGET_COLUMN,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Required columns are missing:\n"
        f"{missing_columns}"
    )


# ============================================================
# DYNAMIC FEATURE DETECTION
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in [
        ID_COLUMN,
        TARGET_COLUMN,
    ]
]

if not feature_columns:

    raise ValueError(
        "No feature columns were detected."
    )


print(
    f"Features detected: "
    f"{len(feature_columns):,}"
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[
    feature_columns
].copy()

y = df[
    TARGET_COLUMN
].copy()


# ============================================================
# ENSURE NUMERIC FEATURES
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# HANDLE INVALID VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

for column in X.columns:

    median_value = X[column].median()

    X[column] = X[column].fillna(
        median_value
    )


# ============================================================
# VALIDATE TARGET
# ============================================================

if y.isna().any():

    raise ValueError(
        "Target column contains missing values."
    )

y = pd.to_numeric(
    y,
    errors="raise"
).astype(int)


unique_classes = sorted(
    y.unique()
)

if len(unique_classes) != 2:

    raise ValueError(
        "Isolation Forest evaluation "
        "currently requires a binary target.\n"
        f"Detected classes: {unique_classes}"
    )


# ============================================================
# DYNAMIC CONTAMINATION
# ============================================================

contamination_config = (
    ANOMALY_CONFIG[
        "contamination"
    ]
)

contamination_method = (
    contamination_config[
        "method"
    ]
)


if contamination_method == (
    "target_prevalence"
):

    positive_class = max(
        unique_classes
    )

    contamination = (
        y == positive_class
    ).mean()

else:

    raise ValueError(
        "Unsupported contamination method:\n"
        f"{contamination_method}"
    )


# Isolation Forest requires contamination
# to be greater than 0 and less than 0.5.
if not (
    0 < contamination < 0.5
):

    raise ValueError(
        "Derived contamination is invalid:\n"
        f"{contamination:.6f}\n\n"
        "Isolation Forest requires a value "
        "between 0 and 0.5."
    )


print(
    "\nDataset-derived contamination:"
)

print(
    f"{contamination:.6f}"
)


# ============================================================
# CREATE ISOLATION FOREST
# ============================================================

print(
    "\nTraining Isolation Forest..."
)

model = IsolationForest(

    n_estimators=ANOMALY_CONFIG[
        "n_estimators"
    ],

    contamination=contamination,

    random_state=RANDOM_STATE,

    n_jobs=ANOMALY_CONFIG[
        "n_jobs"
    ],
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X
)


# ============================================================
# ANOMALY PREDICTIONS
# ============================================================

raw_predictions = model.predict(
    X
)

anomaly_label = np.where(
    raw_predictions == -1,
    1,
    0
)


# ============================================================
# ANOMALY SCORE
# ============================================================

# Isolation Forest's decision_function:
# higher = more normal.
#
# Therefore negate it so:
# higher anomaly_score = more anomalous.

raw_scores = (
    -model.decision_function(X)
)

score_min = raw_scores.min()
score_max = raw_scores.max()

if score_max > score_min:

    anomaly_score = (
        (raw_scores - score_min)
        /
        (
            score_max
            - score_min
        )
    )

else:

    anomaly_score = np.zeros(
        len(raw_scores)
    )


# ============================================================
# RESULT DATAFRAME
# ============================================================

results = df[
    [
        ID_COLUMN,
        TARGET_COLUMN,
    ]
].copy()

results[
    "anomaly_label"
] = anomaly_label

results[
    "anomaly_score"
] = anomaly_score


# ============================================================
# EVALUATION
# ============================================================

print(
    "\nEvaluating anomaly detection..."
)

metrics = {

    "accuracy":
        accuracy_score(
            y,
            anomaly_label
        ),

    "precision":
        precision_score(
            y,
            anomaly_label,
            zero_division=0
        ),

    "recall":
        recall_score(
            y,
            anomaly_label,
            zero_division=0
        ),

    "f1":
        f1_score(
            y,
            anomaly_label,
            zero_division=0
        ),

    "roc_auc":
        roc_auc_score(
            y,
            anomaly_score
        ),
}


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "-" * 70)
print("ISOLATION FOREST RESULTS")
print("-" * 70)

for metric_name, metric_value in metrics.items():

    print(
        f"{metric_name:<12}: "
        f"{metric_value:.4f}"
    )


print("\nClassification report:")

print(
    classification_report(
        y,
        anomaly_label,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(
    y,
    anomaly_label
)

print(
    "Confusion matrix:"
)

print(
    confusion
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE ANOMALY RESULTS
# ============================================================

results_path = (
    DATA_DIR
    / "anomaly_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    DATA_DIR
    / "isolation_forest_metrics.csv"
)

pd.DataFrame(
    [metrics]
).to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_path = (
    DATA_DIR
    / "isolation_forest_confusion_matrix.csv"
)

pd.DataFrame(
    confusion
).to_csv(
    confusion_path,
    index=False
)


# ============================================================
# TOP ANOMALIES
# ============================================================

top_count = min(
    100,
    len(results)
)

top_anomalies = (
    results
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    .head(top_count)
)

top_anomalies_path = (
    DATA_DIR
    / "top_anomalies.csv"
)

top_anomalies.to_csv(
    top_anomalies_path,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "isolation_forest.pkl"
)

joblib.dump(
    model,
    model_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"Samples: {len(df):,}"
)

print(
    f"Features: {len(feature_columns):,}"
)

print(
    f"Contamination: "
    f"{contamination:.6f}"
)

print(
    f"Anomalies detected: "
    f"{anomaly_label.sum():,}"
)

print(
    "\nGenerated files:"
)

print(
    results_path
)

print(
    metrics_path
)

print(
    confusion_path
)

print(
    top_anomalies_path
)

print(
    model_path
)