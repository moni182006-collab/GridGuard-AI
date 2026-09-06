# ============================================================
# GRIDGUARD AI
# PHASE 6 — ISOLATION FOREST ANOMALY DETECTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

import joblib


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path("../data/processed/features.csv")

OUTPUT_PATH = Path(
    "../data/processed/anomaly_results.csv"
)

MODEL_PATH = Path(
    "../models/isolation_forest.pkl"
)

OUTPUT_DIR = Path(
    "../data/processed"
)


# ============================================================
# SETTINGS
# ============================================================

ID_COLUMN = "CONS_NO"
TARGET_COLUMN = "FLAG"

RANDOM_STATE = 42

# Expected approximate suspicious proportion:
# 3615 / 42372 ≈ 8.5%
#
# We use 0.085 as the initial contamination estimate.
CONTAMINATION = 0.085


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GRIDGUARD AI — PHASE 6")
print("ISOLATION FOREST ANOMALY DETECTION")
print("=" * 70)

print("\nLoading feature dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# CHECK DATA
# ============================================================

print("\nMissing values:")

print(df.isnull().sum().sum())


print("\nTarget distribution:")

print(df[TARGET_COLUMN].value_counts())


# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

print("\nPreparing anomaly-detection features...")

# IMPORTANT:
# FLAG is NOT used for training.
# CONS_NO is only an identifier.

X = df.drop(
    columns=[
        ID_COLUMN,
        TARGET_COLUMN
    ]
)

y = df[TARGET_COLUMN]


print("\nFeature matrix shape:", X.shape)

print("\nFeatures used by Isolation Forest:")

for column in X.columns:
    print(" -", column)


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ISOLATION FOREST")
print("=" * 70)

isolation_forest = IsolationForest(
    n_estimators=300,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


isolation_forest.fit(X)


print("\nIsolation Forest training complete.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating anomaly predictions...")

raw_predictions = isolation_forest.predict(X)

# Isolation Forest:
#  1  = normal
# -1  = anomaly

df["anomaly_label"] = np.where(
    raw_predictions == -1,
    1,
    0
)


# ============================================================
# GENERATE ANOMALY SCORE
# ============================================================

print("Generating anomaly scores...")

raw_scores = isolation_forest.decision_function(X)

# Lower decision_function = more anomalous.
#
# Convert it so:
# larger anomaly_score = more unusual

df["anomaly_score"] = -raw_scores


# ============================================================
# NORMALIZE ANOMALY SCORE
# ============================================================

score_min = df["anomaly_score"].min()

score_max = df["anomaly_score"].max()

if score_max > score_min:

    df["anomaly_score_normalized"] = (
        (df["anomaly_score"] - score_min)
        /
        (score_max - score_min)
    )

else:

    df["anomaly_score_normalized"] = 0.0


# ============================================================
# ANOMALY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANOMALY SUMMARY")
print("=" * 70)

total_anomalies = (
    df["anomaly_label"] == 1
).sum()

total_normal = (
    df["anomaly_label"] == 0
).sum()

print(
    f"\nNormal according to Isolation Forest: "
    f"{total_normal}"
)

print(
    f"Anomalies according to Isolation Forest: "
    f"{total_anomalies}"
)

print(
    f"Anomaly percentage: "
    f"{total_anomalies / len(df) * 100:.2f}%"
)


# ============================================================
# EVALUATION AGAINST FLAG
# ============================================================
#
# IMPORTANT:
# FLAG is NOT used for training.
# It is used here ONLY to evaluate how well
# anomalies overlap with known suspicious labels.
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION AGAINST KNOWN LABELS")
print("=" * 70)


anomaly_precision = precision_score(
    y,
    df["anomaly_label"],
    zero_division=0
)

anomaly_recall = recall_score(
    y,
    df["anomaly_label"],
    zero_division=0
)

anomaly_f1 = f1_score(
    y,
    df["anomaly_label"],
    zero_division=0
)


try:

    anomaly_auc = roc_auc_score(
        y,
        df["anomaly_score_normalized"]
    )

except ValueError:

    anomaly_auc = np.nan


print(
    f"\nPrecision: {anomaly_precision:.4f}"
)

print(
    f"Recall   : {anomaly_recall:.4f}"
)

print(
    f"F1 Score : {anomaly_f1:.4f}"
)

print(
    f"ROC-AUC  : {anomaly_auc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y,
        df["anomaly_label"],
        target_names=[
            "Normal",
            "Suspicious"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    df["anomaly_label"]
)

print("\nConfusion Matrix:")

print(cm)


fig, ax = plt.subplots(
    figsize=(6, 5)
)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Suspicious"
    ]
).plot(ax=ax)

plt.title(
    "Isolation Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "isolation_forest_confusion_matrix.png",
    dpi=150
)

plt.show()


# ============================================================
# ANOMALY SCORE DISTRIBUTION
# ============================================================

print("\nCreating anomaly score distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df.loc[
        df[TARGET_COLUMN] == 0,
        "anomaly_score_normalized"
    ],
    bins=50,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    df.loc[
        df[TARGET_COLUMN] == 1,
        "anomaly_score_normalized"
    ],
    bins=50,
    alpha=0.6,
    label="Suspicious"
)

plt.xlabel(
    "Normalized Anomaly Score"
)

plt.ylabel(
    "Number of Households"
)

plt.title(
    "Isolation Forest Anomaly Score Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "anomaly_score_distribution.png",
    dpi=150
)

plt.show()


# ============================================================
# TOP ANOMALOUS HOUSEHOLDS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 MOST ANOMALOUS HOUSEHOLDS")
print("=" * 70)

top_anomalies = df.sort_values(
    by="anomaly_score_normalized",
    ascending=False
).head(20)

display_columns = [
    ID_COLUMN,
    TARGET_COLUMN,
    "anomaly_score",
    "anomaly_score_normalized",
    "anomaly_label",
    "avg_consumption",
    "std_consumption",
    "zero_ratio",
    "missing_ratio",
    "peak_to_average",
    "coefficient_variation"
]

print(
    top_anomalies[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# CHECK HOW MANY TOP ANOMALIES ARE SUSPICIOUS
# ============================================================

print("\nChecking top anomaly overlap...")

for top_n in [100, 500, 1000]:

    top_n_df = df.sort_values(
        by="anomaly_score_normalized",
        ascending=False
    ).head(top_n)

    suspicious_count = (
        top_n_df[TARGET_COLUMN] == 1
    ).sum()

    percentage = (
        suspicious_count / top_n
    ) * 100

    print(
        f"Top {top_n}: "
        f"{suspicious_count} suspicious "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SAVE ANOMALY RESULTS
# ============================================================

result_columns = [
    ID_COLUMN,
    TARGET_COLUMN,

    "anomaly_label",
    "anomaly_score",
    "anomaly_score_normalized",

    "avg_consumption",
    "median_consumption",
    "std_consumption",
    "min_consumption",
    "max_consumption",

    "zero_days",
    "missing_days",
    "zero_ratio",
    "missing_ratio",

    "consumption_range",
    "peak_to_average",
    "coefficient_variation",
    "median_to_average",
    "min_to_average",
    "max_to_median",
    "zero_percentage",
    "missing_percentage",
    "stability_score",
    "consumption_intensity"
]


df[result_columns].to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nAnomaly results saved to: "
    f"{OUTPUT_PATH}"
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    isolation_forest,
    MODEL_PATH
)

print(
    f"Isolation Forest model saved to: "
    f"{MODEL_PATH}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Total Anomalies",
        "Anomaly Percentage"
    ],

    "Value": [
        anomaly_precision,
        anomaly_recall,
        anomaly_f1,
        anomaly_auc,
        total_anomalies,
        total_anomalies / len(df) * 100
    ]
})


metrics.to_csv(
    OUTPUT_DIR /
    "isolation_forest_metrics.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 6 ISOLATION FOREST COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    " - data/processed/anomaly_results.csv"
)

print(
    " - data/processed/isolation_forest_metrics.csv"
)

print(
    " - data/processed/isolation_forest_confusion_matrix.png"
)

print(
    " - data/processed/anomaly_score_distribution.png"
)

print(
    " - models/isolation_forest.pkl"
)

print("\nREADY FOR PHASE 7 — GRAPH CONSTRUCTION")

print("=" * 70)