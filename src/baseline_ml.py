from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

DATA_CONFIG = CONFIG["data"]
ML_CONFIG = CONFIG["baseline_ml"]

RANDOM_STATE = CONFIG["project"]["random_state"]

DATA_DIR = BASE_DIR / DATA_CONFIG["processed_path"]
MODEL_DIR = BASE_DIR / "models"

INPUT_PATH = DATA_DIR / "features.csv"

ID_COLUMN = DATA_CONFIG["id_column"]
TARGET_COLUMN = DATA_CONFIG["target_column"]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GRIDGUARD AI — BASELINE ML")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

feature_columns = [
    column
    for column in df.columns
    if column not in [
        ID_COLUMN,
        TARGET_COLUMN
    ]
]

X = df[feature_columns].copy()
y = df[TARGET_COLUMN].copy()


# ============================================================
# DYNAMIC DATA INFORMATION
# ============================================================

class_counts = y.value_counts()

print(
    f"Samples: {len(df):,}"
)

print(
    f"Features: {len(feature_columns):,}"
)

print("\nClass distribution:")
print(class_counts)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

test_size = ML_CONFIG["test_size"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

lr_config = ML_CONFIG[
    "logistic_regression"
]

logistic_model = LogisticRegression(
    max_iter=lr_config["max_iter"],
    class_weight=lr_config["class_weight"],
    random_state=RANDOM_STATE,
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

lr_probabilities = (
    logistic_model
    .predict_proba(X_test_scaled)[:, 1]
)

lr_predictions = (
    lr_probabilities >= 0.5
).astype(int)


# ============================================================
# RANDOM FOREST
# ============================================================

rf_config = ML_CONFIG[
    "random_forest"
]

random_forest = RandomForestClassifier(
    n_estimators=rf_config["n_estimators"],
    class_weight=rf_config["class_weight"],
    n_jobs=rf_config["n_jobs"],
    random_state=RANDOM_STATE,
)

random_forest.fit(
    X_train,
    y_train
)

rf_probabilities = (
    random_forest
    .predict_proba(X_test)[:, 1]
)

rf_predictions = (
    rf_probabilities >= 0.5
).astype(int)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    predictions,
    probabilities
):

    metrics = {
        "model": name,
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),
    }

    print(
        f"\n{name}"
    )

    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    return metrics


# ============================================================
# EVALUATE
# ============================================================

lr_metrics = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions,
    lr_probabilities,
)

rf_metrics = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities,
)


# ============================================================
# SAVE MODELS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    logistic_model,
    MODEL_DIR / "logistic_regression.pkl"
)

joblib.dump(
    random_forest,
    MODEL_DIR / "random_forest.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    [
        lr_metrics,
        rf_metrics,
    ]
)

results_df.to_csv(
    DATA_DIR / "baseline_ml_results.csv",
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": random_forest.feature_importances_,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    DATA_DIR / "feature_importance.csv",
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    range(len(class_counts)),
    class_counts.index
)

plt.yticks(
    range(len(class_counts)),
    class_counts.index
)

for row in range(cm.shape[0]):

    for column in range(cm.shape[1]):

        plt.text(
            column,
            row,
            cm[row, column],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    DATA_DIR / "random_forest_confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

lr_fpr, lr_tpr, _ = roc_curve(
    y_test,
    lr_probabilities
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression "
          f"(AUC={lr_metrics['roc_auc']:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest "
          f"(AUC={rf_metrics['roc_auc']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Baseline ML ROC Comparison"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    DATA_DIR / "roc_curve_comparison.png",
    dpi=150
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

plot_data = importance_df.head(
    min(20, len(importance_df))
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_data["feature"][::-1],
    plot_data["importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    DATA_DIR / "random_forest_feature_importance.png",
    dpi=150
)

plt.close()


print("\nBaseline ML completed successfully.")