# ============================================================
# GRIDGUARD AI
# PHASE 5 — BASELINE MACHINE LEARNING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)

import joblib


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path("../data/processed/features.csv")

MODEL_DIR = Path("../models")

OUTPUT_DIR = Path("../data/processed")


# ============================================================
# SETTINGS
# ============================================================

ID_COLUMN = "CONS_NO"
TARGET_COLUMN = "FLAG"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GRIDGUARD AI — PHASE 5")
print("BASELINE MACHINE LEARNING")
print("=" * 70)

print("\nLoading feature dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# CHECK DATA
# ============================================================

print("\nTarget distribution:")

print(
    df[TARGET_COLUMN].value_counts()
)


print("\nMissing values:")

print(
    df.isnull().sum().sum()
)


# ============================================================
# CREATE X AND y
# ============================================================

print("\nPreparing features...")

X = df.drop(
    columns=[ID_COLUMN, TARGET_COLUMN]
)

y = df[TARGET_COLUMN]


print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


print("\nFeatures used:")

for column in X.columns:
    print(" -", column)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


print("\nTraining class distribution:")

print(y_train.value_counts())


print("\nTesting class distribution:")

print(y_test.value_counts())


# ============================================================
# FEATURE SCALING
# ============================================================

print("\nScaling features for Logistic Regression...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_predictions = (
    logistic_model.predict(X_test_scaled)
)

logistic_probabilities = (
    logistic_model.predict_proba(X_test_scaled)[:, 1]
)


# ============================================================
# LOGISTIC REGRESSION EVALUATION
# ============================================================

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)


print("\nLOGISTIC REGRESSION RESULTS")

print(
    f"Accuracy : {logistic_accuracy:.4f}"
)

print(
    f"Precision: {logistic_precision:.4f}"
)

print(
    f"Recall   : {logistic_recall:.4f}"
)

print(
    f"F1 Score : {logistic_f1:.4f}"
)

print(
    f"ROC-AUC  : {logistic_auc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=[
            "Normal",
            "Suspicious"
        ],
        zero_division=0
    )
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)


# ============================================================
# RANDOM FOREST PREDICTIONS
# ============================================================

rf_predictions = (
    random_forest.predict(X_test)
)

rf_probabilities = (
    random_forest.predict_proba(X_test)[:, 1]
)


# ============================================================
# RANDOM FOREST EVALUATION
# ============================================================

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)


print("\nRANDOM FOREST RESULTS")

print(
    f"Accuracy : {rf_accuracy:.4f}"
)

print(
    f"Precision: {rf_precision:.4f}"
)

print(
    f"Recall   : {rf_recall:.4f}"
)

print(
    f"F1 Score : {rf_f1:.4f}"
)

print(
    f"ROC-AUC  : {rf_auc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=[
            "Normal",
            "Suspicious"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX — RANDOM FOREST
# ============================================================

print("\nCreating Random Forest confusion matrix...")

cm = confusion_matrix(
    y_test,
    rf_predictions
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
).plot(
    ax=ax
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_DIR / "random_forest_confusion_matrix.png",
    dpi=150
)

plt.show()


# ============================================================
# ROC CURVES
# ============================================================

print("\nCreating ROC curve...")

lr_fpr, lr_tpr, _ = roc_curve(
    y_test,
    logistic_probabilities
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)


plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC={logistic_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC={rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve — GridGuard AI"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve_comparison.png",
    dpi=150
)

plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)


print("\nTop features:")

print(
    feature_importance.head(15)
)


# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top Random Forest Features"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_feature_importance.png",
    dpi=150
)

plt.show()


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)


# ============================================================
# SAVE MODELS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving models...")

joblib.dump(
    logistic_model,
    MODEL_DIR / "logistic_regression.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl"
)

joblib.dump(
    random_forest,
    MODEL_DIR / "random_forest.pkl"
)


print("Models saved successfully.")


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ],

    "Precision": [
        logistic_precision,
        rf_precision
    ],

    "Recall": [
        logistic_recall,
        rf_recall
    ],

    "F1": [
        logistic_f1,
        rf_f1
    ],

    "ROC_AUC": [
        logistic_auc,
        rf_auc
    ]
})


results.to_csv(
    OUTPUT_DIR / "baseline_ml_results.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("BASELINE ML COMPLETE")
print("=" * 70)

print("\nModel comparison:")

print(results.to_string(index=False))

print("\nFiles created:")

print(" - models/logistic_regression.pkl")
print(" - models/scaler.pkl")
print(" - models/random_forest.pkl")

print(" - data/processed/baseline_ml_results.csv")
print(" - data/processed/feature_importance.csv")
print(" - data/processed/random_forest_confusion_matrix.png")
print(" - data/processed/roc_curve_comparison.png")
print(" - data/processed/random_forest_feature_importance.png")

print("\n" + "=" * 70)
print("READY FOR PHASE 6 — ISOLATION FOREST")
print("=" * 70)