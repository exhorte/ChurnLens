"""
Main model training script.

Trains candidate models inside a single fit-once / serve-once pipeline, selects
the best one via cross-validation on the training split only (the test split is
touched exactly once, for the final report), and serializes one artifact.

Usage:
    python train_model.py
"""

import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import clean_data, load_telco_data, validate_data
from src.feature_engineering import separate_features_and_target
from src.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MODELS_DIR = "models"
REPORTS_DIR = "reports"
PIPELINE_PATH = os.path.join(MODELS_DIR, "churn_pipeline.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# Candidate estimators. class_weight handles the ~27% churn imbalance.
CANDIDATES = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Standard binary-classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    print("\n" + "=" * 80)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 80)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load + clean (guardrail inside load_telco_data rejects synthetic data)
    print("\n[1/6] Loading and cleaning dataset...")
    df = clean_data(load_telco_data())
    validate_data(df)
    X, y = separate_features_and_target(df)

    # 2. Hold out a test set we only look at once
    print("\n[2/6] Splitting train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    - Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 3. Model selection via cross-validation on TRAIN ONLY (no test leakage)
    print(f"\n[3/6] Selecting model via {CV_FOLDS}-fold CV (ROC-AUC) on train...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, estimator in CANDIDATES.items():
        scores = cross_val_score(
            build_pipeline(estimator), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        cv_scores[name] = scores.mean()
        print(f"    - {name:22s} CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
    best_name = max(cv_scores, key=cv_scores.get)
    print(f"    => Best by CV: {best_name}")

    # 4. Evaluate every candidate once on the test set (for the comparison report)
    print("\n[4/6] Evaluating candidates on the held-out test set...")
    comparison_rows = []
    fitted = {}
    for name, estimator in CANDIDATES.items():
        pipe = build_pipeline(estimator).fit(X_train, y_train)
        fitted[name] = pipe
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        m = compute_metrics(y_test, pred, proba)
        comparison_rows.append({"Model": name, **m})

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .rename(
            columns={
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1-Score",
                "roc_auc": "ROC-AUC",
            }
        )
        .sort_values("ROC-AUC", ascending=False)
    )
    comparison_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print(comparison_df.to_string(index=False))

    # 5. Persist the best pipeline + metadata as the single serving artifact
    print("\n[5/6] Saving best pipeline...")
    best_pipe = fitted[best_name]
    best_pred = best_pipe.predict(X_test)
    best_proba = best_pipe.predict_proba(X_test)[:, 1]
    best_metrics = compute_metrics(y_test, best_pred, best_proba)
    print(classification_report(y_test, best_pred, target_names=["No Churn", "Churn"], digits=4))

    joblib.dump(best_pipe, PIPELINE_PATH)
    metadata = {
        "model": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_roc_auc": round(cv_scores[best_name], 4),
        "test_metrics": {k: round(v, 4) for k, v in best_metrics.items()},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    with open(METADATA_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2)

    # 6. Feature importance via permutation (estimator-agnostic, per raw feature)
    print("\n[6/6] Computing permutation feature importance...")
    try:
        result = permutation_importance(
            best_pipe, X_test, y_test, n_repeats=5,
            random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
        )
        importance_df = (
            pd.DataFrame({"Feature": X_test.columns, "Importance": result.importances_mean})
            .sort_values("Importance", ascending=False)
            .head(15)
        )
        importance_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False)
        print(importance_df.head(5).to_string(index=False))
    except Exception as exc:  # pragma: no cover - importance is non-critical
        print(f"    - Permutation importance skipped: {exc}")

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Best Model : {best_name}")
    print(f"Test F1    : {best_metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Saved to   : {PIPELINE_PATH}")
    print("\nRun the app with:  streamlit run app.py")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n❌ Error during training: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
