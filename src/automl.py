"""
"Bring your own data" churn trainer (AutoML-lite).

Given an arbitrary tabular dataset and a binary target column, this module
infers feature types, builds a generic preprocessing + model pipeline, trains
and evaluates Logistic Regression and Random Forest, and returns a fitted
pipeline plus metrics. It is deliberately defensive: small / malformed inputs
raise a clear ``DataValidationError`` instead of producing a silently bad model.

No project-specific feature engineering is applied, so the resulting pipeline
depends only on scikit-learn and is portable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MIN_ROWS = 100
MAX_CATEGORIES = 50           # categorical columns with more uniques are treated as ID-like
MAX_TRAIN_ROWS = 50_000       # sample above this to keep training responsive
_POSITIVE_TOKENS = {"yes", "true", "1", "churn", "churned", "y", "t", "positive"}


class DataValidationError(ValueError):
    """Raised when an uploaded dataset cannot be turned into a sound model."""


@dataclass
class TrainResult:
    best_model: str
    results: dict[str, dict[str, Any]]
    numeric: list[str]
    categorical: list[str]
    dropped: list[tuple[str, str]]
    positive_label: Any
    schema: dict[str, dict[str, Any]]
    n_train: int
    n_test: int
    best_pipeline: Any = field(repr=False, default=None)


def infer_feature_types(
    df: pd.DataFrame, target_col: str, exclude: list[str] | None = None
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Split columns into numeric / categorical / dropped (with a reason)."""
    skip = set(exclude or []) | {target_col}
    numeric, categorical, dropped = [], [], []
    n = len(df)
    for col in df.columns:
        if col in skip:
            continue
        s = df[col]
        n_unique = s.nunique(dropna=True)
        if n_unique <= 1:
            dropped.append((col, "constant"))
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        elif n_unique > MAX_CATEGORIES or n_unique / n > 0.5:
            dropped.append((col, f"ID-like / high cardinality ({n_unique} values)"))
        else:
            categorical.append(col)
    return numeric, categorical, dropped


def prepare_target(series: pd.Series) -> tuple[pd.Series, Any]:
    """Map a binary target to 0/1 and return (y, positive_label).

    Positive class = whichever value looks churn-like (yes/true/1/churn...),
    else the value ``1`` for a 0/1 target, else the minority class.
    """
    classes = list(pd.Series(series.dropna().unique()))
    if len(classes) != 2:
        raise DataValidationError(
            f"The target must have exactly 2 distinct values; found {len(classes)}."
        )

    token_matches = [c for c in classes if str(c).strip().lower() in _POSITIVE_TOKENS]
    if len(token_matches) == 1:
        positive = token_matches[0]
    elif set(str(c).strip() for c in classes) == {"0", "1"}:
        positive = next(c for c in classes if str(c).strip() == "1")
    else:
        counts = series.value_counts()
        positive = counts.idxmin()  # minority class is positive (churn/fraud convention)

    y = (series == positive).astype("Int64")
    y[series.isna()] = pd.NA
    return y, positive


def build_generic_pipeline(numeric: list[str], categorical: list[str], estimator) -> Pipeline:
    """Generic preprocessing (impute+scale / impute+one-hot) + estimator."""
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            categorical,
        ))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", pre), ("classifier", estimator)])


def _metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def _schema(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict[str, dict]:
    """Per-column info used to auto-build a prediction form."""
    schema: dict[str, dict] = {}
    for col in numeric:
        s = pd.to_numeric(X[col], errors="coerce")
        schema[col] = {
            "type": "numeric",
            "min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
        }
    for col in categorical:
        schema[col] = {
            "type": "categorical",
            "choices": sorted(X[col].dropna().astype(str).unique().tolist()),
        }
    return schema


def train_on_dataframe(
    df: pd.DataFrame, target_col: str, exclude: list[str] | None = None, random_state: int = 42
) -> TrainResult:
    """Validate, train and evaluate two models on a user-supplied dataset."""
    if target_col not in df.columns:
        raise DataValidationError(f"Target column '{target_col}' not found.")
    if len(df) < MIN_ROWS:
        raise DataValidationError(f"Need at least {MIN_ROWS} rows; got {len(df)}.")

    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(MAX_TRAIN_ROWS, random_state=random_state)

    numeric, categorical, dropped = infer_feature_types(df, target_col, exclude)
    if not numeric and not categorical:
        raise DataValidationError("No usable feature columns were found.")

    y_full, positive = prepare_target(df[target_col])
    keep = y_full.notna()
    X = df.loc[keep, numeric + categorical].reset_index(drop=True)
    y = y_full[keep].astype(int).reset_index(drop=True)

    counts = y.value_counts()
    if len(counts) < 2:
        raise DataValidationError("The target has only one class after cleaning.")
    if counts.min() < 10:
        raise DataValidationError(
            f"The minority class has only {counts.min()} samples; need at least 10."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    folds = int(max(2, min(5, y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, estimator in candidates.items():
        pipe = build_generic_pipeline(numeric, categorical, estimator)
        cv_auc = float(cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        results[name] = {
            "pipeline": pipe,
            "cv_roc_auc": cv_auc,
            "metrics": _metrics(y_test, pred, proba),
            "confusion_matrix": confusion_matrix(y_test, pred),
        }

    best = max(results, key=lambda k: results[k]["cv_roc_auc"])
    best_pipe = results[best]["pipeline"]

    # permutation importance (best, per raw feature)
    try:
        imp = permutation_importance(
            best_pipe, X_test, y_test, n_repeats=5,
            random_state=random_state, scoring="roc_auc", n_jobs=-1)
        results[best]["importance"] = (
            pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:  # pragma: no cover
        results[best]["importance"] = None

    return TrainResult(
        best_model=best, results=results, numeric=numeric, categorical=categorical,
        dropped=dropped, positive_label=positive, schema=_schema(X, numeric, categorical),
        n_train=len(X_train), n_test=len(X_test), best_pipeline=best_pipe,
    )
