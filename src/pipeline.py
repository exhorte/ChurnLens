"""
Single source of truth for the churn-modeling pipeline.

The whole transformation + estimator chain is built here and fitted ONCE, then
serialized as a single artifact (``models/churn_pipeline.pkl``). Inference reuses
the exact same fitted object, which removes by construction the train/serve skew
that used to come from re-fitting encoders at prediction time.

Exposes:
    - ``NUMERIC_FEATURES`` / ``CATEGORICAL_FEATURES``: canonical column lists
    - ``add_engineered_features``: pure, row-wise feature engineering
    - ``FeatureEngineer``: sklearn transformer wrapping the above
    - ``build_pipeline(estimator)``: FE -> preprocessing -> classifier
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Canonical feature contract (post feature-engineering)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "TotalServices",   # engineered
    "ContractRisk",    # engineered
    "ChargeRatio",     # engineered
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "TenureSegment",   # engineered
]

_CONTRACT_RISK = {"Month-to-month": 3, "One year": 2, "Two year": 1}
_SERVICE_KEYS = ("Online", "Streaming", "Device", "Tech")


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``X`` with derived features added.

    Pure and strictly row-wise (no cross-row statistics, no fitted state), so it
    behaves identically on a 7k-row training frame and on a single inference row.
    """
    X = pd.DataFrame(X).copy()

    # The raw Telco CSV stores TotalCharges as text with 11 blank cells; coerce
    # to numeric here (idempotent) and let the imputer handle the resulting NaNs.
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    service_cols = [c for c in X.columns if any(k in c for k in _SERVICE_KEYS)]
    if service_cols:
        X["TotalServices"] = (X[service_cols] == "Yes").sum(axis=1)

    if "Contract" in X.columns:
        X["ContractRisk"] = X["Contract"].map(_CONTRACT_RISK).fillna(2)

    if {"MonthlyCharges", "TotalCharges"} <= set(X.columns):
        X["ChargeRatio"] = X["MonthlyCharges"] / (X["TotalCharges"] + 1)

    if "tenure" in X.columns:
        # bins start at -1 so that tenure == 0 falls into the first segment
        # (the original [0, ...] dropped tenure==0 customers into a NaN bucket).
        X["TenureSegment"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 36, 72],
            labels=["0-1 year", "1-2 years", "2-3 years", "3+ years"],
        ).astype(str)

    return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer that injects engineered features into the pipeline."""

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X):
        return add_engineered_features(X)


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median-impute + standard-scale. Categorical: mode-impute + one-hot."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" makes single-row / unseen categories safe.
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # ignores leftovers such as customerID
    )


def build_pipeline(estimator) -> Pipeline:
    """Compose feature engineering, preprocessing and an estimator into one object."""
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("classifier", estimator),
        ]
    )
