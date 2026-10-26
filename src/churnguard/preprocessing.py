"""Preprocessing building blocks: the Telco feature engineer and the generic preprocessor.

Both live *inside* the scikit-learn pipeline, so they are fitted on training folds only and the
very same object is used at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import DataValidationError

# --- Telco-specific engineered features (opt-in: FeatureSpec.engineer) -------------------
SERVICE_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
# Contract length in months: a neutral, domain-based encoding (not derived from churn rates).
CONTRACT_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}
ENGINEERED_NUMERIC = ("TotalServices", "ContractMonths", "ChargeRatio")
ENGINEERED_CATEGORICAL = ("TenureSegment",)
TENURE_BINS = [-np.inf, 12, 24, 36, np.inf]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-36 months", "37+ months"]
TELCO_REQUIRED = (*SERVICE_COLUMNS, "Contract", "MonthlyCharges", "TotalCharges", "tenure")


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, strictly row-wise feature engineering for the Telco schema.

    Row-wise means it behaves identically on a 7k-row frame and on a single inference row.
    ``TotalCharges`` is blank exactly when ``tenure == 0`` in the Telco data (no bill issued yet,
    checked on the 11 blank rows), so those rows get 0 instead of an imputed median.
    """

    def fit(self, X: pd.DataFrame, y=None) -> TelcoFeatureEngineer:  # noqa: D102
        self._check_columns(X)
        self.is_fitted_ = True  # stateless; lets sklearn's check_is_fitted accept pipeline slices
        return self

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in TELCO_REQUIRED if c not in X.columns]
        if missing:
            raise DataValidationError(f"Missing required column(s): {missing}.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        X = pd.DataFrame(X).copy()
        self._check_columns(X)
        total = pd.to_numeric(X["TotalCharges"], errors="coerce")
        tenure = pd.to_numeric(X["tenure"], errors="coerce")
        total = total.mask(total.isna() & (tenure == 0), 0.0)
        X["TotalCharges"] = total
        X["TotalServices"] = (X[list(SERVICE_COLUMNS)] == "Yes").sum(axis=1)
        X["ContractMonths"] = X["Contract"].map(CONTRACT_MONTHS).astype(float)
        X["ChargeRatio"] = pd.to_numeric(X["MonthlyCharges"], errors="coerce") / (total + 1.0)
        segment = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS)
        X["TenureSegment"] = segment.astype(object)
        return X


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Numeric: median-impute + scale. Categorical: mode-impute + one-hot (unknown-safe)."""
    numeric_steps = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Dense output: required by HistGradientBoosting and cheap at this scale.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_steps, list(numeric)),
            ("cat", categorical_steps, list(categorical)),
        ],
        remainder="drop",
    )
