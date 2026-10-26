"""
Feature engineering helpers for the Customer Churn Prediction project.

Encoding and scaling now live inside the serving pipeline (``src/pipeline.py``),
so this module only keeps the small, stateless helpers used by the training
script, the notebook and the tests.
"""

import pandas as pd

from .pipeline import add_engineered_features


def separate_features_and_target(df: pd.DataFrame, target_col: str = 'Churn') -> tuple:
    """
    Separate features and target variable.

    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column

    Returns:
        tuple: (X features dataframe, y target series)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col].map({'Yes': 1, 'No': 0})

    return X, y


def engineer_features(X: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
    """
    Create and engineer new features (total services, contract risk, charge
    ratio, tenure segment).

    Args:
        X (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features (optional)

    Returns:
        tuple: (X_engineered, X_test_engineered)

    Note:
        The actual logic lives in :func:`src.pipeline.add_engineered_features`,
        the single source of truth shared with the serving pipeline. This wrapper
        only preserves the legacy ``(X, X_test)`` signature used by the tests.
    """
    X = add_engineered_features(X)
    if X_test is not None:
        X_test = add_engineered_features(X_test)
    return X, X_test


if __name__ == "__main__":
    print("Feature engineering module loaded successfully")
