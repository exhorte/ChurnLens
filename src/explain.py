"""
Per-customer prediction explanations.

For the deployed linear model the decomposition is exact:

    logit(churn) = intercept + Σ_i  coef_i · z_i

where ``z`` is the preprocessed feature vector. Each ``coef_i · z_i`` is the
feature's contribution to the churn log-odds — i.e. the (linear) SHAP value.
One-hot columns are aggregated back to their source feature so the output is
human-readable. No extra dependencies, so it runs anywhere the pipeline does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def supports_explanation(pipeline) -> bool:
    """True if the deployed estimator exposes linear coefficients."""
    return hasattr(pipeline.named_steps.get("classifier"), "coef_")


def explain_customer(pipeline, X_row: pd.DataFrame) -> pd.DataFrame:
    """Return per-feature contributions to the churn log-odds for one customer.

    Args:
        pipeline: the fitted churn pipeline (engineer -> preprocess -> classifier)
        X_row: a one-row DataFrame with the raw customer fields

    Returns:
        DataFrame with columns ``feature`` and ``contribution`` (log-odds),
        sorted by absolute impact. Positive = pushes toward churn.
    """
    classifier = pipeline.named_steps["classifier"]
    preprocess = pipeline.named_steps["preprocess"]

    # raw -> engineered -> preprocessed model input (all steps except the classifier)
    z = pipeline[:-1].transform(X_row)
    if hasattr(z, "toarray"):
        z = z.toarray()
    z = np.asarray(z, dtype=float)[0]

    contributions = classifier.coef_[0] * z

    # Map each output column back to its source feature.
    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    sources = list(NUMERIC_FEATURES)
    for feature, categories in zip(CATEGORICAL_FEATURES, ohe.categories_):
        sources += [feature] * len(categories)

    df = pd.DataFrame({"feature": sources, "contribution": contributions})
    agg = df.groupby("feature", as_index=False)["contribution"].sum()
    return (
        agg.reindex(agg["contribution"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )
