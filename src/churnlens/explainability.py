"""Explainability: global importance and per-customer explanations, tied to the real model.

Global (what the model relies on overall)
  * ``native_importance``   - logistic coefficients / tree impurity importances;
  * ``permutation_importance_table`` - drop in a metric when one raw feature is shuffled.
Local (why one customer got this score)
  * linear models: *exact* decomposition of the log-odds (intercept + sum of contributions);
  * any other model: one-at-a-time *baseline substitution* (see ``explain_prediction``).

All of these describe associations the model learned. They do **not** show that a factor
causes churn, and changing a factor will not necessarily change a customer's behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .pipeline import encoded_feature_table, final_estimator
from .schema import DataSchema

ASSOCIATION_NOTICE = (
    "These factors describe associations learned by the model from past data. They do not show "
    "that a factor causes churn, and acting on one may not change a customer's behaviour."
)


def is_linear(pipeline: Pipeline) -> bool:
    coef = getattr(final_estimator(pipeline), "coef_", None)
    return coef is not None and np.ndim(coef) == 2 and np.shape(coef)[0] == 1


# --------------------------------------------------------------------------- global
def native_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """Model-specific importance per encoded input column, or ``None`` if the model has none.

    Logistic regression: coefficient (log-odds per standard deviation of a numeric feature, or
    for a category being present) and its odds ratio. Trees/forests: impurity importance, which
    favours high-cardinality and continuous features - prefer permutation importance to rank.
    """
    estimator = final_estimator(pipeline)
    names = encoded_feature_table(pipeline)
    if is_linear(pipeline):
        coef = np.ravel(estimator.coef_)
        out = names.assign(value=coef, odds_ratio=np.exp(coef), kind="logistic coefficient")
        return out.reindex(out["value"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None
    out = names.assign(value=importances, odds_ratio=np.nan, kind="impurity importance")
    return out.sort_values("value", ascending=False).reset_index(drop=True)


def permutation_importance_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y,
    *,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Drop in ``scoring`` when each *raw* input column is shuffled (model-agnostic).

    Run it on data the model was not fitted on (the test set), after model selection.
    """
    result = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------- local
@dataclass
class LocalExplanation:
    method: str                  # "linear_exact" | "baseline_substitution"
    table: pd.DataFrame          # feature, value, reference, contribution (sorted by |contribution|)
    unit: str                    # what ``contribution`` is measured in
    probability: float
    base_value: float            # intercept (log-odds) or baseline-customer probability
    description: str

    @property
    def logit(self) -> float | None:
        """Predicted log-odds (only meaningful for the exact linear method)."""
        if self.method != "linear_exact":
            return None
        return float(self.base_value + self.table["contribution"].sum())


def _engineered_view(pipeline: Pipeline, row: pd.DataFrame) -> pd.DataFrame:
    step = pipeline.named_steps["engineer"]
    return row if isinstance(step, str) else step.transform(row)


def explain_prediction(
    pipeline: Pipeline, schema: DataSchema, row: pd.DataFrame
) -> LocalExplanation:
    """Explain the score of one customer (``row`` = one validated row of raw features)."""
    if len(row) != 1:
        raise ValueError("explain_prediction expects exactly one row.")
    probability = float(pipeline.predict_proba(row)[0, 1])
    if is_linear(pipeline):
        return _explain_linear(pipeline, row, probability)
    return _explain_by_substitution(pipeline, schema, row, probability)


def _explain_linear(pipeline: Pipeline, row: pd.DataFrame, probability: float) -> LocalExplanation:
    estimator = final_estimator(pipeline)
    z = np.asarray(pipeline[:-1].transform(row), dtype=float)[0]
    names = encoded_feature_table(pipeline)
    if len(names) != len(z):
        raise RuntimeError("Encoded feature table does not match the model input width.")
    contributions = np.ravel(estimator.coef_) * z
    per_source = (
        names.assign(contribution=contributions).groupby("feature", sort=False)["contribution"].sum()
    )
    shown = _engineered_view(pipeline, row).iloc[0]
    table = pd.DataFrame(
        {
            "feature": per_source.index,
            "value": [shown.get(f, np.nan) for f in per_source.index],
            "reference": np.nan,
            "contribution": per_source.to_numpy(),
        }
    )
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
    return LocalExplanation(
        method="linear_exact",
        table=table.reset_index(drop=True),
        unit="log-odds",
        probability=probability,
        base_value=float(np.ravel(estimator.intercept_)[0]),
        description=(
            "Exact decomposition of the logistic model: intercept + sum of contributions = "
            "predicted log-odds of churn. One-hot columns are summed back to their source feature."
        ),
    )


def _explain_by_substitution(
    pipeline: Pipeline, schema: DataSchema, row: pd.DataFrame, probability: float
) -> LocalExplanation:
    baseline = schema.baseline_row()
    reference = pd.DataFrame([baseline], columns=schema.names).astype(row.dtypes.to_dict())
    base_prob = float(pipeline.predict_proba(reference)[0, 1])
    rows = []
    for feature in schema.names:
        altered = row.copy()
        altered[feature] = reference[feature].to_numpy()
        rows.append(
            {
                "feature": feature,
                "value": row[feature].iloc[0],
                "reference": baseline[feature],
                "contribution": probability - float(pipeline.predict_proba(altered)[0, 1]),
            }
        )
    table = pd.DataFrame(rows)
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
    return LocalExplanation(
        method="baseline_substitution",
        table=table.reset_index(drop=True),
        unit="probability points",
        probability=probability,
        base_value=base_prob,
        description=(
            "For each feature: predicted probability minus the probability when that feature alone "
            "is replaced by the typical training value (median / most frequent). Positive = the "
            "customer's actual value raises the score. Contributions are not additive when "
            "features interact; they do not sum exactly to the gap with the typical customer."
        ),
    )


def global_summary(pipeline: Pipeline) -> dict[str, Any]:
    """What kind of global explanation this model supports (drives the UI)."""
    return {
        "linear": is_linear(pipeline),
        "has_native_importance": native_importance(pipeline) is not None,
    }
