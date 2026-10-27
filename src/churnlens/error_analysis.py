"""Error analysis: who does the model get wrong, and what do those customers look like?

Meant for the **final test set**, as a diagnosis after model selection. Nothing here feeds back
into model or threshold selection, and test results are not guarantees of future performance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .evaluation import confusion_counts

OUTCOMES = {
    "TP": "True positive: predicted churn, did churn",
    "FP": "False positive: predicted churn, did not churn",
    "FN": "False negative: predicted stay, did churn",
    "TN": "True negative: predicted stay, did stay",
}


def classify_outcomes(y_true, proba, threshold: float) -> pd.DataFrame:
    """Per-customer table keeping score, threshold, predicted class and actual class apart."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    outcome = np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["TP", "FP", "FN"],
        default="TN",
    )
    return pd.DataFrame(
        {
            "probability": p,
            "threshold": threshold,
            "predicted_class": pred,
            "actual_class": y,
            "outcome": pd.Categorical(outcome, categories=list(OUTCOMES)),
        }
    )


def error_metrics(outcomes: pd.DataFrame) -> dict[str, Any]:
    """Counts and rates derived from a ``classify_outcomes`` table."""
    c = confusion_counts(outcomes["actual_class"], outcomes["predicted_class"])

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        **c,
        "precision": ratio(c["tp"], c["tp"] + c["fp"]),
        "recall": ratio(c["tp"], c["tp"] + c["fn"]),
        "specificity": ratio(c["tn"], c["tn"] + c["fp"]),
        "false_positive_rate": ratio(c["fp"], c["fp"] + c["tn"]),
        "false_negative_rate": ratio(c["fn"], c["fn"] + c["tp"]),
        "negative_predictive_value": ratio(c["tn"], c["tn"] + c["fn"]),
        "flagged_share": ratio(c["tp"] + c["fp"], len(outcomes)),
    }


def confusion_frame(outcomes: pd.DataFrame) -> pd.DataFrame:
    """2x2 matrix, rows = actual class, columns = predicted class."""
    c = confusion_counts(outcomes["actual_class"], outcomes["predicted_class"])
    return pd.DataFrame(
        [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
        index=["Actual: stays", "Actual: churns"],
        columns=["Predicted: stays", "Predicted: churns"],
    )


def compare_groups(
    X: pd.DataFrame, in_group: pd.Series, reference: pd.Series, feature: str
) -> pd.DataFrame:
    """Describe ``feature`` in two groups of customers (e.g. false negatives vs true positives).

    Numeric -> mean, median, count. Categorical -> share of each category in each group, plus
    the difference in percentage points. ``in_group`` / ``reference`` are boolean masks on X.
    """
    a, b = X.loc[in_group, feature], X.loc[reference, feature]
    if pd.api.types.is_numeric_dtype(X[feature]):
        return pd.DataFrame(
            {
                "group": ["selected group", "reference group"],
                "n": [len(a), len(b)],
                "mean": [a.mean(), b.mean()],
                "median": [a.median(), b.median()],
            }
        )
    share_a = a.value_counts(normalize=True)
    share_b = b.value_counts(normalize=True)
    table = pd.DataFrame({"selected_group_%": share_a, "reference_group_%": share_b}).fillna(0.0) * 100
    table["difference_pts"] = table["selected_group_%"] - table["reference_group_%"]
    return table.sort_values("difference_pts", key=np.abs, ascending=False).rename_axis(feature)


def error_profile(
    X: pd.DataFrame, outcomes: pd.DataFrame, error: str = "FN", features: list[str] | None = None
) -> pd.DataFrame:
    """Which features differ most between the errors of one kind and correct predictions?

    ``FN`` is compared with ``TP`` (churners the model caught vs missed); ``FP`` with ``TN``
    (stayers wrongly flagged vs correctly left alone). Numeric features are ranked by the
    standardised mean difference; categorical ones by the largest category share gap.
    These are descriptive associations, not causes.
    """
    if error not in ("FN", "FP"):
        raise ValueError("error must be 'FN' or 'FP'.")
    reference_label = "TP" if error == "FN" else "TN"
    selected = (outcomes["outcome"] == error).to_numpy()
    reference = (outcomes["outcome"] == reference_label).to_numpy()
    X = X.reset_index(drop=True)
    rows = []
    for feature in features or list(X.columns):
        if selected.sum() == 0 or reference.sum() == 0:
            break
        a, b = X.loc[selected, feature], X.loc[reference, feature]
        if pd.api.types.is_numeric_dtype(X[feature]):
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
            effect = float((a.mean() - b.mean()) / pooled) if pooled and pooled > 0 else 0.0
            rows.append(
                {"feature": feature, "kind": "numeric", "effect_size": effect,
                 "detail": f"mean {a.mean():.2f} vs {b.mean():.2f}"}
            )
        else:
            share_a, share_b = a.value_counts(normalize=True), b.value_counts(normalize=True)
            levels = share_a.index.union(share_b.index)       # a level absent from one group counts as 0%
            gap = share_a.reindex(levels, fill_value=0.0) - share_b.reindex(levels, fill_value=0.0)
            if gap.empty:
                continue
            top = gap.abs().idxmax()
            rows.append(
                {"feature": feature, "kind": "categorical", "effect_size": float(gap[top]),
                 "detail": f"'{top}': {100 * a.eq(top).mean():.0f}% vs {100 * b.eq(top).mean():.0f}%"}
            )
    table = pd.DataFrame(rows, columns=["feature", "kind", "effect_size", "detail"])
    table["abs_effect"] = table["effect_size"].abs()
    return table.sort_values("abs_effect", ascending=False).drop(columns="abs_effect").reset_index(drop=True)
