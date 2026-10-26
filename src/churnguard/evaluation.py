"""Metrics, decision-threshold analysis, calibration and risk segments.

Vocabulary used everywhere in the project:

* **score / probability** - the model's predicted probability of the positive class (churn);
* **threshold** - the cut-off turning a probability into a decision;
* **predicted class** - 1 if probability >= threshold, else 0;
* **actual class** - the observed outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD_FREE = ("roc_auc", "average_precision", "log_loss", "brier_score")
THRESHOLD_DEPENDENT = ("precision", "recall", "f1", "balanced_accuracy", "accuracy")
ALL_METRICS = ("roc_auc", "average_precision", *THRESHOLD_DEPENDENT, "log_loss", "brier_score")
# For these, lower is better.
LOWER_IS_BETTER = frozenset({"log_loss", "brier_score"})
_EPS = 1e-15


def cv_scorers() -> dict[str, Any]:
    """Scorers for ``cross_validate`` (precision/recall/F1 use the default 0.5 cut-off)."""
    return {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "log_loss": "neg_log_loss",
        "brier_score": "neg_brier_score",
    }


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """All metrics for one set of predictions; classification metrics use ``threshold``."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y_true and proba must have the same length ({y.shape} vs {p.shape}).")
    if len(np.unique(y)) < 2:
        raise ValueError("Metrics need both classes in y_true.")
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.clip(p, _EPS, 1 - _EPS), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
    }


_BOOTSTRAPPABLE = ("roc_auc", "average_precision", "precision", "recall", "f1")


def _sample_metrics(y: np.ndarray, p: np.ndarray, threshold: float, metrics: tuple[str, ...]) -> dict[str, float]:
    """Only the requested metrics, computed cheaply (a bootstrap evaluates thousands of resamples)."""
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    flagged, positives = int(pred.sum()), int(y.sum())
    precision = tp / flagged if flagged else 0.0
    recall = tp / positives
    out: dict[str, float] = {}
    for name in metrics:
        if name == "roc_auc":
            out[name] = float(roc_auc_score(y, p))
        elif name == "average_precision":
            out[name] = float(average_precision_score(y, p))
        elif name == "precision":
            out[name] = precision
        elif name == "recall":
            out[name] = recall
        else:  # f1
            out[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return out


def bootstrap_intervals(
    y_true,
    proba,
    threshold: float,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    level: float = 0.95,
    metrics: tuple[str, ...] = ("roc_auc", "average_precision", "precision", "recall", "f1"),
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap interval of test metrics: how much a score moves with the sample.

    Resamples *customers* with replacement from fixed predictions (the model is not refitted), so
    it captures test-sample noise only - not training variability, and not drift to new data.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    unknown = set(metrics) - set(_BOOTSTRAPPABLE)
    if unknown:
        raise ValueError(f"Cannot bootstrap {sorted(unknown)}; choose from {list(_BOOTSTRAPPABLE)}.")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        ys, ps = y[idx], p[idx]
        if ys.min() == ys.max():                    # a resample with one class has no ROC/AP
            continue
        for name, value in _sample_metrics(ys, ps, threshold, metrics).items():
            samples[name].append(value)
    alpha = (1 - level) / 2
    return {
        m: (float(np.quantile(v, alpha)), float(np.quantile(v, 1 - alpha))) for m, v in samples.items()
    }


def confusion_counts(y_true, pred) -> dict[str, int]:
    y = np.asarray(y_true).astype(int)
    q = np.asarray(pred).astype(int)
    return {
        "tp": int(((y == 1) & (q == 1)).sum()),
        "fp": int(((y == 0) & (q == 1)).sum()),
        "fn": int(((y == 1) & (q == 0)).sum()),
        "tn": int(((y == 0) & (q == 0)).sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def default_thresholds() -> np.ndarray:
    return np.round(np.arange(0.05, 0.951, 0.05), 2)


def threshold_table(y_true, proba, thresholds=None) -> pd.DataFrame:
    """Operational consequences of each decision threshold.

    ``flagged`` = customers predicted to churn (probability >= threshold).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    grid = default_thresholds() if thresholds is None else np.asarray(thresholds, dtype=float)
    rows = []
    for t in grid:
        c = confusion_counts(y, p >= t)
        flagged = c["tp"] + c["fp"]
        precision = _safe_div(c["tp"], flagged)
        recall = _safe_div(c["tp"], c["tp"] + c["fn"])
        rows.append(
            {
                "threshold": float(t),
                "flagged": flagged,
                "flagged_share": flagged / len(y),
                **c,
                "precision": precision,
                "recall": recall,
                "f1": _safe_div(2 * precision * recall, precision + recall)
                if flagged and c["tp"] else 0.0,
                "false_positive_rate": _safe_div(c["fp"], c["fp"] + c["tn"]),
                "false_negative_rate": _safe_div(c["fn"], c["fn"] + c["tp"]),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ThresholdRecommendation:
    threshold: float
    criterion: str
    score: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "criterion": self.criterion,
            "score_on_validation": self.score,
            "description": self.description,
        }


_CRITERIA = {
    "f1": "maximises F1 (balance of precision and recall)",
    "f2": "maximises F2 (recall counts twice as much as precision)",
    "youden": "maximises Youden's J = recall - false positive rate",
}


def recommend_threshold(
    y_true, proba, criterion: str = "f1", thresholds=None
) -> ThresholdRecommendation:
    """Pick the threshold that maximises ``criterion``.

    Must be called on **validation** predictions (out-of-fold training predictions), never on the
    final test set. Ties go to the lowest threshold. This is a generic statistical default: the
    right operating point is a business choice, see the retention simulator.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {sorted(_CRITERIA)}.")
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2) if thresholds is None else np.asarray(thresholds)
    best_t, best_score = float(grid[0]), -np.inf
    for t in grid:
        pred = (p >= t).astype(int)
        if criterion == "f1":
            score = f1_score(y, pred, zero_division=0)
        elif criterion == "f2":
            score = fbeta_score(y, pred, beta=2, zero_division=0)
        else:
            c = confusion_counts(y, pred)
            score = _safe_div(c["tp"], c["tp"] + c["fn"]) - _safe_div(c["fp"], c["fp"] + c["tn"])
        if np.isfinite(score) and score > best_score + 1e-12:
            best_t, best_score = float(t), float(score)
    return ThresholdRecommendation(
        threshold=best_t,
        criterion=criterion,
        score=best_score,
        description=(
            f"Threshold {best_t:.2f} {_CRITERIA[criterion]} on out-of-fold predictions of the "
            "training set (the test set is not used)."
        ),
    )


def calibration_summary(y_true, proba, n_bins: int = 10) -> dict[str, Any]:
    """Do predicted probabilities match observed frequencies? (uniform-width bins)"""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append(
                {
                    "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                    "n": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "observed_rate": float(y[mask].mean()),
                }
            )
    bins = pd.DataFrame(rows)
    ece = float((bins["n"] * (bins["mean_predicted"] - bins["observed_rate"]).abs()).sum() / len(y))
    return {
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": ece,
        "bins": bins,
    }


@dataclass(frozen=True)
class SegmentRule:
    low_below: float
    high_from: float

    LABELS = ("Low", "Medium", "High")

    def to_dict(self) -> dict[str, float]:
        return {"low_below": self.low_below, "high_from": self.high_from}


def assign_segments(proba, rule: SegmentRule) -> pd.Categorical:
    """Low (< low_below), Medium, High (>= high_from) - a business convention, not a result."""
    p = np.asarray(proba, dtype=float)
    labels = np.where(p < rule.low_below, "Low", np.where(p >= rule.high_from, "High", "Medium"))
    return pd.Categorical(labels, categories=list(SegmentRule.LABELS), ordered=True)


def segment_summary(y_true, proba, rule: SegmentRule) -> pd.DataFrame:
    """Size and observed positive rate of each risk segment (checks the segments mean something)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    seg = assign_segments(p, rule)
    rows = []
    for label in SegmentRule.LABELS:
        mask = np.asarray(seg == label)
        rows.append(
            {
                "segment": label,
                "customers": int(mask.sum()),
                "share": float(mask.mean()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else float("nan"),
                "observed_rate": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)
