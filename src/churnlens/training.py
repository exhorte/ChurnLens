"""Experiment protocol: split -> compare models by CV -> select -> threshold -> test once.

Protocol (see docs/METHODOLOGY.md):

1. one stratified train/test split; the test set is untouched until step 6;
2. column roles / leakage screen and schema are inferred from the **training** split only;
3. every candidate model is cross-validated on the training split, on the *same* folds, inside
   a ``Pipeline`` (so preprocessing is re-fitted per fold);
4. the model is selected from CV scores only (primary metric, optional one-standard-error rule);
5. the decision threshold is chosen on out-of-fold predictions of the selected model;
6. the selected model is refitted on the whole training split and evaluated **once** on the test set.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from . import __version__
from .config import Config
from .data import (
    ColumnRoles,
    DataValidationError,
    PreparedData,
    file_sha256,
    infer_roles,
)
from .evaluation import (
    ALL_METRICS,
    LOWER_IS_BETTER,
    SegmentRule,
    ThresholdRecommendation,
    bootstrap_intervals,
    calibration_summary,
    compute_metrics,
    cv_scorers,
    recommend_threshold,
    threshold_table,
)
from .explainability import permutation_importance_table
from .pipeline import (
    MODEL_REGISTRY,
    FeatureSpec,
    build_pipeline,
    candidate_estimators,
)
from .schema import DataSchema, infer_schema, require_valid

logger = logging.getLogger(__name__)

PIPELINE_FILE = "churn_pipeline.pkl"
METADATA_FILE = "metadata.json"
_MAX_NATIVE_THREADS = min(4, os.cpu_count() or 1)


@dataclass
class ExperimentResult:
    run_id: str
    run_date: str
    config: Config
    spec: FeatureSpec
    schema: DataSchema
    roles: ColumnRoles
    prepared: PreparedData
    train_index: np.ndarray
    test_index: np.ndarray
    cv_folds: pd.DataFrame
    comparison: pd.DataFrame
    selected_key: str
    selection_note: str
    pipeline: Pipeline = field(repr=False)
    oof: pd.DataFrame = field(repr=False)
    threshold: ThresholdRecommendation = None  # type: ignore[assignment]
    test_predictions: pd.DataFrame = field(repr=False, default=None)  # type: ignore[assignment]
    test_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics_at_default: dict[str, float] = field(default_factory=dict)
    test_metrics_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    calibration_oof: dict[str, Any] = field(default_factory=dict)
    permutation: pd.DataFrame = field(repr=False, default=None)  # type: ignore[assignment]
    segments: SegmentRule = None  # type: ignore[assignment]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_label(self) -> str:
        return MODEL_REGISTRY[self.selected_key].label


# ------------------------------------------------------------------------- splitting
def split_indices(y: pd.Series, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stratified split; shared by training and the app so both see the same test set."""
    positions = np.arange(len(y))
    train, test = train_test_split(
        positions,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        stratify=y.to_numpy(),
    )
    return np.sort(train), np.sort(test)


def make_cv(config: Config, y_train: pd.Series) -> StratifiedKFold:
    smallest = int(y_train.value_counts().min())
    n_splits = max(2, min(config.cv.n_splits, smallest))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.cv.random_state)


# --------------------------------------------------------------------- CV comparison
def compare_models_cv(
    spec: FeatureSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    model_keys: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate every candidate on identical folds.

    Returns ``(per_fold, summary)``. ``score_time`` is scikit-learn's time to predict *and* score
    one fold, so it is an upper bound of the prediction time.
    """
    cv = make_cv(config, y_train)
    estimators = candidate_estimators(config, list(model_keys) if model_keys else None)
    fold_rows: list[dict[str, Any]] = []
    for key, estimator in estimators.items():
        if progress:
            progress(f"Cross-validating {MODEL_REGISTRY[key].label}...")
        raw = cross_validate(
            build_pipeline(spec, estimator),
            X_train,
            y_train,
            cv=cv,
            scoring=cv_scorers(),
            error_score="raise",
        )
        for fold in range(cv.get_n_splits()):
            row: dict[str, Any] = {"model": key, "fold": fold + 1}
            for metric in ALL_METRICS:
                value = float(raw[f"test_{metric}"][fold])
                row[metric] = -value if metric in LOWER_IS_BETTER else value
            row["fit_time_s"] = float(raw["fit_time"][fold])
            row["score_time_s"] = float(raw["score_time"][fold])
            fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    return folds, summarize_folds(folds)


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (ddof=1) of every metric across folds, per model."""
    metrics = [*ALL_METRICS, "fit_time_s", "score_time_s"]
    rows = []
    for key, group in folds.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": key,
            "label": MODEL_REGISTRY[key].label,
            "n_folds": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_model(summary: pd.DataFrame, config: Config) -> tuple[str, str]:
    """Choose a model from CV results only. Returns ``(key, explanation)``.

    The dummy baseline is never selectable. With ``rule="one_se"`` the simplest model whose mean
    is within one standard error of the best wins (the more complex model must earn its place).
    """
    metric = config.selection.primary_metric
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown primary metric '{metric}'. Choose from {list(ALL_METRICS)}.")
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    pool = summary[~summary["model"].map(lambda k: MODEL_REGISTRY[k].baseline_only)].copy()
    if pool.empty:
        raise ValueError("No selectable model among the candidates.")
    pool["score"] = sign * pool[f"{metric}_mean"]
    pool["se"] = pool[f"{metric}_std"] / np.sqrt(pool["n_folds"])
    best = pool.loc[pool["score"].idxmax()]
    if config.selection.rule == "best":
        return str(best["model"]), (
            f"Highest mean CV {metric} ({sign * best['score']:.4f}) among the candidates."
        )
    eligible = pool[pool["score"] >= best["score"] - best["se"]].copy()
    eligible["complexity"] = eligible["model"].map(lambda k: MODEL_REGISTRY[k].complexity)
    chosen = eligible.sort_values(["complexity", "score"], ascending=[True, False]).iloc[0]
    if chosen["model"] == best["model"]:
        note = f"Best mean CV {metric} ({sign * best['score']:.4f}), and the simplest eligible model."
    else:
        note = (
            f"Best CV {metric} is {MODEL_REGISTRY[best['model']].label} "
            f"({sign * best['score']:.4f} +/- {best['se']:.4f} SE); "
            f"{MODEL_REGISTRY[chosen['model']].label} ({sign * chosen['score']:.4f}) is within one "
            "standard error and simpler, so it is selected."
        )
    return str(chosen["model"]), note


# ------------------------------------------------------------------- full experiment
def run_experiment(
    prepared: PreparedData,
    config: Config,
    *,
    engineer: bool = False,
    exclude: Sequence[str] = (),
    allow: Sequence[str] = (),
    model_keys: Sequence[str] | None = None,
    data_file: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ExperimentResult:
    """Run the whole protocol on a prepared dataset and return every artefact."""
    # Gradient boosting is ~5x slower when BLAS and OpenMP both spawn one thread per core.
    with threadpool_limits(limits=_MAX_NATIVE_THREADS):
        return _run_experiment(
            prepared, config, engineer, exclude, allow, model_keys, data_file, progress
        )


def _run_experiment(
    prepared: PreparedData,
    config: Config,
    engineer: bool,
    exclude: Sequence[str],
    allow: Sequence[str],
    model_keys: Sequence[str] | None,
    data_file: str | Path | None,
    progress: Callable[[str], None] | None,
) -> ExperimentResult:
    say = progress or (lambda _msg: None)
    X_all, y_all = prepared.frame, prepared.y
    train_idx, test_idx = split_indices(y_all, config)
    X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all.iloc[test_idx]

    say("Screening columns on the training split...")
    roles = infer_roles(
        X_train, y_train, target=prepared.target.column, config=config, exclude=exclude, allow=allow
    )
    if not roles.used:
        raise DataValidationError("No usable feature column remains after screening.")
    spec = FeatureSpec(tuple(roles.numeric), tuple(roles.categorical), engineer=engineer)

    schema = infer_schema(X_train, roles.numeric, roles.categorical)
    X_train = require_valid(X_train[schema.names], schema)
    X_test = require_valid(X_test[schema.names], schema)

    folds, comparison = compare_models_cv(spec, X_train, y_train, config, model_keys, say)
    selected_key, selection_note = select_model(comparison, config)
    say(f"Selected {MODEL_REGISTRY[selected_key].label}. Computing out-of-fold predictions...")

    estimator = candidate_estimators(config, [selected_key])[selected_key]
    cv = make_cv(config, y_train)
    oof_proba = cross_val_predict(
        build_pipeline(spec, estimator), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    oof = pd.DataFrame({"row": train_idx, "y_true": y_train.to_numpy(), "proba": oof_proba})
    recommendation = recommend_threshold(oof["y_true"], oof["proba"], config.threshold.criterion)

    say("Refitting on the full training split and evaluating once on the test set...")
    final = build_pipeline(spec, candidate_estimators(config, [selected_key])[selected_key])
    final.fit(X_train, y_train)
    test_proba = final.predict_proba(X_test)[:, 1]
    test_predictions = pd.DataFrame(
        {"row": test_idx, "y_true": y_test.to_numpy(), "proba": test_proba}
    )
    if prepared.ids is not None:
        test_predictions.insert(0, "customer_id", prepared.ids.iloc[test_idx].to_numpy())
    metrics = compute_metrics(y_test, test_proba, recommendation.threshold)
    metrics_default = compute_metrics(y_test, test_proba, config.threshold.default)
    metrics_ci = bootstrap_intervals(
        y_test, test_proba, recommendation.threshold, seed=config.split.random_state
    )
    # Global explanation of the *already selected* model, measured on data it was not fitted on.
    scoring = config.selection.primary_metric
    permutation = permutation_importance_table(
        final, X_test, y_test,
        scoring=scoring if scoring in ("roc_auc", "average_precision") else "average_precision",
        random_state=config.split.random_state,
    )

    segments = SegmentRule(config.segments.low_below, config.segments.high_from)
    result = ExperimentResult(
        run_id=f"run-{uuid.uuid4().hex[:12]}",  # prefix: never parsed as a number by CSV readers
        run_date=datetime.now(UTC).isoformat(timespec="seconds"),
        config=config,
        spec=spec,
        schema=schema,
        roles=roles,
        prepared=prepared,
        train_index=train_idx,
        test_index=test_idx,
        cv_folds=folds,
        comparison=comparison,
        selected_key=selected_key,
        selection_note=selection_note,
        pipeline=final,
        oof=oof,
        threshold=recommendation,
        test_predictions=test_predictions,
        test_metrics=metrics,
        test_metrics_at_default=metrics_default,
        test_metrics_ci=metrics_ci,
        calibration_oof=calibration_summary(oof["y_true"], oof["proba"]),
        permutation=permutation,
        segments=segments,
    )
    result.metadata = build_metadata(result, data_file)
    return result


# ----------------------------------------------------------------------- persistence
def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "churnlens": __version__,
    }


def _selected_cv_summary(result: ExperimentResult) -> dict[str, float]:
    row = result.comparison.set_index("model").loc[result.selected_key]
    return {
        f"{m}_{stat}": float(row[f"{m}_{stat}"]) for m in ALL_METRICS for stat in ("mean", "std")
    }


def build_metadata(result: ExperimentResult, data_file: str | Path | None) -> dict[str, Any]:
    """Everything needed to identify, audit and reuse the model (JSON-serialisable)."""
    cfg = result.config
    return {
        "project": "ChurnLens",
        "run_id": result.run_id,
        "trained_at": result.run_date,
        "model_key": result.selected_key,
        "model": result.selected_label,
        "model_params": dict(getattr(cfg.models, result.selected_key)),
        "selection": {
            "primary_metric": cfg.selection.primary_metric,
            "rule": cfg.selection.rule,
            "explanation": result.selection_note,
            "candidates": result.comparison["model"].tolist(),
        },
        "cv_scores_selected_model": _selected_cv_summary(result),
        "cv_folds": int(result.comparison["n_folds"].iloc[0]),
        "threshold": result.threshold.to_dict(),
        "segments": result.segments.to_dict(),
        "test_metrics_at_recommended_threshold": result.test_metrics,
        "test_metrics_at_threshold_0.5": result.test_metrics_at_default,
        "test_metrics_ci_95": {k: list(v) for k, v in result.test_metrics_ci.items()},
        "calibration_out_of_fold": {
            k: v for k, v in result.calibration_oof.items() if k != "bins"
        },
        "target": {
            "column": result.prepared.target.column,
            "positive_label": str(result.prepared.target.positive_label),
            "negative_label": str(result.prepared.target.negative_label),
            "label_inferred": result.prepared.target.labels_were_inferred,
        },
        "features": result.spec.to_dict(),
        "ignored_columns": result.roles.ignored,
        "screening_warnings": result.roles.warnings,
        "schema": result.schema.to_dict(),
        "data": {
            "file": Path(data_file).name if data_file else None,
            "file_sha256": file_sha256(data_file) if data_file and Path(data_file).exists() else None,
            "frame_sha256": result.prepared.fingerprint,
            "rows": int(len(result.prepared.y)),
            "n_train": int(len(result.train_index)),
            "n_test": int(len(result.test_index)),
            "positive_rate": float(result.prepared.y.mean()),
            "id_column": result.prepared.id_label,
            "notes": result.prepared.notes,
        },
        "random_state": {"split": cfg.split.random_state, "cv": cfg.cv.random_state},
        "config_fingerprint": cfg.fingerprint(),
        "environment": _environment(),
    }


def save_experiment(result: ExperimentResult, models_dir: Path, reports_dir: Path) -> dict[str, Path]:
    """Write the serving artefact (+ integrity hash) and the reproducible reports."""
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = models_dir / PIPELINE_FILE
    joblib.dump(result.pipeline, pipeline_path)
    metadata = dict(result.metadata)
    metadata["artifact"] = {"file": PIPELINE_FILE, "sha256": file_sha256(pipeline_path)}
    metadata_path = models_dir / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    stamp = {
        "run_id": result.run_id,
        "run_date": result.run_date,
        "config_fingerprint": result.config.fingerprint(),
        "random_state": result.config.cv.random_state,
        "primary_metric": result.config.selection.primary_metric,
        "selection_rule": result.config.selection.rule,
    }
    comparison = result.comparison.copy()
    comparison.insert(2, "selected", comparison["model"] == result.selected_key)
    comparison = comparison.assign(**stamp, score_source="cross-validation on the training split")
    folds = result.cv_folds.assign(run_id=result.run_id)
    paths = {
        "pipeline": pipeline_path,
        "metadata": metadata_path,
        "model_comparison": reports_dir / "model_comparison.csv",
        "cv_folds": reports_dir / "cv_folds.csv",
        "oof_predictions": reports_dir / "oof_predictions.csv",
        "test_predictions": reports_dir / "test_predictions.csv",
        "test_evaluation": reports_dir / "test_evaluation.csv",
        "threshold_analysis": reports_dir / "threshold_analysis_validation.csv",
        "permutation_importance": reports_dir / "permutation_importance_test.csv",
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    folds.to_csv(paths["cv_folds"], index=False)
    result.oof.to_csv(paths["oof_predictions"], index=False)
    # Customer identifiers are data, not results: the committed report keeps only the row position
    # (join back to your own file by `row`), so reports never re-publish a dataset's identifiers.
    result.test_predictions.drop(columns="customer_id", errors="ignore").to_csv(
        paths["test_predictions"], index=False
    )
    evaluation = pd.DataFrame(
        [
            {"setting": "recommended threshold", "threshold": result.threshold.threshold,
             **result.test_metrics},
            {"setting": "default threshold 0.5", "threshold": result.config.threshold.default,
             **result.test_metrics_at_default},
        ]
    ).assign(model=result.selected_label, score_source="final test set (used once)", **{
        k: stamp[k] for k in ("run_id", "run_date")
    })
    evaluation.to_csv(paths["test_evaluation"], index=False)
    threshold_table(result.oof["y_true"], result.oof["proba"]).assign(
        score_source="out-of-fold predictions on the training split", run_id=result.run_id
    ).to_csv(paths["threshold_analysis"], index=False)
    result.permutation.assign(
        score_source="final test set, selected model only", run_id=result.run_id
    ).to_csv(paths["permutation_importance"], index=False)
    return paths
