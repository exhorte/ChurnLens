"""What the app works on: a trained model plus the data around it.

Pure Python (no Streamlit import) so the logic is unit-tested. A ``Workspace`` comes either from
the saved demo model (artefacts in models/ and reports/) or from a model trained in the session
on a user's CSV; the pages only ever see this one shape.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnguard.config import Config
from churnguard.data import (
    DatasetNotFoundError,
    DataValidationError,
    PreparedData,
    load_demo_dataset,
    prepare_dataset,
)
from churnguard.predictions import ModelBundle, ScoredFrame, bundle_from_experiment, load_bundle, score_frame
from churnguard.schema import require_valid
from churnguard.training import ExperimentResult

POPULATION_TEST = "Held-out test customers (unseen during training)"
POPULATION_ALL = "All customers of the dataset (includes training customers)"
POPULATION_UPLOAD = "New customers from an uploaded CSV (no target needed)"


@dataclass
class Workspace:
    name: str
    is_demo: bool
    bundle: ModelBundle
    comparison: pd.DataFrame
    cv_folds: pd.DataFrame | None
    oof: pd.DataFrame
    test_predictions: pd.DataFrame
    test_metrics: dict[str, float]
    test_metrics_default: dict[str, float]
    permutation: pd.DataFrame | None
    prepared: PreparedData | None = None
    ignored_columns: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source_file: str | None = None

    @property
    def has_features(self) -> bool:
        return self.prepared is not None

    @property
    def metadata(self) -> dict:
        return self.bundle.metadata


@dataclass
class Population:
    kind: str
    description: str
    X: pd.DataFrame                       # validated raw features, one row per customer
    ids: pd.Series | None
    actual: pd.Series | None              # observed outcomes (0/1) when known
    warnings: list[str]
    scored: ScoredFrame

    @property
    def table(self) -> pd.DataFrame:
        return self.scored.table

    @property
    def has_actual(self) -> bool:
        return self.actual is not None


# --------------------------------------------------------------------------- builders
def workspace_from_experiment(
    result: ExperimentResult, name: str, source_file: str | None = None
) -> Workspace:
    """Wrap an in-session experiment (a model trained on the user's own data)."""
    return Workspace(
        name=name,
        is_demo=False,
        bundle=bundle_from_experiment(result),
        comparison=result.comparison.assign(selected=lambda d: d["model"] == result.selected_key),
        cv_folds=result.cv_folds,
        oof=result.oof,
        test_predictions=result.test_predictions,
        test_metrics=result.test_metrics,
        test_metrics_default=result.test_metrics_at_default,
        permutation=result.permutation,
        prepared=result.prepared,
        ignored_columns=dict(result.roles.ignored),
        warnings=[*result.roles.warnings, *result.prepared.notes],
        source_file=source_file,
    )


def _read_report(reports_dir: Path, name: str) -> pd.DataFrame | None:
    path = reports_dir / name
    return pd.read_csv(path) if path.exists() else None


def load_demo_workspace(config: Config) -> Workspace:
    """Saved demo model + its reports; the Telco CSV is optional (pages degrade without it)."""
    bundle = load_bundle(config.models_dir)            # raises ModelLoadError with guidance
    reports = config.reports_dir
    comparison = _read_report(reports, "model_comparison.csv")
    oof = _read_report(reports, "oof_predictions.csv")
    test_predictions = _read_report(reports, "test_predictions.csv")
    test_eval = _read_report(reports, "test_evaluation.csv")
    if any(x is None for x in (comparison, oof, test_predictions, test_eval)):
        raise DataValidationError(
            "Reports are missing from reports/: run `python train_model.py` to regenerate them."
        )
    metric_cols = [c for c in test_eval.columns if c in bundle.metadata["test_metrics_at_recommended_threshold"]]
    by_setting = test_eval.set_index("setting")[metric_cols]
    ws = Workspace(
        name="Demo - Telco churn (saved model)",
        is_demo=True,
        bundle=bundle,
        comparison=comparison,
        cv_folds=_read_report(reports, "cv_folds.csv"),
        oof=oof,
        test_predictions=test_predictions,
        test_metrics=by_setting.loc["recommended threshold"].to_dict(),
        test_metrics_default=by_setting.loc["default threshold 0.5"].to_dict(),
        permutation=_read_report(reports, "permutation_importance_test.csv"),
        source_file=bundle.metadata.get("data", {}).get("file"),
    )
    _attach_demo_data(ws, config)
    return ws


def _attach_demo_data(ws: Workspace, config: Config) -> None:
    """Re-read the Telco CSV and check it is the very data the model was trained on."""
    try:
        raw = load_demo_dataset(config)
    except DatasetNotFoundError:
        ws.warnings.append(
            "The Telco CSV is not in data/ (it is not redistributed): the explorer, error analysis "
            "and customer-level pages are limited. See data/README.md. You can still upload a CSV."
        )
        return
    cfg = config
    prepared = prepare_dataset(raw, cfg.data.target, positive_label=cfg.data.positive_label,
                               id_column=cfg.data.id_column, config=cfg)
    expected = ws.metadata.get("data", {}).get("frame_sha256")
    if expected and prepared.fingerprint != expected:
        ws.warnings.append(
            "The Telco CSV on disk differs from the one the model was trained on (content hash "
            "mismatch): customer-level pages are disabled. Retrain with `python train_model.py`."
        )
        return
    rows = ws.test_predictions["row"].to_numpy()
    if not np.array_equal(prepared.y.iloc[rows].to_numpy(), ws.test_predictions["y_true"].to_numpy()):
        ws.warnings.append("Saved test labels do not match the dataset: customer-level pages disabled.")
        return
    ws.prepared = prepared


# ------------------------------------------------------------------------ populations
def _validated(ws: Workspace, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from churnguard.schema import validate_features

    report = validate_features(frame, ws.bundle.schema)
    report.raise_if_errors()
    assert report.frame is not None
    return report.frame, report.warnings


def population_for(
    ws: Workspace,
    kind: str,
    threshold: float,
    uploaded: pd.DataFrame | None = None,
) -> Population:
    """Customers to analyse, scored with the workspace model at ``threshold``."""
    warnings: list[str] = []
    if kind == POPULATION_UPLOAD:
        if uploaded is None:
            raise DataValidationError("Upload a CSV of customers to score.")
        id_col = ws.metadata.get("data", {}).get("id_column")
        ids = uploaded[id_col].astype(str).reset_index(drop=True) if id_col in uploaded.columns else None
        frame = uploaded.reset_index(drop=True)
        X, warnings = _validated(ws, frame)
        actual, description = None, f"{len(X):,} uploaded customers"
    else:
        if ws.prepared is None:
            raise DataValidationError(
                "This view needs the training dataset (data/ CSV for the demo, or the CSV you "
                "trained on). Load it, or choose 'New customers from an uploaded CSV'."
            )
        schema = ws.bundle.schema
        if kind == POPULATION_TEST:
            rows = ws.test_predictions["row"].to_numpy()
            description = f"{len(rows):,} held-out test customers"
        else:
            rows = np.arange(len(ws.prepared.y))
            description = f"{len(rows):,} customers (all rows)"
            warnings.append(
                "This includes the customers the model was trained on: their scores are "
                "optimistic and must not be read as performance on new customers."
            )
        X = require_valid(ws.prepared.frame.iloc[rows][schema.names], schema)
        X = X.reset_index(drop=True)
        ids = ws.prepared.ids.iloc[rows].reset_index(drop=True) if ws.prepared.ids is not None else None
        actual = ws.prepared.y.iloc[rows].reset_index(drop=True)
    scored = score_frame(ws.bundle, X, ids=ids, threshold=threshold)
    return Population(kind, description, X, ids, actual, warnings, scored)
