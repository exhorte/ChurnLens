"""Train a model on a user's CSV: guarded, explicit, and clearly labelled as custom."""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnguard.config import Config
from churnguard.data import (
    POSITIVE_TOKENS,
    ColumnRoles,
    DataValidationError,
    infer_roles,
    prepare_dataset,
    prepare_target,
    standardize_frame,
)
from churnguard.training import run_experiment

from .workspace import Workspace, workspace_from_experiment

_ID_HINTS = ("id", "uuid", "customer", "client", "account")


def suggest_positive_label(values: Sequence) -> str | None:
    """A churn-like value if exactly one class looks like it, else ``None`` (user must choose)."""
    matches = [v for v in values if str(v).strip().lower() in POSITIVE_TOKENS]
    return str(matches[0]) if len(matches) == 1 else None


def suggest_id_column(df: pd.DataFrame, target: str | None) -> str | None:
    """Best guess for an identifier column: name hints plus (near-)unique values."""
    for col in df.columns:
        if col == target:
            continue
        if any(hint in col.lower() for hint in _ID_HINTS) and df[col].nunique() >= 0.9 * len(df):
            return col
    return None


def preview_roles(
    df: pd.DataFrame, target: str, positive_label: str, id_column: str | None,
    exclude: Sequence[str], config: Config,
) -> ColumnRoles:
    """Which columns would be used / ignored / rejected (on all rows - a preview only).

    The real screening in ``run_experiment`` is done on the training split alone.
    """
    cleaned, _ = standardize_frame(df)
    y, _ = prepare_target(cleaned[target], positive_label, column=target)
    keep = y.notna()
    X = cleaned.loc[keep].drop(columns=[target, *([id_column] if id_column else [])])
    return infer_roles(X, y.loc[keep].astype(int), target=target, config=config, exclude=exclude,
                       id_columns=[id_column] if id_column else [])


def train_from_dataframe(
    df: pd.DataFrame,
    *,
    target: str | None,
    positive_label: str | None,
    id_column: str | None,
    exclude: Sequence[str],
    allow: Sequence[str],
    model_keys: Sequence[str],
    config: Config,
    source_name: str,
    progress: Callable[[str], None] | None = None,
) -> Workspace:
    """Validate the user's choices, train, and return a workspace flagged as custom."""
    if not target:
        raise DataValidationError("Choose the target column: it is never assumed to be the last one.")
    if positive_label is None:
        raise DataValidationError("Choose which target value is the positive class (churn).")
    if not model_keys:
        raise DataValidationError("Select at least one model to compare.")
    prepared = prepare_dataset(df, target, positive_label=positive_label, config=config,
                               id_column=id_column)
    result = run_experiment(prepared, config, engineer=False, exclude=exclude, allow=allow,
                            model_keys=list(model_keys), progress=progress)
    return workspace_from_experiment(result, name=f"Custom - {source_name}", source_file=source_name)
