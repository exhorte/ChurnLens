"""Loading a saved model and scoring new customers - the single entry point for app and API.

``joblib.load`` unpickles arbitrary code: only load model files you produced yourself. The
loader verifies the SHA-256 recorded in ``metadata.json`` and the scikit-learn version, which
catches accidental corruption and version drift. It is an integrity check, **not** authentication:
someone who can replace both files can replace the hash too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from .data import file_sha256
from .evaluation import SegmentRule, assign_segments
from .schema import DataSchema, ValidationReport, validate_features
from .training import METADATA_FILE, PIPELINE_FILE


class ModelLoadError(RuntimeError):
    """The saved model cannot be loaded safely. The message tells the user what to do."""


@dataclass
class ModelBundle:
    """A fitted pipeline together with everything needed to use it correctly."""

    pipeline: Pipeline = field(repr=False)
    metadata: dict[str, Any]
    schema: DataSchema
    threshold: float
    segments: SegmentRule

    @property
    def label(self) -> str:
        return str(self.metadata.get("model", "unknown model"))

    @property
    def positive_label(self) -> str:
        return str(self.metadata.get("target", {}).get("positive_label", "1"))

    @property
    def trained_at(self) -> str:
        return str(self.metadata.get("trained_at", "unknown"))


def bundle_from_experiment(result) -> ModelBundle:
    """Wrap an in-memory ``ExperimentResult`` (e.g. a model trained from the app)."""
    return ModelBundle(
        pipeline=result.pipeline,
        metadata=result.metadata,
        schema=result.schema,
        threshold=result.threshold.threshold,
        segments=result.segments,
    )


def load_bundle(models_dir: str | Path, *, verify: bool = True) -> ModelBundle:
    """Load ``churn_pipeline.pkl`` + ``metadata.json`` from ``models_dir``."""
    models_dir = Path(models_dir)
    pipeline_path = models_dir / PIPELINE_FILE
    metadata_path = models_dir / METADATA_FILE
    if not pipeline_path.exists() or not metadata_path.exists():
        raise ModelLoadError(
            f"No trained model in '{models_dir}'. Run `python train_model.py` first."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"{metadata_path.name} is not valid JSON: {exc}") from exc

    if verify:
        expected = metadata.get("artifact", {}).get("sha256")
        if not expected:
            raise ModelLoadError(
                f"{metadata_path.name} has no artifact hash: retrain with `python train_model.py`."
            )
        if file_sha256(pipeline_path) != expected:
            raise ModelLoadError(
                f"{PIPELINE_FILE} does not match the hash in {METADATA_FILE}: the file was "
                "modified or corrupted. Refusing to load it; retrain with `python train_model.py`."
            )
    trained_with = metadata.get("environment", {}).get("scikit_learn")
    if trained_with and trained_with != sklearn.__version__:
        raise ModelLoadError(
            f"The model was trained with scikit-learn {trained_with} but {sklearn.__version__} "
            "is installed. Install the pinned version (requirements.txt) or retrain."
        )
    try:
        pipeline = joblib.load(pipeline_path)
        schema = DataSchema.from_dict(metadata["schema"])
        threshold = float(metadata["threshold"]["threshold"])
        segments = SegmentRule(**metadata["segments"])
    except (KeyError, TypeError, ValueError, AttributeError, ImportError, EOFError) as exc:
        raise ModelLoadError(
            f"The saved model could not be restored ({type(exc).__name__}: {exc}). "
            "Retrain with `python train_model.py`."
        ) from exc
    return ModelBundle(pipeline, metadata, schema, threshold, segments)


@dataclass
class ScoredFrame:
    table: pd.DataFrame              # one row per input row, in input order
    report: ValidationReport
    threshold: float


def score_frame(
    bundle: ModelBundle,
    df: pd.DataFrame,
    *,
    ids: pd.Series | None = None,
    threshold: float | None = None,
    strict_categories: bool = False,
) -> ScoredFrame:
    """Validate ``df`` against the model schema and score it with the saved pipeline.

    Returns probability, predicted class, risk segment and the threshold used. Raises
    ``DataValidationError`` (user-readable) when the data cannot be scored.
    """
    report = validate_features(df, bundle.schema, strict_categories=strict_categories)
    report.raise_if_errors()
    assert report.frame is not None
    used = bundle.threshold if threshold is None else float(threshold)
    if not 0.0 <= used <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    proba = bundle.pipeline.predict_proba(report.frame)[:, 1]
    table = pd.DataFrame(
        {
            "probability": proba,
            "predicted_class": (proba >= used).astype(int),
            "risk_segment": assign_segments(proba, bundle.segments),
        },
        index=df.index,
    )
    if ids is not None:
        table.insert(0, "customer_id", ids.loc[df.index].to_numpy())
    return ScoredFrame(table=table, report=report, threshold=used)


def build_export(
    scored: ScoredFrame,
    bundle: ModelBundle,
    *,
    id_label: str | None = None,
) -> pd.DataFrame:
    """CSV-ready predictions with the metadata a reader needs to interpret them.

    When the data has no identifier, ``customer_id`` is a *technical row number* (1-based
    position in the scored data) and ``id_type`` says so - it is never presented as a customer id.
    """
    table = scored.table.reset_index(drop=True)
    has_id = "customer_id" in table.columns
    out = pd.DataFrame()
    out["customer_id"] = table["customer_id"] if has_id else np.arange(1, len(table) + 1)
    out["id_type"] = (
        f"customer identifier ({id_label})" if has_id and id_label
        else "customer identifier" if has_id
        else "technical row number (the data has no identifier)"
    )
    out["churn_probability"] = table["probability"].round(6)
    out["predicted_churn"] = table["predicted_class"]
    out["predicted_label"] = np.where(
        table["predicted_class"] == 1, bundle.positive_label, "not " + bundle.positive_label
    )
    out["risk_segment"] = table["risk_segment"].astype(str)
    out["decision_threshold"] = scored.threshold
    out["model"] = bundle.label
    out["model_trained_at"] = bundle.trained_at
    out["scored_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return out.sort_values("churn_probability", ascending=False, kind="stable").reset_index(drop=True)
