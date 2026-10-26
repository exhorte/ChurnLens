"""FastAPI scoring service.

Serves the *same* saved pipeline, schema, threshold and risk segments as the Streamlit app, so
both return identical probabilities for identical inputs. The request model is generated from
the schema stored in ``models/metadata.json`` - there is no second, hand-written input contract.

    uvicorn service.main:app --reload        # then open http://127.0.0.1:8000/docs

Environment:
    CHURNGUARD_MODELS_DIR    folder holding churn_pipeline.pkl + metadata.json (default: ./models)
    CHURNGUARD_CORS_ORIGINS  comma-separated origins allowed to call the API from a browser
                             (default: none - cross-origin browser calls are refused)

Security note: the model file is a pickle. Only serve a model you trained yourself; the service
checks its SHA-256 against metadata.json (integrity, not authenticity).
"""
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, create_model

from churnguard import __version__
from churnguard.config import load_config
from churnguard.data import DataValidationError
from churnguard.predictions import ModelBundle, ModelLoadError, load_bundle, score_frame
from churnguard.schema import DataSchema

logger = logging.getLogger("churnguard.service")

MAX_BATCH = 1000
ThresholdQuery = Annotated[
    float | None,
    Query(ge=0.0, le=1.0, description="Decision threshold; defaults to the model's recommended one."),
]


class Prediction(BaseModel):
    churn_probability: float = Field(description="Calibrated-in-validation probability of churn")
    predicted_churn: bool = Field(description="probability >= decision_threshold")
    risk_segment: Literal["Low", "Medium", "High"]
    decision_threshold: float
    model: str


def build_record_model(schema: DataSchema) -> type[BaseModel]:
    """Pydantic request model generated from the training schema (single input contract)."""
    fields: dict[str, Any] = {}
    for spec in schema.columns:
        if spec.kind == "numeric":
            kind = int if spec.integer else float
            fields[spec.name] = (
                kind,
                Field(ge=0 if spec.non_negative else None, allow_inf_nan=False,
                      description=f"Seen in training: {spec.minimum:g} to {spec.maximum:g}"),
            )
        else:
            fields[spec.name] = (Literal[tuple(spec.categories)], Field())  # type: ignore[valid-type]
    example = schema.baseline_row()
    return create_model(
        "CustomerRecord",
        __config__=ConfigDict(extra="forbid", json_schema_extra={"example": example}),
        **fields,
    )


def _models_dir() -> Path:
    override = os.environ.get("CHURNGUARD_MODELS_DIR")
    return Path(override) if override else load_config().models_dir


def create_app(models_dir: str | Path | None = None) -> FastAPI:
    """Build the application; a missing/untrusted model yields 503s with a clear message."""
    bundle: ModelBundle | None = None
    load_error: str | None = None
    try:
        bundle = load_bundle(models_dir or _models_dir())
    except ModelLoadError as exc:
        load_error = str(exc)
        logger.error("Model not loaded: %s", exc)

    app = FastAPI(
        title="ChurnGuard AI scoring API",
        version=__version__,
        description=(
            "Scores customers for churn risk with the saved scikit-learn pipeline. "
            "An analysis aid: probabilities are conditional on the training data and do not "
            "replace business validation."
        ),
    )
    app.state.bundle = bundle
    app.state.load_error = load_error

    origins = [o.strip() for o in os.environ.get("CHURNGUARD_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["*"])

    record_model: type[BaseModel] = (
        build_record_model(bundle.schema) if bundle else create_model("CustomerRecord")
    )

    def require_bundle() -> ModelBundle:
        if app.state.bundle is None:
            raise HTTPException(status_code=503, detail=app.state.load_error or "Model not loaded.")
        return app.state.bundle

    def predict_records(records: list[dict[str, Any]], threshold: float | None) -> list[Prediction]:
        current = require_bundle()
        try:
            scored = score_frame(current, pd.DataFrame(records), threshold=threshold,
                                 strict_categories=True)
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return [
            Prediction(
                churn_probability=round(float(row.probability), 6),
                predicted_churn=bool(row.predicted_class),
                risk_segment=str(row.risk_segment),
                decision_threshold=scored.threshold,
                model=current.label,
            )
            for row in scored.table.itertuples()
        ]

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"service": "churnguard-ai", "docs": "/docs", "health": "/health"}

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, Any]:
        loaded = app.state.bundle is not None
        return {
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "model": app.state.bundle.label if loaded else None,
            "trained_at": app.state.bundle.trained_at if loaded else None,
            "detail": app.state.load_error,
        }

    @app.get("/model", tags=["meta"])
    def model_info() -> dict[str, Any]:
        """Model metadata: selection, validation scores, threshold, data and environment."""
        return require_bundle().metadata

    @app.get("/schema", tags=["meta"])
    def input_schema() -> list[dict[str, Any]]:
        """The input contract: columns, types, allowed categories and training ranges."""
        return require_bundle().schema.to_dict()

    @app.post("/predict", response_model=Prediction, tags=["scoring"])
    def predict(customer: record_model, threshold: ThresholdQuery = None) -> Prediction:  # type: ignore[valid-type]
        """Score a single customer."""
        return predict_records([customer.model_dump()], threshold)[0]

    @app.post("/predict/batch", response_model=list[Prediction], tags=["scoring"])
    def predict_batch(
        customers: list[record_model],  # type: ignore[valid-type]
        threshold: ThresholdQuery = None,
    ) -> list[Prediction]:
        """Score up to 1000 customers in one request."""
        if not customers:
            return []
        if len(customers) > MAX_BATCH:
            raise HTTPException(status_code=413, detail=f"Batch too large (max {MAX_BATCH}).")
        return predict_records([c.model_dump() for c in customers], threshold)

    return app


app = create_app()
