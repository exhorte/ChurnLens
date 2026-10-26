"""
FastAPI scoring service for customer churn.

Loads the single serialized pipeline (``models/churn_pipeline.pkl``) once at
startup and exposes JSON endpoints. The same fitted object the Streamlit app and
the training script use is reused here, so scoring is identical everywhere.

Run locally:
    uvicorn service.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make the project root importable so joblib can rebuild the custom transformer.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import FeatureEngineer  # noqa: E402,F401  (needed for unpickling)

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = ROOT / "models" / "churn_pipeline.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"

MAX_BATCH = 1000
DECISION_THRESHOLD = 0.5


def _load_artifacts():
    """Load the pipeline and metadata; tolerate a missing/empty metadata file."""
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())
    return pipeline, metadata


try:
    PIPELINE, METADATA = _load_artifacts()
except FileNotFoundError:
    PIPELINE, METADATA = None, {}


app = FastAPI(
    title="Customer Churn Prediction API",
    version="2.0.0",
    description="Score telecom customers for churn risk using the trained pipeline.",
)

# Permissive CORS for the demo; restrict `allow_origins` in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Customer(BaseModel):
    """One customer record, validated against the dataset's allowed values."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
            }
        }
    }


class Prediction(BaseModel):
    churn: bool
    churn_probability: float
    risk: Literal["low", "medium", "high"]


def _risk_band(p: float) -> str:
    if p < 0.3:
        return "low"
    if p < 0.7:
        return "medium"
    return "high"


def _score(df: pd.DataFrame) -> list[Prediction]:
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py.")
    proba = PIPELINE.predict_proba(df)[:, 1]
    return [
        Prediction(
            churn=bool(p >= DECISION_THRESHOLD),
            churn_probability=round(float(p), 4),
            risk=_risk_band(float(p)),
        )
        for p in proba
    ]


@app.get("/", tags=["meta"])
def root():
    return {"service": "customer-churn-prediction", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "model_loaded": PIPELINE is not None,
        "model": METADATA.get("model"),
    }


@app.get("/model", tags=["meta"])
def model_info():
    if not METADATA:
        raise HTTPException(status_code=404, detail="Model metadata not available.")
    return METADATA


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict(customer: Customer):
    """Score a single customer."""
    df = pd.DataFrame([customer.model_dump()])
    return _score(df)[0]


@app.post("/predict/batch", response_model=list[Prediction], tags=["scoring"])
def predict_batch(customers: list[Customer]):
    """Score up to 1000 customers in one request."""
    if not customers:
        return []
    if len(customers) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Batch too large (max {MAX_BATCH}).")
    df = pd.DataFrame([c.model_dump() for c in customers])
    return _score(df)
