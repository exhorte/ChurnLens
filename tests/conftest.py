"""Shared fixtures.

IMPORTANT: every dataset generated here is SYNTHETIC. It exists only to test code paths
(validation, leakage guards, formulas, round-trips). No test uses it to claim anything about the
performance of the real model; real-data tests are marked and skipped when the CSV is absent.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnlens.config import Config, ModelsConfig, load_config
from churnlens.data import PreparedData, prepare_dataset
from churnlens.preprocessing import CONTRACT_MONTHS, SERVICE_COLUMNS
from churnlens.training import ExperimentResult, run_experiment


def make_generic_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC generic churn-like data: churn depends on tenure and plan; `noise` is pure noise."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2])
    monthly = np.round(rng.normal(60, 20, n).clip(15, 120), 2)
    noise = rng.normal(0, 1, n)
    logit = -0.5 - 0.05 * (tenure - 30) + np.where(plan == "basic", 0.9, np.where(plan == "plus", 0.0, -0.8))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "tenure": tenure,
            "monthly_charges": monthly,
            "plan": plan,
            "noise": noise,
            "churn": np.where(churn, "Yes", "No"),
        }
    )


def make_telco_like_df(n: int = 700, seed: int = 1) -> pd.DataFrame:
    """SYNTHETIC data using the Telco column names, for the Telco-specific code paths."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(list(CONTRACT_MONTHS), n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(0, 73, n)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    monthly = np.round(rng.uniform(18, 119, n), 2)
    df = pd.DataFrame(
        {
            "customerID": [f"{1000 + i}-SYNTH" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                 "Credit card (automatic)"], n),
            "MonthlyCharges": monthly,
        }
    )
    for col in SERVICE_COLUMNS:
        df[col] = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
    total = np.round(tenure * monthly, 2).astype(object)
    total[tenure == 0] = " "          # blank exactly when tenure == 0, like the real file
    df["TotalCharges"] = total
    month_to_month = np.where(contract == "Month-to-month", 1.0, -0.5)
    logit = -0.6 + 0.9 * month_to_month - 0.04 * (tenure - 30)
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


@pytest.fixture(scope="session")
def fast_config() -> Config:
    """Real defaults, with lighter ensembles so the whole suite stays fast."""
    base = load_config()
    models = ModelsConfig(
        random_forest={"n_estimators": 25, "max_depth": 6, "min_samples_leaf": 5},
        gradient_boosting={"max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 8},
    )
    return replace(base, models=models)


@pytest.fixture(scope="session")
def generic_df() -> pd.DataFrame:
    return make_generic_df()


@pytest.fixture(scope="session")
def telco_like_df() -> pd.DataFrame:
    return make_telco_like_df()


@pytest.fixture(scope="session")
def generic_prepared(generic_df: pd.DataFrame, fast_config: Config) -> PreparedData:
    return prepare_dataset(
        generic_df, "churn", positive_label="Yes", id_column="customer_id", config=fast_config
    )


@pytest.fixture(scope="session")
def generic_experiment(generic_prepared: PreparedData, fast_config: Config) -> ExperimentResult:
    return run_experiment(generic_prepared, fast_config)


@pytest.fixture(scope="session")
def telco_prepared(telco_like_df: pd.DataFrame, fast_config: Config) -> PreparedData:
    return prepare_dataset(
        telco_like_df, "Churn", positive_label="Yes", id_column="customerID", config=fast_config
    )


@pytest.fixture(scope="session")
def telco_experiment(telco_prepared: PreparedData, fast_config: Config) -> ExperimentResult:
    return run_experiment(telco_prepared, fast_config, engineer=True)
