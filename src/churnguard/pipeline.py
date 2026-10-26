"""Model registry and pipeline assembly: [engineer] -> preprocess -> classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import Config
from .preprocessing import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    TelcoFeatureEngineer,
    build_preprocessor,
)


@dataclass(frozen=True)
class ModelEntry:
    key: str
    label: str
    factory: type
    complexity: int          # used by the "one standard error" selection rule
    baseline_only: bool = False
    seeded: bool = True


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "dummy": ModelEntry("dummy", "Dummy (class prior)", DummyClassifier, 0, True, False),
    "logistic_regression": ModelEntry(
        "logistic_regression", "Logistic Regression", LogisticRegression, 1
    ),
    "decision_tree": ModelEntry("decision_tree", "Decision Tree", DecisionTreeClassifier, 2),
    "random_forest": ModelEntry("random_forest", "Random Forest", RandomForestClassifier, 3),
    "gradient_boosting": ModelEntry(
        "gradient_boosting", "Hist. Gradient Boosting", HistGradientBoostingClassifier, 4
    ),
}


@dataclass(frozen=True)
class FeatureSpec:
    """Raw input columns of the model, and whether the Telco feature engineering is applied."""

    raw_numeric: tuple[str, ...]
    raw_categorical: tuple[str, ...]
    engineer: bool = False

    @property
    def model_numeric(self) -> list[str]:
        extra = ENGINEERED_NUMERIC if self.engineer else ()
        return [*self.raw_numeric, *extra]

    @property
    def model_categorical(self) -> list[str]:
        extra = ENGINEERED_CATEGORICAL if self.engineer else ()
        return [*self.raw_categorical, *extra]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_numeric": list(self.raw_numeric),
            "raw_categorical": list(self.raw_categorical),
            "engineer": self.engineer,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeatureSpec:
        return cls(tuple(raw["raw_numeric"]), tuple(raw["raw_categorical"]), bool(raw["engineer"]))


def build_estimator(key: str, params: dict[str, Any], seed: int) -> ClassifierMixin:
    """Instantiate a registry model with explicit hyperparameters and seed."""
    try:
        entry = MODEL_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(MODEL_REGISTRY)}") from exc
    kwargs = dict(params)
    if entry.seeded:
        kwargs["random_state"] = seed
    if key == "random_forest":
        kwargs.setdefault("n_jobs", -1)
    return entry.factory(**kwargs)


def candidate_estimators(config: Config, keys: list[str] | None = None) -> dict[str, ClassifierMixin]:
    """The models compared in a run, built from the configuration."""
    seed = config.cv.random_state
    chosen = keys or list(MODEL_REGISTRY)
    return {
        key: build_estimator(key, getattr(config.models, key), seed) for key in chosen
    }


def build_pipeline(spec: FeatureSpec, estimator: ClassifierMixin) -> Pipeline:
    """[TelcoFeatureEngineer ->] ColumnTransformer -> classifier, one fit-once object."""
    engineer = TelcoFeatureEngineer() if spec.engineer else "passthrough"
    return Pipeline(
        steps=[
            ("engineer", engineer),
            ("preprocess", build_preprocessor(spec.model_numeric, spec.model_categorical)),
            ("classifier", estimator),
        ]
    )


def encoded_feature_table(pipeline: Pipeline) -> pd.DataFrame:
    """One row per model input column: encoded name, source feature, level, readable label.

    Built from the fitted encoders (not from positional assumptions), and checked against
    ``ColumnTransformer.get_feature_names_out`` by the test-suite.
    """
    pre = pipeline.named_steps["preprocess"]
    rows: list[tuple[str, str, str | None, str]] = []
    for name, transformer, columns in pre.transformers_:
        if name == "num":
            rows += [(f"num__{c}", c, None, c) for c in columns]
        elif name == "cat":
            onehot = transformer.named_steps["onehot"]
            for column, levels in zip(columns, onehot.categories_, strict=True):
                rows += [(f"cat__{column}_{lv}", column, str(lv), f"{column} = {lv}") for lv in levels]
    return pd.DataFrame(rows, columns=["encoded", "feature", "level", "label"])


def final_estimator(pipeline: Pipeline) -> ClassifierMixin:
    return pipeline.named_steps["classifier"]


def transform_to_model_input(pipeline: Pipeline, X: pd.DataFrame):
    """Raw frame -> numeric matrix actually fed to the classifier (all steps but the last)."""
    return pipeline[:-1].transform(X)
