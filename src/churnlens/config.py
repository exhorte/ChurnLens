"""Central configuration.

The dataclass defaults below are the single source of truth. ``configs/config.toml`` mirrors
them so a run can be tuned without touching code; ``tests/test_config.py`` fails if the two
drift apart.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# src/churnlens/config.py -> parents[2] is the project root (the folder holding src/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.toml"


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class PathsConfig:
    data_file: str = "data/WA_Fn-UseC_-_Telco_Customer_Churn.csv"
    models_dir: str = "models"
    reports_dir: str = "reports"


@dataclass(frozen=True)
class DataConfig:
    """Demo dataset description (IBM Telco sample data)."""

    target: str = "Churn"
    positive_label: str = "Yes"
    id_column: str = "customerID"
    min_rows: int = 100
    min_minority: int = 10
    max_train_rows: int = 50_000
    max_categories: int = 50
    leakage_auc: float = 0.98


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    random_state: int = 42


@dataclass(frozen=True)
class SelectionConfig:
    # One of: roc_auc, average_precision, f1, recall, precision, balanced_accuracy, accuracy
    primary_metric: str = "average_precision"
    # "best": highest mean. "one_se": simplest model within one standard error of the best.
    rule: str = "one_se"


@dataclass(frozen=True)
class ThresholdConfig:
    # "f1", "f2" (recall-weighted) or "youden". Computed on out-of-fold training predictions only.
    criterion: str = "f1"
    default: float = 0.5


@dataclass(frozen=True)
class SegmentConfig:
    # Business convention, adjustable: Low < low_below <= Medium < high_from <= High
    low_below: float = 0.20
    high_from: float = 0.50


@dataclass(frozen=True)
class EconomicsConfig:
    """Illustrative defaults for the retention simulator. NOT estimated from data."""

    cost_per_contact: float = 15.0
    value_per_retained_customer: float = 250.0
    success_rate: float = 0.20
    budget: float = 3000.0
    max_contacts: int = 500
    risk_threshold: float = 0.50
    top_k: int = 300


@dataclass(frozen=True)
class ModelsConfig:
    """Fixed hyperparameters (no tuning in v1). Keys are model registry keys."""

    dummy: dict[str, Any] = field(default_factory=lambda: {"strategy": "prior"})
    logistic_regression: dict[str, Any] = field(
        default_factory=lambda: {"C": 1.0, "max_iter": 2000}
    )
    decision_tree: dict[str, Any] = field(
        default_factory=lambda: {"max_depth": 5, "min_samples_leaf": 20}
    )
    random_forest: dict[str, Any] = field(
        default_factory=lambda: {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5}
    )
    gradient_boosting: dict[str, Any] = field(
        default_factory=lambda: {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
        }
    )


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)

    def resolve(self, relative: str) -> Path:
        """Resolve a project-relative path from the configuration."""
        return PROJECT_ROOT / relative

    @property
    def data_path(self) -> Path:
        return self.resolve(self.paths.data_file)

    @property
    def models_dir(self) -> Path:
        return self.resolve(self.paths.models_dir)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.paths.reports_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short hash of the configuration that can change a trained model.

        File locations and the retention-simulator defaults do not affect training, so they are left out.
        """
        relevant = {k: v for k, v in self.to_dict().items() if k not in ("paths", "economics")}
        blob = json.dumps(relevant, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _build(cls: type, values: dict[str, Any], where: str) -> Any:
    """Instantiate a (nested) frozen dataclass from a dict, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{where}]: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default):
            if not isinstance(value, dict):
                raise ConfigError(f"[{where}.{name}] must be a table.")
            kwargs[name] = _build(type(default), value, f"{where}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load a TOML configuration file; missing file or keys fall back to the defaults."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        if path:
            raise ConfigError(f"Configuration file not found: {config_path}")
        return Config()
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc
    config = _build(Config, raw, "config")
    _validate(config)
    return config


def _validate(config: Config) -> None:
    if not 0.0 < config.split.test_size < 0.5:
        raise ConfigError("split.test_size must be in (0, 0.5).")
    if config.cv.n_splits < 2:
        raise ConfigError("cv.n_splits must be >= 2.")
    if config.selection.rule not in {"best", "one_se"}:
        raise ConfigError("selection.rule must be 'best' or 'one_se'.")
    if config.threshold.criterion not in {"f1", "f2", "youden"}:
        raise ConfigError("threshold.criterion must be 'f1', 'f2' or 'youden'.")
    if not 0.0 < config.segments.low_below < config.segments.high_from < 1.0:
        raise ConfigError("segments must satisfy 0 < low_below < high_from < 1.")
