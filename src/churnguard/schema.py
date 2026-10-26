"""The input contract of a trained model, and validation of incoming data against it.

The schema is inferred from the *training* split, saved in ``metadata.json`` and reused by the
API, the Streamlit forms and file scoring - so the contract is defined in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data import DataValidationError

_MAX_EXAMPLES = 3


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str                                 # "numeric" | "categorical"
    categories: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    baseline: float | str | None = None       # training median / mode (used by explanations)
    non_negative: bool = False                # True if the training minimum was >= 0
    integer: bool = False                     # True if all training values were whole numbers

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "categories": list(self.categories),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "baseline": self.baseline,
            "non_negative": self.non_negative,
            "integer": self.integer,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ColumnSpec:
        return cls(
            name=raw["name"],
            kind=raw["kind"],
            categories=tuple(raw.get("categories", ())),
            minimum=raw.get("minimum"),
            maximum=raw.get("maximum"),
            baseline=raw.get("baseline"),
            non_negative=bool(raw.get("non_negative", False)),
            integer=bool(raw.get("integer", False)),
        )


@dataclass(frozen=True)
class DataSchema:
    columns: tuple[ColumnSpec, ...]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric(self) -> list[str]:
        return [c.name for c in self.columns if c.kind == "numeric"]

    @property
    def categorical(self) -> list[str]:
        return [c.name for c in self.columns if c.kind == "categorical"]

    def column(self, name: str) -> ColumnSpec:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def baseline_row(self) -> dict[str, Any]:
        """One 'typical' record (training median / mode) used as the explanation baseline."""
        return {c.name: c.baseline for c in self.columns}

    def to_dict(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.columns]

    @classmethod
    def from_dict(cls, raw: list[dict[str, Any]]) -> DataSchema:
        return cls(tuple(ColumnSpec.from_dict(item) for item in raw))


def canonical_text(s: pd.Series) -> pd.Series:
    """Categorical values as stripped strings, missing values kept as NaN."""
    return s.astype(object).where(s.isna(), s.astype(str).str.strip())


def infer_schema(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> DataSchema:
    """Describe the raw input columns of the model from the training data."""
    specs: list[ColumnSpec] = []
    for col in numeric:
        s = pd.to_numeric(X[col], errors="coerce")
        non_null = s.dropna()
        is_integer = bool(np.all(np.mod(non_null, 1) == 0))
        median = float(non_null.median())
        specs.append(
            ColumnSpec(
                name=col,
                kind="numeric",
                minimum=float(non_null.min()),
                maximum=float(non_null.max()),
                # An integer column must keep an integer baseline (median of 0/1 can be 0.5).
                baseline=float(np.floor(median + 0.5)) if is_integer else median,
                non_negative=bool(non_null.min() >= 0),
                integer=is_integer,
            )
        )
    for col in categorical:
        s = canonical_text(X[col]).dropna()
        counts = s.value_counts()
        specs.append(
            ColumnSpec(
                name=col,
                kind="categorical",
                categories=tuple(sorted(counts.index.astype(str))),
                baseline=str(counts.idxmax()),
            )
        )
    return DataSchema(tuple(specs))


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    frame: pd.DataFrame | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise DataValidationError(" ".join(self.errors))


def _examples(values: pd.Series) -> str:
    return ", ".join(repr(v) for v in list(values.unique())[:_MAX_EXAMPLES])


def validate_features(
    df: pd.DataFrame, schema: DataSchema, *, strict_categories: bool = False
) -> ValidationReport:
    """Check ``df`` against the schema and return an aligned, correctly typed copy.

    Columns are matched **by name** (order is irrelevant, extra columns are reported and
    dropped). Errors block scoring; warnings do not. ``strict_categories`` turns unseen
    categories into errors (used by the API); otherwise the pipeline's one-hot encoder ignores them.
    """
    report = ValidationReport()
    if df is None or df.shape[0] == 0:
        report.errors.append("The data has no rows.")
        return report
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in schema.names if c not in df.columns]
    if missing:
        report.errors.append(
            f"Missing required column(s): {missing}. Expected columns: {schema.names}."
        )
        return report
    extra = [c for c in df.columns if c not in schema.names]
    if extra:
        report.warnings.append(f"Ignored column(s) not used by the model: {extra}.")

    out = pd.DataFrame(index=df.index)
    for spec in schema.columns:
        raw = df[spec.name]
        if spec.kind == "numeric":
            _check_numeric(spec, raw, out, report)
        else:
            _check_categorical(spec, raw, out, report, strict_categories)
    if report.ok:
        report.frame = out[schema.names]
    return report


def _check_numeric(
    spec: ColumnSpec, raw: pd.Series, out: pd.DataFrame, report: ValidationReport
) -> None:
    converted = pd.to_numeric(raw, errors="coerce")
    unparseable = raw.notna() & converted.isna()
    if unparseable.any():
        report.errors.append(
            f"Column '{spec.name}' must be numeric: {int(unparseable.sum())} value(s) are not "
            f"numbers (e.g. {_examples(raw[unparseable])})."
        )
        return
    n_missing = int(converted.isna().sum())
    if n_missing:
        report.warnings.append(
            f"Column '{spec.name}': {n_missing} missing value(s) will be imputed with the "
            "training median."
        )
    if spec.non_negative and (converted < 0).any():
        report.errors.append(
            f"Column '{spec.name}' contains negative values (e.g. "
            f"{_examples(converted[converted < 0])}) but was never negative in the training data."
        )
        return
    if spec.minimum is not None and spec.maximum is not None:
        outside = converted.notna() & ((converted < spec.minimum) | (converted > spec.maximum))
        if outside.any():
            report.warnings.append(
                f"Column '{spec.name}': {int(outside.sum())} value(s) outside the training range "
                f"[{spec.minimum:g}, {spec.maximum:g}] - the model is extrapolating."
            )
    out[spec.name] = converted.astype(float)


def _check_categorical(
    spec: ColumnSpec,
    raw: pd.Series,
    out: pd.DataFrame,
    report: ValidationReport,
    strict: bool,
) -> None:
    text = canonical_text(raw)
    n_missing = int(text.isna().sum())
    if n_missing:
        report.warnings.append(
            f"Column '{spec.name}': {n_missing} missing value(s) will be imputed with the "
            "most frequent training category."
        )
    unseen = text.notna() & ~text.isin(spec.categories)
    if unseen.any():
        message = (
            f"Column '{spec.name}': {int(unseen.sum())} value(s) not seen in training "
            f"(e.g. {_examples(text[unseen])}); known values: {list(spec.categories)}."
        )
        (report.errors if strict else report.warnings).append(message)
    out[spec.name] = text


def require_valid(df: pd.DataFrame, schema: DataSchema, **kwargs: Any) -> pd.DataFrame:
    """Validate and return the aligned frame, raising ``DataValidationError`` on errors."""
    report = validate_features(df, schema, **kwargs)
    report.raise_if_errors()
    assert report.frame is not None
    return report.frame
