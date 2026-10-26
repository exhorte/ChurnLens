"""Loading, standardisation, profiling and role inference for tabular churn data.

Nothing here fits a model. Everything that looks at the target (positive-class detection,
leakage screening) is meant to be called on training data only; ``training.run_experiment``
does exactly that.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import Config

POSITIVE_TOKENS = frozenset(
    {"yes", "true", "1", "y", "t", "churn", "churned", "left", "cancelled", "canceled", "positive"}
)
_ID_NAME = re.compile(
    r"(^|[^a-z])(id|uuid|guid)($|[^a-z])|(customer|client).?(id|number|no)($|[^a-z])",
    re.IGNORECASE,
)
_NUMERIC_LIKE_MIN_SHARE = 0.95
_LEAKAGE_SAMPLE = 20_000
_LEAKAGE_WARN_AUC = 0.90


class DataValidationError(ValueError):
    """The data cannot be used for the requested operation. The message is user-facing."""


class DatasetNotFoundError(FileNotFoundError):
    """The demo dataset is not on disk (it is not redistributed with the repository)."""


# --------------------------------------------------------------------------- loading
def read_csv(source: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Read a CSV into a DataFrame, turning every failure into a readable error."""
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.exists():
            raise DatasetNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(
            f"The file could not be read as a CSV ({type(exc).__name__}). "
            "Check the delimiter (comma expected) and the encoding (UTF-8 expected)."
        ) from exc
    if df.shape[0] == 0:
        raise DataValidationError("The file has a header but no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        raise DataValidationError(
            "The file has a single column: check the delimiter (comma expected)."
        )
    return df


def load_demo_dataset(config: Config) -> pd.DataFrame:
    """Load the Telco demo CSV; it is *not* shipped with the repository (see data/README.md)."""
    path = config.data_path
    if not path.exists():
        raise DatasetNotFoundError(
            f"Demo dataset not found at '{path}'. It is not redistributed with this repository: "
            "follow the download instructions in data/README.md."
        )
    df = read_csv(path)
    verify_demo_dataset(df, config)
    return df


def verify_demo_dataset(df: pd.DataFrame, config: Config) -> None:
    """Refuse a placeholder / shuffled file posing as the real Telco data.

    Real data must be used for every reported result; this guard exists so that a synthetic
    stand-in can never be mistaken for it. It is a heuristic, not proof of authenticity.
    """
    id_col, target = config.data.id_column, config.data.target
    if id_col in df.columns and df[id_col].astype(str).str.startswith("ID-").mean() > 0.5:
        raise DataValidationError(
            f"'{id_col}' uses a placeholder 'ID-xxxxx' format: this does not look like the real "
            "Telco dataset (expected e.g. '7590-VHVEG')."
        )
    if {"Contract", target} <= set(df.columns):
        rates = df.groupby("Contract")[target].apply(
            lambda s: (s.astype(str).str.strip() == config.data.positive_label).mean()
        )
        if len(rates) > 1 and (rates.max() - rates.min()) < 0.10:
            raise DataValidationError(
                "No churn signal across Contract types (spread < 0.10): the file is almost "
                "certainly synthetic or shuffled, not the real Telco dataset."
            )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (order- and dtype-insensitive enough for provenance)."""
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- standardising
def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)


def standardize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Strip text, turn blanks into missing values and coerce numeric-looking text columns.

    Row-wise only (no statistic is computed from the data), so it is safe before any split.
    Returns the cleaned frame and a list of human-readable notes on what was changed.
    """
    out = df.copy()
    notes: list[str] = []
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype(int)
            continue
        if not _is_text(s):
            continue
        text = s.astype(object).where(s.isna(), s.astype(str).str.strip())
        blank = text.map(lambda v: isinstance(v, str) and v == "")
        text = text.mask(blank, np.nan)
        n_blank = int(blank.sum())
        parsed = pd.to_numeric(text, errors="coerce")
        non_null = text.notna()
        share = float(parsed[non_null].notna().mean()) if non_null.any() else 0.0
        if share >= _NUMERIC_LIKE_MIN_SHARE:
            out[col] = parsed
            bad = int((non_null & parsed.isna()).sum())
            notes.append(
                f"'{col}' is numeric stored as text: converted to numbers "
                f"({n_blank} blank and {bad} non-numeric value(s) became missing)."
            )
        else:
            out[col] = text
            if n_blank:
                notes.append(f"'{col}': {n_blank} blank value(s) treated as missing.")
    return out, notes


# --------------------------------------------------------------------------- target
@dataclass(frozen=True)
class TargetInfo:
    column: str
    positive_label: Any
    negative_label: Any
    labels_were_inferred: bool


def prepare_target(
    series: pd.Series, positive_label: Any | None = None, column: str = "target"
) -> tuple[pd.Series, TargetInfo]:
    """Map a binary target to 0/1 (1 = positive / churn). Never guesses silently.

    ``positive_label`` is compared as text so values chosen in a UI widget match numeric columns.
    Without it, the label must be recognisable (yes/true/1/churn...); otherwise the caller must
    choose, and a ``DataValidationError`` listing the classes is raised.
    """
    classes = list(pd.Series(series.dropna().unique()))
    if len(classes) != 2:
        shown = [str(c) for c in classes[:6]]
        raise DataValidationError(
            f"The target '{column}' must have exactly 2 distinct values (binary classification); "
            f"found {len(classes)}: {shown}{'...' if len(classes) > 6 else ''}."
        )
    by_text = {str(c).strip(): c for c in classes}
    inferred = False
    if positive_label is not None:
        key = str(positive_label).strip()
        if key not in by_text:
            raise DataValidationError(
                f"Positive class '{positive_label}' not found in target '{column}' "
                f"(values: {sorted(by_text)})."
            )
        positive = by_text[key]
    else:
        matches = [c for c in classes if str(c).strip().lower() in POSITIVE_TOKENS]
        if len(matches) != 1:
            raise DataValidationError(
                f"Cannot tell which class of '{column}' is the churn / positive class "
                f"(values: {sorted(by_text)}). Choose it explicitly."
            )
        positive, inferred = matches[0], True
    negative = next(c for c in classes if c is not positive and c != positive)
    y = pd.Series(pd.NA, index=series.index, dtype="Int64")
    y[series.notna()] = (series[series.notna()] == positive).astype(int)
    return y, TargetInfo(column, positive, negative, inferred)


# -------------------------------------------------------------------------- profile
@dataclass
class DataProfile:
    n_rows: int
    n_columns: int
    n_missing_cells: int
    n_duplicate_rows: int
    constant_columns: list[str]
    columns: pd.DataFrame = field(repr=False)
    target_counts: dict[str, int] | None = None
    positive_rate: float | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.n_rows,
            "columns": self.n_columns,
            "missing_cells": self.n_missing_cells,
            "duplicate_rows": self.n_duplicate_rows,
            "constant_columns": self.constant_columns,
            "positive_rate": self.positive_rate,
        }


def profile_dataframe(
    df: pd.DataFrame, target: str | None = None, positive_label: Any | None = None
) -> DataProfile:
    """Structure and quality overview of a (standardised) DataFrame."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_%": round(100 * float(s.isna().mean()), 2),
                "unique": int(s.nunique(dropna=True)),
                "example": "" if s.dropna().empty else str(s.dropna().iloc[0]),
            }
        )
    table = pd.DataFrame(rows)
    constant = table.loc[table["unique"] <= 1, "column"].tolist()
    profile = DataProfile(
        n_rows=len(df),
        n_columns=df.shape[1],
        n_missing_cells=int(df.isna().sum().sum()),
        n_duplicate_rows=int(df.duplicated().sum()),
        constant_columns=constant,
        columns=table,
    )
    if profile.n_duplicate_rows:
        profile.warnings.append(
            f"{profile.n_duplicate_rows} fully identical row(s): they may be distinct customers "
            "with identical attributes, so they are kept."
        )
    heavy = table.loc[table["missing_%"] > 10, "column"].tolist()
    if heavy:
        profile.warnings.append(f"More than 10% missing values in: {heavy}.")
    if target and target in df.columns:
        counts = df[target].value_counts(dropna=False)
        profile.target_counts = {str(k): int(v) for k, v in counts.items()}
        if positive_label is not None:
            profile.positive_rate = float((df[target].astype(str).str.strip()
                                           == str(positive_label)).mean())
    return profile


# ------------------------------------------------------------------------- roles
@dataclass
class ColumnRoles:
    numeric: list[str]
    categorical: list[str]
    ignored: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    @property
    def used(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    def table(self) -> pd.DataFrame:
        rows = [(c, "numeric", "") for c in self.numeric]
        rows += [(c, "categorical", "") for c in self.categorical]
        rows += [(c, "ignored", why) for c, why in self.ignored.items()]
        return pd.DataFrame(rows, columns=["column", "role", "reason"])


def _single_feature_auc(x: pd.Series, y: pd.Series, numeric: bool, seed: int = 0) -> float:
    """Strength of one column as a stand-alone predictor, in [0.5, 1] (cross-fitted if text)."""
    y = y.astype(int)
    if numeric:
        filled = pd.to_numeric(x, errors="coerce")
        filled = filled.fillna(filled.median())
        if filled.nunique() < 2:
            return 0.5
        auc = roc_auc_score(y, filled)
        return float(max(auc, 1 - auc))
    folds = min(5, int(y.value_counts().min()))
    if folds < 2:
        return 0.5
    scores = pd.Series(np.nan, index=x.index)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(x, y):
        rates = y.iloc[tr].groupby(x.iloc[tr].fillna("<missing>")).mean()
        scores.iloc[va] = x.iloc[va].fillna("<missing>").map(rates).fillna(y.iloc[tr].mean()).values
    auc = roc_auc_score(y, scores)
    return float(max(auc, 1 - auc))


def infer_roles(
    X: pd.DataFrame,
    y: pd.Series | None,
    *,
    target: str = "target",
    config: Config,
    exclude: Sequence[str] = (),
    id_columns: Sequence[str] = (),
    allow: Sequence[str] = (),
) -> ColumnRoles:
    """Decide which columns are numeric features, categorical features, or ignored (with reason).

    Call it on the **training** split: the leakage screen looks at the target.
    ``allow`` lists columns the user explicitly keeps despite an automatic identifier/leakage flag.
    """
    ignored: dict[str, str] = {}
    numeric: list[str] = []
    categorical: list[str] = []
    warnings: list[str] = []
    n = len(X)
    sample = X.sample(_LEAKAGE_SAMPLE, random_state=0) if n > _LEAKAGE_SAMPLE else X
    y_sample = None if y is None else y.loc[sample.index]
    target_key = target.lower()

    for col in X.columns:
        s = X[col]
        n_unique = int(s.nunique(dropna=True))
        is_num = pd.api.types.is_numeric_dtype(s)
        reason = None
        if col in exclude:
            reason = "excluded by you"
        elif col in id_columns or (_ID_NAME.search(col) and col not in allow):
            reason = "identifier (declared or recognised by name)"
        elif pd.api.types.is_datetime64_any_dtype(s):
            reason = "date/time column (not supported in this version)"
        elif n_unique <= 1:
            reason = "constant (or empty)"
        elif col not in allow:
            ratio = n_unique / max(n, 1)
            if is_num and pd.api.types.is_integer_dtype(s) and n_unique > 50 and ratio >= 0.95:
                reason = f"identifier-like integer ({n_unique} distinct values for {n} rows)"
            elif not is_num and (n_unique > config.data.max_categories or ratio > 0.5):
                reason = f"identifier-like or very high cardinality ({n_unique} distinct values)"
        if reason is None and y_sample is not None and col not in allow:
            strength = _single_feature_auc(sample[col], y_sample, is_num)
            if strength >= config.data.leakage_auc:
                reason = (
                    f"probable target leakage: this column alone separates the classes "
                    f"(AUC={strength:.3f}). Keep it only if it is known at prediction time"
                )
            elif strength >= _LEAKAGE_WARN_AUC:
                warnings.append(
                    f"'{col}' is a very strong stand-alone predictor (AUC={strength:.2f}): check "
                    "that it will be available when predicting, not recorded after the outcome."
                )
        if reason is not None:
            ignored[col] = reason
            continue
        (numeric if is_num else categorical).append(col)
        if target_key and target_key in col.lower():
            warnings.append(f"'{col}' has a name close to the target: check it is not derived from it.")
    return ColumnRoles(numeric, categorical, ignored, warnings)


# --------------------------------------------------------------------- prepared data
@dataclass
class PreparedData:
    """Standardised dataset with a validated binary target; still no train/test split."""

    frame: pd.DataFrame          # all columns except target and id, standardised
    y: pd.Series                 # int 0/1, same index as ``frame``
    ids: pd.Series | None        # customer identifiers if the dataset has one
    id_label: str
    target: TargetInfo
    profile: DataProfile
    notes: list[str]
    fingerprint: str


def prepare_dataset(
    df: pd.DataFrame,
    target: str,
    *,
    positive_label: Any | None,
    config: Config,
    id_column: str | None = None,
    random_state: int | None = None,
) -> PreparedData:
    """Validate and standardise a labelled dataset (no fitting, no split)."""
    if df is None or df.empty:
        raise DataValidationError("The dataset is empty.")
    if target not in df.columns:
        raise DataValidationError(
            f"Target column '{target}' not found. Available columns: {list(df.columns)}."
        )
    if id_column is not None and id_column not in df.columns:
        raise DataValidationError(f"Identifier column '{id_column}' not found.")
    if id_column == target:
        raise DataValidationError("The identifier column cannot be the target.")

    cleaned, notes = standardize_frame(df)
    y_all, info = prepare_target(cleaned[target], positive_label, column=target)
    keep = y_all.notna()
    if not keep.all():
        notes.append(f"{int((~keep).sum())} row(s) without a target value were dropped.")
    cleaned, y_all = cleaned.loc[keep], y_all.loc[keep].astype(int)

    if len(cleaned) < config.data.min_rows:
        raise DataValidationError(
            f"Need at least {config.data.min_rows} labelled rows; got {len(cleaned)}."
        )
    minority = int(y_all.value_counts().min())
    if minority < config.data.min_minority:
        raise DataValidationError(
            f"The rarest class has only {minority} row(s); at least {config.data.min_minority} "
            "are required for a meaningful split and cross-validation."
        )
    if len(cleaned) > config.data.max_train_rows:
        seed = config.split.random_state if random_state is None else random_state
        cleaned = cleaned.sample(config.data.max_train_rows, random_state=seed)
        y_all = y_all.loc[cleaned.index]
        notes.append(f"Dataset sampled down to {config.data.max_train_rows:,} rows to stay responsive.")

    cleaned = cleaned.reset_index(drop=True)
    y_all = y_all.reset_index(drop=True)
    ids = None
    id_label = "row_number"
    if id_column is not None:
        ids = cleaned[id_column].astype(str).reset_index(drop=True)
        id_label = id_column
        if ids.duplicated().any():
            notes.append(f"Identifier '{id_column}' is not unique ({int(ids.duplicated().sum())} repeats).")
    frame = cleaned.drop(columns=[target] + ([id_column] if id_column else []))
    profile = profile_dataframe(cleaned, target, info.positive_label)
    return PreparedData(
        frame=frame,
        y=y_all,
        ids=ids,
        id_label=id_label,
        target=info,
        profile=profile,
        notes=notes,
        fingerprint=frame_fingerprint(cleaned),
    )
