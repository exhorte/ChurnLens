"""Data Explorer: structure and quality of a dataset, and training on your own CSV."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from churnlens.data import DataProfile, DataValidationError, read_csv
from churnlens.pipeline import MODEL_REGISTRY

from .common import BLUE, GREEN, HISTORICAL_RATE_NOTE, RED, base_layout, notices, pct
from .context import AppContext
from .train_panel import (
    preview_roles,
    suggest_id_column,
    suggest_positive_label,
    train_from_dataframe,
)


def render(ctx: AppContext) -> None:
    st.title("Data Explorer")
    if ctx.ws is None:
        st.warning("No trained model is available yet. You can still upload a CSV below and train on it.")
        _upload(ctx)
        return
    source = st.radio("Dataset", ["Demo dataset (Telco)", "Upload my own CSV"], horizontal=True,
                      key="explorer_source")
    if source == "Demo dataset (Telco)":
        _demo(ctx)
    else:
        _upload(ctx)


# --------------------------------------------------------------------------- demo
def _demo(ctx: AppContext) -> None:
    ws = ctx.ws
    st.caption("IBM Telco customer churn sample data. Source and licence status: see About / Model Card "
               "and docs/DATA_CARD.md.")
    if not ws.is_demo:
        st.info("A custom model is active; this tab shows the demo dataset only if its CSV is present.")
    prepared = ws.prepared if ws.is_demo else _demo_prepared(ctx)
    if prepared is None:
        data = ws.metadata.get("data", {})
        st.warning("The demo CSV is not available in this installation (it is not redistributed). "
                   "See data/README.md to download it.")
        if data:
            st.write(f"The saved model was trained on {data.get('rows', '?'):,} rows "
                     f"(positive rate {pct(data.get('positive_rate'))}).")
        return
    _profile_view(prepared.profile, prepared.frame.assign(**{prepared.target.column: prepared.y}),
                  prepared.target.column, prepared.notes)


def _demo_prepared(ctx: AppContext):
    from ui.workspace import load_demo_workspace

    try:
        return load_demo_workspace(ctx.config).prepared
    except (DataValidationError, FileNotFoundError, RuntimeError):
        return None


# ------------------------------------------------------------------------- upload
def _upload(ctx: AppContext) -> None:
    st.write("Upload a CSV **with a known binary target** to explore it and train models on it. "
             "To score customers without a target, use the sidebar's *New customers* option instead.")
    uploaded = st.file_uploader("CSV file (UTF-8, comma-separated)", type=["csv"], key="train_csv")
    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return
    try:
        df = read_csv(uploaded)
    except DataValidationError as exc:
        st.error(str(exc))
        return
    st.success(f"Loaded **{df.shape[0]:,} rows × {df.shape[1]} columns** from `{uploaded.name}`.")
    st.dataframe(df.head(8), width="stretch")

    columns = list(df.columns)
    target = st.selectbox("Target column (required - what you want to predict)", [None, *columns],
                          format_func=lambda c: "— choose —" if c is None else c, key="train_target")
    if target is None:
        st.info("Choose the target column. It is never assumed to be the last one.")
        return
    values = df[target].dropna().unique().tolist()
    if len(values) != 2:
        st.error(f"The target must have exactly 2 distinct values; '{target}' has {len(values)}.")
        return
    suggested = suggest_positive_label(values)
    labels = [str(v) for v in values]
    positive = st.selectbox("Positive class (churn)", [None, *labels],
                            index=(labels.index(suggested) + 1) if suggested in labels else 0,
                            format_func=lambda v: "— choose —" if v is None else v, key="train_positive")
    if positive is None:
        st.info("Choose which value means 'churned' (the class to detect).")
        return
    id_default = suggest_id_column(df, target)
    id_options = [None, *[c for c in columns if c != target]]
    id_column = st.selectbox("Customer identifier column (optional)", id_options,
                             index=id_options.index(id_default) if id_default in id_options else 0,
                             format_func=lambda c: "(none)" if c is None else c, key="train_id")
    exclude = st.multiselect("Columns to ignore", [c for c in columns if c not in (target, id_column)],
                             key="train_exclude")

    cfg = ctx.config
    try:
        roles = preview_roles(df, target, positive, id_column, exclude, cfg)
    except DataValidationError as exc:
        st.error(str(exc))
        return
    st.markdown("**Columns the models will use (preview)**")
    st.dataframe(roles.table(), hide_index=True, width="stretch")
    for warning in roles.warnings:
        st.warning(warning)
    flagged = [c for c, why in roles.ignored.items() if "leakage" in why or "identifier" in why]
    allow = st.multiselect(
        "Keep flagged columns anyway (only if they are known at prediction time)", flagged,
        key="train_allow", help="Identifier-like and target-leaking columns are rejected by default.",
    )
    keys = st.multiselect("Models to compare", list(MODEL_REGISTRY), default=list(MODEL_REGISTRY),
                          format_func=lambda k: MODEL_REGISTRY[k].label, key="train_models",
                          help="Gradient boosting is the slowest; drop it for a quicker run.")
    if st.button("Train and compare models", type="primary", key="train_go"):
        _train(ctx, df, uploaded.name, target, positive, id_column, exclude, allow, keys)


def _train(ctx, df, name, target, positive, id_column, exclude, allow, keys) -> None:
    status = st.status("Training...", expanded=True)
    try:
        workspace = train_from_dataframe(
            df, target=target, positive_label=positive, id_column=id_column, exclude=exclude,
            allow=allow, model_keys=keys, config=ctx.config, source_name=name,
            progress=lambda message: status.write(message),
        )
    except DataValidationError as exc:
        status.update(label="Cannot train", state="error")
        st.error(str(exc))
        return
    except ValueError as exc:
        status.update(label="Training failed", state="error")
        st.error(f"Training failed: {exc}")
        return
    status.update(label="Done", state="complete", expanded=False)
    st.session_state["custom_ws"] = workspace
    st.session_state["pending_active"] = "custom"
    st.rerun()


# ----------------------------------------------------------------------- profile view
def _profile_view(profile: DataProfile, frame, target: str, notes: list[str]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile.n_rows:,}")
    c2.metric("Columns", profile.n_columns)
    c3.metric("Missing cells", f"{profile.n_missing_cells:,}")
    c4.metric("Identical rows", f"{profile.n_duplicate_rows:,}")
    notices([*notes, *profile.warnings], expanded=True)

    t1, t2, t3 = st.tabs(["Structure & quality", "Target", "Feature vs churn"])
    with t1:
        st.dataframe(profile.columns, hide_index=True, width="stretch")
        st.caption("`missing` counts true blanks after standardisation (blank text becomes missing).")
    with t2:
        if profile.positive_rate is not None:
            counts = frame[target].value_counts().sort_index()
            fig = go.Figure(go.Bar(x=["Stayed", "Churned"], y=[counts.get(0, 0), counts.get(1, 0)],
                                   marker_color=[GREEN, RED], text=[counts.get(0, 0), counts.get(1, 0)],
                                   textposition="outside"))
            fig.update_layout(title="Target distribution")
            st.plotly_chart(base_layout(fig), width="stretch")
            st.metric("Observed churn rate (historical)", pct(profile.positive_rate), help=HISTORICAL_RATE_NOTE)
    with t3:
        features = [c for c in frame.columns if c != target]
        feature = st.selectbox("Feature", features, key="explorer_feature")
        series = frame[feature]
        if series.dtype.kind in "if" and series.nunique() > 12:
            binned = series.groupby(_quantile_bins(series)).agg(["size"]).join(
                frame[target].groupby(_quantile_bins(series)).mean().rename("rate"))
            labels = [str(i) for i in binned.index]
        else:
            grouped = frame.groupby(feature, observed=True)[target].agg(["size", "mean"]).rename(columns={"mean": "rate"})
            binned, labels = grouped, [str(i) for i in grouped.index]
        fig = go.Figure(go.Bar(x=labels, y=binned["rate"] * 100, marker_color=BLUE,
                               text=[f"{r:.0f}%  (n={n})" for r, n in zip(binned["rate"] * 100, binned["size"], strict=True)]))
        fig.update_layout(title=f"Observed churn rate by {feature}", yaxis_title="Churn rate (%)")
        st.plotly_chart(base_layout(fig, height=380), width="stretch")
        st.caption("Descriptive association in historical data - not evidence that the feature causes churn.")


def _quantile_bins(series):
    import pandas as pd

    return pd.qcut(series, 8, duplicates="drop")
