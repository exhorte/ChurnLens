"""Customer Risk: customers ranked by predicted risk, with a per-customer explanation."""
from __future__ import annotations

import streamlit as st

from churnguard.data import DataValidationError

from .common import NOT_A_GUARANTEE, local_explanation_view, notices, pct
from .context import AppContext

MAX_ROWS = 500


def select_customer(population, key: str):
    """Customer picker shared with the Explainability page. Returns (position, label) or None."""
    ranked = population.table.sort_values("probability", ascending=False)
    ids = population.ids
    options = list(ranked.index[:300])

    def label(pos: int) -> str:
        who = ids.iloc[pos] if ids is not None else f"row {pos + 1}"
        return f"{who} — {pct(ranked.loc[pos, 'probability'])}"

    chosen = st.selectbox("Customer (top 300 by risk shown)", options, format_func=label, key=key)
    return None if chosen is None else (chosen, label(chosen))


def render(ctx: AppContext) -> None:
    ws = ctx.ws
    st.title("Customer Risk")
    st.caption(NOT_A_GUARANTEE)
    population = ctx.population()
    if population is None:
        return
    notices(population.warnings, expanded=True)
    table = population.table.copy()          # already carries customer_id when the data has one
    if population.has_actual:
        table["observed_churn (historical)"] = population.actual.map({1: "churned", 0: "stayed"}).to_numpy()

    st.caption(f"{population.description} · decision threshold {ctx.threshold:.2f}")
    f1, f2, f3 = st.columns([2, 2, 2])
    segments = f1.multiselect("Risk segment", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    min_p = f2.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.01)
    only_flagged = f3.checkbox("Only customers predicted to churn", value=False)
    search = st.text_input("Search customer id", "") if "customer_id" in table.columns else ""

    mask = table["risk_segment"].astype(str).isin(segments) & (table["probability"] >= min_p)
    if only_flagged:
        mask &= table["predicted_class"] == 1
    if search:
        mask &= table["customer_id"].astype(str).str.contains(search, case=False, regex=False)
    view = table[mask].sort_values("probability", ascending=False)

    feature_names = ws.bundle.schema.names
    importance = ws.permutation["feature"].tolist() if ws.permutation is not None else feature_names
    show = st.multiselect("Feature columns to display", feature_names,
                          default=[f for f in importance if f in feature_names][:4], key="risk_cols")
    display = view.join(population.X[show]).head(MAX_ROWS).copy()
    display["risk_segment"] = display["risk_segment"].astype(str)
    st.dataframe(
        display, hide_index=True, width="stretch",
        column_config={"probability": st.column_config.ProgressColumn(
            "churn probability", min_value=0.0, max_value=1.0, format="%.2f")},
    )
    st.caption(f"{len(view):,} customers match the filters; showing the {min(len(view), MAX_ROWS)} riskiest. "
               "Probability = model score; predicted class = probability ≥ threshold.")

    st.markdown("---")
    st.subheader("One customer")
    if view.empty:
        st.info("No customer matches the filters.")
        return
    picked = select_customer(_subset(population, view.index), key="risk_customer")
    if picked is None:
        return
    pos, _ = picked
    c1, c2, c3 = st.columns(3)
    row = population.scored.table.loc[pos]
    c1.metric("Churn probability", pct(float(row["probability"])))
    c2.metric(f"Predicted class (threshold {ctx.threshold:.2f})", "will churn" if row["predicted_class"] else "will stay")
    c3.metric("Risk segment", str(row["risk_segment"]))
    if population.has_actual:
        st.write(f"Observed outcome (historical): **{'churned' if population.actual.iloc[pos] else 'stayed'}**")
    try:
        local_explanation_view(ws, population.X.iloc[[pos]])
    except (DataValidationError, ValueError) as exc:
        st.error(f"Could not explain this customer: {exc}")


class _Subset:
    """A view of a population restricted to some positions (for the customer picker)."""

    def __init__(self, population, index):
        self.table = population.table.loc[index]
        self.ids = population.ids


def _subset(population, index) -> _Subset:
    return _Subset(population, index)
