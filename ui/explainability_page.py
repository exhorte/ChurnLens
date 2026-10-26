"""Explainability: what the model relies on (global) and why one customer scored as they did (local)."""
from __future__ import annotations

import streamlit as st

from churnguard.data import DataValidationError
from churnguard.explainability import ASSOCIATION_NOTICE, is_linear, native_importance

from .common import horizontal_bars, local_explanation_view, signed_bars
from .context import AppContext
from .customer_risk import select_customer


def render(ctx: AppContext) -> None:
    ws = ctx.ws
    st.title("Explainability")
    st.warning(ASSOCIATION_NOTICE)
    global_tab, local_tab = st.tabs(["Global: what the model relies on", "Local: one customer"])
    with global_tab:
        _global(ctx)
    with local_tab:
        _local(ctx)
    st.caption(f"Explanations are computed from the deployed pipeline itself ({ws.bundle.label}), not from a "
               "separate surrogate model.")


def _global(ctx: AppContext) -> None:
    ws = ctx.ws
    pipeline = ws.bundle.pipeline
    native = native_importance(pipeline)
    if native is None:
        st.info(f"{ws.bundle.label} has no built-in importance; use the permutation importance below.")
    elif is_linear(pipeline):
        st.subheader("Logistic regression coefficients")
        top = native.head(15)
        st.plotly_chart(
            signed_bars(top["label"], top["value"], "Largest coefficients (red raises churn log-odds)",
                        "Coefficient (log-odds)"),
            width="stretch",
        )
        st.dataframe(
            top[["label", "value", "odds_ratio"]].rename(columns={"label": "input", "value": "coefficient"}),
            hide_index=True, width="stretch",
        )
        st.caption("Numeric inputs are standardised: a coefficient is the change in log-odds for +1 standard "
                   "deviation. Categorical inputs: the effect of the category being present, relative to the "
                   "model's other categories. Odds ratio = exp(coefficient). Correlated inputs share effect "
                   "between them - read direction and rough size, not exact values.")
    else:
        st.subheader("Built-in importance")
        top = native.head(15)
        st.plotly_chart(horizontal_bars(top["label"], top["value"], "Impurity-based importance", xlabel="importance"),
                        width="stretch")
        st.caption("Impurity importance favours continuous and many-valued features and is computed on the "
                   "training data: prefer the permutation importance below to rank features.")

    st.subheader("Permutation importance (test set)")
    if ws.permutation is None:
        st.info("Permutation importance is unavailable for this model.")
        return
    perm = ws.permutation.head(15)
    st.plotly_chart(
        horizontal_bars(perm["feature"], perm["importance_mean"], "Drop in score when a feature is shuffled",
                        error=perm["importance_std"], xlabel="mean drop (± std over repeats)"),
        width="stretch",
    )
    st.caption("Computed on the held-out test set, after model selection, on the raw input columns. "
               "A feature near zero is not used much *by this model*; strongly correlated features can "
               "mask each other.")


def _local(ctx: AppContext) -> None:
    ws = ctx.ws
    population = ctx.population()
    if population is None:
        return
    picked = select_customer(population, key="explain_customer")
    if picked is None:
        return
    pos, _ = picked
    row = population.scored.table.loc[pos]
    st.metric("Churn probability", f"{100 * row['probability']:.1f}%",
              delta=f"{'above' if row['predicted_class'] else 'below'} the {ctx.threshold:.2f} threshold",
              delta_color="inverse" if row["predicted_class"] else "normal")
    try:
        explanation = local_explanation_view(ws, population.X.iloc[[pos]], top=12)
    except (DataValidationError, ValueError) as exc:
        st.error(f"Could not explain this customer: {exc}")
        return
    with st.expander("Details"):
        shown = explanation.table.astype({"value": str, "reference": str})   # mixed numeric/text values
        st.dataframe(shown, hide_index=True, width="stretch")
        if explanation.logit is not None:
            st.write(f"intercept {explanation.base_value:+.3f} + contributions "
                     f"{explanation.table['contribution'].sum():+.3f} = log-odds {explanation.logit:+.3f}")
        else:
            st.write(f"Probability of a 'typical' customer (training median/mode for every feature): "
                     f"{100 * explanation.base_value:.1f}%")
