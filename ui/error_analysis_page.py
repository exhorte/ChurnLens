"""Error Analysis: false positives, false negatives, and what the decision threshold changes."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from churnguard.error_analysis import (
    OUTCOMES,
    classify_outcomes,
    compare_groups,
    confusion_frame,
    error_metrics,
    error_profile,
)
from churnguard.evaluation import threshold_table

from .common import BLUE, GREEN, GREY, NOT_A_GUARANTEE, RED, base_layout, confusion_heatmap, pct, probability_histogram
from .context import AppContext
from .workspace import POPULATION_TEST


def render(ctx: AppContext) -> None:
    ws = ctx.ws
    st.title("Error Analysis")
    st.warning(
        "**Held-out test customers only.** This page diagnoses the already-selected model on data it "
        "never saw. Do **not** use it to re-pick the model or the threshold (that would leak the test "
        "set into the decision), and do not read it as a guarantee of future performance."
    )
    test = ws.test_predictions
    outcomes = classify_outcomes(test["y_true"], test["proba"], ctx.threshold)
    metrics = error_metrics(outcomes)

    st.subheader(f"At decision threshold {ctx.threshold:.2f}")
    left, right = st.columns([1, 1])
    left.plotly_chart(confusion_heatmap(confusion_frame(outcomes)), width="stretch")
    with right:
        a, b = st.columns(2)
        a.metric("True positives", metrics["tp"], help=OUTCOMES["TP"])
        b.metric("False positives", metrics["fp"], help=OUTCOMES["FP"])
        a.metric("False negatives", metrics["fn"], help=OUTCOMES["FN"])
        b.metric("True negatives", metrics["tn"], help=OUTCOMES["TN"])
        c, d = st.columns(2)
        c.metric("Precision", pct(metrics["precision"]), help="Of customers flagged, the share that did churn.")
        d.metric("Recall", pct(metrics["recall"]), help="Of customers who churned, the share the model flagged.")
        c.metric("False-negative rate", pct(metrics["false_negative_rate"]), help="Churners the model missed.")
        d.metric("False-positive rate", pct(metrics["false_positive_rate"]), help="Stayers wrongly flagged.")
    st.caption("Four things are kept apart: the **probability** (model score), the **threshold** "
               "(decision cut-off), the **predicted class** (probability ≥ threshold) and the "
               "**actual class** (what happened).")
    st.plotly_chart(probability_histogram(test["proba"], ctx.threshold, test["y_true"],
                                          "Scores by actual outcome (overlap = unavoidable errors)"),
                    width="stretch")

    _threshold_analysis(ctx)
    _explore_errors(ctx, outcomes)


def _threshold_analysis(ctx: AppContext) -> None:
    ws = ctx.ws
    st.markdown("---")
    st.subheader("Threshold analysis")
    st.write(f"Recommended threshold: **{ws.bundle.threshold:.2f}** — {ws.metadata['threshold']['description']}")
    cost = st.number_input("Cost per contact (for the campaign-cost column)", min_value=0.0,
                           value=float(ctx.config.economics.cost_per_contact), step=1.0, key="thr_cost")
    show_test = st.checkbox("Also show the same table on the test set (diagnostic only)", value=False)

    def build(y, p):
        table = threshold_table(y, p)
        table["campaign_cost"] = table["flagged"] * cost
        return table

    validation = build(ws.oof["y_true"], ws.oof["proba"])
    st.markdown("**Validation data (out-of-fold predictions on the training split)** - the basis for choosing a threshold")
    _curves(validation, ws.bundle.threshold, ctx.threshold)
    _table(validation)
    if show_test:
        st.markdown("**Test set** - what the same thresholds would have produced on unseen customers")
        _table(build(ws.test_predictions["y_true"], ws.test_predictions["proba"]))
    st.caption("`flagged` = customers contacted. Raising the threshold flags fewer customers: fewer false "
               "positives (cheaper campaign) but more false negatives (missed churners). The right trade-off "
               "depends on contact cost and customer value - see Retention Simulator.")


def _curves(table, recommended: float, current: float) -> None:
    fig = go.Figure()
    for column, name, color in (("precision", "Precision", BLUE), ("recall", "Recall", RED), ("f1", "F1", GREEN),
                                ("flagged_share", "Share flagged", GREY)):
        fig.add_trace(go.Scatter(x=table["threshold"], y=table[column], mode="lines+markers", name=name,
                                 line=dict(color=color)))
    fig.add_vline(x=recommended, line_dash="dot", line_color="black", annotation_text="recommended")
    if abs(current - recommended) > 1e-9:
        fig.add_vline(x=current, line_dash="dash", line_color=RED, annotation_text="selected",
                      annotation_position="bottom right")
    fig.update_layout(xaxis_title="Decision threshold", yaxis_title="Value", yaxis_range=[0, 1])
    st.plotly_chart(base_layout(fig, height=340), width="stretch")


def _table(table) -> None:
    shown = table[["threshold", "flagged", "flagged_share", "fp", "fn", "tp", "precision", "recall", "f1",
                   "campaign_cost"]].copy()
    shown["flagged_share"] = shown["flagged_share"].map(pct)
    for col in ("precision", "recall", "f1"):
        shown[col] = shown[col].map(lambda v: "n/a" if v != v else f"{v:.3f}")
    shown["campaign_cost"] = shown["campaign_cost"].map(lambda v: f"{v:,.0f}")
    st.dataframe(shown.rename(columns={"fp": "false positives", "fn": "false negatives", "tp": "true positives"}),
                 hide_index=True, width="stretch")


def _explore_errors(ctx: AppContext, outcomes) -> None:
    st.markdown("---")
    st.subheader("Who does the model get wrong?")
    population = ctx.population(POPULATION_TEST, show_errors=False)
    if population is None:
        st.info("Customer features are unavailable (the training CSV is not loaded), so only aggregate "
                "figures can be shown. See data/README.md.")
        return
    kind = st.radio("Errors to examine", ["FN", "FP"], horizontal=True, key="err_kind",
                    format_func=lambda k: {"FN": "False negatives - churners the model missed",
                                           "FP": "False positives - stayers wrongly flagged"}[k])
    selected = outcomes["outcome"] == kind
    reference = "TP" if kind == "FN" else "TN"
    st.write(f"**{int(selected.sum())}** customers, compared with **{int((outcomes['outcome'] == reference).sum())}** "
             f"{'true positives (caught churners)' if kind == 'FN' else 'true negatives (correctly left alone)'}.")
    profile = error_profile(population.X, outcomes, kind)
    if profile.empty:
        st.info("Not enough customers in both groups to compare.")
        return
    st.markdown("**Features that differ most** (numeric: standardised mean difference; categorical: largest "
                "share gap in points/100 - not directly comparable across kinds)")
    st.dataframe(profile.head(8).assign(effect_size=lambda d: d["effect_size"].round(2)), hide_index=True, width="stretch")
    feature = st.selectbox("Compare a feature in detail", population.X.columns, key="err_feature")
    detail = compare_groups(population.X, selected.reset_index(drop=True), (outcomes["outcome"] == reference).reset_index(drop=True), feature)
    st.dataframe(detail.round(2), width="stretch")

    st.markdown("**The customers themselves** (most confidently wrong first)")
    cases = population.table.join(population.X).assign(outcome=outcomes["outcome"].astype(str).to_numpy())
    cases = cases[cases["outcome"] == kind].sort_values("probability", ascending=(kind == "FN"))
    cases["risk_segment"] = cases["risk_segment"].astype(str)
    st.dataframe(cases.drop(columns=["outcome"]).head(200), hide_index=True, width="stretch")
    st.caption("Associations only: these differences do not explain why the model erred in a causal sense.")
