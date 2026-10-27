"""Dashboard: the situation at a glance."""
from __future__ import annotations

import streamlit as st

from churnlens.evaluation import SegmentRule, segment_summary

from .common import (
    HISTORICAL_RATE_NOTE,
    NOT_A_GUARANTEE,
    notices,
    num,
    pct,
    probability_histogram,
    segment_bar,
)
from .context import AppContext


def render(ctx: AppContext) -> None:
    ws, meta = ctx.ws, ctx.ws.metadata
    st.title("Dashboard")
    st.caption(NOT_A_GUARANTEE)
    if not ws.is_demo:
        st.info(
            f"**Custom model** trained in this session on `{ws.source_file or 'your CSV'}`. It is not "
            "comparable with the demo model and has had no review beyond the checks shown here."
        )
    notices(ws.warnings)

    population = ctx.population()
    if population is not None:
        notices(population.warnings, expanded=True)
        table = population.table
        flagged = int(table["predicted_class"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers analysed", f"{len(table):,}", help=population.description)
        c2.metric(f"Predicted to churn (threshold {ctx.threshold:.2f})", f"{flagged:,}",
                  delta=pct(flagged / len(table)) + " of customers", delta_color="off")
        c3.metric("Mean predicted churn probability", pct(table["probability"].mean()))
        if population.has_actual:
            c4.metric("Observed churn rate (historical)", pct(float(population.actual.mean())),
                      help=HISTORICAL_RATE_NOTE)
        else:
            c4.metric("Observed churn rate", "unknown", help="New customers have no known outcome.")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(segment_bar(table), width="stretch")
        with right:
            st.plotly_chart(
                probability_histogram(table["probability"], ctx.threshold, population.actual),
                width="stretch",
            )
        if population.has_actual:
            summary = segment_summary(population.actual, table["probability"], ws.bundle.segments)
            st.markdown("**Do the risk segments mean something?** Observed historical churn per segment:")
            st.dataframe(
                summary.assign(share=summary["share"].map(pct), mean_predicted=summary["mean_predicted"].map(pct),
                               observed_rate=summary["observed_rate"].map(pct)),
                hide_index=True, width="stretch",
            )
            st.caption(HISTORICAL_RATE_NOTE)
        seg: SegmentRule = ws.bundle.segments
        st.caption(f"Segments are a business convention: Low < {seg.low_below:.2f} <= Medium < "
                   f"{seg.high_from:.2f} <= High. The decision threshold ({ctx.threshold:.2f}) is separate.")

    st.markdown("---")
    st.subheader("Model in use")
    cv = meta["cv_scores_selected_model"]
    cal = meta["calibration_out_of_fold"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", ws.bundle.label)
    m2.metric("CV Average Precision", f"{num(cv['average_precision_mean'])} ± {num(cv['average_precision_std'])}",
              help="5-fold cross-validation on the training split (validation, not the test set).")
    m3.metric("CV ROC-AUC", f"{num(cv['roc_auc_mean'])} ± {num(cv['roc_auc_std'])}")
    m4.metric("Calibration (out-of-fold)", f"{pct(cal['mean_predicted'])} vs {pct(cal['observed_rate'])}",
              help="Mean predicted churn probability vs observed churn rate on out-of-fold predictions. "
                   "Close values mean probabilities can be read as frequencies.")
    st.caption(f"Trained {ws.bundle.trained_at} · scikit-learn "
               f"{meta['environment']['scikit_learn']} · selection: {meta['selection']['explanation']}")
    gap = abs(cal["mean_predicted"] - cal["observed_rate"])
    if gap > 0.03:
        st.warning(f"Predicted probabilities average {pct(cal['mean_predicted'])} but {pct(cal['observed_rate'])} "
                   "churned on validation data: probabilities are miscalibrated, so economic estimates "
                   "based on them will be biased.")
