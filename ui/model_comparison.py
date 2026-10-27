"""Model Comparison: validation scores for every model, then the single test evaluation."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churnlens.evaluation import ALL_METRICS, LOWER_IS_BETTER, calibration_summary

from .common import BLUE, GREY, NOT_A_GUARANTEE, base_layout, num, pct
from .context import AppContext

LABELS = {
    "roc_auc": "ROC-AUC", "average_precision": "Average precision", "precision": "Precision",
    "recall": "Recall", "f1": "F1", "balanced_accuracy": "Balanced accuracy", "accuracy": "Accuracy",
    "log_loss": "Log loss", "brier_score": "Brier score",
}


def render(ctx: AppContext) -> None:
    ws, meta = ctx.ws, ctx.ws.metadata
    st.title("Model Comparison")
    st.info(
        "**Where these numbers come from.** The table and charts below are **cross-validation scores on "
        f"the training split** ({meta['cv_folds']} stratified folds, identical for every model) - "
        "this is the data used to *choose* the model. The **test set** is shown separately at the "
        "bottom: it was used **once**, for the selected model only, after the choice was made."
    )
    selection = meta["selection"]
    primary = selection["primary_metric"]
    st.success(f"**Selected: {ws.bundle.label}** — {selection['explanation']}")
    st.caption(
        f"Selection criterion: mean CV *{LABELS.get(primary, primary)}* with the "
        f"`{selection['rule']}` rule. Average Precision is the default because the positive class is "
        "the minority and the goal is to rank at-risk customers for a limited campaign; it is "
        "configurable in configs/config.toml. The most complex model is not selected automatically."
    )

    comparison = ws.comparison.copy()
    table = pd.DataFrame({"Model": comparison["label"], "Selected": comparison["selected"].map({True: "✔", False: ""})})
    for metric in ALL_METRICS:
        table[LABELS[metric]] = [f"{m:.3f} ± {s:.3f}" for m, s in
                                 zip(comparison[f"{metric}_mean"], comparison[f"{metric}_std"], strict=True)]
    table["Fit time (s)"] = comparison["fit_time_s_mean"].map(lambda v: f"{v:.2f}")
    table["Score time (s)"] = comparison["score_time_s_mean"].map(lambda v: f"{v:.3f}")
    st.subheader("Cross-validation scores (mean ± standard deviation over folds)")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Precision, recall, F1, balanced accuracy and accuracy use the default 0.5 cut-off for every "
               "model in this table. Score time includes prediction and scoring of one fold. "
               "Log loss and Brier score: lower is better.")
    st.download_button("Download comparison (CSV)", comparison.to_csv(index=False).encode("utf-8"),
                       file_name="model_comparison.csv", mime="text/csv")

    metric = st.selectbox("Metric to visualise", list(ALL_METRICS), index=list(ALL_METRICS).index(primary)
                          if primary in ALL_METRICS else 1, format_func=LABELS.get, key="cmp_metric")
    _metric_chart(ws, comparison, metric)
    _calibration(ws)
    _test_evaluation(ctx)


def _metric_chart(ws, comparison: pd.DataFrame, metric: str) -> None:
    fig = go.Figure()
    folds = ws.cv_folds
    fig.add_trace(go.Bar(x=comparison["label"], y=comparison[f"{metric}_mean"], name="mean",
                         error_y=dict(type="data", array=comparison[f"{metric}_std"]),
                         marker_color=[BLUE if s else GREY for s in comparison["selected"]]))
    if folds is not None:
        label_of = dict(zip(comparison["model"], comparison["label"], strict=True))
        fig.add_trace(go.Scatter(x=folds["model"].map(label_of), y=folds[metric], mode="markers", name="each fold",
                                 marker=dict(color="black", size=6, opacity=0.6)))
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    fig.update_layout(title=f"{LABELS[metric]} by model - cross-validation ({direction})", yaxis_title=LABELS[metric])
    st.plotly_chart(base_layout(fig, height=380), width="stretch")


def _calibration(ws) -> None:
    st.subheader("Calibration of the selected model (out-of-fold predictions)")
    calibration = calibration_summary(ws.oof["y_true"], ws.oof["proba"])
    bins = calibration["bins"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="perfect"))
    fig.add_trace(go.Scatter(x=bins["mean_predicted"], y=bins["observed_rate"], mode="lines+markers",
                             marker=dict(size=bins["n"].clip(upper=800) ** 0.5 / 2 + 4), name="model", line=dict(color=BLUE)))
    fig.update_layout(title="Reliability diagram", xaxis_title="Predicted probability",
                      yaxis_title="Observed churn rate")
    left, right = st.columns([2, 1])
    left.plotly_chart(base_layout(fig, height=340), width="stretch")
    right.metric("Mean predicted", pct(calibration["mean_predicted"]))
    right.metric("Observed rate", pct(calibration["observed_rate"]))
    right.metric("Expected calibration error", num(calibration["expected_calibration_error"]))
    st.caption("Points near the diagonal mean the probabilities can be read as frequencies - a "
               "prerequisite for the retention simulator, which sums probabilities.")


def _test_evaluation(ctx: AppContext) -> None:
    ws = ctx.ws
    st.subheader("Final test-set evaluation (selected model, used once)")
    st.warning("Source: **held-out test set**. These numbers were not used to select the model, tune "
               "it, or choose the threshold. " + NOT_A_GUARANTEE)
    rows = []
    for name, thr, metrics in (
        ("Recommended threshold", ws.bundle.threshold, ws.test_metrics),
        ("Default threshold 0.5", ctx.config.threshold.default, ws.test_metrics_default),
    ):
        rows.append({"Setting": name, "Threshold": f"{thr:.2f}",
                     **{LABELS[m]: f"{metrics[m]:.3f}" for m in ALL_METRICS}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Threshold {ws.bundle.threshold:.2f}: {ws.metadata['threshold']['description']}")
    ci = ws.metadata.get("test_metrics_ci_95")
    if ci:
        st.markdown("**How much could these numbers move? (95% bootstrap interval over test customers)**")
        st.dataframe(pd.DataFrame([{"Metric": LABELS[m], "Value": f"{ws.test_metrics[m]:.3f}",
                                    "95% interval": f"{lo:.3f} to {hi:.3f}"} for m, (lo, hi) in ci.items()]),
                     hide_index=True, width="stretch")
        st.caption("Test-sample noise only (the model is not refitted). It does not cover training variability "
                   "or changes in the customer population.")
