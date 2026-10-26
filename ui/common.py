"""Shared UI helpers: palette, figures, formatting, standard notices."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnguard.explainability import ASSOCIATION_NOTICE, LocalExplanation, explain_prediction

SEGMENT_COLORS = {"Low": "#2e9e6b", "Medium": "#e8a33d", "High": "#d64545"}
BLUE, GREY, RED, GREEN = "#2f6fb3", "#8a94a3", "#d64545", "#2e9e6b"

NOT_A_GUARANTEE = (
    "Analysis aid, not a validated production system. Figures describe this dataset and model; "
    "they are not guarantees of future performance and do not replace business validation."
)
HISTORICAL_RATE_NOTE = (
    "Observed churn rate = what happened historically in this dataset. It is not the future risk "
    "of any customer."
)


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def money(value: float | None) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:,.0f}"


def base_layout(fig: go.Figure, height: int = 340, **kwargs) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                      **kwargs)
    return fig


def segment_bar(table: pd.DataFrame) -> go.Figure:
    counts = table["risk_segment"].astype(str).value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[SEGMENT_COLORS[s] for s in counts.index],
                           text=counts.values, textposition="outside"))
    fig.update_layout(title="Customers by risk segment", yaxis_title="Customers")
    return base_layout(fig)


def probability_histogram(proba, threshold: float, actual=None, title="Distribution of churn probability") -> go.Figure:
    fig = go.Figure()
    p = np.asarray(proba, dtype=float)
    if actual is None:
        fig.add_trace(go.Histogram(x=p, xbins=dict(start=0, end=1, size=0.05), marker_color=BLUE, name="customers"))
    else:
        y = np.asarray(actual).astype(int)
        for value, name, color in ((0, "Actually stayed", GREEN), (1, "Actually churned", RED)):
            fig.add_trace(go.Histogram(x=p[y == value], xbins=dict(start=0, end=1, size=0.05),
                                       marker_color=color, name=name, opacity=0.75))
        fig.update_layout(barmode="overlay")
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="top")
    fig.update_layout(title=title, xaxis_title="Predicted probability of churn", yaxis_title="Customers")
    return base_layout(fig)


def confusion_heatmap(cm: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm.to_numpy(), x=list(cm.columns), y=list(cm.index), colorscale="Blues",
                               showscale=False, text=cm.to_numpy(), texttemplate="%{text}",
                               textfont=dict(size=18)))
    fig.update_layout(yaxis=dict(autorange="reversed"), title="Confusion matrix (rows = actual)")
    return base_layout(fig, height=320)


def horizontal_bars(labels, values, title: str, color: str = BLUE, error=None, xlabel: str = "") -> go.Figure:
    order = np.argsort(np.abs(np.asarray(values)))
    labels, values = np.asarray(labels)[order], np.asarray(values)[order]
    colors = [color] * len(values)
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=np.asarray(error)[order]) if error is not None else None))
    fig.update_layout(title=title, xaxis_title=xlabel)
    return base_layout(fig, height=max(280, 24 * len(values) + 80))


def signed_bars(labels, values, title: str, xlabel: str) -> go.Figure:
    order = np.argsort(np.abs(np.asarray(values)))
    labels, values = np.asarray(labels)[order], np.asarray(values)[order]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker_color=[RED if v > 0 else GREEN for v in values]))
    fig.update_layout(title=title, xaxis_title=xlabel)
    return base_layout(fig, height=max(280, 26 * len(values) + 80))


def local_explanation_view(ws, row: pd.DataFrame, top: int = 10) -> LocalExplanation:
    """Render the explanation of one customer and return it."""
    explanation = explain_prediction(ws.bundle.pipeline, ws.bundle.schema, row)
    table = explanation.table.head(top)
    labels = [f"{f} = {v:g}" if isinstance(v, int | float | np.number) and pd.notna(v) else f"{f} = {v}"
              for f, v in zip(table["feature"], table["value"], strict=True)]
    st.plotly_chart(
        signed_bars(labels, table["contribution"], "What moves this customer's score "
                    "(red raises risk, green lowers it)", f"Contribution ({explanation.unit})"),
        width="stretch",
    )
    st.caption(explanation.description)
    st.caption(ASSOCIATION_NOTICE)
    return explanation


def notices(messages: list[str], expanded: bool = False) -> None:
    messages = list(dict.fromkeys(messages))
    if messages:
        with st.expander(f"{len(messages)} data / model notice(s)", expanded=expanded):
            for message in messages:
                st.warning(message)
