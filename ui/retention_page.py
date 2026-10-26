"""Retention Simulator: scenario comparison under assumptions you control."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from churnguard.retention_simulator import (
    Assumptions,
    SimulationInputError,
    break_even_probability,
    break_even_success_rate,
    sensitivity_to_success_rate,
    simulate_strategies,
)

from .common import BLUE, GREY, RED, base_layout, money, num, pct
from .context import AppContext


def render(ctx: AppContext) -> None:
    eco = ctx.config.economics
    st.title("Retention Simulator")
    st.warning(
        "**A scenario calculator, not a forecast.** Every result is conditional on the assumptions you "
        "enter. The churn model predicts who may leave; it does **not** measure what a campaign would "
        "change, and not every at-risk customer can be retained. The defaults below are illustrative "
        "placeholders, not estimates from data."
    )
    with st.expander("Formulas and units", expanded=False):
        st.latex(r"\text{net benefit} = V \times R \times Q - N \times C")
        st.markdown(
            "- **N** customers contacted · **C** cost per contact · **V** value of a retained customer "
            "(same currency unit as C)\n"
            "- **R** expected churners among the contacted = sum of their predicted churn probabilities\n"
            "- **Q** assumed campaign success rate (share of would-be churners retained)\n"
            "- gross benefit = V × R × Q · cost = N × C · ROI = net benefit / cost (undefined if cost is 0)\n"
            "- Only *Budget-limited* is shaped by the budget and the contact limit (customers are contacted from "
            "the highest risk down); the other strategies are shown as they are and flagged when they would "
            "exceed a limit. See docs/ECONOMIC_ASSUMPTIONS.md."
        )

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Assumptions")
        cost = st.number_input("Cost per contact (C)", 0.0, 100000.0, float(eco.cost_per_contact), 1.0, key="sim_cost")
        value = st.number_input("Value of a retained customer (V)", 0.0, 10_000_000.0,
                                float(eco.value_per_retained_customer), 10.0, key="sim_value")
        success = st.slider("Assumed campaign success rate (Q)", 0.0, 1.0, float(eco.success_rate), 0.01, key="sim_q")
        use_budget = st.checkbox("Limit total budget", value=True, key="sim_use_budget")
        budget = st.number_input("Total budget", 0.0, 100_000_000.0, float(eco.budget), 100.0,
                                 key="sim_budget", disabled=not use_budget)
        use_max = st.checkbox("Limit number of contacts", value=True, key="sim_use_max")
        max_contacts = st.number_input("Maximum customers to contact", 0, 10_000_000, int(eco.max_contacts), 10,
                                       key="sim_max", disabled=not use_max)
        risk_threshold = st.slider("Risk threshold (strategy 4)", 0.0, 1.0, float(eco.risk_threshold), 0.01,
                                   key="sim_threshold")
        top_k = st.number_input("k for 'top k by risk' (strategy 3)", 0, 10_000_000, int(eco.top_k), 10, key="sim_topk")

    population = ctx.population()
    with right:
        if population is None:
            return
        for note in population.warnings:
            st.warning(note)
        proba = population.table["probability"].to_numpy()
        st.caption(f"Scenarios are computed on {population.description} (change it in the sidebar). "
                   f"Sum of probabilities = {proba.sum():,.0f} expected churners among {len(proba):,} customers.")
        _calibration_warning(ctx, population)
        assumptions = Assumptions(cost, value, success, float(budget) if use_budget else None,
                                  int(max_contacts) if use_max else None)
        try:
            table = simulate_strategies(
                proba, assumptions, risk_threshold=risk_threshold, top_k=int(top_k),
                actual=population.actual.to_numpy() if population.has_actual else None,
            )
        except SimulationInputError as exc:
            st.error(str(exc))
            return
        p_star = break_even_probability(cost, value, success)
        if p_star is None:
            st.info("With V×Q = 0 a contact can never pay off.")
        else:
            st.info(f"**Break-even probability p\* = C / (V×Q) = {p_star:.2f}.** One more contact has a positive "
                    "expected net value only for a customer whose churn probability exceeds it. It follows "
                    "from your inputs, so it is exactly as reliable as C, V and Q.")
        _results(table, population.has_actual, assumptions)
        _charts(table)
        _sensitivity(proba, assumptions, risk_threshold, int(top_k))
        st.download_button("Download scenario table (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="retention_scenarios.csv", mime="text/csv")


def _calibration_warning(ctx: AppContext, population) -> None:
    cal = ctx.ws.metadata["calibration_out_of_fold"]
    gap = cal["mean_predicted"] - cal["observed_rate"]
    if abs(gap) > 0.03:
        st.error(f"The model's probabilities are miscalibrated on validation data (mean predicted "
                 f"{pct(cal['mean_predicted'])} vs observed {pct(cal['observed_rate'])}): expected-churner "
                 "figures are biased by roughly that gap.")


def _results(table, has_actual: bool, assumptions: Assumptions) -> None:
    shown = table.copy()
    shown["contacted"] = shown["contacted"].map(lambda v: f"{v:,}")
    shown["within_limits"] = shown["within_limits"].map({True: "yes", False: "NO"})
    for col in ("total_cost", "budget_used", "gross_benefit", "net_benefit"):
        shown[col] = shown[col].map(money)
    shown["expected_churners_targeted"] = shown["expected_churners_targeted"].map(lambda v: f"{v:,.1f}")
    shown["expected_saved"] = shown["expected_saved"].map(lambda v: f"{v:,.1f}")
    shown["roi"] = shown["roi"].map(lambda v: "n/a" if v is None or v != v else pct(v, 0))
    columns = {"strategy": "Strategy", "contacted": "Contacted", "total_cost": "Total cost",
               "expected_churners_targeted": "Expected churners targeted (R)",
               "expected_saved": "Churn avoided (Q×R)", "gross_benefit": "Gross benefit",
               "net_benefit": "Net benefit", "budget_used": "Budget used",
               "within_limits": "Within budget & contact limit?", "roi": "ROI"}
    if has_actual:
        shown["observed_churners_targeted"] = shown["observed_churners_targeted"].map(lambda v: f"{v:,}")
        columns["observed_churners_targeted"] = "Churners among contacted (historical)"
    st.subheader("Strategy comparison")
    st.dataframe(shown[list(columns)].rename(columns=columns), hide_index=True, width="stretch")
    best = table.loc[table["net_benefit"].idxmax()]
    st.caption(f"Highest estimated net benefit under the inputs entered: {best['strategy']} "
               f"({money(best['net_benefit'])}). That ranking is a property of these assumptions, not a "
               "recommendation - the defaults are illustrative, so vary the inputs and read the sensitivity "
               "chart below before drawing any conclusion.")
    rows = []
    for _, r in table.iterrows():
        q = break_even_success_rate(r["contacted"], r["expected_churners_targeted"],
                                    assumptions.cost_per_contact, assumptions.value_per_retained)
        rows.append({"Strategy": r["strategy"], "Break-even success rate Q*": "n/a" if q is None else num(q)})
    with st.expander("Break-even success rate per strategy"):
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Below Q*, the strategy loses money; above it, it gains. Q* > 1 means it cannot pay off at any success rate.")


def _charts(table) -> None:
    fig = go.Figure(go.Bar(x=table["strategy"], y=table["net_benefit"],
                           marker_color=[BLUE if v >= 0 else RED for v in table["net_benefit"]],
                           text=[money(v) for v in table["net_benefit"]], textposition="outside"))
    fig.update_layout(title="Estimated net benefit by strategy (conditional on the assumptions)",
                      yaxis_title="Net benefit")
    fig.add_hline(y=0, line_color=GREY)
    st.plotly_chart(base_layout(fig, height=360), width="stretch")


def _sensitivity(proba, assumptions, risk_threshold, top_k) -> None:
    st.subheader("How sensitive is the answer to the success rate Q?")
    long = sensitivity_to_success_rate(proba, assumptions, risk_threshold=risk_threshold, top_k=top_k)
    fig = go.Figure()
    for name, group in long.groupby("strategy", sort=False):
        fig.add_trace(go.Scatter(x=group["success_rate"], y=group["net_benefit"], mode="lines", name=name))
    fig.add_hline(y=0, line_color=GREY)
    fig.update_layout(xaxis_title="Assumed success rate (Q)", yaxis_title="Net benefit")
    st.plotly_chart(base_layout(fig, height=360), width="stretch")
    st.caption("The ranking of strategies can change with Q: if the best strategy is not stable across the "
               "plausible range of Q, no strategy should be declared the winner.")
