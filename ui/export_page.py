"""Export: predictions as CSV, with the metadata needed to interpret them."""
from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from churnlens.predictions import build_export

from .common import NOT_A_GUARANTEE, notices
from .context import AppContext


def render(ctx: AppContext) -> None:
    ws = ctx.ws
    st.title("Export")
    st.caption(NOT_A_GUARANTEE)
    population = ctx.population()
    if population is None:
        return
    notices(population.warnings, expanded=True)
    only_flagged = st.checkbox("Only customers predicted to churn at the current threshold", value=False)
    export = build_export(population.scored, ws.bundle, id_label=ws.metadata.get("data", {}).get("id_column"))
    if only_flagged:
        export = export[export["predicted_churn"] == 1]
    st.write(f"**{len(export):,} rows** from {population.description}, decision threshold "
             f"**{ctx.threshold:.2f}**, model **{ws.bundle.label}** (trained {ws.bundle.trained_at}).")
    technical_ids = len(export) > 0 and "technical row number" in export["id_type"].iloc[0]
    if technical_ids:
        st.info("The data has no customer identifier: `customer_id` is a **technical row number** (position "
                "in the scored data), not a customer id. Keep the original file to map rows back.")
    st.dataframe(export.head(200), hide_index=True, width="stretch")
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    st.download_button("Download predictions (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"churnlens_predictions_{stamp}.csv", mime="text/csv", type="primary")
    st.caption("Columns: customer_id, id_type, churn_probability, predicted_churn, predicted_label, "
               "risk_segment, decision_threshold, model, model_trained_at, scored_at. The CSV may contain "
               "customer identifiers: handle it according to your data-protection rules.")
