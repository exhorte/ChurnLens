"""ChurnGuard AI - customer retention analysis (Streamlit).

    streamlit run app.py

Pages live in ui/, the machine-learning code in src/churnguard/ (tested without the interface).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from churnguard.config import ConfigError, load_config
from churnguard.data import DataValidationError, read_csv
from churnguard.predictions import ModelLoadError
from ui import (
    customer_risk,
    dashboard,
    data_explorer,
    error_analysis_page,
    explainability_page,
    export_page,
    model_card,
    model_comparison,
    retention_page,
)
from ui.context import AppContext
from ui.workspace import (
    POPULATION_ALL,
    POPULATION_TEST,
    POPULATION_UPLOAD,
    Workspace,
    load_demo_workspace,
)

PAGES = {
    "Dashboard": dashboard.render,
    "Data Explorer": data_explorer.render,
    "Model Comparison": model_comparison.render,
    "Customer Risk": customer_risk.render,
    "Error Analysis": error_analysis_page.render,
    "Explainability": explainability_page.render,
    "Retention Simulator": retention_page.render,
    "Export": export_page.render,
    "About / Model Card": model_card.render,
}
USES_POPULATION = {"Dashboard", "Customer Risk", "Explainability", "Retention Simulator", "Export"}
DEMO_LABEL, CUSTOM_LABEL = "Demo model (saved)", "Custom model (this session)"


@st.cache_resource(show_spinner="Loading the saved model...")
def _demo_workspace(models_dir: str, stamp: float) -> Workspace:
    """Cached per model folder and metadata.json modification time, so retraining invalidates it."""
    return load_demo_workspace(load_config())


def _demo_or_error(config) -> tuple[Workspace | None, str | None]:
    metadata = config.models_dir / "metadata.json"
    stamp = metadata.stat().st_mtime if metadata.exists() else 0.0
    try:
        return _demo_workspace(str(config.models_dir), stamp), None
    except (ModelLoadError, DataValidationError) as exc:
        return None, str(exc)


def _active_workspace(config) -> tuple[Workspace | None, str | None]:
    demo, error = _demo_or_error(config)
    custom: Workspace | None = st.session_state.get("custom_ws")
    if st.session_state.pop("pending_active", None) == "custom":
        st.session_state["active_model"] = CUSTOM_LABEL
    options = ([DEMO_LABEL] if demo else []) + ([CUSTOM_LABEL] if custom else [])
    if st.session_state.get("active_model") not in options:
        st.session_state.pop("active_model", None)
    if len(options) > 1:
        choice = st.sidebar.radio("Active model", options, key="active_model")
    else:
        choice = options[0] if options else None
    if choice == CUSTOM_LABEL:
        return custom, None
    return demo, error


def _sidebar_controls(ws: Workspace, page: str) -> tuple[float, str, pd.DataFrame | None]:
    """Decision threshold and the customers to analyse; keyed per model so switching is safe."""
    run = ws.metadata["run_id"]
    recommended = round(ws.bundle.threshold, 2)
    key = f"threshold::{run}"
    st.session_state.setdefault(key, min(max(recommended, 0.05), 0.95))
    st.sidebar.slider("Decision threshold", 0.05, 0.95, step=0.01, key=key,
                      help="Customers with probability ≥ threshold are predicted to churn.")
    st.sidebar.button("Reset to recommended", key=f"reset::{run}",
                      on_click=lambda: st.session_state.__setitem__(key, min(max(recommended, 0.05), 0.95)))
    st.sidebar.caption(f"Recommended {recommended:.2f}: {ws.metadata['threshold']['criterion'].upper()}-optimal on "
                       "out-of-fold training predictions (test set not used).")

    kind, uploaded = POPULATION_TEST, None
    if page in USES_POPULATION:
        options = ([POPULATION_TEST, POPULATION_ALL] if ws.has_features else []) + [POPULATION_UPLOAD]
        kind = st.sidebar.radio("Customers to analyse", options, key=f"population::{run}")
        if kind == POPULATION_UPLOAD:
            file = st.sidebar.file_uploader("CSV of customers (no target needed)", type=["csv"], key="score_csv")
            if file is not None:
                try:
                    uploaded = read_csv(file)
                except DataValidationError as exc:
                    st.sidebar.error(str(exc))
    return st.session_state[key], kind, uploaded


def main() -> None:
    st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")
    try:
        config = load_config()
    except ConfigError as exc:
        st.error(f"Invalid configuration: {exc}")
        st.stop()

    st.sidebar.markdown("## 🛡️ ChurnGuard AI")
    st.sidebar.caption("Customer retention intelligence - an analysis aid, not a production system.")
    ws, error = _active_workspace(config)
    page = st.sidebar.radio("Section", list(PAGES), key="page")
    st.sidebar.markdown("---")

    if ws is None:
        if page != "Data Explorer":
            st.error(error or "No model available.")
            st.info("Train the demo model with `python train_model.py` (see data/README.md for the dataset), "
                    "or upload a CSV in **Data Explorer** to train on your own data.")
            return
        PAGES[page](AppContext(None, config, 0.5, POPULATION_UPLOAD, None))
        return

    if ws.is_demo:
        st.sidebar.success("Demo model (IBM Telco sample data)")
    else:
        st.sidebar.warning("Custom model: trained in this session, not reviewed")
    threshold, kind, uploaded = _sidebar_controls(ws, page)
    PAGES[page](AppContext(ws, config, threshold, kind, uploaded))


main()
