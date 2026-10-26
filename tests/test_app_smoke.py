"""Render every Streamlit page against the committed demo artefacts (skipped if they are absent)."""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = ROOT / "models" / "churn_pipeline.pkl"
DEMO_CSV = ROOT / "data" / "WA_Fn-UseC_-_Telco_Customer_Churn.csv"

pytestmark = pytest.mark.skipif(
    not ARTEFACTS.exists(), reason="demo model not trained: run `python train_model.py`"
)

PAGES = ["Dashboard", "Data Explorer", "Model Comparison", "Customer Risk", "Error Analysis",
         "Explainability", "Retention Simulator", "Export", "About / Model Card"]


def open_page(page: str):
    at = streamlit_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.session_state["page"] = page
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page):
    at = open_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title and at.title[0].value


def test_threshold_slider_updates_the_dashboard():
    at = open_page("Dashboard")
    key = next(k for k in at.session_state.filtered_state if str(k).startswith("threshold::"))
    before = [m.value for m in at.metric if "Predicted to churn" in m.label][0]
    at.session_state[key] = 0.90
    at.run()
    after = [m.value for m in at.metric if "Predicted to churn" in m.label][0]
    assert not at.exception
    assert int(after.replace(",", "")) < int(before.replace(",", ""))


def test_dashboard_states_that_observed_rate_is_historical():
    at = open_page("Dashboard")
    labels = " ".join(m.label for m in at.metric)
    assert "Observed churn rate (historical)" in labels


def test_model_comparison_separates_validation_from_test():
    at = open_page("Model Comparison")
    text = " ".join(x.value for x in [*at.info, *at.warning, *at.success, *at.subheader])
    assert "cross-validation" in text.lower() and "test" in text.lower() and "once" in text.lower()


def test_retention_page_states_it_is_not_a_forecast():
    at = open_page("Retention Simulator")
    assert any("not a forecast" in w.value.lower() for w in at.warning)
    assert not at.exception


def test_export_page_offers_a_csv_download():
    at = open_page("Export")
    assert not at.exception
    assert any("Download predictions" in b.proto.label for b in at.get("download_button"))


@pytest.mark.skipif(not DEMO_CSV.exists(), reason="Telco CSV not downloaded (see data/README.md)")
def test_error_analysis_exposes_false_negatives_and_positives():
    at = open_page("Error Analysis")
    assert not at.exception
    labels = " ".join(m.label for m in at.metric)
    assert "False negatives" in labels and "False positives" in labels
    at.radio(key="err_kind").set_value("FP").run()
    assert not at.exception


def test_without_a_model_the_app_explains_how_to_get_one(tmp_path, monkeypatch):
    monkeypatch.setattr("churnguard.config.PROJECT_ROOT", tmp_path)
    at = streamlit_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.session_state["page"] = "Dashboard"
    at.run()
    assert not at.exception
    assert any("train_model.py" in str(m.value) for m in [*at.error, *at.info])
