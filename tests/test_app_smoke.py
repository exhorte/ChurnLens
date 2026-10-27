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

needs_csv = pytest.mark.skipif(not DEMO_CSV.exists(), reason="Telco CSV not downloaded (see data/README.md)")

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


@needs_csv
def test_threshold_slider_updates_the_dashboard():
    at = open_page("Dashboard")
    key = next(k for k in at.session_state.filtered_state if str(k).startswith("threshold::"))
    before = [m.value for m in at.metric if "Predicted to churn" in m.label][0]
    at.session_state[key] = 0.90
    at.run()
    after = [m.value for m in at.metric if "Predicted to churn" in m.label][0]
    assert not at.exception
    assert int(after.replace(",", "")) < int(before.replace(",", ""))


@needs_csv
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


@needs_csv
def test_export_page_offers_a_csv_download():
    at = open_page("Export")
    assert not at.exception
    assert any("Download predictions" in b.proto.label for b in at.get("download_button"))


@needs_csv
def test_error_analysis_exposes_false_negatives_and_positives():
    at = open_page("Error Analysis")
    assert not at.exception
    labels = " ".join(m.label for m in at.metric)
    assert "False negatives" in labels and "False positives" in labels
    at.radio(key="err_kind").set_value("FP").run()
    assert not at.exception


def test_without_a_model_the_app_explains_how_to_get_one(tmp_path, monkeypatch):
    monkeypatch.setattr("churnlens.config.PROJECT_ROOT", tmp_path)
    at = streamlit_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.session_state["page"] = "Dashboard"
    at.run()
    assert not at.exception
    assert any("train_model.py" in str(m.value) for m in [*at.error, *at.info])


def test_without_the_csv_every_page_still_renders_and_explains_what_to_do(tmp_path, monkeypatch):
    """CI has the committed model but not the Telco CSV: nothing may crash, guidance must appear."""
    import shutil

    shutil.copytree(ROOT / "models", tmp_path / "models")
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    monkeypatch.setattr("churnlens.config.PROJECT_ROOT", tmp_path)      # no data/ folder there
    for page in PAGES:
        at = streamlit_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.run()
        at.session_state["page"] = page
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])
        if page == "Dashboard":
            assert any("CSV" in str(i.value) for i in [*at.info, *at.warning])


# ------------------------------------------------------- train on my own CSV, through the UI
def _upload_page(csv_bytes: bytes):
    at = open_page("Data Explorer")
    at.radio(key="explorer_source").set_value("Upload my own CSV").run()
    at.file_uploader(key="train_csv").set_value(("mine.csv", csv_bytes, "text/csv")).run()
    return at


def _synthetic_csv() -> bytes:
    from tests.conftest import make_generic_df

    return make_generic_df(400, seed=5).to_csv(index=False).encode("utf-8")


def test_the_target_is_never_guessed_in_the_upload_flow():
    at = _upload_page(_synthetic_csv())
    assert not at.exception
    assert any("never assumed to be the last one" in str(i.value) for i in at.info)
    assert not at.get("button") or all("Train" not in b.proto.label for b in at.get("button"))


def test_a_non_binary_target_gets_a_clear_message():
    at = _upload_page(_synthetic_csv())
    at.selectbox(key="train_target").set_value("plan").run()      # 3 distinct values
    assert not at.exception
    assert any("exactly 2 distinct values" in str(e.value) for e in at.error)


def test_an_invalid_file_does_not_crash_the_app():
    at = _upload_page(b"just one column\n1\n2\n")
    assert not at.exception
    assert any("single column" in str(e.value) for e in at.error)
    at = _upload_page(b"")
    assert not at.exception and any("empty" in str(e.value) for e in at.error)


def test_training_on_an_uploaded_csv_creates_a_labelled_custom_model():
    at = _upload_page(_synthetic_csv())
    at.selectbox(key="train_target").set_value("churn").run()
    at.selectbox(key="train_positive").set_value("Yes").run()
    at.multiselect(key="train_models").set_value(["dummy", "logistic_regression"]).run()
    assert any("Columns the models will use" in str(m.value) for m in at.markdown)
    at.button(key="train_go").click().run(timeout=180)
    assert not at.exception, [e.value for e in at.exception]
    custom = at.session_state["custom_ws"]
    assert custom.name == "Custom - mine.csv" and not custom.is_demo
    assert custom.bundle.metadata["target"]["positive_label"] == "Yes"
    # the whole app now works on the custom model and labels it as such
    at.session_state["page"] = "About / Model Card"
    at.run()
    assert not at.exception
    assert any("Custom model" in str(x.value) for x in [*at.info, *at.warning, *at.sidebar.warning])
