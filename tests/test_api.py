"""API tests on a model trained from SYNTHETIC Telco-like data (not real performance claims)."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churnguard.predictions import load_bundle, score_frame
from churnguard.training import PIPELINE_FILE, save_experiment
from service.main import MAX_BATCH, create_app


@pytest.fixture(scope="module")
def models_dir(telco_experiment, tmp_path_factory):
    folder = tmp_path_factory.mktemp("api_model")
    save_experiment(telco_experiment, folder / "models", folder / "reports")
    return folder / "models"


@pytest.fixture(scope="module")
def client(models_dir):
    return TestClient(create_app(models_dir))


@pytest.fixture(scope="module")
def bundle(models_dir):
    return load_bundle(models_dir)


@pytest.fixture
def record(bundle):
    return dict(bundle.schema.baseline_row())


def test_health_and_metadata(client, bundle):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model_loaded"] and health["model"] == bundle.label
    meta = client.get("/model").json()
    assert meta["run_id"] == bundle.metadata["run_id"] and meta["threshold"]["criterion"] == "f1"
    columns = client.get("/schema").json()
    assert [c["name"] for c in columns] == bundle.schema.names


def test_predict_returns_a_well_formed_prediction(client, record, bundle):
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "predicted_churn", "risk_segment",
                         "decision_threshold", "model"}
    assert 0 <= body["churn_probability"] <= 1 and body["risk_segment"] in {"Low", "Medium", "High"}
    assert body["decision_threshold"] == bundle.threshold and body["model"] == bundle.label
    assert body["predicted_churn"] == (body["churn_probability"] >= bundle.threshold - 1e-6)


def test_api_and_app_scoring_path_agree_for_the_same_input(client, record, bundle):
    """The Streamlit app calls score_frame; the API must return the same numbers."""
    expected = score_frame(bundle, pd.DataFrame([record]), strict_categories=True).table.iloc[0]
    body = client.post("/predict", json=record).json()
    assert body["churn_probability"] == pytest.approx(expected["probability"], abs=1e-6)
    assert body["predicted_churn"] == bool(expected["predicted_class"])
    assert body["risk_segment"] == str(expected["risk_segment"])


def test_batch_equals_individual_predictions(client, bundle):
    base = bundle.schema.baseline_row()
    records = [{**base, "tenure": t, "Contract": c}
               for t, c in [(1, "Month-to-month"), (30, "One year"), (70, "Two year")]]
    batch = client.post("/predict/batch", json=records).json()
    singles = [client.post("/predict", json=r).json() for r in records]
    assert batch == singles
    assert batch[0]["churn_probability"] > batch[2]["churn_probability"]   # sanity on the synthetic signal


def test_threshold_query_changes_the_decision_not_the_score(client, record):
    low = client.post("/predict?threshold=0.0", json=record).json()
    high = client.post("/predict?threshold=1.0", json=record).json()
    assert low["predicted_churn"] is True and high["predicted_churn"] is False
    assert low["churn_probability"] == high["churn_probability"]
    assert low["decision_threshold"] == 0.0
    assert client.post("/predict?threshold=1.5", json=record).status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "Weekly"},           # unseen category
        {"tenure": -3},                   # negative where training was non-negative
        {"MonthlyCharges": "cheap"},      # wrong type
        {"unexpected_field": 1},          # extra field forbidden
    ],
)
def test_invalid_records_are_rejected_with_422(client, record, change):
    assert client.post("/predict", json={**record, **change}).status_code == 422


def test_missing_field_is_rejected_and_named(client, record):
    record.pop("tenure")
    response = client.post("/predict", json=record)
    assert response.status_code == 422 and "tenure" in response.text


def test_empty_batch_and_oversized_batch(client, record):
    assert client.post("/predict/batch", json=[]).json() == []
    too_many = client.post("/predict/batch", json=[record] * (MAX_BATCH + 1))
    assert too_many.status_code == 413 and str(MAX_BATCH) in too_many.json()["detail"]


def test_openapi_documents_the_generated_input_contract(client, bundle):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["CustomerRecord"]
    assert set(schema["properties"]) == set(bundle.schema.names)
    assert set(schema["required"]) == set(bundle.schema.names)
    assert schema["properties"]["Contract"]["enum"] == list(bundle.schema.column("Contract").categories)


def test_missing_model_gives_degraded_health_and_actionable_503(tmp_path):
    client = TestClient(create_app(tmp_path / "empty"))
    health = client.get("/health").json()
    assert health["status"] == "degraded" and health["model_loaded"] is False
    response = client.post("/predict", json={})
    assert response.status_code == 503 and "train_model.py" in response.json()["detail"]
    assert client.get("/model").status_code == 503


def test_tampered_model_is_not_served(models_dir, tmp_path):
    import shutil

    clone = tmp_path / "clone"
    shutil.copytree(models_dir, clone)
    with open(clone / PIPELINE_FILE, "ab") as fh:
        fh.write(b"x")
    client = TestClient(create_app(clone))
    assert client.get("/health").json()["status"] == "degraded"
    assert "does not match the hash" in client.post("/predict", json={}).json()["detail"]


def test_cross_origin_calls_are_refused_unless_configured(models_dir, monkeypatch):
    headers = {"Origin": "https://example.org", "Access-Control-Request-Method": "POST"}
    default = TestClient(create_app(models_dir)).options("/predict", headers=headers)
    assert "access-control-allow-origin" not in default.headers
    monkeypatch.setenv("CHURNGUARD_CORS_ORIGINS", "https://example.org")
    allowed = TestClient(create_app(models_dir)).options("/predict", headers=headers)
    assert allowed.headers.get("access-control-allow-origin") == "https://example.org"


def test_models_dir_can_be_set_by_environment(models_dir, monkeypatch):
    monkeypatch.setenv("CHURNGUARD_MODELS_DIR", str(models_dir))
    assert TestClient(create_app()).get("/health").json()["status"] == "ok"
