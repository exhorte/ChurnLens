"""Logic behind the Streamlit pages (no browser): workspaces, populations, training from a CSV."""
import numpy as np
import pandas as pd
import pytest

from churnlens.data import DataValidationError
from churnlens.predictions import score_frame
from ui.train_panel import (
    preview_roles,
    suggest_id_column,
    suggest_positive_label,
    train_from_dataframe,
)
from ui.workspace import (
    POPULATION_ALL,
    POPULATION_TEST,
    POPULATION_UPLOAD,
    PopulationUnavailableError,
    population_for,
    workspace_from_experiment,
)


@pytest.fixture(scope="module")
def ws(generic_experiment):
    return workspace_from_experiment(generic_experiment, "Custom - synthetic")


# ----------------------------------------------------------------------- populations
def test_test_population_is_the_heldout_split_with_known_outcomes(ws, generic_experiment):
    pop = population_for(ws, POPULATION_TEST, ws.bundle.threshold)
    assert len(pop.X) == len(generic_experiment.test_index) and pop.has_actual
    assert pop.warnings == []
    assert list(pop.ids) == list(generic_experiment.test_predictions["customer_id"])
    np.testing.assert_allclose(pop.table["probability"], generic_experiment.test_predictions["proba"], atol=1e-12)
    assert list(pop.actual) == list(generic_experiment.test_predictions["y_true"])


def test_all_customers_population_warns_that_training_rows_are_included(ws):
    pop = population_for(ws, POPULATION_ALL, 0.5)
    assert len(pop.X) == len(ws.prepared.y)
    assert any("trained on" in w for w in pop.warnings)


def test_threshold_is_applied_to_the_population(ws):
    low = population_for(ws, POPULATION_TEST, 0.05).table["predicted_class"].sum()
    high = population_for(ws, POPULATION_TEST, 0.95).table["predicted_class"].sum()
    assert low > high


def test_uploaded_customers_are_validated_and_scored_without_a_target(ws, generic_df):
    new = generic_df.drop(columns=["churn"]).head(25)
    pop = population_for(ws, POPULATION_UPLOAD, ws.bundle.threshold, uploaded=new)
    assert len(pop.table) == 25 and not pop.has_actual
    assert list(pop.ids) == list(new["customer_id"])
    expected = score_frame(ws.bundle, new.reset_index(drop=True)).table["probability"]
    np.testing.assert_array_equal(pop.table["probability"], expected)


def test_uploaded_file_without_id_column_still_scores(ws, generic_df):
    new = generic_df.drop(columns=["churn", "customer_id"]).head(10)
    pop = population_for(ws, POPULATION_UPLOAD, 0.5, uploaded=new)
    assert pop.ids is None and len(pop.table) == 10


def test_bad_uploads_give_readable_errors(ws, generic_df):
    with pytest.raises(PopulationUnavailableError, match="Upload a CSV"):
        population_for(ws, POPULATION_UPLOAD, 0.5, uploaded=None)
    with pytest.raises(DataValidationError, match="Missing required column"):
        population_for(ws, POPULATION_UPLOAD, 0.5, uploaded=generic_df[["tenure"]])


def test_pages_that_need_features_explain_what_is_missing(ws):
    featureless = type(ws)(**{**ws.__dict__, "prepared": None})
    assert not featureless.has_features
    with pytest.raises(PopulationUnavailableError, match="needs the training dataset"):
        population_for(featureless, POPULATION_TEST, 0.5)


# ------------------------------------------------------------------- train from CSV
def test_target_is_never_guessed_in_training(generic_df, fast_config):
    kw = {"positive_label": "Yes", "id_column": None, "exclude": [], "allow": [],
          "model_keys": ["dummy", "logistic_regression"], "config": fast_config, "source_name": "x.csv"}
    with pytest.raises(DataValidationError, match="never assumed to be the last"):
        train_from_dataframe(generic_df, target=None, **kw)
    with pytest.raises(DataValidationError, match="positive class"):
        train_from_dataframe(generic_df, target="churn", **{**kw, "positive_label": None})
    with pytest.raises(DataValidationError, match="at least one model"):
        train_from_dataframe(generic_df, target="churn", **{**kw, "model_keys": []})
    with pytest.raises(DataValidationError, match="not found"):
        train_from_dataframe(generic_df, target="missing", **kw)


def test_training_on_a_csv_yields_a_labelled_custom_workspace(generic_df, fast_config):
    messages = []
    custom = train_from_dataframe(
        generic_df, target="churn", positive_label="Yes", id_column="customer_id", exclude=["noise"],
        allow=[], model_keys=["dummy", "logistic_regression"], config=fast_config,
        source_name="mine.csv", progress=messages.append,
    )
    assert not custom.is_demo and custom.name == "Custom - mine.csv"
    assert "noise" in custom.ignored_columns and "noise" not in custom.bundle.schema.names
    assert custom.bundle.metadata["target"]["positive_label"] == "Yes"
    assert any("Cross-validating" in m for m in messages)
    assert set(custom.comparison["label"]) == {"Dummy (class prior)", "Logistic Regression"}


def test_only_the_dummy_baseline_cannot_be_trained_as_a_model(generic_df, fast_config):
    with pytest.raises(ValueError, match="selectable"):
        train_from_dataframe(generic_df, target="churn", positive_label="Yes", id_column=None, exclude=[],
                             allow=[], model_keys=["dummy"], config=fast_config, source_name="x.csv")


def test_role_preview_explains_columns_before_training(generic_df, fast_config):
    df = generic_df.assign(cancel_code=np.where(generic_df["churn"] == "Yes", 3, 0))
    roles = preview_roles(df, "churn", "Yes", "customer_id", ["noise"], fast_config)
    assert "cancel_code" in roles.ignored and "leakage" in roles.ignored["cancel_code"]
    assert roles.ignored["noise"] == "excluded by you"
    assert "customer_id" not in roles.used


def test_suggestions_never_pick_ambiguous_values():
    assert suggest_positive_label(["No", "Yes"]) == "Yes"
    assert suggest_positive_label(["A", "B"]) is None
    assert suggest_positive_label([0, 1]) == "1"


def test_id_suggestion_needs_name_and_uniqueness(generic_df):
    assert suggest_id_column(generic_df, "churn") == "customer_id"
    assert suggest_id_column(generic_df.drop(columns="customer_id"), "churn") is None
    low_card = pd.DataFrame({"customer_type": ["a", "b"] * 50, "y": [0, 1] * 50})
    assert suggest_id_column(low_card, "y") is None
