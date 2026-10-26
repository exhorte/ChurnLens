"""Explanations must describe the model that is actually deployed, not an approximation of it."""
import copy

import numpy as np
import pandas as pd
import pytest

from churnguard.explainability import (
    ASSOCIATION_NOTICE,
    explain_prediction,
    is_linear,
    native_importance,
    permutation_importance_table,
)
from churnguard.pipeline import build_pipeline, candidate_estimators, encoded_feature_table
from churnguard.schema import require_valid


@pytest.fixture(scope="module")
def train_xy(generic_experiment):
    r = generic_experiment
    X = require_valid(r.prepared.frame.iloc[r.train_index][r.schema.names], r.schema)
    return X, r.prepared.y.iloc[r.train_index]


@pytest.fixture(scope="module")
def pipelines(generic_experiment, train_xy, fast_config):
    X, y = train_xy
    return {
        key: build_pipeline(generic_experiment.spec, est).fit(X, y)
        for key, est in candidate_estimators(
            fast_config, ["dummy", "logistic_regression", "random_forest", "gradient_boosting"]
        ).items()
    }


@pytest.fixture(scope="module")
def test_xy(generic_experiment):
    r = generic_experiment
    X = require_valid(r.prepared.frame.iloc[r.test_index][r.schema.names], r.schema)
    return X, r.prepared.y.iloc[r.test_index]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------------------- linear: exact identity
def test_linear_explanation_is_an_exact_decomposition_of_the_model(pipelines, test_xy, generic_experiment):
    pipe, (X, _) = pipelines["logistic_regression"], test_xy
    assert is_linear(pipe)
    for i in range(0, 60, 7):
        row = X.iloc[[i]]
        exp = explain_prediction(pipe, generic_experiment.schema, row)
        assert exp.method == "linear_exact" and exp.unit == "log-odds"
        assert exp.logit == pytest.approx(float(pipe.decision_function(row)[0]), abs=1e-9)
        assert sigmoid(exp.logit) == pytest.approx(float(pipe.predict_proba(row)[0, 1]), abs=1e-9)
        assert exp.probability == pytest.approx(float(pipe.predict_proba(row)[0, 1]))


def test_linear_explanation_table_is_sorted_complete_and_readable(pipelines, test_xy, generic_experiment):
    pipe, (X, _) = pipelines["logistic_regression"], test_xy
    exp = explain_prediction(pipe, generic_experiment.schema, X.iloc[[3]])
    contributions = exp.table["contribution"].abs().tolist()
    assert contributions == sorted(contributions, reverse=True)
    assert sorted(exp.table["feature"]) == sorted(generic_experiment.schema.names)  # one-hot summed back
    shown = exp.table.set_index("feature")["value"]
    assert shown["tenure"] == X.iloc[3]["tenure"] and shown["plan"] == X.iloc[3]["plan"]


def test_explanation_follows_the_model_when_the_model_changes(pipelines, test_xy, generic_experiment):
    """Zero the tenure coefficient in a copy of the model: its explanation must react accordingly."""
    original = pipelines["logistic_regression"]
    row = test_xy[0].iloc[[5]]
    before = explain_prediction(original, generic_experiment.schema, row).table.set_index("feature")
    assert before.loc["tenure", "contribution"] != 0

    edited = copy.deepcopy(original)
    names = encoded_feature_table(edited)
    edited.named_steps["classifier"].coef_[0, names.index[names["encoded"] == "num__tenure"][0]] = 0.0
    after_exp = explain_prediction(edited, generic_experiment.schema, row)
    after = after_exp.table.set_index("feature")
    assert after.loc["tenure", "contribution"] == 0
    assert after.loc["plan", "contribution"] == pytest.approx(before.loc["plan", "contribution"])
    assert after_exp.logit == pytest.approx(float(edited.decision_function(row)[0]), abs=1e-9)

    shifted = copy.deepcopy(original)
    shifted.named_steps["classifier"].intercept_ = shifted.named_steps["classifier"].intercept_ + 2.0
    assert explain_prediction(shifted, generic_experiment.schema, row).base_value == pytest.approx(
        float(original.named_steps["classifier"].intercept_[0]) + 2.0
    )


# ---------------------------------------------------- non-linear: baseline substitution
def test_nonlinear_models_use_baseline_substitution(pipelines, test_xy, generic_experiment):
    X = test_xy[0]
    for key in ("random_forest", "gradient_boosting"):
        assert not is_linear(pipelines[key])
        exp = explain_prediction(pipelines[key], generic_experiment.schema, X.iloc[[0]])
        assert exp.method == "baseline_substitution" and exp.unit == "probability points"
        assert exp.logit is None
        assert len(exp.table) == len(generic_experiment.schema.names)


def test_substitution_matches_a_manual_recomputation(pipelines, test_xy, generic_experiment):
    pipe, schema = pipelines["random_forest"], generic_experiment.schema
    row = test_xy[0].iloc[[11]]
    exp = explain_prediction(pipe, schema, row)
    altered = row.copy()
    altered["tenure"] = schema.column("tenure").baseline
    expected = float(pipe.predict_proba(row)[0, 1] - pipe.predict_proba(altered)[0, 1])
    assert exp.table.set_index("feature").loc["tenure", "contribution"] == pytest.approx(expected)


def test_baseline_customer_has_nothing_to_explain(pipelines, generic_experiment):
    schema = generic_experiment.schema
    typical = pd.DataFrame([schema.baseline_row()])[schema.names]
    typical = require_valid(typical, schema)
    exp = explain_prediction(pipelines["gradient_boosting"], schema, typical)
    assert np.allclose(exp.table["contribution"], 0.0, atol=1e-12)
    assert exp.probability == pytest.approx(exp.base_value)


def test_model_without_signal_yields_zero_contributions(pipelines, test_xy, generic_experiment):
    exp = explain_prediction(pipelines["dummy"], generic_experiment.schema, test_xy[0].iloc[[0]])
    assert np.allclose(exp.table["contribution"], 0.0)


def test_extreme_customer_is_explained_by_the_features_that_drive_the_data(pipelines, generic_experiment):
    """SYNTHETIC data: churn depends on tenure and plan only; `noise` and charges are irrelevant."""
    schema = generic_experiment.schema
    risky = pd.DataFrame([{**schema.baseline_row(), "tenure": 1, "plan": "basic"}])[schema.names]
    risky = require_valid(risky, schema)
    # Logistic regression: exact, so the two true drivers must be the two largest contributions.
    linear = explain_prediction(pipelines["logistic_regression"], schema, risky).table
    assert set(linear["feature"].head(2)) == {"tenure", "plan"}
    # Random forest (25 trees): approximate, so only require that the strongest factor is a true driver.
    forest = explain_prediction(pipelines["random_forest"], schema, risky).table
    assert forest["feature"].iloc[0] in {"tenure", "plan"}
    assert forest.iloc[0]["contribution"] > 0            # a 1-month basic customer is above typical risk


def test_explain_prediction_needs_exactly_one_row(pipelines, test_xy, generic_experiment):
    with pytest.raises(ValueError, match="exactly one row"):
        explain_prediction(pipelines["logistic_regression"], generic_experiment.schema, test_xy[0].iloc[:2])


# -------------------------------------------------------------------------- global
def test_native_importance_is_the_models_own_numbers(pipelines):
    lr = pipelines["logistic_regression"]
    table = native_importance(lr)
    coef = lr.named_steps["classifier"].coef_[0]
    by_name = dict(zip(encoded_feature_table(lr)["encoded"], coef, strict=True))
    for _, row in table.iterrows():
        assert row["value"] == pytest.approx(by_name[row["encoded"]])
        assert row["odds_ratio"] == pytest.approx(np.exp(row["value"]))
    assert table["value"].abs().is_monotonic_decreasing
    assert "plan = basic" in table["label"].tolist()

    forest = native_importance(pipelines["random_forest"])
    assert forest["value"].sum() == pytest.approx(1.0) and forest["kind"].iloc[0] == "impurity importance"
    assert native_importance(pipelines["dummy"]) is None
    assert native_importance(pipelines["gradient_boosting"]) is None


def test_permutation_importance_ranks_true_drivers_above_noise(pipelines, test_xy):
    X, y = test_xy
    table = permutation_importance_table(pipelines["logistic_regression"], X, y, n_repeats=5)
    ranked = table.set_index("feature")["importance_mean"]
    assert set(table["feature"]) == set(X.columns)
    assert table["importance_mean"].is_monotonic_decreasing
    assert ranked["tenure"] > ranked["noise"] and ranked["plan"] > ranked["noise"]
    assert abs(ranked["noise"]) < 0.02


def test_permutation_importance_is_zero_for_a_model_that_ignores_features(pipelines, test_xy):
    X, y = test_xy
    table = permutation_importance_table(pipelines["dummy"], X, y, n_repeats=3)
    assert np.allclose(table["importance_mean"], 0.0)


def test_association_notice_is_explicit_about_causality():
    assert "do not show" in ASSOCIATION_NOTICE and "causes" in ASSOCIATION_NOTICE
