from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from churnguard.config import SelectionConfig
from churnguard.data import prepare_dataset
from churnguard.evaluation import ALL_METRICS, compute_metrics, recommend_threshold
from churnguard.training import (
    compare_models_cv,
    make_cv,
    run_experiment,
    select_model,
    split_indices,
    summarize_folds,
)

QUICK = ["dummy", "logistic_regression", "decision_tree"]
TIME_COLUMNS = ("fit_time_s", "score_time_s")


# --------------------------------------------------------------------------- splits
def test_split_is_stratified_disjoint_complete_and_deterministic(generic_prepared, fast_config):
    y = generic_prepared.y
    train, test = split_indices(y, fast_config)
    assert set(train).isdisjoint(test) and len(train) + len(test) == len(y)
    assert len(test) == pytest.approx(fast_config.split.test_size * len(y), abs=1)
    assert y.iloc[train].mean() == pytest.approx(y.mean(), abs=0.01)
    assert y.iloc[test].mean() == pytest.approx(y.mean(), abs=0.02)
    again_train, again_test = split_indices(y, fast_config)
    assert (train == again_train).all() and (test == again_test).all()


def test_cv_folds_are_deterministic_so_every_model_sees_the_same_folds(generic_prepared, fast_config):
    train, _ = split_indices(generic_prepared.y, fast_config)
    X, y = generic_prepared.frame.iloc[train], generic_prepared.y.iloc[train]
    cv_a, cv_b = make_cv(fast_config, y), make_cv(fast_config, y)
    for (tr_a, va_a), (tr_b, va_b) in zip(cv_a.split(X, y), cv_b.split(X, y), strict=True):
        assert (tr_a == tr_b).all() and (va_a == va_b).all()


# ---------------------------------------------------------------- CV comparison table
def test_cv_comparison_has_every_model_metric_fold_and_timing(generic_experiment):
    folds, summary = generic_experiment.cv_folds, generic_experiment.comparison
    assert set(folds["model"]) == {"dummy", "logistic_regression", "decision_tree",
                                   "random_forest", "gradient_boosting"}
    n_folds = int(summary["n_folds"].iloc[0])
    assert (folds.groupby("model").size() == n_folds).all()
    for metric in ALL_METRICS:
        assert {f"{metric}_mean", f"{metric}_std"} <= set(summary.columns)
    assert (folds["fit_time_s"] >= 0).all() and (folds["score_time_s"] >= 0).all()


def test_summary_is_mean_and_sample_std_of_the_folds(generic_experiment):
    folds = generic_experiment.cv_folds
    lr = folds[folds["model"] == "logistic_regression"]["roc_auc"]
    row = generic_experiment.comparison.set_index("model").loc["logistic_regression"]
    assert row["roc_auc_mean"] == pytest.approx(lr.mean())
    assert row["roc_auc_std"] == pytest.approx(lr.std(ddof=1))
    assert summarize_folds(folds).shape == generic_experiment.comparison.shape


def test_dummy_baseline_is_a_floor_the_real_models_beat(generic_experiment):
    s = generic_experiment.comparison.set_index("model")
    assert s.loc["dummy", "roc_auc_mean"] == pytest.approx(0.5)
    assert s.loc["dummy", "average_precision_mean"] == pytest.approx(
        generic_experiment.prepared.y.iloc[generic_experiment.train_index].mean(), abs=0.01)
    # The synthetic data carries a real signal, so a learner must clearly beat the floor.
    assert s.loc["logistic_regression", "roc_auc_mean"] > s.loc["dummy", "roc_auc_mean"] + 0.1


# ------------------------------------------------------------------------- selection
def summary_frame(rows):
    base = dict.fromkeys(ALL_METRICS, 0.0)
    out = []
    for model, mean, std in rows:
        row = {"model": model, "n_folds": 5, **{f"{m}_mean": v for m, v in base.items()},
               **{f"{m}_std": 0.0 for m in base}}
        row["average_precision_mean"], row["average_precision_std"] = mean, std
        row["log_loss_mean"], row["log_loss_std"] = 1 - mean, std
        out.append(row)
    return pd.DataFrame(out)


def cfg_with(config, **kwargs):
    return replace(config, selection=SelectionConfig(**kwargs))


def test_best_rule_takes_the_highest_mean_even_if_complex(fast_config):
    s = summary_frame([("logistic_regression", 0.69, 0.02), ("gradient_boosting", 0.70, 0.02)])
    key, _ = select_model(s, cfg_with(fast_config, primary_metric="average_precision", rule="best"))
    assert key == "gradient_boosting"


def test_one_se_rule_prefers_the_simpler_model_within_noise(fast_config):
    cfg = cfg_with(fast_config, primary_metric="average_precision", rule="one_se")
    # SE = 0.02 / sqrt(5) = 0.0089: a 0.005 gap is inside it ...
    close = summary_frame([("logistic_regression", 0.695, 0.02), ("gradient_boosting", 0.700, 0.02)])
    key, note = select_model(close, cfg)
    assert key == "logistic_regression" and "within one standard error" in note
    # ... a 0.03 gap is not: the more complex model must earn its place.
    far = summary_frame([("logistic_regression", 0.670, 0.02), ("gradient_boosting", 0.700, 0.02)])
    assert select_model(far, cfg)[0] == "gradient_boosting"


def test_dummy_baseline_is_never_selected(fast_config):
    s = summary_frame([("dummy", 0.99, 0.0), ("decision_tree", 0.60, 0.02)])
    assert select_model(s, fast_config)[0] == "decision_tree"


def test_lower_is_better_metrics_are_handled(fast_config):
    s = summary_frame([("logistic_regression", 0.60, 0.0), ("random_forest", 0.70, 0.0)])
    # log_loss_mean = 1 - mean here, so the RF (0.30) beats LR (0.40) on log loss
    key, _ = select_model(s, cfg_with(fast_config, primary_metric="log_loss", rule="best"))
    assert key == "random_forest"


def test_unknown_primary_metric_is_rejected(fast_config):
    with pytest.raises(ValueError, match="Unknown primary metric"):
        select_model(summary_frame([("decision_tree", 0.6, 0.0)]),
                     cfg_with(fast_config, primary_metric="vibes"))


# ------------------------------------------------------------ protocol-level guarantees
def test_experiment_artefacts_are_internally_consistent(generic_experiment):
    r = generic_experiment
    n_train, n_test = len(r.train_index), len(r.test_index)
    assert sorted(r.oof["row"]) == sorted(r.train_index)            # each training row once
    assert set(r.oof["row"]).isdisjoint(r.test_predictions["row"])  # OOF never sees the test set
    assert len(r.test_predictions) == n_test
    assert r.test_predictions["customer_id"].is_unique
    for proba in (r.oof["proba"], r.test_predictions["proba"]):
        assert proba.between(0, 1).all()
    # threshold comes from validation (OOF) predictions and from nothing else
    again = recommend_threshold(r.oof["y_true"], r.oof["proba"], r.config.threshold.criterion)
    assert r.threshold.threshold == again.threshold
    # reported test metrics are exactly the metrics of the stored test predictions
    recomputed = compute_metrics(r.test_predictions["y_true"], r.test_predictions["proba"],
                                 r.threshold.threshold)
    assert r.test_metrics == pytest.approx(recomputed)
    assert r.metadata["data"]["n_train"] == n_train and r.metadata["data"]["n_test"] == n_test


def test_probabilities_are_coherent(generic_experiment):
    r = generic_experiment
    X = r.prepared.frame.iloc[r.test_index][r.schema.names]
    both = r.pipeline.predict_proba(X)
    assert both.shape == (len(X), 2)
    assert np.allclose(both.sum(axis=1), 1.0) and (both >= 0).all() and (both <= 1).all()
    # raising the threshold can only shrink the set of predicted churners
    flagged = [(both[:, 1] >= t).sum() for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert flagged == sorted(flagged, reverse=True)


def test_model_selection_ignores_test_labels_and_is_reproducible(generic_prepared, fast_config, monkeypatch):
    """Flip every TEST label: selection, CV scores, OOF predictions and threshold must not move."""
    train, test = split_indices(generic_prepared.y, fast_config)
    first = run_experiment(generic_prepared, fast_config, model_keys=QUICK)
    again = run_experiment(generic_prepared, fast_config, model_keys=QUICK)

    flipped_y = generic_prepared.y.copy()
    flipped_y.iloc[test] = 1 - flipped_y.iloc[test]
    poisoned = replace(generic_prepared, y=flipped_y)
    # keep the very same split (it is stratified on y, which we just altered)
    monkeypatch.setattr("churnguard.training.split_indices", lambda y, cfg: (train, test))
    flipped = run_experiment(poisoned, fast_config, model_keys=QUICK)

    stable = [c for c in first.comparison.columns if not c.startswith(TIME_COLUMNS)]
    pd.testing.assert_frame_equal(first.comparison[stable], again.comparison[stable])   # reproducible
    pd.testing.assert_frame_equal(first.comparison[stable], flipped.comparison[stable])  # test-blind
    assert first.selected_key == flipped.selected_key
    assert first.threshold.threshold == flipped.threshold.threshold
    pd.testing.assert_frame_equal(first.oof, flipped.oof)
    np.testing.assert_array_equal(first.test_predictions["proba"], again.test_predictions["proba"])
    # ... while the test evaluation, which does use the labels, does change
    assert first.test_metrics["roc_auc"] != pytest.approx(flipped.test_metrics["roc_auc"])


def test_compare_models_cv_accepts_a_model_subset(generic_prepared, fast_config):
    r = run_experiment(generic_prepared, fast_config, model_keys=["dummy", "logistic_regression"])
    assert list(r.comparison["model"]) == ["dummy", "logistic_regression"]
    assert r.selected_key == "logistic_regression"
    folds, _ = compare_models_cv(r.spec, generic_prepared.frame.iloc[r.train_index][r.schema.names],
                                 generic_prepared.y.iloc[r.train_index], fast_config, ["dummy"])
    assert set(folds["model"]) == {"dummy"}


# ----------------------------------------------------------------- leakage guard in use
def test_identifiers_and_target_proxies_are_screened_out_before_training(generic_df, fast_config):
    df = generic_df.copy()
    df["customer_number"] = np.arange(len(df))                      # identifier recognised by its name
    df["ordinal"] = np.arange(len(df))                              # identifier recognised by its values
    df["cancel_code"] = np.where(df["churn"] == "Yes", 7, 0)        # perfect proxy of the target
    prepared = prepare_dataset(df, "churn", positive_label="Yes", id_column="customer_id",
                               config=fast_config)
    r = run_experiment(prepared, fast_config, model_keys=QUICK)
    assert "recognised by name" in r.roles.ignored["customer_number"]
    assert "identifier-like integer" in r.roles.ignored["ordinal"]
    assert "leakage" in r.roles.ignored["cancel_code"]
    assert {"customer_number", "ordinal", "cancel_code"}.isdisjoint(r.schema.names)
    assert r.test_metrics["roc_auc"] < 0.95        # an unscreened proxy would give ~1.0


def test_user_can_knowingly_keep_a_flagged_column(generic_df, fast_config):
    df = generic_df.copy()
    df["cancel_code"] = np.where(df["churn"] == "Yes", 7, 0)
    prepared = prepare_dataset(df, "churn", positive_label="Yes", config=fast_config)
    r = run_experiment(prepared, fast_config, model_keys=QUICK, allow=["cancel_code"])
    assert "cancel_code" in r.schema.names and r.test_metrics["roc_auc"] > 0.99


def test_engineered_telco_pipeline_trains_and_keeps_raw_input_contract(telco_experiment):
    r = telco_experiment
    assert r.spec.engineer
    assert "TotalServices" not in r.schema.names            # derived columns are not user inputs
    assert "Contract" in r.schema.names and "customerID" not in r.schema.names
    assert r.metadata["features"]["engineer"] is True
