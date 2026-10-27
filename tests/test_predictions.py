import json
import shutil

import numpy as np
import pandas as pd
import pytest
import sklearn

from churnlens.data import DataValidationError
from churnlens.predictions import (
    ModelLoadError,
    build_export,
    bundle_from_experiment,
    load_bundle,
    score_frame,
)
from churnlens.training import METADATA_FILE, PIPELINE_FILE, save_experiment


@pytest.fixture(scope="module")
def bundle(generic_experiment):
    return bundle_from_experiment(generic_experiment)


@pytest.fixture(scope="module")
def raw_test(generic_experiment):
    r = generic_experiment
    return r.prepared.frame.iloc[r.test_index]


@pytest.fixture
def saved(generic_experiment, tmp_path):
    models, reports = tmp_path / "models", tmp_path / "reports"
    paths = save_experiment(generic_experiment, models, reports)
    return models, reports, paths


# ------------------------------------------------------------------------- scoring
def test_scoring_returns_probability_class_segment_and_threshold(bundle, raw_test):
    scored = score_frame(bundle, raw_test)
    t = scored.table
    assert list(t.columns) == ["probability", "predicted_class", "risk_segment"]
    assert t["probability"].between(0, 1).all()
    assert (t["predicted_class"] == (t["probability"] >= bundle.threshold).astype(int)).all()
    assert scored.threshold == bundle.threshold
    assert (t.index == raw_test.index).all()
    seg = t["risk_segment"].astype(str)
    assert (seg[t["probability"] < bundle.segments.low_below] == "Low").all()
    assert (seg[t["probability"] >= bundle.segments.high_from] == "High").all()


def test_scoring_path_uses_exactly_the_saved_pipeline(bundle, raw_test):
    aligned = raw_test[bundle.schema.names]
    direct = bundle.pipeline.predict_proba(aligned)[:, 1]
    np.testing.assert_array_equal(score_frame(bundle, raw_test).table["probability"], direct)


def test_threshold_override_changes_decisions_not_scores(bundle, raw_test):
    low, high = score_frame(bundle, raw_test, threshold=0.05), score_frame(bundle, raw_test, threshold=0.95)
    np.testing.assert_array_equal(low.table["probability"], high.table["probability"])
    assert low.table["predicted_class"].sum() > high.table["predicted_class"].sum()
    with pytest.raises(ValueError, match="between 0 and 1"):
        score_frame(bundle, raw_test, threshold=1.5)


def test_column_order_and_extra_columns_do_not_matter(bundle, raw_test):
    base = score_frame(bundle, raw_test).table["probability"].to_numpy()
    shuffled = raw_test[list(reversed(raw_test.columns))].assign(extra_col="x")
    scored = score_frame(bundle, shuffled)
    np.testing.assert_array_equal(scored.table["probability"], base)
    assert any("extra_col" in w for w in scored.report.warnings)


def test_invalid_data_raises_readable_errors(bundle, raw_test):
    with pytest.raises(DataValidationError, match="Missing required column.*tenure"):
        score_frame(bundle, raw_test.drop(columns=["tenure"]))
    bad = raw_test.copy()
    bad["tenure"] = bad["tenure"].astype(object)
    bad.iloc[0, bad.columns.get_loc("tenure")] = "many"
    with pytest.raises(DataValidationError, match="must be numeric"):
        score_frame(bundle, bad)
    with pytest.raises(DataValidationError, match="no rows"):
        score_frame(bundle, raw_test.iloc[0:0])


def test_unknown_category_is_scored_with_a_warning_unless_strict(bundle, raw_test):
    odd = raw_test.copy()
    odd.iloc[0, odd.columns.get_loc("plan")] = "platinum"
    scored = score_frame(bundle, odd)
    assert np.isfinite(scored.table["probability"]).all()
    assert any("platinum" in w for w in scored.report.warnings)
    with pytest.raises(DataValidationError, match="platinum"):
        score_frame(bundle, odd, strict_categories=True)


def test_missing_values_are_imputed_and_reported(bundle, raw_test):
    gappy = raw_test.copy()
    gappy.iloc[0, gappy.columns.get_loc("monthly_charges")] = np.nan
    scored = score_frame(bundle, gappy)
    assert np.isfinite(scored.table["probability"]).all()
    assert any("missing value" in w for w in scored.report.warnings)


# ------------------------------------------------------------------------- export
def test_export_has_required_columns_and_is_sorted_by_risk(bundle, raw_test, generic_experiment):
    ids = generic_experiment.prepared.ids
    scored = score_frame(bundle, raw_test, ids=ids)
    export = build_export(scored, bundle, id_label="customer_id")
    assert {"customer_id", "id_type", "churn_probability", "predicted_churn", "predicted_label",
            "risk_segment", "decision_threshold", "model", "model_trained_at", "scored_at"} <= set(export)
    assert export["churn_probability"].is_monotonic_decreasing
    assert set(export["customer_id"]) == set(ids.iloc[generic_experiment.test_index])
    assert (export["decision_threshold"] == bundle.threshold).all()
    assert (export["model"] == bundle.label).all()
    assert export["id_type"].str.startswith("customer identifier").all()
    roundtrip = pd.read_csv(pd.io.common.StringIO(export.to_csv(index=False)))
    assert len(roundtrip) == len(raw_test)


def test_export_never_invents_a_customer_id(bundle, raw_test):
    export = build_export(score_frame(bundle, raw_test), bundle)
    assert sorted(export["customer_id"]) == list(range(1, len(raw_test) + 1))
    assert export["id_type"].str.contains("technical row number").all()


# ---------------------------------------------------------- persistence round trip
def test_saved_model_reproduces_predictions_exactly(saved, generic_experiment, raw_test):
    models, _, _ = saved
    loaded = load_bundle(models)
    before = generic_experiment.pipeline.predict_proba(raw_test[generic_experiment.schema.names])
    after = loaded.pipeline.predict_proba(raw_test[loaded.schema.names])
    np.testing.assert_array_equal(before, after)
    assert loaded.threshold == generic_experiment.threshold.threshold
    assert loaded.schema == generic_experiment.schema
    assert loaded.segments == generic_experiment.segments


def test_metadata_identifies_model_config_environment_and_data(saved, generic_experiment):
    models, _, _ = saved
    meta = json.loads((models / METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["run_id"] == generic_experiment.run_id and meta["trained_at"]
    assert meta["model"] == generic_experiment.selected_label
    assert meta["environment"]["scikit_learn"] == sklearn.__version__
    assert meta["random_state"] == {"split": 42, "cv": 42}
    assert meta["config_fingerprint"] == generic_experiment.config.fingerprint()
    assert meta["selection"]["primary_metric"] == "average_precision"
    assert meta["data"]["n_train"] + meta["data"]["n_test"] == meta["data"]["rows"]
    assert len(meta["artifact"]["sha256"]) == 64
    assert meta["threshold"]["criterion"] == "f1" and "test set is not used" in meta["threshold"]["description"]
    assert "test_metrics_at_recommended_threshold" in meta
    low, high = meta["test_metrics_ci_95"]["roc_auc"]
    assert low <= meta["test_metrics_at_recommended_threshold"]["roc_auc"] <= high


def test_reports_are_written_with_run_metadata(saved, generic_experiment):
    _, reports, _ = saved
    comparison = pd.read_csv(reports / "model_comparison.csv")
    assert {"run_id", "run_date", "config_fingerprint", "random_state", "primary_metric",
            "selection_rule", "score_source", "selected"} <= set(comparison)
    assert comparison["selected"].sum() == 1
    assert set(comparison["run_id"]) == {generic_experiment.run_id}
    assert comparison["score_source"].str.contains("cross-validation").all()
    folds = pd.read_csv(reports / "cv_folds.csv")
    assert len(folds) == len(comparison) * int(comparison["n_folds"].iloc[0])
    test_eval = pd.read_csv(reports / "test_evaluation.csv")
    assert len(test_eval) == 2 and test_eval["score_source"].str.contains("test set").all()
    assert len(pd.read_csv(reports / "oof_predictions.csv")) == len(generic_experiment.train_index)


def test_saved_reports_never_contain_customer_identifiers(saved, generic_experiment):
    """Identifiers belong to the dataset, not to the results: reports keep the row position only."""
    _, reports, _ = saved
    assert "customer_id" in generic_experiment.test_predictions.columns     # still available in memory
    for name in ("test_predictions.csv", "oof_predictions.csv"):
        report = pd.read_csv(reports / name)
        assert "customer_id" not in report.columns and "row" in report.columns
        assert not report.astype(str).apply(lambda col: col.str.startswith("C000")).any().any()


# ------------------------------------------------------------------- safe loading
def test_tampered_model_file_is_refused(saved):
    models, _, _ = saved
    with open(models / PIPELINE_FILE, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ModelLoadError, match="does not match the hash"):
        load_bundle(models)


def test_model_without_recorded_hash_is_refused_unless_verification_is_off(saved):
    models, _, _ = saved
    meta = json.loads((models / METADATA_FILE).read_text(encoding="utf-8"))
    meta.pop("artifact")
    (models / METADATA_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="no artifact hash"):
        load_bundle(models)
    assert load_bundle(models, verify=False).label


def test_sklearn_version_drift_is_reported(saved):
    models, _, _ = saved
    meta = json.loads((models / METADATA_FILE).read_text(encoding="utf-8"))
    meta["environment"]["scikit_learn"] = "0.0.1"
    (models / METADATA_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="trained with scikit-learn 0.0.1"):
        load_bundle(models)


def test_missing_and_corrupt_artifacts_give_actionable_messages(saved, tmp_path):
    with pytest.raises(ModelLoadError, match="train_model.py"):
        load_bundle(tmp_path / "nothing_here")
    models, _, _ = saved
    (models / METADATA_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="not valid JSON"):
        load_bundle(models)


def test_copied_model_directory_still_loads(saved, tmp_path):
    models, _, _ = saved
    clone = tmp_path / "clone"
    shutil.copytree(models, clone)
    assert load_bundle(clone).threshold > 0
