import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer  # noqa: F401  (documenting the transformer under test)

from churnlens.config import Config
from churnlens.data import DataValidationError
from churnlens.pipeline import (
    MODEL_REGISTRY,
    FeatureSpec,
    build_estimator,
    build_pipeline,
    candidate_estimators,
    encoded_feature_table,
)
from churnlens.preprocessing import (
    CONTRACT_MONTHS,
    SERVICE_COLUMNS,
    TelcoFeatureEngineer,
    build_preprocessor,
)


def telco_row(**overrides):
    row = dict.fromkeys(SERVICE_COLUMNS, "No")
    row.update(Contract="Month-to-month", MonthlyCharges=50.0, TotalCharges=500.0, tenure=10)
    row.update(overrides)
    return pd.DataFrame([row])


# ----------------------------------------------------------------- feature engineer
def test_total_services_counts_exactly_the_six_service_columns():
    row = telco_row(OnlineSecurity="Yes", StreamingTV="Yes", OnlineFoo="Yes", TechnicalNotes="Yes")
    out = TelcoFeatureEngineer().transform(row)
    assert out["TotalServices"].iloc[0] == 2          # unrelated 'Online*'/'Tech*' columns ignored


def test_blank_total_charges_become_zero_only_when_tenure_is_zero():
    rows = pd.concat(
        [telco_row(TotalCharges=np.nan, tenure=0), telco_row(TotalCharges=np.nan, tenure=5)],
        ignore_index=True,
    )
    out = TelcoFeatureEngineer().transform(rows)
    assert out["TotalCharges"].iloc[0] == 0.0
    assert np.isnan(out["TotalCharges"].iloc[1])      # left for the imputer: no silent rule


def test_contract_months_neutral_encoding_and_unseen_is_missing_not_guessed():
    out = TelcoFeatureEngineer().transform(
        pd.concat([telco_row(Contract=c) for c in [*CONTRACT_MONTHS, "Weekly"]], ignore_index=True)
    )
    assert out["ContractMonths"].tolist()[:3] == [1.0, 12.0, 24.0]
    assert np.isnan(out["ContractMonths"].iloc[3])


@pytest.mark.parametrize(
    ("tenure", "segment"),
    [(0, "0-12 months"), (12, "0-12 months"), (13, "13-24 months"), (36, "25-36 months"),
     (37, "37+ months"), (72, "37+ months"), (500, "37+ months")],
)
def test_tenure_segments_cover_boundaries_and_out_of_range(tenure, segment):
    assert TelcoFeatureEngineer().transform(telco_row(tenure=tenure))["TenureSegment"].iloc[0] == segment


def test_charge_ratio_uses_corrected_total():
    out = TelcoFeatureEngineer().transform(telco_row(MonthlyCharges=40.0, TotalCharges=np.nan, tenure=0))
    assert out["ChargeRatio"].iloc[0] == pytest.approx(40.0)    # 40 / (0 + 1)


def test_engineering_is_row_wise_and_does_not_mutate_input(telco_like_df):
    X = telco_like_df.drop(columns=["customerID", "Churn"])
    X = X.assign(TotalCharges=pd.to_numeric(X["TotalCharges"], errors="coerce"))
    snapshot = X.copy()
    batch = TelcoFeatureEngineer().transform(X)
    pd.testing.assert_frame_equal(X, snapshot, check_dtype=False)
    single = TelcoFeatureEngineer().transform(X.iloc[[7]])
    for col in ("TotalServices", "ContractMonths", "ChargeRatio", "TenureSegment", "TotalCharges"):
        a, b = batch[col].iloc[7], single[col].iloc[0]
        assert (a == b) or (pd.isna(a) and pd.isna(b))


def test_engineer_reports_missing_raw_columns_not_derived_ones():
    with pytest.raises(DataValidationError, match="Missing required column") as err:
        TelcoFeatureEngineer().transform(telco_row().drop(columns=["tenure"]))
    assert "TenureSegment" not in str(err.value)


# --------------------------------------------------------------------- preprocessor
def test_preprocessor_imputes_scales_and_ignores_unknown_categories():
    train = pd.DataFrame({"n": [1.0, 2.0, np.nan, 4.0], "c": ["a", "b", None, "a"]})
    pre = build_preprocessor(["n"], ["c"]).fit(train)
    out = pre.transform(pd.DataFrame({"n": [np.nan], "c": ["zzz"]}))
    assert np.isfinite(out).all()
    assert out.shape == (1, 3)                         # 1 numeric + 2 learned categories
    assert out[0, 1:].sum() == 0                       # unseen category -> all zeros
    # A missing numeric value is imputed with the TRAINING median (2.0), then standardised with
    # the training mean/std of the imputed column - nothing is learned from the new row.
    imputed_train = np.array([1.0, 2.0, 2.0, 4.0])
    expected = (2.0 - imputed_train.mean()) / imputed_train.std()
    assert out[0, 0] == pytest.approx(expected)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(imputed_train.mean())


# ------------------------------------------------------------------------- pipeline
def test_encoded_feature_table_matches_sklearn_feature_names(generic_experiment):
    pipe = generic_experiment.pipeline
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    table = encoded_feature_table(pipe)
    assert table["encoded"].tolist() == names.tolist()
    assert "plan = basic" in table["label"].tolist()


def test_registry_and_config_cover_the_same_models():
    cfg = Config()
    assert set(MODEL_REGISTRY) == {"dummy", "logistic_regression", "decision_tree",
                                   "random_forest", "gradient_boosting"}
    for key in MODEL_REGISTRY:
        assert hasattr(cfg.models, key)


def test_estimators_are_seeded_and_unknown_key_is_explicit():
    cfg = Config()
    est = candidate_estimators(cfg)
    assert est["random_forest"].random_state == cfg.cv.random_state
    assert est["gradient_boosting"].random_state == cfg.cv.random_state
    assert not hasattr(est["dummy"], "random_state") or est["dummy"].random_state is None
    with pytest.raises(KeyError, match="Unknown model"):
        build_estimator("svm", {}, 0)


def test_feature_spec_lists_engineered_columns_only_when_enabled():
    plain = FeatureSpec(("a",), ("b",), engineer=False)
    eng = FeatureSpec(("a",), ("b",), engineer=True)
    assert plain.model_numeric == ["a"] and plain.model_categorical == ["b"]
    assert "TotalServices" in eng.model_numeric and "TenureSegment" in eng.model_categorical
    assert FeatureSpec.from_dict(eng.to_dict()) == eng


def test_pipeline_fits_and_predicts_unseen_category(generic_df, fast_config):
    spec = FeatureSpec(("tenure", "monthly_charges"), ("plan",))
    y = (generic_df["churn"] == "Yes").astype(int)
    pipe = build_pipeline(spec, candidate_estimators(fast_config, ["logistic_regression"])["logistic_regression"])
    pipe.fit(generic_df[["tenure", "monthly_charges", "plan"]], y)
    new = pd.DataFrame({"tenure": [3], "monthly_charges": [50.0], "plan": ["never-seen"]})
    proba = pipe.predict_proba(new)
    assert proba.shape == (1, 2) and proba.sum() == pytest.approx(1.0)
