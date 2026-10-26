import io

import numpy as np
import pandas as pd
import pytest

from churnguard.data import (
    DatasetNotFoundError,
    DataValidationError,
    infer_roles,
    prepare_dataset,
    prepare_target,
    profile_dataframe,
    read_csv,
    standardize_frame,
    verify_demo_dataset,
)


# ------------------------------------------------------------------------ read_csv
@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("a,b\n", "no rows"),
        ("only_one_column\n1\n2\n", "single column"),
        ("a,b\n1,2\n1,2,3,4\n", "could not be read"),
    ],
)
def test_read_csv_rejects_unusable_files(text, message):
    with pytest.raises(DataValidationError, match=message):
        read_csv(io.StringIO(text))


def test_read_csv_missing_file_and_good_file(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        read_csv(tmp_path / "missing.csv")
    path = tmp_path / "ok.csv"
    path.write_text(" a , b \n1,x\n2,y\n")
    df = read_csv(path)
    assert list(df.columns) == ["a", "b"]          # header whitespace stripped
    assert len(df) == 2


# --------------------------------------------------------------------- standardize
def test_numeric_text_with_blanks_becomes_numeric_and_missing():
    df = pd.DataFrame({"total": ["10.5", " ", "3", "4"], "city": ["a", " b ", "", "a"]})
    out, notes = standardize_frame(df)
    assert pd.api.types.is_numeric_dtype(out["total"])
    assert out["total"].isna().sum() == 1 and out["total"].iloc[0] == 10.5
    assert out["city"].tolist()[1] == "b"           # stripped
    assert out["city"].isna().sum() == 1            # blank -> missing
    assert any("total" in n for n in notes)


def test_standardize_does_not_mutate_input_or_convert_real_text():
    df = pd.DataFrame({"plan": ["basic", "pro", "basic"], "flag": [True, False, True]})
    snapshot = df.copy()
    out, _ = standardize_frame(df)
    pd.testing.assert_frame_equal(df, snapshot)
    assert out["plan"].tolist() == ["basic", "pro", "basic"]
    assert out["flag"].tolist() == [1, 0, 1]


# ------------------------------------------------------------------------- target
def test_prepare_target_maps_positive_class_to_one():
    y, info = prepare_target(pd.Series(["No", "Yes", "Yes", "No"]), column="Churn")
    assert y.tolist() == [0, 1, 1, 0]
    assert info.positive_label == "Yes" and info.negative_label == "No"
    assert info.labels_were_inferred


def test_prepare_target_explicit_label_overrides_inference():
    y, info = prepare_target(pd.Series(["stayed", "left", "left"]), positive_label="stayed")
    assert y.tolist() == [1, 0, 0]
    assert not info.labels_were_inferred


def test_prepare_target_matches_numeric_labels_given_as_text():
    y, _ = prepare_target(pd.Series([0, 1, 1, 0]), positive_label="1")
    assert y.tolist() == [0, 1, 1, 0]


def test_prepare_target_refuses_to_guess_ambiguous_labels():
    with pytest.raises(DataValidationError, match="Choose it explicitly"):
        prepare_target(pd.Series(["A", "B", "A"]))


@pytest.mark.parametrize("values", [["a", "b", "c"], ["only"] * 5])
def test_prepare_target_requires_exactly_two_classes(values):
    with pytest.raises(DataValidationError, match="exactly 2"):
        prepare_target(pd.Series(values))


def test_prepare_target_unknown_positive_label_and_missing_values():
    with pytest.raises(DataValidationError, match="not found"):
        prepare_target(pd.Series(["a", "b"]), positive_label="z")
    y, _ = prepare_target(pd.Series(["Yes", None, "No"]))
    assert pd.isna(y.iloc[1]) and y.dropna().tolist() == [1, 0]


# -------------------------------------------------------------------- prepare_dataset
def test_prepare_dataset_uses_named_target_not_last_column(generic_df, fast_config):
    df = generic_df[["churn", "customer_id", "tenure", "plan", "monthly_charges"]]  # target first
    prepared = prepare_dataset(df, "churn", positive_label="Yes", id_column="customer_id",
                               config=fast_config)
    assert "churn" not in prepared.frame.columns and "customer_id" not in prepared.frame.columns
    assert prepared.y.isin([0, 1]).all()
    assert prepared.id_label == "customer_id" and prepared.ids is not None
    assert len(prepared.fingerprint) == 64


def test_prepare_dataset_without_id_column_reports_row_number(generic_df, fast_config):
    prepared = prepare_dataset(generic_df.drop(columns="customer_id"), "churn",
                               positive_label="Yes", config=fast_config)
    assert prepared.ids is None and prepared.id_label == "row_number"


def test_prepare_dataset_errors_are_explicit(generic_df, fast_config):
    kw = {"positive_label": "Yes", "config": fast_config}
    with pytest.raises(DataValidationError, match="Target column 'nope' not found"):
        prepare_dataset(generic_df, "nope", **kw)
    with pytest.raises(DataValidationError, match="Identifier column 'zzz' not found"):
        prepare_dataset(generic_df, "churn", id_column="zzz", **kw)
    with pytest.raises(DataValidationError, match="cannot be the target"):
        prepare_dataset(generic_df, "churn", id_column="churn", **kw)
    with pytest.raises(DataValidationError, match="at least"):
        prepare_dataset(generic_df.head(40), "churn", **kw)
    with pytest.raises(DataValidationError, match="empty"):
        prepare_dataset(generic_df.iloc[0:0], "churn", **kw)


def test_prepare_dataset_rejects_tiny_minority_class(fast_config):
    df = pd.DataFrame({"x": np.arange(300), "y": ["No"] * 295 + ["Yes"] * 5})
    with pytest.raises(DataValidationError, match="rarest class"):
        prepare_dataset(df, "y", positive_label="Yes", config=fast_config)


def test_prepare_dataset_drops_rows_without_target_and_says_so(generic_df, fast_config):
    df = generic_df.copy()
    df.loc[:9, "churn"] = None
    prepared = prepare_dataset(df, "churn", positive_label="Yes", config=fast_config)
    assert len(prepared.y) == len(df) - 10
    assert any("without a target" in n for n in prepared.notes)


# -------------------------------------------------------------------------- profile
def test_profile_counts_missing_duplicates_and_constants():
    df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", "y"], "c": [7, 7, 7, 7],
                       "t": ["Y", "Y", "N", "N"]})
    p = profile_dataframe(df, target="t", positive_label="Y")
    assert p.n_rows == 4 and p.n_missing_cells == 1
    assert p.constant_columns == ["c"]
    assert p.positive_rate == 0.5 and p.target_counts == {"Y": 2, "N": 2}


# ---------------------------------------------------------------------------- roles
@pytest.fixture
def role_frame(fast_config):
    rng = np.random.default_rng(3)
    n = 400
    y = pd.Series(rng.integers(0, 2, n))
    X = pd.DataFrame(
        {
            "customer_id": [f"id{i}" for i in range(n)],
            "row_no": np.arange(n),
            "const": 1,
            "free_text": [f"unique text {i}" for i in range(n)],
            "proxy_num": y.to_numpy() * 10 + 5,
            "proxy_cat": np.where(y == 1, "closed", "open"),
            "age": rng.integers(18, 80, n),
            "plan": rng.choice(["a", "b", "c"], n),
            "signup": pd.date_range("2020-01-01", periods=n),
        }
    )
    return X, y


def test_roles_flag_identifiers_constants_text_dates_and_leakage(role_frame, fast_config):
    X, y = role_frame
    roles = infer_roles(X, y, target="churn", config=fast_config)
    assert roles.numeric == ["age"] and roles.categorical == ["plan"]
    why = roles.ignored
    assert "identifier" in why["customer_id"]
    assert "identifier-like integer" in why["row_no"]
    assert "constant" in why["const"]
    assert "cardinality" in why["free_text"]
    assert "leakage" in why["proxy_num"] and "leakage" in why["proxy_cat"]
    assert "date" in why["signup"]


def test_roles_user_overrides(role_frame, fast_config):
    X, y = role_frame
    roles = infer_roles(X, y, config=fast_config, allow=["proxy_num"], exclude=["age"])
    assert "proxy_num" in roles.numeric               # explicit keep beats the leakage screen
    assert roles.ignored["age"] == "excluded by you"
    assert set(roles.table()["role"]) == {"numeric", "categorical", "ignored"}


def test_roles_warn_about_target_like_names(fast_config):
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 200))
    X = pd.DataFrame({"churn_score_old": rng.normal(size=200), "plan": rng.choice(["a", "b"], 200)})
    roles = infer_roles(X, y, target="churn", config=fast_config)
    assert any("churn_score_old" in w for w in roles.warnings)


def test_roles_do_not_flag_ordinary_columns_of_the_telco_like_data(telco_experiment):
    assert telco_experiment.roles.ignored == {}
    assert "SeniorCitizen" in telco_experiment.roles.numeric


# --------------------------------------------------------------------- demo guard
def test_demo_guard_rejects_placeholder_ids(fast_config):
    df = pd.DataFrame({"customerID": [f"ID-{i:05d}" for i in range(50)], "Churn": ["Yes", "No"] * 25})
    with pytest.raises(DataValidationError, match="placeholder"):
        verify_demo_dataset(df, fast_config)


def test_demo_guard_rejects_data_without_contract_signal(fast_config):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "customerID": [f"{i}-AB" for i in range(6000)],   # large n: noise spread << 0.10
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], 6000),
            "Churn": rng.choice(["Yes", "No"], 6000),
        }
    )
    with pytest.raises(DataValidationError, match="No churn signal"):
        verify_demo_dataset(df, fast_config)
