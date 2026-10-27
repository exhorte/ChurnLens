import numpy as np
import pandas as pd
import pytest

from churnlens.data import DataValidationError
from churnlens.schema import DataSchema, infer_schema, require_valid, validate_features


@pytest.fixture
def schema():
    train = pd.DataFrame(
        {
            "tenure": [0, 5, 10, 70],
            "balance": [10.5, 20.0, 30.25, 40.0],
            "plan": ["basic", "pro", "basic", "plus"],
        }
    )
    return infer_schema(train, ["tenure", "balance"], ["plan"])


def good_frame():
    return pd.DataFrame({"tenure": [3, 8], "balance": [15.0, 25.0], "plan": ["pro", "basic"]})


def test_schema_describes_training_data(schema):
    tenure = schema.column("tenure")
    # median of [0, 5, 10, 70] is 7.5, but an integer column keeps an integer baseline
    assert (tenure.minimum, tenure.maximum, tenure.baseline) == (0.0, 70.0, 8.0)
    assert tenure.non_negative and tenure.integer
    balance = schema.column("balance")
    assert not balance.integer and balance.baseline == 25.125     # float column: plain median
    plan = schema.column("plan")
    assert plan.categories == ("basic", "plus", "pro") and plan.baseline == "basic"
    assert schema.names == ["tenure", "balance", "plan"]


def test_schema_roundtrips_through_json_types(schema):
    import json

    again = DataSchema.from_dict(json.loads(json.dumps(schema.to_dict())))
    assert again == schema


def test_valid_frame_passes_and_is_typed(schema):
    report = validate_features(good_frame(), schema)
    assert report.ok and not report.warnings
    assert report.frame["tenure"].dtype == float


def test_columns_are_matched_by_name_not_position(schema):
    shuffled = good_frame()[["plan", "balance", "tenure"]]
    out = require_valid(shuffled, schema)
    assert list(out.columns) == schema.names
    assert out["plan"].tolist() == ["pro", "basic"]


def test_missing_columns_are_named_in_the_error(schema):
    report = validate_features(good_frame().drop(columns=["tenure", "plan"]), schema)
    assert not report.ok
    assert "tenure" in report.errors[0] and "plan" in report.errors[0]
    with pytest.raises(DataValidationError, match="Missing required column"):
        report.raise_if_errors()


def test_extra_columns_are_ignored_with_a_warning(schema):
    frame = good_frame().assign(unused=1)
    report = validate_features(frame, schema)
    assert report.ok and "unused" in report.warnings[0]
    assert "unused" not in report.frame.columns


def test_non_numeric_value_in_numeric_column_is_an_error(schema):
    frame = good_frame()
    frame["tenure"] = ["3", "abc"]
    report = validate_features(frame, schema)
    assert any("must be numeric" in e and "abc" in e for e in report.errors)


def test_numeric_text_is_accepted(schema):
    frame = good_frame()
    frame["balance"] = ["15.5", "25"]
    assert validate_features(frame, schema).ok


def test_negative_values_rejected_when_training_was_non_negative(schema):
    frame = good_frame()
    frame.loc[0, "tenure"] = -4
    report = validate_features(frame, schema)
    assert any("negative" in e for e in report.errors)


def test_out_of_range_is_only_a_warning(schema):
    frame = good_frame()
    frame.loc[0, "balance"] = 9999.0
    report = validate_features(frame, schema)
    assert report.ok and any("outside the training range" in w for w in report.warnings)


def test_unseen_category_warns_by_default_and_errors_when_strict(schema):
    frame = good_frame()
    frame.loc[0, "plan"] = "platinum"
    lenient = validate_features(frame, schema)
    assert lenient.ok and any("platinum" in w for w in lenient.warnings)
    strict = validate_features(frame, schema, strict_categories=True)
    assert not strict.ok and "platinum" in strict.errors[0]


def test_missing_values_are_reported_not_rejected(schema):
    frame = good_frame()
    frame.loc[0, "balance"] = np.nan
    frame.loc[1, "plan"] = None
    report = validate_features(frame, schema)
    assert report.ok
    assert sum("missing value" in w for w in report.warnings) == 2


def test_empty_frame_and_input_not_mutated(schema):
    assert not validate_features(good_frame().iloc[0:0], schema).ok
    frame = good_frame()
    snapshot = frame.copy()
    validate_features(frame, schema)
    pd.testing.assert_frame_equal(frame, snapshot)
