import numpy as np
import pandas as pd
import pytest

from churnguard.error_analysis import (
    OUTCOMES,
    classify_outcomes,
    compare_groups,
    confusion_frame,
    error_metrics,
    error_profile,
)

Y = [1, 1, 0, 0, 1, 0]
P = [0.9, 0.2, 0.8, 0.1, 0.6, 0.4]


def test_outcomes_keep_score_threshold_prediction_and_actual_apart():
    out = classify_outcomes(Y, P, 0.5)
    assert out["outcome"].astype(str).tolist() == ["TP", "FN", "FP", "TN", "TP", "TN"]
    assert out["predicted_class"].tolist() == [1, 0, 1, 0, 1, 0]
    assert out["actual_class"].tolist() == Y
    assert (out["threshold"] == 0.5).all() and out["probability"].tolist() == P
    assert set(out["outcome"].cat.categories) == set(OUTCOMES)


def test_changing_the_threshold_moves_customers_between_outcomes():
    low = classify_outcomes(Y, P, 0.15)["outcome"].astype(str).tolist()
    high = classify_outcomes(Y, P, 0.85)["outcome"].astype(str).tolist()
    assert low == ["TP", "TP", "FP", "TN", "TP", "FP"]      # 0.2 and 0.4 now flagged
    assert high == ["TP", "FN", "TN", "TN", "FN", "TN"]


def test_error_metrics_counts_and_rates():
    m = error_metrics(classify_outcomes(Y, P, 0.5))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["false_negative_rate"] == pytest.approx(1 / 3)
    assert m["negative_predictive_value"] == pytest.approx(2 / 3)
    assert m["flagged_share"] == pytest.approx(3 / 6)
    assert m["recall"] + m["false_negative_rate"] == pytest.approx(1.0)


def test_rates_are_nan_not_zero_when_undefined():
    m = error_metrics(classify_outcomes([0, 0, 1, 1], [0.1, 0.2, 0.1, 0.3], 0.9))   # nobody flagged
    assert np.isnan(m["precision"]) and m["recall"] == 0.0


def test_confusion_frame_layout():
    cm = confusion_frame(classify_outcomes(Y, P, 0.5))
    assert cm.loc["Actual: churns", "Predicted: churns"] == 2      # TP
    assert cm.loc["Actual: churns", "Predicted: stays"] == 1       # FN
    assert cm.loc["Actual: stays", "Predicted: churns"] == 1       # FP
    assert cm.loc["Actual: stays", "Predicted: stays"] == 2        # TN
    assert cm.to_numpy().sum() == len(Y)


def planted_errors():
    """SYNTHETIC: missed churners (FN) have low tenure; caught churners (TP) have high tenure."""
    rng = np.random.default_rng(0)
    n = 400
    outcome = np.repeat(["TP", "FN", "TN", "FP"], n // 4)
    X = pd.DataFrame(
        {
            "tenure": np.where(outcome == "FN", rng.normal(5, 2, n), rng.normal(40, 5, n)),
            "plan": np.where(np.isin(outcome, ["FP"]), "pro", rng.choice(["basic", "plus"], n)),
            "noise": rng.normal(0, 1, n),
        }
    )
    frame = pd.DataFrame({"outcome": pd.Categorical(outcome, categories=list(OUTCOMES))})
    return X, frame


def test_error_profile_finds_the_planted_driver_of_false_negatives():
    X, outcomes = planted_errors()
    profile = error_profile(X, outcomes, "FN")
    assert profile.iloc[0]["feature"] == "tenure"
    assert profile.iloc[0]["effect_size"] < -2            # FN much lower than TP
    assert profile.set_index("feature").loc["noise", "effect_size"] == pytest.approx(0, abs=0.5)


def test_error_profile_finds_the_planted_categorical_driver_of_false_positives():
    X, outcomes = planted_errors()
    top = error_profile(X, outcomes, "FP").iloc[0]
    assert top["feature"] == "plan" and top["kind"] == "categorical" and top["effect_size"] > 0.5
    assert "'pro'" in top["detail"]


def test_error_profile_input_checks_and_empty_groups():
    X, outcomes = planted_errors()
    with pytest.raises(ValueError, match="'FN' or 'FP'"):
        error_profile(X, outcomes, "TP")
    only_tp = pd.DataFrame({"outcome": pd.Categorical(["TP"] * 400, categories=list(OUTCOMES))})
    assert error_profile(X, only_tp, "FN").empty


def test_compare_groups_numeric_and_categorical():
    X = pd.DataFrame({"age": [20, 30, 40, 50], "plan": ["a", "a", "b", "b"]})
    first, second = pd.Series([True, True, False, False]), pd.Series([False, False, True, True])
    num = compare_groups(X, first, second, "age")
    assert num.set_index("group").loc["selected group", "mean"] == 25
    cat = compare_groups(X, first, second, "plan")
    assert cat.loc["a", "difference_pts"] == pytest.approx(100.0)
