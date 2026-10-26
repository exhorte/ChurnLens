import numpy as np
import pandas as pd
import pytest

from churnguard.evaluation import (
    ALL_METRICS,
    SegmentRule,
    assign_segments,
    bootstrap_intervals,
    calibration_summary,
    compute_metrics,
    confusion_counts,
    cv_scorers,
    recommend_threshold,
    segment_summary,
    threshold_table,
)

# Hand-checkable example: 3 churners, 5 stayers.
Y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.4, 0.6, 0.3, 0.2, 0.1, 0.05])


def test_metrics_match_hand_computation():
    m = compute_metrics(Y, P, threshold=0.5)
    # threshold 0.5 -> predicted [1,1,0,1,0,0,0,0]: tp=2 fp=1 fn=1 tn=4
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    # ROC-AUC: 14 of the 15 (churner, stayer) pairs are ranked correctly
    assert m["roc_auc"] == pytest.approx(14 / 15)
    # Average precision: precisions 1, 1 and 3/4 at the three recall steps
    assert m["average_precision"] == pytest.approx((1 + 1 + 0.75) / 3)
    assert m["brier_score"] == pytest.approx(np.mean((P - Y) ** 2))
    expected_log_loss = -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    assert m["log_loss"] == pytest.approx(expected_log_loss)
    assert set(m) == set(ALL_METRICS)


def test_threshold_changes_only_threshold_dependent_metrics():
    a, b = compute_metrics(Y, P, 0.3), compute_metrics(Y, P, 0.7)
    for free in ("roc_auc", "average_precision", "log_loss", "brier_score"):
        assert a[free] == b[free]
    # threshold 0.3 flags 5 customers (recall 1.0, precision 0.6); 0.7 flags 2 (recall 2/3, precision 1.0)
    assert (a["recall"], a["precision"]) == pytest.approx((1.0, 0.6))
    assert (b["recall"], b["precision"]) == pytest.approx((2 / 3, 1.0))


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError, match="both classes"):
        compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="same length"):
        compute_metrics([0, 1], [0.1, 0.2, 0.3])


def test_cv_scorers_cover_every_reported_metric():
    assert set(cv_scorers()) == set(ALL_METRICS)


def test_confusion_counts():
    assert confusion_counts(Y, P >= 0.5) == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}


def test_threshold_table_is_internally_consistent():
    table = threshold_table(Y, P, thresholds=[0.05, 0.25, 0.5, 0.75, 0.95])
    assert (table[["tp", "fp", "fn", "tn"]].sum(axis=1) == len(Y)).all()
    assert (table["flagged"] == table["tp"] + table["fp"]).all()
    assert table["flagged"].is_monotonic_decreasing                  # higher bar, fewer flagged
    assert table["recall"].is_monotonic_decreasing
    row = table.loc[table["threshold"] == 0.5].iloc[0]
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 1, 1, 4)
    assert row["flagged_share"] == pytest.approx(3 / 8)
    assert row["false_positive_rate"] == pytest.approx(1 / 5)
    assert row["false_negative_rate"] == pytest.approx(1 / 3)
    # nobody flagged: precision is undefined (NaN), not a silent 0 or 1
    assert np.isnan(table.iloc[-1]["precision"]) and table.iloc[-1]["flagged"] == 0


def test_recommended_threshold_sits_in_the_gap_of_separable_scores():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 60)
    p = np.concatenate([rng.uniform(0, 0.4, 60), rng.uniform(0.6, 1.0, 60)])
    rec = recommend_threshold(y, p, "f1")
    assert 0.4 <= rec.threshold <= 0.6 and rec.score == pytest.approx(1.0)
    assert "out-of-fold" in rec.description


def test_recall_weighted_criterion_never_picks_a_higher_threshold_than_f1():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.3).astype(int)
    p = np.clip(0.3 + 0.25 * (y - 0.3) + rng.normal(0, 0.2, 2000), 0.001, 0.999)
    f1, f2 = recommend_threshold(y, p, "f1"), recommend_threshold(y, p, "f2")
    assert f2.threshold <= f1.threshold
    assert recommend_threshold(y, p, "youden").criterion == "youden"


def test_unknown_threshold_criterion_is_rejected():
    with pytest.raises(ValueError, match="criterion"):
        recommend_threshold(Y, P, "accuracy")


def test_calibration_distinguishes_calibrated_from_inflated_probabilities():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(int)
    good = calibration_summary(y, p)
    assert good["expected_calibration_error"] < 0.03
    assert good["mean_predicted"] == pytest.approx(good["observed_rate"], abs=0.02)
    inflated = calibration_summary(y, np.clip(p * 0.5 + 0.4, 0, 1))
    assert inflated["expected_calibration_error"] > 0.1
    assert inflated["expected_calibration_error"] > 5 * good["expected_calibration_error"]
    assert inflated["mean_predicted"] > inflated["observed_rate"] + 0.1
    assert list(good["bins"].columns) == ["bin", "n", "mean_predicted", "observed_rate"]


def test_segments_follow_the_cutoffs_exactly():
    rule = SegmentRule(0.20, 0.50)
    seg = assign_segments([0.0, 0.19, 0.20, 0.49, 0.50, 1.0], rule)
    assert [str(s) for s in seg] == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_segment_summary_reports_observed_rates():
    rule = SegmentRule(0.3, 0.7)
    summary = segment_summary(Y, P, rule)
    assert summary["customers"].sum() == len(Y)
    assert summary["share"].sum() == pytest.approx(1.0)
    high = summary.set_index("segment").loc["High"]
    assert high["customers"] == 2 and high["observed_rate"] == 1.0
    assert isinstance(summary, pd.DataFrame)


def noisy_scores(n, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.3 + 0.3 * (y - 0.3) + rng.normal(0, 0.2, n), 0.001, 0.999)
    return y, p


def test_bootstrap_interval_brackets_the_point_estimate_and_is_reproducible():
    y, p = noisy_scores(1500)
    first = bootstrap_intervals(y, p, 0.35, n_boot=200, seed=1)
    again = bootstrap_intervals(y, p, 0.35, n_boot=200, seed=1)
    assert first == again
    point = compute_metrics(y, p, 0.35)
    for metric, (low, high) in first.items():
        assert low <= point[metric] <= high, metric
        assert low < high


def test_bootstrap_interval_narrows_with_more_customers():
    small = bootstrap_intervals(*noisy_scores(300), 0.35, n_boot=300)["roc_auc"]
    large = bootstrap_intervals(*noisy_scores(6000), 0.35, n_boot=300)["roc_auc"]
    assert (large[1] - large[0]) < (small[1] - small[0]) / 2


def test_bootstrap_matches_the_point_metrics_formulas_and_rejects_unknown_metrics():
    y, p = noisy_scores(800)
    assert bootstrap_intervals(y, p, 0.35, n_boot=50, metrics=("recall",))["recall"][0] > 0
    with pytest.raises(ValueError, match="Cannot bootstrap"):
        bootstrap_intervals(y, p, 0.35, n_boot=10, metrics=("log_loss",))
