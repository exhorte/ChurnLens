import numpy as np
import pytest

from churnlens.retention_simulator import (
    Assumptions,
    SimulationInputError,
    break_even_probability,
    break_even_success_rate,
    sensitivity_to_success_rate,
    simulate_strategies,
)

P = np.array([0.9, 0.5, 0.1])


def run(assumptions, proba=P, threshold=0.5, top_k=2, actual=None):
    table = simulate_strategies(proba, assumptions, risk_threshold=threshold, top_k=top_k, actual=actual)
    return table.set_index("strategy")


A = Assumptions(cost_per_contact=10, value_per_retained=100, success_rate=0.5)


# --------------------------------------------------------------------------- formula
def test_contact_everyone_matches_the_published_formula_by_hand():
    r = run(A).loc["Contact everyone"]
    # N=3, C=10 -> cost 30 ; R = 0.9+0.5+0.1 = 1.5 ; saved = Q*R = 0.75 ; gross = V*saved = 75
    assert r["contacted"] == 3 and r["total_cost"] == pytest.approx(30)
    assert r["expected_churners_targeted"] == pytest.approx(1.5)
    assert r["expected_saved"] == pytest.approx(0.75)
    assert r["gross_benefit"] == pytest.approx(75)
    assert r["net_benefit"] == pytest.approx(100 * 1.5 * 0.5 - 3 * 10) == pytest.approx(45)
    assert r["roi"] == pytest.approx(45 / 30)
    assert r["budget_used"] == pytest.approx(30)


def test_net_benefit_always_equals_V_R_Q_minus_N_C():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    a = Assumptions(7.5, 180.0, 0.35, budget=900, max_contacts=200)
    for _, r in simulate_strategies(p, a, risk_threshold=0.6, top_k=120).iterrows():
        expected = a.value_per_retained * r["expected_churners_targeted"] * a.success_rate \
            - r["contacted"] * a.cost_per_contact
        assert r["net_benefit"] == pytest.approx(expected)
        assert r["gross_benefit"] - r["total_cost"] == pytest.approx(r["net_benefit"])


def test_no_campaign_costs_and_earns_nothing_and_has_no_roi():
    r = run(A).loc["No campaign"]
    assert r["contacted"] == 0 and r["total_cost"] == 0 and r["net_benefit"] == 0
    assert r["expected_churners_targeted"] == 0 and r["roi"] is None or np.isnan(r["roi"])


# -------------------------------------------------------------------------- strategies
def test_top_k_contacts_the_highest_risk_customers():
    r = run(A, top_k=2).loc["Top 2 by risk"]
    assert r["contacted"] == 2 and r["expected_churners_targeted"] == pytest.approx(1.4)


def test_threshold_strategy_selects_scores_at_or_above_the_threshold():
    r = run(A, threshold=0.5).loc["Risk >= 0.50"]
    assert r["contacted"] == 2                               # 0.9 and 0.5 (>= is inclusive)
    assert run(A, threshold=0.95).loc["Risk >= 0.95"]["contacted"] == 0
    assert run(A, threshold=0.0).loc["Risk >= 0.00"]["contacted"] == 3


def test_budget_shapes_only_the_budget_limited_strategy_and_flags_the_others():
    tight = Assumptions(10, 100, 0.5, budget=25)              # floor(25/10) = 2 contacts
    table = run(tight)
    limited = table.loc["Budget-limited"]
    assert limited["contacted"] == 2 and limited["total_cost"] == pytest.approx(20)
    assert limited["within_limits"] and limited["budget_used"] <= tight.budget
    everyone = table.loc["Contact everyone"]
    assert everyone["contacted"] == 3 and everyone["total_cost"] == pytest.approx(30)
    assert not everyone["within_limits"]                       # visible, not silently capped
    assert table.loc["No campaign"]["within_limits"]
    # the budget-limited customers are the highest-risk ones
    assert limited["expected_churners_targeted"] == pytest.approx(0.9 + 0.5)


def test_max_contacts_limit_and_both_limits_together():
    one = run(Assumptions(10, 100, 0.5, max_contacts=1))
    assert one.loc["Budget-limited"]["contacted"] == 1
    assert one.loc["Contact everyone"]["contacted"] == 3 and not one.loc["Contact everyone"]["within_limits"]
    both = Assumptions(10, 100, 0.5, budget=1000, max_contacts=2)
    assert run(both).loc["Budget-limited"]["contacted"] == 2
    tight = Assumptions(10, 100, 0.5, budget=15, max_contacts=2)
    assert run(tight).loc["Budget-limited"]["contacted"] == 1       # the stricter limit wins


def test_strategies_within_the_limits_are_marked_feasible():
    loose = run(Assumptions(10, 100, 0.5, budget=1000, max_contacts=10))
    assert loose["within_limits"].all()
    unlimited = run(A)
    assert unlimited["within_limits"].all()
    assert unlimited.loc["Budget-limited"]["contacted"] == 3       # no limit set: nothing to limit


def test_zero_budget_means_the_budget_limited_strategy_contacts_nobody():
    table = run(Assumptions(10, 100, 0.5, budget=0))
    assert table.loc["Budget-limited"]["contacted"] == 0
    assert not table.loc["Contact everyone"]["within_limits"]


def test_free_contacts_ignore_the_budget_but_not_max_contacts():
    free = Assumptions(0, 100, 0.5, budget=1, max_contacts=2)
    table = run(free)
    limited = table.loc["Budget-limited"]
    assert limited["contacted"] == 2 and limited["total_cost"] == 0
    assert limited["roi"] is None or np.isnan(limited["roi"])
    assert not table.loc["Contact everyone"]["within_limits"]      # 3 contacts > max_contacts


def test_zero_success_rate_loses_exactly_the_campaign_cost():
    r = run(Assumptions(10, 100, 0.0)).loc["Contact everyone"]
    assert r["net_benefit"] == pytest.approx(-30) and r["roi"] == pytest.approx(-1.0)


def test_perfect_success_rate_and_zero_value_edge_cases():
    r = run(Assumptions(10, 100, 1.0)).loc["Contact everyone"]
    assert r["net_benefit"] == pytest.approx(100 * 1.5 - 30)
    assert run(Assumptions(10, 0, 0.5)).loc["Contact everyone"]["gross_benefit"] == 0


def test_ties_are_broken_deterministically_by_input_order():
    p = np.array([0.5, 0.5, 0.5, 0.9])
    y = np.array([1, 0, 0, 1])
    r = run(Assumptions(1, 1, 0.5), proba=p, top_k=2, actual=y).loc["Top 2 by risk"]
    assert r["observed_churners_targeted"] == 2        # 0.9 first, then the first 0.5 (a churner)


def test_observed_churners_are_a_historical_reference_only():
    y = np.array([1, 0, 0])
    r = run(A, actual=y)
    assert r.loc["Contact everyone"]["observed_churners_targeted"] == 1
    assert r.loc["No campaign"]["observed_churners_targeted"] == 0
    assert run(A)["observed_churners_targeted"].isna().all()


def test_break_even_probability_makes_a_single_contact_net_zero():
    p_star = break_even_probability(10, 100, 0.5)
    assert p_star == pytest.approx(0.2)
    r = run(A, proba=np.array([p_star]), top_k=1).loc["Contact everyone"]
    assert r["net_benefit"] == pytest.approx(0.0, abs=1e-9)
    assert run(A, proba=np.array([0.3]), top_k=1).loc["Contact everyone"]["net_benefit"] > 0
    assert run(A, proba=np.array([0.1]), top_k=1).loc["Contact everyone"]["net_benefit"] < 0
    assert break_even_probability(10, 0, 0.5) is None and break_even_probability(10, 100, 0) is None


def test_a_single_customer_works():
    r = run(A, proba=np.array([0.8]), top_k=1).loc["Contact everyone"]
    assert r["net_benefit"] == pytest.approx(100 * 0.8 * 0.5 - 10)


# ------------------------------------------------------------------- derived helpers
def test_sensitivity_is_monotonic_in_success_rate_and_crosses_zero_at_break_even():
    rng = np.random.default_rng(1)
    p = rng.random(300)
    long = sensitivity_to_success_rate(p, A, risk_threshold=0.5, top_k=100)
    for _, group in long.groupby("strategy"):
        net = group.sort_values("success_rate")["net_benefit"]
        assert net.is_monotonic_increasing
    everyone = simulate_strategies(p, A, risk_threshold=0.5, top_k=100).set_index("strategy").loc["Contact everyone"]
    q = break_even_success_rate(everyone["contacted"], everyone["expected_churners_targeted"], 10, 100)
    at_q = Assumptions(10, 100, q)
    net = simulate_strategies(p, at_q, risk_threshold=0.5, top_k=100).set_index("strategy").loc["Contact everyone", "net_benefit"]
    assert net == pytest.approx(0, abs=1e-6)
    assert break_even_success_rate(10, 0.0, 10, 100) is None


# ------------------------------------------------------------------- invalid inputs
@pytest.mark.parametrize(
    "kwargs",
    [
        {"cost_per_contact": -1}, {"value_per_retained": -5}, {"success_rate": 1.5},
        {"success_rate": -0.1}, {"budget": -10}, {"max_contacts": -3}, {"max_contacts": 2.5},
        {"cost_per_contact": float("nan")}, {"value_per_retained": float("inf")},
        {"success_rate": float("nan")},
    ],
)
def test_invalid_assumptions_are_rejected(kwargs):
    values = {"cost_per_contact": 10, "value_per_retained": 100, "success_rate": 0.5, **kwargs}
    with pytest.raises(SimulationInputError):
        run(Assumptions(**values))


@pytest.mark.parametrize(
    "proba",
    [[], [0.5, 1.2], [-0.1, 0.5], [0.5, np.nan], [0.5, np.inf], [[0.1, 0.2], [0.3, 0.4]]],
)
def test_invalid_probabilities_are_rejected(proba):
    with pytest.raises(SimulationInputError):
        run(A, proba=np.asarray(proba, dtype=float))


def test_invalid_threshold_top_k_and_actual_length_are_rejected():
    with pytest.raises(SimulationInputError, match="risk_threshold"):
        run(A, threshold=1.5)
    with pytest.raises(SimulationInputError, match="top_k"):
        run(A, top_k=-1)
    with pytest.raises(SimulationInputError, match="same length"):
        run(A, actual=np.array([1, 0]))
