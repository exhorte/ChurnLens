"""Retention campaign simulator: a transparent scenario calculator, not a causal estimate.

Formulas (see docs/ECONOMIC_ASSUMPTIONS.md). For the set S of contacted customers:

    N      = |S|                                   customers contacted
    cost   = N * C                                 C = cost per contact          (currency)
    R      = sum_{i in S} p_i                      expected churners among S  (probabilities)
    saved  = Q * R                                 Q = assumed success rate in [0, 1]
    gross  = V * saved                             V = value of a retained customer (currency)
    net    = V * R * Q - N * C                     <- the simple model asked for
    ROI    = net / cost        (undefined if cost == 0)

A single contact has a positive expected net value only if p_i > C / (V * Q): the break-even
probability. It follows directly from the formulas, so it is only as good as C, V and Q.

``p_i`` must be calibrated probabilities for ``R`` to mean "expected churners"; a wrongly
calibrated model biases every figure. Results are conditional on the assumptions entered.
They are **not** measured effects: churn data alone cannot show what a campaign would change,
and not every at-risk customer can be retained (that is what Q is for).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


class SimulationInputError(ValueError):
    """An invalid assumption or probability vector."""


@dataclass(frozen=True)
class Assumptions:
    cost_per_contact: float            # C, currency per customer contacted
    value_per_retained: float          # V, currency per retained customer
    success_rate: float                # Q in [0, 1]
    budget: float | None = None        # total spend cap (None = no cap)
    max_contacts: int | None = None    # cap on contacted customers (None = no cap)

    def validate(self) -> None:
        for name in ("cost_per_contact", "value_per_retained"):
            value = getattr(self, name)
            if not _is_finite(value) or value < 0:
                raise SimulationInputError(f"{name} must be a finite number >= 0 (got {value!r}).")
        if not _is_finite(self.success_rate) or not 0.0 <= self.success_rate <= 1.0:
            raise SimulationInputError(f"success_rate must be in [0, 1] (got {self.success_rate!r}).")
        if self.budget is not None and (not _is_finite(self.budget) or self.budget < 0):
            raise SimulationInputError(f"budget must be >= 0 or None (got {self.budget!r}).")
        if self.max_contacts is not None and (
            int(self.max_contacts) != self.max_contacts or self.max_contacts < 0
        ):
            raise SimulationInputError(
                f"max_contacts must be a whole number >= 0 or None (got {self.max_contacts!r})."
            )

    def contact_cap(self, n_customers: int) -> int:
        """Largest number of contacts allowed by the budget and the max-contacts limit."""
        cap = n_customers
        if self.max_contacts is not None:
            cap = min(cap, int(self.max_contacts))
        if self.budget is not None and self.cost_per_contact > 0:
            cap = min(cap, int(math.floor(self.budget / self.cost_per_contact + 1e-9)))
        return max(cap, 0)


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    description: str
    contacted: int                      # N
    within_limits: bool                 # cost <= budget and N <= max_contacts (when those are set)
    total_cost: float                   # N * C
    budget_used: float                  # = total_cost (kept explicit for the comparison table)
    expected_churners_targeted: float   # R
    expected_saved: float               # Q * R
    gross_benefit: float                # V * Q * R
    net_benefit: float                  # gross - cost
    roi: float | None                   # net / cost, None when cost == 0
    observed_churners_targeted: int | None = None  # historical reference, not an effect

    def to_dict(self) -> dict:
        return asdict(self)


def _is_finite(value: float) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and math.isfinite(value)


def _check_probabilities(proba) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise SimulationInputError("Probabilities must be a non-empty 1-D sequence.")
    if not np.all(np.isfinite(p)):
        raise SimulationInputError("Probabilities contain missing or infinite values.")
    if p.min() < 0 or p.max() > 1:
        raise SimulationInputError("Probabilities must lie in [0, 1].")
    return p


def evaluate_selection(
    name: str,
    description: str,
    proba: np.ndarray,
    selected: np.ndarray,
    assumptions: Assumptions,
    actual: np.ndarray | None = None,
) -> StrategyResult:
    """Apply the formulas to a chosen set of customers (an array of row positions)."""
    n = len(selected)
    expected = float(proba[selected].sum()) if n else 0.0
    cost = n * assumptions.cost_per_contact
    saved = assumptions.success_rate * expected
    gross = assumptions.value_per_retained * saved
    net = gross - cost
    within_budget = assumptions.budget is None or cost <= assumptions.budget + 1e-9
    within_contacts = assumptions.max_contacts is None or n <= assumptions.max_contacts
    return StrategyResult(
        strategy=name,
        description=description,
        contacted=n,
        within_limits=bool(within_budget and within_contacts),
        total_cost=cost,
        budget_used=cost,
        expected_churners_targeted=expected,
        expected_saved=saved,
        gross_benefit=gross,
        net_benefit=net,
        roi=(net / cost) if cost > 0 else None,
        observed_churners_targeted=None if actual is None else int(actual[selected].sum()),
    )


def simulate_strategies(
    proba,
    assumptions: Assumptions,
    *,
    risk_threshold: float,
    top_k: int,
    actual=None,
) -> pd.DataFrame:
    """Compare the five standard strategies on the same customers and assumptions.

    1. no campaign          2. contact everyone
    3. top-k by risk        4. everyone with probability >= ``risk_threshold``
    5. budget-limited: the highest-risk customers until the budget or ``max_contacts`` is reached

    Only strategy 5 is *shaped* by the budget and contact limit. The others are shown as they
    are and flagged ``within_limits = False`` when they would exceed them, so the cost of ignoring
    a constraint stays visible. Customers are always contacted from the highest score down.
    ``actual`` (0/1 outcomes) is optional and only adds the historical count of churners among
    the contacted - a reference, not an effect.
    """
    assumptions.validate()
    p = _check_probabilities(proba)
    if not _is_finite(risk_threshold) or not 0.0 <= risk_threshold <= 1.0:
        raise SimulationInputError(f"risk_threshold must be in [0, 1] (got {risk_threshold!r}).")
    if int(top_k) != top_k or top_k < 0:
        raise SimulationInputError(f"top_k must be a whole number >= 0 (got {top_k!r}).")
    y = None
    if actual is not None:
        y = np.asarray(actual).astype(int)
        if y.shape != p.shape:
            raise SimulationInputError("actual and proba must have the same length.")

    n_customers = len(p)
    order = np.argsort(-p, kind="stable")          # highest risk first
    above_threshold = int((p >= risk_threshold).sum())
    plans = [
        ("No campaign", "Contact nobody (reference).", order[:0]),
        ("Contact everyone", "Contact every customer.", order),
        (f"Top {int(top_k)} by risk", "Contact the k customers with the highest churn probability.",
         order[: int(top_k)]),
        (f"Risk >= {risk_threshold:.2f}", "Contact customers whose probability reaches the threshold.",
         order[:above_threshold]),
        ("Budget-limited", "Contact the highest-risk customers until the budget or contact limit is reached.",
         order[: assumptions.contact_cap(n_customers)]),
    ]
    results = [evaluate_selection(name, desc, p, idx, assumptions, y) for name, desc, idx in plans]
    return pd.DataFrame([r.to_dict() for r in results])


def sensitivity_to_success_rate(
    proba,
    assumptions: Assumptions,
    *,
    risk_threshold: float,
    top_k: int,
    rates=None,
) -> pd.DataFrame:
    """Net benefit of each strategy as the assumed success rate Q varies (long format)."""
    grid = np.round(np.linspace(0.0, 1.0, 11), 2) if rates is None else np.asarray(rates, dtype=float)
    frames = []
    for q in grid:
        varied = Assumptions(
            assumptions.cost_per_contact, assumptions.value_per_retained, float(q),
            assumptions.budget, assumptions.max_contacts,
        )
        table = simulate_strategies(proba, varied, risk_threshold=risk_threshold, top_k=top_k)
        frames.append(table[["strategy", "net_benefit"]].assign(success_rate=float(q)))
    return pd.concat(frames, ignore_index=True)


def break_even_success_rate(contacted: int, expected_churners: float, cost_per_contact: float,
                            value_per_retained: float) -> float | None:
    """Q at which net benefit is zero: N*C / (V*R). ``None`` when V*R is zero."""
    denominator = value_per_retained * expected_churners
    if denominator <= 0:
        return None
    return float(contacted * cost_per_contact / denominator)


def break_even_probability(cost_per_contact: float, value_per_retained: float, success_rate: float) -> float | None:
    """Churn probability above which one more contact has a positive expected net value.

    p* = C / (V * Q). ``None`` when V * Q is zero (a contact can never pay off).
    """
    gain_if_churner = value_per_retained * success_rate
    if gain_if_churner <= 0:
        return None
    return float(cost_per_contact / gain_if_churner)
