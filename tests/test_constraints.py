"""Tests for portfolio constraint classes."""

from __future__ import annotations

import cvxpy as cp  # type: ignore[import-untyped]
import numpy as np

from waypoint.constraints import DEFAULT_CONSTRAINTS, LongOnly, SumToOne, WeightBounds

# ---------------------------------------------------------------------------
# LongOnly
# ---------------------------------------------------------------------------

def test_long_only_produces_one_constraint() -> None:
    w = cp.Variable(3)
    constraints = LongOnly().to_cvxpy(w, ["A", "B", "C"])
    assert len(constraints) == 1


def test_long_only_enforces_non_negative() -> None:
    """Optimal solution under LongOnly must have all weights >= 0."""
    w = cp.Variable(3)
    constraints = LongOnly().to_cvxpy(w, ["A", "B", "C"])
    constraints.append(cp.sum(w) == 1)
    # Minimise w[0] — should be driven to 0, not negative
    prob = cp.Problem(cp.Minimize(w[0]), constraints)
    prob.solve()
    assert w.value is not None
    assert all(v >= -1e-6 for v in w.value)


# ---------------------------------------------------------------------------
# WeightBounds
# ---------------------------------------------------------------------------

def test_weight_bounds_produces_two_constraints() -> None:
    w = cp.Variable(2)
    constraints = WeightBounds(min_weight=0.05, max_weight=0.6).to_cvxpy(w, ["A", "B"])
    assert len(constraints) == 2


def test_weight_bounds_respects_lower_bound() -> None:
    w = cp.Variable(2)
    lower = 0.1
    constraints = WeightBounds(min_weight=lower, max_weight=1.0).to_cvxpy(w, ["A", "B"])
    constraints.append(cp.sum(w) == 1)
    prob = cp.Problem(cp.Minimize(w[0]), constraints)
    prob.solve()
    assert w.value is not None
    assert w.value[0] >= lower - 1e-6


def test_weight_bounds_respects_upper_bound() -> None:
    w = cp.Variable(2)
    upper = 0.7
    constraints = WeightBounds(min_weight=0.0, max_weight=upper).to_cvxpy(w, ["A", "B"])
    constraints.append(cp.sum(w) == 1)
    prob = cp.Problem(cp.Maximize(w[0]), constraints)
    prob.solve()
    assert w.value is not None
    assert w.value[0] <= upper + 1e-6


def test_weight_bounds_default_values() -> None:
    wb = WeightBounds()
    assert wb.min_weight == 0.0
    assert wb.max_weight == 1.0


# ---------------------------------------------------------------------------
# SumToOne
# ---------------------------------------------------------------------------

def test_sum_to_one_produces_one_constraint() -> None:
    w = cp.Variable(3)
    constraints = SumToOne().to_cvxpy(w, ["A", "B", "C"])
    assert len(constraints) == 1


def test_sum_to_one_enforces_equality() -> None:
    """Optimal solution must have weights summing to exactly 1."""
    w = cp.Variable(3)
    constraints = SumToOne().to_cvxpy(w, ["A", "B", "C"])
    constraints.append(w >= 0)
    sigma = np.eye(3)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), constraints)
    prob.solve()
    assert w.value is not None
    assert abs(np.sum(w.value) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# DEFAULT_CONSTRAINTS
# ---------------------------------------------------------------------------

def test_default_constraints_are_long_only_and_sum_to_one() -> None:
    types = {type(c) for c in DEFAULT_CONSTRAINTS}
    assert LongOnly in types
    assert SumToOne in types


def test_default_constraints_combined() -> None:
    """Applying default constraints should yield non-negative weights summing to 1."""
    w = cp.Variable(3)
    all_constraints: list = []
    for c in DEFAULT_CONSTRAINTS:
        all_constraints.extend(c.to_cvxpy(w, ["A", "B", "C"]))
    sigma = np.eye(3)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), all_constraints)
    prob.solve()
    assert w.value is not None
    assert abs(np.sum(w.value) - 1.0) < 1e-5
    assert all(v >= -1e-6 for v in w.value)
