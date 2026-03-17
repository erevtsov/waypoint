"""Tests for Aggregate and Portfolio.initial_wealth."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.aggregate import Aggregate
from waypoint.assets import Asset
from waypoint.portfolio import Portfolio


def _port(
    name: str,
    slots: dict,
    weights: dict,
    initial_wealth: float | None = None,
) -> Portfolio:
    return Portfolio(slots, weights=weights, name=name, initial_wealth=initial_wealth)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(name: str, ticker: str, n: int = 100, seed: int = 42) -> Asset:
    rng = np.random.default_rng(seed=seed)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = rng.normal(0.0003, 0.01, n).tolist()
    returns = pl.DataFrame({"date": dates, "returns": values})
    return Asset(name=name, ticker=ticker, returns=returns, frequency="daily")


def _make_portfolio(name: str, initial_wealth: float | None = None, seed: int = 42) -> Portfolio:
    eq = _make_asset("Equities", "SPY", seed=seed)
    fi = _make_asset("Bonds", "AGG", seed=seed + 1)
    return Portfolio(
        {"eq": eq, "fi": fi},
        weights={"eq": 0.6, "fi": 0.4},
        name=name,
        initial_wealth=initial_wealth,
    )


# ---------------------------------------------------------------------------
# Portfolio.initial_wealth
# ---------------------------------------------------------------------------

def test_portfolio_initial_wealth_defaults_none() -> None:
    p = _make_portfolio("taxable")
    assert p.initial_wealth is None


def test_portfolio_initial_wealth_set_at_construction() -> None:
    p = _make_portfolio("taxable", initial_wealth=500_000.0)
    assert p.initial_wealth == 500_000.0


def test_portfolio_initial_wealth_is_mutable() -> None:
    p = _make_portfolio("taxable")
    p.initial_wealth = 250_000.0
    assert p.initial_wealth == 250_000.0


# ---------------------------------------------------------------------------
# Aggregate construction
# ---------------------------------------------------------------------------

def test_aggregate_construction() -> None:
    taxable = _make_portfolio("taxable", 500_000.0)
    retirement = _make_portfolio("401k", 300_000.0, seed=10)
    agg = Aggregate([taxable, retirement])
    assert agg.names == ["taxable", "401k"]


def test_aggregate_requires_initial_wealth() -> None:
    p = _make_portfolio("taxable")  # initial_wealth=None
    with pytest.raises(ValueError, match="initial_wealth"):
        Aggregate([p])


def test_aggregate_requires_unique_names() -> None:
    p1 = _make_portfolio("taxable", 500_000.0)
    p2 = _make_portfolio("taxable", 300_000.0, seed=10)
    with pytest.raises(ValueError, match="Duplicates"):
        Aggregate([p1, p2])


def test_aggregate_requires_at_least_one_portfolio() -> None:
    with pytest.raises(ValueError):
        Aggregate([])


# ---------------------------------------------------------------------------
# wealth_weights
# ---------------------------------------------------------------------------

def test_wealth_weights_sum_to_one() -> None:
    taxable = _make_portfolio("taxable", 500_000.0)
    retirement = _make_portfolio("401k", 300_000.0, seed=10)
    roth = _make_portfolio("roth", 200_000.0, seed=20)
    agg = Aggregate([taxable, retirement, roth])
    weights = agg.wealth_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_wealth_weights_proportional() -> None:
    taxable = _make_portfolio("taxable", 600_000.0)
    retirement = _make_portfolio("401k", 400_000.0, seed=10)
    agg = Aggregate([taxable, retirement])
    weights = agg.wealth_weights()
    assert abs(weights["taxable"] - 0.6) < 1e-9
    assert abs(weights["401k"] - 0.4) < 1e-9


def test_wealth_weights_single_portfolio() -> None:
    p = _make_portfolio("taxable", 1_000_000.0)
    agg = Aggregate([p])
    assert agg.wealth_weights() == {"taxable": 1.0}


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------

def test_flatten_weights_sum_to_one() -> None:
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=1)
    taxable = _port("taxable", {"eq": eq, "fi": fi}, {"eq": 0.6, "fi": 0.4}, 600_000.0)
    retirement = _port("401k", {"eq": eq, "fi": fi}, {"eq": 0.5, "fi": 0.5}, 400_000.0)
    agg = Aggregate([taxable, retirement])
    flat = agg.flatten()
    assert abs(sum(flat.weights.values()) - 1.0) < 1e-9


def test_flatten_initial_wealth_is_total() -> None:
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=1)
    taxable = _port("taxable", {"eq": eq, "fi": fi}, {"eq": 0.6, "fi": 0.4}, 600_000.0)
    retirement = _port("401k", {"eq": eq, "fi": fi}, {"eq": 0.5, "fi": 0.5}, 400_000.0)
    agg = Aggregate([taxable, retirement])
    flat = agg.flatten()
    assert flat.initial_wealth == 1_000_000.0


def test_flatten_single_portfolio_preserves_weights() -> None:
    p = _make_portfolio("taxable", 500_000.0)
    agg = Aggregate([p])
    flat = agg.flatten()
    assert abs(flat.weights["eq"] - 0.6) < 1e-9
    assert abs(flat.weights["fi"] - 0.4) < 1e-9


def test_flatten_wealth_weighted_combined_weights() -> None:
    """Two equal-wealth portfolios with same slots: combined = average of each."""
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=1)
    p1 = _port("p1", {"eq": eq, "fi": fi}, {"eq": 0.8, "fi": 0.2}, 500_000.0)
    p2 = _port("p2", {"eq": eq, "fi": fi}, {"eq": 0.4, "fi": 0.6}, 500_000.0)
    agg = Aggregate([p1, p2])
    flat = agg.flatten()
    assert abs(flat.weights["eq"] - 0.6) < 1e-9
    assert abs(flat.weights["fi"] - 0.4) < 1e-9


def test_flatten_disjoint_slots_combined() -> None:
    """Portfolios with entirely different slots both appear in the flattened result."""
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=1)
    p1 = Portfolio({"eq": eq}, weights={"eq": 1.0}, name="p1", initial_wealth=600_000.0)
    p2 = Portfolio({"fi": fi}, weights={"fi": 1.0}, name="p2", initial_wealth=400_000.0)
    agg = Aggregate([p1, p2])
    flat = agg.flatten()
    assert abs(flat.weights["eq"] - 0.6) < 1e-9
    assert abs(flat.weights["fi"] - 0.4) < 1e-9


def test_flatten_conflicting_slot_names_raises() -> None:
    """Same slot name mapped to different assets raises ValueError."""
    eq1 = _make_asset("Equities", "SPY", seed=1)
    eq2 = _make_asset("Equities", "QQQ", seed=2)
    p1 = Portfolio({"eq": eq1}, weights={"eq": 1.0}, name="p1", initial_wealth=500_000.0)
    p2 = Portfolio({"eq": eq2}, weights={"eq": 1.0}, name="p2", initial_wealth=500_000.0)
    agg = Aggregate([p1, p2])
    with pytest.raises(ValueError, match="different assets"):
        agg.flatten()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_returns_per_portfolio_results() -> None:
    """run() calls analytic.compute per portfolio and keys by name."""

    class _MockAnalytic:
        def compute(self, portfolio: Portfolio, multiplier: float = 1.0) -> float:
            return portfolio.initial_wealth * multiplier  # type: ignore[operator]

    taxable = _make_portfolio("taxable", 500_000.0)
    retirement = _make_portfolio("401k", 300_000.0, seed=10)
    agg = Aggregate([taxable, retirement])
    results = agg.run(_MockAnalytic(), multiplier=2.0)
    assert results["taxable"] == 1_000_000.0
    assert results["401k"] == 600_000.0


def test_run_keys_match_portfolio_names() -> None:
    taxable = _make_portfolio("taxable", 500_000.0)
    retirement = _make_portfolio("401k", 300_000.0, seed=10)
    agg = Aggregate([taxable, retirement])

    class _MockAnalytic:
        def compute(self, portfolio: Portfolio) -> str:
            return portfolio.name

    results = agg.run(_MockAnalytic())
    assert set(results.keys()) == {"taxable", "401k"}
