"""Tests for the Portfolio class."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.assets import Asset
from waypoint.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(name: str, ticker: str, n: int = 100, seed: int = 42) -> Asset:
    rng = np.random.default_rng(seed=seed)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = rng.normal(0.0003, 0.01, n).tolist()
    returns = pl.DataFrame({"date": dates, "returns": values})
    return Asset(name=name, ticker=ticker, returns=returns, frequency="daily")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_portfolio_construction() -> None:
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=7)
    p = Portfolio({"eq": eq, "fi": fi}, weights={"eq": 0.6, "fi": 0.4})
    assert p.name == ""
    assert set(p.names) == {"eq", "fi"}


def test_portfolio_name() -> None:
    eq = _make_asset("Equities", "SPY")
    p = Portfolio({"eq": eq}, weights={"eq": 1.0}, name="My Portfolio")
    assert p.name == "My Portfolio"


def test_weights_are_normalised() -> None:
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=7)
    p = Portfolio({"eq": eq, "fi": fi}, weights={"eq": 60.0, "fi": 40.0})
    assert abs(p.weights["eq"] - 0.6) < 1e-9
    assert abs(p.weights["fi"] - 0.4) < 1e-9


def test_weights_sum_to_one() -> None:
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=7)
    p = Portfolio({"eq": eq, "fi": fi}, weights={"eq": 0.6, "fi": 0.4})
    assert abs(sum(p.weights.values()) - 1.0) < 1e-9


def test_mismatched_keys_raises() -> None:
    eq = _make_asset("Equities", "SPY")
    with pytest.raises(ValueError, match="same keys"):
        Portfolio({"eq": eq}, weights={"bonds": 1.0})


def test_empty_slots_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Portfolio({}, weights={})


def test_zero_weight_sum_raises_when_normalizing() -> None:
    eq = _make_asset("Equities", "SPY")
    with pytest.raises(ValueError, match="sum to zero"):
        Portfolio({"eq": eq}, weights={"eq": 0.0})


def test_normalize_weights_false_preserves_weights() -> None:
    """Long-short portfolio: weights are stored as-is when normalize=False."""
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=7)
    # 150% long equity, 50% short bonds → net 100%, gross 200%
    p = Portfolio({"eq": eq, "fi": fi}, weights={"eq": 1.5, "fi": -0.5},
                  normalize_weights=False)
    assert abs(p.weights["eq"] - 1.5) < 1e-9
    assert abs(p.weights["fi"] - (-0.5)) < 1e-9


def test_normalize_weights_false_zero_sum_allowed() -> None:
    """Dollar-neutral portfolio: weights summing to zero is valid when not normalising."""
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=7)
    p = Portfolio({"eq": eq, "fi": fi}, weights={"eq": 1.0, "fi": -1.0},
                  normalize_weights=False)
    assert abs(sum(p.weights.values())) < 1e-9  # net zero


# ---------------------------------------------------------------------------
# get_returns
# ---------------------------------------------------------------------------

def test_get_returns_wide_schema() -> None:
    eq = _make_asset("Equities", "SPY", n=50)
    fi = _make_asset("Bonds", "AGG", n=50, seed=7)
    p = Portfolio({"eq": eq, "fi": fi}, weights={"eq": 0.6, "fi": 0.4})
    wide = p.get_returns()
    assert wide.columns == ["date", "eq", "fi"]
    assert wide["date"].dtype == pl.Date


def test_get_returns_aligned_on_date() -> None:
    """get_returns inner-joins on date — only overlapping dates survive."""
    dates_a = [date(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    dates_b = [date(2020, 1, 3) + timedelta(days=i) for i in range(5)]
    a = Asset(
        name="A", ticker="A",
        returns=pl.DataFrame({"date": dates_a, "returns": [0.01] * 5}),
        frequency="daily",
    )
    b = Asset(
        name="B", ticker="B",
        returns=pl.DataFrame({"date": dates_b, "returns": [0.02] * 5}),
        frequency="daily",
    )
    p = Portfolio({"a": a, "b": b}, weights={"a": 0.5, "b": 0.5})
    wide = p.get_returns()
    # Overlap is Jan 3–5 (3 rows)
    assert len(wide) == 3
    assert wide["date"][0] == date(2020, 1, 3)


def test_get_returns_date_filter() -> None:
    eq = _make_asset("Equities", "SPY", n=30)
    p = Portfolio({"eq": eq}, weights={"eq": 1.0})
    full = p.get_returns()
    filtered = p.get_returns(start=date(2020, 1, 5), end=date(2020, 1, 15))
    assert len(filtered) < len(full)
    assert filtered["date"].min() >= date(2020, 1, 5)
    assert filtered["date"].max() <= date(2020, 1, 15)


def test_get_returns_cached() -> None:
    """Second call with same range must not re-join — returns the cached object."""
    eq = _make_asset("Equities", "SPY", n=30)
    p = Portfolio({"eq": eq}, weights={"eq": 1.0})
    result1 = p.get_returns()
    result2 = p.get_returns()
    assert result1 is result2  # exact same object from cache


def test_assetdef_slots_require_dates() -> None:
    from waypoint.asset_def import AssetDef
    ad = AssetDef(name="X", symbol="SPY", vendor="yfinance", frequency="daily")
    p = Portfolio({"x": ad}, weights={"x": 1.0})
    with pytest.raises(ValueError, match="start and end are required"):
        p.get_returns()


# ---------------------------------------------------------------------------
# portfolio_returns
# ---------------------------------------------------------------------------

def test_portfolio_returns_schema() -> None:
    eq = _make_asset("Equities", "SPY", n=50)
    fi = _make_asset("Bonds", "AGG", n=50, seed=7)
    p = Portfolio({"eq": eq, "fi": fi}, weights={"eq": 0.6, "fi": 0.4})
    pr = p.portfolio_returns()
    assert pr.columns == ["date", "returns"]
    assert pr["returns"].dtype in (pl.Float32, pl.Float64)


def test_portfolio_returns_weighted_sum() -> None:
    """Portfolio return = weighted sum of asset returns on each date."""
    d = date(2020, 1, 2)
    a = Asset(
        name="A", ticker="A",
        returns=pl.DataFrame({"date": [d], "returns": [0.10]}),
        frequency="daily",
    )
    b = Asset(
        name="B", ticker="B",
        returns=pl.DataFrame({"date": [d], "returns": [0.20]}),
        frequency="daily",
    )
    p = Portfolio({"a": a, "b": b}, weights={"a": 0.5, "b": 0.5})
    pr = p.portfolio_returns()
    assert abs(pr["returns"][0] - 0.15) < 1e-9  # 0.5*0.10 + 0.5*0.20
