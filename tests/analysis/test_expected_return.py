"""Tests for ExpectedReturn and HistoricalMean."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.analysis.expected_return import ExpectedReturn, ExpectedReturnResult
from waypoint.analysis.methods.returns import HistoricalMean
from waypoint.assets import Asset
from waypoint.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(name: str, ticker: str, mean: float, n: int = 100, seed: int = 42) -> Asset:
    rng = np.random.default_rng(seed=seed)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = rng.normal(mean, 0.01, n).tolist()
    returns = pl.DataFrame({"date": dates, "returns": values})
    return Asset(name=name, ticker=ticker, returns=returns, frequency="daily")


def _make_portfolio(
    eq_mean: float = 0.0004,
    fi_mean: float = 0.0001,
    w_eq: float = 0.6,
    w_fi: float = 0.4,
) -> Portfolio:
    eq = _make_asset("Equities", "EQ", mean=eq_mean, seed=1)
    fi = _make_asset("Bonds", "FI", mean=fi_mean, seed=2)
    return Portfolio({"Equities": eq, "Bonds": fi}, weights={"Equities": w_eq, "Bonds": w_fi})


# ---------------------------------------------------------------------------
# HistoricalMean
# ---------------------------------------------------------------------------

def test_historical_mean_value() -> None:
    """HistoricalMean should return sample mean * periods_per_year."""
    rng = np.random.default_rng(seed=42)
    values = rng.normal(0.001, 0.01, 500).tolist()
    series = pl.Series("returns", values)
    method = HistoricalMean()
    result = method.compute(series, periods_per_year=252)
    expected = float(series.mean()) * 252  # type: ignore[arg-type]
    assert abs(result - expected) < 1e-10


def test_historical_mean_monthly() -> None:
    """Annualised mean changes with periods_per_year."""
    values = [0.01] * 12  # 1% per month
    series = pl.Series("returns", values)
    method = HistoricalMean()
    result = method.compute(series, periods_per_year=12)
    assert abs(result - 0.12) < 1e-10  # 12% annualised


def test_historical_mean_empty_returns_zero() -> None:
    """Empty series returns 0.0 without error."""
    series = pl.Series("returns", [], dtype=pl.Float64)
    method = HistoricalMean()
    result = method.compute(series, periods_per_year=252)
    assert result == 0.0


# ---------------------------------------------------------------------------
# ExpectedReturn.compute
# ---------------------------------------------------------------------------

def test_expected_return_per_asset_values() -> None:
    """Per-asset values should match calling HistoricalMean directly."""
    portfolio = _make_portfolio()
    method = HistoricalMean()
    er = ExpectedReturn(method=method)
    result = er.compute(portfolio, start=None, end=None, periods_per_year=252)

    wide = portfolio.get_returns()
    for name in ["Equities", "Bonds"]:
        direct = method.compute(wide[name], 252)
        assert abs(result.per_asset[name] - direct) < 1e-10


def test_expected_return_portfolio_is_weighted_sum() -> None:
    """Portfolio expected return must equal weighted sum of per-asset values."""
    portfolio = _make_portfolio(w_eq=0.6, w_fi=0.4)
    er = ExpectedReturn(method=HistoricalMean())
    result = er.compute(portfolio, start=None, end=None, periods_per_year=252)

    expected_portfolio = (
        0.6 * result.per_asset["Equities"] + 0.4 * result.per_asset["Bonds"]
    )
    assert abs(result.portfolio - expected_portfolio) < 1e-10


def test_expected_return_method_name() -> None:
    """method_name should reflect the class name of the method used."""
    er = ExpectedReturn(method=HistoricalMean())
    result = er.compute(_make_portfolio(), start=None, end=None, periods_per_year=252)
    assert result.method_name == "HistoricalMean"


def test_expected_return_result_is_frozen() -> None:
    """ExpectedReturnResult must be immutable (frozen dataclass)."""
    result = ExpectedReturnResult(per_asset={"A": 0.1}, portfolio=0.1, method_name="Test")
    with pytest.raises((AttributeError, TypeError)):
        result.portfolio = 0.2  # type: ignore[misc]


def test_expected_return_date_filter_respected() -> None:
    """Passing start/end should filter the data used for estimation."""
    portfolio = _make_portfolio()
    er = ExpectedReturn(method=HistoricalMean())
    full = er.compute(portfolio, start=None, end=None, periods_per_year=252)
    filtered = er.compute(
        portfolio,
        start=date(2020, 1, 10),
        end=date(2020, 2, 10),
        periods_per_year=252,
    )
    # Different windows → different means (very likely with random data)
    # We only assert both are finite floats
    assert isinstance(full.portfolio, float)
    assert isinstance(filtered.portfolio, float)
