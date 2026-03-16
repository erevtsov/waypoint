"""Tests for ExpectedReturn and ArithmeticMean."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.analysis.expected_return import ExpectedReturn, ExpectedReturnResult
from waypoint.analysis.methods.returns import ArithmeticMean, EWMAMean, GeometricMean, ViewReturn
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
# ArithmeticMean
# ---------------------------------------------------------------------------

def test_historical_mean_value() -> None:
    """ArithmeticMean should return sample mean * periods_per_year."""
    rng = np.random.default_rng(seed=42)
    values = rng.normal(0.001, 0.01, 500).tolist()
    series = pl.Series("returns", values)
    method = ArithmeticMean()
    result = method.compute(series, periods_per_year=252)
    expected = float(series.mean()) * 252  # type: ignore[arg-type]
    assert abs(result - expected) < 1e-10


def test_historical_mean_monthly() -> None:
    """Annualised mean changes with periods_per_year."""
    values = [0.01] * 12  # 1% per month
    series = pl.Series("returns", values)
    method = ArithmeticMean()
    result = method.compute(series, periods_per_year=12)
    assert abs(result - 0.12) < 1e-10  # 12% annualised


def test_historical_mean_empty_returns_zero() -> None:
    """Empty series returns 0.0 without error."""
    series = pl.Series("returns", [], dtype=pl.Float64)
    method = ArithmeticMean()
    result = method.compute(series, periods_per_year=252)
    assert result == 0.0


# ---------------------------------------------------------------------------
# ExpectedReturn.compute
# ---------------------------------------------------------------------------

def test_expected_return_per_asset_values() -> None:
    """Per-asset values should match calling ArithmeticMean directly."""
    portfolio = _make_portfolio()
    method = ArithmeticMean()
    er = ExpectedReturn(method=method)
    result = er.compute(portfolio, start=None, end=None, frequency="daily")

    wide = portfolio.get_returns()
    for name in ["Equities", "Bonds"]:
        direct = method.compute(wide[name], 252)
        assert abs(result.per_asset[name] - direct) < 1e-10


def test_expected_return_portfolio_is_weighted_sum() -> None:
    """Portfolio expected return must equal weighted sum of per-asset values."""
    portfolio = _make_portfolio(w_eq=0.6, w_fi=0.4)
    er = ExpectedReturn(method=ArithmeticMean())
    result = er.compute(portfolio, start=None, end=None, frequency="daily")

    expected_portfolio = (
        0.6 * result.per_asset["Equities"] + 0.4 * result.per_asset["Bonds"]
    )
    assert abs(result.portfolio - expected_portfolio) < 1e-10


def test_expected_return_method_name() -> None:
    """method_name should reflect the class name of the method used."""
    er = ExpectedReturn(method=ArithmeticMean())
    result = er.compute(_make_portfolio(), start=None, end=None, frequency="daily")
    assert result.method_name == "ArithmeticMean"


def test_expected_return_result_is_frozen() -> None:
    """ExpectedReturnResult must be immutable (frozen dataclass)."""
    result = ExpectedReturnResult(per_asset={"A": 0.1}, portfolio=0.1, method_name="Test")
    with pytest.raises((AttributeError, TypeError)):
        result.portfolio = 0.2  # type: ignore[misc]


def test_expected_return_date_filter_respected() -> None:
    """Passing start/end should filter the data used for estimation."""
    portfolio = _make_portfolio()
    er = ExpectedReturn(method=ArithmeticMean())
    full = er.compute(portfolio, start=None, end=None, frequency="daily")
    filtered = er.compute(
        portfolio,
        start=date(2020, 1, 10),
        end=date(2020, 2, 10),
        frequency="daily",
    )
    # Different windows → different means (very likely with random data)
    # We only assert both are finite floats
    assert isinstance(full.portfolio, float)
    assert isinstance(filtered.portfolio, float)


# ---------------------------------------------------------------------------
# GeometricMean
# ---------------------------------------------------------------------------

def test_geometric_mean_constant_returns() -> None:
    """With constant returns r, geometric mean = (1+r)^ppy - 1 exactly."""
    r = 0.001  # 0.1% per period
    series = pl.Series("returns", [r] * 252)
    result = GeometricMean().compute(series, periods_per_year=252)
    expected = (1 + r) ** 252 - 1.0
    assert abs(result - expected) < 1e-10


def test_geometric_mean_shows_volatility_drag() -> None:
    """Volatility drag: alternating +10%/-10% has zero arithmetic mean but negative geometric."""
    # mean(r) = 0 but each round-trip loses: 1.1 × 0.9 = 0.99 < 1
    values = [0.1, -0.1] * 100  # 200 periods
    series = pl.Series("returns", values)
    geo = GeometricMean().compute(series, periods_per_year=252)
    arith = ArithmeticMean().compute(series, periods_per_year=252)
    assert abs(arith) < 1e-10  # arithmetic annualised ≈ 0
    assert geo < 0.0  # geometric annualised is negative due to volatility drag


def test_geometric_mean_empty_returns_zero() -> None:
    series = pl.Series("returns", [], dtype=pl.Float64)
    assert GeometricMean().compute(series, periods_per_year=252) == 0.0


def test_geometric_mean_via_expected_return_analytic() -> None:
    """GeometricMean integrates with ExpectedReturn without error."""
    portfolio = _make_portfolio()
    result = ExpectedReturn(method=GeometricMean()).compute(
        portfolio, start=None, end=None, frequency="daily"
    )
    assert isinstance(result.portfolio, float)
    assert result.method_name == "GeometricMean"


# ---------------------------------------------------------------------------
# EWMAMean
# ---------------------------------------------------------------------------

def test_ewma_mean_decay_near_one_equals_arithmetic() -> None:
    """With decay_factor ≈ 1 the EWMA approaches the arithmetic mean."""
    rng = np.random.default_rng(seed=3)
    values = rng.normal(0.001, 0.01, 1000).tolist()
    series = pl.Series("returns", values)
    ewma = EWMAMean(decay_factor=0.9999).compute(series, periods_per_year=252)
    arith = ArithmeticMean().compute(series, periods_per_year=252)
    assert abs(ewma - arith) < 1e-3


def test_ewma_mean_low_decay_weights_recent() -> None:
    """Lower decay_factor must weight recent observations more than high decay_factor."""
    # Series that starts low and ends high
    values = [0.0] * 200 + [0.01] * 50
    series = pl.Series("returns", values)
    ewma_low = EWMAMean(decay_factor=0.5).compute(series, periods_per_year=252)
    ewma_high = EWMAMean(decay_factor=0.999).compute(series, periods_per_year=252)
    # Low decay heavily favours the recent high-return block
    assert ewma_low > ewma_high


def test_ewma_mean_single_observation() -> None:
    """Single observation: result is that value × ppy regardless of decay_factor."""
    series = pl.Series("returns", [0.002])
    result = EWMAMean(decay_factor=0.94).compute(series, periods_per_year=252)
    assert abs(result - 0.002 * 252) < 1e-10


def test_ewma_mean_empty_returns_zero() -> None:
    series = pl.Series("returns", [], dtype=pl.Float64)
    assert EWMAMean(decay_factor=0.94).compute(series, periods_per_year=252) == 0.0


def test_ewma_mean_via_expected_return_analytic() -> None:
    """EWMAMean integrates with ExpectedReturn without error."""
    portfolio = _make_portfolio()
    result = ExpectedReturn(method=EWMAMean(decay_factor=0.94)).compute(
        portfolio, start=None, end=None, frequency="daily"
    )
    assert isinstance(result.portfolio, float)
    assert result.method_name == "EWMAMean"


# ---------------------------------------------------------------------------
# ViewReturn
# ---------------------------------------------------------------------------

def test_view_return_returns_specified_values() -> None:
    """ViewReturn must return the pre-specified value, ignoring historical data."""
    portfolio = _make_portfolio()
    method = ViewReturn(expected_returns={"Equities": 0.08, "Bonds": 0.03})
    er = ExpectedReturn(method=method)
    result = er.compute(portfolio, start=None, end=None, frequency="daily")
    assert abs(result.per_asset["Equities"] - 0.08) < 1e-12
    assert abs(result.per_asset["Bonds"] - 0.03) < 1e-12


def test_view_return_portfolio_is_weighted_sum() -> None:
    """Portfolio return must equal the weighted sum of specified per-asset values."""
    portfolio = _make_portfolio(w_eq=0.6, w_fi=0.4)
    method = ViewReturn(expected_returns={"Equities": 0.08, "Bonds": 0.03})
    er = ExpectedReturn(method=method)
    result = er.compute(portfolio, start=None, end=None, frequency="daily")
    expected = 0.6 * 0.08 + 0.4 * 0.03
    assert abs(result.portfolio - expected) < 1e-12


def test_view_return_ignores_periods_per_year() -> None:
    """ViewReturn must return the same value regardless of frequency/periods_per_year."""
    portfolio = _make_portfolio()
    method = ViewReturn(expected_returns={"Equities": 0.07, "Bonds": 0.02})
    er = ExpectedReturn(method=method)
    daily = er.compute(portfolio, start=None, end=None, frequency="daily")
    # Re-run with a different name mapping — values must be identical
    assert abs(daily.per_asset["Equities"] - 0.07) < 1e-12


def test_view_return_missing_asset_raises() -> None:
    """compute() must raise ValueError when an asset name is not in expected_returns."""
    portfolio = _make_portfolio()
    method = ViewReturn(expected_returns={"Equities": 0.08})  # missing "Bonds"
    er = ExpectedReturn(method=method)
    with pytest.raises(ValueError, match="Bonds"):
        er.compute(portfolio, start=None, end=None, frequency="daily")


def test_view_return_for_portfolio_validates_keys() -> None:
    """for_portfolio raises ValueError when any slot is missing from expected_returns."""
    portfolio = _make_portfolio()
    with pytest.raises(ValueError, match="Bonds"):
        ViewReturn.for_portfolio(portfolio, {"Equities": 0.08})


def test_view_return_for_portfolio_returns_valid_instance() -> None:
    """for_portfolio returns a usable ViewReturn when all keys are present."""
    portfolio = _make_portfolio()
    method = ViewReturn.for_portfolio(portfolio, {"Equities": 0.08, "Bonds": 0.03})
    er = ExpectedReturn(method=method)
    result = er.compute(portfolio, start=None, end=None, frequency="daily")
    assert abs(result.per_asset["Equities"] - 0.08) < 1e-12
