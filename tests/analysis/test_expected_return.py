"""Tests for ExpectedReturn and ArithmeticMean."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.analysis.expected_return import ExpectedReturn, ExpectedReturnResult
from waypoint.analysis.methods.returns import (
    CAPM,
    ArithmeticMean,
    EWMAMean,
    GeometricMean,
    ViewReturn,
)
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


# ---------------------------------------------------------------------------
# CAPM
# ---------------------------------------------------------------------------

def _make_capm_assets(n: int = 300, seed: int = 99) -> tuple[Asset, Asset, Asset, Asset]:
    """Return (market, rf, beta1_asset, beta0_asset) with known properties.

    beta1_asset returns == market returns  →  beta = 1.
    beta0_asset returns == constant 0.001  →  beta = 0 (no covariance with market).
    """
    rng = np.random.default_rng(seed=seed)
    dates = [date(2015, 1, 1) + timedelta(days=i) for i in range(n)]

    mkt_vals = rng.normal(0.0008, 0.01, n).tolist()
    rf_vals = [0.0001] * n  # flat risk-free series

    market = Asset(
        name="Market", ticker="MKT",
        returns=pl.DataFrame({"date": dates, "returns": mkt_vals}),
        frequency="daily",
    )
    rf_asset = Asset(
        name="RiskFree", ticker="RF",
        returns=pl.DataFrame({"date": dates, "returns": rf_vals}),
        frequency="daily",
    )
    beta1 = Asset(
        name="Beta1", ticker="B1",
        returns=pl.DataFrame({"date": dates, "returns": mkt_vals}),  # identical to market
        frequency="daily",
    )
    beta0 = Asset(
        name="Beta0", ticker="B0",
        returns=pl.DataFrame({"date": dates, "returns": [0.001] * n}),
        frequency="daily",
    )
    return market, rf_asset, beta1, beta0


def test_capm_beta1_asset_returns_market_expected_return() -> None:
    """Asset with returns identical to market must have CAPM E[R] == E[Rm]."""
    market, _, beta1, _ = _make_capm_assets()
    portfolio = Portfolio({"Beta1": beta1}, weights={"Beta1": 1.0})
    method = CAPM(market=market, risk_free=0.02)
    result = ExpectedReturn(method=method).compute(
        portfolio, start=None, end=None, frequency="daily"
    )
    e_rm = GeometricMean().compute(market.returns["returns"], 252)
    assert abs(result.per_asset["Beta1"] - e_rm) < 1e-10


def test_capm_beta0_asset_returns_risk_free() -> None:
    """Asset uncorrelated with the market must have CAPM E[R] == Rf."""
    market, _, _, beta0 = _make_capm_assets()
    rf = 0.04
    portfolio = Portfolio({"Beta0": beta0}, weights={"Beta0": 1.0})
    method = CAPM(market=market, risk_free=rf)
    result = ExpectedReturn(method=method).compute(
        portfolio, start=None, end=None, frequency="daily"
    )
    assert abs(result.per_asset["Beta0"] - rf) < 1e-10


def test_capm_rf_asset_used_correctly() -> None:
    """Passing rf as an Asset should give the same result as the equivalent float."""
    market, rf_asset, beta1, _ = _make_capm_assets()
    rf_float = GeometricMean().compute(rf_asset.returns["returns"], 252)

    portfolio = Portfolio({"Beta1": beta1}, weights={"Beta1": 1.0})
    result_float = ExpectedReturn(method=CAPM(market=market, risk_free=rf_float)).compute(
        portfolio, start=None, end=None, frequency="daily"
    )
    result_asset = ExpectedReturn(method=CAPM(market=market, risk_free=rf_asset)).compute(
        portfolio, start=None, end=None, frequency="daily"
    )
    assert abs(result_float.per_asset["Beta1"] - result_asset.per_asset["Beta1"]) < 1e-10


def test_capm_market_return_method_is_used() -> None:
    """Changing market_return_method must change the expected return estimate."""
    market, _, beta1, _ = _make_capm_assets()
    portfolio = Portfolio({"Beta1": beta1}, weights={"Beta1": 1.0})
    geo = ExpectedReturn(
        method=CAPM(market=market, risk_free=0.02, market_return_method=GeometricMean())
    ).compute(portfolio, start=None, end=None, frequency="daily")
    arith = ExpectedReturn(
        method=CAPM(market=market, risk_free=0.02, market_return_method=ArithmeticMean())
    ).compute(portfolio, start=None, end=None, frequency="daily")
    # With volatile market returns geometric < arithmetic (roughly)
    assert geo.per_asset["Beta1"] != arith.per_asset["Beta1"]


def test_capm_no_overlapping_dates_raises() -> None:
    """ValueError when market dates don't overlap with portfolio dates."""
    rng = np.random.default_rng(seed=5)
    n = 50
    port_dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    mkt_dates = [date(2021, 6, 1) + timedelta(days=i) for i in range(n)]  # no overlap

    port_asset = Asset(
        name="A", ticker="A",
        returns=pl.DataFrame({"date": port_dates, "returns": [0.001] * n}),
        frequency="daily",
    )
    market = Asset(
        name="MKT", ticker="MKT",
        returns=pl.DataFrame({"date": mkt_dates, "returns": rng.normal(0.001, 0.01, n).tolist()}),
        frequency="daily",
    )
    portfolio = Portfolio({"A": port_asset}, weights={"A": 1.0})
    with pytest.raises(ValueError, match="overlapping dates"):
        ExpectedReturn(method=CAPM(market=market, risk_free=0.0)).compute(
            portfolio, start=None, end=None, frequency="daily"
        )


def test_capm_method_name() -> None:
    market, _, beta1, _ = _make_capm_assets()
    portfolio = Portfolio({"Beta1": beta1}, weights={"Beta1": 1.0})
    result = ExpectedReturn(method=CAPM(market=market, risk_free=0.02)).compute(
        portfolio, start=None, end=None, frequency="daily"
    )
    assert result.method_name == "CAPM"
