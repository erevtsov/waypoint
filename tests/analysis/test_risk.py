"""Tests for Risk and SampleCovariance."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.analysis.methods.risk import SampleCovariance
from waypoint.analysis.risk import Risk, RiskResult
from waypoint.assets import Asset
from waypoint.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(name: str, ticker: str, n: int = 200, seed: int = 42) -> Asset:
    rng = np.random.default_rng(seed=seed)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = rng.normal(0.0003, 0.01, n).tolist()
    returns = pl.DataFrame({"date": dates, "returns": values})
    return Asset(name=name, ticker=ticker, returns=returns, frequency="daily")


def _two_asset_portfolio(w1: float = 0.6, w2: float = 0.4) -> Portfolio:
    eq = _make_asset("Equities", "EQ", seed=1)
    fi = _make_asset("Bonds", "FI", seed=2)
    return Portfolio({"Equities": eq, "Bonds": fi}, weights={"Equities": w1, "Bonds": w2})


# ---------------------------------------------------------------------------
# SampleCovariance
# ---------------------------------------------------------------------------

def test_sample_covariance_shape() -> None:
    """Output matrix must be n_assets × n_assets."""
    rng = np.random.default_rng(seed=42)
    data = pl.DataFrame({
        "A": rng.normal(0.001, 0.01, 100).tolist(),
        "B": rng.normal(0.001, 0.01, 100).tolist(),
        "C": rng.normal(0.001, 0.01, 100).tolist(),
    })
    method = SampleCovariance()
    cov = method.compute(data, periods_per_year=252)
    assert cov.shape == (3, 3)


def test_sample_covariance_is_symmetric() -> None:
    """Covariance matrix must be symmetric."""
    rng = np.random.default_rng(seed=42)
    data = pl.DataFrame({
        "A": rng.normal(0.001, 0.01, 200).tolist(),
        "B": rng.normal(0.001, 0.01, 200).tolist(),
    })
    method = SampleCovariance()
    cov = method.compute(data, periods_per_year=252)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_sample_covariance_diagonal_positive() -> None:
    """Diagonal elements (variances) must be positive."""
    rng = np.random.default_rng(seed=42)
    data = pl.DataFrame({
        "A": rng.normal(0.001, 0.01, 200).tolist(),
        "B": rng.normal(0.001, 0.01, 200).tolist(),
    })
    method = SampleCovariance()
    cov = method.compute(data, periods_per_year=252)
    assert all(cov[i, i] > 0 for i in range(2))


def test_sample_covariance_scales_with_periods() -> None:
    """Covariance should scale linearly with periods_per_year."""
    rng = np.random.default_rng(seed=42)
    data = pl.DataFrame({
        "A": rng.normal(0.001, 0.01, 200).tolist(),
        "B": rng.normal(0.001, 0.01, 200).tolist(),
    })
    method = SampleCovariance()
    cov_12 = method.compute(data, periods_per_year=12)
    cov_252 = method.compute(data, periods_per_year=252)
    ratio = cov_252 / cov_12
    np.testing.assert_allclose(ratio, 252 / 12, rtol=1e-10)


# ---------------------------------------------------------------------------
# Risk.compute
# ---------------------------------------------------------------------------

def test_risk_covariance_columns_are_asset_names() -> None:
    """Covariance DataFrame columns must be the asset names."""
    portfolio = _two_asset_portfolio()
    risk = Risk(method=SampleCovariance())
    result = risk.compute(portfolio, start=None, end=None, frequency="daily")
    assert result.covariance.columns == ["Equities", "Bonds"]


def test_risk_covariance_shape_matches_assets() -> None:
    """Covariance DataFrame must be square with n_assets rows and columns."""
    portfolio = _two_asset_portfolio()
    risk = Risk(method=SampleCovariance())
    result = risk.compute(portfolio, start=None, end=None, frequency="daily")
    n = len(portfolio.names)
    assert result.covariance.shape == (n, n)


def test_risk_portfolio_volatility_non_negative() -> None:
    """Portfolio volatility must always be >= 0."""
    portfolio = _two_asset_portfolio()
    risk = Risk(method=SampleCovariance())
    result = risk.compute(portfolio, start=None, end=None, frequency="daily")
    assert result.portfolio_volatility >= 0.0


def test_risk_per_asset_volatility_non_negative() -> None:
    """Per-asset volatilities must all be >= 0."""
    portfolio = _two_asset_portfolio()
    risk = Risk(method=SampleCovariance())
    result = risk.compute(portfolio, start=None, end=None, frequency="daily")
    for vol in result.volatilities.values():
        assert vol >= 0.0


def test_risk_portfolio_volatility_consistent() -> None:
    """Portfolio volatility = sqrt(w^T Sigma w) must match manual calculation."""
    portfolio = _two_asset_portfolio(w1=0.6, w2=0.4)
    risk = Risk(method=SampleCovariance())
    result = risk.compute(portfolio, start=None, end=None, frequency="daily")

    sigma = result.covariance.to_numpy()
    w = np.array([0.6, 0.4])
    manual_vol = float(np.sqrt(w @ sigma @ w))
    assert abs(result.portfolio_volatility - manual_vol) < 1e-10


def test_risk_method_name() -> None:
    """method_name should reflect the class name of the method used."""
    portfolio = _two_asset_portfolio()
    risk = Risk(method=SampleCovariance())
    result = risk.compute(portfolio, start=None, end=None, frequency="daily")
    assert result.method_name == "SampleCovariance"


def test_risk_result_is_frozen() -> None:
    """RiskResult must be immutable (frozen dataclass)."""
    cov = pl.DataFrame({"A": [0.01], "B": [0.005]})
    result = RiskResult(
        covariance=cov,
        volatilities={"A": 0.1, "B": 0.07},
        portfolio_volatility=0.09,
        method_name="Test",
    )
    with pytest.raises((AttributeError, TypeError)):
        result.portfolio_volatility = 0.5  # type: ignore[misc]
