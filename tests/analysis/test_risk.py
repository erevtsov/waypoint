"""Tests for Risk and SampleCovariance."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.analysis.methods.risk import SampleCovariance, ViewRisk
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


# ---------------------------------------------------------------------------
# ViewRisk — historical correlations mode
# ---------------------------------------------------------------------------

def test_view_risk_historical_corr_shape() -> None:
    """ViewRisk must return a (n_assets, n_assets) matrix."""
    portfolio = _two_asset_portfolio()
    method = ViewRisk(volatilities={"Equities": 0.15, "Bonds": 0.06})
    result = Risk(method=method).compute(portfolio, start=None, end=None, frequency="daily")
    assert result.covariance.shape == (2, 2)


def test_view_risk_historical_corr_uses_custom_vols() -> None:
    """Diagonal volatilities in the result must match the user-specified values."""
    portfolio = _two_asset_portfolio()
    method = ViewRisk(volatilities={"Equities": 0.15, "Bonds": 0.06})
    result = Risk(method=method).compute(portfolio, start=None, end=None, frequency="daily")
    assert abs(result.volatilities["Equities"] - 0.15) < 1e-10
    assert abs(result.volatilities["Bonds"] - 0.06) < 1e-10


def test_view_risk_historical_corr_preserves_correlation_sign() -> None:
    """Off-diagonal sign of the covariance must come from historical correlations."""
    rng = np.random.default_rng(seed=5)
    n = 300
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    base = rng.normal(0.001, 0.01, n)
    # Positively correlated pair
    a = Asset(
        name="A", ticker="A",
        returns=pl.DataFrame({"date": dates, "returns": (base + rng.normal(0, 0.005, n)).tolist()}),
        frequency="daily",
    )
    b = Asset(
        name="B", ticker="B",
        returns=pl.DataFrame({"date": dates, "returns": (base + rng.normal(0, 0.005, n)).tolist()}),
        frequency="daily",
    )
    portfolio = Portfolio({"A": a, "B": b}, weights={"A": 0.5, "B": 0.5})
    method = ViewRisk(volatilities={"A": 0.20, "B": 0.10})
    result = Risk(method=method).compute(portfolio, start=None, end=None, frequency="daily")
    # Off-diagonal must be positive (positively correlated assets)
    sigma = result.covariance.to_numpy()
    assert sigma[0, 1] > 0, "Expected positive off-diagonal for positively correlated assets"


# ---------------------------------------------------------------------------
# ViewRisk — manual correlation matrix mode
# ---------------------------------------------------------------------------

def test_view_risk_manual_corr_uses_provided_matrix() -> None:
    """When correlation_matrix is supplied, historical data is not used for correlations."""
    portfolio = _two_asset_portfolio()
    manual_corr = np.array([[1.0, -0.5], [-0.5, 1.0]])  # imposed negative correlation
    method = ViewRisk(
        volatilities={"Equities": 0.15, "Bonds": 0.06},
        correlation_matrix=manual_corr,
    )
    result = Risk(method=method).compute(portfolio, start=None, end=None, frequency="daily")
    sigma = result.covariance.to_numpy()
    # Off-diagonal must be negative (imposed by manual_corr)
    assert sigma[0, 1] < 0


def test_view_risk_manual_corr_diagonal_matches_custom_vols() -> None:
    """Diagonal entries of the covariance must equal σ_i² for the user-specified vols."""
    portfolio = _two_asset_portfolio()
    manual_corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    method = ViewRisk(
        volatilities={"Equities": 0.15, "Bonds": 0.06},
        correlation_matrix=manual_corr,
    )
    result = Risk(method=method).compute(portfolio, start=None, end=None, frequency="daily")
    sigma = result.covariance.to_numpy()
    assert abs(sigma[0, 0] - 0.15**2) < 1e-10
    assert abs(sigma[1, 1] - 0.06**2) < 1e-10


# ---------------------------------------------------------------------------
# ViewRisk — mutual exclusion invariant
# ---------------------------------------------------------------------------

def test_view_risk_both_sources_raises() -> None:
    """Supplying both correlation_matrix and correlation_method must raise at construction."""
    with pytest.raises(ValueError, match="not both"):
        ViewRisk(
            volatilities={"Equities": 0.15, "Bonds": 0.06},
            correlation_matrix=np.eye(2),
            correlation_method=SampleCovariance(),
        )


def test_view_risk_neither_source_defaults_to_sample_covariance() -> None:
    """Omitting both sources must default to SampleCovariance."""
    method = ViewRisk(volatilities={"Equities": 0.15, "Bonds": 0.06})
    assert method.correlation_method is not None
    assert isinstance(method.correlation_method, SampleCovariance)
    assert method.correlation_matrix is None


# ---------------------------------------------------------------------------
# ViewRisk — for_portfolio classmethod
# ---------------------------------------------------------------------------

def test_view_risk_for_portfolio_validates_missing_vols() -> None:
    """for_portfolio raises ValueError when a slot is missing from volatilities."""
    portfolio = _two_asset_portfolio()
    with pytest.raises(ValueError, match="Bonds"):
        ViewRisk.for_portfolio(portfolio, volatilities={"Equities": 0.15})


def test_view_risk_for_portfolio_validates_corr_matrix_shape() -> None:
    """for_portfolio raises ValueError when correlation_matrix has wrong shape."""
    portfolio = _two_asset_portfolio()
    bad_corr = np.eye(3)  # 3×3 for a 2-asset portfolio
    with pytest.raises(ValueError, match="shape"):
        ViewRisk.for_portfolio(
            portfolio,
            volatilities={"Equities": 0.15, "Bonds": 0.06},
            correlation_matrix=bad_corr,
        )


def test_view_risk_for_portfolio_returns_valid_instance() -> None:
    """for_portfolio returns a usable ViewRisk when inputs are valid."""
    portfolio = _two_asset_portfolio()
    method = ViewRisk.for_portfolio(
        portfolio,
        volatilities={"Equities": 0.15, "Bonds": 0.06},
    )
    result = Risk(method=method).compute(portfolio, start=None, end=None, frequency="daily")
    assert abs(result.volatilities["Equities"] - 0.15) < 1e-10


def test_view_risk_missing_asset_at_compute_raises() -> None:
    """compute() raises ValueError when a portfolio asset is not in volatilities."""
    portfolio = _two_asset_portfolio()
    method = ViewRisk(volatilities={"Equities": 0.15})  # missing "Bonds"
    with pytest.raises(ValueError, match="Bonds"):
        Risk(method=method).compute(portfolio, start=None, end=None, frequency="daily")
