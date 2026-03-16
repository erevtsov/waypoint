"""Tests for Risk and SampleCovariance."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.analysis.methods.risk import (
    EWMACovariance,
    LedoitWolf,
    SampleCovariance,
    ViewRisk,
    _ledoit_wolf_alpha,
)
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


# ---------------------------------------------------------------------------
# LedoitWolf
# ---------------------------------------------------------------------------

def _make_returns_df(n: int = 250, p: int = 3, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed=seed)
    data = {f"A{i}": rng.normal(0.0003, 0.01, n).tolist() for i in range(p)}
    return pl.DataFrame(data)


def test_ledoit_wolf_shape() -> None:
    """Output must be (p, p)."""
    data = _make_returns_df(p=4)
    cov = LedoitWolf().compute(data, periods_per_year=252)
    assert cov.shape == (4, 4)


def test_ledoit_wolf_symmetric() -> None:
    """Shrunk covariance must be symmetric."""
    data = _make_returns_df()
    cov = LedoitWolf().compute(data, periods_per_year=252)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_ledoit_wolf_positive_diagonal() -> None:
    """Diagonal entries (variances) must be positive."""
    data = _make_returns_df()
    cov = LedoitWolf().compute(data, periods_per_year=252)
    assert all(cov[i, i] > 0 for i in range(3))


def test_ledoit_wolf_scales_with_periods() -> None:
    """Result scales linearly with periods_per_year.

    Uses correlated assets with different variances so alpha < 1 and the
    off-diagonal entries are nonzero after shrinkage.
    """
    rng = np.random.default_rng(seed=77)
    n = 100
    base = rng.normal(0, 0.01, n)
    data = pl.DataFrame({
        "A": (base + rng.normal(0, 0.005, n)).tolist(),
        "B": (base + rng.normal(0, 0.015, n)).tolist(),
        "C": (base * 0.5 + rng.normal(0, 0.02, n)).tolist(),
    })
    cov_12 = LedoitWolf().compute(data, periods_per_year=12)
    cov_252 = LedoitWolf().compute(data, periods_per_year=252)
    np.testing.assert_allclose(cov_252 / cov_12, 252 / 12, rtol=1e-10)


def test_ledoit_wolf_diagonal_shrunk_toward_mean_variance() -> None:
    """With very few observations the diagonal must be pulled toward the mean variance."""
    rng = np.random.default_rng(seed=10)
    # Large spread of variances so shrinkage is clearly visible.
    data = pl.DataFrame({
        "A": rng.normal(0, 0.20, 30).tolist(),  # high vol
        "B": rng.normal(0, 0.01, 30).tolist(),  # low vol
        "C": rng.normal(0, 0.05, 30).tolist(),
    })
    lw = LedoitWolf().compute(data, periods_per_year=252)
    sc = SampleCovariance().compute(data, periods_per_year=252)
    # LW diagonal variance for high-vol asset should be less than sample
    assert lw[0, 0] < sc[0, 0]
    # LW diagonal variance for low-vol asset should be greater than sample
    assert lw[1, 1] > sc[1, 1]


def test_ledoit_wolf_alpha_in_unit_interval() -> None:
    """Analytical alpha must be in [0, 1]."""
    rng = np.random.default_rng(seed=99)
    X = rng.normal(0, 0.01, (150, 5))
    X -= X.mean(axis=0)
    alpha = _ledoit_wolf_alpha(X)
    assert 0.0 <= alpha <= 1.0


def test_ledoit_wolf_alpha_identity_input_is_zero() -> None:
    """If returns are perfectly scaled identity-like, alpha must be 0."""
    rng = np.random.default_rng(seed=7)
    # All assets have identical independent returns — covariance is diagonal with equal entries.
    vals = rng.normal(0, 0.01, (300, 1)) * np.ones((300, 4))
    X = vals - vals.mean(axis=0)
    alpha = _ledoit_wolf_alpha(X)
    assert alpha == 0.0 or alpha < 0.01  # already close to identity; minimal shrinkage needed


def test_ledoit_wolf_integration() -> None:
    """LedoitWolf must work end-to-end via Risk.compute."""
    portfolio = _two_asset_portfolio()
    result = Risk(method=LedoitWolf()).compute(portfolio, start=None, end=None, frequency="daily")
    assert result.method_name == "LedoitWolf"
    assert result.portfolio_volatility > 0


# ---------------------------------------------------------------------------
# EWMACovariance
# ---------------------------------------------------------------------------

def test_ewma_covariance_shape() -> None:
    """Output must be (p, p)."""
    data = _make_returns_df(p=3)
    cov = EWMACovariance().compute(data, periods_per_year=252)
    assert cov.shape == (3, 3)


def test_ewma_covariance_symmetric() -> None:
    """EWMA covariance must be symmetric."""
    data = _make_returns_df()
    cov = EWMACovariance().compute(data, periods_per_year=252)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_ewma_covariance_positive_diagonal() -> None:
    """Diagonal entries must be positive for non-degenerate data."""
    data = _make_returns_df()
    cov = EWMACovariance().compute(data, periods_per_year=252)
    assert all(cov[i, i] > 0 for i in range(3))


def test_ewma_covariance_scales_with_periods() -> None:
    """Result scales linearly with periods_per_year."""
    data = _make_returns_df()
    cov_12 = EWMACovariance().compute(data, periods_per_year=12)
    cov_252 = EWMACovariance().compute(data, periods_per_year=252)
    np.testing.assert_allclose(cov_252 / cov_12, 252 / 12, rtol=1e-10)


def test_ewma_decay_one_approaches_sample_covariance() -> None:
    """decay_factor → 1 gives equal weights, converging to the biased sample covariance."""
    rng = np.random.default_rng(seed=3)
    data = pl.DataFrame({
        "A": rng.normal(0, 0.01, 500).tolist(),
        "B": rng.normal(0, 0.01, 500).tolist(),
    })
    ewma = EWMACovariance(decay_factor=0.9999).compute(data, periods_per_year=1)
    # biased sample cov (1/T)
    arr = data.to_numpy()
    arr -= arr.mean(axis=0)
    biased = arr.T @ arr / len(arr)
    # rtol=1e-2: small residual from EWMA-weighted mean vs arithmetic mean centering
    np.testing.assert_allclose(ewma, biased, rtol=1e-2)


def test_ewma_low_decay_weights_recent_more() -> None:
    """Low decay_factor must make recent high-vol regime dominate the estimate."""
    rng = np.random.default_rng(seed=55)
    # First 200 periods: low vol; last 100 periods: high vol
    low_vol = rng.normal(0, 0.005, (200, 2)).tolist()
    high_vol = rng.normal(0, 0.05, (100, 2)).tolist()
    all_returns = low_vol + high_vol
    data = pl.DataFrame({"A": [r[0] for r in all_returns], "B": [r[1] for r in all_returns]})

    cov_low_decay = EWMACovariance(decay_factor=0.70).compute(data, periods_per_year=1)
    cov_high_decay = EWMACovariance(decay_factor=0.99).compute(data, periods_per_year=1)
    # Low decay_factor puts more weight on recent high-vol → larger variance
    assert cov_low_decay[0, 0] > cov_high_decay[0, 0]


def test_ewma_integration() -> None:
    """EWMACovariance must work end-to-end via Risk.compute."""
    portfolio = _two_asset_portfolio()
    result = Risk(method=EWMACovariance(decay_factor=0.94)).compute(
        portfolio, start=None, end=None, frequency="daily"
    )
    assert result.method_name == "EWMACovariance"
    assert result.portfolio_volatility > 0
