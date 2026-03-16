"""Tests for Optimizer and EfficientFrontierResult."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from waypoint.analysis.expected_return import ExpectedReturn
from waypoint.analysis.methods.returns import ArithmeticMean
from waypoint.analysis.methods.risk import SampleCovariance
from waypoint.analysis.optimizer import EfficientFrontierResult, Optimizer
from waypoint.analysis.risk import Risk
from waypoint.assets import Asset
from waypoint.constraints import LongOnly, SumToOne
from waypoint.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERIODS_PER_YEAR = 252
N_PERIODS = 500
N_POINTS = 20  # fewer points for speed in tests


def _make_asset(name: str, ticker: str, mean: float, std: float, seed: int) -> Asset:
    rng = np.random.default_rng(seed=seed)
    dates = [date(2019, 1, 2) + timedelta(days=i) for i in range(N_PERIODS)]
    values = rng.normal(mean, std, N_PERIODS).tolist()
    return Asset(
        name=name, ticker=ticker,
        returns=pl.DataFrame({"date": dates, "returns": values}),
        frequency="daily",
    )


def _make_portfolio() -> Portfolio:
    eq = _make_asset("Equities", "EQ", mean=0.0004, std=0.012, seed=1)
    fi = _make_asset("Bonds", "FI", mean=0.0001, std=0.003, seed=2)
    alt = _make_asset("Alternatives", "ALT", mean=0.0003, std=0.007, seed=3)
    return Portfolio(
        {"Equities": eq, "Bonds": fi, "Alternatives": alt},
        weights={"Equities": 0.6, "Bonds": 0.3, "Alternatives": 0.1},
    )


def _make_optimizer() -> Optimizer:
    return Optimizer(
        return_model=ExpectedReturn(method=ArithmeticMean()),
        risk_model=Risk(method=SampleCovariance()),
        constraints=[LongOnly(), SumToOne()],
    )


# ---------------------------------------------------------------------------
# EfficientFrontierResult helpers
# ---------------------------------------------------------------------------

def test_frontier_has_correct_number_of_columns() -> None:
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    # expected_return + 3 asset columns
    assert set(result.weights.columns) == {"expected_return", "Equities", "Bonds", "Alternatives"}


def test_frontier_asset_names_match_portfolio() -> None:
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    assert set(result.asset_names) == {"Equities", "Bonds", "Alternatives"}


def test_frontier_risks_are_non_negative() -> None:
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    assert all(r >= 0.0 for r in result.risks.to_list())


def test_frontier_risks_are_non_decreasing() -> None:
    """Risks must be sorted ascending (frontier sorted by risk)."""
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    risks = result.risks.to_list()
    for i in range(1, len(risks)):
        assert risks[i] >= risks[i - 1] - 1e-8


def test_frontier_returns_are_non_decreasing() -> None:
    """Expected returns should be non-decreasing along the frontier."""
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    returns = result.expected_returns.to_list()
    for i in range(1, len(returns)):
        # Allow small numerical slack
        assert returns[i] >= returns[i - 1] - 1e-5


def test_frontier_weights_sum_to_one() -> None:
    """With SumToOne constraint, each row's weights must sum to 1."""
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    for row in result.weights.iter_rows(named=True):
        weight_sum = sum(row[name] for name in result.asset_names)
        assert abs(weight_sum - 1.0) < 1e-4


def test_frontier_weights_non_negative_with_long_only() -> None:
    """With LongOnly, all weights must be >= 0."""
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    for row in result.weights.iter_rows(named=True):
        for name in result.asset_names:
            assert row[name] >= -1e-5


# ---------------------------------------------------------------------------
# optimal_sharpe
# ---------------------------------------------------------------------------

def test_optimal_sharpe_returns_valid_weights() -> None:
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    sharpe_weights = result.optimal_sharpe(risk_free_rate=0.02)
    assert set(sharpe_weights.keys()) == set(result.asset_names)
    weight_sum = sum(sharpe_weights.values())
    assert abs(weight_sum - 1.0) < 1e-4


def test_optimal_sharpe_weights_are_floats() -> None:
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    sharpe_weights = result.optimal_sharpe()
    for v in sharpe_weights.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# portfolio_at / min_volatility_portfolio / max_sharpe_portfolio
# ---------------------------------------------------------------------------

def _make_frontier() -> tuple[Portfolio, EfficientFrontierResult]:
    portfolio = _make_portfolio()
    optimizer = _make_optimizer()
    result = optimizer.efficient_frontier(
        portfolio, start=None, end=None,
        frequency="daily", n_points=N_POINTS,
    )
    return portfolio, result


def test_portfolio_at_returns_portfolio_instance() -> None:
    from waypoint.portfolio import Portfolio as P
    source, result = _make_frontier()
    p = result.portfolio_at(source, 0)
    assert isinstance(p, P)


def test_portfolio_at_weights_match_frontier_row() -> None:
    source, result = _make_frontier()
    idx = 5
    p = result.portfolio_at(source, idx)
    row = result.weights.row(idx, named=True)
    for name in result.asset_names:
        assert abs(p.weights[name] - row[name]) < 1e-10


def test_portfolio_at_shares_asset_slots() -> None:
    """portfolio_at must reuse the source assets, not copy them."""
    source, result = _make_frontier()
    p = result.portfolio_at(source, 0)
    for name in result.asset_names:
        assert p.slots[name] is source.slots[name]


def test_min_volatility_portfolio_has_lowest_risk() -> None:
    source, result = _make_frontier()
    min_vol_p = result.min_volatility_portfolio(source)
    # Weights should match row 0 (lowest risk)
    row0 = result.weights.row(0, named=True)
    for name in result.asset_names:
        assert abs(min_vol_p.weights[name] - row0[name]) < 1e-10


def test_max_sharpe_portfolio_weights_match_optimal_sharpe() -> None:
    source, result = _make_frontier()
    rf = 0.02
    sharpe_weights = result.optimal_sharpe(risk_free_rate=rf)
    sharpe_p = result.max_sharpe_portfolio(source, risk_free_rate=rf)
    for name in result.asset_names:
        assert abs(sharpe_p.weights[name] - sharpe_weights[name]) < 1e-10


def test_max_sharpe_portfolio_weights_sum_to_one() -> None:
    source, result = _make_frontier()
    p = result.max_sharpe_portfolio(source)
    assert abs(sum(p.weights.values()) - 1.0) < 1e-4
