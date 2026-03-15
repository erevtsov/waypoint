"""Tests for LeveragedAsset."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.assets import Asset, LeveragedAsset
from waypoint.enums import Frequency
from waypoint.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 120  # months of monthly data


def _make_asset(mean: float = 0.005, std: float = 0.02, seed: int = 7) -> Asset:
    """Monthly asset with N returns."""
    rng = np.random.default_rng(seed=seed)
    dates = [date(2015, 1, 1) + timedelta(days=30 * i) for i in range(N)]
    values = rng.normal(mean, std, N).tolist()
    return Asset(
        name="Base Asset",
        ticker="BA",
        returns=pl.DataFrame({"date": dates, "returns": values}),
        frequency="monthly",
    )


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_default_name_inherits_from_asset() -> None:
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06)
    assert la.name == asset.name


def test_custom_name_overrides() -> None:
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06, name="My Condo")
    assert la.name == "My Condo"


def test_invalid_leverage_ratio_raises() -> None:
    asset = _make_asset()
    with pytest.raises(ValueError, match="leverage_ratio"):
        LeveragedAsset(asset=asset, leverage_ratio=0.0, financing_cost=0.06)


def test_negative_leverage_ratio_raises() -> None:
    asset = _make_asset()
    with pytest.raises(ValueError, match="leverage_ratio"):
        LeveragedAsset(asset=asset, leverage_ratio=-1.0, financing_cost=0.06)


def test_negative_financing_cost_raises() -> None:
    asset = _make_asset()
    with pytest.raises(ValueError, match="financing_cost"):
        LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=-0.01)


# ---------------------------------------------------------------------------
# Metadata delegation
# ---------------------------------------------------------------------------


def test_ticker_delegates_to_asset() -> None:
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06)
    assert la.ticker == asset.ticker


def test_frequency_delegates_to_asset() -> None:
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06)
    assert la.frequency == Frequency.MONTHLY


def test_periods_per_year_delegates_to_asset() -> None:
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06)
    assert la.periods_per_year == 12


# ---------------------------------------------------------------------------
# Leverage formula
# ---------------------------------------------------------------------------


def test_no_leverage_returns_unchanged() -> None:
    """leverage_ratio=1.0 with any financing_cost leaves returns unchanged."""
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.0, financing_cost=0.10)
    np.testing.assert_array_almost_equal(
        la.returns["returns"].to_numpy(),
        asset.returns["returns"].to_numpy(),
    )


def test_leverage_amplifies_returns() -> None:
    """2x leverage doubles the asset return minus a financing drag."""
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=2.0, financing_cost=0.0)
    expected = asset.returns["returns"].to_numpy() * 2.0
    np.testing.assert_array_almost_equal(la.returns["returns"].to_numpy(), expected)


def test_financing_cost_creates_drag() -> None:
    """With positive financing_cost the leveraged return is less than leverage * asset return."""
    asset = _make_asset(mean=0.01)
    la_no_cost = LeveragedAsset(asset=asset, leverage_ratio=2.0, financing_cost=0.0)
    la_with_cost = LeveragedAsset(asset=asset, leverage_ratio=2.0, financing_cost=0.06)
    # Every period: la_with_cost = la_no_cost - borrowed * cost_per_period
    drag = 1.0 * (0.06 / 12)  # borrowed = leverage - 1 = 1
    np.testing.assert_array_almost_equal(
        la_with_cost.returns["returns"].to_numpy(),
        la_no_cost.returns["returns"].to_numpy() - drag,
    )


def test_leverage_formula_exact_values() -> None:
    """Spot-check the formula with deterministic single-row input."""
    single_row = pl.DataFrame(
        {"date": [date(2020, 1, 1)], "returns": [0.10]}  # 10% return
    )
    asset = Asset(name="X", ticker="X", returns=single_row, frequency="monthly")
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06)
    # r_lev = 1.5 * 0.10 - 0.5 * (0.06/12) = 0.15 - 0.0025 = 0.1475
    expected = 1.5 * 0.10 - 0.5 * (0.06 / 12)
    result = la.returns["returns"][0]
    assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# get_returns date filtering
# ---------------------------------------------------------------------------


def test_get_returns_filters_dates() -> None:
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06)
    start = date(2016, 1, 1)
    end = date(2016, 12, 31)
    filtered = la.get_returns(start, end)
    assert all(start <= d <= end for d in filtered["date"].to_list())
    assert len(filtered) > 0


def test_get_returns_has_correct_columns() -> None:
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06)
    result = la.get_returns(date(2016, 1, 1), date(2016, 12, 31))
    assert result.columns == ["date", "returns"]


# ---------------------------------------------------------------------------
# Portfolio integration
# ---------------------------------------------------------------------------


def test_leveraged_asset_in_portfolio() -> None:
    """LeveragedAsset should be accepted in Portfolio.slots."""
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=1.5, financing_cost=0.06, name="Condo")
    portfolio = Portfolio({"Condo": la}, weights={"Condo": 1.0})
    wide = portfolio.get_returns()
    assert "Condo" in wide.columns
    assert len(wide) == N


def test_portfolio_returns_with_leveraged_asset_match_formula() -> None:
    """Portfolio with one LeveragedAsset should equal the leveraged return series."""
    asset = _make_asset()
    la = LeveragedAsset(asset=asset, leverage_ratio=2.0, financing_cost=0.0, name="Condo")
    portfolio = Portfolio({"Condo": la}, weights={"Condo": 1.0})
    port_rets = portfolio.portfolio_returns()["returns"].to_numpy()
    expected = asset.returns["returns"].to_numpy() * 2.0
    np.testing.assert_array_almost_equal(port_rets, expected)
