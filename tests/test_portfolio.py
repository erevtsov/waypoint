"""Tests for the Portfolio class."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.assets import Asset
from waypoint.enums import Frequency
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


# ---------------------------------------------------------------------------
# Resampling — frequency parameter and end-of-period date alignment
# ---------------------------------------------------------------------------


def _make_daily_asset(start: date, n_days: int, value: float = 0.001, seed: int = 1) -> Asset:
    """Daily asset spanning exactly *n_days* starting from *start*."""
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rng = np.random.default_rng(seed=seed)
    values = rng.normal(value, 0.005, n_days).tolist()
    return Asset(
        name="X", ticker="X",
        returns=pl.DataFrame({"date": dates, "returns": values}),
        frequency="daily",
    )


def test_monthly_resample_fewer_rows_than_daily() -> None:
    """Resampling daily → monthly should produce fewer rows."""
    asset = _make_daily_asset(date(2020, 1, 1), n_days=365)
    p = Portfolio({"X": asset}, weights={"X": 1.0})
    daily = p.get_returns()
    monthly = p.get_returns(frequency=Frequency.MONTHLY)
    assert len(monthly) < len(daily)
    assert len(monthly) <= 12  # at most 12 months in a year


def test_monthly_resample_end_of_month_dates() -> None:
    """Resampled monthly dates must be the last day of each month."""
    asset = _make_daily_asset(date(2022, 1, 1), n_days=365)
    p = Portfolio({"X": asset}, weights={"X": 1.0})
    monthly = p.get_returns(frequency=Frequency.MONTHLY)
    for d in monthly["date"].to_list():
        # The last day of a month has no tomorrow in the same month.
        next_day = d + timedelta(days=1)
        assert next_day.month != d.month, f"{d} is not the last day of its month"


def _make_quarterly_asset(name: str, ticker: str, n: int = 8) -> Asset:
    """Quarterly asset with *n* end-of-quarter observations starting 2020-Q1."""
    quarter_ends = [
        date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31),
        date(2021, 3, 31), date(2021, 6, 30), date(2021, 9, 30), date(2021, 12, 31),
    ]
    rng = np.random.default_rng(seed=99)
    values = rng.normal(0.02, 0.03, n).tolist()
    returns = pl.DataFrame({"date": quarter_ends[:n], "returns": values})
    return Asset(name=name, ticker=ticker, returns=returns, frequency="quarterly")


# ---------------------------------------------------------------------------
# native_frequency
# ---------------------------------------------------------------------------

def test_native_frequency_all_daily() -> None:
    """All-daily portfolio → native frequency is daily."""
    eq = _make_asset("Equities", "SPY")
    fi = _make_asset("Bonds", "AGG", seed=7)
    p = Portfolio({"eq": eq, "fi": fi}, weights={"eq": 0.6, "fi": 0.4})
    assert p.native_frequency == Frequency.DAILY


def test_native_frequency_mixed_daily_quarterly() -> None:
    """Portfolio with a quarterly slot → native frequency is quarterly."""
    daily = _make_asset("Equities", "SPY")
    quarterly = _make_quarterly_asset("HPI", "HPI")
    p = Portfolio({"d": daily, "q": quarterly}, weights={"d": 0.5, "q": 0.5})
    assert p.native_frequency == Frequency.QUARTERLY


def test_native_frequency_all_quarterly() -> None:
    """All-quarterly portfolio → native frequency is quarterly."""
    q1 = _make_quarterly_asset("HPI1", "A")
    q2 = _make_quarterly_asset("HPI2", "B", n=8)
    p = Portfolio({"q1": q1, "q2": q2}, weights={"q1": 0.5, "q2": 0.5})
    assert p.native_frequency == Frequency.QUARTERLY


def test_get_returns_raises_when_frequency_too_fine() -> None:
    """Requesting monthly from a portfolio with a quarterly slot raises ValueError."""
    daily = _make_asset("Equities", "SPY", n=500)
    quarterly = _make_quarterly_asset("HPI", "HPI")
    p = Portfolio({"d": daily, "q": quarterly}, weights={"d": 0.5, "q": 0.5})
    with pytest.raises(ValueError, match="finer"):
        p.get_returns(frequency=Frequency.MONTHLY)


def test_get_returns_raises_names_offending_slots() -> None:
    """The ValueError must name which slot(s) are too coarse."""
    daily = _make_asset("Equities", "SPY", n=500)
    quarterly = _make_quarterly_asset("Condo HPI", "HPI")
    p = Portfolio(
        {"Equities": daily, "Condo HPI": quarterly},
        weights={"Equities": 0.5, "Condo HPI": 0.5},
    )
    with pytest.raises(ValueError, match="Condo HPI"):
        p.get_returns(frequency=Frequency.DAILY)


def test_get_returns_at_native_frequency_does_not_raise() -> None:
    """Requesting exactly the native frequency must succeed."""
    daily = _make_asset("Equities", "SPY", n=500)
    quarterly = _make_quarterly_asset("HPI", "HPI")
    p = Portfolio({"d": daily, "q": quarterly}, weights={"d": 0.5, "q": 0.5})
    result = p.get_returns(frequency=Frequency.QUARTERLY)
    assert len(result) > 0


def test_quarterly_asset_accepted() -> None:
    """Asset with frequency='quarterly' should be constructed without error."""
    dates = [date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31)]
    asset = Asset(
        name="Q", ticker="Q",
        returns=pl.DataFrame({"date": dates, "returns": [0.02] * 4}),
        frequency="quarterly",
    )
    assert asset.frequency == Frequency.QUARTERLY
    assert asset.periods_per_year == 4


def test_quarterly_resample_fewer_rows_than_monthly() -> None:
    """Resampling daily → quarterly should produce fewer rows than monthly."""
    asset = _make_daily_asset(date(2018, 1, 1), n_days=730)
    p = Portfolio({"X": asset}, weights={"X": 1.0})
    monthly = p.get_returns(frequency=Frequency.MONTHLY)
    quarterly = p.get_returns(frequency=Frequency.QUARTERLY)
    assert len(quarterly) < len(monthly)


def test_quarterly_resample_end_of_quarter_dates() -> None:
    """Resampled quarterly dates must be the last day of each quarter (Mar 31, Jun 30, ...)."""
    # Two full years of daily data starting Jan 1 — polars quarters start Jan 1
    asset = _make_daily_asset(date(2021, 1, 1), n_days=730)
    p = Portfolio({"X": asset}, weights={"X": 1.0})
    quarterly = p.get_returns(frequency=Frequency.QUARTERLY)
    quarter_ends = {(3, 31), (6, 30), (9, 30), (12, 31)}
    for d in quarterly["date"].to_list():
        assert (d.month, d.day) in quarter_ends, f"Unexpected quarter-end date: {d}"


def test_mixed_frequency_portfolio_resamples_daily_before_join() -> None:
    """A portfolio with daily + quarterly assets at quarterly frequency should
    compound the daily asset over each quarter, not sample a single daily return."""
    # Quarterly asset: 4 end-of-quarter dates with a fixed 2% quarterly return
    q_dates = [date(2021, 3, 31), date(2021, 6, 30), date(2021, 9, 30), date(2021, 12, 31)]
    quarterly_asset = Asset(
        name="Q", ticker="Q",
        returns=pl.DataFrame({"date": q_dates, "returns": [0.02] * 4}),
        frequency="quarterly",
    )
    # Daily asset: every calendar day in 2021 with constant 0.001 daily return
    daily_r = 0.001
    d_dates = [date(2021, 1, 1) + timedelta(days=i) for i in range(365)]
    daily_asset = Asset(
        name="D", ticker="D",
        returns=pl.DataFrame({"date": d_dates, "returns": [daily_r] * 365}),
        frequency="daily",
    )
    p = Portfolio({"Q": quarterly_asset, "D": daily_asset}, weights={"Q": 0.5, "D": 0.5})
    wide = p.get_returns(frequency=Frequency.QUARTERLY)

    # Must have 4 quarterly rows (not fewer due to sparse inner-join before resampling)
    assert len(wide) == 4, f"Expected 4 quarterly rows, got {len(wide)}"
    # Quarterly asset values should be unchanged (single observation per quarter)
    for val in wide["Q"].to_list():
        assert abs(val - 0.02) < 1e-9, f"Quarterly asset return should be 0.02, got {val}"
    # Daily asset should be compounded, not a single-day sample (~0.001)
    for val in wide["D"].to_list():
        assert val > 0.05, (
            f"Daily asset should compound to >5% per quarter, got {val:.4f}. "
            "Did the daily return get sampled at a single date instead of compounded?"
        )


def test_quarterly_resample_compounds_correctly() -> None:
    """A constant daily return should compound correctly to quarterly."""
    # 63 trading days per quarter in our 252-day/year model; use exact daily returns
    daily_r = 0.001  # 0.1% per day
    n_days = 63
    dates = [date(2021, 1, 1) + timedelta(days=i) for i in range(n_days)]
    asset = Asset(
        name="X", ticker="X",
        returns=pl.DataFrame({"date": dates, "returns": [daily_r] * n_days}),
        frequency="daily",
    )
    p = Portfolio({"X": asset}, weights={"X": 1.0})
    quarterly = p.get_returns(frequency=Frequency.QUARTERLY)
    expected = (1 + daily_r) ** n_days - 1
    assert len(quarterly) == 1
    assert abs(quarterly["X"][0] - expected) < 1e-10
