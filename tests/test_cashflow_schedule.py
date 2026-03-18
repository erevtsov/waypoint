"""Tests for cashflow schedule computation on MultiWealthSimulationResult."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import waypoint as wp
from waypoint.analysis.simulation import _compute_annual_cashflows
from waypoint.assets import Asset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(name: str, ticker: str, n_months: int = 120, seed: int = 0) -> Asset:
    rng = np.random.default_rng(seed=seed)
    dates: list[date] = []
    for i in range(n_months):
        year = 2010 + (i // 12)
        month = (i % 12) + 1
        eom = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        dates.append(eom)
    returns = rng.normal(0.006, 0.03, n_months).tolist()
    return Asset(name=name, ticker=ticker,
                 returns=pl.DataFrame({"date": dates, "returns": returns}),
                 frequency="monthly")


# ---------------------------------------------------------------------------
# Fixtures — a minimal two-account simulation
# ---------------------------------------------------------------------------

HORIZON = 5  # years
N_SIMS = 100
INFLATION = 0.03


@pytest.fixture(scope="module")
def two_account_result():
    """Minimal two-account simulation with predictable cashflows."""
    eq = _make_asset("Equity", "EQ", seed=1)

    acct_a = wp.Portfolio(
        slots={"eq": eq}, weights={"eq": 1.0},
        name="acct_a", initial_wealth=100_000.0,
    )
    acct_b = wp.Portfolio(
        slots={"eq": eq}, weights={"eq": 1.0},
        name="acct_b", initial_wealth=50_000.0,
    )
    agg = wp.Aggregate([acct_a, acct_b])

    cashflows = {
        "acct_a": [
            # Fixed annual contribution of $10k real
            wp.cashflows.PeriodicCashflow(
                amount=10_000.0, frequency="annual", real=True
            ),
        ],
        "acct_b": [
            # Withdrawal of $5k real starting year 3
            wp.cashflows.PeriodicCashflow(
                amount=-5_000.0, frequency="annual", real=True, start_year=3.0
            ),
        ],
    }

    sim = wp.analytics.MultiWealthSimulation(
        method=wp.sim.MonteCarlo(seed=7),
        cashflows=cashflows,
        horizon_years=HORIZON,
        n_simulations=N_SIMS,
        inflation_rate=INFLATION,
    )
    return sim.compute(agg, start="2010-01-31", end="2019-12-31",
                       frequency="monthly", real=True)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_cashflow_schedule_row_count(two_account_result) -> None:
    """One row per year in the horizon."""
    df = two_account_result.cashflow_schedule
    assert len(df) == HORIZON


def test_cashflow_schedule_columns(two_account_result) -> None:
    """year, one per account, and total."""
    df = two_account_result.cashflow_schedule
    assert "year" in df.columns
    assert "acct_a" in df.columns
    assert "acct_b" in df.columns
    assert "total" in df.columns


def test_cashflow_schedule_year_column(two_account_result) -> None:
    """year column runs 1..horizon_years."""
    df = two_account_result.cashflow_schedule
    assert df["year"].to_list() == list(range(1, HORIZON + 1))


def test_cashflow_schedule_total_equals_sum(two_account_result) -> None:
    """total column is the row-wise sum of account columns."""
    df = two_account_result.cashflow_schedule
    recomputed = (df["acct_a"] + df["acct_b"]).to_list()
    np.testing.assert_allclose(df["total"].to_list(), recomputed, rtol=1e-9)


# ---------------------------------------------------------------------------
# Economic sanity
# ---------------------------------------------------------------------------

def test_acct_a_contributions_positive_all_years(two_account_result) -> None:
    """acct_a has only contributions — all annual values should be positive."""
    df = two_account_result.cashflow_schedule
    assert all(v > 0 for v in df["acct_a"].to_list())


def test_acct_b_zero_before_withdrawal_start(two_account_result) -> None:
    """acct_b has no cashflow before year 3."""
    df = two_account_result.cashflow_schedule
    # years 1 and 2 have no cashflows for acct_b
    early = df.filter(pl.col("year") < 3)["acct_b"].to_list()
    np.testing.assert_allclose(early, [0.0, 0.0], atol=1e-9)


def test_acct_b_withdrawals_negative_from_year_3(two_account_result) -> None:
    """acct_b withdrawals start in year 3 — those values must be negative."""
    df = two_account_result.cashflow_schedule
    late = df.filter(pl.col("year") >= 3)["acct_b"].to_list()
    assert all(v < 0 for v in late)


def test_cashflow_schedule_is_real(two_account_result) -> None:
    """Schedule follows the real flag: values are real when real=True.

    A $10k real annual contribution deflated by (1+0.03)^1 ≈ 1.03 gives
    ~$10k in real year-1 dollars.  Check it's in a reasonable range.
    """
    assert two_account_result.total.is_real is True
    df = two_account_result.cashflow_schedule
    y1 = float(df.filter(pl.col("year") == 1)["acct_a"][0])
    assert 8_000 < y1 < 12_000  # within ±20% of $10k real


# ---------------------------------------------------------------------------
# Unit tests for _compute_annual_cashflows
# ---------------------------------------------------------------------------

def test_compute_annual_cashflows_dollar_no_inflation() -> None:
    """A fixed $1k/year contribution sums to $1k per year exactly."""
    cf = wp.cashflows.PeriodicCashflow(amount=1_000.0, frequency="annual", real=False)
    path = np.full(61, 100_000.0)  # 5 years monthly, constant portfolio value
    result = _compute_annual_cashflows([cf], path, 12, 0.0, 5)
    np.testing.assert_allclose(result, [1_000.0] * 5, rtol=1e-9)


def test_compute_annual_cashflows_monthly_dollar() -> None:
    """$100/month sums to $1200/year."""
    cf = wp.cashflows.PeriodicCashflow(amount=100.0, frequency="monthly", real=False)
    path = np.full(61, 50_000.0)
    result = _compute_annual_cashflows([cf], path, 12, 0.0, 5)
    np.testing.assert_allclose(result, [1_200.0] * 5, rtol=1e-9)


def test_compute_annual_cashflows_start_year() -> None:
    """Cashflow with start_year=2.0 is zero in year 1, active from year 2 onward.

    ``amount_at`` uses a strict ``<`` check: current_year < start_year → skip.
    So at current_year == start_year the cashflow fires, meaning year 2 is active.
    """
    cf = wp.cashflows.PeriodicCashflow(
        amount=1_000.0, frequency="annual", real=False, start_year=2.0
    )
    path = np.full(49, 100_000.0)  # 4 years monthly (ppy=12)
    result = _compute_annual_cashflows([cf], path, 12, 0.0, 4)
    assert result[0] == 0.0    # year 1: current_year(12/12=1.0) < 2.0 → zero
    assert result[1] > 0.0     # year 2: current_year(24/12=2.0) == start → fires
    assert result[2] > 0.0    # year 3
    assert result[3] > 0.0    # year 4


def test_compute_annual_cashflows_pct_portfolio() -> None:
    """pct_portfolio cashflow scales with the provided median path."""
    # 10% annual withdrawal from a $100k portfolio → $10k/year
    cf = wp.cashflows.PeriodicCashflow(
        amount=-0.10, frequency="annual", mode="pct_portfolio", real=False
    )
    # Quarterly (ppy=4): 5 years → 20 periods + 1
    path = np.full(21, 100_000.0)
    result = _compute_annual_cashflows([cf], path, 4, 0.0, 5)
    np.testing.assert_allclose(result, [-10_000.0] * 5, rtol=1e-6)


def test_compute_annual_cashflows_multiple_cashflows() -> None:
    """Multiple cashflows per account are summed correctly."""
    contrib = wp.cashflows.PeriodicCashflow(amount=12_000.0, frequency="annual", real=False)
    withdraw = wp.cashflows.PeriodicCashflow(amount=-6_000.0, frequency="annual", real=False)
    path = np.full(37, 100_000.0)  # 3 years monthly
    result = _compute_annual_cashflows([contrib, withdraw], path, 12, 0.0, 3)
    np.testing.assert_allclose(result, [6_000.0] * 3, rtol=1e-9)
