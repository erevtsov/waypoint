"""Integration test: retirement feasibility analysis using the waypoint library.

Scenario
--------
* Current age 42, retire at 65 (23-year accumulation), horizon to age 88 (46 years).
* One child, age 10; college starts in year 8 (4 years, years 8–12).
* Five accounts: brokerage, 401k, Roth IRA, 529, HSA.
* Monthly simulation frequency, 500 paths.
* All numbers are deliberately different from the personal/retirement.ipynb notebook.

The test exercises the full waypoint analytics stack:
``Asset → Portfolio → Aggregate → MultiWealthSimulation → MultiWealthSimulationResult``.
Assertions check economic sanity rather than exact numerical values so that the
test remains valid even if the simulation method or parameter-estimation
implementation changes.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import waypoint as wp
from waypoint.assets import Asset

# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------

CURRENT_AGE = 42
RETIREMENT_AGE = 65
HORIZON_AGE = 88

RETIREMENT_YEAR = RETIREMENT_AGE - CURRENT_AGE  # 23
HORIZON_YEARS = HORIZON_AGE - CURRENT_AGE        # 46

KID_AGE = 10
KID_COLLEGE_START = 18 - KID_AGE   # 8
KID_COLLEGE_END = KID_COLLEGE_START + 4  # 12

INFLATION_RATE = 0.035
N_SIMULATIONS = 500

DATA_START = "2006-01-31"
DATA_END = "2024-12-31"
FREQUENCY = "monthly"

# Initial account balances (different from notebook)
W_BROKERAGE = 110_000.0
W_401K = 310_000.0
W_ROTH = 19_000.0
W_529 = 10_000.0
W_HSA = 18_000.0
TOTAL_INITIAL_WEALTH = W_BROKERAGE + W_401K + W_ROTH + W_529 + W_HSA  # 467_000


# ---------------------------------------------------------------------------
# Synthetic assets (no network access required)
# ---------------------------------------------------------------------------

def _make_monthly_asset(name: str, ticker: str, mean: float, std: float, seed: int) -> Asset:
    """Create a synthetic monthly return Asset covering 2006-01 through 2024-12.

    Generates 228 month-end dates with normally distributed returns.  The
    parameters are chosen to produce realistic long-run equity and bond
    return profiles without requiring live market data.
    """
    n_months = (2024 - 2006) * 12 + 12  # 228
    rng = np.random.default_rng(seed=seed)
    returns_vals = rng.normal(mean, std, n_months).tolist()

    dates: list[date] = []
    for i in range(n_months):
        year = 2006 + (i // 12)
        month = (i % 12) + 1
        if month == 12:
            eom = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            eom = date(year, month + 1, 1) - timedelta(days=1)
        dates.append(eom)

    return Asset(
        name=name,
        ticker=ticker,
        returns=pl.DataFrame({"date": dates, "returns": returns_vals}),
        frequency="monthly",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def assets() -> dict[str, Asset]:
    """Three synthetic assets covering the full data window."""
    return {
        "us_eq":   _make_monthly_asset("US Equity",   "VTI", mean=0.0065, std=0.040, seed=10),
        "intl_eq": _make_monthly_asset("Intl Equity", "EFA", mean=0.005,  std=0.042, seed=11),
        "bonds":   _make_monthly_asset("Bonds",       "AGG", mean=0.0025, std=0.010, seed=12),
    }


@pytest.fixture(scope="module")
def portfolios(assets: dict[str, Asset]) -> dict[str, wp.Portfolio]:
    """Five account portfolios with realistic allocations and initial balances."""
    us_eq   = assets["us_eq"]
    intl_eq = assets["intl_eq"]
    bonds   = assets["bonds"]

    brokerage = wp.Portfolio(
        slots={"us_eq": us_eq, "intl_eq": intl_eq, "bonds": bonds},
        weights={"us_eq": 0.40, "intl_eq": 0.35, "bonds": 0.25},
        name="brokerage",
        initial_wealth=W_BROKERAGE,
    )
    k401 = wp.Portfolio(
        slots={"us_eq": us_eq, "bonds": bonds},
        weights={"us_eq": 0.65, "bonds": 0.35},
        name="k401",
        initial_wealth=W_401K,
    )
    roth = wp.Portfolio(
        slots={"us_eq": us_eq},
        weights={"us_eq": 1.0},
        name="roth",
        initial_wealth=W_ROTH,
    )
    plan529 = wp.Portfolio(
        slots={"us_eq": us_eq},
        weights={"us_eq": 1.0},
        name="plan529",
        initial_wealth=W_529,
    )
    hsa = wp.Portfolio(
        slots={"us_eq": us_eq, "bonds": bonds},
        weights={"us_eq": 0.80, "bonds": 0.20},
        name="hsa",
        initial_wealth=W_HSA,
    )
    return {"brokerage": brokerage, "k401": k401, "roth": roth, "plan529": plan529, "hsa": hsa}


@pytest.fixture(scope="module")
def cashflow_map() -> dict[str, list]:
    """Per-account cashflow definitions."""
    brokerage_cfs = [
        # Annual contributions into equity slots only, pre-retirement
        wp.cashflows.PeriodicCashflow(
            amount=10_000.0,
            frequency="annual",
            real=True,
            end_year=float(RETIREMENT_YEAR),
            slots=("us_eq", "intl_eq"),
        ),
        # Nominal side income (dividends / rental income), flat in dollar terms
        wp.cashflows.PeriodicCashflow(
            amount=400.0,
            frequency="monthly",
            real=False,
        ),
        # Modest supplementary retirement draw: $18k/year real
        wp.cashflows.PeriodicCashflow(
            amount=-18_000.0,
            frequency="annual",
            real=True,
            start_year=float(RETIREMENT_YEAR),
        ),
    ]
    k401_cfs = [
        wp.cashflows.PeriodicCashflow(
            amount=2_000.0,
            frequency="monthly",
            real=True,
            end_year=float(RETIREMENT_YEAR),
        ),
    ]
    roth_cfs = [
        wp.cashflows.PeriodicCashflow(
            amount=7_000.0,
            frequency="annual",
            real=True,
            end_year=float(RETIREMENT_YEAR),
        ),
    ]
    plan529_cfs = [
        wp.cashflows.PeriodicCashflow(
            amount=450.0,
            frequency="monthly",
            real=True,
            end_year=float(KID_COLLEGE_START),
        ),
        wp.cashflows.PeriodicCashflow(
            amount=-0.25,
            frequency="annual",
            mode="pct_portfolio",
            start_year=float(KID_COLLEGE_START),
            end_year=float(KID_COLLEGE_END),
        ),
    ]
    hsa_cfs = [
        wp.cashflows.PeriodicCashflow(
            amount=450.0,
            frequency="monthly",
            real=True,
            end_year=float(RETIREMENT_YEAR),
        ),
    ]
    return {
        "brokerage": brokerage_cfs,
        "k401":      k401_cfs,
        "roth":      roth_cfs,
        "plan529":   plan529_cfs,
        "hsa":       hsa_cfs,
    }


@pytest.fixture(scope="module")
def sim_result(portfolios: dict[str, wp.Portfolio], cashflow_map: dict[str, list]):
    """Run MultiWealthSimulation and cache the result for all tests."""
    aggregate = wp.Aggregate(list(portfolios.values()))
    sim = wp.analytics.MultiWealthSimulation(
        method=wp.sim.MonteCarlo(seed=42),
        cashflows=cashflow_map,
        horizon_years=HORIZON_YEARS,
        n_simulations=N_SIMULATIONS,
        inflation_rate=INFLATION_RATE,
    )
    return sim.compute(
        aggregate,
        start=DATA_START,
        end=DATA_END,
        frequency=FREQUENCY,
        real=True,
    )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

def test_total_paths_shape(sim_result) -> None:
    """Total paths array has the correct (n_sims, n_periods+1) shape."""
    expected_periods = HORIZON_YEARS * 12 + 1
    assert sim_result.total.paths.shape == (N_SIMULATIONS, expected_periods)


def test_all_accounts_present(sim_result) -> None:
    assert set(sim_result.accounts.keys()) == {"brokerage", "k401", "roth", "plan529", "hsa"}


def test_account_paths_sum_to_total(sim_result) -> None:
    """Per-account paths should exactly reconstitute the aggregate total."""
    reconstructed = sum(r.paths for r in sim_result.accounts.values())
    np.testing.assert_allclose(reconstructed, sim_result.total.paths, rtol=1e-10)


def test_initial_period_equals_total_wealth(sim_result) -> None:
    """Period-0 wealth across all simulations must equal the sum of initial balances."""
    initial = sim_result.total.paths[:, 0]
    np.testing.assert_allclose(initial, TOTAL_INITIAL_WEALTH, rtol=1e-6)


def test_percentile_df_columns(sim_result) -> None:
    df = sim_result.total.percentile_df
    for col in ("period", "p5", "p25", "p50", "p75", "p95"):
        assert col in df.columns


def test_is_real_flag(sim_result) -> None:
    assert sim_result.total.is_real is True


# ---------------------------------------------------------------------------
# Economic sanity tests
# ---------------------------------------------------------------------------

def test_summary_percentile_ordering(sim_result) -> None:
    stats = sim_result.total.summary()
    assert stats["p5_terminal"] < stats["median_terminal"] < stats["p95_terminal"]


def test_median_terminal_wealth_positive(sim_result) -> None:
    stats = sim_result.total.summary()
    assert stats["median_terminal"] > 0
    assert stats["p5_terminal"] > 0


def test_median_wealth_exceeds_initial_at_retirement(sim_result) -> None:
    """Substantial contributions + returns over 23 years should lift median wealth."""
    retirement_period = RETIREMENT_YEAR * 12
    median_at_retirement = float(np.median(sim_result.total.paths[:, retirement_period]))
    assert median_at_retirement > TOTAL_INITIAL_WEALTH


def test_401k_dominates_brokerage_at_retirement(sim_result) -> None:
    """401k starts larger and receives more contributions; should dwarf brokerage by retirement."""
    retirement_period = RETIREMENT_YEAR * 12
    median_401k = float(np.median(sim_result.accounts["k401"].paths[:, retirement_period]))
    median_brokerage = float(
        np.median(sim_result.accounts["brokerage"].paths[:, retirement_period])
    )
    assert median_401k > median_brokerage


def test_529_substantially_drawn_down_by_college_end(sim_result) -> None:
    """Four years of 25%-of-portfolio annual withdrawals should significantly reduce the 529."""
    college_start_period = KID_COLLEGE_START * 12
    college_end_period = KID_COLLEGE_END * 12
    median_at_start = float(
        np.median(sim_result.accounts["plan529"].paths[:, college_start_period])
    )
    median_at_end = float(np.median(sim_result.accounts["plan529"].paths[:, college_end_period]))
    # Rough bound: should be less than the peak at college start
    assert median_at_end < median_at_start


def test_brokerage_positive_at_retirement(sim_result) -> None:
    """Brokerage should accumulate meaningfully during the contribution phase."""
    retirement_period = RETIREMENT_YEAR * 12
    median_brokerage_at_retirement = float(
        np.median(sim_result.accounts["brokerage"].paths[:, retirement_period])
    )
    assert median_brokerage_at_retirement > W_BROKERAGE


def test_per_account_initial_wealth_correct(sim_result) -> None:
    """Period-0 of each account should match its configured initial wealth."""
    expected = {
        "brokerage": W_BROKERAGE,
        "k401":      W_401K,
        "roth":      W_ROTH,
        "plan529":   W_529,
        "hsa":       W_HSA,
    }
    for name, w in expected.items():
        initial = sim_result.accounts[name].paths[:, 0]
        np.testing.assert_allclose(initial, w, rtol=1e-6, err_msg=f"Account {name!r}")
