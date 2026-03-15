"""Tests for WealthSimulation and SimulationResult."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.analysis.methods.simulation import Bootstrap, MonteCarlo
from waypoint.analysis.simulation import SimulationResult, WealthSimulation
from waypoint.assets import Asset
from waypoint.cashflows import PeriodicCashflow
from waypoint.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PERIODS = 200
PERIODS_PER_YEAR = 12
HORIZON_YEARS = 10
N_SIMULATIONS = 200  # small for test speed


def _make_asset(name: str, ticker: str, mean: float, std: float, seed: int) -> Asset:
    """Create a daily Asset with N_PERIODS * 12 rows."""
    n = N_PERIODS * 12
    rng = np.random.default_rng(seed=seed)
    dates = [date(2010, 1, 4) + timedelta(days=i) for i in range(n)]
    values = rng.normal(mean, std, n).tolist()
    return Asset(
        name=name, ticker=ticker,
        returns=pl.DataFrame({"date": dates, "returns": values}),
        frequency="daily",
    )


def _make_portfolio(positive_mu: bool = True) -> Portfolio:
    mean = 0.0003 if positive_mu else -0.0001
    eq = _make_asset("Equities", "EQ", mean=mean, std=0.01, seed=10)
    fi = _make_asset("Bonds", "FI", mean=0.0001, std=0.003, seed=11)
    return Portfolio(
        {"Equities": eq, "Bonds": fi},
        weights={"Equities": 0.6, "Bonds": 0.4},
    )


def _make_simulation(
    n_simulations: int = N_SIMULATIONS,
    cashflows: list | None = None,
    positive_mu: bool = True,
) -> SimulationResult:
    portfolio = _make_portfolio(positive_mu=positive_mu)
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        cashflows=cashflows,
        horizon_years=HORIZON_YEARS,
        initial_wealth=1_000_000.0,
        n_simulations=n_simulations,
    )
    return sim.compute(portfolio, start=None, end=None, frequency="monthly")


# ---------------------------------------------------------------------------
# Shape and structure
# ---------------------------------------------------------------------------

def test_paths_shape() -> None:
    """paths must have shape (n_simulations, horizon_years * periods_per_year + 1)."""
    result = _make_simulation()
    expected_cols = HORIZON_YEARS * PERIODS_PER_YEAR + 1
    assert result.paths.shape == (N_SIMULATIONS, expected_cols)


def test_paths_initial_wealth() -> None:
    """Column 0 of paths must equal initial_wealth for all simulations."""
    result = _make_simulation()
    np.testing.assert_array_equal(result.paths[:, 0], 1_000_000.0)


def test_percentile_df_columns() -> None:
    """percentile_df must have the correct columns."""
    result = _make_simulation()
    assert result.percentile_df.columns == ["period", "p5", "p25", "p50", "p75", "p95"]


def test_percentile_df_row_count() -> None:
    """One row per period including period 0."""
    result = _make_simulation()
    expected_rows = HORIZON_YEARS * PERIODS_PER_YEAR + 1
    assert len(result.percentile_df) == expected_rows


def test_percentile_ordering() -> None:
    """p5 <= p25 <= p50 <= p75 <= p95 at every period."""
    result = _make_simulation()
    for row in result.percentile_df.iter_rows(named=True):
        assert row["p5"] <= row["p25"] + 1e-9
        assert row["p25"] <= row["p50"] + 1e-9
        assert row["p50"] <= row["p75"] + 1e-9
        assert row["p75"] <= row["p95"] + 1e-9


# ---------------------------------------------------------------------------
# Economic properties
# ---------------------------------------------------------------------------

def test_median_path_grows_with_positive_mu() -> None:
    """With positive expected return and no cashflows, median terminal > initial."""
    result = _make_simulation(positive_mu=True)
    median_terminal = result.summary()["median_terminal"]
    assert median_terminal > result.initial_wealth


def test_cashflows_reduce_terminal_wealth_for_withdrawals() -> None:
    """Regular withdrawals should reduce median terminal wealth vs no cashflows."""
    withdrawal = PeriodicCashflow(amount=-2000.0, frequency="monthly", mode="dollar")
    with_withdrawals = _make_simulation(cashflows=[withdrawal])
    without = _make_simulation(cashflows=None)

    med_with = with_withdrawals.summary()["median_terminal"]
    med_without = without.summary()["median_terminal"]
    assert med_with < med_without


def test_cashflows_increase_terminal_wealth_for_contributions() -> None:
    """Regular contributions should increase median terminal wealth vs no cashflows."""
    contribution = PeriodicCashflow(amount=2000.0, frequency="monthly", mode="dollar")
    with_contributions = _make_simulation(cashflows=[contribution])
    without = _make_simulation(cashflows=None)

    med_with = with_contributions.summary()["median_terminal"]
    med_without = without.summary()["median_terminal"]
    assert med_with > med_without


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_keys() -> None:
    result = _make_simulation()
    summary = result.summary()
    assert set(summary.keys()) == {"median_terminal", "p5_terminal", "p95_terminal"}


def test_summary_ordering() -> None:
    """p5 <= median <= p95 in the summary."""
    result = _make_simulation(n_simulations=500)
    summary = result.summary()
    assert summary["p5_terminal"] <= summary["median_terminal"]
    assert summary["median_terminal"] <= summary["p95_terminal"]


# ---------------------------------------------------------------------------
# Bootstrap method
# ---------------------------------------------------------------------------

def test_bootstrap_shape() -> None:
    """Bootstrap simulation must produce paths with the correct shape."""
    portfolio = _make_portfolio()
    hist = portfolio.portfolio_returns()["returns"].to_numpy()
    sim = WealthSimulation(
        method=Bootstrap(historical_returns=hist, block_size=6, seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=100.0,
        n_simulations=50,
    )
    result = sim.compute(portfolio, start=None, end=None, frequency="monthly")
    expected_cols = HORIZON_YEARS * PERIODS_PER_YEAR + 1
    assert result.paths.shape == (50, expected_cols)


# ---------------------------------------------------------------------------
# SimulationResult frozen
# ---------------------------------------------------------------------------

def test_simulation_result_is_frozen() -> None:
    result = _make_simulation()
    with pytest.raises((AttributeError, TypeError)):
        result.horizon_years = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# start_date — date x-axis
# ---------------------------------------------------------------------------

def test_percentile_df_has_date_column_when_start_date_given() -> None:
    portfolio = _make_portfolio()
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=1_000_000.0,
        n_simulations=N_SIMULATIONS,
    )
    result = sim.compute(
        portfolio, start=None, end=None, frequency="monthly",
        start_date=date(2025, 1, 1),
    )
    assert "date" in result.percentile_df.columns
    assert result.start_date == date(2025, 1, 1)


def test_percentile_df_no_date_column_without_start_date() -> None:
    result = _make_simulation()
    assert "date" not in result.percentile_df.columns
    assert result.start_date is None


def test_start_date_first_row_equals_start_date() -> None:
    start = date(2025, 1, 1)
    portfolio = _make_portfolio()
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=1_000_000.0,
        n_simulations=N_SIMULATIONS,
    )
    result = sim.compute(
        portfolio, start=None, end=None, frequency="monthly",
        start_date=start,
    )
    assert result.percentile_df["date"][0] == start


def test_start_date_accepts_string() -> None:
    portfolio = _make_portfolio()
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=1_000_000.0,
        n_simulations=N_SIMULATIONS,
    )
    result = sim.compute(
        portfolio, start=None, end=None, frequency="monthly",
        start_date="2025-01-01",
    )
    assert result.start_date == date(2025, 1, 1)
    assert "date" in result.percentile_df.columns


# ---------------------------------------------------------------------------
# real mode
# ---------------------------------------------------------------------------

def test_real_mode_reduces_terminal_wealth() -> None:
    """Real paths must be smaller than nominal when inflation_rate > 0."""
    portfolio = _make_portfolio()
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=1_000_000.0,
        n_simulations=N_SIMULATIONS,
        inflation_rate=0.03,
    )
    nominal = sim.compute(portfolio, start=None, end=None, frequency="monthly", real=False)
    real = sim.compute(portfolio, start=None, end=None, frequency="monthly", real=True)

    assert real.summary()["median_terminal"] < nominal.summary()["median_terminal"]
    assert real.is_real is True
    assert nominal.is_real is False


def test_real_mode_initial_wealth_unchanged() -> None:
    """Deflation starts at period 1; period-0 wealth must equal initial_wealth."""
    portfolio = _make_portfolio()
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=1_000_000.0,
        n_simulations=N_SIMULATIONS,
        inflation_rate=0.03,
    )
    result = sim.compute(portfolio, start=None, end=None, frequency="monthly", real=True)
    np.testing.assert_array_equal(result.paths[:, 0], 1_000_000.0)


def test_zero_inflation_real_equals_nominal() -> None:
    """With inflation_rate=0 real and nominal paths should be identical."""
    portfolio = _make_portfolio()
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=1_000_000.0,
        n_simulations=N_SIMULATIONS,
        inflation_rate=0.0,
    )
    nominal = sim.compute(portfolio, start=None, end=None, frequency="monthly", real=False)
    real = sim.compute(portfolio, start=None, end=None, frequency="monthly", real=True)
    np.testing.assert_array_almost_equal(real.paths, nominal.paths)
