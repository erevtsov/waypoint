"""Tests for WealthSimulation and SimulationResult."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.aggregate import Aggregate
from waypoint.analysis.methods.simulation import Bootstrap, MonteCarlo
from waypoint.analysis.simulation import (
    MultiWealthSimulation,
    SimulationResult,
    WealthSimulation,
)
from waypoint.assets import Asset
from waypoint.cashflows import PeriodicCashflow
from waypoint.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PERIODS = 200
PERIODS_PER_YEAR = 252
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
    mean = 0.001 if positive_mu else -0.001
    eq = _make_asset("Equities", "EQ", mean=mean, std=0.01, seed=10)
    fi = _make_asset("Bonds", "FI", mean=0.0005, std=0.003, seed=11)
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
    return sim.compute(portfolio, start=None, end=None, frequency="daily")


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
    result = sim.compute(portfolio, start=None, end=None, frequency="daily")
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
# Per-asset allocation
# ---------------------------------------------------------------------------

def test_allocation_dollar_keys_match_portfolio() -> None:
    """allocation_dollar must have one entry per portfolio slot."""
    result = _make_simulation()
    assert set(result.allocation_dollar.keys()) == {"Equities", "Bonds"}


def test_allocation_dollar_period_0_equals_initial_values() -> None:
    """At t=0 all simulations start at initial_wealth × weight; all percentiles equal that value."""
    result = _make_simulation()
    for name, w in result.weights.items():
        expected = 1_000_000.0 * w
        # All sims identical at t=0 so every percentile equals expected
        for col in ["p5", "p25", "p50", "p75", "p95"]:
            assert abs(result.allocation_dollar[name][col][0] - expected) < 1e-6


def test_allocation_dollar_period_0_sums_to_initial_wealth() -> None:
    """Sum of per-asset values at t=0 must equal initial_wealth."""
    result = _make_simulation()
    total_at_0 = sum(result.allocation_dollar[n]["p50"][0] for n in result.allocation_dollar)
    assert abs(total_at_0 - 1_000_000.0) < 1e-6


def test_weights_match_portfolio() -> None:
    """result.weights must equal the initial portfolio weights."""
    result = _make_simulation()
    assert abs(result.weights["Equities"] - 0.6) < 1e-10
    assert abs(result.weights["Bonds"] - 0.4) < 1e-10


def test_allocation_dollar_has_same_date_column_as_percentile_df() -> None:
    """When start_date is given, allocation_dollar DFs must also have a date column."""
    portfolio = _make_portfolio()
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=1_000_000.0,
        n_simulations=N_SIMULATIONS,
    )
    result = sim.compute(
        portfolio, start=None, end=None, frequency="daily",
        start_date=date(2025, 1, 1),
    )
    for df in result.allocation_dollar.values():
        assert "date" in df.columns


# ---------------------------------------------------------------------------
# Cashflow routing via slots
# ---------------------------------------------------------------------------

def _make_zero_return_asset(name: str, ticker: str, n: int = 200) -> Asset:
    """Asset with exactly zero returns every period."""
    dates = [date(2010, 1, 1) + timedelta(days=i) for i in range(n)]
    returns = pl.DataFrame({"date": dates, "returns": [0.0] * n})
    return Asset(name=name, ticker=ticker, returns=returns, frequency="daily")


def test_cashflow_routing_does_not_affect_excluded_slot() -> None:
    """A slot excluded from cashflow slots must not receive any cashflow."""
    a = _make_zero_return_asset("A", "A")
    b = _make_zero_return_asset("B", "B")
    portfolio = Portfolio({"A": a, "B": b}, weights={"A": 0.5, "B": 0.5})

    # Cashflow targets only slot A
    cf = PeriodicCashflow(amount=1_000.0, frequency="monthly", mode="dollar", slots=("A",))
    sim = WealthSimulation(
        method=Bootstrap(historical_returns=np.zeros(200), block_size=5, seed=42),
        cashflows=[cf],
        horizon_years=1,
        initial_wealth=100_000.0,
        n_simulations=50,
    )
    result = sim.compute(portfolio, start=None, end=None, frequency="monthly")

    b_initial = 100_000.0 * 0.5
    # B has zero returns and receives no cashflows — its value must stay constant
    for col in ["p5", "p25", "p50", "p75", "p95"]:
        np.testing.assert_allclose(
            result.allocation_dollar["B"][col].to_numpy(),
            b_initial,
            atol=1e-6,
        )


def test_cashflow_routing_targets_correct_slot() -> None:
    """A slot that IS the cashflow target must grow beyond its initial value."""
    a = _make_zero_return_asset("A", "A")
    b = _make_zero_return_asset("B", "B")
    portfolio = Portfolio({"A": a, "B": b}, weights={"A": 0.5, "B": 0.5})

    cf = PeriodicCashflow(amount=1_000.0, frequency="monthly", mode="dollar", slots=("A",))
    sim = WealthSimulation(
        method=Bootstrap(historical_returns=np.zeros(200), block_size=5, seed=42),
        cashflows=[cf],
        horizon_years=1,
        initial_wealth=100_000.0,
        n_simulations=50,
    )
    result = sim.compute(portfolio, start=None, end=None, frequency="monthly")

    a_initial = 100_000.0 * 0.5
    # A must have grown by accumulated cashflows
    assert result.allocation_dollar["A"]["p50"][-1] > a_initial


def test_cashflow_routing_invalid_slot_raises() -> None:
    """Specifying a slot not in the portfolio must raise ValueError."""
    a = _make_zero_return_asset("A", "A")
    portfolio = Portfolio({"A": a}, weights={"A": 1.0})

    cf = PeriodicCashflow(amount=1_000.0, frequency="monthly", mode="dollar", slots=("Z",))
    sim = WealthSimulation(
        method=Bootstrap(historical_returns=np.zeros(200), block_size=5, seed=42),
        cashflows=[cf],
        horizon_years=1,
        initial_wealth=100_000.0,
        n_simulations=10,
    )
    with pytest.raises(ValueError, match="Z"):
        sim.compute(portfolio, start=None, end=None, frequency="monthly")


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
        portfolio, start=None, end=None, frequency="daily",
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
        portfolio, start=None, end=None, frequency="daily",
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
        portfolio, start=None, end=None, frequency="daily",
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
    nominal = sim.compute(portfolio, start=None, end=None, frequency="daily", real=False)
    real = sim.compute(portfolio, start=None, end=None, frequency="daily", real=True)

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
    result = sim.compute(portfolio, start=None, end=None, frequency="daily", real=True)
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
    nominal = sim.compute(portfolio, start=None, end=None, frequency="daily", real=False)
    real = sim.compute(portfolio, start=None, end=None, frequency="daily", real=True)
    np.testing.assert_array_almost_equal(real.paths, nominal.paths)


# ---------------------------------------------------------------------------
# MultiWealthSimulation helpers
# ---------------------------------------------------------------------------

def _make_zero_return_portfolio(
    name: str,
    initial_wealth: float,
    slot_names: list[str],
    seed: int = 0,
) -> Portfolio:
    """Portfolio with near-zero returns for deterministic testing."""
    n = 300
    assets = {}
    weights = {}
    for i, sn in enumerate(slot_names):
        dates = [date(2010, 1, 1) + timedelta(days=j) for j in range(n)]
        returns = pl.DataFrame({"date": dates, "returns": [0.0] * n})
        assets[sn] = Asset(
            name=sn, ticker=sn.upper(),
            returns=returns, frequency="daily",
        )
        weights[sn] = 1.0 / len(slot_names)
    return Portfolio(assets, weights=weights, name=name, initial_wealth=initial_wealth)


def _make_aggregate() -> Aggregate:
    eq = Asset(
        name="EQ", ticker="EQ",
        returns=pl.DataFrame({
            "date": [date(2010, 1, 1) + timedelta(days=i) for i in range(300)],
            "returns": np.random.default_rng(1).normal(0.001, 0.01, 300).tolist(),
        }),
        frequency="daily",
    )
    fi = Asset(
        name="FI", ticker="FI",
        returns=pl.DataFrame({
            "date": [date(2010, 1, 1) + timedelta(days=i) for i in range(300)],
            "returns": np.random.default_rng(2).normal(0.0003, 0.003, 300).tolist(),
        }),
        frequency="daily",
    )
    taxable = Portfolio(
        {"EQ": eq, "FI": fi}, weights={"EQ": 0.7, "FI": 0.3},
        name="taxable", initial_wealth=600_000.0,
    )
    retirement = Portfolio(
        {"EQ": eq, "FI": fi}, weights={"EQ": 0.6, "FI": 0.4},
        name="401k", initial_wealth=400_000.0,
    )
    return Aggregate([taxable, retirement])


# ---------------------------------------------------------------------------
# MultiWealthSimulation — structure
# ---------------------------------------------------------------------------

def test_multi_result_has_all_accounts() -> None:
    agg = _make_aggregate()
    sim = MultiWealthSimulation(method=MonteCarlo(seed=42), horizon_years=5, n_simulations=50)
    result = sim.compute(agg, start=None, end=None, frequency="monthly")
    assert set(result.accounts.keys()) == {"taxable", "401k"}


def test_multi_result_total_paths_shape() -> None:
    agg = _make_aggregate()
    sim = MultiWealthSimulation(method=MonteCarlo(seed=42), horizon_years=5, n_simulations=50)
    result = sim.compute(agg, start=None, end=None, frequency="monthly")
    expected_cols = 5 * 12 + 1  # 5 years * 12 months + t=0
    assert result.total.paths.shape == (50, expected_cols)


def test_multi_result_account_paths_shape() -> None:
    agg = _make_aggregate()
    sim = MultiWealthSimulation(method=MonteCarlo(seed=42), horizon_years=5, n_simulations=50)
    result = sim.compute(agg, start=None, end=None, frequency="monthly")
    expected_cols = 5 * 12 + 1
    for acct_result in result.accounts.values():
        assert acct_result.paths.shape == (50, expected_cols)


def test_multi_total_initial_wealth_is_sum() -> None:
    """Total wealth at t=0 must equal sum of account initial wealths."""
    agg = _make_aggregate()
    sim = MultiWealthSimulation(method=MonteCarlo(seed=42), horizon_years=5, n_simulations=50)
    result = sim.compute(agg, start=None, end=None, frequency="monthly")
    np.testing.assert_allclose(result.total.paths[:, 0], 1_000_000.0, rtol=1e-6)


def test_multi_account_initial_wealth_correct() -> None:
    """Each account's t=0 wealth must match its initial_wealth."""
    agg = _make_aggregate()
    sim = MultiWealthSimulation(method=MonteCarlo(seed=42), horizon_years=5, n_simulations=50)
    result = sim.compute(agg, start=None, end=None, frequency="monthly")
    np.testing.assert_allclose(result.accounts["taxable"].paths[:, 0], 600_000.0, rtol=1e-6)
    np.testing.assert_allclose(result.accounts["401k"].paths[:, 0], 400_000.0, rtol=1e-6)


def test_multi_total_equals_sum_of_accounts() -> None:
    """Total paths must equal the sum of per-account paths at every period."""
    agg = _make_aggregate()
    sim = MultiWealthSimulation(method=MonteCarlo(seed=42), horizon_years=5, n_simulations=50)
    result = sim.compute(agg, start=None, end=None, frequency="monthly")
    reconstructed = (
        result.accounts["taxable"].paths + result.accounts["401k"].paths
    )
    np.testing.assert_allclose(result.total.paths, reconstructed, rtol=1e-9)


def test_multi_total_allocation_dollar_keyed_by_account() -> None:
    """total.allocation_dollar must be keyed by account name."""
    agg = _make_aggregate()
    sim = MultiWealthSimulation(method=MonteCarlo(seed=42), horizon_years=5, n_simulations=50)
    result = sim.compute(agg, start=None, end=None, frequency="monthly")
    assert set(result.total.allocation_dollar.keys()) == {"taxable", "401k"}


# ---------------------------------------------------------------------------
# MultiWealthSimulation — cashflows routed per account
# ---------------------------------------------------------------------------

def test_multi_account_cashflow_only_affects_target_account() -> None:
    """A cashflow for 'taxable' must not change '401k' balance."""
    n = 300
    dates = [date(2010, 1, 1) + timedelta(days=i) for i in range(n)]
    zero_returns = pl.DataFrame({"date": dates, "returns": [0.0] * n})
    eq = Asset(name="EQ", ticker="EQ", returns=zero_returns, frequency="daily")

    taxable = Portfolio({"EQ": eq}, weights={"EQ": 1.0}, name="taxable", initial_wealth=500_000.0)
    retirement = Portfolio(
        {"EQ": eq}, weights={"EQ": 1.0}, name="401k", initial_wealth=300_000.0
    )
    agg = Aggregate([taxable, retirement])

    contribution = PeriodicCashflow(amount=1_000.0, frequency="monthly", mode="dollar")
    sim = MultiWealthSimulation(
        method=Bootstrap(
            historical_returns=np.zeros(300), block_size=5, seed=42
        ),
        cashflows={"taxable": [contribution]},
        horizon_years=2,
        n_simulations=20,
    )
    result = sim.compute(agg, start=None, end=None, frequency="monthly")

    # 401k has zero returns and no cashflows — value must stay flat
    np.testing.assert_allclose(
        result.accounts["401k"].paths,
        300_000.0,
        atol=1e-4,
    )
    # taxable receives contributions — must grow above initial
    assert result.accounts["taxable"].paths[:, -1].mean() > 500_000.0


# ---------------------------------------------------------------------------
# MultiWealthSimulation — is_real flag
# ---------------------------------------------------------------------------

def test_multi_real_mode_reduces_terminal_wealth() -> None:
    agg = _make_aggregate()
    sim = MultiWealthSimulation(
        method=MonteCarlo(seed=42), horizon_years=5, n_simulations=50,
        inflation_rate=0.03,
    )
    nominal = sim.compute(agg, start=None, end=None, frequency="monthly", real=False)
    real = sim.compute(agg, start=None, end=None, frequency="monthly", real=True)
    assert real.total.summary()["median_terminal"] < nominal.total.summary()["median_terminal"]
    assert real.total.is_real is True
