"""Integration test: retirement feasibility analysis using a parallel implementation.

Only the data-layer ``Asset`` dataclass is imported from waypoint — its
``.returns`` attribute provides a ``pl.DataFrame[date, returns]``.  All
simulation logic is implemented from scratch in this file:

* Expected-return estimation (arithmetic mean)
* Sample covariance estimation
* Monte Carlo return simulation (multivariate normal with PSD projection)
* Cashflow scheduling (dollar and pct_portfolio modes, real/nominal, start/end years)
* Per-asset wealth path accumulation with proportional cashflow routing
* Real (inflation-adjusted) deflation
* Multi-account aggregation (joint return draws, per-account paths)
* Summary statistics

The scenario mirrors ``test_retirement_feasibility_waypoint.py`` in structure
(same ages, same account types, same horizon) but uses different initial balances
and cashflow amounts, and is fully independent of the waypoint analytics layer.

In production the assets would come from::

    asset = wp.fetch(wp.catalog.equities.US_TOTAL_MARKET, start="...", end="...")

For test isolation the same synthetic assets are constructed directly using the
``Asset`` dataclass — the only waypoint import in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import polars as pl
import pytest

# ── Single waypoint import: the data-layer Asset dataclass only ─────────────
from waypoint.assets import Asset

# ---------------------------------------------------------------------------
# Scenario constants (same structure as waypoint test; different balances)
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
PERIODS_PER_YEAR = 12  # monthly simulation

DATA_START = "2006-01-31"
DATA_END   = "2024-12-31"

W_BROKERAGE = 110_000.0
W_401K      = 310_000.0
W_ROTH      =  19_000.0
W_529       =  10_000.0
W_HSA       =  18_000.0
TOTAL_INITIAL_WEALTH = W_BROKERAGE + W_401K + W_ROTH + W_529 + W_HSA  # 467_000


# ---------------------------------------------------------------------------
# Synthetic assets — same seeds / parameters as the waypoint test so both
# tests operate on comparable historical data.
# ---------------------------------------------------------------------------

def _make_monthly_asset(name: str, ticker: str, mean: float, std: float, seed: int) -> Asset:
    """Create a synthetic monthly return Asset covering 2006-01 through 2024-12."""
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


@pytest.fixture(scope="module")
def raw_assets() -> dict[str, Asset]:
    """Synthetic assets.

    In production these would be obtained via ``wp.fetch()`` — the data layer
    is the only waypoint component used in this parallel implementation.
    """
    return {
        "us_eq":   _make_monthly_asset("US Equity",   "VTI", mean=0.0065, std=0.040, seed=10),
        "intl_eq": _make_monthly_asset("Intl Equity", "EFA", mean=0.005,  std=0.042, seed=11),
        "bonds":   _make_monthly_asset("Bonds",       "AGG", mean=0.0025, std=0.010, seed=12),
    }


# ---------------------------------------------------------------------------
# Parallel simulation: parameter estimation
# ---------------------------------------------------------------------------

def build_return_matrix(assets: list[Asset], start: str, end: str) -> np.ndarray:
    """Inner-join per-asset returns on date and return a (T, n_assets) numpy array.

    Only ``Asset.returns`` (a ``pl.DataFrame``) is accessed — no waypoint
    analytics methods are called.
    """
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)

    frames = [
        asset.returns
        .filter((pl.col("date") >= start_dt) & (pl.col("date") <= end_dt))
        .rename({"returns": asset.ticker})
        for asset in assets
    ]

    wide = frames[0]
    for f in frames[1:]:
        wide = wide.join(f, on="date", how="inner")

    return_cols = [c for c in wide.columns if c != "date"]
    return wide.select(return_cols).to_numpy()


def estimate_params(
    return_matrix: np.ndarray,
    periods_per_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic mean and sample covariance, both annualized.

    Parameters
    ----------
    return_matrix:
        (T, n_assets) matrix of per-period decimal returns.
    periods_per_year:
        Multiplier to annualise (12 for monthly, 4 for quarterly, etc.).

    Returns
    -------
    mu_annual : (n_assets,) annualized expected return per asset.
    cov_annual : (n_assets, n_assets) annualized sample covariance matrix.
    """
    mu_annual  = return_matrix.mean(axis=0) * periods_per_year
    cov_annual = np.cov(return_matrix.T) * periods_per_year
    if cov_annual.ndim == 0:
        cov_annual = cov_annual.reshape(1, 1)
    return mu_annual, cov_annual


# ---------------------------------------------------------------------------
# Parallel simulation: Monte Carlo return draws
# ---------------------------------------------------------------------------

def monte_carlo(
    mu_per_period: np.ndarray,
    cov_per_period: np.ndarray,
    n_periods: int,
    n_simulations: int,
    seed: int = 42,
) -> np.ndarray:
    """Draw correlated per-period returns from a multivariate normal distribution.

    Projects ``cov_per_period`` to the nearest positive-semidefinite matrix
    (clips negative eigenvalues) so near-singular covariance matrices arising
    from short histories or highly correlated assets do not raise errors.

    Returns
    -------
    np.ndarray of shape (n_simulations, n_periods, n_assets).
    """
    rng = np.random.default_rng(seed=seed)

    min_eigenvalue = 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov_per_period)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    draws = rng.multivariate_normal(
        mean=mu_per_period,
        cov=cov_psd,
        size=(n_simulations, n_periods),
        method="eigh",
    )
    return draws  # (n_sims, n_periods, n_assets)


# ---------------------------------------------------------------------------
# Parallel simulation: cashflow scheduling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CashflowSpec:
    """Minimal cashflow specification for the parallel simulator.

    Parameters
    ----------
    amount:
        Base cash flow.  Positive = contribution, negative = withdrawal.
        For ``mode="pct_portfolio"``, interpreted as a fraction of portfolio value.
    frequency:
        ``"monthly"`` (fires every period when ppy=12) or ``"annual"`` (fires
        once per year, at period multiples of ppy).
    real:
        When ``True``, scales the dollar amount by cumulative inflation.
        Ignored for ``mode="pct_portfolio"``.
    effective_tax_rate:
        Gross-up rate for withdrawals.  Portfolio is drawn by
        ``amount / (1 - rate)`` so the net receipt equals ``amount``.
    start_year / end_year:
        Year bounds (floating-point, 0-indexed) outside which the cashflow is
        suppressed.
    mode:
        ``"dollar"`` (default) or ``"pct_portfolio"``.
    target_indices:
        Indices into the account's asset list that receive this cashflow.
        ``None`` = distribute proportionally across all assets.
    """

    amount:             float
    frequency:          Literal["monthly", "annual"] = "monthly"
    real:               bool  = False
    effective_tax_rate: float = 0.0
    start_year:         float | None = None
    end_year:           float | None = None
    mode:               Literal["dollar", "pct_portfolio"] = "dollar"
    target_indices:     tuple[int, ...] | None = None


def cashflow_amount_at(
    cf: CashflowSpec,
    period: int,
    ppy: int,
    cumulative_inflation: float,
    portfolio_value: float,
) -> float:
    """Return the net portfolio impact of *cf* at *period* (period 0 = inception).

    Mirrors the logic of ``waypoint.cashflows.PeriodicCashflow.amount_at``
    without importing that class.

    Parameters
    ----------
    cf:
        Cashflow specification.
    period:
        Current simulation period (1-indexed; 0 never fires).
    ppy:
        Periods per year (12 for monthly).
    cumulative_inflation:
        ``(1 + annual_rate) ** (period / ppy)`` computed by the caller.
    portfolio_value:
        Current account value before the cashflow (required for pct_portfolio mode).

    Returns
    -------
    float
        Net portfolio impact (negative = outflow including tax gross-up).
    """
    if period == 0:
        return 0.0

    current_year = period / ppy
    if cf.start_year is not None and current_year < cf.start_year:
        return 0.0
    if cf.end_year is not None and current_year > cf.end_year:
        return 0.0

    cf_ppy = 12 if cf.frequency == "monthly" else 1

    if cf_ppy >= ppy:
        count = cf_ppy // ppy
    else:
        every = ppy // cf_ppy
        if period % every != 0:
            return 0.0
        count = 1

    if cf.mode == "pct_portfolio":
        base = cf.amount * count * portfolio_value
    else:
        base = cf.amount * count
        if cf.real:
            base *= cumulative_inflation

    # Gross up withdrawals for tax
    if base < 0.0 and cf.effective_tax_rate > 0.0:
        return base / (1.0 - cf.effective_tax_rate)
    return base


# ---------------------------------------------------------------------------
# Parallel simulation: wealth path accumulation
# ---------------------------------------------------------------------------

def _distribute_to_targets(
    values: np.ndarray,
    target_indices: list[int],
    amount: float,
) -> None:
    """Add *amount* to *target_indices* slots proportionally by current value (in-place)."""
    target_vals = values[target_indices]
    total = float(target_vals.sum())
    if total > 0.0:
        values[target_indices] += amount * (target_vals / total)
    else:
        values[target_indices] += amount / len(target_indices)


def build_wealth_paths(
    draws: np.ndarray,
    initial_values: np.ndarray,
    cashflows: list[CashflowSpec],
    ppy: int,
    inflation_rate: float,
) -> np.ndarray:
    """Accumulate per-asset wealth paths from return draws and cashflows.

    Parameters
    ----------
    draws:
        (n_sims, n_periods, n_assets) per-period return array.
    initial_values:
        (n_assets,) starting dollar allocation per asset.
    cashflows:
        Ordered list of cashflow specifications for this account.
    ppy:
        Periods per year.
    inflation_rate:
        Annual rate used for cumulative inflation compounding.

    Returns
    -------
    np.ndarray
        (n_sims, n_periods + 1, n_assets) wealth array.
    """
    n_sims, n_periods, n_assets = draws.shape
    paths = np.empty((n_sims, n_periods + 1, n_assets))
    paths[:, 0, :] = initial_values

    period_inflation = (1.0 + inflation_rate) ** (1.0 / ppy) if inflation_rate else 1.0

    # Resolve target index lists once
    resolved_targets: list[list[int]] = []
    for cf in cashflows:
        if cf.target_indices is not None:
            resolved_targets.append(list(cf.target_indices))
        else:
            resolved_targets.append(list(range(n_assets)))

    for t in range(1, n_periods + 1):
        paths[:, t, :] = paths[:, t - 1, :] * (1.0 + draws[:, t - 1, :])

        if not cashflows:
            continue

        cumulative_inflation = period_inflation ** t

        for cf, target_indices in zip(cashflows, resolved_targets):
            if cf.mode == "pct_portfolio":
                for sim_idx in range(n_sims):
                    portfolio_value = float(paths[sim_idx, t, :].sum())
                    amount = cashflow_amount_at(
                        cf, t, ppy, cumulative_inflation, portfolio_value
                    )
                    if amount != 0.0:
                        _distribute_to_targets(paths[sim_idx, t, :], target_indices, amount)
            else:
                amount = cashflow_amount_at(cf, t, ppy, cumulative_inflation, 0.0)
                if amount == 0.0:
                    continue
                target_vals = paths[:, t, target_indices]
                total = target_vals.sum(axis=1, keepdims=True)
                positive = total > 0.0
                safe_total = np.where(positive, total, 1.0)
                proportional = target_vals / safe_total
                equal = np.full(target_vals.shape, 1.0 / len(target_indices))
                fractions = np.where(positive, proportional, equal)
                paths[:, t, target_indices] += amount * fractions

    return paths


# ---------------------------------------------------------------------------
# Parallel simulation: multi-account driver
# ---------------------------------------------------------------------------

@dataclass
class AccountSpec:
    """One investment account in the parallel multi-account simulation."""

    name:           str
    assets:         list[Asset]
    weights:        list[float]   # must sum to 1.0; same order as assets
    initial_wealth: float
    cashflows:      list[CashflowSpec]

    # Set by the simulator after the shared universe is built.
    _global_indices: list[int] = field(default_factory=list, init=False, repr=False)


def simulate_multi_account(
    accounts: list[AccountSpec],
    start: str,
    end: str,
    ppy: int,
    horizon_years: int,
    n_simulations: int,
    inflation_rate: float,
    seed: int = 42,
    real: bool = True,
) -> dict[str, np.ndarray]:
    """Run a joint multi-account Monte Carlo simulation.

    All unique assets across all accounts are simulated **jointly** from a
    single set of correlated return draws.  Per-account cashflows are applied
    independently so that withdrawals from one account do not affect another.

    Parameters
    ----------
    accounts:
        List of account specifications.  Asset ordering within each account
        determines cashflow ``target_indices``.
    start, end:
        Historical date window for parameter estimation (ISO-8601 strings).
    ppy:
        Periods per year (12 for monthly).
    horizon_years:
        Number of years to simulate.
    n_simulations:
        Number of independent Monte Carlo paths.
    inflation_rate:
        Annual inflation rate for real-terms deflation and real cashflows.
    seed:
        RNG seed for reproducibility.
    real:
        When ``True``, deflate all paths to real (constant-dollar) terms.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are account names plus ``"total"``; values are
        (n_sims, n_periods + 1) total wealth paths.
    """
    # 1. Build shared asset universe (deduplicate by ticker, preserve order)
    ticker_to_asset: dict[str, Asset] = {}
    for account in accounts:
        for asset in account.assets:
            if asset.ticker not in ticker_to_asset:
                ticker_to_asset[asset.ticker] = asset

    universe_assets  = list(ticker_to_asset.values())
    universe_tickers = [a.ticker for a in universe_assets]
    ticker_to_idx    = {t: i for i, t in enumerate(universe_tickers)}

    for account in accounts:
        account._global_indices = [ticker_to_idx[a.ticker] for a in account.assets]

    # 2. Estimate joint parameters from the shared return matrix
    return_matrix    = build_return_matrix(universe_assets, start, end)
    mu_annual, cov_a = estimate_params(return_matrix, ppy)
    mu_per_period    = mu_annual / ppy
    cov_per_period   = cov_a / ppy

    # 3. Single joint draw for all accounts
    n_periods = horizon_years * ppy
    draws = monte_carlo(mu_per_period, cov_per_period, n_periods, n_simulations, seed=seed)
    # draws: (n_sims, n_periods, n_universe_assets)

    # 4. Per-account accumulation using sliced draws
    account_total_paths: dict[str, np.ndarray] = {}

    for account in accounts:
        acct_draws = draws[:, :, account._global_indices]

        initial_values = np.array(
            [w * account.initial_wealth for w in account.weights]
        )
        acct_asset_paths = build_wealth_paths(
            acct_draws, initial_values, account.cashflows, ppy, inflation_rate
        )

        if real and inflation_rate:
            for t in range(1, n_periods + 1):
                acct_asset_paths[:, t, :] /= (1.0 + inflation_rate) ** (t / ppy)

        account_total_paths[account.name] = acct_asset_paths.sum(axis=2)

    # 5. Total = sum of per-account paths
    total_paths: np.ndarray = sum(account_total_paths.values())  # type: ignore[assignment]
    return {**account_total_paths, "total": total_paths}


def summarize(paths: np.ndarray) -> dict[str, float]:
    """Terminal wealth distribution statistics."""
    terminal = paths[:, -1]
    return {
        "median_terminal": float(np.median(terminal)),
        "p5_terminal":     float(np.percentile(terminal, 5)),
        "p95_terminal":    float(np.percentile(terminal, 95)),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def account_specs(raw_assets: dict[str, Asset]) -> list[AccountSpec]:
    """Build account specifications using only the data-layer Asset objects."""
    us_eq   = raw_assets["us_eq"]
    intl_eq = raw_assets["intl_eq"]
    bonds   = raw_assets["bonds"]

    brokerage = AccountSpec(
        name="brokerage",
        assets=[us_eq, intl_eq, bonds],
        weights=[0.40, 0.35, 0.25],
        initial_wealth=W_BROKERAGE,
        cashflows=[
            # Annual contributions into equity slots [us_eq=0, intl_eq=1]
            CashflowSpec(
                amount=10_000.0,
                frequency="annual",
                real=True,
                end_year=float(RETIREMENT_YEAR),
                target_indices=(0, 1),
            ),
            # Nominal side income — all slots
            CashflowSpec(amount=400.0, frequency="monthly", real=False),
            # Modest supplementary retirement draw: $18k/year real
            CashflowSpec(
                amount=-18_000.0,
                frequency="annual",
                real=True,
                start_year=float(RETIREMENT_YEAR),
            ),
        ],
    )

    k401 = AccountSpec(
        name="k401",
        assets=[us_eq, bonds],
        weights=[0.65, 0.35],
        initial_wealth=W_401K,
        cashflows=[
            CashflowSpec(
                amount=2_000.0,
                frequency="monthly",
                real=True,
                end_year=float(RETIREMENT_YEAR),
            ),
        ],
    )

    roth = AccountSpec(
        name="roth",
        assets=[us_eq],
        weights=[1.0],
        initial_wealth=W_ROTH,
        cashflows=[
            CashflowSpec(
                amount=7_000.0,
                frequency="annual",
                real=True,
                end_year=float(RETIREMENT_YEAR),
            ),
        ],
    )

    plan529 = AccountSpec(
        name="plan529",
        assets=[us_eq],
        weights=[1.0],
        initial_wealth=W_529,
        cashflows=[
            CashflowSpec(
                amount=450.0,
                frequency="monthly",
                real=True,
                end_year=float(KID_COLLEGE_START),
            ),
            CashflowSpec(
                amount=-0.25,
                frequency="annual",
                mode="pct_portfolio",
                start_year=float(KID_COLLEGE_START),
                end_year=float(KID_COLLEGE_END),
            ),
        ],
    )

    hsa = AccountSpec(
        name="hsa",
        assets=[us_eq, bonds],
        weights=[0.80, 0.20],
        initial_wealth=W_HSA,
        cashflows=[
            CashflowSpec(
                amount=450.0,
                frequency="monthly",
                real=True,
                end_year=float(RETIREMENT_YEAR),
            ),
        ],
    )

    return [brokerage, k401, roth, plan529, hsa]


@pytest.fixture(scope="module")
def sim_paths(
    raw_assets: dict[str, Asset], account_specs: list[AccountSpec]
) -> dict[str, np.ndarray]:
    """Run the parallel simulation and cache results for all tests."""
    return simulate_multi_account(
        accounts=account_specs,
        start=DATA_START,
        end=DATA_END,
        ppy=PERIODS_PER_YEAR,
        horizon_years=HORIZON_YEARS,
        n_simulations=N_SIMULATIONS,
        inflation_rate=INFLATION_RATE,
        seed=42,
        real=True,
    )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

def test_total_paths_shape(sim_paths: dict[str, np.ndarray]) -> None:
    n_periods = HORIZON_YEARS * PERIODS_PER_YEAR + 1
    assert sim_paths["total"].shape == (N_SIMULATIONS, n_periods)


def test_all_accounts_present(sim_paths: dict[str, np.ndarray]) -> None:
    expected_keys = {"brokerage", "k401", "roth", "plan529", "hsa", "total"}
    assert set(sim_paths.keys()) == expected_keys


def test_account_paths_sum_to_total(sim_paths: dict[str, np.ndarray]) -> None:
    account_names = ["brokerage", "k401", "roth", "plan529", "hsa"]
    reconstructed = sum(sim_paths[name] for name in account_names)
    np.testing.assert_allclose(reconstructed, sim_paths["total"], rtol=1e-10)


def test_initial_wealth_correct(sim_paths: dict[str, np.ndarray]) -> None:
    np.testing.assert_allclose(sim_paths["total"][:, 0], TOTAL_INITIAL_WEALTH, rtol=1e-10)


def test_per_account_initial_wealth(sim_paths: dict[str, np.ndarray]) -> None:
    expected = {
        "brokerage": W_BROKERAGE,
        "k401":      W_401K,
        "roth":      W_ROTH,
        "plan529":   W_529,
        "hsa":       W_HSA,
    }
    for name, w in expected.items():
        np.testing.assert_allclose(sim_paths[name][:, 0], w, rtol=1e-10, err_msg=name)


# ---------------------------------------------------------------------------
# Economic sanity tests
# ---------------------------------------------------------------------------

def test_median_terminal_wealth_positive(sim_paths: dict[str, np.ndarray]) -> None:
    stats = summarize(sim_paths["total"])
    assert stats["median_terminal"] > 0
    assert stats["p5_terminal"] > 0


def test_percentile_ordering(sim_paths: dict[str, np.ndarray]) -> None:
    stats = summarize(sim_paths["total"])
    assert stats["p5_terminal"] < stats["median_terminal"] < stats["p95_terminal"]


def test_median_wealth_grows_to_retirement(sim_paths: dict[str, np.ndarray]) -> None:
    retirement_period = RETIREMENT_YEAR * PERIODS_PER_YEAR
    median_at_retirement = float(np.median(sim_paths["total"][:, retirement_period]))
    assert median_at_retirement > TOTAL_INITIAL_WEALTH


def test_401k_largest_at_retirement(sim_paths: dict[str, np.ndarray]) -> None:
    retirement_period = RETIREMENT_YEAR * PERIODS_PER_YEAR
    median_401k       = float(np.median(sim_paths["k401"][:, retirement_period]))
    median_brokerage  = float(np.median(sim_paths["brokerage"][:, retirement_period]))
    assert median_401k > median_brokerage


def test_529_drawn_down_after_college(sim_paths: dict[str, np.ndarray]) -> None:
    college_start_period = KID_COLLEGE_START * PERIODS_PER_YEAR
    college_end_period   = KID_COLLEGE_END * PERIODS_PER_YEAR
    median_at_start = float(np.median(sim_paths["plan529"][:, college_start_period]))
    median_at_end   = float(np.median(sim_paths["plan529"][:, college_end_period]))
    assert median_at_end < median_at_start


def test_brokerage_positive_at_retirement(sim_paths: dict[str, np.ndarray]) -> None:
    """Brokerage should accumulate meaningfully during the contribution phase."""
    retirement_period = RETIREMENT_YEAR * PERIODS_PER_YEAR
    median_at_retirement = float(np.median(sim_paths["brokerage"][:, retirement_period]))
    assert median_at_retirement > W_BROKERAGE


# ---------------------------------------------------------------------------
# Unit tests for parallel helper functions
# ---------------------------------------------------------------------------

def test_build_return_matrix_shape(raw_assets: dict[str, Asset]) -> None:
    matrix = build_return_matrix(list(raw_assets.values()), DATA_START, DATA_END)
    assert matrix.shape[0] > 0
    assert matrix.shape[1] == len(raw_assets)


def test_estimate_params_shapes(raw_assets: dict[str, Asset]) -> None:
    matrix = build_return_matrix(list(raw_assets.values()), DATA_START, DATA_END)
    mu, cov = estimate_params(matrix, PERIODS_PER_YEAR)
    n = len(raw_assets)
    assert mu.shape == (n,)
    assert cov.shape == (n, n)


def test_monte_carlo_shape() -> None:
    mu  = np.array([0.005, 0.004, 0.002])
    cov = np.diag([0.002, 0.0018, 0.0003])
    draws = monte_carlo(mu, cov, n_periods=120, n_simulations=100, seed=1)
    assert draws.shape == (100, 120, 3)


def test_cashflow_amount_at_monthly_fires_every_period() -> None:
    cf = CashflowSpec(amount=1_000.0, frequency="monthly")
    amounts = [cashflow_amount_at(cf, t, 12, 1.0, 0.0) for t in range(13)]
    assert amounts[0] == 0.0
    assert all(a == 1_000.0 for a in amounts[1:])


def test_cashflow_amount_at_annual_fires_yearly() -> None:
    cf = CashflowSpec(amount=12_000.0, frequency="annual")
    # Monthly sim (ppy=12); should fire at periods 12, 24, 36
    fired = [t for t in range(1, 37) if cashflow_amount_at(cf, t, 12, 1.0, 0.0) != 0.0]
    assert fired == [12, 24, 36]


def test_cashflow_real_scales_with_inflation() -> None:
    cf = CashflowSpec(amount=10_000.0, frequency="annual", real=True)
    infl_y1  = (1.0 + INFLATION_RATE) ** 1.0
    infl_y10 = (1.0 + INFLATION_RATE) ** 10.0
    amt_y1  = cashflow_amount_at(cf, 12,  12, infl_y1,  0.0)
    amt_y10 = cashflow_amount_at(cf, 120, 12, infl_y10, 0.0)
    assert amt_y10 > amt_y1
    assert abs(amt_y1 - 10_000.0 * infl_y1) < 1e-9


def test_cashflow_start_end_year_filtering() -> None:
    cf = CashflowSpec(amount=1_000.0, frequency="annual", start_year=2.0, end_year=5.0)
    fired_years = [
        t / 12
        for t in range(1, 73)
        if cashflow_amount_at(cf, t, 12, 1.0, 0.0) != 0.0
    ]
    assert all(2.0 <= y <= 5.0 for y in fired_years)


def test_cashflow_withdrawal_grossed_up_for_tax() -> None:
    cf = CashflowSpec(amount=-10_000.0, frequency="annual", effective_tax_rate=0.20)
    # Net withdrawal $10k; gross-up: -10_000 / 0.80 = -12_500
    amount = cashflow_amount_at(cf, 12, 12, 1.0, 0.0)
    assert abs(amount - (-12_500.0)) < 1e-9


def test_cashflow_pct_portfolio_mode() -> None:
    cf = CashflowSpec(amount=-0.25, frequency="annual", mode="pct_portfolio")
    portfolio_value = 100_000.0
    amount = cashflow_amount_at(cf, 12, 12, 1.0, portfolio_value)
    assert abs(amount - (-25_000.0)) < 1e-9
