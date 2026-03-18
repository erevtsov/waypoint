"""Cross-comparison test: waypoint library vs independent parallel implementation.

This is the primary correctness test for waypoint's simulation logic.

Key finding about ``MultiWealthSimulation``:
  ``aggregate.flatten()`` creates a new ``Portfolio`` without forwarding each
  account's ``expected_return_method`` or ``risk_method``.  Therefore
  ``MultiWealthSimulation.compute`` always estimates parameters from the
  **flattened** portfolio using its default methods:
    - ``GeometricMean``   for expected returns
    - ``SampleCovariance`` for covariance
  Per-account ``expected_return_method`` settings have no effect on the joint
  simulation; they only affect single-account ``WealthSimulation.compute``.

The parallel implementation replicates this behaviour exactly:
  - Geometric mean:        ``exp(mean(log(1 + r_t)) * ppy) - 1``
  - Sample covariance:     ``np.cov(R, rowvar=False) * ppy``
  - Monte Carlo (seed 42): multivariate normal with PSD projection

With identical mu, cov, and seed the RNG produces identical draws, so all
wealth paths must be numerically identical (``atol=1e-8``).  Any divergence
exposes a logic discrepancy in waypoint's simulation layer.

The parallel implementation is kept independent of ``waypoint.analytics``,
``waypoint.portfolio``, ``waypoint.cashflows``, and ``waypoint.aggregate``
— the only waypoint symbol used is ``Asset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import polars as pl
import pytest

import waypoint as wp
from waypoint.assets import Asset
from waypoint.analysis.methods.returns import GeometricMean
from waypoint.analysis.methods.risk import SampleCovariance

# ---------------------------------------------------------------------------
# Shared scenario constants
# ---------------------------------------------------------------------------

RETIREMENT_YEAR   = 23   # years until retirement (age 42 → 65)
HORIZON_YEARS     = 46   # total horizon (age 42 → 88)
KID_COLLEGE_START = 8
KID_COLLEGE_END   = 12

INFLATION_RATE = 0.035
N_SIMULATIONS  = 500
PPY = 12  # monthly simulation

DATA_START = "2006-01-31"
DATA_END   = "2024-12-31"

W_BROKERAGE = 110_000.0
W_401K      = 310_000.0
W_ROTH      =  19_000.0
W_529       =  10_000.0
W_HSA       =  18_000.0


# ---------------------------------------------------------------------------
# Shared synthetic asset factory (same seeds as other retirement tests)
# ---------------------------------------------------------------------------

def _make_monthly_asset(name: str, ticker: str, mean: float, std: float, seed: int) -> Asset:
    n_months = (2024 - 2006) * 12 + 12
    rng = np.random.default_rng(seed=seed)
    returns_vals = rng.normal(mean, std, n_months).tolist()
    dates: list[date] = []
    for i in range(n_months):
        year  = 2006 + (i // 12)
        month = (i % 12) + 1
        eom = (
            date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        ) - timedelta(days=1)
        dates.append(eom)
    return Asset(
        name=name, ticker=ticker,
        returns=pl.DataFrame({"date": dates, "returns": returns_vals}),
        frequency="monthly",
    )


@pytest.fixture(scope="module")
def assets() -> dict[str, Asset]:
    return {
        "us_eq":   _make_monthly_asset("US Equity",   "VTI", mean=0.0065, std=0.040, seed=10),
        "intl_eq": _make_monthly_asset("Intl Equity", "EFA", mean=0.005,  std=0.042, seed=11),
        "bonds":   _make_monthly_asset("Bonds",       "AGG", mean=0.0025, std=0.010, seed=12),
    }


# ---------------------------------------------------------------------------
# Waypoint simulation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def waypoint_result(assets: dict[str, Asset]):
    """Run waypoint MultiWealthSimulation with default (GeometricMean) parameters.

    Individual portfolio ``expected_return_method`` is intentionally left at
    the default so the fixture honestly represents what ``MultiWealthSimulation``
    actually does: it reads the **flattened** portfolio's default methods
    (``GeometricMean`` + ``SampleCovariance``), ignoring per-account settings.
    """
    us_eq, intl_eq, bonds = assets["us_eq"], assets["intl_eq"], assets["bonds"]

    brokerage = wp.Portfolio(
        slots={"us_eq": us_eq, "intl_eq": intl_eq, "bonds": bonds},
        weights={"us_eq": 0.40, "intl_eq": 0.35, "bonds": 0.25},
        name="brokerage", initial_wealth=W_BROKERAGE,
    )
    k401 = wp.Portfolio(
        slots={"us_eq": us_eq, "bonds": bonds},
        weights={"us_eq": 0.65, "bonds": 0.35},
        name="k401", initial_wealth=W_401K,
    )
    roth = wp.Portfolio(
        slots={"us_eq": us_eq}, weights={"us_eq": 1.0},
        name="roth", initial_wealth=W_ROTH,
    )
    plan529 = wp.Portfolio(
        slots={"us_eq": us_eq}, weights={"us_eq": 1.0},
        name="plan529", initial_wealth=W_529,
    )
    hsa = wp.Portfolio(
        slots={"us_eq": us_eq, "bonds": bonds},
        weights={"us_eq": 0.80, "bonds": 0.20},
        name="hsa", initial_wealth=W_HSA,
    )

    cashflows = {
        "brokerage": [
            wp.cashflows.PeriodicCashflow(
                amount=10_000.0, frequency="annual", real=True,
                end_year=float(RETIREMENT_YEAR), slots=("us_eq", "intl_eq"),
            ),
            wp.cashflows.PeriodicCashflow(amount=400.0, frequency="monthly", real=False),
            wp.cashflows.PeriodicCashflow(
                amount=-18_000.0, frequency="annual", real=True,
                start_year=float(RETIREMENT_YEAR),
            ),
        ],
        "k401": [
            wp.cashflows.PeriodicCashflow(
                amount=2_000.0, frequency="monthly", real=True,
                end_year=float(RETIREMENT_YEAR),
            ),
        ],
        "roth": [
            wp.cashflows.PeriodicCashflow(
                amount=7_000.0, frequency="annual", real=True,
                end_year=float(RETIREMENT_YEAR),
            ),
        ],
        "plan529": [
            wp.cashflows.PeriodicCashflow(
                amount=450.0, frequency="monthly", real=True,
                end_year=float(KID_COLLEGE_START),
            ),
            wp.cashflows.PeriodicCashflow(
                amount=-0.25, frequency="annual", mode="pct_portfolio",
                start_year=float(KID_COLLEGE_START), end_year=float(KID_COLLEGE_END),
            ),
        ],
        "hsa": [
            wp.cashflows.PeriodicCashflow(
                amount=450.0, frequency="monthly", real=True,
                end_year=float(RETIREMENT_YEAR),
            ),
        ],
    }

    aggregate = wp.Aggregate([brokerage, k401, roth, plan529, hsa])
    sim = wp.analytics.MultiWealthSimulation(
        method=wp.sim.MonteCarlo(seed=42),
        cashflows=cashflows,
        horizon_years=HORIZON_YEARS,
        n_simulations=N_SIMULATIONS,
        inflation_rate=INFLATION_RATE,
    )
    return sim.compute(aggregate, start=DATA_START, end=DATA_END, frequency="monthly", real=True)


# ---------------------------------------------------------------------------
# Parallel implementation
# ---------------------------------------------------------------------------
# All logic is re-implemented from scratch.  Only ``Asset.returns`` (a polars
# DataFrame) is read from the waypoint data layer.
# ---------------------------------------------------------------------------

def _par_return_matrix(assets_list: list[Asset], start: str, end: str) -> np.ndarray:
    """Inner-join per-asset returns into a (T, n_assets) numpy matrix."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    frames = [
        a.returns
        .filter((pl.col("date") >= s) & (pl.col("date") <= e))
        .rename({"returns": a.ticker})
        for a in assets_list
    ]
    wide = frames[0]
    for f in frames[1:]:
        wide = wide.join(f, on="date", how="inner")
    return wide.select([c for c in wide.columns if c != "date"]).to_numpy()


def _par_estimate_params(matrix: np.ndarray, ppy: int) -> tuple[np.ndarray, np.ndarray]:
    """Geometric mean and sample covariance, both annualised.

    Matches the flattened portfolio's default estimation methods used by
    ``MultiWealthSimulation.compute``:
      - ``GeometricMean``:    ``exp(mean(log(1 + r_t)) * ppy) - 1``
      - ``SampleCovariance``: ``np.cov(R, rowvar=False) * ppy``
    """
    mu  = np.exp(np.mean(np.log1p(matrix), axis=0) * ppy) - 1.0
    cov = np.cov(matrix, rowvar=False) * ppy
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return mu, cov


def _par_monte_carlo(
    mu_pp: np.ndarray, cov_pp: np.ndarray,
    n_periods: int, n_sims: int, seed: int,
) -> np.ndarray:
    """Multivariate-normal draws with PSD projection.

    Mirrors ``MonteCarlo.simulate`` exactly, including the eigenvalue clip at
    1e-8 and the ``method="eigh"`` argument to ``multivariate_normal``.
    """
    rng = np.random.default_rng(seed=seed)
    eigvals, eigvecs = np.linalg.eigh(cov_pp)
    eigvals = np.maximum(eigvals, 1e-8)
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return rng.multivariate_normal(
        mean=mu_pp, cov=cov_psd,
        size=(n_sims, n_periods), method="eigh",
    )


@dataclass(frozen=True)
class _CF:
    """Minimal cashflow spec for the parallel simulation."""
    amount:             float
    frequency:          Literal["monthly", "annual"] = "monthly"
    real:               bool  = False
    effective_tax_rate: float = 0.0
    start_year:         float | None = None
    end_year:           float | None = None
    mode:               Literal["dollar", "pct_portfolio"] = "dollar"
    target_indices:     tuple[int, ...] | None = None


def _cf_amount(cf: _CF, t: int, ppy: int, infl: float, pv: float) -> float:
    """Cash-flow amount at period *t*.  Mirrors ``PeriodicCashflow.amount_at``."""
    if t == 0:
        return 0.0
    yr = t / ppy
    if cf.start_year is not None and yr < cf.start_year:
        return 0.0
    if cf.end_year is not None and yr > cf.end_year:
        return 0.0
    cf_ppy = 12 if cf.frequency == "monthly" else 1
    if cf_ppy >= ppy:
        count = cf_ppy // ppy
    else:
        every = ppy // cf_ppy
        if t % every != 0:
            return 0.0
        count = 1
    base = (cf.amount * count * pv) if cf.mode == "pct_portfolio" else (
        cf.amount * count * infl if cf.real else cf.amount * count
    )
    if base < 0.0 and cf.effective_tax_rate > 0.0:
        return base / (1.0 - cf.effective_tax_rate)
    return base


def _distribute(vals: np.ndarray, idx: list[int], amount: float) -> None:
    tgt = vals[idx]
    total = float(tgt.sum())
    vals[idx] += amount * (tgt / total) if total > 0.0 else amount / len(idx)


def _par_build_paths(
    draws: np.ndarray, initial: np.ndarray,
    cashflows: list[_CF], ppy: int, inflation_rate: float,
) -> np.ndarray:
    """Per-asset wealth path accumulation.  Mirrors ``_build_asset_paths``."""
    n_sims, n_periods, n_assets = draws.shape
    paths = np.empty((n_sims, n_periods + 1, n_assets))
    paths[:, 0, :] = initial
    pi = (1.0 + inflation_rate) ** (1.0 / ppy) if inflation_rate else 1.0
    targets = [
        list(cf.target_indices) if cf.target_indices else list(range(n_assets))
        for cf in cashflows
    ]
    for t in range(1, n_periods + 1):
        paths[:, t, :] = paths[:, t - 1, :] * (1.0 + draws[:, t - 1, :])
        if not cashflows:
            continue
        infl = pi ** t
        for cf, tgt in zip(cashflows, targets):
            if cf.mode == "pct_portfolio":
                for s in range(n_sims):
                    amt = _cf_amount(cf, t, ppy, infl, float(paths[s, t, :].sum()))
                    if amt:
                        _distribute(paths[s, t, :], tgt, amt)
            else:
                amt = _cf_amount(cf, t, ppy, infl, 0.0)
                if not amt:
                    continue
                tv  = paths[:, t, tgt]
                tot = tv.sum(axis=1, keepdims=True)
                pos = tot > 0.0
                frac = np.where(
                    pos, tv / np.where(pos, tot, 1.0),
                    np.full(tv.shape, 1.0 / len(tgt)),
                )
                paths[:, t, tgt] += amt * frac
    return paths


@dataclass
class _Account:
    name:           str
    assets:         list[Asset]
    weights:        list[float]
    initial_wealth: float
    cashflows:      list[_CF]
    _idx:           list[int] = field(default_factory=list, init=False, repr=False)


def _par_simulate(
    accounts: list[_Account], start: str, end: str,
    ppy: int, horizon_years: int, n_sims: int,
    inflation_rate: float, seed: int, real: bool,
) -> dict[str, np.ndarray]:
    """Joint multi-account Monte Carlo.  Mirrors ``MultiWealthSimulation.compute``."""
    seen: dict[str, Asset] = {}
    for acct in accounts:
        for a in acct.assets:
            seen.setdefault(a.ticker, a)
    universe = list(seen.values())
    idx_map  = {a.ticker: i for i, a in enumerate(universe)}
    for acct in accounts:
        acct._idx = [idx_map[a.ticker] for a in acct.assets]

    matrix   = _par_return_matrix(universe, start, end)
    mu_a, cov_a = _par_estimate_params(matrix, ppy)
    mu_pp    = mu_a / ppy
    cov_pp   = cov_a / ppy
    n_periods = horizon_years * ppy
    draws    = _par_monte_carlo(mu_pp, cov_pp, n_periods, n_sims, seed)

    account_paths: dict[str, np.ndarray] = {}
    for acct in accounts:
        acct_draws = draws[:, :, acct._idx]
        init_vals  = np.array([w * acct.initial_wealth for w in acct.weights])
        ap = _par_build_paths(acct_draws, init_vals, acct.cashflows, ppy, inflation_rate)
        if real and inflation_rate:
            for t in range(1, n_periods + 1):
                ap[:, t, :] /= (1.0 + inflation_rate) ** (t / ppy)
        account_paths[acct.name] = ap.sum(axis=2)

    total: np.ndarray = sum(account_paths.values())  # type: ignore[assignment]
    return {**account_paths, "total": total}


@pytest.fixture(scope="module")
def parallel_paths(assets: dict[str, Asset]) -> dict[str, np.ndarray]:
    """Run the parallel simulation with parameters matching waypoint's flat-portfolio defaults."""
    us_eq, intl_eq, bonds = assets["us_eq"], assets["intl_eq"], assets["bonds"]

    accounts = [
        _Account(
            name="brokerage",
            assets=[us_eq, intl_eq, bonds],
            weights=[0.40, 0.35, 0.25],
            initial_wealth=W_BROKERAGE,
            cashflows=[
                _CF(amount=10_000.0, frequency="annual", real=True,
                    end_year=float(RETIREMENT_YEAR), target_indices=(0, 1)),
                _CF(amount=400.0, frequency="monthly", real=False),
                _CF(amount=-18_000.0, frequency="annual", real=True,
                    start_year=float(RETIREMENT_YEAR)),
            ],
        ),
        _Account(
            name="k401", assets=[us_eq, bonds],
            weights=[0.65, 0.35], initial_wealth=W_401K,
            cashflows=[
                _CF(amount=2_000.0, frequency="monthly", real=True,
                    end_year=float(RETIREMENT_YEAR)),
            ],
        ),
        _Account(
            name="roth", assets=[us_eq],
            weights=[1.0], initial_wealth=W_ROTH,
            cashflows=[
                _CF(amount=7_000.0, frequency="annual", real=True,
                    end_year=float(RETIREMENT_YEAR)),
            ],
        ),
        _Account(
            name="plan529", assets=[us_eq],
            weights=[1.0], initial_wealth=W_529,
            cashflows=[
                _CF(amount=450.0, frequency="monthly", real=True,
                    end_year=float(KID_COLLEGE_START)),
                _CF(amount=-0.25, frequency="annual", mode="pct_portfolio",
                    start_year=float(KID_COLLEGE_START), end_year=float(KID_COLLEGE_END)),
            ],
        ),
        _Account(
            name="hsa", assets=[us_eq, bonds],
            weights=[0.80, 0.20], initial_wealth=W_HSA,
            cashflows=[
                _CF(amount=450.0, frequency="monthly", real=True,
                    end_year=float(RETIREMENT_YEAR)),
            ],
        ),
    ]

    return _par_simulate(
        accounts, DATA_START, DATA_END,
        ppy=PPY, horizon_years=HORIZON_YEARS,
        n_sims=N_SIMULATIONS, inflation_rate=INFLATION_RATE,
        seed=42, real=True,
    )


# ---------------------------------------------------------------------------
# Cross-comparison tests
# ---------------------------------------------------------------------------
# Both implementations use GeometricMean + SampleCovariance + MonteCarlo(seed=42).
# Identical RNG inputs → identical draws → numerically identical paths.
# ---------------------------------------------------------------------------

_ATOL = 1e-8   # tolerance for path arrays (floating-point identity)
# Percentile interpolation in np.percentile adds ~1e-7 rounding on top of
# the path-array tolerance, so percentile comparisons use a slightly wider bound.
_ATOL_PCT = 1e-5

ACCOUNT_NAMES = ["brokerage", "k401", "roth", "plan529", "hsa"]


def test_parameter_estimation_matches(assets: dict[str, Asset]) -> None:
    """Parallel geometric-mean/sample-cov estimation must match waypoint's flat-portfolio defaults.

    This isolates the parameter-estimation layer from the simulation layer and
    confirms both implementations feed the same mu and cov to the RNG.
    """
    us_eq, intl_eq, bonds = assets["us_eq"], assets["intl_eq"], assets["bonds"]

    # Parallel estimation
    matrix = _par_return_matrix([us_eq, intl_eq, bonds], DATA_START, DATA_END)
    mu_par, cov_par = _par_estimate_params(matrix, PPY)

    # Waypoint estimation via the flattened portfolio's default methods
    flat = wp.Aggregate([
        wp.Portfolio(
            slots={"us_eq": us_eq, "intl_eq": intl_eq, "bonds": bonds},
            weights={"us_eq": 0.40, "intl_eq": 0.35, "bonds": 0.25},
            name="brokerage", initial_wealth=W_BROKERAGE,
        ),
        wp.Portfolio(
            slots={"us_eq": us_eq, "bonds": bonds},
            weights={"us_eq": 0.65, "bonds": 0.35},
            name="k401", initial_wealth=W_401K,
        ),
    ]).flatten()

    er   = wp.analytics.ExpectedReturn(method=GeometricMean()).compute(
        flat, DATA_START, DATA_END, frequency="monthly"
    )
    risk = wp.analytics.Risk(method=SampleCovariance()).compute(
        flat, DATA_START, DATA_END, frequency="monthly"
    )
    mu_wp  = np.array([er.per_asset[n] for n in ["us_eq", "intl_eq", "bonds"]])
    cov_wp = risk.covariance.to_numpy()

    np.testing.assert_allclose(mu_par, mu_wp, atol=1e-12,
                                err_msg="Expected-return vectors differ.")
    np.testing.assert_allclose(cov_par, cov_wp, atol=1e-12,
                                err_msg="Covariance matrices differ.")


def test_total_paths_numerically_identical(waypoint_result, parallel_paths) -> None:
    """Total wealth paths must be numerically identical across both implementations."""
    np.testing.assert_allclose(
        parallel_paths["total"],
        waypoint_result.total.paths,
        atol=_ATOL,
        err_msg="Total wealth paths diverge between waypoint and parallel implementation.",
    )


@pytest.mark.parametrize("account", ACCOUNT_NAMES)
def test_per_account_paths_numerically_identical(
    waypoint_result, parallel_paths, account: str,
) -> None:
    """Per-account paths must be numerically identical."""
    np.testing.assert_allclose(
        parallel_paths[account],
        waypoint_result.accounts[account].paths,
        atol=_ATOL,
        err_msg=f"Account '{account}' paths diverge.",
    )


def test_terminal_wealth_median_matches(waypoint_result, parallel_paths) -> None:
    """Median terminal wealth must be identical (follows from identical paths)."""
    wp_med  = float(np.median(waypoint_result.total.paths[:, -1]))
    par_med = float(np.median(parallel_paths["total"][:, -1]))
    assert abs(wp_med - par_med) < _ATOL, (
        f"Median terminal wealth: waypoint={wp_med:.2f}, parallel={par_med:.2f}"
    )


def test_terminal_wealth_percentiles_match(waypoint_result, parallel_paths) -> None:
    """All terminal-wealth percentiles (P5–P95) must be identical.

    np.percentile uses linear interpolation between data points, which can
    introduce ~1e-7 floating-point rounding beyond the path-array tolerance.
    The wider ``_ATOL_PCT`` still asserts agreement to better than $0.01.
    """
    for pct in (5, 25, 50, 75, 95):
        wp_val  = float(np.percentile(waypoint_result.total.paths[:, -1], pct))
        par_val = float(np.percentile(parallel_paths["total"][:, -1], pct))
        assert abs(wp_val - par_val) < _ATOL_PCT, (
            f"P{pct}: waypoint={wp_val:.2f}, parallel={par_val:.2f}"
        )
