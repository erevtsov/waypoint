"""Wealth simulation for long-horizon portfolio analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import polars as pl

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio

from waypoint.analysis.expected_return import ExpectedReturn
from waypoint.analysis.methods.simulation import SimulationMethod
from waypoint.analysis.risk import Risk
from waypoint.cashflows import CashflowDefinition, PeriodicCashflow
from waypoint.enums import PERIODS_PER_YEAR, CashflowMode, Frequency


@dataclass(frozen=True)
class SimulationResult:
    """Result of a wealth simulation.

    Attributes
    ----------
    paths:
        (n_simulations, n_periods + 1) array of total portfolio values.
        Column 0 is the initial wealth; column t is the value after period t.
    percentile_df:
        ``pl.DataFrame`` with columns ``["period", "p5", "p25", "p50", "p75", "p95"]``
        plus an optional ``"date"`` column when ``start_date`` was provided.
    weights:
        Initial portfolio weights.  Under the buy-and-hold simulation model
        these are only the *starting* allocation — asset values drift over time
        as their returns diverge and cashflow routing may differ across slots.
    allocation_dollar:
        Per-asset dollar allocations derived from the simulation.  Each value
        is a ``pl.DataFrame`` with the same structure as ``percentile_df``
        (``period``, ``p5``–``p95``, optional ``date``) representing the
        percentile distribution of that asset's value path.
    initial_wealth:
        Starting portfolio value.
    horizon_years:
        Simulation horizon in years.
    start_date:
        Calendar date of period 0, if provided at compute time.
    is_real:
        ``True`` when the paths are expressed in real (inflation-adjusted) terms.
    """

    paths: np.ndarray
    percentile_df: pl.DataFrame
    weights: dict[str, float]
    allocation_dollar: dict[str, pl.DataFrame]
    initial_wealth: float
    horizon_years: int
    start_date: date | None = None
    is_real: bool = False

    def summary(self) -> dict[str, float]:
        """Return summary statistics of the terminal wealth distribution.

        Returns
        -------
        dict[str, float]
            Keys: ``"median_terminal"``, ``"p5_terminal"``, ``"p95_terminal"``.
        """
        terminal = self.paths[:, -1]
        return {
            "median_terminal": float(np.median(terminal)),
            "p5_terminal": float(np.percentile(terminal, 5)),
            "p95_terminal": float(np.percentile(terminal, 95)),
        }

    def plot(self) -> go.Figure:
        """Fan chart of percentile wealth paths."""
        from waypoint.analysis.viz import plot_wealth_simulation

        return plot_wealth_simulation(self)

    def plot_allocation(self) -> go.Figure:
        """Stacked area chart of median per-asset dollar values over time."""
        from waypoint.analysis.viz import plot_allocation_dollar

        return plot_allocation_dollar(self)


@dataclass
class WealthSimulation:
    """Simulates long-horizon portfolio wealth under a given return model.

    Each asset is simulated independently using its own return path (buy-and-hold,
    no rebalancing).  Cashflows can be routed to specific slots via each
    cashflow's ``slots`` field; ``slots=None`` distributes to all assets
    proportionally by their current values.

    Parameters
    ----------
    method:
        Simulation method (e.g. MonteCarlo or Bootstrap).
    cashflows:
        Optional list of periodic or lump-sum cash flows.
    horizon_years:
        Simulation horizon in years.
    initial_wealth:
        Starting portfolio value.
    n_simulations:
        Number of independent simulation paths.
    inflation_rate:
        Annual inflation rate used for real-terms deflation (default 0.0).
    """

    method: SimulationMethod
    cashflows: list[CashflowDefinition] | None = field(default=None)
    horizon_years: int = field(default=30)
    initial_wealth: float = field(default=1.0)
    n_simulations: int = field(default=1000)
    inflation_rate: float = field(default=0.0)

    def compute(
        self,
        portfolio: Portfolio,
        start: date | str | None,
        end: date | str | None,
        frequency: Frequency | str = Frequency.DAILY,
        start_date: date | str | None = None,
        real: bool = False,
    ) -> SimulationResult:
        """Run the wealth simulation.

        Simulates per-asset return paths using the full covariance structure,
        then applies cashflows routed to each cashflow's ``slots``.  Asset
        values are never rebalanced — weights drift as returns diverge.

        Parameters
        ----------
        portfolio:
            Portfolio to simulate.
        start, end:
            Historical date range used to estimate parameters.
        frequency:
            Observation frequency of the portfolio returns.
        start_date:
            Calendar date of period 0 (today).  When provided, the
            ``percentile_df`` and ``allocation_dollar`` DFs gain a ``"date"``
            column.
        real:
            When ``True``, deflate paths to real (inflation-adjusted) terms
            using ``self.inflation_rate``.

        Returns
        -------
        SimulationResult
        """
        freq = Frequency(frequency)
        periods_per_year = PERIODS_PER_YEAR[freq]

        asset_names = portfolio.names
        n_assets = len(asset_names)
        weights = portfolio.weights
        initial_values = np.array([weights[n] * self.initial_wealth for n in asset_names])

        # Estimate per-asset annualised mu vector and covariance matrix,
        # then scale down to per-period quantities for the simulation engine.
        er_result = ExpectedReturn(method=portfolio.expected_return_method).compute(
            portfolio, start, end, frequency=freq
        )
        risk_result = Risk(method=portfolio.risk_method).compute(
            portfolio, start, end, frequency=freq
        )

        mu_per_period = np.array(
            [er_result.per_asset[n] / periods_per_year for n in asset_names]
        )
        sigma_per_period = risk_result.covariance.to_numpy() / periods_per_year

        n_periods = self.horizon_years * periods_per_year

        # Draw per-asset period returns.
        raw_draws = self.method.simulate(
            mu_per_period, sigma_per_period, n_periods, self.n_simulations
        )

        # Normalise to (n_sims, n_periods, n_assets).
        # MonteCarlo returns (n_sims, n_periods, n_assets) for multivariate or
        # (n_sims, n_periods) for univariate / single-asset.
        # Bootstrap always returns (n_sims, n_periods) — broadcast same return
        # to all assets (approximates constant-correlation, no per-asset split).
        if raw_draws.ndim == 2:
            raw_draws = np.repeat(raw_draws[:, :, np.newaxis], n_assets, axis=2)

        cashflows = self.cashflows or []
        asset_paths = self._build_asset_paths(
            raw_draws, initial_values, asset_names, cashflows, periods_per_year
        )
        # asset_paths: (n_sims, n_periods + 1, n_assets)

        if real and self.inflation_rate:
            for t in range(1, n_periods + 1):
                asset_paths[:, t, :] /= (1.0 + self.inflation_rate) ** (t / periods_per_year)

        # Total portfolio path = sum across assets.
        paths: np.ndarray = asset_paths.sum(axis=2)

        parsed_start: date | None = None
        if start_date is not None:
            parsed_start = (
                date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
            )

        percentile_df = self._compute_percentiles(paths, parsed_start, periods_per_year)
        allocation_dollar = {
            name: self._compute_percentiles(asset_paths[:, :, i], parsed_start, periods_per_year)
            for i, name in enumerate(asset_names)
        }

        return SimulationResult(
            paths=paths,
            percentile_df=percentile_df,
            weights=weights,
            allocation_dollar=allocation_dollar,
            initial_wealth=self.initial_wealth,
            horizon_years=self.horizon_years,
            start_date=parsed_start,
            is_real=real,
        )

    def _build_asset_paths(
        self,
        per_asset_returns: np.ndarray,
        initial_values: np.ndarray,
        asset_names: list[str],
        cashflows: list[CashflowDefinition],
        periods_per_year: int,
    ) -> np.ndarray:
        """Build per-asset wealth paths from return draws and routed cashflows.

        Parameters
        ----------
        per_asset_returns:
            (n_sims, n_periods, n_assets) array of per-period, per-asset returns.
        initial_values:
            (n_assets,) starting dollar value for each asset.
        asset_names:
            Ordered asset names matching the last axis of ``per_asset_returns``.
        cashflows:
            Cash flow definitions; each may specify ``slots`` for routing.
        periods_per_year:
            Used for cashflow scheduling and inflation compounding.

        Returns
        -------
        np.ndarray
            (n_sims, n_periods + 1, n_assets) array of per-asset dollar values.
        """
        n_sims, n_periods, n_assets = per_asset_returns.shape
        asset_paths = np.empty((n_sims, n_periods + 1, n_assets))
        asset_paths[:, 0, :] = initial_values

        # Inflation factor per period, derived from cashflows that declare one.
        inflation_rates = {
            cf.inflation_rate
            for cf in cashflows
            if isinstance(cf, PeriodicCashflow) and cf.inflation_rate
        }
        avg_inflation = sum(inflation_rates) / len(inflation_rates) if inflation_rates else 0.0
        period_inflation = (
            (1.0 + avg_inflation) ** (1.0 / periods_per_year) if avg_inflation else 1.0
        )

        # Pre-resolve slot indices for each cashflow; validate once up-front.
        asset_name_to_idx = {n: i for i, n in enumerate(asset_names)}
        routing: list[list[int]] = []
        for cf in cashflows:
            slots = cf.slots
            if slots:
                missing = [s for s in slots if s not in asset_name_to_idx]
                if missing:
                    raise ValueError(
                        f"Cashflow slots {missing} not found in portfolio slots {asset_names}."
                    )
                routing.append([asset_name_to_idx[s] for s in slots])
            else:
                routing.append(list(range(n_assets)))

        for t in range(1, n_periods + 1):
            # Grow each asset by its own return for this period.
            asset_paths[:, t, :] = asset_paths[:, t - 1, :] * (
                1.0 + per_asset_returns[:, t - 1, :]
            )

            if not cashflows:
                continue

            cumulative_inflation = period_inflation ** t

            for cf, target_indices in zip(cashflows, routing):
                is_pct_mode = (
                    isinstance(cf, PeriodicCashflow) and cf.mode != CashflowMode.DOLLAR
                )

                if is_pct_mode:
                    # Amount depends on total portfolio value — loop per simulation.
                    for sim_idx in range(n_sims):
                        portfolio_value = float(asset_paths[sim_idx, t, :].sum())
                        amount = cf.amount_at(
                            t, periods_per_year, portfolio_value, cumulative_inflation
                        )
                        if amount == 0.0:
                            continue
                        self._distribute(asset_paths, sim_idx, t, target_indices, amount)
                else:
                    # Dollar / lump-sum: same amount for every simulation — vectorise.
                    amount = cf.amount_at(t, periods_per_year, 0.0, cumulative_inflation)
                    if amount == 0.0:
                        continue
                    target_vals = asset_paths[:, t, target_indices]  # (n_sims, n_targets)
                    total_target = target_vals.sum(axis=1, keepdims=True)  # (n_sims, 1)
                    positive = total_target > 0.0  # (n_sims, 1)
                    safe_total = np.where(positive, total_target, 1.0)
                    proportional = target_vals / safe_total
                    equal = np.full(target_vals.shape, 1.0 / len(target_indices))
                    fractions = np.where(positive, proportional, equal)
                    asset_paths[:, t, target_indices] += amount * fractions

        return asset_paths

    @staticmethod
    def _distribute(
        asset_paths: np.ndarray,
        sim_idx: int,
        t: int,
        target_indices: list[int],
        amount: float,
    ) -> None:
        """Apply ``amount`` to target slots proportionally by their current values."""
        target_vals = asset_paths[sim_idx, t, target_indices]
        total_target = float(target_vals.sum())
        if total_target > 0.0:
            fractions = target_vals / total_target
            asset_paths[sim_idx, t, target_indices] += amount * fractions
        else:
            asset_paths[sim_idx, t, target_indices] += amount / len(target_indices)

    @staticmethod
    def _compute_percentiles(
        paths: np.ndarray,
        start_date: date | None = None,
        periods_per_year: int = 252,
    ) -> pl.DataFrame:
        """Compute percentile bands across simulation paths.

        Parameters
        ----------
        paths:
            (n_simulations, n_periods + 1) wealth array.
        start_date:
            When provided, a ``"date"`` column is added to the output.
        periods_per_year:
            Used to convert period index to a calendar offset from
            ``start_date``.

        Returns
        -------
        pl.DataFrame
            Columns: ``["period", "p5", "p25", "p50", "p75", "p95"]``,
            plus ``"date"`` when ``start_date`` is not ``None``.
        """
        n_periods_plus_one = paths.shape[1]
        percentile_levels = [5, 25, 50, 75, 95]
        pct_values: dict[str, list[float]] = {f"p{p}": [] for p in percentile_levels}
        periods: list[int] = list(range(n_periods_plus_one))

        for t in range(n_periods_plus_one):
            col = paths[:, t]
            for p in percentile_levels:
                pct_values[f"p{p}"].append(float(np.percentile(col, p)))

        base: dict[str, object] = {"period": periods, **pct_values}
        if start_date is not None:
            days_per_period = 365.25 / periods_per_year
            dates = [
                start_date + timedelta(days=round(t * days_per_period))
                for t in range(n_periods_plus_one)
            ]
            base["date"] = dates
        return pl.DataFrame(base)
