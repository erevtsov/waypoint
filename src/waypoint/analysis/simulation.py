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
from waypoint.cashflows import CashflowDefinition
from waypoint.enums import PERIODS_PER_YEAR, Frequency


@dataclass(frozen=True)
class SimulationResult:
    """Result of a wealth simulation.

    Attributes
    ----------
    paths:
        (n_simulations, n_periods + 1) array of portfolio values.
        Column 0 is the initial wealth; column t is the value after period t.
    percentile_df:
        ``pl.DataFrame`` with columns ``["period", "p5", "p25", "p50", "p75", "p95"]``
        plus an optional ``"date"`` column when ``start_date`` was provided.
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


@dataclass
class WealthSimulation:
    """Simulates long-horizon portfolio wealth under a given return model.

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

        Estimates portfolio return and volatility from historical data, then
        simulates ``n_simulations`` paths of
        ``horizon_years * PERIODS_PER_YEAR[frequency]`` periods.  At each
        period, the portfolio grows by the simulated return and then cash flows
        are applied.

        Parameters
        ----------
        portfolio:
            Portfolio to simulate.
        start, end:
            Historical date range used to estimate parameters.
        frequency:
            Observation frequency of the portfolio returns.  Determines the
            number of simulation steps per year.  Accepts a ``Frequency``
            member or its lowercase string equivalent.
        start_date:
            Calendar date of period 0 (today).  When provided, the
            ``percentile_df`` gains a ``"date"`` column for the x-axis.
        real:
            When ``True``, deflate paths to real (inflation-adjusted) terms
            using ``self.inflation_rate``.

        Returns
        -------
        SimulationResult
        """
        freq = Frequency(frequency)
        periods_per_year = PERIODS_PER_YEAR[freq]

        # Estimate annualised mu and sigma via portfolio's configured methods,
        # then convert to per-period quantities for the simulation engine.
        er_result = ExpectedReturn(method=portfolio.expected_return_method).compute(
            portfolio, start, end, frequency=freq
        )
        risk_result = Risk(method=portfolio.risk_method).compute(
            portfolio, start, end, frequency=freq
        )

        mu_per_period = er_result.portfolio / periods_per_year
        variance_per_period = risk_result.portfolio_volatility**2 / periods_per_year

        mu = np.array([mu_per_period])
        sigma = np.array([[variance_per_period]])

        n_periods = self.horizon_years * periods_per_year

        # Simulate period returns: shape (n_simulations, n_periods)
        raw_draws = self.method.simulate(mu, sigma, n_periods, self.n_simulations)

        # raw_draws may be (n_simulations, n_periods, 1) for multivariate case
        if raw_draws.ndim == 3 and raw_draws.shape[2] == 1:
            raw_draws = raw_draws[:, :, 0]

        cashflows = self.cashflows or []
        paths = self._build_paths(raw_draws, cashflows, periods_per_year)

        if real and self.inflation_rate:
            for t in range(1, n_periods + 1):
                paths[:, t] /= (1.0 + self.inflation_rate) ** (t / periods_per_year)

        parsed_start: date | None = None
        if start_date is not None:
            parsed_start = (
                date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
            )

        percentile_df = self._compute_percentiles(paths, parsed_start, periods_per_year)

        return SimulationResult(
            paths=paths,
            percentile_df=percentile_df,
            initial_wealth=self.initial_wealth,
            horizon_years=self.horizon_years,
            start_date=parsed_start,
            is_real=real,
        )

    def _build_paths(
        self,
        period_returns: np.ndarray,
        cashflows: list[CashflowDefinition],
        periods_per_year: int,
    ) -> np.ndarray:
        """Build wealth paths from period returns and cash flows.

        Parameters
        ----------
        period_returns:
            (n_simulations, n_periods) array of period returns.
        cashflows:
            List of cash flow definitions.
        periods_per_year:
            Number of periods per year (for cash flow timing).

        Returns
        -------
        np.ndarray
            (n_simulations, n_periods + 1) array of wealth values.
        """
        n_sims, n_periods = period_returns.shape
        paths = np.empty((n_sims, n_periods + 1))
        paths[:, 0] = self.initial_wealth

        # Annual inflation factor per period
        annual_inflation_rates = {
            getattr(cf, "inflation_rate")
            for cf in cashflows
            if hasattr(cf, "inflation_rate")
        }
        # Use average inflation rate across all cashflows that have it
        avg_inflation = (
            sum(annual_inflation_rates) / len(annual_inflation_rates)
            if annual_inflation_rates
            else 0.0
        )
        period_inflation = (
            (1.0 + avg_inflation) ** (1.0 / periods_per_year) if avg_inflation else 1.0
        )

        for t in range(1, n_periods + 1):
            # Apply period return
            paths[:, t] = paths[:, t - 1] * (1.0 + period_returns[:, t - 1])

            # Apply cash flows
            if cashflows:
                cumulative_inflation = period_inflation ** t
                for sim_idx in range(n_sims):
                    pv = float(paths[sim_idx, t])
                    cf_total = sum(
                        cf.amount_at(t, periods_per_year, pv, cumulative_inflation)
                        for cf in cashflows
                    )
                    paths[sim_idx, t] += cf_total

        return paths

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
        pct_values: dict[str, list[float]] = {
            f"p{p}": [] for p in percentile_levels
        }
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
