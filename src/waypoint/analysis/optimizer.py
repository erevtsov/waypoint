"""Mean-variance efficient frontier optimiser."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
import plotly.graph_objects as go
import polars as pl

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio

from waypoint.analysis.expected_return import ExpectedReturn
from waypoint.analysis.risk import Risk
from waypoint.constraints import Constraint


@dataclass
class EfficientFrontierResult:
    """Result of an efficient frontier computation.

    Attributes
    ----------
    weights:
        DataFrame with columns ``["expected_return", asset1, asset2, ...]``.
        Each row is one frontier point, sorted by risk ascending.
    expected_returns:
        Series of annualised expected returns, one per frontier point.
    risks:
        Series of annualised volatilities, one per frontier point.
    asset_names:
        Ordered list of asset names.
    """

    weights: pl.DataFrame
    expected_returns: pl.Series
    risks: pl.Series
    asset_names: list[str]

    def optimal_sharpe(self, risk_free_rate: float = 0.0) -> dict[str, float]:
        """Return the weights at the maximum Sharpe ratio frontier point.

        Parameters
        ----------
        risk_free_rate:
            Annualised risk-free rate used in the Sharpe calculation.

        Returns
        -------
        dict[str, float]
            Mapping of asset name → weight at the max-Sharpe point.
        """
        returns_list = self.expected_returns.to_list()
        risks_list = self.risks.to_list()

        best_idx = 0
        best_sharpe = -math.inf
        for i, (ret, risk) in enumerate(zip(returns_list, risks_list)):
            if risk > 0.0:
                sharpe = (ret - risk_free_rate) / risk
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_idx = i

        row = self.weights.row(best_idx, named=True)
        return {name: float(row[name]) for name in self.asset_names}

    def plot(self) -> go.Figure:
        """Plot the efficient frontier as a risk vs return scatter."""
        from waypoint.analysis.viz import plot_efficient_frontier

        return plot_efficient_frontier(self)


@dataclass
class Optimizer:
    """Mean-variance efficient frontier builder.

    Parameters
    ----------
    return_model:
        ``ExpectedReturn`` instance used to estimate per-asset expected returns.
    risk_model:
        ``Risk`` instance used to estimate the covariance matrix.
    constraints:
        List of ``Constraint`` objects applied to every sub-problem.
    """

    return_model: ExpectedReturn
    risk_model: Risk
    constraints: list[Constraint]

    def efficient_frontier(
        self,
        portfolio: Portfolio,
        start: date | str | None,
        end: date | str | None,
        periods_per_year: int,
        n_points: int = 50,
    ) -> EfficientFrontierResult:
        """Compute the efficient frontier.

        Solves n_points minimum-variance problems, each with a different return
        target uniformly spaced between the minimum and maximum feasible returns.
        Infeasible targets are skipped silently.  The result is sorted by risk
        (volatility) ascending.

        Parameters
        ----------
        portfolio:
            Portfolio whose assets define the investment universe.
        start, end:
            Historical date range used to estimate mu and Sigma.
        periods_per_year:
            Annualisation factor.
        n_points:
            Number of return targets to try along the frontier.

        Returns
        -------
        EfficientFrontierResult
        """
        er_result = self.return_model.compute(portfolio, start, end, periods_per_year)
        risk_result = self.risk_model.compute(portfolio, start, end, periods_per_year)

        asset_names = list(er_result.per_asset.keys())
        n_assets = len(asset_names)

        mu = np.array([er_result.per_asset[name] for name in asset_names])
        sigma = risk_result.covariance.to_numpy()

        # Find min and max feasible returns by solving two LP-like problems
        min_return = self._solve_extreme_return(mu, sigma, n_assets, minimize=True)
        max_return = self._solve_extreme_return(mu, sigma, n_assets, minimize=False)

        if min_return is None or max_return is None:
            raise ValueError("Could not determine feasible return range — check constraints.")

        return_targets = np.linspace(min_return, max_return, n_points)

        rows_weights: list[list[float]] = []
        rows_returns: list[float] = []
        rows_risks: list[float] = []

        for target in return_targets:
            w_var = cp.Variable(n_assets)
            point_constraints: list[Any] = [
                c for con in self.constraints for c in con.to_cvxpy(w_var, asset_names)
            ]
            point_constraints.append(mu @ w_var >= target)

            objective = cp.Minimize(cp.quad_form(w_var, sigma))  # type: ignore[attr-defined]
            problem = cp.Problem(objective, point_constraints)

            try:
                problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
            except cp.SolverError:
                continue

            if problem.status not in ("optimal", "optimal_inaccurate"):
                continue
            if w_var.value is None:
                continue

            w_opt: np.ndarray = w_var.value
            port_variance = float(w_opt @ sigma @ w_opt)
            port_vol = math.sqrt(max(port_variance, 0.0))
            port_return = float(mu @ w_opt)

            rows_weights.append(w_opt.tolist())
            rows_returns.append(port_return)
            rows_risks.append(port_vol)

        if not rows_weights:
            raise ValueError("No feasible frontier points found.")

        # Sort all rows by risk ascending
        sort_order = sorted(range(len(rows_risks)), key=lambda i: rows_risks[i])
        sorted_weights_rows = [rows_weights[i] for i in sort_order]
        sorted_returns = [rows_returns[i] for i in sort_order]
        sorted_risks = [rows_risks[i] for i in sort_order]

        weight_data: dict[str, list[float]] = {name: [] for name in asset_names}
        for row in sorted_weights_rows:
            for i, name in enumerate(asset_names):
                weight_data[name].append(row[i])

        weights_df = pl.DataFrame({"expected_return": sorted_returns, **weight_data})

        return EfficientFrontierResult(
            weights=weights_df,
            expected_returns=pl.Series("expected_return", sorted_returns),
            risks=pl.Series("risk", sorted_risks),
            asset_names=asset_names,
        )

    def _solve_extreme_return(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        n_assets: int,
        minimize: bool,
    ) -> float | None:
        """Solve for the min or max feasible expected return under the constraints.

        Parameters
        ----------
        mu:
            Expected return vector.
        sigma:
            Covariance matrix (used to ensure the problem is well-formed, but
            the objective here is purely on the expected return).
        n_assets:
            Number of assets.
        minimize:
            If True, find minimum feasible return; if False, find maximum.

        Returns
        -------
        float | None
            Optimal objective value, or None if the problem is infeasible.
        """
        w_var = cp.Variable(n_assets)
        constraints: list[Any] = [
            c for con in self.constraints for c in con.to_cvxpy(w_var, self._dummy_names(n_assets))
        ]

        if minimize:
            objective = cp.Minimize(mu @ w_var)
        else:
            objective = cp.Maximize(mu @ w_var)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
        except cp.SolverError:
            return None

        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None
        if w_var.value is None:
            return None

        return float(mu @ w_var.value)

    @staticmethod
    def _dummy_names(n: int) -> list[str]:
        """Generate placeholder asset names for constraint building."""
        return [f"asset_{i}" for i in range(n)]
