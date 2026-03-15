"""Risk computation for portfolios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio

from waypoint.analysis.methods.risk import RiskMethod


@dataclass(frozen=True)
class RiskResult:
    """Result of a risk (covariance) computation.

    Attributes
    ----------
    covariance:
        Annualised covariance matrix as a ``pl.DataFrame``.  Columns are asset
        names; row order matches column order.
    volatilities:
        Per-asset annualised volatility (standard deviation), keyed by name.
    portfolio_volatility:
        Portfolio-level annualised volatility = sqrt(w^T Sigma w).
    method_name:
        Name of the method used.
    """

    covariance: pl.DataFrame
    volatilities: dict[str, float]
    portfolio_volatility: float
    method_name: str


@dataclass
class Risk:
    """Computes covariance and volatility for a portfolio using a pluggable method.

    Parameters
    ----------
    method:
        An object implementing the ``RiskMethod`` protocol.
    """

    method: RiskMethod

    def compute(
        self,
        portfolio: Portfolio,
        start: date | str | None,
        end: date | str | None,
        periods_per_year: int,
    ) -> RiskResult:
        """Compute annualised covariance and volatility for a portfolio.

        Parameters
        ----------
        portfolio:
            Portfolio whose asset returns are used.
        start, end:
            Date range for the historical window.
        periods_per_year:
            Number of periods per calendar year for annualisation.

        Returns
        -------
        RiskResult
        """
        wide = portfolio.get_returns(start, end)
        asset_cols = [c for c in wide.columns if c != "date"]
        returns_only = wide.select(asset_cols)

        sigma: np.ndarray = self.method.compute(returns_only, periods_per_year)

        weights = portfolio.weights
        weight_vec = np.array([weights[name] for name in asset_cols])

        portfolio_variance = float(weight_vec @ sigma @ weight_vec)
        portfolio_volatility = math.sqrt(max(portfolio_variance, 0.0))

        volatilities: dict[str, float] = {
            name: math.sqrt(max(float(sigma[i, i]), 0.0))
            for i, name in enumerate(asset_cols)
        }

        cov_df = pl.DataFrame(
            {name: sigma[:, i].tolist() for i, name in enumerate(asset_cols)}
        )

        return RiskResult(
            covariance=cov_df,
            volatilities=volatilities,
            portfolio_volatility=portfolio_volatility,
            method_name=type(self.method).__name__,
        )
