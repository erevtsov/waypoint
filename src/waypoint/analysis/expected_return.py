"""Expected return computation for portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, cast

import plotly.graph_objects as go

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio

from waypoint.analysis.methods.returns import PortfolioReturnMethod, ReturnMethod
from waypoint.enums import PERIODS_PER_YEAR, Frequency


@dataclass(frozen=True)
class ExpectedReturnResult:
    """Result of an expected return computation.

    Attributes
    ----------
    per_asset:
        Mapping of asset name to annualised expected return.
    portfolio:
        Portfolio-level expected return = weighted sum of per-asset values.
    method_name:
        Name of the method used (for provenance tracking).
    """

    per_asset: dict[str, float]
    portfolio: float
    method_name: str

    def plot(self) -> go.Figure:
        """Horizontal bar chart of per-asset annualised expected returns."""
        from waypoint.analysis.viz import plot_expected_return

        return plot_expected_return(self)


@dataclass
class ExpectedReturn:
    """Computes expected returns for a portfolio using a pluggable method.

    Parameters
    ----------
    method:
        An object implementing the ``ReturnMethod`` protocol.
    """

    method: ReturnMethod | PortfolioReturnMethod

    def compute(
        self,
        portfolio: Portfolio,
        start: date | str | None,
        end: date | str | None,
        frequency: Frequency | str,
    ) -> ExpectedReturnResult:
        """Compute annualised expected returns for each asset and the portfolio.

        Parameters
        ----------
        portfolio:
            Portfolio whose asset returns are used.
        start, end:
            Date range for the historical window.
        frequency:
            Observation frequency used to annualise returns.  Accepts a
            ``Frequency`` member or its lowercase string equivalent.

        Returns
        -------
        ExpectedReturnResult
        """
        freq = Frequency(frequency)
        periods_per_year = PERIODS_PER_YEAR[freq]
        wide = portfolio.get_returns(start, end, frequency=freq)
        asset_cols = [c for c in wide.columns if c != "date"]

        weights = portfolio.weights

        per_asset: dict[str, float] = {}
        if getattr(self.method, "_portfolio_level", False):
            per_asset = cast(PortfolioReturnMethod, self.method).compute(
                wide, weights, periods_per_year
            )
        else:
            per_asset_method = cast(ReturnMethod, self.method)
            for col in asset_cols:
                per_asset[col] = per_asset_method.compute(wide[col], periods_per_year)
        portfolio_return = sum(weights[name] * per_asset[name] for name in asset_cols)

        return ExpectedReturnResult(
            per_asset=per_asset,
            portfolio=portfolio_return,
            method_name=type(self.method).__name__,
        )
