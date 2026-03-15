"""Expected return computation for portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio

from waypoint.analysis.methods.returns import ReturnMethod
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


@dataclass
class ExpectedReturn:
    """Computes expected returns for a portfolio using a pluggable method.

    Parameters
    ----------
    method:
        An object implementing the ``ReturnMethod`` protocol.
    """

    method: ReturnMethod

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

        per_asset: dict[str, float] = {}
        for col in asset_cols:
            per_asset[col] = self.method.compute(wide[col], periods_per_year)

        weights = portfolio.weights
        portfolio_return = sum(weights[name] * per_asset[name] for name in asset_cols)

        return ExpectedReturnResult(
            per_asset=per_asset,
            portfolio=portfolio_return,
            method_name=type(self.method).__name__,
        )
