"""Return estimation method protocols and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio


@runtime_checkable
class ReturnMethod(Protocol):
    """Protocol for expected-return estimation methods."""

    def compute(self, returns: pl.Series, periods_per_year: int) -> float:
        """Compute annualised expected return from a return series.

        Parameters
        ----------
        returns:
            Decimal periodic returns (0.01 = 1%).
        periods_per_year:
            Number of periods per calendar year (252 daily, 12 monthly, etc.).

        Returns
        -------
        float
            Annualised expected return as a decimal.
        """
        ...


@dataclass(frozen=True)
class HistoricalMean:
    """Arithmetic annualised mean of historical returns.

    Expected return = sample mean of periodic returns * periods_per_year.
    """

    def compute(self, returns: pl.Series, periods_per_year: int) -> float:
        """Return arithmetic annualised mean."""
        mean = returns.mean()
        if mean is None:
            return 0.0
        # cast through int conversion path to satisfy strict mypy typing
        return float(mean) * periods_per_year  # type: ignore[arg-type]


@dataclass(frozen=True)
class ViewReturn:
    """User-specified annualised expected returns — a forward-looking return view.

    Ignores historical return data entirely; the supplied values are returned
    as-is.  The asset is identified by ``pl.Series.name``, which matches the
    slot name in the portfolio.

    Parameters
    ----------
    expected_returns:
        Mapping of asset slot name → annualised expected return (decimal).
        Every asset in the portfolio must have an entry.

    Raises
    ------
    ValueError
        If an asset name is not found in ``expected_returns`` at compute time.
    """

    expected_returns: dict[str, float]

    @classmethod
    def for_portfolio(
        cls,
        portfolio: Portfolio,
        expected_returns: dict[str, float],
    ) -> ViewReturn:
        """Construct and validate against a portfolio's slot names.

        Raises ``ValueError`` immediately if any slot is missing from
        ``expected_returns``, rather than failing silently at compute time.
        """
        missing = sorted(set(portfolio.names) - set(expected_returns))
        if missing:
            raise ValueError(
                f"ViewReturn: missing expected_returns for slot(s) {missing}. "
                f"Portfolio slots: {portfolio.names}"
            )
        return cls(expected_returns=expected_returns)

    def compute(self, returns: pl.Series, periods_per_year: int) -> float:
        """Return the pre-specified annualised expected return for this asset."""
        name = returns.name
        if name not in self.expected_returns:
            raise ValueError(
                f"ViewReturn: no expected return specified for asset '{name}'. "
                f"Provided keys: {sorted(self.expected_returns)}"
            )
        return self.expected_returns[name]
