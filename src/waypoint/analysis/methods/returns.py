"""Return estimation method protocols and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl


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
