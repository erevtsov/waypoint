"""Return estimation method protocols and implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
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
class ArithmeticMean:
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
class GeometricMean:
    """Geometric (compounded) annualised mean of historical returns.

    Computes ``exp(mean(log(1 + r)) × ppy) − 1``, which equals the
    constant per-period return that would produce the same terminal wealth
    as the historical return sequence.  Preferred over ``ArithmeticMean``
    for long-horizon simulations because the arithmetic mean overstates
    expected compound growth by approximately ½σ².
    """

    def compute(self, returns: pl.Series, periods_per_year: int) -> float:
        """Return the geometric annualised mean."""
        values = returns.drop_nulls().to_numpy()
        if len(values) == 0:
            return 0.0
        return float(np.exp(np.mean(np.log1p(values)) * periods_per_year) - 1.0)


@dataclass(frozen=True)
class EWMAMean:
    """Exponentially weighted annualised mean of historical returns.

    Assigns geometrically decaying weights to past observations so that
    recent returns receive more weight.  Useful when you believe recent
    market regimes are more informative than older ones.

    The weight assigned to an observation ``k`` periods in the past is
    proportional to ``decay_factor ** k``.  Weights are normalised to
    sum to 1 before computing the mean.

    Parameters
    ----------
    decay_factor:
        Per-period decay rate λ ∈ (0, 1).  Higher values retain more
        history; lower values react faster to recent observations.
        Common choices: 0.94 (RiskMetrics daily), 0.97 (RiskMetrics monthly).
    """

    decay_factor: float = field(default=0.94)

    def compute(self, returns: pl.Series, periods_per_year: int) -> float:
        """Return the EWMA annualised mean."""
        values = returns.drop_nulls().to_numpy()
        n = len(values)
        if n == 0:
            return 0.0
        # Oldest observation has the lowest weight; most recent = 1.
        weights = self.decay_factor ** np.arange(n - 1, -1, -1, dtype=np.float64)
        weights /= weights.sum()
        return float(np.dot(weights, values)) * periods_per_year


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
