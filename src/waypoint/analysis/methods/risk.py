"""Risk estimation method protocols and implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio


@runtime_checkable
class RiskMethod(Protocol):
    """Protocol for covariance estimation methods."""

    def compute(self, returns_df: pl.DataFrame, periods_per_year: int) -> np.ndarray:
        """Compute annualised covariance matrix.

        Parameters
        ----------
        returns_df:
            Wide DataFrame of decimal periodic returns — one column per asset,
            no date column.
        periods_per_year:
            Number of periods per calendar year used to annualise the result.

        Returns
        -------
        np.ndarray
            Annualised covariance matrix of shape (n_assets, n_assets).
        """
        ...


@dataclass(frozen=True)
class SampleCovariance:
    """Sample covariance estimator scaled to annual frequency.

    Covariance = sample_covariance(returns) * periods_per_year.
    """

    def compute(self, returns_df: pl.DataFrame, periods_per_year: int) -> np.ndarray:
        """Return annualised sample covariance matrix."""
        data = returns_df.to_numpy()
        # np.cov returns a 0-D scalar for a single asset; atleast_2d ensures
        # the result always has shape (n_assets, n_assets) per the protocol contract.
        cov: np.ndarray = np.atleast_2d(np.cov(data, rowvar=False)) * periods_per_year
        return cov


@dataclass(frozen=True)
class ViewRisk:
    """User-specified volatilities blended with a correlation structure.

    Reconstructs the covariance matrix as:

        Σ = diag(σ) @ Corr @ diag(σ)

    where ``σ`` are the user-supplied annualised volatilities.  Exactly one
    correlation source must be active per instance:

    * **``correlation_matrix``** — a fully manual ``(n_assets, n_assets)``
      correlation matrix.  Rows/columns ordered to match ``volatilities``
      key insertion order (= portfolio slot order).
    * **``correlation_method``** — a ``RiskMethod`` that derives correlations
      from historical returns.  Omit ``correlation_matrix`` to use this
      mode; defaults to ``SampleCovariance`` when neither is supplied.

    Supplying both raises ``ValueError`` at construction time.

    Parameters
    ----------
    volatilities:
        Mapping of asset slot name → annualised volatility (decimal, e.g.
        0.15 for 15%).  Every asset in the portfolio must have an entry.
    correlation_matrix:
        ``(n_assets, n_assets)`` correlation matrix.  Mutually exclusive
        with ``correlation_method``.
    correlation_method:
        ``RiskMethod`` used to estimate correlations from historical data.
        Mutually exclusive with ``correlation_matrix``.  When both are
        omitted, defaults to ``SampleCovariance()``.

    Raises
    ------
    ValueError
        At construction if both ``correlation_matrix`` and
        ``correlation_method`` are supplied, or at compute time if any
        portfolio asset is missing from ``volatilities``.
    """

    volatilities: dict[str, float]
    correlation_matrix: np.ndarray | None = field(default=None, hash=False, compare=False)
    correlation_method: RiskMethod | None = field(default=None, hash=False, compare=False)

    def __post_init__(self) -> None:
        has_matrix = self.correlation_matrix is not None
        has_method = self.correlation_method is not None
        if has_matrix and has_method:
            raise ValueError(
                "ViewRisk: supply either correlation_matrix or correlation_method, not both."
            )
        if not has_matrix and not has_method:
            # Default: derive correlations from sample covariance
            object.__setattr__(self, "correlation_method", SampleCovariance())

    @classmethod
    def for_portfolio(
        cls,
        portfolio: Portfolio,
        volatilities: dict[str, float],
        correlation_matrix: np.ndarray | None = None,
        correlation_method: RiskMethod | None = None,
    ) -> ViewRisk:
        """Construct and validate against a portfolio's slot names.

        Raises ``ValueError`` immediately if any slot is missing from
        ``volatilities``, if ``correlation_matrix`` has the wrong shape, or
        if both correlation sources are supplied.
        """
        missing = sorted(set(portfolio.names) - set(volatilities))
        if missing:
            raise ValueError(
                f"ViewRisk: missing volatilities for slot(s) {missing}. "
                f"Portfolio slots: {portfolio.names}"
            )
        n = len(portfolio.names)
        if correlation_matrix is not None and correlation_matrix.shape != (n, n):
            raise ValueError(
                f"ViewRisk: correlation_matrix shape {correlation_matrix.shape} "
                f"does not match portfolio size ({n}, {n})."
            )
        return cls(
            volatilities=volatilities,
            correlation_matrix=correlation_matrix,
            correlation_method=correlation_method,
        )

    def compute(self, returns_df: pl.DataFrame, periods_per_year: int) -> np.ndarray:
        """Return annualised covariance using custom vols and the configured correlations."""
        cols = returns_df.columns
        missing = [c for c in cols if c not in self.volatilities]
        if missing:
            raise ValueError(
                f"ViewRisk: missing volatilities for asset(s) {missing}. "
                f"Provided: {sorted(self.volatilities)}"
            )

        if self.correlation_matrix is not None:
            corr = self.correlation_matrix
        else:
            assert self.correlation_method is not None  # guaranteed by __post_init__
            hist_cov: np.ndarray = self.correlation_method.compute(returns_df, periods_per_year)
            hist_vols = np.sqrt(np.maximum(np.diag(hist_cov), 0.0))
            # Avoid division by zero for degenerate (zero-variance) assets
            safe_vols = np.where(hist_vols > 0, hist_vols, 1.0)
            corr = hist_cov / np.outer(safe_vols, safe_vols)

        custom_vols = np.array([self.volatilities[c] for c in cols])
        result: np.ndarray = np.diag(custom_vols) @ corr @ np.diag(custom_vols)
        return result
