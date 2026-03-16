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


def _ledoit_wolf_alpha(X: np.ndarray) -> float:
    """Compute the analytical Ledoit-Wolf shrinkage intensity toward the scaled identity.

    Implements the Oracle estimator from Ledoit & Wolf, "A well-conditioned
    estimator for large-dimensional covariance matrices", JMVA 2004.

    Parameters
    ----------
    X:
        ``(T, p)`` centered (demeaned) returns matrix.

    Returns
    -------
    float
        Shrinkage intensity α ∈ [0, 1]; 0.0 when ``T ≤ 1`` or the sample
        covariance is already proportional to the identity.
    """
    T, p = X.shape
    if T <= 1:
        return 0.0

    S = X.T @ X / T  # biased (1/T) sample covariance — required by LW formula
    trace_s = float(np.trace(S))
    trace_s2 = float(np.trace(S @ S))

    # Squared Frobenius distance from S to scaled identity target, normalised by p.
    delta = (trace_s2 - trace_s**2 / p) / p
    if delta == 0.0:
        return 0.0  # already proportional to identity

    # Oracle beta: average squared Frobenius distance of rank-1 sample to S.
    # beta = (1/(T²p)) * Σ_t ||x_t x_t' − S||²_F
    #      = (1/(T²p)) * [Σ_t ||x_t||^4  −  T * trace(S²)]
    sq_norms = np.sum(X**2, axis=1)  # ||x_t||² for each observation t
    beta_raw = (float(np.sum(sq_norms**2)) - T * trace_s2) / (T**2 * p)
    beta = min(beta_raw, delta)  # positive-part clamp

    return float(np.clip(beta / delta, 0.0, 1.0))


@dataclass(frozen=True)
class LedoitWolf:
    """Ledoit-Wolf analytical shrinkage estimator toward the scaled identity.

    Shrinks the sample covariance matrix toward ``μI`` (where
    ``μ = trace(S) / p``), pulling noisy off-diagonal entries toward zero
    without introducing free parameters.  The shrinkage intensity is
    determined analytically from the data.

    Preferred over ``SampleCovariance`` when the number of assets is large
    relative to the number of observations (high p/T ratio), where the sample
    covariance matrix becomes ill-conditioned.

    Reference
    ---------
    Ledoit & Wolf, "A well-conditioned estimator for large-dimensional
    covariance matrices", Journal of Multivariate Analysis, 2004.
    """

    def compute(self, returns_df: pl.DataFrame, periods_per_year: int) -> np.ndarray:
        """Return the annualised Ledoit-Wolf shrunk covariance matrix."""
        data = returns_df.to_numpy()
        X = data - data.mean(axis=0)  # center

        alpha = _ledoit_wolf_alpha(X)

        T, p = X.shape
        # Use unbiased S for the final estimate (ddof=1) so it is consistent
        # with SampleCovariance; alpha is still derived from biased S as LW requires.
        S = np.atleast_2d(np.cov(X, rowvar=False)) if T > 1 else X.T @ X
        mu = float(np.trace(S)) / p
        shrunk: np.ndarray = (1.0 - alpha) * S + alpha * mu * np.eye(p)
        return shrunk * periods_per_year


@dataclass(frozen=True)
class EWMACovariance:
    """Exponentially weighted covariance estimator.

    Assigns geometrically decaying weights to past observations so that recent
    return pairs receive more weight.  Useful when volatility regimes shift
    over time and you want the risk estimate to adapt faster than a sample
    covariance would.

    The weight for an observation ``k`` periods in the past is proportional to
    ``decay_factor ** k``.  Weights are normalised to sum to 1.  Both the
    weighted mean and the weighted covariance are computed from the same decay
    schedule.

    Parameters
    ----------
    decay_factor:
        Per-period decay rate λ ∈ (0, 1).  Higher values retain more history;
        lower values react faster to recent observations.  Common choices:
        0.94 (RiskMetrics daily), 0.97 (RiskMetrics monthly).
    """

    decay_factor: float = field(default=0.94)

    def compute(self, returns_df: pl.DataFrame, periods_per_year: int) -> np.ndarray:
        """Return the annualised EWMA covariance matrix."""
        data = returns_df.to_numpy()  # (T, p)
        T, p = data.shape

        # Weights: oldest observation has the lowest weight; most recent = 1.
        weights = self.decay_factor ** np.arange(T - 1, -1, -1, dtype=np.float64)
        weights /= weights.sum()

        # Weighted mean then weighted covariance.
        mu = weights @ data  # (p,)
        X_c = data - mu  # (T, p) centred
        cov: np.ndarray = (X_c * weights[:, np.newaxis]).T @ X_c
        return cov * periods_per_year


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
