"""Return estimation method protocols and implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
import polars as pl

from waypoint.assets import Asset

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


class PortfolioReturnMethod(Protocol):
    """Protocol for portfolio-level expected-return methods.

    Unlike ``ReturnMethod``, these methods receive the full wide DataFrame
    (all assets and dates) and return a mapping of asset name → annualised
    expected return.  Implementations must set the class variable
    ``_portfolio_level = True`` so that ``ExpectedReturn`` can dispatch
    correctly without relying on ``isinstance`` signature matching.
    """

    _portfolio_level: ClassVar[Literal[True]]

    def compute(
        self,
        wide: pl.DataFrame,
        weights: dict[str, float],
        periods_per_year: int,
    ) -> dict[str, float]:
        """Compute annualised expected returns for all portfolio assets.

        Parameters
        ----------
        wide:
            DataFrame with a ``"date"`` column and one column per asset.
        weights:
            Portfolio weights keyed by asset name.
        periods_per_year:
            Used to annualise return estimates.

        Returns
        -------
        dict[str, float]
            Mapping of asset name to annualised expected return.
        """
        ...


def _capm_expected_return(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    e_rm: float,
    rf: float,
) -> float:
    """Compute CAPM expected return for a single asset.

    Parameters
    ----------
    asset_returns:
        Array of periodic returns for the asset, aligned with ``market_returns``.
    market_returns:
        Array of periodic market returns.
    e_rm:
        Annualised expected market return.
    rf:
        Annualised risk-free rate.

    Returns
    -------
    float
        Annualised CAPM expected return: ``rf + beta * (e_rm - rf)``.

    Raises
    ------
    ValueError
        If ``market_returns`` has zero variance.
    """
    var_m = float(np.var(market_returns, ddof=1))
    if var_m == 0.0:
        raise ValueError("CAPM: market asset has zero return variance.")
    cov_im = float(np.cov(asset_returns, market_returns, ddof=1)[0, 1])
    beta = cov_im / var_m
    return rf + beta * (e_rm - rf)


@dataclass(frozen=True)
class CAPM:
    """CAPM single-factor expected return model.

    Estimates per-asset expected returns as:

        E[Ri] = Rf + βi × (E[Rm] − Rf)

    where ``βi = cov(Ri, Rm) / var(Rm)`` is computed from the overlapping
    historical returns of the portfolio, market, and (if an Asset)
    risk-free series.

    Parameters
    ----------
    market:
        Asset representing the market benchmark (e.g. a broad equity index).
    risk_free:
        Risk-free rate as a constant annualised decimal (e.g. ``0.04``) or
        an Asset whose return series represents the risk-free instrument
        (e.g. a short-term T-bill ETF).  Must be supplied explicitly — there
        is no default.
    market_return_method:
        Method used to estimate ``E[Rm]`` from the aligned market return
        series.  Defaults to ``GeometricMean``.
    """

    _portfolio_level: ClassVar[Literal[True]] = True

    market: Asset
    risk_free: Asset | float
    market_return_method: ReturnMethod = field(default_factory=GeometricMean)

    def compute(
        self,
        wide: pl.DataFrame,
        weights: dict[str, float],
        periods_per_year: int,
    ) -> dict[str, float]:
        """Return CAPM expected returns for each asset in the portfolio.

        Parameters
        ----------
        wide:
            Wide DataFrame with a ``"date"`` column and one column per asset.
        weights:
            Portfolio weights (not used by CAPM; present to satisfy protocol).
        periods_per_year:
            Used to annualise the expected market and risk-free returns.

        Returns
        -------
        dict[str, float]
            Mapping of asset name to annualised CAPM expected return.

        Raises
        ------
        ValueError
            If there are no overlapping dates between the portfolio and the
            market or risk-free assets, or if the market return variance is zero.
        """
        asset_cols = [c for c in wide.columns if c != "date"]

        # Align market returns to the portfolio date window.
        mkt_df = self.market.returns.rename({"returns": "__market__"})
        aligned = wide.join(mkt_df, on="date", how="inner")
        if len(aligned) == 0:
            raise ValueError(
                "CAPM: no overlapping dates between portfolio and market asset."
            )

        # Optionally align risk-free returns.
        if isinstance(self.risk_free, float):
            rf_annualised = self.risk_free
        else:
            rf_df = self.risk_free.returns.rename({"returns": "__rf__"})
            aligned = aligned.join(rf_df, on="date", how="inner")
            if len(aligned) == 0:
                raise ValueError(
                    "CAPM: no overlapping dates between portfolio, market, "
                    "and risk-free asset."
                )
            rf_annualised = self.market_return_method.compute(
                aligned["__rf__"], periods_per_year
            )

        mkt_returns = aligned["__market__"].to_numpy()
        e_rm = self.market_return_method.compute(aligned["__market__"], periods_per_year)

        per_asset: dict[str, float] = {}
        for col in asset_cols:
            per_asset[col] = _capm_expected_return(
                asset_returns=aligned[col].to_numpy(),
                market_returns=mkt_returns,
                e_rm=e_rm,
                rf=rf_annualised,
            )

        return per_asset
