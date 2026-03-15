"""Asset and LeveragedAsset: investable instruments with return series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from waypoint.asset_def import AssetDef
from waypoint.enums import ASSET_FREQUENCIES, PERIODS_PER_YEAR, Frequency


@dataclass
class Asset:
    """A single investable instrument with a date-indexed return series.

    Parameters
    ----------
    name:
        Display name (e.g. "US Large Cap Equities").
    ticker:
        Vendor-native symbol (e.g. "SPY").
    returns:
        ``pl.DataFrame`` with columns ``"date"`` (``pl.Date``) and
        ``"returns"`` (``pl.Float64``) of decimal periodic returns (0.01 = 1%).
    frequency:
        Observation frequency.  Accepts a ``Frequency`` member or its
        lowercase string equivalent (e.g. ``"daily"``).
    asset_class:
        Top-level classification (e.g. "Equities").
    sub_asset_class:
        Second-level classification (e.g. "Large Cap").
    geography:
        Geographic scope (e.g. "US").
    """

    name: str
    ticker: str
    returns: pl.DataFrame
    frequency: Frequency
    asset_class: str = ""
    sub_asset_class: str = ""
    geography: str = ""

    def __post_init__(self) -> None:
        self.frequency = Frequency(self.frequency)
        cols = self.returns.columns
        if cols != ["date", "returns"]:
            raise TypeError(
                f"Asset.returns must be a DataFrame with columns ['date', 'returns'], got {cols}"
            )
        if self.returns["date"].dtype != pl.Date:
            raise TypeError(
                f"Asset.returns['date'] must be pl.Date, got {self.returns['date'].dtype}"
            )
        if self.returns["returns"].dtype not in (pl.Float32, pl.Float64):
            raise TypeError(
                f"Asset.returns['returns'] must be a float column, "
                f"got {self.returns['returns'].dtype}"
            )
        if self.frequency not in ASSET_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(ASSET_FREQUENCIES)}, "
                f"got {self.frequency!r}"
            )

    @property
    def periods_per_year(self) -> int:
        """Number of periods per calendar year implied by this asset's frequency."""
        return PERIODS_PER_YEAR[self.frequency]

    def get_returns(self, start: date | str, end: date | str) -> pl.DataFrame:
        """Return the ``["date", "returns"]`` DataFrame filtered to *[start, end]*.

        Provides a uniform interface with ``AssetDef`` and ``Factor`` so that
        analytics can accept any of the three without special-casing.
        """
        start_dt = date.fromisoformat(start) if isinstance(start, str) else start
        end_dt = date.fromisoformat(end) if isinstance(end, str) else end
        return self.returns.filter(
            (pl.col("date") >= start_dt) & (pl.col("date") <= end_dt)
        )

    @classmethod
    def from_asset_def(cls, asset_def: AssetDef, returns: pl.DataFrame) -> Asset:
        """Construct an Asset from an ``AssetDef`` and a return DataFrame.

        Copies all metadata fields from the asset definition so they flow
        through to portfolio construction and downstream analysis.
        """
        return cls(
            name=asset_def.name,
            ticker=asset_def.symbol,
            returns=returns,
            frequency=asset_def.frequency,
            asset_class=asset_def.asset_class,
            sub_asset_class=asset_def.sub_asset_class,
            geography=asset_def.geography,
        )


@dataclass(frozen=True)
class LeveragedAsset:
    """An asset wrapper that applies a constant-leverage return transformation.

    Models any leveraged position: a mortgaged property, a margin account, or
    a leveraged ETF.  The leveraged period return is computed as::

        r_lev = leverage_ratio × r_asset
                − (leverage_ratio − 1) × (financing_cost / periods_per_year)

    This assumes *constant* leverage rebalancing each period (analogous to a
    leveraged ETF), not an amortising fixed-amount loan.  For a mortgage, the
    leverage ratio decreases over time as principal is repaid; using the ratio
    at inception is a reasonable approximation for long-horizon simulations.

    Parameters
    ----------
    asset:
        The underlying asset whose returns are leveraged.
    leverage_ratio:
        Total asset exposure divided by equity.  A $600 K property with $400 K
        equity has leverage_ratio = 1.5.  Must be > 0.  Use 1.0 for no leverage.
    financing_cost:
        Annual rate on the borrowed portion (e.g. ``0.065`` for 6.5%).  Must
        be >= 0.
    name:
        Display name.  Defaults to the underlying asset's name when omitted.
    """

    asset: Asset
    leverage_ratio: float
    financing_cost: float
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.asset.name)
        if self.leverage_ratio <= 0:
            raise ValueError(f"leverage_ratio must be > 0, got {self.leverage_ratio}")
        if self.financing_cost < 0:
            raise ValueError(f"financing_cost must be >= 0, got {self.financing_cost}")

    @property
    def ticker(self) -> str:
        """Ticker of the underlying asset."""
        return self.asset.ticker

    @property
    def frequency(self) -> Frequency:
        """Observation frequency of the underlying asset."""
        return self.asset.frequency

    @property
    def periods_per_year(self) -> int:
        """Periods per year implied by the underlying asset's frequency."""
        return self.asset.periods_per_year

    @property
    def returns(self) -> pl.DataFrame:
        """Full leveraged return series derived from the underlying asset."""
        return self._apply_leverage(self.asset.returns)

    def get_returns(self, start: date | str, end: date | str) -> pl.DataFrame:
        """Return the leveraged ``["date", "returns"]`` DataFrame filtered to *[start, end]*."""
        return self._apply_leverage(self.asset.get_returns(start, end))

    def _apply_leverage(self, df: pl.DataFrame) -> pl.DataFrame:
        cost_per_period = self.financing_cost / self.periods_per_year
        borrowed = self.leverage_ratio - 1.0
        return df.with_columns(
            (self.leverage_ratio * pl.col("returns") - borrowed * cost_per_period).alias("returns")
        )
