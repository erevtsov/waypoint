"""Asset: a single investable instrument with a named return series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from waypoint.asset_def import AssetDef

PERIODS_PER_YEAR: dict[str, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


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
        Observation frequency: "daily" | "weekly" | "monthly".
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
    frequency: str
    asset_class: str = ""
    sub_asset_class: str = ""
    geography: str = ""

    def __post_init__(self) -> None:
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
        from waypoint.asset_def import VALID_FREQUENCIES  # avoid circular at module level

        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(VALID_FREQUENCIES)}, "
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
