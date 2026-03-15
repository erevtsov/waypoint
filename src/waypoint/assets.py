"""Asset: a single investable instrument with a named return series."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from waypoint.instruments import Instrument

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
        Date-indexed ``pl.Series`` of decimal periodic returns (0.01 = 1%).
        Must have dtype ``pl.Date`` as its index column name ``"date"``.
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
    returns: pl.Series
    frequency: str
    asset_class: str = ""
    sub_asset_class: str = ""
    geography: str = ""

    def __post_init__(self) -> None:
        if self.returns.dtype not in (pl.Float32, pl.Float64):
            raise TypeError(
                f"Asset.returns must be a float Series, got dtype {self.returns.dtype}"
            )
        from waypoint.instruments import VALID_FREQUENCIES  # avoid circular at module level

        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(VALID_FREQUENCIES)}, "
                f"got {self.frequency!r}"
            )

    @property
    def periods_per_year(self) -> int:
        """Number of periods per calendar year implied by this asset's frequency."""
        return PERIODS_PER_YEAR[self.frequency]

    @classmethod
    def from_instrument(cls, instrument: Instrument, returns: pl.Series) -> Asset:
        """Construct an Asset from an ``Instrument`` and a return series.

        Copies all metadata fields from the instrument so they flow through
        to portfolio construction and downstream analysis.
        """
        return cls(
            name=instrument.name,
            ticker=instrument.symbol,
            returns=returns,
            frequency=instrument.frequency,
            asset_class=instrument.asset_class,
            sub_asset_class=instrument.sub_asset_class,
            geography=instrument.geography,
        )
