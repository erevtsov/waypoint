"""IndicatorDef: definition of a macro / signal time series.

Unlike ``AssetDef``, an ``IndicatorDef`` describes a series that is fetched
and stored as raw *levels* (e.g. a yield in percent, an index value) rather
than converted to decimal periodic returns.  Use ``waypoint.data.fetch`` with
an ``IndicatorDef`` to get back an ``Indicator`` object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from waypoint.enums import ASSET_FREQUENCIES, Frequency

VALID_VENDORS: frozenset[str] = frozenset({"yfinance", "fred", "eodhd"})


@dataclass(frozen=True)
class IndicatorDef:
    """Immutable definition of a macro / signal series.

    Parameters
    ----------
    name:
        Display name (e.g. "US 10-Year Treasury Yield").
    symbol:
        Vendor-native symbol (e.g. "DGS10").
    vendor:
        Data vendor: ``"yfinance"``, ``"fred"``, or ``"eodhd"``.
    frequency:
        Observation frequency.  Accepts a ``Frequency`` member or its
        lowercase string equivalent (e.g. ``"daily"``).
    unit:
        Optional description of the value unit (e.g. ``"percent"``).
        No automatic conversion is applied; purely informational.
    asset_class:
        Top-level classification (e.g. ``"Macro"``).
    sub_asset_class:
        Second-level classification (e.g. ``"Real Rates"``).
    geography:
        Geographic scope (e.g. ``"US"``).
    """

    name: str
    symbol: str
    vendor: str
    frequency: Frequency = field(default=Frequency.DAILY)
    unit: str = ""
    asset_class: str = ""
    sub_asset_class: str = ""
    geography: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency", Frequency(self.frequency))
        if self.vendor not in VALID_VENDORS:
            raise ValueError(
                f"vendor must be one of {sorted(VALID_VENDORS)}, got {self.vendor!r}"
            )
        if self.frequency not in ASSET_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(ASSET_FREQUENCIES)}, "
                f"got {self.frequency!r}"
            )
