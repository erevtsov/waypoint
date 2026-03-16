"""AssetDef: immutable definition of an investable instrument."""

from __future__ import annotations

from dataclasses import dataclass, field

from waypoint.enums import ASSET_FREQUENCIES, Frequency

VALID_VENDORS: frozenset[str] = frozenset({"yfinance", "fred", "eodhd"})

# Re-exported for any code that imports VALID_FREQUENCIES from here directly.
VALID_FREQUENCIES: frozenset[Frequency] = ASSET_FREQUENCIES

VALID_NORMALIZATIONS: frozenset[str] = frozenset({"pct_change", "rate_to_daily"})


@dataclass(frozen=True)
class AssetDef:
    """Immutable definition of an investable instrument.

    Carries everything needed to fetch, identify, and classify a security.
    Decouples the user-facing display name from the vendor-native symbol so
    portfolios can be described in semantic terms ("US Large Cap Equities")
    while the data layer handles the symbol/vendor detail.

    Parameters
    ----------
    name:
        Human-readable display name. Becomes ``Asset.name`` after fetching.
    symbol:
        Vendor-native ticker or series ID (e.g. "SPY", "DFII10").
    vendor:
        Data provider: ``"yfinance"``, ``"fred"``, or ``"eodhd"``.
    frequency:
        Observation frequency.  Accepts a ``Frequency`` member or its
        lowercase string equivalent (e.g. ``"daily"``).
    normalization:
        How raw vendor data is converted to decimal periodic returns.
        ``"pct_change"`` (default) applies ``pct_change`` on closing prices —
        appropriate for equity/ETF price series.
        ``"rate_to_daily"`` divides the annualized rate by 100 and 360 —
        appropriate for money-market rate series such as T-bill yields
        (e.g. FRED DTB3) that use the ACT/360 day-count convention.
    asset_class:
        Top-level classification (e.g. ``"Equities"``).
    sub_asset_class:
        Second-level classification (e.g. ``"Large Cap"``).
    geography:
        Geographic scope (e.g. ``"US"``).
    """

    name: str
    symbol: str
    vendor: str
    frequency: Frequency = field(default=Frequency.DAILY)
    normalization: str = "pct_change"
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
        if self.normalization not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {sorted(VALID_NORMALIZATIONS)}, "
                f"got {self.normalization!r}"
            )
