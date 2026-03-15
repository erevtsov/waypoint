"""Instrument: a named, vendor-sourced security definition."""

from __future__ import annotations

from dataclasses import dataclass

VALID_VENDORS: frozenset[str] = frozenset({"yfinance", "fred", "eodhd"})
VALID_FREQUENCIES: frozenset[str] = frozenset({"daily", "weekly", "monthly"})


@dataclass(frozen=True)
class Instrument:
    """A named, vendor-sourced security definition.

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
        Data provider: "yfinance" | "fred" | "eodhd".
    frequency:
        Observation frequency: "daily" | "weekly" | "monthly".
    asset_class:
        Top-level classification: "Equities" | "Fixed Income" | "Alternatives" | "Macro".
    sub_asset_class:
        Second-level classification: "Large Cap", "Aggregate", "TIPS", etc.
    geography:
        Geographic scope: "US" | "International" | "Emerging" | "Global".
    """

    name: str
    symbol: str
    vendor: str
    frequency: str
    asset_class: str = ""
    sub_asset_class: str = ""
    geography: str = ""

    def __post_init__(self) -> None:
        if self.vendor not in VALID_VENDORS:
            raise ValueError(
                f"vendor must be one of {sorted(VALID_VENDORS)}, got {self.vendor!r}"
            )
        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(VALID_FREQUENCIES)}, got {self.frequency!r}"
            )
