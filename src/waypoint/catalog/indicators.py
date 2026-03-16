"""Indicator definitions — raw level/rate series (no pct_change)."""

from waypoint.enums import Frequency
from waypoint.indicator_def import IndicatorDef

US_10Y_YIELD = IndicatorDef(
    name="US 10-Year Treasury Yield",
    symbol="DGS10",
    vendor="fred",
    frequency=Frequency.DAILY,
    unit="percent",
    asset_class="Macro",
    sub_asset_class="Risk-Free Rate",
    geography="US",
)

REAL_RATE_10Y = IndicatorDef(
    name="US 10-Year Real Rate",
    symbol="DFII10",
    vendor="fred",
    frequency=Frequency.DAILY,
    unit="percent",
    asset_class="Macro",
    sub_asset_class="Real Rates",
    geography="US",
)
