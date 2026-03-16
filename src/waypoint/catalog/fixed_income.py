"""Fixed income and macro asset definitions."""

from waypoint.asset_def import AssetDef
from waypoint.enums import Frequency

US_AGG_BONDS = AssetDef(
    name="US Aggregate Bonds",
    symbol="AGG",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Fixed Income",
    sub_asset_class="Aggregate",
    geography="US",
)

US_TIPS = AssetDef(
    name="US TIPS",
    symbol="TIP",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Fixed Income",
    sub_asset_class="Inflation-Linked",
    geography="US",
)

CPI_YOY = AssetDef(
    name="CPI YoY",
    symbol="CPIAUCSL",
    vendor="fred",
    frequency=Frequency.MONTHLY,
    asset_class="Macro",
    sub_asset_class="Inflation",
    geography="US",
)

RISK_FREE_RATE = AssetDef(
    name="3-Month T-Bill Rate",
    symbol="DTB3",
    vendor="fred",
    frequency=Frequency.DAILY,
    normalization="rate_to_daily",
    asset_class="Fixed Income",
    sub_asset_class="Risk-Free Rate",
    geography="US",
)
