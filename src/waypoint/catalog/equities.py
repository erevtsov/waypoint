"""Equity asset definitions."""

from waypoint.asset_def import AssetDef
from waypoint.enums import Frequency

US_LARGE_CAP = AssetDef(
    name="US Large Cap Equities",
    symbol="^SPX",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Large Cap",
    geography="US",
)

US_TOTAL_MARKET = AssetDef(
    name="US Total Market",
    symbol="^CRSPTM1",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Total Market",
    geography="US",
)

US_LARGE_CAP_GROWTH = AssetDef(
    name="US Large Cap Growth",
    symbol="^CRSPLCG1",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Large Cap Growth",
    geography="US",
)

NASDAQ_100 = AssetDef(
    name="NASDAQ-100",
    symbol="^NDX",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Large Cap Growth",
    geography="US",
)

RUSSELL_1000 = AssetDef(
    name="Russell 1000",
    symbol="^RUI",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Large Cap",
    geography="US",
)

US_SMALL_CAP = AssetDef(
    name="US Small Cap Equities",
    symbol="^RUT",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Small Cap",
    geography="US",
)

INTL_DEVELOPED = AssetDef(
    name="Intl Developed Equities",
    symbol="EFA",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Developed",
    geography="International",
)

EUROPE_DEVELOPED = AssetDef(
    name="Europe Developed Equities",
    symbol="VGK",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Developed",
    geography="Europe",
)

EMERGING = AssetDef(
    name="Emerging Markets",
    symbol="EEM",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Emerging",
    geography="Emerging",
)

CHINA_TECH = AssetDef(
    name="China Technology",
    symbol="CQQQ",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Technology",
    geography="China",
)

US_FINANCIALS = AssetDef(
    name="Dow Jones US Financials",
    symbol="^DJUSFN",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Financials",
    geography="US",
)
