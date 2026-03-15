"""Built-in asset and indicator catalog.

Pre-defined ``AssetDef`` and ``IndicatorDef`` constants for commonly used
instruments and macro series.  Import these directly instead of constructing
definitions from scratch.

Usage::

    import waypoint as wp

    equities = wp.fetch(wp.catalog.US_LARGE_CAP, start="2020-01-01", end="2024-12-31")
    rf = wp.fetch(wp.catalog.US_10Y_YIELD, start="2024-01-01", end="2024-12-31")
    risk_free_rate = float(rf.values["value"].tail(1).item()) / 100

Custom definitions can be created alongside catalog entries::

    from waypoint.asset_def import AssetDef

    MY_FUND = AssetDef(name="My Alt Fund", symbol="XYZ", vendor="eodhd",
                       frequency="daily", asset_class="Alternatives",
                       geography="Global")
"""

from waypoint.asset_def import AssetDef
from waypoint.enums import Frequency
from waypoint.indicator_def import IndicatorDef

# ---------------------------------------------------------------------------
# Equities
# ---------------------------------------------------------------------------
US_LARGE_CAP = AssetDef(
    name="US Large Cap Equities",
    symbol="^SPX",
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

EMERGING = AssetDef(
    name="Emerging Markets",
    symbol="EEM",
    vendor="yfinance",
    frequency=Frequency.DAILY,
    asset_class="Equities",
    sub_asset_class="Emerging",
    geography="Emerging",
)

# ---------------------------------------------------------------------------
# Fixed income
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Macro / FRED — AssetDef (pct_change applied; values are meaningful as returns)
# ---------------------------------------------------------------------------
CPI_YOY = AssetDef(
    name="CPI YoY",
    symbol="CPIAUCSL",
    vendor="fred",
    frequency=Frequency.MONTHLY,
    asset_class="Macro",
    sub_asset_class="Inflation",
    geography="US",
)

# ---------------------------------------------------------------------------
# Indicators / FRED — IndicatorDef (raw levels; no pct_change)
# ---------------------------------------------------------------------------
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
