"""Built-in asset definition catalog.

Pre-defined ``AssetDef`` constants for commonly used assets. Import these
directly instead of constructing ``AssetDef`` objects from scratch.

Usage::

    from waypoint.catalog import US_LARGE_CAP, REAL_RATE_10Y
    from waypoint.data import fetch

    spy = fetch(US_LARGE_CAP, start="2020-01-01", end="2024-12-31")

Custom asset definitions can be defined alongside catalog entries::

    from waypoint.asset_def import AssetDef

    MY_FUND = AssetDef("My Alt Fund", symbol="XYZ", vendor="eodhd",
                       frequency="daily", asset_class="Alternatives",
                       geography="Global")
"""

from waypoint.asset_def import AssetDef

# ---------------------------------------------------------------------------
# Equities
# ---------------------------------------------------------------------------
US_LARGE_CAP = AssetDef(
    name="US Large Cap Equities",
    symbol="^SPX",
    vendor="yfinance",
    frequency="daily",
    asset_class="Equities",
    sub_asset_class="Large Cap",
    geography="US",
)

US_SMALL_CAP = AssetDef(
    name="US Small Cap Equities",
    symbol="^RUT",
    vendor="yfinance",
    frequency="daily",
    asset_class="Equities",
    sub_asset_class="Small Cap",
    geography="US",
)

INTL_DEVELOPED = AssetDef(
    name="Intl Developed Equities",
    symbol="EFA",
    vendor="yfinance",
    frequency="daily",
    asset_class="Equities",
    sub_asset_class="Developed",
    geography="International",
)

EMERGING = AssetDef(
    name="Emerging Markets",
    symbol="EEM",
    vendor="yfinance",
    frequency="daily",
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
    frequency="daily",
    asset_class="Fixed Income",
    sub_asset_class="Aggregate",
    geography="US",
)

US_TIPS = AssetDef(
    name="US TIPS",
    symbol="TIP",
    vendor="yfinance",
    frequency="daily",
    asset_class="Fixed Income",
    sub_asset_class="Inflation-Linked",
    geography="US",
)

# ---------------------------------------------------------------------------
# Macro / FRED
# ---------------------------------------------------------------------------
REAL_RATE_10Y = AssetDef(
    name="10Y Real Rate",
    symbol="DFII10",
    vendor="fred",
    frequency="daily",
    asset_class="Macro",
    sub_asset_class="Real Rates",
    geography="US",
)

CPI_YOY = AssetDef(
    name="CPI YoY",
    symbol="CPIAUCSL",
    vendor="fred",
    frequency="monthly",
    asset_class="Macro",
    sub_asset_class="Inflation",
    geography="US",
)
