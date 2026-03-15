"""Built-in instrument catalog.

Pre-defined ``Instrument`` constants for commonly used assets. Import these
directly instead of constructing ``Instrument`` objects from scratch.

Usage::

    from waypoint.catalog import US_LARGE_CAP, REAL_RATE_10Y
    from waypoint.data import fetch

    spy = fetch(US_LARGE_CAP, start="2020-01-01", end="2024-12-31")

Custom instruments can be defined alongside catalog entries::

    from waypoint.instruments import Instrument

    MY_FUND = Instrument("My Alt Fund", symbol="XYZ", vendor="eodhd",
                         frequency="daily", asset_class="Alternatives",
                         geography="Global")
"""

from waypoint.instruments import Instrument

# ---------------------------------------------------------------------------
# Equities
# ---------------------------------------------------------------------------
US_LARGE_CAP = Instrument(
    name="US Large Cap Equities",
    symbol="SPY",
    vendor="yfinance",
    frequency="daily",
    asset_class="Equities",
    sub_asset_class="Large Cap",
    geography="US",
)

US_SMALL_CAP = Instrument(
    name="US Small Cap Equities",
    symbol="IWM",
    vendor="yfinance",
    frequency="daily",
    asset_class="Equities",
    sub_asset_class="Small Cap",
    geography="US",
)

INTL_DEVELOPED = Instrument(
    name="Intl Developed Equities",
    symbol="EFA",
    vendor="yfinance",
    frequency="daily",
    asset_class="Equities",
    sub_asset_class="Developed",
    geography="International",
)

EMERGING = Instrument(
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
US_AGG_BONDS = Instrument(
    name="US Aggregate Bonds",
    symbol="AGG",
    vendor="yfinance",
    frequency="daily",
    asset_class="Fixed Income",
    sub_asset_class="Aggregate",
    geography="US",
)

US_TIPS = Instrument(
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
REAL_RATE_10Y = Instrument(
    name="10Y Real Rate",
    symbol="DFII10",
    vendor="fred",
    frequency="daily",
    asset_class="Macro",
    sub_asset_class="Real Rates",
    geography="US",
)

CPI_YOY = Instrument(
    name="CPI YoY",
    symbol="CPIAUCSL",
    vendor="fred",
    frequency="monthly",
    asset_class="Macro",
    sub_asset_class="Inflation",
    geography="US",
)
