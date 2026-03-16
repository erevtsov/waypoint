"""Built-in asset and indicator catalog.

Pre-defined ``AssetDef`` and ``IndicatorDef`` constants for commonly used
instruments and macro series, organised into submodules by asset class:

* ``wp.catalog.equities``     — equity indices
* ``wp.catalog.fixed_income`` — bonds and inflation series
* ``wp.catalog.real_estate``  — house price indices
* ``wp.catalog.indicators``   — raw level/rate series (``IndicatorDef``)

Usage::

    import waypoint as wp

    equities = wp.fetch(wp.catalog.equities.US_LARGE_CAP, start="2020-01-01", end="2024-12-31")
    bonds    = wp.fetch(wp.catalog.fixed_income.US_AGG_BONDS, start="2020-01-01", end="2024-12-31")
    hpi      = wp.fetch(wp.catalog.real_estate.MA_HPI, start="2010-01-01", end="2024-12-31")
    rf       = wp.fetch(wp.catalog.indicators.US_10Y_YIELD, start="2024-01-01", end="2024-12-31")
    risk_free_rate = float(rf.values["value"].tail(1).item()) / 100

Custom definitions can be created alongside catalog entries::

    from waypoint.asset_def import AssetDef

    MY_FUND = AssetDef(name="My Alt Fund", symbol="XYZ", vendor="eodhd",
                       frequency="daily", asset_class="Alternatives",
                       geography="Global")
"""

from waypoint.catalog import equities, fixed_income, indicators, real_estate

__all__ = [
    "equities",
    "fixed_income",
    "real_estate",
    "indicators",
]
