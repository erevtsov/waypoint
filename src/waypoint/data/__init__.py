"""Data sub-package — vendor-agnostic fetch API with local parquet cache.

Usage::

    import waypoint as wp

    # Returns an Asset (pct_change applied)
    spy = wp.fetch(wp.catalog.US_LARGE_CAP, start="2020-01-01", end="2024-12-31")

    # Returns an Indicator (raw levels, no pct_change)
    rf = wp.fetch(wp.catalog.US_10Y_YIELD, start="2020-01-01", end="2024-12-31")
    risk_free_rate = float(rf.values["value"].tail(1).item()) / 100
"""

from __future__ import annotations

from datetime import date
from typing import overload

from dotenv import load_dotenv

from waypoint.asset_def import AssetDef
from waypoint.assets import Asset
from waypoint.data.cache import load_or_fetch, snap_to_month_boundaries
from waypoint.data.normalize import to_returns
from waypoint.data.providers import get_provider
from waypoint.enums import Frequency
from waypoint.indicator_def import IndicatorDef
from waypoint.indicators import Indicator

load_dotenv(override=False)  # shell env vars take precedence over .env


@overload
def fetch(
    instrument: AssetDef,
    start: str | date,
    end: str | date,
    *,
    force_refresh: bool = ...,
) -> Asset: ...


@overload
def fetch(
    instrument: IndicatorDef,
    start: str | date,
    end: str | date,
    *,
    force_refresh: bool = ...,
) -> Indicator: ...


def fetch(
    instrument: AssetDef | IndicatorDef,
    start: str | date,
    end: str | date,
    *,
    force_refresh: bool = False,
) -> Asset | Indicator:
    """Fetch historical data for *instrument* and return an ``Asset`` or ``Indicator``.

    * ``AssetDef`` → ``Asset``: raw prices are converted to decimal periodic
      returns via ``pct_change``.
    * ``IndicatorDef`` → ``Indicator``: raw levels are returned as-is
      (no ``pct_change``); useful for yield series, rate series, etc.

    The vendor is determined by ``instrument.vendor``; the caller never
    interacts with vendor-specific APIs.  Data is cached locally in parquet
    under ``~/.waypoint/cache/{vendor}/{symbol}.parquet`` (overridable via the
    ``WAYPOINT_CACHE_DIR`` environment variable).

    For daily-frequency instruments, the date range is automatically snapped
    to full calendar-month boundaries before fetching so that daily→monthly
    resampling always has complete months.

    Parameters
    ----------
    instrument:
        An ``AssetDef`` or ``IndicatorDef`` (from ``waypoint.catalog`` or
        a custom definition).
    start:
        Start date as ``date`` or ISO-8601 string (e.g. ``"2020-01-01"``).
    end:
        End date as ``date`` or ISO-8601 string (e.g. ``"2024-12-31"``).
    force_refresh:
        If True, bypass the cache and re-fetch from the vendor, overwriting
        any existing cached data for this (vendor, symbol) pair.

    Returns
    -------
    Asset | Indicator
        ``Asset`` when given an ``AssetDef``; ``Indicator`` when given an
        ``IndicatorDef``.
    """
    start_dt = date.fromisoformat(start) if isinstance(start, str) else start
    end_dt = date.fromisoformat(end) if isinstance(end, str) else end

    if instrument.frequency == Frequency.DAILY:
        start_dt, end_dt = snap_to_month_boundaries(start_dt, end_dt)

    provider = get_provider(instrument.vendor)
    raw_df = load_or_fetch(
        vendor=instrument.vendor,
        symbol=instrument.symbol,
        start=start_dt,
        end=end_dt,
        fetch_fn=provider,
        force_refresh=force_refresh,
    )

    if isinstance(instrument, IndicatorDef):
        import polars as pl

        values = raw_df.rename({"close": "value"}).with_columns(
            pl.col("value").cast(pl.Float64)
        )
        return Indicator.from_indicator_def(instrument, values)

    returns = to_returns(raw_df)
    return Asset.from_asset_def(instrument, returns)


__all__ = ["fetch"]
