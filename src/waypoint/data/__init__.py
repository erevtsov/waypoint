"""Data sub-package — vendor-agnostic fetch API with local parquet cache.

Usage::

    from waypoint.data import fetch
    from waypoint.catalog import US_LARGE_CAP

    spy = fetch(US_LARGE_CAP, start="2020-01-01", end="2024-12-31")
    spy = fetch(US_LARGE_CAP, start="2020-01-01", end="2024-12-31", force_refresh=True)
"""

from __future__ import annotations

from datetime import date

from dotenv import load_dotenv

from waypoint.asset_def import AssetDef
from waypoint.assets import Asset
from waypoint.data.cache import load_or_fetch, snap_to_month_boundaries
from waypoint.data.normalize import to_returns
from waypoint.data.providers import get_provider

load_dotenv(override=False)  # shell env vars take precedence over .env


def fetch(
    instrument: AssetDef,
    start: str | date,
    end: str | date,
    *,
    force_refresh: bool = False,
) -> Asset:
    """Fetch historical return data for *instrument* and return an ``Asset``.

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
        Security definition (from ``waypoint.catalog`` or a custom
        ``AssetDef``).
    start:
        Start date as ``date`` or ISO-8601 string (e.g. ``"2020-01-01"``).
    end:
        End date as ``date`` or ISO-8601 string (e.g. ``"2024-12-31"``).
    force_refresh:
        If True, bypass the cache and re-fetch from the vendor, overwriting
        any existing cached data for this (vendor, symbol) pair.

    Returns
    -------
    Asset
        Asset whose ``name``, ``ticker``, and metadata mirror *instrument*.
        ``returns`` is a ``pl.DataFrame`` with columns ``"date"`` and ``"returns"``.
    """
    start_dt = date.fromisoformat(start) if isinstance(start, str) else start
    end_dt = date.fromisoformat(end) if isinstance(end, str) else end

    if instrument.frequency == "daily":
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

    returns = to_returns(raw_df)
    return Asset.from_asset_def(instrument, returns)


__all__ = ["fetch"]
