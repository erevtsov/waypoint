"""Parquet-based local cache for vendor price data.

Cache layout::

    {cache_root}/{vendor}/{symbol}.parquet

Each file stores the full history for a (vendor, symbol) pair as a sorted
DataFrame with columns ``"date"`` and ``"close"``.  The vendor is part of
the key so switching ``AssetDef.vendor`` always starts with a clean cache.

Staleness rule: date-gap only.  The cache is only re-fetched if the requested
date range extends beyond what is already stored.  Historical prices do not
change, so already-cached rows are never overwritten — except when
``force_refresh=True``, which discards the cached file entirely.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from waypoint.data.providers.base import Provider

_DEFAULT_CACHE_ROOT = Path.home() / ".waypoint" / "cache"


def _cache_root() -> Path:
    env = os.environ.get("WAYPOINT_CACHE_DIR")
    return Path(env) if env else _DEFAULT_CACHE_ROOT


def _cache_path(vendor: str, symbol: str) -> Path:
    return _cache_root() / vendor / f"{symbol}.parquet"


def _read(path: Path) -> pl.DataFrame | None:
    """Return cached DataFrame or None if the file does not exist."""
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _write(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def snap_to_month_boundaries(start: date, end: date) -> tuple[date, date]:
    """Expand a date range to full calendar-month boundaries.

    For daily-frequency instruments, waypoint always fetches complete months
    so that daily→monthly resampling never has partial-month boundary issues.

    Parameters
    ----------
    start:
        Requested start date (inclusive).
    end:
        Requested end date (inclusive).

    Returns
    -------
    tuple[date, date]
        (snapped_start, snapped_end) where snapped_start is the first day of
        start's month and snapped_end is the last day of end's month.
    """
    snapped_start = start.replace(day=1)
    # Last day of end's month: first day of next month minus one day
    if end.month == 12:
        next_month_first = end.replace(year=end.year + 1, month=1, day=1)
    else:
        next_month_first = end.replace(month=end.month + 1, day=1)
    snapped_end = next_month_first - timedelta(days=1)
    return snapped_start, snapped_end


def load_or_fetch(
    vendor: str,
    symbol: str,
    start: date,
    end: date,
    fetch_fn: Provider,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Return a ``date``/``close`` DataFrame for the requested range.

    Checks the local parquet cache first.  Only calls *fetch_fn* for date
    ranges not already present in the cache (date-gap fill strategy).

    Parameters
    ----------
    vendor:
        Vendor name, used as the top-level cache directory.
    symbol:
        Vendor-native symbol.
    start, end:
        Requested date range (inclusive).  Should already be month-snapped
        for daily instruments before calling this function.
    fetch_fn:
        Provider instance whose ``fetch_raw`` is called for missing date ranges.
    force_refresh:
        If True, discard the cached file and re-fetch the full range from the
        vendor.
    """
    path = _cache_path(vendor, symbol)

    if force_refresh and path.exists():
        path.unlink()

    cached = _read(path)

    if cached is not None and not force_refresh:
        cached_min: date = cached["date"].min()  # type: ignore[assignment]
        cached_max: date = cached["date"].max()  # type: ignore[assignment]

        # Determine which, if any, date ranges are missing
        need_before = start < cached_min
        need_after = end > cached_max

        new_frames: list[pl.DataFrame] = []

        if need_before:
            # Gap may be entirely non-trading days (holidays/weekends); skip if empty
            try:
                new_frames.append(
                    fetch_fn.fetch_raw(symbol, start, cached_min - timedelta(days=1))
                )
            except ValueError:
                pass
        if need_after:
            # Gap may be entirely non-trading days (holidays/weekends); skip if empty
            try:
                new_frames.append(
                    fetch_fn.fetch_raw(symbol, cached_max + timedelta(days=1), end)
                )
            except ValueError:
                pass

        if new_frames:
            combined = pl.concat([cached, *new_frames]).unique("date").sort("date")
            _write(path, combined)
            cached = combined

        return cached.filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        )

    # No cache or force_refresh — fetch full range
    fresh: pl.DataFrame = fetch_fn.fetch_raw(symbol, start, end)
    _write(path, fresh.sort("date"))
    return fresh
