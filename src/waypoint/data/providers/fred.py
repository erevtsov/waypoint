"""FRED provider — macroeconomic series via the Federal Reserve Bank of St. Louis."""

from __future__ import annotations

from datetime import date

import polars as pl


class FredProvider:
    """Fetches economic time series from FRED via the ``fredapi`` library.

    Requires a FRED API key set in the ``FRED_API_KEY`` environment variable.
    ``fredapi`` is an optional dependency; install with::

        uv add waypoint[fred]
    """

    def fetch_raw(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        """Return FRED series data as a Polars DataFrame."""
        import os

        try:
            from fredapi import Fred
        except ImportError as exc:
            raise ImportError(
                "fredapi is required for this provider. "
                "Install it with: uv add waypoint[fred]"
            ) from exc

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise OSError(
                "FRED_API_KEY environment variable is not set. "
                "Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        fred = Fred(api_key=api_key)
        series = fred.get_series(
            symbol,
            observation_start=start.isoformat(),
            observation_end=end.isoformat(),
        )
        if series is None or series.empty:
            raise ValueError(
                f"FRED returned no data for series {symbol!r} "
                f"between {start} and {end}."
            )

        df = pl.DataFrame(
            {
                "date": series.index.to_list(),
                "close": series.values.tolist(),
            }
        )
        # FRED sometimes returns IEEE-754 NaN for missing observations.
        # Polars `drop_nulls` only removes Polars null, not NaN floats, so we
        # must convert NaN → null first before dropping.
        return (
            df.with_columns(pl.col("date").cast(pl.Date))
            .with_columns(pl.col("close").fill_nan(None))
            .drop_nulls()
        )
