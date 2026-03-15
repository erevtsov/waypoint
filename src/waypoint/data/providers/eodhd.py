"""EODHD provider — global equities and fundamentals via EOD Historical Data."""

from __future__ import annotations

from datetime import date

import polars as pl


class EodhdProvider:
    """Fetches price data from EOD Historical Data via the ``eodhd`` library.

    Requires an EODHD API key set in the ``EODHD_API_KEY`` environment variable.
    ``eodhd`` is an optional dependency; install with::

        uv add waypoint[eodhd]
    """

    def fetch_raw(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        """Return adjusted-close price data as a Polars DataFrame."""
        import os

        try:
            from eodhd import APIClient
        except ImportError as exc:
            raise ImportError(
                "eodhd is required for this provider. "
                "Install it with: uv add waypoint[eodhd]"
            ) from exc

        api_key = os.environ.get("EODHD_API_KEY")
        if not api_key:
            raise OSError(
                "EODHD_API_KEY environment variable is not set. "
                "Obtain a key at https://eodhd.com/"
            )

        client = APIClient(api_key)
        raw = client.get_historical_data(
            symbol,
            "d",
            date_from=start.isoformat(),
            date_to=end.isoformat(),
        )
        if not raw:
            raise ValueError(
                f"EODHD returned no data for symbol {symbol!r} "
                f"between {start} and {end}."
            )

        df = pl.DataFrame(raw)
        return (
            df.select(["date", "adjusted_close"])
            .rename({"adjusted_close": "close"})
            .with_columns(pl.col("date").cast(pl.Date))
        )
