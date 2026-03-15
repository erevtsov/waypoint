"""yfinance provider — equities, ETFs, and crypto via Yahoo Finance."""

from __future__ import annotations

from datetime import date

import polars as pl


class YFinanceProvider:
    """Fetches adjusted-close price data via the ``yfinance`` library.

    ``yfinance`` is an optional dependency; importing this module will raise
    ``ImportError`` if it is not installed.  Install with::

        uv add waypoint[yfinance]
    """

    def fetch_raw(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        """Return daily OHLCV data from Yahoo Finance as a Polars DataFrame."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for this provider. "
                "Install it with: uv add waypoint[yfinance]"
            ) from exc

        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
        )
        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for symbol {symbol!r} "
                f"between {start} and {end}."
            )

        df = pl.from_pandas(raw.reset_index()[["Date", "Close"]])
        return df.rename({"Date": "date", "Close": "close"}).with_columns(
            pl.col("date").cast(pl.Date)
        )
