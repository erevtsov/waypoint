"""Normalize raw vendor price/value data to decimal periodic returns."""

from __future__ import annotations

import polars as pl


def to_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a raw price DataFrame to a date/returns DataFrame.

    Parameters
    ----------
    df:
        DataFrame with columns ``"date"`` (``pl.Date``) and ``"close"``
        (``pl.Float64``).  Rows must be sorted ascending by date and must not
        contain nulls in the ``"close"`` column.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``"date"`` (``pl.Date``) and ``"returns"``
        (``pl.Float64``) of decimal periodic returns.
        The first row is dropped (no predecessor to diff against),
        so the output has one fewer row than the input.
    """
    df = df.sort("date")
    result = df.select(
        pl.col("date"),
        pl.col("close").pct_change().alias("returns"),
    )
    # Drop the first row (null returns from pct_change)
    return result.drop_nulls()


MONEY_MARKET_DAY_COUNT = 360
"""Day-count denominator for ACT/360 money-market conventions (e.g. T-bill rates)."""


def rate_to_daily_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Convert an annualized rate DataFrame to a daily decimal return DataFrame.

    Uses the ACT/360 money-market day-count convention::

        r_daily = annualized_rate / 100 / 360

    Parameters
    ----------
    df:
        DataFrame with columns ``"date"`` (``pl.Date``) and ``"close"``
        (``pl.Float64``) where ``"close"`` holds annualized rates in percent
        (e.g. ``5.25`` for 5.25% per year).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``"date"`` (``pl.Date``) and ``"returns"``
        (``pl.Float64``) of decimal daily returns.
    """
    result = df.select(
        pl.col("date"),
        (pl.col("close") / 100 / MONEY_MARKET_DAY_COUNT).alias("returns"),
    )
    # FRED omits rate observations on US federal holidays; those days appear
    # in the series with NaN values.  Drop them so they don't propagate into
    # downstream computations (covariance, resampling, etc.).
    return result.with_columns(pl.col("returns").fill_nan(None)).drop_nulls()
