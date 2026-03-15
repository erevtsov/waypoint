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
