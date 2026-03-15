"""Normalize raw vendor price/value data to decimal periodic returns."""

from __future__ import annotations

import polars as pl


def to_returns(df: pl.DataFrame) -> pl.Series:
    """Convert a raw price DataFrame to a decimal periodic return Series.

    Parameters
    ----------
    df:
        DataFrame with columns ``"date"`` (``pl.Date``) and ``"close"``
        (``pl.Float64``).  Rows must be sorted ascending by date and must not
        contain nulls in the ``"close"`` column.

    Returns
    -------
    pl.Series
        Series of decimal periodic returns named ``"returns"``.
        The first observation is dropped (no predecessor to diff against),
        so the output has one fewer row than the input.
    """
    df = df.sort("date")
    pct = df.select(
        pl.col("close").pct_change().alias("returns")
    ).get_column("returns")
    # Drop the first row (null from pct_change) and return
    return pct.drop_nulls()
