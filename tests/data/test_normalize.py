"""Tests for data normalization (prices → decimal returns)."""

import polars as pl

from waypoint.data.normalize import to_returns


def _price_df(prices: list[float]) -> pl.DataFrame:
    from datetime import date, timedelta

    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(len(prices))]
    return pl.DataFrame({"date": dates, "close": prices})


def test_to_returns_length() -> None:
    df = _price_df([100.0, 101.0, 99.0, 102.0])
    returns = to_returns(df)
    # One fewer row than input (first row has no predecessor)
    assert len(returns) == 3


def test_to_returns_values() -> None:
    df = _price_df([100.0, 110.0, 99.0])
    returns = to_returns(df)
    assert abs(returns[0] - 0.10) < 1e-9    # +10%
    assert abs(returns[1] - (-0.10)) < 1e-9  # -10% (110 → 99)


def test_to_returns_series_name() -> None:
    df = _price_df([100.0, 105.0])
    returns = to_returns(df)
    assert returns.name == "returns"


def test_to_returns_no_nulls() -> None:
    df = _price_df([100.0, 101.0, 102.0, 103.0])
    returns = to_returns(df)
    assert returns.null_count() == 0
