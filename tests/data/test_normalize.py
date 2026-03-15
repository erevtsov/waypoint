"""Tests for data normalization (prices → decimal returns)."""

from datetime import date, timedelta

import polars as pl

from waypoint.data.normalize import to_returns


def _price_df(prices: list[float]) -> pl.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(len(prices))]
    return pl.DataFrame({"date": dates, "close": prices})


def test_to_returns_length() -> None:
    df = _price_df([100.0, 101.0, 99.0, 102.0])
    result = to_returns(df)
    # One fewer row than input (first row has no predecessor)
    assert len(result) == 3


def test_to_returns_values() -> None:
    df = _price_df([100.0, 110.0, 99.0])
    result = to_returns(df)
    assert abs(result["returns"][0] - 0.10) < 1e-9    # +10%
    assert abs(result["returns"][1] - (-0.10)) < 1e-9  # -10% (110 → 99)


def test_to_returns_schema() -> None:
    df = _price_df([100.0, 105.0])
    result = to_returns(df)
    assert result.columns == ["date", "returns"]
    assert result["date"].dtype == pl.Date
    assert result["returns"].dtype in (pl.Float32, pl.Float64)


def test_to_returns_dates_preserved() -> None:
    prices = [100.0, 101.0, 102.0, 103.0]
    df = _price_df(prices)
    result = to_returns(df)
    # Date for each return is the date of the new price (row 1 onward)
    expected_dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(1, len(prices))]
    assert result["date"].to_list() == expected_dates


def test_to_returns_no_nulls() -> None:
    df = _price_df([100.0, 101.0, 102.0, 103.0])
    result = to_returns(df)
    assert result["returns"].null_count() == 0
    assert result["date"].null_count() == 0
