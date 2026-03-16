"""Tests for data normalization functions."""

from datetime import date

import polars as pl
import pytest

from waypoint.data.normalize import MONEY_MARKET_DAY_COUNT, rate_to_daily_returns, to_returns


def _make_price_df(closes: list[float]) -> pl.DataFrame:
    dates = [date(2024, 1, d + 1) for d in range(len(closes))]
    return pl.DataFrame({"date": dates, "close": closes})


def _make_rate_df(rates: list[float]) -> pl.DataFrame:
    dates = [date(2024, 1, d + 1) for d in range(len(rates))]
    return pl.DataFrame({"date": dates, "close": rates})


def test_to_returns_basic() -> None:
    df = _make_price_df([100.0, 110.0, 99.0])
    result = to_returns(df)
    assert result.columns == ["date", "returns"]
    assert len(result) == 2  # first row dropped
    assert result["returns"][0] == pytest.approx(0.10)
    assert result["returns"][1] == pytest.approx((99.0 - 110.0) / 110.0)


def test_to_returns_drops_first_row() -> None:
    df = _make_price_df([50.0, 55.0])
    result = to_returns(df)
    assert len(result) == 1
    assert result["date"][0] == date(2024, 1, 2)


def test_rate_to_daily_returns_schema() -> None:
    df = _make_rate_df([5.0, 5.25, 5.10])
    result = rate_to_daily_returns(df)
    assert result.columns == ["date", "returns"]
    assert len(result) == 3  # no row dropped — no differencing


def test_rate_to_daily_returns_values() -> None:
    df = _make_rate_df([5.25])
    result = rate_to_daily_returns(df)
    expected = 5.25 / 100 / MONEY_MARKET_DAY_COUNT
    assert result["returns"][0] == pytest.approx(expected)


def test_rate_to_daily_returns_zero_rate() -> None:
    df = _make_rate_df([0.0])
    result = rate_to_daily_returns(df)
    assert result["returns"][0] == pytest.approx(0.0)
