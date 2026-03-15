"""Tests for the yfinance provider."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


def _make_fake_history(dates: list[date], closes: list[float]) -> MagicMock:
    """Return a fake pandas DataFrame-like object as yfinance would."""
    import pandas as pd

    raw = pd.DataFrame(
        {"Close": closes},
        index=pd.DatetimeIndex([str(d) for d in dates], name="Date"),
    )
    mock = MagicMock()
    mock.empty = False
    mock.reset_index.return_value = raw.reset_index()
    return mock


@pytest.fixture()
def provider():
    from waypoint.data.providers.yfinance import YFinanceProvider

    return YFinanceProvider()


def test_fetch_raw_passes_exclusive_end_to_yfinance(provider) -> None:
    """yfinance end is exclusive; provider must add one day to convert from inclusive."""
    start = date(2020, 1, 2)
    end = date(2020, 1, 5)
    expected_yf_end = (end + timedelta(days=1)).isoformat()

    fake_history = _make_fake_history(
        [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4), date(2020, 1, 5)],
        [100.0, 101.0, 102.0, 103.0],
    )

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_history
        mock_ticker_cls.return_value = mock_ticker

        provider.fetch_raw("SPY", start, end)

    call_kwargs = mock_ticker.history.call_args.kwargs
    assert call_kwargs["end"] == expected_yf_end, (
        f"Expected yfinance end={expected_yf_end!r} (exclusive), got {call_kwargs['end']!r}. "
        "Off-by-one here causes cache to re-fetch on every second call."
    )


def test_fetch_raw_returns_date_and_close_columns(provider) -> None:
    fake_history = _make_fake_history(
        [date(2020, 1, 2), date(2020, 1, 3)],
        [100.0, 101.0],
    )

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_history
        mock_ticker_cls.return_value = mock_ticker

        result = provider.fetch_raw("SPY", date(2020, 1, 2), date(2020, 1, 3))

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["date", "close"]
    assert result["date"].dtype == pl.Date
