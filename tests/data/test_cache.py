"""Tests for the parquet cache layer."""

from datetime import date

import polars as pl
import pytest

from waypoint.data.cache import snap_to_month_boundaries


def test_snap_already_full_month() -> None:
    start, end = snap_to_month_boundaries(date(2020, 1, 1), date(2020, 3, 31))
    assert start == date(2020, 1, 1)
    assert end == date(2020, 3, 31)


def test_snap_mid_month_start_and_end() -> None:
    start, end = snap_to_month_boundaries(date(2020, 3, 15), date(2020, 11, 20))
    assert start == date(2020, 3, 1)
    assert end == date(2020, 11, 30)


def test_snap_december_year_boundary() -> None:
    start, end = snap_to_month_boundaries(date(2020, 12, 10), date(2020, 12, 20))
    assert start == date(2020, 12, 1)
    assert end == date(2020, 12, 31)


def test_snap_single_day_in_february_leap_year() -> None:
    start, end = snap_to_month_boundaries(date(2020, 2, 15), date(2020, 2, 15))
    assert start == date(2020, 2, 1)
    assert end == date(2020, 2, 29)  # 2020 is a leap year


def test_load_or_fetch_writes_and_reads_cache(tmp_path: pytest.fixture) -> None:  # type: ignore[type-arg]
    """load_or_fetch should write a parquet file and return the data."""
    import os

    os.environ["WAYPOINT_CACHE_DIR"] = str(tmp_path)

    try:
        from waypoint.data.cache import load_or_fetch

        prices = [100.0, 101.0, 102.0, 103.0]
        dates = [
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 1, 4),
            date(2020, 1, 5),
        ]
        fake_df = pl.DataFrame({"date": dates, "close": prices})

        class _FakeProvider:
            called: int = 0

            def fetch_raw(
                self, symbol: str, start: date, end: date
            ) -> pl.DataFrame:
                self.called += 1
                return fake_df

        provider = _FakeProvider()
        result = load_or_fetch(
            vendor="test_vendor",
            symbol="TST",
            start=date(2020, 1, 2),
            end=date(2020, 1, 5),
            fetch_fn=provider,
        )

        assert len(result) == 4
        assert provider.called == 1

        # Second call should hit cache — provider should NOT be called again
        result2 = load_or_fetch(
            vendor="test_vendor",
            symbol="TST",
            start=date(2020, 1, 2),
            end=date(2020, 1, 5),
            fetch_fn=provider,
        )
        assert len(result2) == 4
        assert provider.called == 1  # still 1 — cache hit

    finally:
        del os.environ["WAYPOINT_CACHE_DIR"]


def test_force_refresh_bypasses_cache(tmp_path: pytest.fixture) -> None:  # type: ignore[type-arg]
    import os

    os.environ["WAYPOINT_CACHE_DIR"] = str(tmp_path)

    try:
        from waypoint.data.cache import load_or_fetch

        dates = [date(2020, 1, 2), date(2020, 1, 3)]
        fake_df = pl.DataFrame({"date": dates, "close": [100.0, 101.0]})

        class _FakeProvider:
            called: int = 0

            def fetch_raw(
                self, symbol: str, start: date, end: date
            ) -> pl.DataFrame:
                self.called += 1
                return fake_df

        provider = _FakeProvider()

        # Populate cache
        load_or_fetch("test_vendor", "TST2", date(2020, 1, 2), date(2020, 1, 3), provider)
        assert provider.called == 1

        # force_refresh should call the provider again
        load_or_fetch(
            "test_vendor", "TST2", date(2020, 1, 2), date(2020, 1, 3), provider, force_refresh=True
        )
        assert provider.called == 2

    finally:
        del os.environ["WAYPOINT_CACHE_DIR"]
