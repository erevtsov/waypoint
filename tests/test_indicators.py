"""Tests for the Indicator class."""

from datetime import date, timedelta

import polars as pl
import pytest

from waypoint.enums import Frequency
from waypoint.indicator_def import IndicatorDef
from waypoint.indicators import Indicator


def _make_values(n: int = 10) -> pl.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": dates, "value": [float(i) for i in range(n)]})


def test_indicator_construction() -> None:
    ind = Indicator(name="X", symbol="X", values=_make_values(), frequency="daily")
    assert ind.name == "X"
    assert ind.frequency is Frequency.DAILY


def test_indicator_frequency_normalised_to_enum() -> None:
    ind = Indicator(name="X", symbol="X", values=_make_values(), frequency="monthly")
    assert ind.frequency is Frequency.MONTHLY


def test_indicator_rejects_wrong_columns() -> None:
    bad = pl.DataFrame({"date": [date(2020, 1, 1)], "close": [1.0]})
    with pytest.raises(TypeError, match="columns"):
        Indicator(name="X", symbol="X", values=bad, frequency="daily")


def test_indicator_rejects_non_date_column() -> None:
    bad = pl.DataFrame({"date": ["2020-01-01"], "value": [1.0]})
    with pytest.raises(TypeError, match="pl.Date"):
        Indicator(name="X", symbol="X", values=bad, frequency="daily")


def test_indicator_rejects_non_float_values() -> None:
    bad = pl.DataFrame({"date": [date(2020, 1, 1)], "value": [1]})
    with pytest.raises(TypeError, match="float"):
        Indicator(name="X", symbol="X", values=bad, frequency="daily")


def test_get_values_filters_range() -> None:
    ind = Indicator(name="X", symbol="X", values=_make_values(20), frequency="daily")
    filtered = ind.get_values(start=date(2020, 1, 5), end=date(2020, 1, 10))
    assert filtered["date"].min() >= date(2020, 1, 5)
    assert filtered["date"].max() <= date(2020, 1, 10)


def test_from_indicator_def() -> None:
    defn = IndicatorDef(
        name="10Y Yield", symbol="DGS10", vendor="fred",
        frequency=Frequency.DAILY, unit="percent",
        asset_class="Macro", sub_asset_class="Risk-Free Rate", geography="US",
    )
    ind = Indicator.from_indicator_def(defn, _make_values())
    assert ind.name == "10Y Yield"
    assert ind.symbol == "DGS10"
    assert ind.unit == "percent"
    assert ind.frequency is Frequency.DAILY
