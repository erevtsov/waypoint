"""Tests for the Asset dataclass."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.assets import PERIODS_PER_YEAR, Asset
from waypoint.instruments import Instrument


def _make_returns(n: int = 100) -> pl.DataFrame:
    rng = np.random.default_rng(seed=42)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = rng.normal(0.0003, 0.01, n).tolist()
    return pl.DataFrame({"date": dates, "returns": values})


def test_asset_construction() -> None:
    asset = Asset(
        name="US Large Cap",
        ticker="SPY",
        returns=_make_returns(),
        frequency="daily",
        asset_class="Equities",
    )
    assert asset.name == "US Large Cap"
    assert asset.ticker == "SPY"
    assert asset.frequency == "daily"
    assert asset.asset_class == "Equities"


def test_asset_metadata_defaults_to_empty_string() -> None:
    asset = Asset(name="X", ticker="X", returns=_make_returns(), frequency="daily")
    assert asset.sub_asset_class == ""
    assert asset.geography == ""


def test_asset_returns_schema() -> None:
    asset = Asset(name="X", ticker="X", returns=_make_returns(), frequency="daily")
    assert asset.returns.columns == ["date", "returns"]
    assert asset.returns["date"].dtype == pl.Date
    assert asset.returns["returns"].dtype in (pl.Float32, pl.Float64)


def test_asset_rejects_wrong_columns() -> None:
    bad = pl.DataFrame({"date": [date(2020, 1, 1)], "close": [1.0]})
    with pytest.raises(TypeError, match="columns"):
        Asset(name="X", ticker="X", returns=bad, frequency="daily")


def test_asset_rejects_non_date_index() -> None:
    bad = pl.DataFrame({"date": ["2020-01-01"], "returns": [0.01]})
    with pytest.raises(TypeError, match="pl.Date"):
        Asset(name="X", ticker="X", returns=bad, frequency="daily")


def test_asset_rejects_non_float_returns() -> None:
    bad = pl.DataFrame({"date": [date(2020, 1, 1)], "returns": [1]})  # Int64
    with pytest.raises(TypeError, match="float column"):
        Asset(name="X", ticker="X", returns=bad, frequency="daily")


def test_asset_rejects_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="frequency must be one of"):
        Asset(name="X", ticker="X", returns=_make_returns(), frequency="hourly")


def test_periods_per_year_property() -> None:
    for freq, expected in PERIODS_PER_YEAR.items():
        asset = Asset(name="X", ticker="X", returns=_make_returns(), frequency=freq)
        assert asset.periods_per_year == expected


def test_from_instrument() -> None:
    inst = Instrument(
        name="Test",
        symbol="TST",
        vendor="yfinance",
        frequency="daily",
        asset_class="Equities",
        sub_asset_class="Large Cap",
        geography="US",
    )
    returns = _make_returns()
    asset = Asset.from_instrument(inst, returns)

    assert asset.name == inst.name
    assert asset.ticker == inst.symbol
    assert asset.frequency == inst.frequency
    assert asset.asset_class == inst.asset_class
    assert asset.sub_asset_class == inst.sub_asset_class
    assert asset.geography == inst.geography


def test_cross_asset_join_on_date() -> None:
    """Assets from different sources can be aligned by joining on 'date'."""
    dates_a = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]
    dates_b = [date(2020, 1, 3), date(2020, 1, 4), date(2020, 1, 5)]

    asset_a = Asset(
        name="A", ticker="A",
        returns=pl.DataFrame({"date": dates_a, "returns": [0.01, 0.02, -0.01]}),
        frequency="daily",
    )
    asset_b = Asset(
        name="B", ticker="B",
        returns=pl.DataFrame({"date": dates_b, "returns": [0.005, -0.003, 0.007]}),
        frequency="daily",
    )

    combined = asset_a.returns.join(
        asset_b.returns.rename({"returns": "returns_b"}),
        on="date",
        how="inner",
    )

    assert combined.columns == ["date", "returns", "returns_b"]
    assert len(combined) == 2  # only Jan 3 and Jan 4 overlap
    assert combined["date"].to_list() == [date(2020, 1, 3), date(2020, 1, 4)]
