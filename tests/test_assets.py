"""Tests for the Asset dataclass."""

import numpy as np
import polars as pl
import pytest

from waypoint.assets import PERIODS_PER_YEAR, Asset
from waypoint.instruments import Instrument


def _make_returns(n: int = 100) -> pl.Series:
    rng = np.random.default_rng(seed=42)
    return pl.Series("returns", rng.normal(0.0003, 0.01, n).tolist())


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


def test_asset_rejects_non_float_series() -> None:
    with pytest.raises(TypeError, match="float Series"):
        Asset(
            name="X",
            ticker="X",
            returns=pl.Series("r", [1, 2, 3]),  # Int64, not Float
            frequency="daily",
        )


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
