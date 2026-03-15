"""Tests for the Instrument dataclass."""

import pytest

from waypoint.instruments import Instrument


def test_instrument_construction() -> None:
    inst = Instrument(
        name="US Large Cap",
        symbol="SPY",
        vendor="yfinance",
        frequency="daily",
        asset_class="Equities",
        sub_asset_class="Large Cap",
        geography="US",
    )
    assert inst.name == "US Large Cap"
    assert inst.symbol == "SPY"
    assert inst.vendor == "yfinance"
    assert inst.frequency == "daily"
    assert inst.asset_class == "Equities"
    assert inst.sub_asset_class == "Large Cap"
    assert inst.geography == "US"


def test_instrument_metadata_defaults_to_empty_string() -> None:
    inst = Instrument(name="X", symbol="X", vendor="fred", frequency="monthly")
    assert inst.asset_class == ""
    assert inst.sub_asset_class == ""
    assert inst.geography == ""


def test_instrument_invalid_vendor() -> None:
    with pytest.raises(ValueError, match="vendor must be one of"):
        Instrument(name="X", symbol="X", vendor="bloomberg", frequency="daily")


def test_instrument_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="frequency must be one of"):
        Instrument(name="X", symbol="X", vendor="yfinance", frequency="hourly")


def test_instrument_is_frozen() -> None:
    inst = Instrument(name="X", symbol="X", vendor="yfinance", frequency="daily")
    with pytest.raises(Exception):
        inst.name = "Y"  # type: ignore[misc]
