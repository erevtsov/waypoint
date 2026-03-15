"""Tests for the AssetDef dataclass."""

import pytest

from waypoint.asset_def import AssetDef


def test_asset_def_construction() -> None:
    ad = AssetDef(
        name="US Large Cap",
        symbol="SPY",
        vendor="yfinance",
        frequency="daily",
        asset_class="Equities",
        sub_asset_class="Large Cap",
        geography="US",
    )
    assert ad.name == "US Large Cap"
    assert ad.symbol == "SPY"
    assert ad.vendor == "yfinance"
    assert ad.frequency == "daily"
    assert ad.asset_class == "Equities"
    assert ad.sub_asset_class == "Large Cap"
    assert ad.geography == "US"


def test_asset_def_metadata_defaults_to_empty_string() -> None:
    ad = AssetDef(name="X", symbol="X", vendor="fred", frequency="monthly")
    assert ad.asset_class == ""
    assert ad.sub_asset_class == ""
    assert ad.geography == ""


def test_asset_def_invalid_vendor() -> None:
    with pytest.raises(ValueError, match="vendor must be one of"):
        AssetDef(name="X", symbol="X", vendor="bloomberg", frequency="daily")


def test_asset_def_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="frequency must be one of"):
        AssetDef(name="X", symbol="X", vendor="yfinance", frequency="hourly")


def test_asset_def_is_frozen() -> None:
    ad = AssetDef(name="X", symbol="X", vendor="yfinance", frequency="daily")
    with pytest.raises(Exception):
        ad.name = "Y"  # type: ignore[misc]
