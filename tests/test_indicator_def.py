"""Tests for IndicatorDef."""

import pytest

from waypoint.enums import Frequency
from waypoint.indicator_def import IndicatorDef


def test_indicator_def_construction() -> None:
    ind = IndicatorDef(name="10Y Yield", symbol="DGS10", vendor="fred", frequency="daily")
    assert ind.name == "10Y Yield"
    assert ind.symbol == "DGS10"
    assert ind.vendor == "fred"


def test_indicator_def_frequency_normalised_to_enum() -> None:
    ind = IndicatorDef(name="X", symbol="X", vendor="fred", frequency="daily")
    assert ind.frequency is Frequency.DAILY


def test_indicator_def_accepts_frequency_enum() -> None:
    ind = IndicatorDef(name="X", symbol="X", vendor="fred", frequency=Frequency.MONTHLY)
    assert ind.frequency is Frequency.MONTHLY


def test_indicator_def_invalid_vendor_raises() -> None:
    with pytest.raises(ValueError, match="vendor"):
        IndicatorDef(name="X", symbol="X", vendor="bloomberg", frequency="daily")


def test_indicator_def_invalid_frequency_raises() -> None:
    with pytest.raises(ValueError):
        IndicatorDef(name="X", symbol="X", vendor="fred", frequency="annual")


def test_indicator_def_is_frozen() -> None:
    ind = IndicatorDef(name="X", symbol="X", vendor="fred", frequency="daily")
    with pytest.raises((AttributeError, TypeError)):
        ind.name = "Y"  # type: ignore[misc]


def test_indicator_def_metadata_defaults_to_empty_string() -> None:
    ind = IndicatorDef(name="X", symbol="X", vendor="fred", frequency="daily")
    assert ind.unit == ""
    assert ind.asset_class == ""
    assert ind.sub_asset_class == ""
    assert ind.geography == ""
