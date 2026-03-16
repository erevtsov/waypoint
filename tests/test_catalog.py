"""Tests for the built-in asset definition catalog."""

from waypoint import catalog
from waypoint.asset_def import AssetDef
from waypoint.enums import Frequency
from waypoint.indicator_def import IndicatorDef


def test_asset_catalog_entries_are_asset_defs() -> None:
    asset_entries = [
        catalog.US_LARGE_CAP,
        catalog.US_SMALL_CAP,
        catalog.INTL_DEVELOPED,
        catalog.EMERGING,
        catalog.US_AGG_BONDS,
        catalog.US_TIPS,
        catalog.CPI_YOY,
    ]
    for entry in asset_entries:
        assert isinstance(entry, AssetDef)


def test_indicator_catalog_entries_are_indicator_defs() -> None:
    indicator_entries = [
        catalog.US_10Y_YIELD,
        catalog.REAL_RATE_10Y,
    ]
    for entry in indicator_entries:
        assert isinstance(entry, IndicatorDef)


def test_catalog_entries_have_non_empty_metadata() -> None:
    all_entries = [
        catalog.US_LARGE_CAP,
        catalog.US_SMALL_CAP,
        catalog.INTL_DEVELOPED,
        catalog.EMERGING,
        catalog.US_AGG_BONDS,
        catalog.US_TIPS,
        catalog.CPI_YOY,
        catalog.US_10Y_YIELD,
        catalog.REAL_RATE_10Y,
    ]
    for entry in all_entries:
        assert entry.asset_class != "", f"{entry.name} missing asset_class"
        assert entry.sub_asset_class != "", f"{entry.name} missing sub_asset_class"
        assert entry.geography != "", f"{entry.name} missing geography"


def test_cpi_is_monthly() -> None:
    assert catalog.CPI_YOY.frequency == Frequency.MONTHLY
    assert catalog.CPI_YOY.frequency == "monthly"  # StrEnum equality with str


def test_equity_asset_defs_are_daily() -> None:
    for ad in [catalog.US_LARGE_CAP, catalog.US_SMALL_CAP, catalog.INTL_DEVELOPED]:
        assert ad.frequency == Frequency.DAILY
        assert ad.frequency == "daily"  # StrEnum equality with str


def test_indicator_defs_have_units() -> None:
    assert catalog.US_10Y_YIELD.unit == "percent"
    assert catalog.REAL_RATE_10Y.unit == "percent"


def test_hpi_entries_are_asset_defs() -> None:
    assert isinstance(catalog.MA_HPI, AssetDef)
    assert isinstance(catalog.BOSTON_HPI, AssetDef)


def test_hpi_entries_are_quarterly() -> None:
    assert catalog.MA_HPI.frequency == Frequency.QUARTERLY
    assert catalog.BOSTON_HPI.frequency == Frequency.QUARTERLY
