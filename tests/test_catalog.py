"""Tests for the built-in asset definition catalog."""

from waypoint import catalog
from waypoint.asset_def import AssetDef


def test_all_catalog_entries_are_asset_defs() -> None:
    entries = [
        catalog.US_LARGE_CAP,
        catalog.US_SMALL_CAP,
        catalog.INTL_DEVELOPED,
        catalog.EMERGING,
        catalog.US_AGG_BONDS,
        catalog.US_TIPS,
        catalog.REAL_RATE_10Y,
        catalog.CPI_YOY,
    ]
    for entry in entries:
        assert isinstance(entry, AssetDef)


def test_catalog_entries_have_non_empty_metadata() -> None:
    entries = [
        catalog.US_LARGE_CAP,
        catalog.US_SMALL_CAP,
        catalog.INTL_DEVELOPED,
        catalog.EMERGING,
        catalog.US_AGG_BONDS,
        catalog.US_TIPS,
        catalog.REAL_RATE_10Y,
        catalog.CPI_YOY,
    ]
    for entry in entries:
        assert entry.asset_class != "", f"{entry.name} missing asset_class"
        assert entry.sub_asset_class != "", f"{entry.name} missing sub_asset_class"
        assert entry.geography != "", f"{entry.name} missing geography"


def test_cpi_is_monthly() -> None:
    assert catalog.CPI_YOY.frequency == "monthly"


def test_equity_asset_defs_are_daily() -> None:
    for ad in [catalog.US_LARGE_CAP, catalog.US_SMALL_CAP, catalog.INTL_DEVELOPED]:
        assert ad.frequency == "daily"
