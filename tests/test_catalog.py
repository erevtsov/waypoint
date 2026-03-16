"""Tests for the built-in asset definition catalog."""

from waypoint import catalog
from waypoint.asset_def import AssetDef
from waypoint.enums import Frequency
from waypoint.indicator_def import IndicatorDef

# ---------------------------------------------------------------------------
# equities submodule
# ---------------------------------------------------------------------------

def test_equities_entries_are_asset_defs() -> None:
    entries = [
        catalog.equities.US_LARGE_CAP,
        catalog.equities.US_TOTAL_MARKET,
        catalog.equities.US_LARGE_CAP_GROWTH,
        catalog.equities.NASDAQ_100,
        catalog.equities.RUSSELL_1000,
        catalog.equities.US_SMALL_CAP,
        catalog.equities.INTL_DEVELOPED,
        catalog.equities.EUROPE_DEVELOPED,
        catalog.equities.EMERGING,
        catalog.equities.CHINA_TECH,
    ]
    for entry in entries:
        assert isinstance(entry, AssetDef)


def test_equities_are_daily() -> None:
    for ad in [
        catalog.equities.US_LARGE_CAP,
        catalog.equities.US_SMALL_CAP,
        catalog.equities.INTL_DEVELOPED,
    ]:
        assert ad.frequency == Frequency.DAILY
        assert ad.frequency == "daily"  # StrEnum equality with str


def test_equities_have_non_empty_metadata() -> None:
    for entry in [catalog.equities.US_LARGE_CAP, catalog.equities.EMERGING]:
        assert entry.asset_class != ""
        assert entry.sub_asset_class != ""
        assert entry.geography != ""


# ---------------------------------------------------------------------------
# fixed_income submodule
# ---------------------------------------------------------------------------

def test_fixed_income_entries_are_asset_defs() -> None:
    for entry in [
        catalog.fixed_income.US_AGG_BONDS,
        catalog.fixed_income.US_TIPS,
        catalog.fixed_income.CPI_YOY,
    ]:
        assert isinstance(entry, AssetDef)


def test_cpi_is_monthly() -> None:
    assert catalog.fixed_income.CPI_YOY.frequency == Frequency.MONTHLY
    assert catalog.fixed_income.CPI_YOY.frequency == "monthly"  # StrEnum equality with str


# ---------------------------------------------------------------------------
# real_estate submodule
# ---------------------------------------------------------------------------

def test_real_estate_entries_are_asset_defs() -> None:
    assert isinstance(catalog.real_estate.MA_HPI, AssetDef)
    assert isinstance(catalog.real_estate.BOSTON_HPI, AssetDef)


def test_real_estate_entries_are_quarterly() -> None:
    assert catalog.real_estate.MA_HPI.frequency == Frequency.QUARTERLY
    assert catalog.real_estate.BOSTON_HPI.frequency == Frequency.QUARTERLY


# ---------------------------------------------------------------------------
# indicators submodule
# ---------------------------------------------------------------------------

def test_indicator_entries_are_indicator_defs() -> None:
    for entry in [catalog.indicators.US_10Y_YIELD, catalog.indicators.REAL_RATE_10Y]:
        assert isinstance(entry, IndicatorDef)


def test_indicator_defs_have_units() -> None:
    assert catalog.indicators.US_10Y_YIELD.unit == "percent"
    assert catalog.indicators.REAL_RATE_10Y.unit == "percent"


def test_indicators_have_non_empty_metadata() -> None:
    for entry in [catalog.indicators.US_10Y_YIELD, catalog.indicators.REAL_RATE_10Y]:
        assert entry.asset_class != ""
        assert entry.sub_asset_class != ""
        assert entry.geography != ""
