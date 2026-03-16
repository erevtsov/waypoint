"""Tests for Frequency and CashflowMode enums."""

import pytest

from waypoint.enums import PERIODS_PER_YEAR, CashflowMode, Frequency

# ---------------------------------------------------------------------------
# Frequency
# ---------------------------------------------------------------------------

def test_frequency_str_equality() -> None:
    assert Frequency.DAILY == "daily"
    assert Frequency.WEEKLY == "weekly"
    assert Frequency.MONTHLY == "monthly"
    assert Frequency.QUARTERLY == "quarterly"
    assert Frequency.ANNUAL == "annual"


def test_frequency_from_string() -> None:
    assert Frequency("daily") is Frequency.DAILY
    assert Frequency("monthly") is Frequency.MONTHLY
    assert Frequency("quarterly") is Frequency.QUARTERLY


def test_frequency_invalid_raises() -> None:
    with pytest.raises(ValueError):
        Frequency("hourly")


def test_periods_per_year_values() -> None:
    assert PERIODS_PER_YEAR[Frequency.DAILY] == 252
    assert PERIODS_PER_YEAR[Frequency.WEEKLY] == 52
    assert PERIODS_PER_YEAR[Frequency.MONTHLY] == 12
    assert PERIODS_PER_YEAR[Frequency.QUARTERLY] == 4
    assert PERIODS_PER_YEAR[Frequency.ANNUAL] == 1


def test_periods_per_year_str_lookup() -> None:
    """StrEnum keys support plain-string lookups."""
    assert PERIODS_PER_YEAR["daily"] == 252
    assert PERIODS_PER_YEAR["monthly"] == 12


# ---------------------------------------------------------------------------
# CashflowMode
# ---------------------------------------------------------------------------

def test_cashflow_mode_str_equality() -> None:
    assert CashflowMode.DOLLAR == "dollar"
    assert CashflowMode.PCT_PORTFOLIO == "pct_portfolio"


def test_cashflow_mode_from_string() -> None:
    assert CashflowMode("dollar") is CashflowMode.DOLLAR
    assert CashflowMode("pct_portfolio") is CashflowMode.PCT_PORTFOLIO


def test_cashflow_mode_invalid_raises() -> None:
    with pytest.raises(ValueError):
        CashflowMode("absolute")
