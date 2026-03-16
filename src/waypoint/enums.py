"""Shared enumerations for Waypoint.

All enums subclass ``StrEnum`` so that members compare equal to their
lowercase string counterparts.  Callers may pass either the enum member
or the equivalent lowercase string; normalisation is performed in each
dataclass ``__post_init__``.
"""

from __future__ import annotations

from enum import StrEnum


class Frequency(StrEnum):
    """Observation / simulation frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class CashflowMode(StrEnum):
    """Determines how a periodic cash flow amount is calculated."""

    DOLLAR = "dollar"
    PCT_PORTFOLIO = "pct_portfolio"


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

#: Number of periods per calendar year for each frequency.
PERIODS_PER_YEAR: dict[Frequency, int] = {
    Frequency.DAILY: 252,
    Frequency.WEEKLY: 52,
    Frequency.MONTHLY: 12,
    Frequency.QUARTERLY: 4,
    Frequency.ANNUAL: 1,
}

#: Frequencies that are valid for ``AssetDef`` / ``Asset`` (no ANNUAL).
ASSET_FREQUENCIES: frozenset[Frequency] = frozenset(
    {Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY, Frequency.QUARTERLY}
)

#: Frequencies that are valid for ``PeriodicCashflow`` (no WEEKLY / DAILY).
CASHFLOW_FREQUENCIES: frozenset[Frequency] = frozenset(
    {Frequency.MONTHLY, Frequency.ANNUAL}
)
