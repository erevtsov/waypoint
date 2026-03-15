"""Cash flow definitions for wealth simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

# Valid cashflow mode strings
CASHFLOW_MODE_DOLLAR = "dollar"
CASHFLOW_MODE_PCT_PORTFOLIO = "pct_portfolio"
CASHFLOW_MODE_PCT_PORTFOLIO_INFLATION_ADJUSTED = "pct_portfolio_inflation_adjusted"

VALID_CASHFLOW_MODES: frozenset[str] = frozenset({
    CASHFLOW_MODE_DOLLAR,
    CASHFLOW_MODE_PCT_PORTFOLIO,
    CASHFLOW_MODE_PCT_PORTFOLIO_INFLATION_ADJUSTED,
})

VALID_FREQUENCIES: frozenset[str] = frozenset({"monthly", "annual"})

# Periods per year for cashflow frequencies
CASHFLOW_PERIODS: dict[str, int] = {"monthly": 12, "annual": 1}


@dataclass(frozen=True)
class PeriodicCashflow:
    """A recurring cash flow applied at fixed intervals.

    Positive ``amount`` means a contribution (inflow); negative means a
    withdrawal (outflow).

    Parameters
    ----------
    amount:
        Base cash flow amount.
    frequency:
        ``"monthly"`` or ``"annual"``.
    mode:
        ``"dollar"``, ``"pct_portfolio"``, or
        ``"pct_portfolio_inflation_adjusted"``.
    inflation_rate:
        Annual inflation rate (e.g. 0.03 for 3%).  Only used when
        ``mode == "dollar"`` — dollar amounts grow by this rate each year.
    """

    amount: float
    frequency: str
    mode: str = field(default=CASHFLOW_MODE_DOLLAR)
    inflation_rate: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(VALID_FREQUENCIES)}, "
                f"got {self.frequency!r}"
            )
        if self.mode not in VALID_CASHFLOW_MODES:
            raise ValueError(
                f"mode must be one of {sorted(VALID_CASHFLOW_MODES)}, "
                f"got {self.mode!r}"
            )

    def amount_at(
        self,
        period: int,
        periods_per_year: int,
        portfolio_value: float,
        cumulative_inflation: float,
    ) -> float:
        """Return the cash flow amount at the given simulation period.

        Returns 0.0 when the cash flow does not fall on this period.

        Parameters
        ----------
        period:
            Current simulation period (0-indexed).
        periods_per_year:
            Number of simulation periods per year.
        portfolio_value:
            Current portfolio value before the cash flow is applied.
        cumulative_inflation:
            Compound inflation factor since inception (1.0 = no inflation).
        """
        cashflow_every = periods_per_year // CASHFLOW_PERIODS[self.frequency]
        # period 0 is the starting period — cashflows apply from period 1 onward
        if period == 0:
            return 0.0
        if period % cashflow_every != 0:
            return 0.0

        if self.mode == CASHFLOW_MODE_DOLLAR:
            return self.amount * cumulative_inflation
        elif self.mode == CASHFLOW_MODE_PCT_PORTFOLIO:
            return self.amount * portfolio_value
        elif self.mode == CASHFLOW_MODE_PCT_PORTFOLIO_INFLATION_ADJUSTED:
            return self.amount * portfolio_value
        return 0.0


@dataclass(frozen=True)
class LumpSum:
    """A one-time cash flow at a specific point in time.

    Positive ``amount`` means a contribution; negative means a withdrawal.

    Parameters
    ----------
    amount:
        Cash flow amount.
    at_year:
        Year at which the cash flow occurs (e.g. 5.0 = end of year 5).
    """

    amount: float
    at_year: float

    def amount_at(
        self,
        period: int,
        periods_per_year: int,
        portfolio_value: float,
        cumulative_inflation: float,
    ) -> float:
        """Return the cash flow amount at the given simulation period.

        Returns 0.0 for all periods except the one corresponding to ``at_year``.

        Parameters
        ----------
        period:
            Current simulation period (0-indexed).
        periods_per_year:
            Number of simulation periods per year.
        portfolio_value:
            Current portfolio value (unused for LumpSum).
        cumulative_inflation:
            Cumulative inflation factor (unused for LumpSum).
        """
        target_period = round(self.at_year * periods_per_year)
        if period == target_period:
            return self.amount
        return 0.0


# Union type for cashflow definitions
CashflowDefinition = PeriodicCashflow | LumpSum
