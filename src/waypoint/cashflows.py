"""Cash flow definitions for wealth simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from waypoint.enums import CASHFLOW_FREQUENCIES, CashflowMode, Frequency

#: Number of cashflow events per year, keyed by frequency.
CASHFLOW_PERIODS: dict[Frequency, int] = {
    Frequency.MONTHLY: 12,
    Frequency.ANNUAL: 1,
}


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
        ``"monthly"`` or ``"annual"``.  Accepts a ``Frequency`` member or
        its lowercase string equivalent.
    mode:
        ``"dollar"``, ``"pct_portfolio"``, or
        ``"pct_portfolio_inflation_adjusted"``.  Accepts a ``CashflowMode``
        member or its lowercase string equivalent.
    inflation_rate:
        Annual inflation rate (e.g. 0.03 for 3%).  Only used when
        ``mode == CashflowMode.DOLLAR`` — dollar amounts grow by this rate
        each year.
    slots:
        Portfolio slot names that receive this cash flow.  ``None`` (default)
        means all slots receive the cash flow proportionally.  Use this to
        restrict contributions or withdrawals to liquid assets only — e.g.
        ``slots=("Equities", "Bonds")`` keeps the cash flow out of a real
        estate slot that cannot be incrementally purchased.
    """

    amount: float
    frequency: Frequency = field(default=Frequency.MONTHLY)
    mode: CashflowMode = field(default=CashflowMode.DOLLAR)
    inflation_rate: float = field(default=0.0)
    slots: tuple[str, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency", Frequency(self.frequency))
        object.__setattr__(self, "mode", CashflowMode(self.mode))
        if isinstance(self.slots, list):
            object.__setattr__(self, "slots", tuple(self.slots))
        if self.frequency not in CASHFLOW_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(CASHFLOW_FREQUENCIES)}, "
                f"got {self.frequency!r}"
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
        cashflow_periods = CASHFLOW_PERIODS[self.frequency]
        # period 0 is the starting period — cashflows apply from period 1 onward
        if period == 0:
            return 0.0
        if cashflow_periods >= periods_per_year:
            # Cashflow fires more often than simulation steps: bundle multiple
            # payments into each period (e.g. 3 monthly payments per quarter).
            count = cashflow_periods // periods_per_year
        else:
            # Cashflow fires less often: only apply on the matching boundary.
            cashflow_every = periods_per_year // cashflow_periods
            if period % cashflow_every != 0:
                return 0.0
            count = 1

        if self.mode == CashflowMode.DOLLAR:
            return self.amount * count * cumulative_inflation
        elif self.mode == CashflowMode.PCT_PORTFOLIO:
            return self.amount * count * portfolio_value
        elif self.mode == CashflowMode.PCT_PORTFOLIO_INFLATION_ADJUSTED:
            return self.amount * count * portfolio_value
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
    slots:
        Portfolio slot names that receive this cash flow.  ``None`` (default)
        means all slots receive the cash flow proportionally.
    """

    amount: float
    at_year: float
    slots: tuple[str, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        if isinstance(self.slots, list):
            object.__setattr__(self, "slots", tuple(self.slots))

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
