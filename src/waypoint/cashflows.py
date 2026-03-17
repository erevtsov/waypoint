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
        Base cash flow amount.  For withdrawals (``amount < 0``) with
        ``effective_tax_rate > 0``, this is the **net** (after-tax) amount
        the beneficiary receives; the portfolio is drawn down by
        ``amount / (1 - effective_tax_rate)``.
    frequency:
        ``"monthly"`` or ``"annual"``.  Accepts a ``Frequency`` member or
        its lowercase string equivalent.
    mode:
        ``"dollar"`` or ``"pct_portfolio"``.  Accepts a ``CashflowMode``
        member or its lowercase string equivalent.
    real:
        When ``True``, ``amount`` is expressed in today's dollars and is
        scaled by the simulation's cumulative inflation factor at each
        period.  Only applies to ``mode="dollar"``.  Defaults to ``False``
        (nominal — amount is in future dollars as stated).
    effective_tax_rate:
        Marginal tax rate applied to withdrawals (``amount < 0``).  The
        portfolio is drawn down by ``|amount| / (1 - rate)`` so that the
        net receipt equals ``|amount|``.  Has no effect on contributions.
        Defaults to ``0.0`` (no tax adjustment).
    slots:
        Portfolio slot names that receive this cash flow.  ``None`` (default)
        means all slots receive the cash flow proportionally.  Use this to
        restrict contributions or withdrawals to liquid assets only.
    start_year:
        Simulation year (0-indexed) at which this cashflow becomes active.
        ``None`` means active from the start.  E.g. ``start_year=5.0`` means
        the cashflow does not fire before year 5.
    end_year:
        Simulation year (0-indexed) after which this cashflow stops firing.
        ``None`` means active indefinitely.  E.g. ``end_year=30.0`` means the
        cashflow stops after year 30.
    """

    amount: float
    frequency: Frequency = field(default=Frequency.MONTHLY)
    mode: CashflowMode = field(default=CashflowMode.DOLLAR)
    real: bool = field(default=False)
    effective_tax_rate: float = field(default=0.0)
    slots: tuple[str, ...] | None = field(default=None)
    start_year: float | None = field(default=None)
    end_year: float | None = field(default=None)

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
        """Return the portfolio impact at the given simulation period.

        For withdrawals with ``effective_tax_rate > 0``, returns the gross
        draw-down amount (larger than the net ``amount``).  Returns 0.0 when
        the cash flow does not fall on this period.

        Parameters
        ----------
        period:
            Current simulation period (0-indexed).
        periods_per_year:
            Number of simulation periods per year.
        portfolio_value:
            Current portfolio value before the cash flow is applied.
        cumulative_inflation:
            Compound inflation factor since inception (1.0 = no inflation),
            supplied by ``WealthSimulation`` from its ``inflation_rate``.
        """
        cashflow_periods = CASHFLOW_PERIODS[self.frequency]
        # period 0 is the starting period — cashflows apply from period 1 onward
        if period == 0:
            return 0.0
        current_year = period / periods_per_year
        if self.start_year is not None and current_year < self.start_year:
            return 0.0
        if self.end_year is not None and current_year > self.end_year:
            return 0.0
        if cashflow_periods >= periods_per_year:
            count = cashflow_periods // periods_per_year
        else:
            cashflow_every = periods_per_year // cashflow_periods
            if period % cashflow_every != 0:
                return 0.0
            count = 1

        if self.mode == CashflowMode.DOLLAR:
            base = self.amount * count
            if self.real:
                base *= cumulative_inflation
        else:  # PCT_PORTFOLIO
            base = self.amount * count * portfolio_value

        return _apply_tax(base, self.effective_tax_rate)


@dataclass(frozen=True)
class LumpSum:
    """A one-time cash flow at a specific point in time.

    Positive ``amount`` means a contribution; negative means a withdrawal.

    Parameters
    ----------
    amount:
        Cash flow amount.  For withdrawals (``amount < 0``) with
        ``effective_tax_rate > 0``, this is the **net** (after-tax) amount;
        the portfolio is drawn down by ``amount / (1 - effective_tax_rate)``.
    at_year:
        Year at which the cash flow occurs (e.g. 5.0 = end of year 5).
    real:
        When ``True``, ``amount`` is expressed in today's dollars and is
        scaled by the simulation's cumulative inflation factor at the time
        of the event.  Defaults to ``False`` (nominal).
    effective_tax_rate:
        Marginal tax rate applied to withdrawals (``amount < 0``).  The
        portfolio is drawn down by ``|amount| / (1 - rate)``.  Defaults to
        ``0.0`` (no tax adjustment).
    slots:
        Portfolio slot names that receive this cash flow.  ``None`` (default)
        means all slots receive the cash flow proportionally.
    start_year:
        If set, the lump sum is suppressed when ``at_year < start_year``.
    end_year:
        If set, the lump sum is suppressed when ``at_year > end_year``.
    """

    amount: float
    at_year: float
    real: bool = field(default=False)
    effective_tax_rate: float = field(default=0.0)
    slots: tuple[str, ...] | None = field(default=None)
    start_year: float | None = field(default=None)
    end_year: float | None = field(default=None)

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
        """Return the portfolio impact at the given simulation period.

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
            Cumulative inflation factor supplied by ``WealthSimulation``.
            Applied when ``real=True``.
        """
        target_period = round(self.at_year * periods_per_year)
        if period != target_period:
            return 0.0
        if self.start_year is not None and self.at_year < self.start_year:
            return 0.0
        if self.end_year is not None and self.at_year > self.end_year:
            return 0.0
        base = self.amount * cumulative_inflation if self.real else self.amount
        return _apply_tax(base, self.effective_tax_rate)


def _apply_tax(amount: float, effective_tax_rate: float) -> float:
    """Gross up a withdrawal by the effective tax rate.

    Contributions (positive amounts) are returned unchanged — they are
    assumed to be already specified in post-tax terms.

    Parameters
    ----------
    amount:
        Net cash flow amount (negative = withdrawal).
    effective_tax_rate:
        Marginal rate ∈ [0, 1).  If 0.0, no adjustment is made.
    """
    if amount < 0.0 and effective_tax_rate > 0.0:
        return amount / (1.0 - effective_tax_rate)
    return amount


# Union type for cashflow definitions
CashflowDefinition = PeriodicCashflow | LumpSum
