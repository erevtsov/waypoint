"""Social Security benefit estimation utilities.

These helpers implement the SSA Primary Insurance Amount (PIA) formula and
claiming-age adjustments so that Social Security income can be modelled as a
:class:`~waypoint.cashflows.PeriodicCashflow` inside a ``WealthSimulation``.

Typical usage::

    import waypoint as wp

    # Monthly benefit for someone born 1962 with $5,000 AIME claiming at 67
    benefit = wp.social_security.monthly_benefit(
        aime=5_000.0, birth_year=1962, claim_age=67.0
    )

    # Turn it into a simulation cashflow (real = COLA-adjusted)
    cf = wp.social_security.as_cashflow(
        aime=5_000.0, birth_year=1962, claim_age=67.0
    )
"""

from __future__ import annotations

from waypoint.cashflows import PeriodicCashflow
from waypoint.enums import CashflowMode, Frequency

# ---------------------------------------------------------------------------
# SSA formula constants
# ---------------------------------------------------------------------------

#: Default SSA bend points for 2024 (dollars / month).
BEND_POINTS_2024: tuple[float, float] = (1_174.0, 7_078.0)

#: PIA formula replacement rates for each tier.
_PIA_RATES: tuple[float, float, float] = (0.90, 0.32, 0.15)

#: Early-claiming monthly reduction for the first 36 months before FRA.
_EARLY_RATE_FIRST_36: float = 5 / 9 / 100  # 5/9 of 1 % per month

#: Early-claiming monthly reduction beyond 36 months before FRA.
_EARLY_RATE_BEYOND_36: float = 5 / 12 / 100  # 5/12 of 1 % per month

#: Delayed-claiming monthly credit after FRA (8 % / year = 2/3 % / month).
_DELAYED_RATE: float = 2 / 3 / 100

#: Earliest allowed claiming age.
MIN_CLAIM_AGE: float = 62.0

#: Latest age after which delayed credits stop accumulating.
MAX_CLAIM_AGE: float = 70.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def full_retirement_age(birth_year: int) -> float:
    """Return the Full Retirement Age (FRA) in years for a given birth year.

    Based on the SSA schedule:

    * ≤ 1937: 65
    * 1938–1942: 65 + (birth_year − 1937) × 2 months
    * 1943–1954: 66
    * 1955–1959: 66 + (birth_year − 1954) × 2 months
    * ≥ 1960: 67

    Parameters
    ----------
    birth_year:
        Four-digit birth year.

    Returns
    -------
    float
        FRA expressed in decimal years (e.g. 66.5 = 66 years 6 months).
    """
    if birth_year <= 1937:
        return 65.0
    if birth_year <= 1942:
        return 65.0 + (birth_year - 1937) * 2 / 12
    if birth_year <= 1954:
        return 66.0
    if birth_year <= 1959:
        return 66.0 + (birth_year - 1954) * 2 / 12
    return 67.0


def primary_insurance_amount(
    aime: float,
    bend_points: tuple[float, float] = BEND_POINTS_2024,
) -> float:
    """Compute the Primary Insurance Amount (PIA) from AIME.

    The PIA is the monthly benefit payable at Full Retirement Age, calculated
    using the SSA three-tier bend-point formula::

        PIA = 90% × min(AIME, bp1)
            + 32% × max(0, min(AIME, bp2) − bp1)
            + 15% × max(0, AIME − bp2)

    Parameters
    ----------
    aime:
        Average Indexed Monthly Earnings (dollars / month).  This is the
        average of the highest 35 years of inflation-indexed annual earnings
        divided by 12.
    bend_points:
        Two-element tuple ``(bp1, bp2)`` of SSA bend points in dollars per
        month.  Defaults to :data:`BEND_POINTS_2024`.

    Returns
    -------
    float
        Estimated monthly PIA in dollars.
    """
    bp1, bp2 = bend_points
    tier1 = min(aime, bp1)
    tier2 = max(0.0, min(aime, bp2) - bp1)
    tier3 = max(0.0, aime - bp2)
    return _PIA_RATES[0] * tier1 + _PIA_RATES[1] * tier2 + _PIA_RATES[2] * tier3


def monthly_benefit(
    aime: float,
    birth_year: int,
    claim_age: float,
    bend_points: tuple[float, float] = BEND_POINTS_2024,
) -> float:
    """Estimate the monthly Social Security retirement benefit.

    Computes the PIA from ``aime`` and adjusts it for early or delayed
    claiming relative to the claimant's Full Retirement Age.

    Parameters
    ----------
    aime:
        Average Indexed Monthly Earnings (dollars / month).
    birth_year:
        Claimant's four-digit birth year.
    claim_age:
        Age (in decimal years) at which benefits begin.
        Must be in [``MIN_CLAIM_AGE``, ``MAX_CLAIM_AGE``] = [62, 70].
    bend_points:
        SSA bend points.  Defaults to :data:`BEND_POINTS_2024`.

    Returns
    -------
    float
        Estimated monthly benefit in nominal dollars, rounded to the
        nearest cent.

    Raises
    ------
    ValueError
        If ``claim_age`` is outside [62, 70].
    """
    if not (MIN_CLAIM_AGE <= claim_age <= MAX_CLAIM_AGE):
        raise ValueError(
            f"claim_age must be between {MIN_CLAIM_AGE} and {MAX_CLAIM_AGE}, "
            f"got {claim_age}"
        )

    fra = full_retirement_age(birth_year)
    pia = primary_insurance_amount(aime, bend_points)

    fra_months = round(fra * 12)
    claim_months = round(claim_age * 12)
    months_from_fra = claim_months - fra_months

    factor = _claiming_adjustment(months_from_fra)
    return round(pia * factor, 2)


def as_cashflow(
    aime: float,
    birth_year: int,
    claim_age: float,
    bend_points: tuple[float, float] = BEND_POINTS_2024,
    real: bool = True,
    effective_tax_rate: float = 0.0,
    start_year: float | None = None,
    end_year: float | None = None,
) -> PeriodicCashflow:
    """Create a monthly :class:`~waypoint.cashflows.PeriodicCashflow` for SS income.

    A convenience wrapper that estimates the monthly benefit and packages it
    as a cashflow ready for use in :class:`~waypoint.analysis.simulation.WealthSimulation`.

    Parameters
    ----------
    aime:
        Average Indexed Monthly Earnings (dollars / month).
    birth_year:
        Claimant's four-digit birth year.
    claim_age:
        Age at which benefits begin.  Must be in [62, 70].
    bend_points:
        SSA bend points.  Defaults to :data:`BEND_POINTS_2024`.
    real:
        When ``True`` (default), the benefit is treated as inflation-adjusted
        (mirroring Social Security's annual COLA).  Pass ``False`` to keep the
        benefit fixed in nominal dollars.
    effective_tax_rate:
        Marginal rate applied to SS benefits at withdrawal time (up to 85 % of
        benefits may be taxable above certain income thresholds).  Defaults to
        ``0.0`` (no tax adjustment).
    start_year:
        Simulation year (0-indexed) at which SS income begins.  Typically set
        to the number of years from simulation start until claiming age.
        ``None`` (default) means active from the start.
    end_year:
        Simulation year after which SS income stops (e.g. end of life horizon).
        ``None`` (default) means active indefinitely.

    Returns
    -------
    PeriodicCashflow
        Monthly inflow representing the estimated Social Security benefit.
    """
    amount = monthly_benefit(aime, birth_year, claim_age, bend_points)
    return PeriodicCashflow(
        amount=amount,
        frequency=Frequency.MONTHLY,
        mode=CashflowMode.DOLLAR,
        real=real,
        effective_tax_rate=effective_tax_rate,
        start_year=start_year,
        end_year=end_year,
    )


# ---------------------------------------------------------------------------
# AIME estimators
# ---------------------------------------------------------------------------

#: Number of months in the 35-year AIME averaging window.
_AIME_MONTHS: int = 35 * 12  # 420


def estimate_aime(
    annual_salary: float,
    career_years: int = 35,
) -> float:
    """Estimate AIME assuming flat nominal career earnings.

    The SSA computes AIME as the average of the highest 35 years of
    *wage-indexed* monthly earnings (dividing by 420 months).  Years
    below 35 count as zero.  This function omits wage indexing and is
    therefore an approximation suitable for financial planning estimates.

    Parameters
    ----------
    annual_salary:
        Representative annual earnings (current dollars).
    career_years:
        Number of years worked with earnings at ``annual_salary``.
        Must be in [1, 35]; values above 35 are clamped to 35 (additional
        years do not raise AIME once the top-35 window is full).

    Returns
    -------
    float
        Estimated AIME in dollars per month.

    Examples
    --------
    A worker earning $80 000 / year for a full 35-year career::

        >>> estimate_aime(80_000)
        1904.76...

    The same worker with only 20 years of earnings::

        >>> estimate_aime(80_000, career_years=20)
        1088.43...
    """
    if career_years < 1:
        raise ValueError(f"career_years must be at least 1, got {career_years}")

    years_counted = min(career_years, 35)
    total_monthly_earnings = annual_salary / 12 * years_counted
    return round(total_monthly_earnings / 35, 2)


def estimate_aime_from_history(annual_earnings: list[float]) -> float:
    """Estimate AIME from a list of annual earnings (no wage indexing).

    Takes up to 35 highest-earning years from ``annual_earnings``, sums
    the monthly equivalents, and divides by 420.  Wage indexing is omitted;
    this is an approximation suitable for planning purposes.

    Parameters
    ----------
    annual_earnings:
        List of annual earnings figures in any order (nominal dollars).
        May contain any number of years; only the top 35 are used.

    Returns
    -------
    float
        Estimated AIME in dollars per month.

    Raises
    ------
    ValueError
        If ``annual_earnings`` is empty.
    """
    if not annual_earnings:
        raise ValueError("annual_earnings must contain at least one value.")

    top_35 = sorted(annual_earnings, reverse=True)[:35]
    total_monthly = sum(e / 12 for e in top_35)
    return round(total_monthly / 35, 2)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _claiming_adjustment(months_from_fra: int) -> float:
    """Return the benefit factor for claiming ``months_from_fra`` months from FRA.

    Positive values mean delayed claiming (credit); negative mean early
    claiming (reduction).

    Parameters
    ----------
    months_from_fra:
        Signed integer: positive = delayed, negative = early.
    """
    if months_from_fra >= 0:
        return 1.0 + months_from_fra * _DELAYED_RATE

    months_early = -months_from_fra
    reduction = min(months_early, 36) * _EARLY_RATE_FIRST_36
    if months_early > 36:
        reduction += (months_early - 36) * _EARLY_RATE_BEYOND_36
    return 1.0 - reduction
