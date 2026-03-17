"""Tests for social_security utilities."""

from __future__ import annotations

import pytest

from waypoint.cashflows import PeriodicCashflow
from waypoint.enums import CashflowMode, Frequency
from waypoint.social_security import (
    BEND_POINTS_2024,
    _claiming_adjustment,
    as_cashflow,
    estimate_aime,
    estimate_aime_from_history,
    full_retirement_age,
    monthly_benefit,
    primary_insurance_amount,
)

# ---------------------------------------------------------------------------
# full_retirement_age
# ---------------------------------------------------------------------------


def test_fra_before_1938() -> None:
    assert full_retirement_age(1930) == 65.0
    assert full_retirement_age(1937) == 65.0


def test_fra_transition_1938_to_1942() -> None:
    assert abs(full_retirement_age(1938) - (65.0 + 2 / 12)) < 1e-9
    assert abs(full_retirement_age(1940) - (65.0 + 6 / 12)) < 1e-9
    assert abs(full_retirement_age(1942) - (65.0 + 10 / 12)) < 1e-9


def test_fra_1943_to_1954() -> None:
    for year in range(1943, 1955):
        assert full_retirement_age(year) == 66.0


def test_fra_transition_1955_to_1959() -> None:
    assert abs(full_retirement_age(1955) - (66.0 + 2 / 12)) < 1e-9
    assert abs(full_retirement_age(1957) - (66.0 + 6 / 12)) < 1e-9
    assert abs(full_retirement_age(1959) - (66.0 + 10 / 12)) < 1e-9


def test_fra_1960_and_later() -> None:
    assert full_retirement_age(1960) == 67.0
    assert full_retirement_age(1990) == 67.0
    assert full_retirement_age(2000) == 67.0


# ---------------------------------------------------------------------------
# primary_insurance_amount
# ---------------------------------------------------------------------------


def test_pia_zero_aime() -> None:
    assert primary_insurance_amount(0.0) == 0.0


def test_pia_below_first_bend_point() -> None:
    """90% of AIME when AIME ≤ bp1."""
    aime = 500.0
    expected = 0.90 * aime
    assert abs(primary_insurance_amount(aime) - expected) < 1e-9


def test_pia_between_bend_points() -> None:
    """90% of bp1 + 32% of (AIME − bp1) when bp1 < AIME ≤ bp2."""
    bp1, bp2 = BEND_POINTS_2024
    aime = (bp1 + bp2) / 2
    expected = 0.90 * bp1 + 0.32 * (aime - bp1)
    assert abs(primary_insurance_amount(aime) - expected) < 1e-9


def test_pia_above_second_bend_point() -> None:
    """All three tiers apply when AIME > bp2."""
    bp1, bp2 = BEND_POINTS_2024
    aime = bp2 + 1_000.0
    expected = 0.90 * bp1 + 0.32 * (bp2 - bp1) + 0.15 * 1_000.0
    assert abs(primary_insurance_amount(aime) - expected) < 1e-9


def test_pia_custom_bend_points() -> None:
    pia = primary_insurance_amount(2_000.0, bend_points=(1_000.0, 6_000.0))
    expected = 0.90 * 1_000.0 + 0.32 * 1_000.0
    assert abs(pia - expected) < 1e-9


# ---------------------------------------------------------------------------
# _claiming_adjustment
# ---------------------------------------------------------------------------


def test_claiming_adjustment_at_fra() -> None:
    """No adjustment when claiming exactly at FRA."""
    assert abs(_claiming_adjustment(0) - 1.0) < 1e-9


def test_claiming_adjustment_early_within_36_months() -> None:
    """5/9% reduction per month for first 36 months."""
    factor = _claiming_adjustment(-12)
    expected = 1.0 - 12 * (5 / 9 / 100)
    assert abs(factor - expected) < 1e-9


def test_claiming_adjustment_early_beyond_36_months() -> None:
    """5/9% for first 36 months then 5/12% beyond."""
    months_early = 60
    expected = 1.0 - 36 * (5 / 9 / 100) - 24 * (5 / 12 / 100)
    assert abs(_claiming_adjustment(-months_early) - expected) < 1e-9


def test_claiming_adjustment_delayed() -> None:
    """2/3% credit per month after FRA."""
    factor = _claiming_adjustment(24)
    expected = 1.0 + 24 * (2 / 3 / 100)
    assert abs(factor - expected) < 1e-9


# ---------------------------------------------------------------------------
# monthly_benefit
# ---------------------------------------------------------------------------


def test_monthly_benefit_at_fra() -> None:
    """Benefit at FRA equals PIA exactly."""
    aime = 5_000.0
    birth_year = 1960  # FRA = 67
    benefit = monthly_benefit(aime, birth_year, claim_age=67.0)
    pia = primary_insurance_amount(aime)
    assert abs(benefit - round(pia, 2)) < 0.01


def test_monthly_benefit_early_claim_62_fra67() -> None:
    """Claiming at 62 with FRA=67 gives ~70% of PIA (30% reduction)."""
    aime = 5_000.0
    birth_year = 1960  # FRA = 67
    benefit = monthly_benefit(aime, birth_year, claim_age=62.0)
    pia = primary_insurance_amount(aime)
    # 36 months @ 5/9% + 24 months @ 5/12% = 20% + 10% = 30% reduction
    expected_factor = 1.0 - 36 * (5 / 9 / 100) - 24 * (5 / 12 / 100)
    assert abs(benefit - round(pia * expected_factor, 2)) < 0.01


def test_monthly_benefit_delayed_claim_70_fra67() -> None:
    """Claiming at 70 with FRA=67 gives 24% increase (36 months × 2/3%)."""
    aime = 5_000.0
    birth_year = 1960
    benefit = monthly_benefit(aime, birth_year, claim_age=70.0)
    pia = primary_insurance_amount(aime)
    expected_factor = 1.0 + 36 * (2 / 3 / 100)
    assert abs(benefit - round(pia * expected_factor, 2)) < 0.01


def test_monthly_benefit_rejects_below_min() -> None:
    with pytest.raises(ValueError, match="claim_age"):
        monthly_benefit(3_000.0, 1960, claim_age=61.0)


def test_monthly_benefit_rejects_above_max() -> None:
    with pytest.raises(ValueError, match="claim_age"):
        monthly_benefit(3_000.0, 1960, claim_age=71.0)


# ---------------------------------------------------------------------------
# as_cashflow
# ---------------------------------------------------------------------------


def test_as_cashflow_returns_periodic_cashflow() -> None:
    cf = as_cashflow(aime=5_000.0, birth_year=1960, claim_age=67.0)
    assert isinstance(cf, PeriodicCashflow)


def test_as_cashflow_amount_matches_monthly_benefit() -> None:
    aime = 5_000.0
    birth_year = 1960
    claim_age = 67.0
    cf = as_cashflow(aime, birth_year, claim_age)
    expected = monthly_benefit(aime, birth_year, claim_age)
    assert abs(cf.amount - expected) < 0.01


def test_as_cashflow_defaults() -> None:
    """Default cashflow is monthly, real, dollar-mode, no tax."""
    cf = as_cashflow(aime=4_000.0, birth_year=1962, claim_age=65.0)
    assert cf.frequency == Frequency.MONTHLY
    assert cf.mode == CashflowMode.DOLLAR
    assert cf.real is True
    assert cf.effective_tax_rate == 0.0


def test_as_cashflow_nominal_with_tax() -> None:
    cf = as_cashflow(
        aime=4_000.0,
        birth_year=1962,
        claim_age=65.0,
        real=False,
        effective_tax_rate=0.22,
    )
    assert cf.real is False
    assert abs(cf.effective_tax_rate - 0.22) < 1e-9


def test_as_cashflow_start_end_year_passed_through() -> None:
    cf = as_cashflow(
        aime=4_000.0,
        birth_year=1962,
        claim_age=67.0,
        start_year=5.0,
        end_year=30.0,
    )
    assert cf.start_year == 5.0
    assert cf.end_year == 30.0


def test_as_cashflow_start_year_suppresses_early_periods() -> None:
    """SS cashflow with start_year=5 must not fire before year 5."""
    cf = as_cashflow(aime=4_000.0, birth_year=1962, claim_age=67.0, start_year=5.0)
    # period 48 = year 4 (< 5) → 0
    assert cf.amount_at(48, 12, 0.0, 1.0) == 0.0
    # period 60 = year 5 → fires
    assert cf.amount_at(60, 12, 0.0, 1.0) > 0.0


# ---------------------------------------------------------------------------
# estimate_aime
# ---------------------------------------------------------------------------


def test_estimate_aime_full_career() -> None:
    """35-year career at $80k/yr → AIME = 80_000 / 12."""
    aime = estimate_aime(80_000.0, career_years=35)
    assert abs(aime - round(80_000.0 / 12, 2)) < 0.01


def test_estimate_aime_partial_career() -> None:
    """20-year career at $60k/yr → AIME = 60_000/12 * 20/35."""
    aime = estimate_aime(60_000.0, career_years=20)
    expected = round(60_000.0 / 12 * 20 / 35, 2)
    assert abs(aime - expected) < 0.01


def test_estimate_aime_over_35_years_clamped() -> None:
    """40 years same as 35 — extra years don't raise AIME."""
    assert estimate_aime(100_000.0, career_years=40) == estimate_aime(
        100_000.0, career_years=35
    )


def test_estimate_aime_rejects_zero_years() -> None:
    with pytest.raises(ValueError, match="career_years"):
        estimate_aime(50_000.0, career_years=0)


# ---------------------------------------------------------------------------
# estimate_aime_from_history
# ---------------------------------------------------------------------------


def test_estimate_aime_from_history_uses_top_35() -> None:
    """Only the 35 highest-earning years count."""
    # 40 years: 35 years at $100k, 5 years at $10k
    earnings = [100_000.0] * 35 + [10_000.0] * 5
    aime = estimate_aime_from_history(earnings)
    expected = estimate_aime(100_000.0, career_years=35)
    assert abs(aime - expected) < 0.01


def test_estimate_aime_from_history_single_year() -> None:
    aime = estimate_aime_from_history([60_000.0])
    expected = round(60_000.0 / 12 / 35, 2)
    assert abs(aime - expected) < 0.01


def test_estimate_aime_from_history_empty_raises() -> None:
    with pytest.raises(ValueError, match="annual_earnings"):
        estimate_aime_from_history([])


def test_estimate_aime_from_history_matches_flat_career() -> None:
    """Flat 35-year history should match estimate_aime."""
    salary = 75_000.0
    history = [salary] * 35
    assert estimate_aime_from_history(history) == estimate_aime(salary, career_years=35)
