"""Tests for cashflow definitions."""

from __future__ import annotations

import pytest

from waypoint.cashflows import LumpSum, PeriodicCashflow, _apply_tax


def _at(cf: PeriodicCashflow | LumpSum, period: int, pv: float = 0.0, ci: float = 1.0) -> float:
    """Shorthand to call amount_at with periods_per_year=12."""
    return cf.amount_at(
        period=period,
        periods_per_year=12,
        portfolio_value=pv,
        cumulative_inflation=ci,
    )


# ---------------------------------------------------------------------------
# PeriodicCashflow — dollar mode
# ---------------------------------------------------------------------------

def test_periodic_cashflow_dollar_fires_monthly() -> None:
    """Monthly cashflow should fire at every period when periods_per_year=12."""
    cf = PeriodicCashflow(amount=1000.0, frequency="monthly", mode="dollar")
    assert abs(_at(cf, 1, pv=100_000) - 1000.0) < 1e-9


def test_periodic_cashflow_annual_fires_at_year_boundary() -> None:
    """Annual cashflow fires every 12 periods when periods_per_year=12."""
    cf = PeriodicCashflow(amount=12_000.0, frequency="annual", mode="dollar")
    assert abs(_at(cf, 12) - 12_000.0) < 1e-9
    assert _at(cf, 1) == 0.0


def test_periodic_cashflow_does_not_fire_at_period_zero() -> None:
    """Period 0 is the starting period; no cashflow fires."""
    cf = PeriodicCashflow(amount=500.0, frequency="monthly", mode="dollar")
    assert _at(cf, 0) == 0.0


def test_periodic_cashflow_zero_outside_schedule() -> None:
    """Annual cashflow returns 0 at non-year periods."""
    cf = PeriodicCashflow(amount=1000.0, frequency="annual", mode="dollar")
    for period in [1, 2, 3, 5, 7, 11]:
        assert _at(cf, period) == 0.0


def test_periodic_cashflow_negative_is_withdrawal() -> None:
    """Negative amounts represent withdrawals."""
    cf = PeriodicCashflow(amount=-500.0, frequency="monthly", mode="dollar")
    assert _at(cf, 1, pv=10_000) < 0.0


# ---------------------------------------------------------------------------
# PeriodicCashflow — real vs nominal
# ---------------------------------------------------------------------------

def test_periodic_cashflow_real_scales_with_inflation() -> None:
    """real=True scales the dollar amount by cumulative_inflation."""
    cf = PeriodicCashflow(amount=1000.0, frequency="monthly", mode="dollar", real=True)
    cumulative = 1.1
    assert abs(_at(cf, 1, ci=cumulative) - 1000.0 * cumulative) < 1e-9


def test_periodic_cashflow_nominal_ignores_inflation() -> None:
    """real=False (default) leaves the dollar amount unchanged by inflation."""
    cf = PeriodicCashflow(amount=2000.0, frequency="monthly", mode="dollar")
    assert abs(_at(cf, 1, ci=1.5) - 2000.0) < 1e-9


def test_periodic_cashflow_pct_portfolio_ignores_real_flag() -> None:
    """pct_portfolio mode is always relative to current portfolio value; real has no effect."""
    cf_nominal = PeriodicCashflow(amount=0.01, frequency="monthly", mode="pct_portfolio")
    cf_real = PeriodicCashflow(amount=0.01, frequency="monthly", mode="pct_portfolio", real=True)
    pv = 500_000.0
    assert abs(_at(cf_nominal, 1, pv=pv, ci=1.3) - _at(cf_real, 1, pv=pv, ci=1.3)) < 1e-9


# ---------------------------------------------------------------------------
# PeriodicCashflow — effective_tax_rate
# ---------------------------------------------------------------------------

def test_periodic_cashflow_tax_grosses_up_withdrawal() -> None:
    """Withdrawal is grossed up so the portfolio impact covers taxes."""
    cf = PeriodicCashflow(
        amount=-80_000.0, frequency="annual", mode="dollar", effective_tax_rate=0.25
    )
    # Expected portfolio draw: -80_000 / (1 - 0.25) = -106_666.67
    result = cf.amount_at(
        period=12, periods_per_year=12, portfolio_value=0.0, cumulative_inflation=1.0
    )
    assert abs(result - (-80_000.0 / 0.75)) < 1e-6


def test_periodic_cashflow_tax_no_effect_on_contribution() -> None:
    """effective_tax_rate must not affect positive (contribution) cashflows."""
    cf = PeriodicCashflow(
        amount=10_000.0, frequency="monthly", mode="dollar", effective_tax_rate=0.30
    )
    assert abs(_at(cf, 1) - 10_000.0) < 1e-9


def test_periodic_cashflow_zero_tax_rate_unchanged() -> None:
    """effective_tax_rate=0 leaves the amount unchanged."""
    cf = PeriodicCashflow(
        amount=-5_000.0, frequency="monthly", mode="dollar", effective_tax_rate=0.0
    )
    assert abs(_at(cf, 1) - (-5_000.0)) < 1e-9


def test_periodic_cashflow_real_withdrawal_with_tax() -> None:
    """real=True inflation scaling applied before tax gross-up."""
    cf = PeriodicCashflow(
        amount=-80_000.0, frequency="annual", mode="dollar",
        real=True, effective_tax_rate=0.25,
    )
    ci = 1.1
    # nominal net = -80_000 * 1.1 = -88_000; gross = -88_000 / 0.75
    result = cf.amount_at(
        period=12, periods_per_year=12, portfolio_value=0.0, cumulative_inflation=ci
    )
    expected = (-80_000.0 * ci) / 0.75
    assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# PeriodicCashflow — pct_portfolio mode
# ---------------------------------------------------------------------------

def test_periodic_cashflow_pct_portfolio() -> None:
    """pct_portfolio mode returns amount * portfolio_value."""
    cf = PeriodicCashflow(amount=0.01, frequency="monthly", mode="pct_portfolio")
    pv = 500_000.0
    assert abs(_at(cf, 1, pv=pv) - 0.01 * pv) < 1e-9


def test_periodic_cashflow_pct_portfolio_with_tax() -> None:
    """pct_portfolio withdrawal is grossed up by tax rate."""
    cf = PeriodicCashflow(
        amount=-0.04, frequency="annual", mode="pct_portfolio", effective_tax_rate=0.25
    )
    pv = 1_000_000.0
    # net = -0.04 * pv = -40_000; gross = -40_000 / 0.75
    result = cf.amount_at(
        period=12, periods_per_year=12, portfolio_value=pv, cumulative_inflation=1.0
    )
    assert abs(result - (-40_000.0 / 0.75)) < 1e-6


# ---------------------------------------------------------------------------
# PeriodicCashflow — validation
# ---------------------------------------------------------------------------

def test_periodic_cashflow_monthly_with_quarterly_simulation() -> None:
    """Monthly cashflow bundled into quarterly simulation periods (3 payments/quarter)."""
    cf = PeriodicCashflow(amount=1000.0, frequency="monthly", mode="dollar")
    result = cf.amount_at(
        period=1, periods_per_year=4, portfolio_value=0.0, cumulative_inflation=1.0
    )
    assert abs(result - 3000.0) < 1e-9


def test_periodic_cashflow_annual_with_quarterly_simulation() -> None:
    """Annual cashflow fires once every 4 quarters."""
    cf = PeriodicCashflow(amount=12_000.0, frequency="annual", mode="dollar")
    result_at_4 = cf.amount_at(
        period=4, periods_per_year=4, portfolio_value=0.0, cumulative_inflation=1.0
    )
    result_at_1 = cf.amount_at(
        period=1, periods_per_year=4, portfolio_value=0.0, cumulative_inflation=1.0
    )
    assert abs(result_at_4 - 12_000.0) < 1e-9
    assert result_at_1 == 0.0


def test_periodic_cashflow_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        PeriodicCashflow(amount=100.0, frequency="weekly")


def test_periodic_cashflow_invalid_mode() -> None:
    with pytest.raises(ValueError):
        PeriodicCashflow(amount=100.0, frequency="monthly", mode="absolute")


def test_periodic_cashflow_is_frozen() -> None:
    cf = PeriodicCashflow(amount=100.0, frequency="monthly")
    with pytest.raises((AttributeError, TypeError)):
        cf.amount = 200.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LumpSum
# ---------------------------------------------------------------------------

def test_lump_sum_fires_at_target_period() -> None:
    """LumpSum fires exactly at round(at_year * periods_per_year)."""
    cf = LumpSum(amount=50_000.0, at_year=5.0)
    assert abs(_at(cf, 60) - 50_000.0) < 1e-9


def test_lump_sum_zero_outside_target() -> None:
    """LumpSum returns 0 for all other periods."""
    cf = LumpSum(amount=50_000.0, at_year=5.0)
    for period in [0, 1, 59, 61, 120]:
        assert _at(cf, period) == 0.0


def test_lump_sum_negative_is_withdrawal() -> None:
    """Negative lump sum represents a withdrawal at the given year."""
    cf = LumpSum(amount=-10_000.0, at_year=10.0)
    assert _at(cf, 120) < 0.0


def test_lump_sum_fractional_year() -> None:
    """Fractional years are rounded to the nearest period."""
    cf = LumpSum(amount=1000.0, at_year=0.5)
    assert abs(_at(cf, 6) - 1000.0) < 1e-9


def test_lump_sum_is_frozen() -> None:
    cf = LumpSum(amount=1000.0, at_year=1.0)
    with pytest.raises((AttributeError, TypeError)):
        cf.amount = 2000.0  # type: ignore[misc]


def test_lump_sum_real_scales_with_inflation() -> None:
    """real=True scales the lump sum by cumulative inflation at the firing period."""
    cf = LumpSum(amount=100_000.0, at_year=5.0, real=True)
    ci = 1.1628  # ~3% for 5 years
    result = _at(cf, 60, ci=ci)
    assert abs(result - 100_000.0 * ci) < 1e-6


def test_lump_sum_nominal_ignores_inflation() -> None:
    """real=False (default) — amount is unaffected by cumulative inflation."""
    cf = LumpSum(amount=100_000.0, at_year=5.0)
    assert abs(_at(cf, 60, ci=1.5) - 100_000.0) < 1e-9


def test_lump_sum_tax_grosses_up_withdrawal() -> None:
    """Lump sum withdrawal is grossed up by effective_tax_rate."""
    cf = LumpSum(amount=-50_000.0, at_year=10.0, effective_tax_rate=0.25)
    result = _at(cf, 120)
    assert abs(result - (-50_000.0 / 0.75)) < 1e-6


def test_lump_sum_tax_no_effect_on_contribution() -> None:
    """Positive lump sum is not affected by effective_tax_rate."""
    cf = LumpSum(amount=50_000.0, at_year=5.0, effective_tax_rate=0.30)
    assert abs(_at(cf, 60) - 50_000.0) < 1e-9


def test_lump_sum_real_withdrawal_with_tax() -> None:
    """real=True inflation scaling applied before tax gross-up."""
    cf = LumpSum(amount=-50_000.0, at_year=5.0, real=True, effective_tax_rate=0.25)
    ci = 1.16
    result = _at(cf, 60, ci=ci)
    expected = (-50_000.0 * ci) / 0.75
    assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# slots field
# ---------------------------------------------------------------------------

def test_periodic_cashflow_slots_default_none() -> None:
    cf = PeriodicCashflow(amount=1000.0, frequency="monthly")
    assert cf.slots is None


def test_periodic_cashflow_slots_tuple_stored() -> None:
    cf = PeriodicCashflow(amount=500.0, frequency="monthly", slots=("A", "B"))
    assert cf.slots == ("A", "B")


def test_periodic_cashflow_slots_list_coerced_to_tuple() -> None:
    cf = PeriodicCashflow(amount=500.0, frequency="monthly", slots=["A", "B"])  # type: ignore[arg-type]
    assert isinstance(cf.slots, tuple)
    assert cf.slots == ("A", "B")


def test_lump_sum_slots_default_none() -> None:
    cf = LumpSum(amount=10_000.0, at_year=5.0)
    assert cf.slots is None


def test_lump_sum_slots_list_coerced_to_tuple() -> None:
    cf = LumpSum(amount=10_000.0, at_year=5.0, slots=["X"])  # type: ignore[arg-type]
    assert isinstance(cf.slots, tuple)
    assert cf.slots == ("X",)


# ---------------------------------------------------------------------------
# _apply_tax helper
# ---------------------------------------------------------------------------

def test_apply_tax_withdrawal_grosses_up() -> None:
    assert abs(_apply_tax(-80_000.0, 0.25) - (-80_000.0 / 0.75)) < 1e-6


def test_apply_tax_contribution_unchanged() -> None:
    assert _apply_tax(10_000.0, 0.25) == 10_000.0


def test_apply_tax_zero_rate_unchanged() -> None:
    assert _apply_tax(-5_000.0, 0.0) == -5_000.0
