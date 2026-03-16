"""Tests for cashflow definitions."""

from __future__ import annotations

import pytest

from waypoint.cashflows import LumpSum, PeriodicCashflow


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
    # period=1, periods_per_year=12 → fires every 1 period (12/12=1)
    assert abs(_at(cf, 1, pv=100_000) - 1000.0) < 1e-9


def test_periodic_cashflow_annual_fires_at_year_boundary() -> None:
    """Annual cashflow fires every 12 periods when periods_per_year=12."""
    cf = PeriodicCashflow(amount=12_000.0, frequency="annual", mode="dollar")
    # Should fire at period=12, not at period=1
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
# PeriodicCashflow — inflation adjustment
# ---------------------------------------------------------------------------

def test_periodic_cashflow_dollar_inflation_grows_amount() -> None:
    """With inflation_rate > 0, dollar cashflows grow by cumulative_inflation."""
    cf = PeriodicCashflow(
        amount=1000.0, frequency="monthly", mode="dollar", inflation_rate=0.03
    )
    cumulative = 1.1  # 10% cumulative inflation
    assert abs(_at(cf, 12, ci=cumulative) - 1000.0 * cumulative) < 1e-9


def test_periodic_cashflow_dollar_no_inflation() -> None:
    """Without inflation, dollar cashflow equals the raw amount."""
    cf = PeriodicCashflow(amount=2000.0, frequency="monthly", mode="dollar", inflation_rate=0.0)
    assert abs(_at(cf, 1) - 2000.0) < 1e-9


# ---------------------------------------------------------------------------
# PeriodicCashflow — pct_portfolio mode
# ---------------------------------------------------------------------------

def test_periodic_cashflow_pct_portfolio() -> None:
    """pct_portfolio mode returns amount * portfolio_value."""
    cf = PeriodicCashflow(amount=0.01, frequency="monthly", mode="pct_portfolio")
    pv = 500_000.0
    assert abs(_at(cf, 1, pv=pv) - 0.01 * pv) < 1e-9


def test_periodic_cashflow_pct_portfolio_inflation_adjusted() -> None:
    """pct_portfolio_inflation_adjusted mode returns amount * portfolio_value."""
    cf = PeriodicCashflow(
        amount=0.02, frequency="monthly", mode="pct_portfolio_inflation_adjusted"
    )
    pv = 200_000.0
    assert abs(_at(cf, 1, pv=pv, ci=1.5) - 0.02 * pv) < 1e-9


# ---------------------------------------------------------------------------
# PeriodicCashflow — validation
# ---------------------------------------------------------------------------

def test_periodic_cashflow_monthly_with_quarterly_simulation() -> None:
    """Monthly cashflow bundled into quarterly simulation periods (3 payments/quarter)."""
    cf = PeriodicCashflow(amount=1000.0, frequency="monthly", mode="dollar")
    # periods_per_year=4 (quarterly); cashflow fires 12/year → 3 per quarter
    result = cf.amount_at(
        period=1, periods_per_year=4, portfolio_value=0.0, cumulative_inflation=1.0
    )
    assert abs(result - 3000.0) < 1e-9


def test_periodic_cashflow_annual_with_quarterly_simulation() -> None:
    """Annual cashflow fires once every 4 quarters."""
    cf = PeriodicCashflow(amount=12_000.0, frequency="annual", mode="dollar")
    # periods_per_year=4; fires every 4th period
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
    # periods_per_year=12 → target_period = round(5.0 * 12) = 60
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
    # round(0.5 * 12) = 6
    assert abs(_at(cf, 6) - 1000.0) < 1e-9


def test_lump_sum_is_frozen() -> None:
    cf = LumpSum(amount=1000.0, at_year=1.0)
    with pytest.raises((AttributeError, TypeError)):
        cf.amount = 2000.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# slots field
# ---------------------------------------------------------------------------

def test_periodic_cashflow_slots_default_none() -> None:
    """slots defaults to None (all assets receive the cashflow)."""
    cf = PeriodicCashflow(amount=1000.0, frequency="monthly")
    assert cf.slots is None


def test_periodic_cashflow_slots_tuple_stored() -> None:
    """Tuple of slot names is stored as-is."""
    cf = PeriodicCashflow(amount=500.0, frequency="monthly", slots=("A", "B"))
    assert cf.slots == ("A", "B")


def test_periodic_cashflow_slots_list_coerced_to_tuple() -> None:
    """A list passed as slots is coerced to tuple."""
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
