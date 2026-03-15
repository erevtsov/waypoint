"""Tests for ComparisonResult and compare_scenarios."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from waypoint.analysis.compare import ComparisonResult
from waypoint.analysis.methods.simulation import MonteCarlo
from waypoint.analysis.simulation import WealthSimulation
from waypoint.assets import Asset
from waypoint.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PERIODS = 200
HORIZON_YEARS = 5
N_SIMULATIONS = 100


def _make_asset(mean: float, seed: int) -> Asset:
    n = N_PERIODS * 12
    rng = np.random.default_rng(seed=seed)
    dates = [date(2010, 1, 4) + timedelta(days=i) for i in range(n)]
    values = rng.normal(mean, 0.01, n).tolist()
    return Asset(
        name="Asset",
        ticker="A",
        returns=pl.DataFrame({"date": dates, "returns": values}),
        frequency="daily",
    )


def _make_result(mean: float, seed: int, initial_wealth: float = 1_000_000.0) -> object:
    portfolio = Portfolio(
        {"A": _make_asset(mean, seed)},
        weights={"A": 1.0},
    )
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=HORIZON_YEARS,
        initial_wealth=initial_wealth,
        n_simulations=N_SIMULATIONS,
    )
    return sim.compute(portfolio, start=None, end=None, frequency="daily")


# ---------------------------------------------------------------------------
# compare_scenarios validation
# ---------------------------------------------------------------------------


def test_compare_requires_at_least_two_scenarios() -> None:
    result = _make_result(mean=0.001, seed=10)
    with pytest.raises(ValueError, match="at least two"):
        ComparisonResult.from_scenarios({"only": result})


def test_compare_returns_comparison_result() -> None:
    a = _make_result(mean=0.001, seed=10)
    b = _make_result(mean=0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"A": a, "B": b})  # type: ignore[arg-type]
    assert isinstance(cr, ComparisonResult)


def test_compare_preserves_scenario_labels() -> None:
    a = _make_result(mean=0.001, seed=10)
    b = _make_result(mean=0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"Keep": a, "Sell": b})  # type: ignore[arg-type]
    assert list(cr.scenarios.keys()) == ["Keep", "Sell"]


# ---------------------------------------------------------------------------
# prob_wins
# ---------------------------------------------------------------------------


def test_prob_wins_between_zero_and_one() -> None:
    a = _make_result(mean=0.001, seed=10)
    b = _make_result(mean=0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"A": a, "B": b})  # type: ignore[arg-type]
    p = cr.prob_wins("A", "B")
    assert 0.0 <= p <= 1.0


def test_prob_wins_higher_mean_favoured() -> None:
    """Scenario with clearly higher expected return should win most of the time."""
    high = _make_result(mean=0.002, seed=10)
    low = _make_result(mean=-0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"High": high, "Low": low})  # type: ignore[arg-type]
    assert cr.prob_wins("High", "Low") > 0.6


def test_prob_wins_plus_loses_equals_one_for_distinct_paths() -> None:
    """prob_wins(A,B) + prob_wins(B,A) should be ~1 (ignoring exact ties)."""
    a = _make_result(mean=0.001, seed=10)
    b = _make_result(mean=0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"A": a, "B": b})  # type: ignore[arg-type]
    total = cr.prob_wins("A", "B") + cr.prob_wins("B", "A")
    # Total may be slightly < 1 if some paths tie exactly, but should be close.
    assert abs(total - 1.0) < 0.05


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_row_count() -> None:
    a = _make_result(mean=0.001, seed=10)
    b = _make_result(mean=0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"A": a, "B": b})  # type: ignore[arg-type]
    df = cr.summary()
    assert len(df) == 2


def test_summary_columns() -> None:
    a = _make_result(mean=0.001, seed=10)
    b = _make_result(mean=0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"A": a, "B": b})  # type: ignore[arg-type]
    df = cr.summary()
    assert df.columns == ["scenario", "initial_wealth", "p5", "p50", "p95"]


def test_summary_scenario_labels_match() -> None:
    a = _make_result(mean=0.001, seed=10)
    b = _make_result(mean=0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"Keep": a, "Sell": b})  # type: ignore[arg-type]
    assert cr.summary()["scenario"].to_list() == ["Keep", "Sell"]


def test_summary_percentile_ordering() -> None:
    """p5 <= p50 <= p95 in the summary."""
    a = _make_result(mean=0.001, seed=10, initial_wealth=500_000.0)
    b = _make_result(mean=0.001, seed=11, initial_wealth=800_000.0)
    cr = ComparisonResult.from_scenarios({"A": a, "B": b})  # type: ignore[arg-type]
    for row in cr.summary().iter_rows(named=True):
        assert row["p5"] <= row["p50"] + 1e-9
        assert row["p50"] <= row["p95"] + 1e-9


def test_summary_initial_wealth_preserved() -> None:
    a = _make_result(mean=0.001, seed=10, initial_wealth=500_000.0)
    b = _make_result(mean=0.001, seed=11, initial_wealth=800_000.0)
    cr = ComparisonResult.from_scenarios({"A": a, "B": b})  # type: ignore[arg-type]
    rows = {r["scenario"]: r for r in cr.summary().iter_rows(named=True)}
    assert rows["A"]["initial_wealth"] == 500_000.0
    assert rows["B"]["initial_wealth"] == 800_000.0


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------


def test_comparison_result_is_frozen() -> None:
    a = _make_result(mean=0.001, seed=10)
    b = _make_result(mean=0.001, seed=11)
    cr = ComparisonResult.from_scenarios({"A": a, "B": b})  # type: ignore[arg-type]
    with pytest.raises((AttributeError, TypeError)):
        cr.scenarios = {}  # type: ignore[misc]
