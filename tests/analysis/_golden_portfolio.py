"""Shared golden-fixture portfolio builder for regression tests.

Deterministic seeded data reused across the `test_*_golden.py` files so
ExpectedReturn, Risk, Optimizer, and WealthSimulation golden tests all pin
against the exact same synthetic universe.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from waypoint.assets import Asset
from waypoint.portfolio import Portfolio

GOLDEN_N_PERIODS = 504


def _make_asset(name: str, ticker: str, mean: float, std: float, seed: int) -> Asset:
    rng = np.random.default_rng(seed=seed)
    dates = [date(2020, 1, 2) + timedelta(days=i) for i in range(GOLDEN_N_PERIODS)]
    values = rng.normal(mean, std, GOLDEN_N_PERIODS).tolist()
    return Asset(
        name=name,
        ticker=ticker,
        returns=pl.DataFrame({"date": dates, "returns": values}),
        frequency="daily",
    )


def make_golden_portfolio() -> Portfolio:
    """Deterministic 3-asset portfolio used by all golden regression tests.

    Do not change the seeds, means, stds, or weights below — golden values
    in the sibling test files were computed against this exact universe.
    Changing this function invalidates every golden fixture.
    """
    eq = _make_asset("Equities", "EQ", mean=0.0004, std=0.012, seed=101)
    fi = _make_asset("Bonds", "FI", mean=0.0001, std=0.004, seed=102)
    cash = _make_asset("Cash", "CASH", mean=0.00003, std=0.0008, seed=103)
    return Portfolio(
        {"Equities": eq, "Bonds": fi, "Cash": cash},
        weights={"Equities": 0.6, "Bonds": 0.3, "Cash": 0.1},
    )
