"""Golden regression fixture for WealthSimulation — pins current numeric output.

MonteCarlo(seed=42) is fully deterministic given a fixed numpy version, so
this pins tightly. A numpy upgrade that changes its RNG algorithm would
legitimately fail this test — that's the intended catch, not a false
positive; verify before updating the constants below.
"""

from __future__ import annotations

from waypoint.analysis.methods.simulation import MonteCarlo
from waypoint.analysis.simulation import WealthSimulation

from ._golden_portfolio import make_golden_portfolio

_TOLERANCE = 1e-6


def test_wealth_simulation_golden() -> None:
    portfolio = make_golden_portfolio()
    sim = WealthSimulation(
        method=MonteCarlo(seed=42),
        horizon_years=5,
        initial_wealth=100_000.0,
        n_simulations=500,
    )
    result = sim.compute(portfolio, start=None, end=None, frequency="daily")

    assert result.paths.shape == (500, 1261)

    summary = result.summary()
    expected = {
        "median_terminal": 84141.86530615282,
        "p5_terminal": 61290.09092477395,
        "p95_terminal": 134258.70426150376,
    }
    for key, exp in expected.items():
        actual = summary[key]
        assert abs(actual - exp) < _TOLERANCE, f"{key}: got {actual!r}, expected {exp!r}"
