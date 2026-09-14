"""Golden regression fixture for Optimizer — pins current numeric output.

Optimizer uses the cvxpy CLARABEL solver, which can differ in the noise
floor of near-zero weights across solver/platform versions. Tolerance here
is looser than the pure-numpy golden tests for that reason.

If any of these tests fail after a code change, verify the new values are
intentional before updating the constants — don't update them reflexively
just to make the test pass.
"""

from __future__ import annotations

from waypoint.analysis.expected_return import ExpectedReturn
from waypoint.analysis.methods.returns import GeometricMean
from waypoint.analysis.methods.risk import SampleCovariance
from waypoint.analysis.optimizer import Optimizer
from waypoint.analysis.risk import Risk
from waypoint.constraints import LongOnly, SumToOne

from ._golden_portfolio import make_golden_portfolio

_TOLERANCE = 1e-4


def _make_golden_optimizer() -> Optimizer:
    return Optimizer(
        return_model=ExpectedReturn(method=GeometricMean()),
        risk_model=Risk(method=SampleCovariance()),
        constraints=[LongOnly(), SumToOne()],
    )


def test_optimizer_frontier_golden() -> None:
    portfolio = make_golden_portfolio()
    frontier = _make_golden_optimizer().efficient_frontier(
        portfolio, start=None, end=None, frequency="daily", n_points=10
    )

    assert len(frontier.risks) == 10
    assert abs(frontier.risks[0] - 0.012416190846918607) < _TOLERANCE
    assert abs(frontier.risks[-1] - 0.012631269502637566) < _TOLERANCE
    assert abs(frontier.expected_returns[0] - 0.006342147978274902) < _TOLERANCE
    assert abs(frontier.expected_returns[-1] - 0.0075540286093250305) < _TOLERANCE


def test_optimizer_min_volatility_weights_golden() -> None:
    portfolio = make_golden_portfolio()
    frontier = _make_golden_optimizer().efficient_frontier(
        portfolio, start=None, end=None, frequency="daily", n_points=10
    )
    weights = frontier.min_volatility_portfolio(portfolio).weights

    expected = {
        "Equities": 0.004646675816906101,
        "Bonds": 0.032581614362263374,
        "Cash": 0.9627717098208305,
    }
    for name, exp in expected.items():
        actual = weights[name]
        assert abs(actual - exp) < _TOLERANCE, f"{name}: got {actual!r}, expected {exp!r}"


def test_optimizer_max_sharpe_weights_golden() -> None:
    portfolio = make_golden_portfolio()
    frontier = _make_golden_optimizer().efficient_frontier(
        portfolio, start=None, end=None, frequency="daily", n_points=10
    )
    weights = frontier.max_sharpe_portfolio(portfolio, risk_free_rate=0.02).weights

    expected = {"Equities": 0.0, "Bonds": 0.0, "Cash": 1.0}
    for name, exp in expected.items():
        actual = weights[name]
        assert abs(actual - exp) < 1e-3, f"{name}: got {actual!r}, expected {exp!r}"
