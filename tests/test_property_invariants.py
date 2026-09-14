"""Property-based tests for invariants that must never silently break.

Not exhaustive coverage — a handful of checks on math that would be
expensive to get wrong: weight normalization, risk non-negativity, and
the compounding identity underlying GeometricMean.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from waypoint.analysis.methods.returns import GeometricMean
from waypoint.analysis.methods.risk import SampleCovariance
from waypoint.assets import Asset
from waypoint.portfolio import Portfolio

_POSITIVE_WEIGHT = st.floats(
    min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False
)
_SMALL_RETURN = st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False)


def _dates(n: int) -> list[date]:
    return [date(2021, 1, 4) + timedelta(days=i) for i in range(n)]


@given(w_equities=_POSITIVE_WEIGHT, w_bonds=_POSITIVE_WEIGHT)
def test_portfolio_weights_always_normalize_to_one(w_equities: float, w_bonds: float) -> None:
    """Portfolio must normalize any positive weight pair to sum to 1.0."""
    dates = _dates(10)
    eq = Asset(
        name="Equities",
        ticker="EQ",
        returns=pl.DataFrame({"date": dates, "returns": [0.001] * 10}),
        frequency="daily",
    )
    fi = Asset(
        name="Bonds",
        ticker="FI",
        returns=pl.DataFrame({"date": dates, "returns": [0.0005] * 10}),
        frequency="daily",
    )
    portfolio = Portfolio(
        {"Equities": eq, "Bonds": fi},
        weights={"Equities": w_equities, "Bonds": w_bonds},
    )
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-9


@given(returns=st.lists(st.tuples(_SMALL_RETURN, _SMALL_RETURN), min_size=5, max_size=50))
def test_sample_covariance_diagonal_never_negative(returns: list[tuple[float, float]]) -> None:
    """SampleCovariance's diagonal (variance) must never be negative, for any input returns."""
    eq_vals = [r[0] for r in returns]
    fi_vals = [r[1] for r in returns]
    data = pl.DataFrame({"Equities": eq_vals, "Bonds": fi_vals})
    cov = SampleCovariance().compute(data, periods_per_year=1)
    assert cov[0, 0] >= 0.0
    assert cov[1, 1] >= 0.0


@given(returns=st.lists(_SMALL_RETURN, min_size=2, max_size=50))
def test_geometric_mean_compounding_identity(returns: list[float]) -> None:
    """(1 + per-period geometric mean)^n must equal the product of (1 + r_i).

    This is the defining identity of a geometric mean: GeometricMean's
    per-period result compounded n times must reproduce the same terminal
    wealth as applying the actual return sequence.
    """
    series = pl.Series("returns", returns)
    per_period = GeometricMean().compute(series, periods_per_year=1)

    n = len(returns)
    compounded_from_mean = (1.0 + per_period) ** n
    actual_compounded = float(np.prod([1.0 + r for r in returns]))

    assert math.isclose(compounded_from_mean, actual_compounded, rel_tol=1e-6, abs_tol=1e-9)
