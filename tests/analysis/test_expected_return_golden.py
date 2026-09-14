"""Golden regression fixture for ExpectedReturn — pins current numeric output.

If this test fails after a code change, the change altered expected-return
math. Verify the new values are intentional before updating the constants
below; do not update them reflexively just to make the test pass.
"""

from __future__ import annotations

from waypoint.analysis.expected_return import ExpectedReturn
from waypoint.analysis.methods.returns import GeometricMean

from ._golden_portfolio import make_golden_portfolio

_TOLERANCE = 1e-9


def test_expected_return_golden() -> None:
    portfolio = make_golden_portfolio()
    result = ExpectedReturn(method=GeometricMean()).compute(
        portfolio, start=None, end=None, frequency="daily"
    )

    expected_per_asset = {
        "Equities": -0.025740420049084167,
        "Bonds": -0.024892866590215057,
        "Cash": 0.007554028718397943,
    }
    for name, exp in expected_per_asset.items():
        actual = result.per_asset[name]
        assert abs(actual - exp) < _TOLERANCE, f"{name}: got {actual!r}, expected {exp!r}"

    expected_portfolio = -0.022156709134675222
    assert abs(result.portfolio - expected_portfolio) < _TOLERANCE
