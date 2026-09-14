"""Golden regression fixture for Risk — pins current numeric output."""

from __future__ import annotations

from waypoint.analysis.methods.risk import SampleCovariance
from waypoint.analysis.risk import Risk

from ._golden_portfolio import make_golden_portfolio

_TOLERANCE = 1e-9


def test_risk_golden() -> None:
    portfolio = make_golden_portfolio()
    result = Risk(method=SampleCovariance()).compute(
        portfolio, start=None, end=None, frequency="daily"
    )

    expected_vol = {
        "Equities": 0.1948824374384027,
        "Bonds": 0.06436698797479568,
        "Cash": 0.012631269543783947,
    }
    for name, exp in expected_vol.items():
        actual = result.volatilities[name]
        assert abs(actual - exp) < _TOLERANCE, f"{name}: got {actual!r}, expected {exp!r}"

    assert abs(result.portfolio_volatility - 0.11845938509147998) < _TOLERANCE
    assert result.covariance.columns == ["Equities", "Bonds", "Cash"]

    expected_cov_rows = [
        [0.037979164421932936, -3.5888643841376995e-05, -2.1949329677198423e-05],
        [-3.5888643841376995e-05, 0.004143109140947491, 2.008738852835055e-05],
        [-2.1949329677198423e-05, 2.008738852835055e-05, 0.00015954897028772392],
    ]
    for row_idx, row in enumerate(result.covariance.iter_rows()):
        for col_idx, val in enumerate(row):
            exp = expected_cov_rows[row_idx][col_idx]
            assert abs(val - exp) < _TOLERANCE, f"cov[{row_idx}][{col_idx}]: got {val!r}"
