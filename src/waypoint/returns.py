"""Return estimation methods — re-exported for the ``wp.returns`` namespace."""

from waypoint.analysis.methods.returns import (
    CAPM,
    ArithmeticMean,
    EWMAMean,
    GeometricMean,
    PortfolioReturnMethod,
    ViewReturn,
)

__all__ = [
    "ArithmeticMean",
    "CAPM",
    "EWMAMean",
    "GeometricMean",
    "PortfolioReturnMethod",
    "ViewReturn",
]
