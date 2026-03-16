"""Return estimation methods — re-exported for the ``wp.returns`` namespace."""

from waypoint.analysis.methods.returns import ArithmeticMean, EWMAMean, GeometricMean, ViewReturn

__all__ = ["EWMAMean", "GeometricMean", "ArithmeticMean", "ViewReturn"]
