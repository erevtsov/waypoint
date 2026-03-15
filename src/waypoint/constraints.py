"""Portfolio constraint definitions for the optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import cvxpy as cp


@runtime_checkable
class Constraint(Protocol):
    """Protocol for portfolio constraints used in optimisation."""

    def to_cvxpy(self, w: Any, asset_names: list[str]) -> list[Any]:
        """Convert this constraint to one or more cvxpy constraint objects.

        Parameters
        ----------
        w:
            cvxpy Variable of shape (n_assets,) representing portfolio weights.
        asset_names:
            Ordered list of asset names (same order as w).

        Returns
        -------
        list[Any]
            List of cvxpy constraints.
        """
        ...


@dataclass(frozen=True)
class LongOnly:
    """All weights must be non-negative: w >= 0."""

    def to_cvxpy(self, w: Any, asset_names: list[str]) -> list[Any]:
        return [w >= 0]


@dataclass(frozen=True)
class WeightBounds:
    """Each weight must satisfy min_weight <= w_i <= max_weight.

    Parameters
    ----------
    min_weight:
        Lower bound on each individual weight (default 0.0).
    max_weight:
        Upper bound on each individual weight (default 1.0).
    """

    min_weight: float = field(default=0.0)
    max_weight: float = field(default=1.0)

    def to_cvxpy(self, w: Any, asset_names: list[str]) -> list[Any]:
        return [w >= self.min_weight, w <= self.max_weight]


@dataclass(frozen=True)
class SumToOne:
    """Portfolio weights must sum to exactly 1: sum(w) == 1."""

    def to_cvxpy(self, w: Any, asset_names: list[str]) -> list[Any]:
        return [cp.sum(w) == 1]  # type: ignore[attr-defined]


DEFAULT_CONSTRAINTS: list[Constraint] = [LongOnly(), SumToOne()]
