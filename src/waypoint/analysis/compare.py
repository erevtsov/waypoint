"""Scenario comparison for WealthSimulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import polars as pl

if TYPE_CHECKING:
    from waypoint.analysis.simulation import SimulationResult


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side comparison of two or more ``WealthSimulation`` results.

    Construct via ``ComparisonResult.from_scenarios`` rather than directly,
    so that the minimum-scenario validation is applied.

    Parameters
    ----------
    scenarios:
        Mapping of scenario label to ``SimulationResult``.  Dict insertion
        order is preserved (Python 3.7+).
    """

    scenarios: dict[str, SimulationResult]

    @classmethod
    def from_scenarios(cls, scenarios: dict[str, SimulationResult]) -> ComparisonResult:
        """Build a ``ComparisonResult`` from a mapping of named simulation results.

        Parameters
        ----------
        scenarios:
            Mapping of scenario name to ``SimulationResult``.  Must contain at
            least two entries.

        Returns
        -------
        ComparisonResult
        """
        if len(scenarios) < 2:
            raise ValueError("from_scenarios requires at least two scenarios.")
        return cls(scenarios=scenarios)

    def prob_wins(self, a: str, b: str) -> float:
        """Fraction of paths where scenario *a* terminal wealth exceeds *b*.

        Uses ``min(n_sims_a, n_sims_b)`` paths.  Paths are treated as
        independent draws; scenarios need not share random seeds.

        Parameters
        ----------
        a, b:
            Scenario labels that must exist in ``self.scenarios``.
        """
        terminal_a = self.scenarios[a].paths[:, -1]
        terminal_b = self.scenarios[b].paths[:, -1]
        n = min(len(terminal_a), len(terminal_b))
        return float(np.mean(terminal_a[:n] > terminal_b[:n]))

    def summary(self) -> pl.DataFrame:
        """Terminal wealth statistics for each scenario.

        Returns
        -------
        pl.DataFrame
            One row per scenario with columns ``scenario``, ``initial_wealth``,
            ``p5``, ``p50``, ``p95``.
        """
        rows = []
        for label, result in self.scenarios.items():
            s = result.summary()
            rows.append(
                {
                    "scenario": label,
                    "initial_wealth": result.initial_wealth,
                    "p5": s["p5_terminal"],
                    "p50": s["median_terminal"],
                    "p95": s["p95_terminal"],
                }
            )
        return pl.DataFrame(rows)

    def plot(self) -> go.Figure:
        """Overlaid fan charts for all scenarios."""
        from waypoint.analysis.viz import plot_comparison

        return plot_comparison(self)
