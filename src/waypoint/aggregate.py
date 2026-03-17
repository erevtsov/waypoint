"""Aggregate: a collection of portfolios representing a total wealth picture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio


class Aggregate:
    """A collection of portfolios representing total wealth across accounts.

    Provides a unified view across multiple ``Portfolio`` instances (e.g.
    taxable brokerage, 401k, Roth IRA).  Each portfolio must have
    ``initial_wealth`` set — this is used to compute relative weights for
    aggregation and flattening.

    Parameters
    ----------
    portfolios:
        Portfolios to aggregate.  Each must have ``initial_wealth`` set and
        a unique ``name``.

    Raises
    ------
    ValueError
        If any portfolio is missing ``initial_wealth``, or if portfolio names
        are not unique.
    """

    def __init__(self, portfolios: list[Portfolio] | tuple[Portfolio, ...]) -> None:
        self._portfolios: tuple[Portfolio, ...] = tuple(portfolios)

        missing = [p.name for p in self._portfolios if p.initial_wealth is None]
        if missing:
            raise ValueError(
                f"initial_wealth is required for all portfolios in an Aggregate. "
                f"Missing: {missing}"
            )

        names = [p.name for p in self._portfolios]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Portfolio names must be unique within an Aggregate. "
                f"Duplicates: {dupes}"
            )

        if not self._portfolios:
            raise ValueError("Aggregate must contain at least one portfolio.")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def portfolios(self) -> tuple[Portfolio, ...]:
        """Portfolios in this aggregate (read-only view)."""
        return self._portfolios

    @property
    def names(self) -> list[str]:
        """Ordered list of portfolio names."""
        return [p.name for p in self._portfolios]

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def wealth_weights(self) -> dict[str, float]:
        """Relative weight of each portfolio by ``initial_wealth``.

        Returns
        -------
        dict[str, float]
            Mapping of portfolio name → weight (sums to 1.0).
        """
        total = sum(p.initial_wealth for p in self._portfolios)  # type: ignore[misc]
        return {p.name: p.initial_wealth / total for p in self._portfolios}  # type: ignore[operator]

    def flatten(self) -> Portfolio:
        """Collapse all portfolios into a single wealth-weighted portfolio.

        Slot weights are scaled by each account's wealth weight and summed
        across accounts.  If the same slot name appears in multiple portfolios,
        the weights are additive (total exposure is preserved).

        Raises
        ------
        ValueError
            If the same slot name maps to different assets across portfolios.

        Returns
        -------
        Portfolio
            A new portfolio whose weights reflect total wealth allocation.
            ``initial_wealth`` is set to the sum of all account wealths.
        """
        from waypoint.portfolio import Portfolio

        weights_map = self.wealth_weights()
        combined_weights: dict[str, float] = {}
        combined_slots: dict[str, Any] = {}

        for portfolio in self._portfolios:
            account_weight = weights_map[portfolio.name]
            for slot_name, slot_weight in portfolio.weights.items():
                asset_weight = account_weight * slot_weight
                if slot_name in combined_slots:
                    existing = combined_slots[slot_name]
                    incoming = portfolio.slots[slot_name]
                    if existing is not incoming:
                        raise ValueError(
                            f"Slot {slot_name!r} maps to different assets across portfolios. "
                            f"Cannot flatten. Use run() to inspect per-account results."
                        )
                    combined_weights[slot_name] += asset_weight
                else:
                    combined_slots[slot_name] = portfolio.slots[slot_name]
                    combined_weights[slot_name] = asset_weight

        total_wealth = sum(p.initial_wealth for p in self._portfolios)  # type: ignore[misc]
        return Portfolio(
            slots=combined_slots,
            weights=combined_weights,
            normalize_weights=False,
            initial_wealth=total_wealth,
        )

    def run(self, analytic: Any, **kwargs: Any) -> dict[str, Any]:
        """Run ``analytic.compute(portfolio, **kwargs)`` for each portfolio.

        Parameters
        ----------
        analytic:
            Any analytic with a ``compute(portfolio, ...)`` method.
        **kwargs:
            Passed through to ``analytic.compute``.

        Returns
        -------
        dict[str, Any]
            Mapping of portfolio name → analytic result.
        """
        return {p.name: analytic.compute(p, **kwargs) for p in self._portfolios}
