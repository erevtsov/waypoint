"""Aggregate: a collection of portfolios representing a total wealth picture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from waypoint.portfolio import Portfolio


def _safe_eq(a: object, b: object) -> bool:
    """Compare two objects, returning ``False`` if ``__eq__`` raises or returns non-bool.

    Handles method types (e.g. ``CAPM``) whose fields contain ``numpy`` arrays
    or ``Asset`` objects whose equality check returns an array rather than a scalar.
    """
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return type(a) is type(b)


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

    def _combined_slots_and_weights(
        self,
    ) -> tuple[dict[str, Any], dict[str, float], float]:
        """Combine slot definitions and wealth-weighted weights across all portfolios.

        Returns
        -------
        tuple[dict, dict, float]
            ``(combined_slots, combined_weights, total_wealth)`` — ready to pass
            directly to :class:`~waypoint.portfolio.Portfolio`.

        Raises
        ------
        ValueError
            If the same slot name maps to different assets across portfolios.
        """
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
        return combined_slots, combined_weights, total_wealth

    def flatten(self) -> Portfolio:
        """Collapse all portfolios into a single wealth-weighted portfolio.

        Slot weights are scaled by each account's wealth weight and summed
        across accounts.  If the same slot name appears in multiple portfolios,
        the weights are additive (total exposure is preserved).

        Method propagation
        ------------------
        If all portfolios share the same ``expected_return_method`` the flat
        portfolio inherits it; likewise for ``risk_method``.  If any two
        portfolios disagree, :exc:`ValueError` is raised — set the same method
        on every portfolio before flattening, or use a simulation analytic that
        accepts an explicit method parameter.

        Raises
        ------
        ValueError
            If the same slot name maps to different assets across portfolios,
            or if portfolios have differing ``expected_return_method`` /
            ``risk_method`` settings.

        Returns
        -------
        Portfolio
            A new portfolio whose weights reflect total wealth allocation.
            ``initial_wealth`` is set to the sum of all account wealths.
        """
        from waypoint.portfolio import Portfolio

        combined_slots, combined_weights, total_wealth = self._combined_slots_and_weights()

        er_methods = [p.expected_return_method for p in self._portfolios]
        if not all(_safe_eq(m, er_methods[0]) for m in er_methods[1:]):
            method_names = {
                p.name: type(p.expected_return_method).__name__ for p in self._portfolios
            }
            raise ValueError(
                f"Portfolios have differing expected_return_method settings: {method_names}. "
                f"Set the same method on all portfolios before flattening."
            )

        risk_methods = [p.risk_method for p in self._portfolios]
        if not all(_safe_eq(m, risk_methods[0]) for m in risk_methods[1:]):
            method_names = {p.name: type(p.risk_method).__name__ for p in self._portfolios}
            raise ValueError(
                f"Portfolios have differing risk_method settings: {method_names}. "
                f"Set the same method on all portfolios before flattening."
            )

        return Portfolio(
            slots=combined_slots,
            weights=combined_weights,
            normalize_weights=False,
            initial_wealth=total_wealth,
            expected_return_method=er_methods[0],
            risk_method=risk_methods[0],
        )

    def data_window(
        self,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
    ) -> Any:
        """Per-asset date coverage and effective inner-join window across all accounts.

        Delegates to the flattened portfolio's ``data_window``.  See
        ``Portfolio.data_window`` for column descriptions.
        """
        from waypoint.portfolio import Portfolio

        combined_slots, combined_weights, total_wealth = self._combined_slots_and_weights()
        flat = Portfolio(
            slots=combined_slots,
            weights=combined_weights,
            normalize_weights=False,
            initial_wealth=total_wealth,
        )
        return flat.data_window(start, end, frequency)

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
