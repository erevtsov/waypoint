"""Provider Protocol — the interface every data vendor must implement."""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class Provider(Protocol):
    """Fetch raw price or value data for a single symbol.

    Each vendor module implements this protocol and registers itself in the
    provider registry (``providers/__init__.py``).  Callers never interact
    with providers directly; they go through ``waypoint.data.fetch()``.
    """

    def fetch_raw(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        """Return raw vendor data for *symbol* over [start, end] inclusive.

        The returned DataFrame must contain at least two columns:
        - ``"date"``  — ``pl.Date`` dtype, one row per observation
        - ``"close"`` — ``pl.Float64`` dtype, adjusted close (equities) or
          level value (macro series)

        Rows outside [start, end] may be included; ``normalize.to_returns``
        will trim to the requested range after computing returns.
        """
        ...
