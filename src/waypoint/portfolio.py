"""Portfolio: a named, weighted collection of assets."""

from __future__ import annotations

from datetime import date
from functools import reduce
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from waypoint.asset_def import AssetDef
    from waypoint.assets import Asset

from waypoint.enums import Frequency

# Sentinel used as cache key when no date range is specified
_NO_DATE = date(1, 1, 1)

# Slot value can be a pre-loaded Asset or a definition to fetch lazily
Slot = "Asset | AssetDef"

_WEIGHT_TOLERANCE = 1e-9

# Polars group_by_dynamic interval string for each non-daily frequency
_RESAMPLE_EVERY: dict[Frequency, str] = {
    Frequency.WEEKLY: "1w",
    Frequency.MONTHLY: "1mo",
    Frequency.ANNUAL: "1y",
}


def _resample_wide(wide: pl.DataFrame, to_freq: Frequency) -> pl.DataFrame:
    """Compound per-period returns in *wide* to the target frequency.

    No-op when *to_freq* is ``Frequency.DAILY``.  Each column other than
    ``"date"`` is treated as a decimal return series and compounded within
    each calendar interval.
    """
    every = _RESAMPLE_EVERY.get(to_freq)
    if every is None:
        return wide  # DAILY -- nothing to resample
    return_cols = [c for c in wide.columns if c != "date"]
    return (
        wide.sort("date")
        .group_by_dynamic("date", every=every)
        .agg([((1 + pl.col(c)).product() - 1).alias(c) for c in return_cols])
        .sort("date")
    )


class Portfolio:
    """A named, weighted collection of assets or asset definitions.

    Supports two construction styles:

    *Definitions-first* (common): pass ``AssetDef`` values; data is fetched
    on demand when ``get_returns`` is called.  Re-running analytics for a
    different date range is as simple as passing new ``start``/``end`` args.

    *Asset-first*: pass pre-loaded ``Asset`` values; ``get_returns`` filters
    the existing data to the requested range -- no network calls.

    Parameters
    ----------
    slots:
        Mapping of display name -> ``Asset`` or ``AssetDef``.
    weights:
        Mapping of display name -> weight (must use the same keys as *slots*).
        By default, weights are normalised to sum to 1.0.  Pass
        ``normalize_weights=False`` for long-short portfolios where gross
        exposure may exceed 1.0 or weights are already in their intended form.
    name:
        Optional display name for the portfolio.
    normalize_weights:
        If ``True`` (default), weights are divided by their sum so they total
        1.0.  If ``False``, weights are stored as supplied; the caller is
        responsible for ensuring they represent the intended exposures.
    """

    def __init__(
        self,
        slots: dict[str, Asset | AssetDef],
        weights: dict[str, float],
        name: str = "",
        normalize_weights: bool = True,
    ) -> None:
        if slots.keys() != weights.keys():
            raise ValueError(
                f"slots and weights must have the same keys. "
                f"slots={sorted(slots)}, weights={sorted(weights)}"
            )
        if not slots:
            raise ValueError("Portfolio must contain at least one slot.")

        if normalize_weights:
            total = sum(weights.values())
            if total == 0:
                raise ValueError(
                    "Weights sum to zero -- cannot normalise. "
                    "Pass normalize_weights=False for long-short portfolios."
                )
            self._weights: dict[str, float] = {k: v / total for k, v in weights.items()}
        else:
            self._weights = dict(weights)

        self._slots: dict[str, Asset | AssetDef] = dict(slots)
        self.name = name

        # Mutable cache: keyed by (start, end, frequency) -> wide pl.DataFrame
        self._cache: dict[tuple[date, date, Frequency | None], pl.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def slots(self) -> dict[str, Asset | AssetDef]:
        """The slot definitions (read-only view)."""
        return dict(self._slots)

    @property
    def weights(self) -> dict[str, float]:
        """Normalised weights (read-only view, sum = 1.0)."""
        return dict(self._weights)

    @property
    def names(self) -> list[str]:
        """Ordered list of slot names."""
        return list(self._slots)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_returns(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        frequency: Frequency | str | None = None,
    ) -> pl.DataFrame:
        """Return a wide DataFrame with one column per slot, aligned on ``"date"``.

        For ``AssetDef`` slots, data is fetched from the vendor (using the
        parquet cache).  For ``Asset`` slots, the existing return series is
        filtered to the requested range.  The result is cached in memory so
        repeated calls with the same range are free.

        Parameters
        ----------
        start, end:
            Date range (inclusive).  Required when any slot is an ``AssetDef``;
            ignored for the alignment operation when all slots are ``Asset``.
        frequency:
            When provided, the raw (daily) returns are compounded up to this
            frequency before being returned.  Accepts a ``Frequency`` member
            or its lowercase string equivalent.  ``None`` returns the native
            resolution of the underlying data (typically daily).
        """
        from waypoint.asset_def import AssetDef
        from waypoint.assets import Asset
        from waypoint.data import fetch

        has_defs = any(isinstance(s, AssetDef) for s in self._slots.values())

        if has_defs and (start is None or end is None):
            raise ValueError(
                "start and end are required when the portfolio contains AssetDef slots."
            )

        start_dt: date | None = date.fromisoformat(start) if isinstance(start, str) else start
        end_dt: date | None = date.fromisoformat(end) if isinstance(end, str) else end
        freq: Frequency | None = Frequency(frequency) if frequency is not None else None

        cache_key = (start_dt or _NO_DATE, end_dt or _NO_DATE, freq)
        if cache_key in self._cache:
            return self._cache[cache_key]

        frames: list[pl.DataFrame] = []
        for slot_name, slot in self._slots.items():
            if isinstance(slot, AssetDef):
                asset = fetch(slot, start=start_dt, end=end_dt)  # type: ignore[arg-type]
                returns_df = asset.returns
            else:
                assert isinstance(slot, Asset)
                returns_df = (
                    slot.get_returns(start_dt, end_dt)  # type: ignore[arg-type]
                    if start_dt is not None
                    else slot.returns
                )
            frames.append(returns_df.rename({"returns": slot_name}))

        # Inner-join all frames on date so every column covers the same dates
        wide = frames[0]
        for frame in frames[1:]:
            wide = wide.join(frame, on="date", how="inner")

        if freq is not None:
            wide = _resample_wide(wide, freq)

        self._cache[cache_key] = wide
        return wide

    def portfolio_returns(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        frequency: Frequency | str | None = None,
    ) -> pl.DataFrame:
        """Return a ``["date", "returns"]`` DataFrame of portfolio-level returns.

        Portfolio return at each date = weighted sum of slot returns.

        Parameters
        ----------
        start, end:
            Date range (inclusive).
        frequency:
            When provided, slot returns are resampled to this frequency before
            computing the weighted sum.  See ``get_returns`` for details.
        """
        wide = self.get_returns(start, end, frequency=frequency)
        asset_cols = [c for c in wide.columns if c != "date"]
        weighted: pl.Expr = reduce(
            lambda acc, name: acc + pl.col(name) * self._weights[name],
            asset_cols,
            pl.lit(0.0),
        )
        return wide.select(pl.col("date"), weighted.alias("returns"))
