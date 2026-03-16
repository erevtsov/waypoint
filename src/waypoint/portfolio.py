"""Portfolio: a named, weighted collection of assets."""

from __future__ import annotations

from datetime import date
from functools import reduce
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from waypoint.analysis.methods.returns import ReturnMethod
    from waypoint.analysis.methods.risk import RiskMethod
    from waypoint.asset_def import AssetDef
    from waypoint.assets import Asset, LeveragedAsset

from waypoint.analysis.methods.returns import GeometricMean
from waypoint.analysis.methods.risk import SampleCovariance
from waypoint.enums import PERIODS_PER_YEAR, Frequency

# Sentinel used as cache key when no date range is specified
_NO_DATE = date(1, 1, 1)

# Slot value can be a pre-loaded Asset, a leveraged wrapper, or a definition to fetch lazily
Slot = "Asset | LeveragedAsset | AssetDef"

_WEIGHT_TOLERANCE = 1e-9

# Polars group_by_dynamic interval string for each non-daily frequency
_RESAMPLE_EVERY: dict[Frequency, str] = {
    Frequency.WEEKLY: "1w",
    Frequency.MONTHLY: "1mo",
    Frequency.QUARTERLY: "1q",
    Frequency.ANNUAL: "1y",
}


def _resample_wide(wide: pl.DataFrame, to_freq: Frequency) -> pl.DataFrame:
    """Compound per-period returns in *wide* to the target frequency.

    No-op when *to_freq* is ``Frequency.DAILY``.  Each column other than
    ``"date"`` is treated as a decimal return series and compounded within
    each calendar interval.

    Dates are aligned to the *end* of each period: ``label="right"`` in
    polars produces the start of the following interval, so we subtract one
    day to obtain the conventional end-of-period date (e.g. Mar 31 for Q1,
    Jan 31 for January).
    """
    every = _RESAMPLE_EVERY.get(to_freq)
    if every is None:
        return wide  # DAILY -- nothing to resample
    return_cols = [c for c in wide.columns if c != "date"]
    return (
        wide.sort("date")
        .group_by_dynamic("date", every=every, label="right")
        .agg([((1 + pl.col(c)).product() - 1).alias(c) for c in return_cols])
        .with_columns(pl.col("date") - pl.duration(days=1))
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
        slots: dict[str, Asset | LeveragedAsset | AssetDef],
        weights: dict[str, float],
        name: str = "",
        normalize_weights: bool = True,
        expected_return_method: ReturnMethod | None = None,
        risk_method: RiskMethod | None = None,
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

        self._slots: dict[str, Asset | LeveragedAsset | AssetDef] = dict(slots)
        self.name = name

        self._expected_return_method: ReturnMethod = (
            expected_return_method if expected_return_method is not None else GeometricMean()
        )
        self._risk_method: RiskMethod = (
            risk_method if risk_method is not None else SampleCovariance()
        )

        # Mutable cache: keyed by (start, end, frequency) -> wide pl.DataFrame
        self._cache: dict[tuple[date, date, Frequency | None], pl.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def slots(self) -> dict[str, Asset | LeveragedAsset | AssetDef]:
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

    @property
    def expected_return_method(self) -> ReturnMethod:
        """Method used to estimate expected returns for this portfolio."""
        return self._expected_return_method

    @expected_return_method.setter
    def expected_return_method(self, method: ReturnMethod) -> None:
        self._expected_return_method = method

    @property
    def risk_method(self) -> RiskMethod:
        """Method used to estimate covariance / risk for this portfolio."""
        return self._risk_method

    @risk_method.setter
    def risk_method(self, method: RiskMethod) -> None:
        self._risk_method = method

    @property
    def native_frequency(self) -> Frequency:
        """Coarsest native frequency across all portfolio slots.

        This is the finest resolution at which the full portfolio can be
        evaluated without disaggregating any asset — i.e. the coarsest slot
        frequency limits the whole portfolio.  Requesting any frequency finer
        than this in ``get_returns`` raises a ``ValueError``.
        """
        return min(
            (slot.frequency for slot in self._slots.values()),
            key=lambda f: PERIODS_PER_YEAR[f],
        )

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
        from waypoint.assets import Asset, LeveragedAsset
        from waypoint.data import fetch

        has_defs = any(isinstance(s, AssetDef) for s in self._slots.values())

        if has_defs and (start is None or end is None):
            raise ValueError(
                "start and end are required when the portfolio contains AssetDef slots."
            )

        start_dt: date | None = date.fromisoformat(start) if isinstance(start, str) else start
        end_dt: date | None = date.fromisoformat(end) if isinstance(end, str) else end
        freq: Frequency | None = Frequency(frequency) if frequency is not None else None

        if freq is not None:
            requested_ppy = PERIODS_PER_YEAR[freq]
            native_ppy = PERIODS_PER_YEAR[self.native_frequency]
            if requested_ppy > native_ppy:
                offending = [
                    name
                    for name, slot in self._slots.items()
                    if PERIODS_PER_YEAR[slot.frequency] < requested_ppy
                ]
                raise ValueError(
                    f"Requested frequency {str(freq)!r} ({requested_ppy} periods/year) is finer "
                    f"than the native frequency of slot(s) {offending}. "
                    f"Use {str(self.native_frequency)!r} or coarser."
                )

        cache_key = (start_dt or _NO_DATE, end_dt or _NO_DATE, freq)
        if cache_key in self._cache:
            return self._cache[cache_key]

        frames: list[pl.DataFrame] = []
        for slot_name, slot in self._slots.items():
            if isinstance(slot, AssetDef):
                asset = fetch(slot, start=start_dt, end=end_dt)  # type: ignore[arg-type]
                returns_df = asset.returns
            else:
                assert isinstance(slot, (Asset, LeveragedAsset))
                returns_df = (
                    slot.get_returns(start_dt, end_dt)  # type: ignore[arg-type]
                    if start_dt is not None
                    else slot.returns
                )
            # Resample each asset individually before joining so that higher-frequency
            # assets are compounded to the target frequency (e.g. daily → quarterly)
            # rather than being sampled at the sparse dates of lower-frequency assets.
            single = returns_df.rename({"returns": slot_name})
            if freq is not None:
                single = _resample_wide(single, freq)
            frames.append(single)

        # Inner-join all (already-resampled) frames on date
        wide = frames[0]
        for frame in frames[1:]:
            wide = wide.join(frame, on="date", how="inner")

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
