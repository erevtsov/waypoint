"""Indicator: a macro / signal time series stored as raw levels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from waypoint.enums import Frequency
from waypoint.indicator_def import IndicatorDef


@dataclass
class Indicator:
    """A macro or signal time series stored as raw levels (not returns).

    Returned by ``waypoint.data.fetch`` when given an ``IndicatorDef``.

    Parameters
    ----------
    name:
        Display name (e.g. "US 10-Year Treasury Yield").
    symbol:
        Vendor-native symbol (e.g. "DGS10").
    values:
        ``pl.DataFrame`` with columns ``"date"`` (``pl.Date``) and
        ``"value"`` (``pl.Float64``).  Values are raw levels; no
        unit conversion is applied.
    frequency:
        Observation frequency.
    unit:
        Optional description of the value unit (e.g. ``"percent"``).
    """

    name: str
    symbol: str
    values: pl.DataFrame
    frequency: Frequency
    unit: str = ""

    def __post_init__(self) -> None:
        cols = self.values.columns
        if cols != ["date", "value"]:
            raise TypeError(
                f"Indicator.values must have columns ['date', 'value'], got {cols}"
            )
        if self.values["date"].dtype != pl.Date:
            raise TypeError(
                f"Indicator.values['date'] must be pl.Date, "
                f"got {self.values['date'].dtype}"
            )
        if self.values["value"].dtype not in (pl.Float32, pl.Float64):
            raise TypeError(
                f"Indicator.values['value'] must be a float column, "
                f"got {self.values['value'].dtype}"
            )
        self.frequency = Frequency(self.frequency)

    def get_values(self, start: date | str, end: date | str) -> pl.DataFrame:
        """Return ``["date", "value"]`` filtered to *[start, end]* (inclusive)."""
        start_dt = date.fromisoformat(start) if isinstance(start, str) else start
        end_dt = date.fromisoformat(end) if isinstance(end, str) else end
        return self.values.filter(
            (pl.col("date") >= start_dt) & (pl.col("date") <= end_dt)
        )

    @classmethod
    def from_indicator_def(
        cls, indicator_def: IndicatorDef, values: pl.DataFrame
    ) -> Indicator:
        """Construct an ``Indicator`` from an ``IndicatorDef`` and a values DataFrame."""
        return cls(
            name=indicator_def.name,
            symbol=indicator_def.symbol,
            values=values,
            frequency=indicator_def.frequency,
            unit=indicator_def.unit,
        )
