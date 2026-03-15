"""Waypoint: financial portfolio advisor library."""

from importlib.metadata import version

from waypoint import analytics, cashflows, catalog, returns, risk, sim
from waypoint.asset_def import AssetDef
from waypoint.assets import Asset, LeveragedAsset
from waypoint.constraints import DEFAULT_CONSTRAINTS, LongOnly, SumToOne, WeightBounds
from waypoint.data import fetch
from waypoint.enums import CashflowMode, Frequency
from waypoint.indicator_def import IndicatorDef
from waypoint.indicators import Indicator
from waypoint.portfolio import Portfolio

__version__: str = version("waypoint")

__all__ = [
    "__version__",
    # sub-modules
    "analytics",
    "catalog",
    "cashflows",
    "returns",
    "risk",
    "sim",
    # enums
    "CashflowMode",
    "Frequency",
    # definitions
    "AssetDef",
    "IndicatorDef",
    # domain objects
    "Asset",
    "LeveragedAsset",
    "Indicator",
    "Portfolio",
    # data
    "fetch",
    # constraints
    "DEFAULT_CONSTRAINTS",
    "LongOnly",
    "SumToOne",
    "WeightBounds",
]
