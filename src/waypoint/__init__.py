"""Waypoint: financial portfolio advisor library."""

from importlib.metadata import version

from waypoint import catalog
from waypoint.analysis.expected_return import ExpectedReturn, ExpectedReturnResult
from waypoint.analysis.methods.returns import HistoricalMean
from waypoint.analysis.methods.risk import SampleCovariance
from waypoint.analysis.methods.simulation import Bootstrap, MonteCarlo
from waypoint.analysis.optimizer import EfficientFrontierResult, Optimizer
from waypoint.analysis.risk import Risk, RiskResult
from waypoint.analysis.simulation import SimulationResult, WealthSimulation
from waypoint.asset_def import AssetDef
from waypoint.assets import Asset
from waypoint.cashflows import CashflowDefinition, LumpSum, PeriodicCashflow
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
    "catalog",
    # enums
    "CashflowMode",
    "Frequency",
    # definitions
    "AssetDef",
    "IndicatorDef",
    # domain objects
    "Asset",
    "Indicator",
    "Portfolio",
    # data
    "fetch",
    # cashflows
    "CashflowDefinition",
    "LumpSum",
    "PeriodicCashflow",
    # constraints
    "DEFAULT_CONSTRAINTS",
    "LongOnly",
    "SumToOne",
    "WeightBounds",
    # analytics
    "ExpectedReturn",
    "ExpectedReturnResult",
    "Risk",
    "RiskResult",
    "Optimizer",
    "EfficientFrontierResult",
    "WealthSimulation",
    "SimulationResult",
    # methods
    "HistoricalMean",
    "SampleCovariance",
    "Bootstrap",
    "MonteCarlo",
]
