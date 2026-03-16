"""Real estate asset definitions (FHFA House Price Indices)."""

from waypoint.asset_def import AssetDef
from waypoint.enums import Frequency

MA_HPI = AssetDef(
    name="Massachusetts House Price Index",
    symbol="MASTHPI",
    vendor="fred",
    frequency=Frequency.QUARTERLY,
    asset_class="Real Estate",
    sub_asset_class="House Price Index",
    geography="Massachusetts",
)

BOSTON_HPI = AssetDef(
    name="Boston Metro Area House Price Index",
    symbol="ATNHPIUS14454Q",
    vendor="fred",
    frequency=Frequency.QUARTERLY,
    asset_class="Real Estate",
    sub_asset_class="House Price Index",
    geography="Massachusetts",
)
