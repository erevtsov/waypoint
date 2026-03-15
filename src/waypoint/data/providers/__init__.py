"""Provider registry — maps vendor names to provider instances."""

from __future__ import annotations

from waypoint.data.providers.base import Provider
from waypoint.data.providers.eodhd import EodhdProvider
from waypoint.data.providers.fred import FredProvider
from waypoint.data.providers.yfinance import YFinanceProvider

_REGISTRY: dict[str, Provider] = {
    "yfinance": YFinanceProvider(),
    "fred": FredProvider(),
    "eodhd": EodhdProvider(),
}


def get_provider(vendor: str) -> Provider:
    """Return the provider instance for *vendor*.

    Raises
    ------
    KeyError
        If *vendor* is not in the registry.
    """
    try:
        return _REGISTRY[vendor]
    except KeyError:
        raise KeyError(
            f"Unknown vendor {vendor!r}. "
            f"Available vendors: {sorted(_REGISTRY)}"
        ) from None


__all__ = ["Provider", "get_provider"]
