"""Risk estimation method protocols and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class RiskMethod(Protocol):
    """Protocol for covariance estimation methods."""

    def compute(self, returns_df: pl.DataFrame, periods_per_year: int) -> np.ndarray:
        """Compute annualised covariance matrix.

        Parameters
        ----------
        returns_df:
            Wide DataFrame of decimal periodic returns — one column per asset,
            no date column.
        periods_per_year:
            Number of periods per calendar year used to annualise the result.

        Returns
        -------
        np.ndarray
            Annualised covariance matrix of shape (n_assets, n_assets).
        """
        ...


@dataclass(frozen=True)
class SampleCovariance:
    """Sample covariance estimator scaled to annual frequency.

    Covariance = sample_covariance(returns) * periods_per_year.
    """

    def compute(self, returns_df: pl.DataFrame, periods_per_year: int) -> np.ndarray:
        """Return annualised sample covariance matrix."""
        data = returns_df.to_numpy()
        # np.cov returns a 0-D scalar for a single asset; atleast_2d ensures
        # the result always has shape (n_assets, n_assets) per the protocol contract.
        cov: np.ndarray = np.atleast_2d(np.cov(data, rowvar=False)) * periods_per_year
        return cov
