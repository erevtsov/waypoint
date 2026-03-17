"""Simulation method protocols and implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SimulationMethod(Protocol):
    """Protocol for multi-asset return simulation methods."""

    def simulate(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        n_periods: int,
        n_simulations: int,
    ) -> np.ndarray:
        """Generate simulated period returns.

        Parameters
        ----------
        mu:
            Expected return vector of shape (n_assets,).
        sigma:
            Covariance matrix of shape (n_assets, n_assets).
        n_periods:
            Number of periods to simulate.
        n_simulations:
            Number of independent simulation paths.

        Returns
        -------
        np.ndarray
            Array of shape (n_simulations, n_periods) containing portfolio
            *period returns* (not wealth levels).
        """
        ...


@dataclass(frozen=True)
class MonteCarlo:
    """Monte Carlo simulation drawing from a multivariate normal distribution.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    """

    seed: int = field(default=42)

    def simulate(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        n_periods: int,
        n_simulations: int,
    ) -> np.ndarray:
        """Draw portfolio period returns from a multivariate normal.

        The portfolio returns are computed as the dot product of the
        multi-asset simulated returns with the weight vector implied by
        mu / sigma.  However, since this method receives the *portfolio*
        scalar mu and sigma, it draws from a univariate normal when both
        are scalar, and from a multivariate normal when they are arrays.

        In practice, mu and sigma here represent portfolio-level parameters
        (scalar mu, scalar variance) computed by the caller.

        Returns
        -------
        np.ndarray
            Shape (n_simulations, n_periods).
        """
        rng = np.random.default_rng(seed=self.seed)
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_2d(sigma)

        if mu_arr.shape == (1,) and sigma_arr.shape == (1, 1):
            # Univariate case: draw scalar returns
            std = float(np.sqrt(sigma_arr[0, 0]))
            draws: np.ndarray = rng.normal(
                loc=float(mu_arr[0]),
                scale=std,
                size=(n_simulations, n_periods),
            )
        else:
            # Multivariate case: draw and compute portfolio return
            # Returns shape: (n_simulations * n_periods, n_assets)
            if not np.all(np.isfinite(sigma_arr)):
                bad = np.argwhere(~np.isfinite(sigma_arr))
                raise ValueError(
                    f"Covariance matrix contains non-finite values at indices {bad.tolist()}. "
                    "Check that all assets have overlapping return history and no zero-variance series."
                )
            # Project onto the nearest positive-definite matrix: decompose via
            # eigh (symmetric eigendecomposition), clip any negative eigenvalues
            # to a small floor, then reconstruct. This handles near-singular or
            # rank-deficient covariance matrices that arise when n_assets is
            # large relative to the number of historical observations.
            MIN_EIGENVALUE = 1e-8
            eigvals, eigvecs = np.linalg.eigh(sigma_arr)
            eigvals = np.maximum(eigvals, MIN_EIGENVALUE)
            sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            draws_mv: np.ndarray = rng.multivariate_normal(
                mean=mu_arr,
                cov=sigma_psd,
                size=(n_simulations, n_periods),
                method="eigh",
            )
            draws = draws_mv

        return draws


@dataclass(frozen=True)
class Bootstrap:
    """Block bootstrap simulation drawing blocks from historical returns.

    Draws overlapping blocks of length ``block_size`` from
    ``historical_returns`` to fill the requested simulation length.
    mu and sigma parameters are ignored.

    Parameters
    ----------
    historical_returns:
        1-D array of historical portfolio period returns to resample.
    block_size:
        Length of each resampled block.
    seed:
        Random seed for reproducibility.
    """

    historical_returns: np.ndarray
    block_size: int = field(default=12)
    seed: int = field(default=42)

    def simulate(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        n_periods: int,
        n_simulations: int,
    ) -> np.ndarray:
        """Generate paths by block-bootstrapping from historical returns.

        Parameters
        ----------
        mu:
            Ignored for bootstrap.
        sigma:
            Ignored for bootstrap.
        n_periods:
            Number of periods per path.
        n_simulations:
            Number of simulation paths.

        Returns
        -------
        np.ndarray
            Shape (n_simulations, n_periods).
        """
        rng = np.random.default_rng(seed=self.seed)
        hist = self.historical_returns
        n_hist = len(hist)
        max_start = n_hist - self.block_size

        if max_start <= 0:
            raise ValueError(
                f"historical_returns length ({n_hist}) must exceed "
                f"block_size ({self.block_size})."
            )

        # Determine how many blocks needed to cover n_periods
        n_blocks = (n_periods + self.block_size - 1) // self.block_size
        result = np.empty((n_simulations, n_periods))

        for sim_idx in range(n_simulations):
            blocks: list[np.ndarray] = []
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            for start in starts:
                blocks.append(hist[start : start + self.block_size])
            path_full = np.concatenate(blocks)[:n_periods]
            result[sim_idx] = path_full

        return result
