"""Correlated GBM Monte Carlo engine with Cholesky decomposition."""

from __future__ import annotations

import logging
import time

import numpy as np

from .market_data import MarketData

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """Generates correlated asset paths under risk-neutral GBM dynamics."""

    def __init__(self, n_paths: int = 100_000, seed: int = 42) -> None:
        self.n_paths = n_paths
        self.seed = seed
        logger.info("MonteCarloEngine created | n_paths=%d | seed=%d", n_paths, seed)

    def simulate(
        self, market_data: MarketData, observation_dates: np.ndarray
    ) -> np.ndarray:
        """
        Simulate correlated GBM paths at given observation dates.

        Parameters
        ----------
        market_data : MarketData
        observation_dates : np.ndarray
            Year fractions of observation dates (e.g. [0.25, 0.50, ...]).

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_assets, n_dates)
            Simulated spot prices at each observation date.
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        n_assets = market_data.n_assets
        n_dates = len(observation_dates)

        logger.debug(
            "Simulating %d paths × %d assets × %d dates | spots=%s | vols=%s",
            self.n_paths, n_assets, n_dates,
            market_data.spots, market_data.volatilities,
        )

        L = np.linalg.cholesky(market_data.correlation_matrix)

        times = np.concatenate([[0.0], observation_dates])
        dt = np.diff(times)

        paths = np.empty((self.n_paths, n_assets, n_dates))
        S_prev = np.broadcast_to(market_data.spots, (self.n_paths, n_assets)).copy()

        r = market_data.risk_free_rate
        q = market_data.dividend_yields
        sigma = market_data.volatilities

        for t in range(n_dates):
            Z = rng.standard_normal((self.n_paths, n_assets))
            W = Z @ L.T

            drift = (r - q - 0.5 * sigma**2) * dt[t]
            diffusion = sigma * np.sqrt(dt[t]) * W

            S_prev = S_prev * np.exp(drift + diffusion)
            paths[:, :, t] = S_prev

        elapsed = time.perf_counter() - t0
        logger.info(
            "Simulation done in %.3fs | terminal mean=%s",
            elapsed,
            np.array2string(paths[:, :, -1].mean(axis=0), precision=2),
        )
        return paths
