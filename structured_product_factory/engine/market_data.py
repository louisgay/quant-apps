"""Market data container for multi-asset Monte Carlo simulation."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Stores spot prices, volatilities, correlations and rates for a basket."""

    tickers: List[str]
    spots: np.ndarray
    volatilities: np.ndarray
    correlation_matrix: np.ndarray
    risk_free_rate: float = 0.04
    dividend_yields: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        n = len(self.tickers)
        self.spots = np.asarray(self.spots, dtype=np.float64)
        self.volatilities = np.asarray(self.volatilities, dtype=np.float64)
        self.correlation_matrix = np.asarray(self.correlation_matrix, dtype=np.float64)
        if self.dividend_yields.size == 0:
            self.dividend_yields = np.zeros(n)
        self.dividend_yields = np.asarray(self.dividend_yields, dtype=np.float64)
        self._validate(n)
        logger.info(
            "MarketData created | tickers=%s | spots=%s | vols=%s | r=%.3f",
            self.tickers, self.spots, self.volatilities, self.risk_free_rate,
        )

    def _validate(self, n: int) -> None:
        assert self.spots.shape == (n,), f"spots shape {self.spots.shape} != ({n},)"
        assert self.volatilities.shape == (n,), "volatilities shape mismatch"
        assert self.correlation_matrix.shape == (n, n), "correlation matrix shape mismatch"
        assert np.allclose(self.correlation_matrix, self.correlation_matrix.T, atol=1e-8), (
            "correlation matrix must be symmetric"
        )
        eigvals = np.linalg.eigvalsh(self.correlation_matrix)
        assert np.all(eigvals > -1e-8), "correlation matrix must be positive semi-definite"

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    def bump_spot(self, asset_index: int, relative_bump: float) -> MarketData:
        """Return a copy with one spot bumped by a relative factor (e.g. 1.01 = +1%)."""
        new = copy.deepcopy(self)
        new.spots[asset_index] *= relative_bump
        return new

    def bump_vol(self, asset_index: int, absolute_bump: float) -> MarketData:
        """Return a copy with one vol bumped by an absolute amount (e.g. 0.01 = +1vol pt)."""
        new = copy.deepcopy(self)
        new.volatilities[asset_index] += absolute_bump
        return new

    def bump_correlation(self, absolute_bump: float) -> MarketData:
        """
        Uniformly bump all off-diagonal correlations and re-project to nearest
        valid correlation matrix via eigenvalue flooring.
        """
        new = copy.deepcopy(self)
        n = self.n_assets
        bumped = new.correlation_matrix.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    bumped[i, j] = np.clip(bumped[i, j] + absolute_bump, -0.99, 0.99)
        # naively bumping off-diag can break PSD — floor eigenvalues then renormalize
        eigvals, eigvecs = np.linalg.eigh(bumped)
        eigvals = np.maximum(eigvals, 1e-8)
        bumped = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalize diagonal to 1
        d = np.sqrt(np.diag(bumped))
        bumped = bumped / np.outer(d, d)
        new.correlation_matrix = bumped
        return new
