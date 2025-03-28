"""Greeks via finite differences and correlation sensitivity analysis."""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import numpy as np

from .market_data import MarketData
from .monte_carlo import MonteCarloEngine
from .products import AutocallableBase

logger = logging.getLogger(__name__)


def _reprice(
    engine: MonteCarloEngine,
    product: AutocallableBase,
    market_data: MarketData,
    initial_spots: np.ndarray,
) -> float:
    """
    Helper: simulate from current market_data spots, but evaluate payoff
    relative to the fixed initial_spots (initial fixing).
    """
    paths = engine.simulate(market_data, product.observation_dates)
    result = product.price(paths, initial_spots, market_data.risk_free_rate)
    return result["price"]


class GreeksCalculator:
    """Compute Greeks by central finite differences."""

    def __init__(
        self,
        engine: MonteCarloEngine,
        product: AutocallableBase,
        market_data: MarketData,
    ) -> None:
        self.engine = engine
        self.product = product
        self.market_data = market_data
        # The initial fixing is frozen at construction time
        self.initial_spots = market_data.spots.copy()
        self._base_price: float | None = None
        logger.info(
            "GreeksCalculator initialized | underlyings=%s | spots=%s | vols=%s",
            product.underlyings, market_data.spots, market_data.volatilities,
        )

    @property
    def base_price(self) -> float:
        if self._base_price is None:
            t0 = time.perf_counter()
            self._base_price = _reprice(
                self.engine, self.product, self.market_data, self.initial_spots
            )
            logger.info(
                "Base price computed: %.2f (%.3fs)",
                self._base_price, time.perf_counter() - t0,
            )
        return self._base_price

    # ------------------------------------------------------------------
    # Delta: dV/dS  (per 1 unit move in spot)
    # ------------------------------------------------------------------
    def delta(self, bump_pct: float = 0.01) -> Dict[str, float]:
        """
        Central-difference delta for each underlying.
        bump_pct: relative bump size (0.01 = 1%).
        Returns delta per unit of spot move.
        """
        logger.info("Computing Delta (bump=%.2f%%) ...", bump_pct * 100)
        t0 = time.perf_counter()
        deltas = {}
        for i, ticker in enumerate(self.product.underlyings):
            md_up = self.market_data.bump_spot(i, 1 + bump_pct)
            md_down = self.market_data.bump_spot(i, 1 - bump_pct)
            p_up = _reprice(self.engine, self.product, md_up, self.initial_spots)
            p_down = _reprice(self.engine, self.product, md_down, self.initial_spots)
            dS = 2 * bump_pct * self.market_data.spots[i]
            deltas[ticker] = (p_up - p_down) / dS
            logger.info(
                "  Delta[%s]: p_up=%.2f  p_down=%.2f  dS=%.4f  => delta=%+.2f",
                ticker, p_up, p_down, dS, deltas[ticker],
            )
        logger.info("Delta done in %.3fs", time.perf_counter() - t0)
        return deltas

    # ------------------------------------------------------------------
    # Gamma: d²V/dS²
    # ------------------------------------------------------------------
    def gamma(self, bump_pct: float = 0.01) -> Dict[str, float]:
        logger.info("Computing Gamma (bump=%.2f%%) ...", bump_pct * 100)
        t0 = time.perf_counter()
        gammas = {}
        for i, ticker in enumerate(self.product.underlyings):
            md_up = self.market_data.bump_spot(i, 1 + bump_pct)
            md_down = self.market_data.bump_spot(i, 1 - bump_pct)
            p_up = _reprice(self.engine, self.product, md_up, self.initial_spots)
            p_down = _reprice(self.engine, self.product, md_down, self.initial_spots)
            dS = bump_pct * self.market_data.spots[i]
            gammas[ticker] = (p_up - 2 * self.base_price + p_down) / (dS**2)
            logger.info(
                "  Gamma[%s]: p_up=%.2f  base=%.2f  p_down=%.2f  => gamma=%+.4f",
                ticker, p_up, self.base_price, p_down, gammas[ticker],
            )
        logger.info("Gamma done in %.3fs", time.perf_counter() - t0)
        return gammas

    # ------------------------------------------------------------------
    # Vega: dV/dσ  (per 1 vol point)
    # ------------------------------------------------------------------
    def vega(self, bump_vol: float = 0.01) -> Dict[str, float]:
        """Vega per 1 percentage-point move in volatility."""
        logger.info("Computing Vega (bump=%.0f bps) ...", bump_vol * 10000)
        t0 = time.perf_counter()
        vegas = {}
        for i, ticker in enumerate(self.product.underlyings):
            md_up = self.market_data.bump_vol(i, bump_vol)
            md_down = self.market_data.bump_vol(i, -bump_vol)
            p_up = _reprice(self.engine, self.product, md_up, self.initial_spots)
            p_down = _reprice(self.engine, self.product, md_down, self.initial_spots)
            vegas[ticker] = (p_up - p_down) / (2 * bump_vol)
            logger.info(
                "  Vega[%s]: p_up=%.2f  p_down=%.2f  => vega=%+.2f",
                ticker, p_up, p_down, vegas[ticker],
            )
        logger.info("Vega done in %.3fs", time.perf_counter() - t0)
        return vegas

    # ------------------------------------------------------------------
    # Correlation sensitivity
    # ------------------------------------------------------------------
    def correlation_sensitivity(
        self, bumps: List[float] | None = None
    ) -> Dict[str, float]:
        """
        Price the product under uniform correlation shocks.
        Returns {bump_label: price_pct}.
        """
        if bumps is None:
            bumps = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]

        logger.info("Computing correlation sensitivity for %d scenarios ...", len(bumps))
        t0 = time.perf_counter()
        results = {}
        for b in bumps:
            md = self.market_data.bump_correlation(b)
            paths = self.engine.simulate(md, self.product.observation_dates)
            res = self.product.price(paths, self.initial_spots, md.risk_free_rate)
            label = f"{b:+.0%}"
            results[label] = res["price_pct"]
            logger.info("  corr bump %s => price=%.2f%%", label, res["price_pct"])
        logger.info("Correlation sensitivity done in %.3fs", time.perf_counter() - t0)
        return results

    def full_report(self) -> Dict:
        """Compute all greeks and sensitivities."""
        return {
            "base_price": self.base_price,
            "base_price_pct": self.base_price / self.product.notional * 100,
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "correlation_sensitivity": self.correlation_sensitivity(),
        }
