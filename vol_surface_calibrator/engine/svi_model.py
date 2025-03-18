"""SVI (Stochastic Volatility Inspired) smile and surface calibration.

Raw SVI parameterisation (Gatheral 2004):

    w(k) = a + b * [ rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) ]

where:
    k     = ln(K / F)          log-moneyness
    w     = sigma_impl^2 * T   total implied variance
    a     : vertical shift
    b > 0 : overall slope / wings level
    |rho| < 1 : rotation (skew direction)
    m     : horizontal translation
    sigma > 0 : ATM curvature

No-arbitrage condition at the slice level:
    a + b * sigma * sqrt(1 - rho^2) >= 0   (ensures w >= 0 everywhere)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-slice SVI
# ---------------------------------------------------------------------------

@dataclass
class SVISlice:
    """Calibrated SVI parameters for one maturity."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    expiry: str
    rmse: float = 0.0

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Compute w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
        k = np.asarray(k, dtype=np.float64)
        dk = k - self.m
        return self.a + self.b * (self.rho * dk + np.sqrt(dk**2 + self.sigma**2))

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        """Convert total variance to implied volatility: sigma = sqrt(w / T)."""
        w = self.total_variance(k)
        return np.sqrt(np.maximum(w, 0.0) / self.T)

    def is_arbitrage_free(self) -> bool:
        """Check the basic Gatheral no-negative-variance condition."""
        return self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2) >= -1e-10

    def to_dict(self) -> Dict[str, float]:
        return {
            "a": self.a, "b": self.b, "rho": self.rho,
            "m": self.m, "sigma": self.sigma,
            "T": self.T, "expiry": self.expiry, "rmse": self.rmse,
        }


# ---------------------------------------------------------------------------
# SVI Surface (collection of slices)
# ---------------------------------------------------------------------------

@dataclass
class SVISurface:
    """Calibrated vol surface made of per-maturity SVI slices."""

    slices: List[SVISlice]
    ticker: str = ""

    def get_slice(self, expiry: str) -> Optional[SVISlice]:
        for s in self.slices:
            if s.expiry == expiry:
                return s
        return None

    @property
    def expiries(self) -> List[str]:
        return [s.expiry for s in self.slices]

    def implied_vol_grid(
        self,
        k_grid: np.ndarray,
        T_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate IV on a (k, T) meshgrid by interpolating between slices.

        Returns shape (len(T_grid), len(k_grid)).
        """
        # Build total-variance spline across T for each k
        slice_Ts = np.array([s.T for s in self.slices])
        order = np.argsort(slice_Ts)
        sorted_slices = [self.slices[i] for i in order]
        sorted_Ts = slice_Ts[order]

        iv_surface = np.full((len(T_grid), len(k_grid)), np.nan)

        for j, T_val in enumerate(T_grid):
            # Linear interpolation of total variance in T (calendar arbitrage-safe)
            if T_val <= sorted_Ts[0]:
                w_row = sorted_slices[0].total_variance(k_grid)
                scale = T_val / sorted_Ts[0]
                w_row = w_row * scale
            elif T_val >= sorted_Ts[-1]:
                w_row = sorted_slices[-1].total_variance(k_grid)
                scale = T_val / sorted_Ts[-1]
                w_row = w_row * scale
            else:
                idx = np.searchsorted(sorted_Ts, T_val) - 1
                T_lo, T_hi = sorted_Ts[idx], sorted_Ts[idx + 1]
                w_lo = sorted_slices[idx].total_variance(k_grid)
                w_hi = sorted_slices[idx + 1].total_variance(k_grid)
                alpha = (T_val - T_lo) / (T_hi - T_lo)
                w_row = (1 - alpha) * w_lo + alpha * w_hi

            iv_surface[j, :] = np.sqrt(np.maximum(w_row, 0.0) / T_val)

        return iv_surface

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.slices])


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class SVICalibrator:
    """Calibrate raw SVI to market implied volatilities."""

    def __init__(self, use_global_optimizer: bool = False) -> None:
        self.use_global = use_global_optimizer

    @staticmethod
    def _svi_w(params: np.ndarray, k: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = params
        dk = k - m
        return a + b * (rho * dk + np.sqrt(dk**2 + sigma**2))

    @staticmethod
    def _objective(params: np.ndarray, k: np.ndarray, w_market: np.ndarray) -> float:
        w_model = SVICalibrator._svi_w(params, k)
        return float(np.mean((w_model - w_market) ** 2))

    def calibrate_slice(
        self,
        k: np.ndarray,
        w_market: np.ndarray,
        T: float,
        expiry: str,
    ) -> SVISlice:
        """
        Calibrate SVI to a single maturity slice.

        Parameters
        ----------
        k : array of log-moneyness values
        w_market : array of market total implied variances
        T : time to expiry (years)
        expiry : expiry label
        """
        k = np.asarray(k, dtype=np.float64)
        w_market = np.asarray(w_market, dtype=np.float64)

        if len(k) < 5:
            logger.warning("Slice %s has only %d points, skipping", expiry, len(k))
            # Return a flat slice
            atm_var = float(np.median(w_market))
            return SVISlice(a=atm_var, b=1e-6, rho=0.0, m=0.0,
                            sigma=0.01, T=T, expiry=expiry, rmse=0.0)

        # Initial guess: flat vol + slight skew
        w_atm = float(np.interp(0.0, k, w_market)) if len(k) > 1 else w_market.mean()
        x0 = np.array([w_atm, 0.1, -0.3, 0.0, 0.1])

        bounds = [
            (-0.5, max(w_market) * 3),   # a
            (1e-4, 3.0),                  # b
            (-0.999, 0.999),              # rho
            (k.min() - 0.5, k.max() + 0.5),  # m
            (1e-4, 3.0),                  # sigma
        ]

        # No-arbitrage constraint: a + b*sigma*sqrt(1-rho^2) >= 0
        constraint = {
            "type": "ineq",
            "fun": lambda p: p[0] + p[1] * p[4] * np.sqrt(1 - p[2]**2),
        }

        if self.use_global:
            res = differential_evolution(
                self._objective, bounds=bounds,
                args=(k, w_market), seed=42,
                maxiter=500, tol=1e-10,
            )
        else:
            res = minimize(
                self._objective, x0, args=(k, w_market),
                method="SLSQP", bounds=bounds,
                constraints=constraint,
                options={"maxiter": 500, "ftol": 1e-12},
            )
            # SLSQP alone wasn't enough — short-dated smiles have too many local minima
            if not res.success or res.fun > 1e-4:
                logger.debug("SLSQP suboptimal for %s (loss=%.2e), trying DE...",
                             expiry, res.fun)
                res2 = differential_evolution(
                    self._objective, bounds=bounds,
                    args=(k, w_market), seed=42,
                    maxiter=500, tol=1e-10,
                )
                if res2.fun < res.fun:
                    res = res2

        a, b, rho, m, sigma = res.x
        rmse = float(np.sqrt(res.fun))

        slc = SVISlice(a=a, b=b, rho=rho, m=m, sigma=sigma,
                       T=T, expiry=expiry, rmse=rmse)

        logger.info(
            "Slice %s (T=%.3f): a=%.4f b=%.4f rho=%.3f m=%.4f sigma=%.4f | "
            "RMSE=%.6f | arb_free=%s",
            expiry, T, a, b, rho, m, sigma, rmse, slc.is_arbitrage_free(),
        )
        return slc

    def calibrate_surface(self, df: pd.DataFrame, ticker: str = "") -> SVISurface:
        """
        Calibrate an SVI surface from a DataFrame with columns:
        expiry, T, log_moneyness, total_var.
        """
        logger.info("Calibrating SVI surface for %s (%d expiries) ...",
                     ticker, df["expiry"].nunique())

        slices: List[SVISlice] = []
        for expiry, grp in df.groupby("expiry"):
            k = grp["log_moneyness"].values
            w = grp["total_var"].values
            T = float(grp["T"].iloc[0])

            slc = self.calibrate_slice(k, w, T, str(expiry))
            slices.append(slc)

        surface = SVISurface(slices=slices, ticker=ticker)
        logger.info("Surface calibrated: %d slices", len(slices))
        return surface
