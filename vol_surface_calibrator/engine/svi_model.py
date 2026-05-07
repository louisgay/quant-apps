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

    # Skip expiries shorter than this (must match data_fetcher.min_fetch_T)
    MIN_T_DEFAULT = 1 / 365

    def __init__(
        self,
        use_global_optimizer: bool = False,
        min_T: float = MIN_T_DEFAULT,
    ) -> None:
        self.use_global = use_global_optimizer
        self.min_T = min_T

    @staticmethod
    def _svi_w(params: np.ndarray, k: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = params
        dk = k - m
        return a + b * (rho * dk + np.sqrt(dk**2 + sigma**2))

    @staticmethod
    def _objective(params: np.ndarray, k: np.ndarray, w_market: np.ndarray) -> float:
        w_model = SVICalibrator._svi_w(params, k)
        return float(np.mean((w_model - w_market) ** 2))

    @staticmethod
    def _objective_iv(
        params: np.ndarray, k: np.ndarray, w_market: np.ndarray, T: float,
    ) -> float:
        """MSE in implied-volatility space (better conditioned than total var)."""
        w_model = SVICalibrator._svi_w(params, k)
        iv_model = np.sqrt(np.maximum(w_model, 1e-12) / T)
        iv_market = np.sqrt(np.maximum(w_market, 1e-12) / T)
        return float(np.mean((iv_model - iv_market) ** 2))

    @staticmethod
    def _estimate_initial_params(
        k: np.ndarray, w_market: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """Derive a data-driven initial guess for (a, b, rho, m, sigma)."""
        order = np.argsort(k)
        k_s, w_s = k[order], w_market[order]

        w_atm = float(np.interp(0.0, k_s, w_s))

        k_mid = float(np.median(k_s))
        left_mask = k_s < k_mid - 0.03
        right_mask = k_s > k_mid + 0.03

        # Left wing: asymptotic slope magnitude ≈ b*(1 − ρ)
        if left_mask.sum() >= 2:
            lm_slope = abs(np.polyfit(k_s[left_mask], w_s[left_mask], 1)[0])
        else:
            lm_slope = 0.1

        # Right wing: asymptotic slope ≈ b*(1 + ρ)
        if right_mask.sum() >= 2:
            rm_slope = max(np.polyfit(k_s[right_mask], w_s[right_mask], 1)[0], 1e-4)
        else:
            rm_slope = max(lm_slope * 0.3, 0.01)

        b_est = max((lm_slope + rm_slope) / 2, 0.01)
        denom = lm_slope + rm_slope
        rho_est = float(np.clip(
            (rm_slope - lm_slope) / denom if denom > 1e-10 else -0.4,
            -0.95, 0.95,
        ))

        return w_atm, b_est, rho_est, 0.0, 0.1

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
            atm_var = float(np.median(w_market))
            return SVISlice(a=atm_var, b=1e-6, rho=0.0, m=0.0,
                            sigma=0.01, T=T, expiry=expiry, rmse=0.0)

        # Sort by log-moneyness so np.interp works correctly
        order = np.argsort(k)
        k = k[order]
        w_market = w_market[order]

        # ------------------------------------------------------------------
        # Data-driven initial guess
        # ------------------------------------------------------------------
        a0, b0, rho0, m0, sig0 = self._estimate_initial_params(k, w_market)

        k_range = k.max() - k.min()
        w_max = float(w_market.max())

        bounds = [
            (-w_max, w_max * 3),                     # a
            (1e-4, max(5.0 * b0, 1.0)),              # b — data-adaptive
            (-0.999, 0.999),                          # rho
            (k.min() - 0.1, k.max() + 0.1),          # m — within data range
            (1e-4, max(k_range * 0.8, 0.3)),          # sigma — tighter
        ]

        constraint = {
            "type": "ineq",
            "fun": lambda p: p[0] + p[1] * p[4] * np.sqrt(1 - p[2]**2),
        }

        def obj_fn(p: np.ndarray) -> float:
            return self._objective_iv(p, k, w_market, T)

        def _clip_x0(x0: np.ndarray) -> np.ndarray:
            return np.array([
                np.clip(x0[j], bounds[j][0] + 1e-6, bounds[j][1] - 1e-6)
                for j in range(5)
            ])

        # ------------------------------------------------------------------
        # Multi-start SLSQP
        # ------------------------------------------------------------------
        best_res = None
        x0_candidates = [
            np.array([a0, b0, rho0, m0, sig0]),
            np.array([a0 * 0.5, b0 * 0.5, max(rho0 - 0.2, -0.95), m0, sig0 * 2]),
            np.array([a0 * 1.5, b0 * 2.0, min(rho0 + 0.2, 0.95), m0, sig0 * 0.5]),
        ]

        if not self.use_global:
            for x0 in x0_candidates:
                try:
                    res = minimize(
                        obj_fn, _clip_x0(x0),
                        method="SLSQP", bounds=bounds,
                        constraints=constraint,
                        options={"maxiter": 500, "ftol": 1e-12},
                    )
                    if best_res is None or res.fun < best_res.fun:
                        best_res = res
                except Exception:
                    continue

        # DE fallback / primary (if use_global or SLSQP result is poor)
        iv_rmse_thresh = 0.005 ** 2  # 0.5 % IV RMSE
        if best_res is None or best_res.fun > iv_rmse_thresh or self.use_global:
            if best_res is not None:
                logger.debug("SLSQP suboptimal for %s (loss=%.2e), trying DE ...",
                             expiry, best_res.fun)
            try:
                res_de = differential_evolution(
                    obj_fn, bounds=bounds,
                    seed=42, maxiter=1000, tol=1e-12,
                )
                if best_res is None or res_de.fun < best_res.fun:
                    best_res = res_de
            except Exception:
                pass

        if best_res is None:
            atm_var = float(np.median(w_market))
            return SVISlice(a=atm_var, b=1e-6, rho=0.0, m=0.0,
                            sigma=0.01, T=T, expiry=expiry, rmse=0.0)

        a, b, rho, m, sigma = best_res.x

        # IV-based RMSE (interpretable: e.g. 0.005 = 0.5 % IV)
        w_fit = self._svi_w(best_res.x, k)
        iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
        iv_mkt = np.sqrt(np.maximum(w_market, 0.0) / T)
        rmse = float(np.sqrt(np.mean((iv_fit - iv_mkt) ** 2)))

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

        Expiries with T < min_T are skipped (unreliable near-expiry smiles).
        """
        n_before = df["expiry"].nunique()

        # Filter out very short-dated expiries
        df_cal = df[df["T"] >= self.min_T]
        n_after = df_cal["expiry"].nunique()
        if n_after < n_before:
            logger.info("Skipped %d short-dated expiries (T < %.4f)",
                        n_before - n_after, self.min_T)
        if df_cal.empty:
            logger.warning("All expiries filtered out; falling back to unfiltered data")
            df_cal = df

        logger.info("Calibrating SVI surface for %s (%d expiries) ...",
                     ticker, df_cal["expiry"].nunique())

        slices: List[SVISlice] = []
        for expiry, grp in df_cal.groupby("expiry"):
            k = grp["log_moneyness"].values
            w = grp["total_var"].values
            T = float(grp["T"].iloc[0])

            slc = self.calibrate_slice(k, w, T, str(expiry))
            slices.append(slc)

        surface = SVISurface(slices=slices, ticker=ticker)
        logger.info("Surface calibrated: %d slices", len(slices))
        return surface
