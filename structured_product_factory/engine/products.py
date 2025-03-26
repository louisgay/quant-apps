"""Autocallable product definitions and pricing logic."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

FREQ_MAP = {"monthly": 12, "quarterly": 4, "semi-annual": 2, "annual": 1}


@dataclass
class ComponentResult:
    """Aggregated MC result for a single payoff component."""

    mean: float
    std_error: float

    @property
    def pct(self) -> float:
        """Value expressed in % of notional (set after construction)."""
        return self._pct

    @pct.setter
    def pct(self, value: float) -> None:
        self._pct = value

    @property
    def std_error_pct(self) -> float:
        return self._std_error_pct

    @std_error_pct.setter
    def std_error_pct(self, value: float) -> None:
        self._std_error_pct = value

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std_error": self.std_error,
            "pct": self.pct,
            "std_error_pct": self.std_error_pct,
        }


def _aggregate(arr: np.ndarray, notional: float) -> ComponentResult:
    """Compute mean / std_error / pct from a per-path array."""
    n = len(arr)
    cr = ComponentResult(mean=float(arr.mean()), std_error=float(arr.std() / np.sqrt(n)))
    cr.pct = cr.mean / notional * 100
    cr.std_error_pct = cr.std_error / notional * 100
    return cr


@dataclass
class AutocallableBase:
    """
    Base class for worst-of autocallable notes on a basket of equities.

    Provides shared fields and properties for observation scheduling.
    Subclasses must implement their own ``price()`` method.
    """

    underlyings: List[str]
    notional: float = 1_000_000.0
    maturity_years: float = 3.0
    observation_frequency: str = "quarterly"
    autocall_barrier: float = 1.00   # 100% of initial
    coupon_rate: float = 0.08        # 8% p.a.
    put_barrier: float = 0.60        # 60%

    @property
    def n_obs_per_year(self) -> int:
        return FREQ_MAP[self.observation_frequency]

    @property
    def observation_dates(self) -> np.ndarray:
        """Year-fraction observation schedule."""
        n_obs = int(self.maturity_years * self.n_obs_per_year)
        return np.array([(i + 1) / self.n_obs_per_year for i in range(n_obs)])

    @property
    def coupon_per_period(self) -> float:
        return self.coupon_rate / self.n_obs_per_year


@dataclass
class AutocallablePhoenix(AutocallableBase):
    """
    Worst-of Autocallable Phoenix Note on a basket of equities.

    Cash-flow rules (at each observation date, checked on worst performer):
      1. **Autocall**: if worst-of >= autocall_barrier -> redeem at par + coupon.
      2. **Coupon**: if worst-of >= coupon_barrier -> pay periodic coupon
         (with optional memory: catch-up on missed coupons).
      3. **No coupon**: otherwise, coupon is missed (stored if memory=True).

    At maturity (if not autocalled):
      - worst-of >= put_barrier -> redeem at par.
      - worst-of <  put_barrier -> redeem at par x worst-of performance (capital loss).
    """

    coupon_barrier: float = 0.70     # 70%
    memory_feature: bool = True

    # ------------------------------------------------------------------
    # Core pricing with full decomposition
    # ------------------------------------------------------------------
    def price(
        self,
        paths: np.ndarray,
        initial_spots: np.ndarray,
        risk_free_rate: float,
    ) -> Dict:
        """
        Price the product and return a component-level decomposition.

        The total product value is broken down into four additive bricks:

            Full Price = ZCB + Coupons + Short Put

        where ZCB itself decomposes as:

            ZCB = ZCB_to_maturity + Autocall Option Value

        so equivalently:

            Full Price = ZCB_to_maturity + Autocall Option + Coupons + Put

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_assets, n_dates)
        initial_spots : np.ndarray, shape (n_assets,)
        risk_free_rate : float

        Returns
        -------
        dict
            Top-level pricing results (price, std_error, price_pct, ...)
            plus a ``"components"`` sub-dict with per-brick statistics.
        """
        t0 = time.perf_counter()
        obs_dates = self.observation_dates
        n_paths, n_assets, n_dates = paths.shape
        assert n_dates == len(obs_dates)

        logger.debug(
            "Pricing %d paths | initial_spots=%s | obs_dates=%d",
            n_paths, initial_spots, n_dates,
        )

        # Relative performance (vs initial fixing)
        perf = paths / initial_spots[np.newaxis, :, np.newaxis]
        worst = perf.min(axis=1)  # (n_paths, n_dates)

        N = self.notional
        coupon_amt = self.coupon_per_period * N
        df_T = np.exp(-risk_free_rate * obs_dates[-1])

        # ---- Per-path component accumulators ----
        pv_zcb = np.zeros(n_paths)         # Capital redemption (at exit time)
        pv_coupons = np.zeros(n_paths)     # All digital coupon payments
        pv_put = np.zeros(n_paths)         # Capital loss below put barrier

        called = np.zeros(n_paths, dtype=bool)
        missed = np.zeros(n_paths)
        life = np.full(n_paths, obs_dates[-1])

        # ---- Observation-by-observation loop ----
        for t in range(n_dates):
            active = ~called
            df = np.exp(-risk_free_rate * obs_dates[t])

            # --- Digital Coupons ---
            earns_coupon = active & (worst[:, t] >= self.coupon_barrier)
            no_coupon = active & (worst[:, t] < self.coupon_barrier)

            if self.memory_feature:
                pv_coupons[earns_coupon] += (
                    (1.0 + missed[earns_coupon]) * coupon_amt * df
                )
                missed[earns_coupon] = 0.0
                missed[no_coupon] += 1.0
            else:
                pv_coupons[earns_coupon] += coupon_amt * df

            # --- Autocall (ZCB at early exit) ---
            ac = active & (worst[:, t] >= self.autocall_barrier)
            pv_zcb[ac] = N * df          # capital back at t
            life[ac] = obs_dates[t]
            called[ac] = True

        # ---- Maturity: paths still alive ----
        still_active = ~called

        # ZCB at maturity for non-autocalled paths
        pv_zcb[still_active] = N * df_T

        # Short Put: loss if worst-of breaches put barrier
        loss_mask = still_active & (worst[:, -1] < self.put_barrier)
        pv_put[loss_mask] = N * (worst[loss_mask, -1] - 1.0) * df_T  # negative

        # ---- Autocall Option = ZCB_actual - ZCB_to_maturity ----
        zcb_maturity_scalar = N * df_T                       # deterministic bond floor
        pv_autocall_option = pv_zcb - zcb_maturity_scalar    # >= 0 for autocalled

        # ---- Full price = ZCB + Coupons + Put ----
        pv_full = pv_zcb + pv_coupons + pv_put

        # ---- Aggregate statistics ----
        comp_zcb = _aggregate(pv_zcb, N)
        comp_coupons = _aggregate(pv_coupons, N)
        comp_put = _aggregate(pv_put, N)
        comp_ac_option = _aggregate(pv_autocall_option, N)
        comp_full = _aggregate(pv_full, N)

        # Verification: sum of 3 bricks == full price
        decomp_check = abs(
            comp_zcb.mean + comp_coupons.mean + comp_put.mean - comp_full.mean
        )

        elapsed = time.perf_counter() - t0

        logger.info(
            "Pricing done in %.3fs | full=%.2f%% | ZCB=%.2f%% | Coupons=%.2f%% "
            "| Put=%.2f%% | AC_option=%.2f%% | check=%.2e",
            elapsed,
            comp_full.pct, comp_zcb.pct, comp_coupons.pct,
            comp_put.pct, comp_ac_option.pct, decomp_check,
        )

        return {
            # --- Legacy top-level keys (backward-compatible) ---
            "price": comp_full.mean,
            "std_error": comp_full.std_error,
            "price_pct": comp_full.pct,
            "autocall_prob": float(called.mean()),
            "avg_life_years": float(life.mean()),
            "coupon_barrier_breach_prob": float(
                (worst < self.coupon_barrier).any(axis=1).mean()
            ),
            "capital_loss_prob": float(
                (~called & (worst[:, -1] < self.put_barrier)).mean()
            ),
            # --- Component decomposition ---
            "components": {
                "zcb": comp_zcb.to_dict(),
                "coupons": comp_coupons.to_dict(),
                "short_put": comp_put.to_dict(),
                "autocall_option": comp_ac_option.to_dict(),
            },
            "decomposition_check": decomp_check,
        }


@dataclass
class AutocallableAthena(AutocallableBase):
    """
    Worst-of Autocallable Athena Note on a basket of equities.

    Cash-flow rules (at each observation date, checked on worst performer):
      1. **Autocall**: if worst-of >= autocall_barrier -> redeem at par
         + accrued coupon (coupon_rate x time elapsed).
      2. **No coupon** paid at intermediate observations if no autocall.

    At maturity (if not autocalled):
      - worst-of >= put_barrier -> redeem at par (no coupon).
      - worst-of <  put_barrier -> redeem at par x worst-of performance (capital loss).
    """

    # ------------------------------------------------------------------
    # Core pricing with full decomposition
    # ------------------------------------------------------------------
    def price(
        self,
        paths: np.ndarray,
        initial_spots: np.ndarray,
        risk_free_rate: float,
    ) -> Dict:
        """
        Price the Athena product with component-level decomposition.

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_assets, n_dates)
        initial_spots : np.ndarray, shape (n_assets,)
        risk_free_rate : float

        Returns
        -------
        dict
        """
        t0 = time.perf_counter()
        obs_dates = self.observation_dates
        n_paths, n_assets, n_dates = paths.shape
        assert n_dates == len(obs_dates)

        logger.debug(
            "Pricing Athena %d paths | initial_spots=%s | obs_dates=%d",
            n_paths, initial_spots, n_dates,
        )

        # Relative performance (vs initial fixing)
        perf = paths / initial_spots[np.newaxis, :, np.newaxis]
        worst = perf.min(axis=1)  # (n_paths, n_dates)

        N = self.notional
        df_T = np.exp(-risk_free_rate * obs_dates[-1])

        # ---- Per-path component accumulators ----
        pv_zcb = np.zeros(n_paths)         # Capital redemption (at exit time)
        pv_coupons = np.zeros(n_paths)     # Athena coupon (paid only at autocall)
        pv_put = np.zeros(n_paths)         # Capital loss below put barrier

        called = np.zeros(n_paths, dtype=bool)
        life = np.full(n_paths, obs_dates[-1])

        # ---- Observation-by-observation loop ----
        for t in range(n_dates):
            active = ~called
            df = np.exp(-risk_free_rate * obs_dates[t])

            # --- Autocall check ---
            ac = active & (worst[:, t] >= self.autocall_barrier)

            # Capital back at t
            pv_zcb[ac] = N * df

            # Athena coupon: coupon_rate x time elapsed, paid only at autocall
            pv_coupons[ac] = self.coupon_rate * obs_dates[t] * N * df

            life[ac] = obs_dates[t]
            called[ac] = True

        # ---- Maturity: paths still alive ----
        still_active = ~called

        # ZCB at maturity for non-autocalled paths (no coupon)
        pv_zcb[still_active] = N * df_T

        # Short Put: loss if worst-of breaches put barrier
        loss_mask = still_active & (worst[:, -1] < self.put_barrier)
        pv_put[loss_mask] = N * (worst[loss_mask, -1] - 1.0) * df_T  # negative

        # ---- Autocall Option = ZCB_actual - ZCB_to_maturity ----
        zcb_maturity_scalar = N * df_T
        pv_autocall_option = pv_zcb - zcb_maturity_scalar

        # ---- Full price = ZCB + Coupons + Put ----
        pv_full = pv_zcb + pv_coupons + pv_put

        # ---- Aggregate statistics ----
        comp_zcb = _aggregate(pv_zcb, N)
        comp_coupons = _aggregate(pv_coupons, N)
        comp_put = _aggregate(pv_put, N)
        comp_ac_option = _aggregate(pv_autocall_option, N)
        comp_full = _aggregate(pv_full, N)

        # Verification: sum of 3 bricks == full price
        decomp_check = abs(
            comp_zcb.mean + comp_coupons.mean + comp_put.mean - comp_full.mean
        )

        elapsed = time.perf_counter() - t0

        logger.info(
            "Athena pricing done in %.3fs | full=%.2f%% | ZCB=%.2f%% | Coupons=%.2f%% "
            "| Put=%.2f%% | AC_option=%.2f%% | check=%.2e",
            elapsed,
            comp_full.pct, comp_zcb.pct, comp_coupons.pct,
            comp_put.pct, comp_ac_option.pct, decomp_check,
        )

        return {
            "price": comp_full.mean,
            "std_error": comp_full.std_error,
            "price_pct": comp_full.pct,
            "autocall_prob": float(called.mean()),
            "avg_life_years": float(life.mean()),
            "capital_loss_prob": float(
                (~called & (worst[:, -1] < self.put_barrier)).mean()
            ),
            # --- Component decomposition ---
            "components": {
                "zcb": comp_zcb.to_dict(),
                "coupons": comp_coupons.to_dict(),
                "short_put": comp_put.to_dict(),
                "autocall_option": comp_ac_option.to_dict(),
            },
            "decomposition_check": decomp_check,
        }
