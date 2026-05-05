"""SVI volatility smile with user-friendly constructor.

Raw SVI parameterisation (Gatheral 2004):
    w(k) = a + b * [ rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) ]

where k = ln(K/S) is log-moneyness and w = IV^2 * T is total implied variance.

This module provides a VolSmile that can be constructed from 3 intuitive sliders
(atm_vol, skew, curvature) or from raw SVI parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Reference maturity for SVI parameterization (1 year)
T_REF = 1.0


@dataclass
class VolSmile:
    """SVI volatility smile for a single reference maturity.

    Attributes
    ----------
    a, b, rho, m, sigma_svi : float
        Raw SVI parameters.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma_svi: float

    @classmethod
    def from_simple(cls, atm_vol: float, skew: float = 0.0, curvature: float = 0.0) -> VolSmile:
        """Create a smile from 3 intuitive parameters.

        Parameters
        ----------
        atm_vol : float
            At-the-money implied volatility (e.g. 0.20 for 20%).
        skew : float
            Skew strength (0 = symmetric, positive = downside skew). Range ~[0, 1].
        curvature : float
            Smile curvature / wing steepness. Range ~[0, 2].
        """
        m = 0.0
        sigma_svi = 0.1
        rho = -skew
        b = curvature * 0.5
        a = atm_vol ** 2 * T_REF - b * sigma_svi

        # No-arbitrage floor: a >= -b * sigma_svi * sqrt(1 - rho^2)
        arb_floor = -b * sigma_svi * np.sqrt(1 - rho ** 2)
        a = max(a, arb_floor)

        return cls(a=a, b=b, rho=rho, m=m, sigma_svi=sigma_svi)

    @classmethod
    def from_raw_svi(cls, a: float, b: float, rho: float, m: float, sigma_svi: float) -> VolSmile:
        """Create from raw SVI parameters (advanced override)."""
        return cls(a=a, b=b, rho=rho, m=m, sigma_svi=sigma_svi)

    @classmethod
    def flat(cls, atm_vol: float) -> VolSmile:
        """Create a flat smile (pure BS, no skew or curvature)."""
        return cls.from_simple(atm_vol, skew=0.0, curvature=0.0)

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Compute w(k) for log-moneyness array k at reference maturity."""
        k = np.asarray(k, dtype=float)
        dk = k - self.m
        w = self.a + self.b * (self.rho * dk + np.sqrt(dk ** 2 + self.sigma_svi ** 2))
        return np.maximum(w, 0.0)

    def get_iv_for_strike(self, K, S, T) -> np.ndarray:
        """Get implied volatility for given strikes, spot, and time to expiry.

        Uses calendar-safe scaling: w(k, T) = w_ref(k) * T / T_ref
        so that total variance is linear in T.

        Parameters
        ----------
        K : scalar or array
            Strike prices.
        S : float
            Current spot price.
        T : float
            Time to expiry in years.

        Returns
        -------
        iv : ndarray
            Implied volatilities for each strike.
        """
        K = np.asarray(K, dtype=float)
        T = float(T)
        if T <= 0:
            return np.full_like(K, 0.0)

        k = np.log(K / S)  # log-moneyness
        w_ref = self.total_variance(k)  # total variance at T_ref
        w_T = w_ref * T / T_REF  # scale to actual maturity
        iv = np.sqrt(np.maximum(w_T, 0.0) / T)
        return iv

    def is_arbitrage_free(self) -> bool:
        """Check the basic no-negative-variance condition."""
        return self.a + self.b * self.sigma_svi * np.sqrt(1 - self.rho ** 2) >= -1e-10
