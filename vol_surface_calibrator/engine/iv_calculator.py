"""Black-Scholes implied volatility via Brent's root-finding algorithm."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-Scholes closed forms
# ---------------------------------------------------------------------------

def bs_call_price(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """European call price under Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """European put price under Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_vega(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes vega (dC/dσ), useful for Newton step."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * np.sqrt(T) * norm.pdf(d1))


# ---------------------------------------------------------------------------
# Implied Volatility
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    sigma_lo: float = 1e-4,
    sigma_hi: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Optional[float]:
    """
    Compute implied volatility using Brent's method.

    Returns None if the root cannot be found (e.g. price below intrinsic
    value or above upper bound).
    """
    if T <= 0 or market_price <= 0:
        return None

    bs_func = bs_call_price if option_type == "call" else bs_put_price

    # Check bounds
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if option_type == "call" \
        else max(K * np.exp(-r * T) - S, 0.0)
    if market_price < intrinsic - tol:
        return None  # below intrinsic → no valid IV

    price_lo = bs_func(S, K, T, r, sigma_lo)
    price_hi = bs_func(S, K, T, r, sigma_hi)

    if market_price < price_lo or market_price > price_hi:
        return None

    try:
        iv = brentq(
            lambda sigma: bs_func(S, K, T, r, sigma) - market_price,
            sigma_lo,
            sigma_hi,
            xtol=tol,
            maxiter=max_iter,
        )
        return float(iv)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Vectorized IV computation for a full chain
# ---------------------------------------------------------------------------

def compute_iv_chain(
    df: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """
    Add an 'iv' column to a DataFrame containing option chain data.

    Expected columns: strike, T, mid_price, option_type.
    Rows where IV cannot be computed are dropped.
    """
    logger.info("Computing IV for %d options (S=%.2f, r=%.4f) ...",
                len(df), spot, risk_free_rate)

    ivs: list[Optional[float]] = []
    for _, row in df.iterrows():
        iv = implied_volatility(
            market_price=row["mid_price"],
            S=spot,
            K=row["strike"],
            T=row["T"],
            r=risk_free_rate,
            option_type=row["option_type"],
        )
        ivs.append(iv)

    result = df.copy()
    result["iv"] = ivs
    n_before = len(result)
    result = result.dropna(subset=["iv"]).reset_index(drop=True)

    # Remove extreme outliers (IV > 300% or < 1%)
    result = result[(result["iv"] > 0.01) & (result["iv"] < 3.0)].reset_index(drop=True)

    logger.info("IV computed: %d / %d options valid", len(result), n_before)

    # Derived fields for SVI
    result["log_moneyness"] = np.log(result["strike"] / result["forward"])
    result["total_var"] = result["iv"] ** 2 * result["T"]

    return result
