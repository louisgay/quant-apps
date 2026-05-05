"""Vectorized Black-Scholes pricing and Greeks with continuous dividend yield.

All functions accept scalar or ndarray inputs via np.asarray + broadcasting.
The sigma parameter can be array-valued (one IV per strike) for skew-aware pricing.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, q, sigma):
    """Compute d1 and d2 with continuous dividend yield q."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Vectorized Black-Scholes price with continuous dividend yield."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    discount_S = S * np.exp(-q * T)
    discount_K = K * np.exp(-r * T)
    if option_type == "call":
        price = discount_S * norm.cdf(d1) - discount_K * norm.cdf(d2)
    else:
        price = discount_K * norm.cdf(-d2) - discount_S * norm.cdf(-d1)
    return np.nan_to_num(np.asarray(price, dtype=float))


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes delta."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.nan_to_num(np.exp(-q * T) * norm.cdf(d1))
    return np.nan_to_num(np.exp(-q * T) * (norm.cdf(d1) - 1))


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Black-Scholes gamma (same for call and put)."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1_d2(S, K, T, r, q, sigma)
        g = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return np.nan_to_num(g)


def bs_theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes theta (per year)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1_d2(S, K, T, r, q, sigma)
        discount_S = S * np.exp(-q * T)
        discount_K = K * np.exp(-r * T)
        first = -discount_S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        theta = first + q * discount_S * norm.cdf(d1) - r * discount_K * norm.cdf(d2)
    else:
        theta = first - q * discount_S * norm.cdf(-d1) + r * discount_K * norm.cdf(-d2)
    return np.nan_to_num(theta)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Black-Scholes vega (per 1 unit of sigma, same for call and put)."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1_d2(S, K, T, r, q, sigma)
        v = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return np.nan_to_num(v)


def bs_rho(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes rho (sensitivity to risk-free rate)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, d2 = _d1_d2(S, K, T, r, q, sigma)
    discount_K = K * np.exp(-r * T)
    if option_type == "call":
        rho = T * discount_K * norm.cdf(d2)
    else:
        rho = -T * discount_K * norm.cdf(-d2)
    return np.nan_to_num(rho)


def bs_vanna(S, K, T, r, sigma, q=0.0):
    """Black-Scholes vanna (d^2V/dS/dsigma). Same for call and put."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1_d2(S, K, T, r, q, sigma)
        vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma
    return np.nan_to_num(vanna)


def bs_volga(S, K, T, r, sigma, q=0.0):
    """Black-Scholes volga / vomma (d^2V/dsigma^2). Same for call and put."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1_d2(S, K, T, r, q, sigma)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        volga = vega * d1 * d2 / sigma
    return np.nan_to_num(volga)
