"""Black-Scholes analytical pricing and Greeks."""

import numpy as np
from scipy.stats import norm

from .models import SurfaceData


def bs_price(S, K, T, r, sigma, option_type="call"):
    """Vectorized Black-Scholes price (scalar or ndarray)."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.nan_to_num(np.asarray(price, dtype=float))


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes delta."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return np.nan_to_num(norm.cdf(d1))
    return np.nan_to_num(norm.cdf(d1) - 1)


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma (same for call and put)."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        g = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return np.nan_to_num(g)


def bs_theta(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes theta (per year)."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        first = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        theta = first - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = first + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return np.nan_to_num(theta)


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (per 1 unit of sigma, same for call and put)."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        v = S * norm.pdf(d1) * np.sqrt(T)
    return np.nan_to_num(v)


def compute_price_surface(K, r, sigma, option_type="put",
                          S_min=50.0, S_max=150.0, n_S=100,
                          T_min=0.01, T_max=1.0, n_T=100):
    """Compute a 3D BS price surface over (S, T) grid."""
    S_values = np.linspace(S_min, S_max, n_S)
    T_values = np.linspace(T_min, T_max, n_T)
    S_grid, T_grid = np.meshgrid(S_values, T_values)
    V_surface = bs_price(S_grid, K, T_grid, r, sigma, option_type)
    return SurfaceData(S_values=S_values, T_values=T_values, V_surface=V_surface)
