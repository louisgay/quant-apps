"""Portfolio-level Greeks aggregation.

Computes individual and aggregate Greeks for multi-leg option strategies,
with skew-adjusted IV from the vol smile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pricing import bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho, bs_vanna, bs_volga
from .strategy import Strategy
from .vol_surface import VolSmile


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for a multi-leg position."""

    delta: float
    gamma: float
    theta: float
    theta_daily: float
    vega: float
    rho: float
    vanna: float
    volga: float


def compute_portfolio_greeks(
    strategy: Strategy,
    S: float,
    r: float,
    q: float,
    vol_smile: VolSmile,
    t_elapsed: float = 0.0,
) -> PortfolioGreeks:
    """Compute aggregate Greeks for all legs in a strategy.

    Parameters
    ----------
    strategy : Strategy
        The multi-leg strategy.
    S : float
        Current spot price.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    vol_smile : VolSmile
        Volatility smile for IV lookup.
    t_elapsed : float
        Time elapsed since entry (years).

    Returns
    -------
    PortfolioGreeks
        Aggregated Greeks.
    """
    total_delta = 0.0
    total_gamma = 0.0
    total_theta = 0.0
    total_vega = 0.0
    total_rho = 0.0
    total_vanna = 0.0
    total_volga = 0.0

    for leg in strategy.legs:
        T_remaining = max(leg.T - t_elapsed, 0.0)
        if T_remaining <= 1e-10:
            continue

        sign = 1.0 if leg.direction == "long" else -1.0
        iv = float(vol_smile.get_iv_for_strike(leg.strike, S, T_remaining))
        if iv <= 0:
            continue

        mult = sign * leg.quantity

        total_delta += mult * float(bs_delta(S, leg.strike, T_remaining, r, iv, leg.option_type, q))
        total_gamma += mult * float(bs_gamma(S, leg.strike, T_remaining, r, iv, q))
        total_theta += mult * float(bs_theta(S, leg.strike, T_remaining, r, iv, leg.option_type, q))
        total_vega += mult * float(bs_vega(S, leg.strike, T_remaining, r, iv, q))
        total_rho += mult * float(bs_rho(S, leg.strike, T_remaining, r, iv, leg.option_type, q))
        total_vanna += mult * float(bs_vanna(S, leg.strike, T_remaining, r, iv, q))
        total_volga += mult * float(bs_volga(S, leg.strike, T_remaining, r, iv, q))

    return PortfolioGreeks(
        delta=total_delta,
        gamma=total_gamma,
        theta=total_theta,
        theta_daily=total_theta / 365.0,
        vega=total_vega,
        rho=total_rho,
        vanna=total_vanna,
        volga=total_volga,
    )


def compute_greeks_vs_spot(
    strategy: Strategy,
    S_range: np.ndarray,
    r: float,
    q: float,
    vol_smile: VolSmile,
    t_elapsed: float = 0.0,
) -> dict:
    """Compute Greeks arrays across a range of spot prices.

    Returns
    -------
    dict with keys: S_range, delta, gamma, theta, vega, rho, vanna, volga
        Each value is an ndarray of same length as S_range.
    """
    S_range = np.asarray(S_range, dtype=float)
    n = len(S_range)
    result = {
        "S_range": S_range,
        "delta": np.zeros(n),
        "gamma": np.zeros(n),
        "theta": np.zeros(n),
        "vega": np.zeros(n),
        "rho": np.zeros(n),
        "vanna": np.zeros(n),
        "volga": np.zeros(n),
    }

    for i, S_val in enumerate(S_range):
        g = compute_portfolio_greeks(strategy, S_val, r, q, vol_smile, t_elapsed)
        result["delta"][i] = g.delta
        result["gamma"][i] = g.gamma
        result["theta"][i] = g.theta
        result["vega"][i] = g.vega
        result["rho"][i] = g.rho
        result["vanna"][i] = g.vanna
        result["volga"][i] = g.volga

    return result


def compute_greeks_vs_time(
    strategy: Strategy,
    S: float,
    t_range: np.ndarray,
    r: float,
    q: float,
    vol_smile: VolSmile,
) -> dict:
    """Compute Greeks arrays over time (t_elapsed values).

    Returns
    -------
    dict with keys: t_range, delta, gamma, theta, vega, rho, vanna, volga
        Each value is an ndarray of same length as t_range.
    """
    t_range = np.asarray(t_range, dtype=float)
    n = len(t_range)
    result = {
        "t_range": t_range,
        "delta": np.zeros(n),
        "gamma": np.zeros(n),
        "theta": np.zeros(n),
        "vega": np.zeros(n),
        "rho": np.zeros(n),
        "vanna": np.zeros(n),
        "volga": np.zeros(n),
    }

    for i, t_val in enumerate(t_range):
        g = compute_portfolio_greeks(strategy, S, r, q, vol_smile, t_val)
        result["delta"][i] = g.delta
        result["gamma"][i] = g.gamma
        result["theta"][i] = g.theta
        result["vega"][i] = g.vega
        result["rho"][i] = g.rho
        result["vanna"][i] = g.vanna
        result["volga"][i] = g.volga

    return result
