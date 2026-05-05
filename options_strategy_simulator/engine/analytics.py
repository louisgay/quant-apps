"""Chart data structures and grid computations.

Produces payoff diagrams, P&L surfaces, and Greeks grids for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .greeks import compute_greeks_vs_spot, compute_greeks_vs_time, compute_portfolio_greeks
from .strategy import Strategy
from .vol_surface import VolSmile


@dataclass
class PayoffResult:
    """Payoff diagram data."""

    S_range: np.ndarray
    payoff: np.ndarray
    pnl_at_expiry: np.ndarray
    entry_cost: float
    max_profit: float
    max_loss: float
    breakeven_points: np.ndarray


@dataclass
class PnLSurface:
    """P&L surface over spot and time."""

    S_range: np.ndarray
    t_range: np.ndarray
    pnl_grid: np.ndarray  # shape (n_t, n_S)


@dataclass
class GreeksGrid:
    """Greeks over a single axis (spot or time)."""

    x_values: np.ndarray
    x_label: str
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    vega: np.ndarray
    rho: np.ndarray


def find_breakevens(S_range: np.ndarray, pnl: np.ndarray) -> np.ndarray:
    """Find breakeven points via sign-change detection + linear interpolation.

    Parameters
    ----------
    S_range : ndarray
        Spot prices.
    pnl : ndarray
        P&L values corresponding to S_range.

    Returns
    -------
    breakevens : ndarray
        Interpolated spot prices where P&L crosses zero.
    """
    breakevens = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i + 1] < 0:
            # Linear interpolation
            frac = abs(pnl[i]) / (abs(pnl[i]) + abs(pnl[i + 1]))
            be = S_range[i] + frac * (S_range[i + 1] - S_range[i])
            breakevens.append(be)
    return np.array(breakevens)


def compute_payoff_diagram(
    strategy: Strategy,
    S_current: float,
    r: float,
    q: float,
    smile: VolSmile,
    S_min: float | None = None,
    S_max: float | None = None,
    n_points: int = 200,
) -> PayoffResult:
    """Compute payoff/P&L at expiry for a strategy.

    Parameters
    ----------
    strategy : Strategy
        The option strategy.
    S_current : float
        Current spot price.
    r, q : float
        Risk-free rate and dividend yield.
    smile : VolSmile
        Volatility smile.
    S_min, S_max : float, optional
        Spot range. Defaults to 0.5*S to 1.5*S.
    n_points : int
        Number of grid points.

    Returns
    -------
    PayoffResult
    """
    if S_min is None:
        S_min = S_current * 0.5
    if S_max is None:
        S_max = S_current * 1.5

    S_range = np.linspace(S_min, S_max, n_points)
    entry_cost = strategy.compute_entry_cost(S_current, r, q, smile)

    if strategy._is_single_expiry():
        # Single-expiry: use intrinsic payoff (original behavior)
        payoff = strategy.compute_payoff(S_range)
        pnl_at_expiry = payoff - entry_cost
    else:
        # Multi-expiry: evaluate at the near-term expiry using compute_pnl.
        # This gives intrinsic for near-term expired legs and BS value for
        # far-term live legs, which is the meaningful "payoff diagram" for
        # calendar/diagonal strategies.
        min_T = min(leg.T for leg in strategy.legs)
        pnl_at_expiry = strategy.compute_pnl(
            S_range, min_T, S_current, r, q, smile
        )
        # The "payoff" (position value before entry cost) for display
        payoff = pnl_at_expiry + entry_cost

    breakevens = find_breakevens(S_range, pnl_at_expiry)

    max_profit = float(np.max(pnl_at_expiry))
    max_loss = float(np.min(pnl_at_expiry))

    return PayoffResult(
        S_range=S_range,
        payoff=payoff,
        pnl_at_expiry=pnl_at_expiry,
        entry_cost=entry_cost,
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven_points=breakevens,
    )


def compute_pnl_surface(
    strategy: Strategy,
    S_current: float,
    r: float,
    q: float,
    smile: VolSmile,
    S_min: float | None = None,
    S_max: float | None = None,
    n_S: int = 200,
    n_t: int = 6,
) -> PnLSurface:
    """Compute P&L grid over spot and time.

    Parameters
    ----------
    strategy : Strategy
        The option strategy.
    S_current : float
        Current spot price.
    r, q : float
        Risk-free rate and dividend yield.
    smile : VolSmile
        Volatility smile.
    S_min, S_max : float, optional
        Spot range.
    n_S : int
        Number of spot grid points.
    n_t : int
        Number of time slices (from t=0 to t=max_T).

    Returns
    -------
    PnLSurface
    """
    if S_min is None:
        S_min = S_current * 0.5
    if S_max is None:
        S_max = S_current * 1.5

    # Find the maximum expiry across all legs
    max_T = max(leg.T for leg in strategy.legs) if strategy.legs else 0.25

    S_range = np.linspace(S_min, S_max, n_S)
    t_range = np.linspace(0, max_T, n_t)

    # For multi-expiry strategies, insert intermediate expiry dates into
    # the time grid to ensure proper resolution around leg expirations.
    if not strategy._is_single_expiry():
        expiries = strategy._unique_expiries()
        tol = (max_T / n_t) * 0.1  # tolerance for deduplication
        for T_i in expiries:
            if T_i < max_T - tol:  # skip the final expiry (already at end)
                # Only insert if not already close to an existing grid point
                if not np.any(np.abs(t_range - T_i) < tol):
                    t_range = np.append(t_range, T_i)
        t_range = np.sort(np.unique(t_range))

    pnl_grid = np.zeros((len(t_range), n_S))

    for i, t_elapsed in enumerate(t_range):
        pnl_grid[i, :] = strategy.compute_pnl(S_range, t_elapsed, S_current, r, q, smile)

    return PnLSurface(S_range=S_range, t_range=t_range, pnl_grid=pnl_grid)


def compute_greeks_over_spot(
    strategy: Strategy,
    S_current: float,
    r: float,
    q: float,
    smile: VolSmile,
    t_elapsed: float = 0.0,
    S_min: float | None = None,
    S_max: float | None = None,
    n_points: int = 200,
) -> GreeksGrid:
    """Compute Greeks across spot prices.

    Returns
    -------
    GreeksGrid
    """
    if S_min is None:
        S_min = S_current * 0.5
    if S_max is None:
        S_max = S_current * 1.5

    S_range = np.linspace(S_min, S_max, n_points)
    data = compute_greeks_vs_spot(strategy, S_range, r, q, smile, t_elapsed)

    return GreeksGrid(
        x_values=S_range,
        x_label="Spot Price",
        delta=data["delta"],
        gamma=data["gamma"],
        theta=data["theta"],
        vega=data["vega"],
        rho=data["rho"],
    )


def compute_greeks_over_time(
    strategy: Strategy,
    S: float,
    r: float,
    q: float,
    smile: VolSmile,
    n_points: int = 200,
) -> GreeksGrid:
    """Compute Greeks over time-to-expiry.

    Returns
    -------
    GreeksGrid
    """
    max_T = max(leg.T for leg in strategy.legs) if strategy.legs else 0.25
    t_range = np.linspace(0, max_T * 0.99, n_points)  # avoid exact expiry

    data = compute_greeks_vs_time(strategy, S, t_range, r, q, smile)

    return GreeksGrid(
        x_values=t_range,
        x_label="Time Elapsed (years)",
        delta=data["delta"],
        gamma=data["gamma"],
        theta=data["theta"],
        vega=data["vega"],
        rho=data["rho"],
    )
