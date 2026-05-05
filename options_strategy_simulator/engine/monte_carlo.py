"""Monte Carlo simulation for option strategy P&L distribution.

Simulates terminal spot prices using GBM with Student-t innovations to capture
fat tails, then evaluates strategy P&L across all paths.

The key insight: a payoff diagram shows what *could* happen, but Monte Carlo
with fat-tailed returns shows what's *likely* to happen and how bad the tail
scenarios are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from .strategy import Strategy
from .vol_surface import VolSmile


@dataclass
class SimulationResult:
    """Monte Carlo simulation output.

    Attributes
    ----------
    pnl_samples : ndarray
        Raw P&L for each simulated path (n_paths,).
    spot_samples : ndarray
        Terminal spot prices (n_paths,).
    prob_profit : float
        Fraction of paths with P&L > 0.
    expected_pnl : float
        Mean P&L across all paths.
    median_pnl : float
        Median P&L.
    percentiles : dict
        P&L at key percentiles {5, 10, 25, 50, 75, 90, 95}.
    var_95 : float
        Value-at-Risk at 95% confidence (5th percentile loss, reported positive).
    var_99 : float
        Value-at-Risk at 99% confidence.
    cvar_95 : float
        Conditional VaR / Expected Shortfall at 95% (mean of worst 5%).
    cvar_99 : float
        Conditional VaR / Expected Shortfall at 99%.
    entry_cost : float
        Net premium paid/received at entry.
    n_paths : int
        Number of simulated paths.
    df : float
        Degrees of freedom used for Student-t innovations.
    """

    pnl_samples: np.ndarray
    spot_samples: np.ndarray
    prob_profit: float
    expected_pnl: float
    median_pnl: float
    percentiles: dict
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    entry_cost: float
    n_paths: int
    df: float


def simulate_strategy(
    strategy: Strategy,
    S: float,
    r: float,
    q: float,
    smile: VolSmile,
    n_paths: int = 50_000,
    df: float = 5.0,
    seed: int | None = 42,
) -> SimulationResult:
    """Run Monte Carlo simulation of strategy P&L at expiry.

    Generates terminal spot prices via GBM with Student-t innovations:
        S_T = S * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    where Z ~ t(df) scaled to unit variance.

    Using Student-t instead of normal captures fat tails — real markets have
    more extreme moves than normal suggests. The df parameter controls tail
    fatness: df=30 ≈ normal, df=5 = moderate tails, df=3 = heavy tails.

    Parameters
    ----------
    strategy : Strategy
        The option strategy to simulate.
    S : float
        Current spot price.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    smile : VolSmile
        Volatility smile (used for ATM vol as the diffusion vol).
    n_paths : int
        Number of Monte Carlo paths.
    df : float
        Degrees of freedom for Student-t innovations.
        Lower = fatter tails. Must be > 2 for finite variance.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    SimulationResult
    """
    if df <= 2:
        df = 2.1  # need finite variance

    rng = np.random.default_rng(seed)

    # Use ATM IV as the diffusion volatility
    max_T = max(leg.T for leg in strategy.legs)
    sigma = float(smile.get_iv_for_strike(S, S, max_T))

    entry_cost = strategy.compute_entry_cost(S, r, q, smile)

    if strategy._is_single_expiry():
        # === Single-expiry fast path (original behavior) ===
        # Generate Student-t innovations scaled to unit variance
        # Var(t_df) = df / (df - 2), so scale by sqrt((df-2)/df)
        raw = student_t.rvs(df, size=n_paths, random_state=rng)
        scale_factor = np.sqrt((df - 2) / df)
        Z = raw * scale_factor  # now Var(Z) ≈ 1

        # Simulate terminal spot: S_T = S * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        drift = (r - q - 0.5 * sigma ** 2) * max_T
        diffusion = sigma * np.sqrt(max_T) * Z
        spot_samples = S * np.exp(drift + diffusion)

        # Compute strategy payoff at expiry for each path
        payoff = strategy.compute_payoff(spot_samples)
        pnl_samples = payoff - entry_cost
    else:
        # === Multi-expiry path (calendar spreads, diagonals) ===
        # Simulate spot to the first (near-term) expiry, then evaluate
        # the full position: expired legs at intrinsic, live legs repriced
        # via BS. This captures time value of far-dated legs which is where
        # calendar spread value comes from.
        min_T = min(leg.T for leg in strategy.legs)
        scale_factor = np.sqrt((df - 2) / df)

        # Simulate spot to the near-term expiry
        raw = student_t.rvs(df, size=n_paths, random_state=rng)
        Z = raw * scale_factor
        drift = (r - q - 0.5 * sigma ** 2) * min_T
        diffusion = sigma * np.sqrt(min_T) * Z
        spot_samples = S * np.exp(drift + diffusion)

        # Evaluate position at near-term expiry: expired legs → intrinsic,
        # live legs → BS repriced (captures time value of far-dated legs)
        position_value = strategy.compute_payoff_at_time(
            spot_samples, min_T, r, q, smile
        )
        pnl_samples = position_value - entry_cost

    # Statistics
    prob_profit = float(np.mean(pnl_samples > 0))
    expected_pnl = float(np.mean(pnl_samples))
    median_pnl = float(np.median(pnl_samples))

    pct_keys = [5, 10, 25, 50, 75, 90, 95]
    pct_values = np.percentile(pnl_samples, pct_keys)
    percentiles = dict(zip(pct_keys, pct_values.tolist()))

    # VaR: the loss at a given confidence level (reported as positive number)
    var_95 = -float(np.percentile(pnl_samples, 5))
    var_99 = -float(np.percentile(pnl_samples, 1))

    # CVaR / Expected Shortfall: mean of losses beyond VaR
    sorted_pnl = np.sort(pnl_samples)
    n_5pct = max(int(n_paths * 0.05), 1)
    n_1pct = max(int(n_paths * 0.01), 1)
    cvar_95 = -float(np.mean(sorted_pnl[:n_5pct]))
    cvar_99 = -float(np.mean(sorted_pnl[:n_1pct]))

    return SimulationResult(
        pnl_samples=pnl_samples,
        spot_samples=spot_samples,
        prob_profit=prob_profit,
        expected_pnl=expected_pnl,
        median_pnl=median_pnl,
        percentiles=percentiles,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        entry_cost=entry_cost,
        n_paths=n_paths,
        df=df,
    )
