"""Options Strategy Simulator engine — public API."""

from .pricing import (
    bs_price,
    bs_delta,
    bs_gamma,
    bs_theta,
    bs_vega,
    bs_rho,
    bs_vanna,
    bs_volga,
)
from .vol_surface import VolSmile
from .strategy import (
    OptionLeg,
    Strategy,
    straddle,
    strangle,
    bull_call_spread,
    bear_put_spread,
    butterfly,
    iron_condor,
    collar,
    ratio_spread,
    calendar_spread,
)
from .greeks import (
    PortfolioGreeks,
    compute_portfolio_greeks,
    compute_greeks_vs_spot,
    compute_greeks_vs_time,
)
from .analytics import (
    PayoffResult,
    PnLSurface,
    GreeksGrid,
    find_breakevens,
    compute_payoff_diagram,
    compute_pnl_surface,
    compute_greeks_over_spot,
    compute_greeks_over_time,
)
from .monte_carlo import SimulationResult, simulate_strategy

__all__ = [
    # Pricing
    "bs_price", "bs_delta", "bs_gamma", "bs_theta", "bs_vega", "bs_rho",
    "bs_vanna", "bs_volga",
    # Vol surface
    "VolSmile",
    # Strategy
    "OptionLeg", "Strategy",
    "straddle", "strangle", "bull_call_spread", "bear_put_spread",
    "butterfly", "iron_condor", "collar", "ratio_spread", "calendar_spread",
    # Greeks
    "PortfolioGreeks", "compute_portfolio_greeks",
    "compute_greeks_vs_spot", "compute_greeks_vs_time",
    # Analytics
    "PayoffResult", "PnLSurface", "GreeksGrid", "find_breakevens",
    "compute_payoff_diagram", "compute_pnl_surface",
    "compute_greeks_over_spot", "compute_greeks_over_time",
    # Monte Carlo
    "SimulationResult", "simulate_strategy",
]
