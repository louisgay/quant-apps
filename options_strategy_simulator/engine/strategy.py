"""Option strategy data model and preset factories.

OptionLeg represents a single option position. Strategy holds a list of legs
and provides payoff, entry cost, and P&L computation with skew-adjusted IV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .pricing import bs_price
from .vol_surface import VolSmile


@dataclass
class OptionLeg:
    """A single option leg in a strategy.

    Attributes
    ----------
    option_type : str
        "call" or "put".
    direction : str
        "long" or "short".
    strike : float
        Strike price.
    T : float
        Time to expiry in years.
    quantity : int
        Number of contracts.
    """

    option_type: str  # "call" or "put"
    direction: str    # "long" or "short"
    strike: float
    T: float
    quantity: int = 1


@dataclass
class Strategy:
    """Multi-leg option strategy.

    Attributes
    ----------
    legs : list of OptionLeg
        The option legs composing this strategy.
    name : str
        Human-readable strategy name.
    """

    legs: List[OptionLeg] = field(default_factory=list)
    name: str = "Custom"

    def _unique_expiries(self) -> list[float]:
        """Sorted unique expiry times across all legs."""
        return sorted(set(leg.T for leg in self.legs))

    def _is_single_expiry(self) -> bool:
        """True if all legs share the same expiration (within tolerance)."""
        return len(self._unique_expiries()) <= 1

    def compute_payoff_at_time(
        self,
        S: np.ndarray,
        t_elapsed: float,
        r: float,
        q: float,
        smile: VolSmile,
    ) -> np.ndarray:
        """Value of position at time t for given spot prices.

        Expired legs use intrinsic value, live legs use BS repricing.
        Does NOT subtract entry cost (returns raw position value).

        Parameters
        ----------
        S : ndarray
            Spot prices at time t.
        t_elapsed : float
            Time elapsed since trade entry (years).
        r : float
            Risk-free rate.
        q : float
            Continuous dividend yield.
        smile : VolSmile
            Volatility smile.

        Returns
        -------
        value : ndarray
            Position value at time t across S array.
        """
        S = np.asarray(S, dtype=float)
        value = np.zeros_like(S)

        for leg in self.legs:
            T_remaining = max(leg.T - t_elapsed, 0.0)
            sign = 1.0 if leg.direction == "long" else -1.0

            if T_remaining <= 1e-10:
                # At or past expiry — use intrinsic value
                if leg.option_type == "call":
                    leg_value = np.maximum(S - leg.strike, 0.0)
                else:
                    leg_value = np.maximum(leg.strike - S, 0.0)
            else:
                iv = smile.get_iv_for_strike(leg.strike, S, T_remaining)
                leg_value = bs_price(S, leg.strike, T_remaining, r, iv, leg.option_type, q)

            value += sign * leg.quantity * leg_value

        return value

    def compute_payoff(self, S_range: np.ndarray) -> np.ndarray:
        """Compute intrinsic payoff at expiry across a range of spot prices.

        Parameters
        ----------
        S_range : ndarray
            Array of spot prices at expiry.

        Returns
        -------
        payoff : ndarray
            Net payoff per unit (not including premium).
        """
        S_range = np.asarray(S_range, dtype=float)
        payoff = np.zeros_like(S_range)
        for leg in self.legs:
            sign = 1.0 if leg.direction == "long" else -1.0
            if leg.option_type == "call":
                leg_payoff = np.maximum(S_range - leg.strike, 0.0)
            else:
                leg_payoff = np.maximum(leg.strike - S_range, 0.0)
            payoff += sign * leg.quantity * leg_payoff
        return payoff

    def compute_entry_cost(self, S: float, r: float, q: float, smile: VolSmile) -> float:
        """Compute net entry cost (debit > 0, credit < 0).

        Each leg is priced at its own skew-adjusted IV from the smile.

        Parameters
        ----------
        S : float
            Current spot price.
        r : float
            Risk-free rate.
        q : float
            Continuous dividend yield.
        smile : VolSmile
            Volatility smile for IV lookup.

        Returns
        -------
        cost : float
            Net premium paid (positive) or received (negative).
        """
        cost = 0.0
        for leg in self.legs:
            iv = float(smile.get_iv_for_strike(leg.strike, S, leg.T))
            price = float(bs_price(S, leg.strike, leg.T, r, iv, leg.option_type, q))
            sign = 1.0 if leg.direction == "long" else -1.0
            cost += sign * leg.quantity * price
        return cost

    def compute_pnl(
        self,
        S_range: np.ndarray,
        t_elapsed: float,
        S_current: float,
        r: float,
        q: float,
        smile: VolSmile,
    ) -> np.ndarray:
        """Compute P&L at a given time for a range of spot prices.

        Parameters
        ----------
        S_range : ndarray
            Array of hypothetical spot prices.
        t_elapsed : float
            Time elapsed since entry (years).
        S_current : float
            Spot price at entry (for entry cost computation).
        r : float
            Risk-free rate.
        q : float
            Continuous dividend yield.
        smile : VolSmile
            Volatility smile.

        Returns
        -------
        pnl : ndarray
            P&L array across S_range.
        """
        S_range = np.asarray(S_range, dtype=float)
        entry_cost = self.compute_entry_cost(S_current, r, q, smile)
        current_value = np.zeros_like(S_range)

        for leg in self.legs:
            T_remaining = max(leg.T - t_elapsed, 0.0)
            sign = 1.0 if leg.direction == "long" else -1.0

            if T_remaining <= 1e-10:
                # At or past expiry — use intrinsic value
                if leg.option_type == "call":
                    leg_value = np.maximum(S_range - leg.strike, 0.0)
                else:
                    leg_value = np.maximum(leg.strike - S_range, 0.0)
            else:
                iv = smile.get_iv_for_strike(leg.strike, S_range, T_remaining)
                leg_value = bs_price(S_range, leg.strike, T_remaining, r, iv, leg.option_type, q)

            current_value += sign * leg.quantity * leg_value

        return current_value - entry_cost


# ---------------------------------------------------------------------------
# Preset strategy factories
# ---------------------------------------------------------------------------

def straddle(strike: float, T: float = 0.25) -> Strategy:
    """Long straddle: buy call + buy put at same strike."""
    return Strategy(
        legs=[
            OptionLeg("call", "long", strike, T),
            OptionLeg("put", "long", strike, T),
        ],
        name="Straddle",
    )


def strangle(put_strike: float, call_strike: float, T: float = 0.25) -> Strategy:
    """Long strangle: buy OTM put + buy OTM call."""
    return Strategy(
        legs=[
            OptionLeg("put", "long", put_strike, T),
            OptionLeg("call", "long", call_strike, T),
        ],
        name="Strangle",
    )


def bull_call_spread(lower_strike: float, upper_strike: float, T: float = 0.25) -> Strategy:
    """Bull call spread: buy lower call, sell higher call."""
    return Strategy(
        legs=[
            OptionLeg("call", "long", lower_strike, T),
            OptionLeg("call", "short", upper_strike, T),
        ],
        name="Bull Call Spread",
    )


def bear_put_spread(lower_strike: float, upper_strike: float, T: float = 0.25) -> Strategy:
    """Bear put spread: buy higher put, sell lower put."""
    return Strategy(
        legs=[
            OptionLeg("put", "short", lower_strike, T),
            OptionLeg("put", "long", upper_strike, T),
        ],
        name="Bear Put Spread",
    )


def butterfly(lower: float, middle: float, upper: float, T: float = 0.25) -> Strategy:
    """Long call butterfly: buy 1 lower, sell 2 middle, buy 1 upper."""
    return Strategy(
        legs=[
            OptionLeg("call", "long", lower, T),
            OptionLeg("call", "short", middle, T, quantity=2),
            OptionLeg("call", "long", upper, T),
        ],
        name="Butterfly",
    )


def iron_condor(
    put_lower: float,
    put_upper: float,
    call_lower: float,
    call_upper: float,
    T: float = 0.25,
) -> Strategy:
    """Iron condor: bull put spread + bear call spread."""
    return Strategy(
        legs=[
            OptionLeg("put", "short", put_upper, T),
            OptionLeg("put", "long", put_lower, T),
            OptionLeg("call", "short", call_lower, T),
            OptionLeg("call", "long", call_upper, T),
        ],
        name="Iron Condor",
    )


def collar(put_strike: float, call_strike: float, T: float = 0.25) -> Strategy:
    """Collar: buy put + sell call (assumes underlying held separately)."""
    return Strategy(
        legs=[
            OptionLeg("put", "long", put_strike, T),
            OptionLeg("call", "short", call_strike, T),
        ],
        name="Collar",
    )


def ratio_spread(
    lower_strike: float,
    upper_strike: float,
    T: float = 0.25,
    ratio: int = 2,
) -> Strategy:
    """Call ratio spread: buy 1 lower call, sell `ratio` upper calls."""
    return Strategy(
        legs=[
            OptionLeg("call", "long", lower_strike, T),
            OptionLeg("call", "short", upper_strike, T, quantity=ratio),
        ],
        name="Ratio Spread",
    )


def calendar_spread(strike: float, T_near: float = 0.083, T_far: float = 0.25) -> Strategy:
    """Calendar spread: sell near-term call, buy far-term call at same strike."""
    return Strategy(
        legs=[
            OptionLeg("call", "short", strike, T_near),
            OptionLeg("call", "long", strike, T_far),
        ],
        name="Calendar Spread",
    )
