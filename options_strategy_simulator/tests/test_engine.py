"""Tests for the options strategy simulator engine."""

import numpy as np
import pytest

from engine.pricing import (
    bs_price, bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho,
    bs_vanna, bs_volga,
)
from engine.vol_surface import VolSmile
from engine.strategy import (
    OptionLeg, Strategy,
    straddle, strangle, bull_call_spread, bear_put_spread,
    butterfly, iron_condor, collar, ratio_spread, calendar_spread,
)
from engine.monte_carlo import simulate_strategy, SimulationResult
from engine.greeks import (
    PortfolioGreeks, compute_portfolio_greeks,
    compute_greeks_vs_spot, compute_greeks_vs_time,
)
from engine.analytics import (
    find_breakevens, compute_payoff_diagram, compute_pnl_surface,
    compute_greeks_over_spot, compute_greeks_over_time,
)


# =========================================================================
# TestPricing
# =========================================================================

class TestPricing:
    """Tests for Black-Scholes pricing functions."""

    def test_put_call_parity_no_dividend(self):
        """C - P = S - K*exp(-rT) when q=0."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
        call = bs_price(S, K, T, r, sigma, "call", q=0.0)
        put = bs_price(S, K, T, r, sigma, "put", q=0.0)
        expected = S - K * np.exp(-r * T)
        assert abs(call - put - expected) < 1e-10

    def test_put_call_parity_with_dividend(self):
        """C - P = S*exp(-qT) - K*exp(-rT) when q > 0."""
        S, K, T, r, sigma, q = 100.0, 100.0, 0.5, 0.05, 0.25, 0.02
        call = bs_price(S, K, T, r, sigma, "call", q=q)
        put = bs_price(S, K, T, r, sigma, "put", q=q)
        expected = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(call - put - expected) < 1e-10

    def test_vectorization_spot(self):
        """bs_price accepts array of spot prices."""
        S = np.array([90, 100, 110])
        prices = bs_price(S, 100, 0.25, 0.05, 0.20, "call")
        assert prices.shape == (3,)
        assert prices[0] < prices[1] < prices[2]  # call value increases with spot

    def test_vectorization_sigma(self):
        """bs_price accepts array of sigmas (one per strike)."""
        K = np.array([90, 100, 110])
        sigma = np.array([0.25, 0.20, 0.18])  # skew: higher IV for lower strikes
        prices = bs_price(100, K, 0.25, 0.05, sigma, "put")
        assert prices.shape == (3,)

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S*exp(-qT) - K*exp(-rT)."""
        S, K, T, r, sigma, q = 200.0, 100.0, 0.25, 0.05, 0.20, 0.0
        price = bs_price(S, K, T, r, sigma, "call", q)
        intrinsic = S - K * np.exp(-r * T)
        assert abs(price - intrinsic) < 0.5  # close to intrinsic

    def test_deep_otm_call(self):
        """Deep OTM call ≈ 0."""
        price = bs_price(50, 200, 0.25, 0.05, 0.20, "call")
        assert price < 0.01

    def test_delta_call_sign(self):
        """Call delta is positive."""
        d = bs_delta(100, 100, 0.25, 0.05, 0.20, "call")
        assert 0 < d < 1

    def test_delta_put_sign(self):
        """Put delta is negative."""
        d = bs_delta(100, 100, 0.25, 0.05, 0.20, "put")
        assert -1 < d < 0

    def test_gamma_positive(self):
        """Gamma is always positive."""
        g = bs_gamma(100, 100, 0.25, 0.05, 0.20)
        assert g > 0

    def test_theta_call_negative(self):
        """ATM call theta is typically negative."""
        t = bs_theta(100, 100, 0.25, 0.05, 0.20, "call")
        assert t < 0

    def test_vega_positive(self):
        """Vega is always positive."""
        v = bs_vega(100, 100, 0.25, 0.05, 0.20)
        assert v > 0

    def test_rho_call_positive(self):
        """Call rho is positive."""
        r_val = bs_rho(100, 100, 0.25, 0.05, 0.20, "call")
        assert r_val > 0

    def test_rho_put_negative(self):
        """Put rho is negative."""
        r_val = bs_rho(100, 100, 0.25, 0.05, 0.20, "put")
        assert r_val < 0

    def test_dividend_reduces_call_price(self):
        """Higher dividend yield reduces call price."""
        call_no_div = bs_price(100, 100, 0.5, 0.05, 0.20, "call", q=0.0)
        call_with_div = bs_price(100, 100, 0.5, 0.05, 0.20, "call", q=0.03)
        assert call_with_div < call_no_div

    def test_dividend_increases_put_price(self):
        """Higher dividend yield increases put price."""
        put_no_div = bs_price(100, 100, 0.5, 0.05, 0.20, "put", q=0.0)
        put_with_div = bs_price(100, 100, 0.5, 0.05, 0.20, "put", q=0.03)
        assert put_with_div > put_no_div


# =========================================================================
# TestVolSmile
# =========================================================================

class TestVolSmile:
    """Tests for the SVI volatility smile."""

    def test_flat_matches_atm(self):
        """Flat smile returns ATM vol for all strikes."""
        smile = VolSmile.flat(0.20)
        K = np.array([80, 90, 100, 110, 120])
        iv = smile.get_iv_for_strike(K, 100, 0.25)
        np.testing.assert_allclose(iv, 0.20, atol=1e-6)

    def test_skew_produces_put_premium(self):
        """Positive skew gives higher IV for OTM puts (lower strikes)."""
        smile = VolSmile.from_simple(atm_vol=0.20, skew=0.5, curvature=0.3)
        iv_otm_put = smile.get_iv_for_strike(85, 100, 0.25)
        iv_atm = smile.get_iv_for_strike(100, 100, 0.25)
        assert iv_otm_put > iv_atm

    def test_curvature_increases_wings(self):
        """Higher curvature increases IV for both wings."""
        smile_low = VolSmile.from_simple(atm_vol=0.20, skew=0.0, curvature=0.2)
        smile_high = VolSmile.from_simple(atm_vol=0.20, skew=0.0, curvature=1.0)
        K_wing = np.array([80, 120])
        iv_low = smile_low.get_iv_for_strike(K_wing, 100, 0.25)
        iv_high = smile_high.get_iv_for_strike(K_wing, 100, 0.25)
        assert np.all(iv_high > iv_low)

    def test_arbitrage_free_from_simple(self):
        """from_simple always produces arbitrage-free smiles."""
        for skew in [0, 0.3, 0.7, 0.99]:
            for curv in [0, 0.5, 1.0, 2.0]:
                smile = VolSmile.from_simple(0.20, skew, curv)
                assert smile.is_arbitrage_free(), f"Failed for skew={skew}, curv={curv}"

    def test_from_raw_svi(self):
        """from_raw_svi stores parameters directly."""
        smile = VolSmile.from_raw_svi(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma_svi=0.1)
        assert smile.a == 0.04
        assert smile.rho == -0.3

    def test_zero_time_returns_zero(self):
        """IV for T=0 returns zeros."""
        smile = VolSmile.flat(0.20)
        iv = smile.get_iv_for_strike(100, 100, 0.0)
        assert iv == 0.0

    def test_calendar_scaling(self):
        """Longer maturity keeps total variance linear."""
        smile = VolSmile.from_simple(0.20, 0.3, 0.5)
        iv_short = smile.get_iv_for_strike(90, 100, 0.25)
        iv_long = smile.get_iv_for_strike(90, 100, 1.0)
        # Total variance should be proportional to T
        w_short = iv_short ** 2 * 0.25
        w_long = iv_long ** 2 * 1.0
        np.testing.assert_allclose(w_long / w_short, 1.0 / 0.25, rtol=1e-6)


# =========================================================================
# TestStrategy
# =========================================================================

class TestStrategy:
    """Tests for strategy payoff shapes and presets."""

    def test_straddle_v_shape(self):
        """Straddle payoff is V-shaped (positive at both wings)."""
        s = straddle(100)
        S_range = np.linspace(70, 130, 100)
        payoff = s.compute_payoff(S_range)
        assert payoff[0] > 0   # deep put side
        assert payoff[-1] > 0  # deep call side
        assert payoff[len(payoff) // 2] < payoff[0]  # minimum near ATM

    def test_butterfly_tent_shape(self):
        """Butterfly has tent shape: max payoff at middle, zero at wings."""
        s = butterfly(90, 100, 110)
        S_range = np.array([80, 90, 100, 110, 120])
        payoff = s.compute_payoff(S_range)
        assert payoff[0] == 0   # below lower
        assert payoff[-1] == 0  # above upper
        assert payoff[2] == 10  # max at middle (110-100 spread width)

    def test_iron_condor_bounded(self):
        """Iron condor has bounded payoff."""
        s = iron_condor(85, 90, 110, 115)
        S_range = np.linspace(70, 140, 200)
        payoff = s.compute_payoff(S_range)
        # Payoff is bounded between -5 and 5
        assert np.max(payoff) <= 5.0 + 1e-10
        assert np.min(payoff) >= -5.0 - 1e-10

    def test_bull_call_spread_direction(self):
        """Bull call spread: max gain above upper strike, max loss below lower."""
        s = bull_call_spread(95, 105)
        payoff_low = s.compute_payoff(np.array([80]))[0]
        payoff_high = s.compute_payoff(np.array([120]))[0]
        assert payoff_low == 0
        assert payoff_high == 10

    def test_preset_leg_counts(self):
        """Verify each preset creates the expected number of legs."""
        assert len(straddle(100).legs) == 2
        assert len(strangle(90, 110).legs) == 2
        assert len(bull_call_spread(95, 105).legs) == 2
        assert len(bear_put_spread(95, 105).legs) == 2
        assert len(butterfly(90, 100, 110).legs) == 3
        assert len(iron_condor(85, 90, 110, 115).legs) == 4
        assert len(collar(90, 110).legs) == 2
        assert len(ratio_spread(95, 105).legs) == 2
        assert len(calendar_spread(100).legs) == 2

    def test_entry_cost_long_call_positive(self):
        """Buying a single call has positive entry cost."""
        s = Strategy(legs=[OptionLeg("call", "long", 100, 0.25)])
        smile = VolSmile.flat(0.20)
        cost = s.compute_entry_cost(100, 0.05, 0.0, smile)
        assert cost > 0

    def test_entry_cost_short_call_negative(self):
        """Selling a single call has negative entry cost (credit)."""
        s = Strategy(legs=[OptionLeg("call", "short", 100, 0.25)])
        smile = VolSmile.flat(0.20)
        cost = s.compute_entry_cost(100, 0.05, 0.0, smile)
        assert cost < 0

    def test_pnl_at_expiry_matches_payoff(self):
        """P&L at expiry should equal payoff minus entry cost."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        S_range = np.linspace(80, 120, 50)
        max_T = max(leg.T for leg in s.legs)
        pnl = s.compute_pnl(S_range, max_T, 100, 0.05, 0.0, smile)
        payoff = s.compute_payoff(S_range)
        entry = s.compute_entry_cost(100, 0.05, 0.0, smile)
        np.testing.assert_allclose(pnl, payoff - entry, atol=1e-6)

    def test_calendar_spread_different_expiries(self):
        """Calendar spread legs have different maturities."""
        s = calendar_spread(100, T_near=0.083, T_far=0.25)
        Ts = [leg.T for leg in s.legs]
        assert Ts[0] != Ts[1]


# =========================================================================
# TestGreeks
# =========================================================================

class TestGreeks:
    """Tests for portfolio Greeks computation."""

    def test_straddle_near_zero_delta(self):
        """ATM straddle has near-zero delta."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        g = compute_portfolio_greeks(s, 100, 0.05, 0.0, smile)
        assert abs(g.delta) < 0.15  # non-zero r shifts call delta above 0.5

    def test_long_straddle_positive_gamma(self):
        """Long straddle has positive gamma."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        g = compute_portfolio_greeks(s, 100, 0.05, 0.0, smile)
        assert g.gamma > 0

    def test_long_straddle_positive_vega(self):
        """Long straddle has positive vega."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        g = compute_portfolio_greeks(s, 100, 0.05, 0.0, smile)
        assert g.vega > 0

    def test_long_straddle_negative_theta(self):
        """Long straddle has negative theta (time decay)."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        g = compute_portfolio_greeks(s, 100, 0.05, 0.0, smile)
        assert g.theta < 0

    def test_short_straddle_negative_gamma(self):
        """Short straddle has negative gamma."""
        s = Strategy(legs=[
            OptionLeg("call", "short", 100, 0.25),
            OptionLeg("put", "short", 100, 0.25),
        ])
        smile = VolSmile.flat(0.20)
        g = compute_portfolio_greeks(s, 100, 0.05, 0.0, smile)
        assert g.gamma < 0

    def test_short_straddle_negative_vega(self):
        """Short straddle has negative vega."""
        s = Strategy(legs=[
            OptionLeg("call", "short", 100, 0.25),
            OptionLeg("put", "short", 100, 0.25),
        ])
        smile = VolSmile.flat(0.20)
        g = compute_portfolio_greeks(s, 100, 0.05, 0.0, smile)
        assert g.vega < 0

    def test_theta_daily_conversion(self):
        """theta_daily = theta / 365."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        g = compute_portfolio_greeks(s, 100, 0.05, 0.0, smile)
        np.testing.assert_allclose(g.theta_daily, g.theta / 365.0)

    def test_greeks_vs_spot_shape(self):
        """compute_greeks_vs_spot returns correct shapes."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        S_range = np.linspace(80, 120, 50)
        data = compute_greeks_vs_spot(s, S_range, 0.05, 0.0, smile)
        assert data["delta"].shape == (50,)
        assert data["gamma"].shape == (50,)

    def test_greeks_vs_time_shape(self):
        """compute_greeks_vs_time returns correct shapes."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        t_range = np.linspace(0, 0.2, 30)
        data = compute_greeks_vs_time(s, 100, t_range, 0.05, 0.0, smile)
        assert data["delta"].shape == (30,)


# =========================================================================
# TestAnalytics
# =========================================================================

class TestAnalytics:
    """Tests for analytics / chart data computations."""

    def test_payoff_diagram_shape(self):
        """compute_payoff_diagram returns correct shapes."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        result = compute_payoff_diagram(s, 100, 0.05, 0.0, smile, n_points=100)
        assert result.S_range.shape == (100,)
        assert result.payoff.shape == (100,)
        assert result.pnl_at_expiry.shape == (100,)

    def test_straddle_two_breakevens(self):
        """Straddle should have exactly 2 breakeven points."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        result = compute_payoff_diagram(s, 100, 0.05, 0.0, smile, n_points=500)
        assert len(result.breakeven_points) == 2

    def test_butterfly_two_breakevens(self):
        """Butterfly should have 2 breakeven points."""
        s = butterfly(90, 100, 110)
        smile = VolSmile.flat(0.20)
        result = compute_payoff_diagram(s, 100, 0.05, 0.0, smile, n_points=500)
        assert len(result.breakeven_points) == 2

    def test_pnl_surface_shape(self):
        """compute_pnl_surface returns correct grid shape."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        surface = compute_pnl_surface(s, 100, 0.05, 0.0, smile, n_S=50, n_t=6)
        assert surface.pnl_grid.shape == (6, 50)
        assert surface.S_range.shape == (50,)
        assert surface.t_range.shape == (6,)

    def test_pnl_at_expiry_matches_payoff_diagram(self):
        """Last time slice of P&L surface ≈ payoff diagram P&L at expiry."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        result = compute_payoff_diagram(s, 100, 0.05, 0.0, smile, n_points=50)
        surface = compute_pnl_surface(s, 100, 0.05, 0.0, smile, n_S=50, n_t=6)
        # Last time slice is at max_T (expiry)
        np.testing.assert_allclose(
            surface.pnl_grid[-1, :], result.pnl_at_expiry, atol=0.5
        )

    def test_greeks_over_spot_shape(self):
        """compute_greeks_over_spot returns GreeksGrid with correct shapes."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        grid = compute_greeks_over_spot(s, 100, 0.05, 0.0, smile, n_points=80)
        assert grid.x_values.shape == (80,)
        assert grid.delta.shape == (80,)
        assert grid.gamma.shape == (80,)

    def test_greeks_over_time_shape(self):
        """compute_greeks_over_time returns GreeksGrid with correct shapes."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        grid = compute_greeks_over_time(s, 100, 0.05, 0.0, smile, n_points=60)
        assert grid.x_values.shape == (60,)
        assert grid.delta.shape == (60,)

    def test_find_breakevens_basic(self):
        """find_breakevens detects sign changes correctly."""
        S = np.array([1, 2, 3, 4, 5])
        pnl = np.array([-2, -1, 1, 2, 3])
        be = find_breakevens(S, pnl)
        assert len(be) == 1
        assert 2.0 < be[0] < 3.0  # between S=2 and S=3

    def test_max_profit_loss(self):
        """PayoffResult captures correct max profit and max loss."""
        s = bull_call_spread(95, 105)
        smile = VolSmile.flat(0.20)
        result = compute_payoff_diagram(s, 100, 0.05, 0.0, smile, n_points=500)
        # Max payoff for bull call spread is upper - lower = 10
        # Max profit = 10 - entry_cost
        assert result.max_profit > 0
        assert result.max_loss < 0


# =========================================================================
# TestMonteCarlo
# =========================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo simulation."""

    def test_output_shape(self):
        """SimulationResult has correct array sizes."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        sim = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=1000)
        assert sim.pnl_samples.shape == (1000,)
        assert sim.spot_samples.shape == (1000,)
        assert sim.n_paths == 1000

    def test_prob_profit_range(self):
        """Probability of profit is between 0 and 1."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        sim = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=5000)
        assert 0 <= sim.prob_profit <= 1

    def test_percentile_ordering(self):
        """Percentiles are monotonically increasing."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        sim = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=10000)
        keys = [5, 10, 25, 50, 75, 90, 95]
        vals = [sim.percentiles[k] for k in keys]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]

    def test_var_positive_for_long_straddle(self):
        """Long straddle has a cost, so VaR should be positive (there is downside)."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        sim = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=10000)
        assert sim.var_95 > 0

    def test_cvar_ge_var(self):
        """CVaR (expected shortfall) >= VaR by definition."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        sim = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=10000)
        assert sim.cvar_95 >= sim.var_95 - 1e-10
        assert sim.cvar_99 >= sim.var_99 - 1e-10

    def test_deterministic_with_seed(self):
        """Same seed produces identical results."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        sim1 = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=1000, seed=123)
        sim2 = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=1000, seed=123)
        np.testing.assert_array_equal(sim1.pnl_samples, sim2.pnl_samples)

    def test_fat_tails_more_extreme(self):
        """Lower df produces wider spot distribution (fatter tails)."""
        s = straddle(100)
        smile = VolSmile.flat(0.30)
        sim_fat = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=50000, df=3.0, seed=42)
        sim_thin = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=50000, df=30.0, seed=42)
        # Fat tails should produce wider spot dispersion
        spot_range_fat = np.percentile(sim_fat.spot_samples, 99) - np.percentile(sim_fat.spot_samples, 1)
        spot_range_thin = np.percentile(sim_thin.spot_samples, 99) - np.percentile(sim_thin.spot_samples, 1)
        assert spot_range_fat > spot_range_thin

    def test_entry_cost_matches_analytical(self):
        """MC entry cost matches strategy.compute_entry_cost."""
        s = bull_call_spread(95, 105)
        smile = VolSmile.flat(0.20)
        sim = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=1000)
        analytical_cost = s.compute_entry_cost(100, 0.05, 0.0, smile)
        np.testing.assert_allclose(sim.entry_cost, analytical_cost)

    def test_iron_condor_bounded_pnl(self):
        """Iron condor MC P&L should be bounded."""
        s = iron_condor(85, 90, 110, 115)
        smile = VolSmile.flat(0.20)
        sim = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=10000)
        entry = sim.entry_cost
        # Max payoff is 5 (width of spreads), max loss is -5
        # P&L = payoff - entry, so bounded by [-5-entry, 5-entry]
        assert np.all(sim.pnl_samples >= -5 - abs(entry) - 0.01)
        assert np.all(sim.pnl_samples <= 5 + abs(entry) + 0.01)

    def test_df_floor(self):
        """df <= 2 is clamped to 2.1 (needs finite variance)."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        sim = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=100, df=1.5)
        assert sim.df == 2.1  # clamped internally
        assert sim.pnl_samples.shape == (100,)  # still runs


# =========================================================================
# TestMultiExpiry
# =========================================================================

class TestMultiExpiry:
    """Tests for multi-expiry strategies (calendar spreads, diagonals)."""

    def test_mc_calendar_spread_not_flat(self):
        """Calendar spread MC P&L distribution should have variance, not be flat."""
        cs = calendar_spread(100, T_near=30/365, T_far=90/365)
        smile = VolSmile.from_simple(0.20, skew=0.4, curvature=0.5)
        sim = simulate_strategy(cs, 100, 0.045, 0.013, smile, n_paths=50_000, df=5, seed=42)
        # P&L should NOT all be the same (the original bug: all = -entry_cost)
        assert not np.allclose(sim.pnl_samples, -sim.entry_cost), \
            "MC P&L is flat — multi-expiry path not working"
        assert np.std(sim.pnl_samples) > 0.01

    def test_mc_calendar_spread_pop_positive(self):
        """Calendar spread should have positive probability of profit."""
        cs = calendar_spread(100, T_near=30/365, T_far=90/365)
        smile = VolSmile.from_simple(0.20, skew=0.4, curvature=0.5)
        sim = simulate_strategy(cs, 100, 0.045, 0.013, smile, n_paths=50_000, df=5, seed=42)
        assert sim.prob_profit > 0, "PoP should be > 0 for calendar spread"

    def test_mc_single_expiry_unchanged(self):
        """Single-expiry straddle MC should give identical results to a fresh run."""
        s = straddle(100)
        smile = VolSmile.flat(0.20)
        sim1 = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=5000, seed=99)
        sim2 = simulate_strategy(s, 100, 0.05, 0.0, smile, n_paths=5000, seed=99)
        np.testing.assert_array_equal(sim1.pnl_samples, sim2.pnl_samples)
        np.testing.assert_array_equal(sim1.spot_samples, sim2.spot_samples)

    def test_payoff_diagram_calendar_has_breakevens(self):
        """Calendar spread payoff diagram should have non-trivial shape with breakevens."""
        cs = calendar_spread(100, T_near=30/365, T_far=90/365)
        smile = VolSmile.from_simple(0.20, skew=0.4, curvature=0.5)
        result = compute_payoff_diagram(cs, 100, 0.045, 0.013, smile, n_points=500)
        # Should have breakeven points (not a flat line)
        assert len(result.breakeven_points) > 0, "Calendar payoff diagram has no breakevens"
        # P&L should not be constant
        assert np.std(result.pnl_at_expiry) > 0.01

    def test_pnl_surface_includes_intermediate_expiry(self):
        """P&L surface time grid should include the near-term expiry date."""
        T_near = 30 / 365
        T_far = 90 / 365
        cs = calendar_spread(100, T_near=T_near, T_far=T_far)
        smile = VolSmile.from_simple(0.20, skew=0.4, curvature=0.5)
        surface = compute_pnl_surface(cs, 100, 0.045, 0.013, smile, n_S=50, n_t=10)
        # The near-term expiry should be present in the time grid
        assert np.any(np.abs(surface.t_range - T_near) < 1e-6), \
            f"Near-term expiry {T_near} not found in t_range: {surface.t_range}"

    def test_greeks_at_intermediate_expiry(self):
        """Greeks can be computed at the near-leg expiry without error."""
        T_near = 30 / 365
        T_far = 90 / 365
        cs = calendar_spread(100, T_near=T_near, T_far=T_far)
        smile = VolSmile.from_simple(0.20, skew=0.4, curvature=0.5)
        # Should not raise — the near leg is at intrinsic, the far leg still has value
        grid = compute_greeks_over_spot(cs, 100, 0.045, 0.013, smile, t_elapsed=T_near, n_points=50)
        assert grid.delta.shape == (50,)
        assert grid.vega.shape == (50,)
