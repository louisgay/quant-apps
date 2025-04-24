"""Tests for the PDE option pricing engine (~30 tests)."""

import numpy as np
import pytest

from engine.analytics import (
    bs_price,
    bs_delta,
    bs_gamma,
    bs_theta,
    bs_vega,
    compute_price_surface,
)
from engine.models import GridConfig
from engine.solvers import (
    price_european_american,
    price_barrier_local_vol,
    extract_free_boundary,
    price_american_dividends_psor,
)


# ── helpers ──────────────────────────────────────────────────────────────────

K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
COARSE = GridConfig(N=200, M=200)
FINE = GridConfig(N=500, M=500)


# ═══════════════════════════════════════════════════════════════════════════════
# Black-Scholes analytical
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlackScholes:

    def test_put_call_parity(self):
        S = 100.0
        c = bs_price(S, K, T, r, sigma, "call")
        p = bs_price(S, K, T, r, sigma, "put")
        assert abs(c - p - S + K * np.exp(-r * T)) < 1e-10

    def test_deep_itm_call(self):
        price = bs_price(200.0, K, T, r, sigma, "call")
        assert price > 95.0

    def test_deep_otm_call(self):
        price = bs_price(10.0, K, T, r, sigma, "call")
        assert price < 0.01

    def test_vectorized(self):
        S_arr = np.array([80.0, 100.0, 120.0])
        prices = bs_price(S_arr, K, T, r, sigma, "call")
        assert prices.shape == (3,)
        assert prices[0] < prices[1] < prices[2]

    def test_delta_call_positive(self):
        d = bs_delta(100.0, K, T, r, sigma, "call")
        assert 0 < d < 1

    def test_delta_put_negative(self):
        d = bs_delta(100.0, K, T, r, sigma, "put")
        assert -1 < d < 0

    def test_gamma_positive(self):
        g = bs_gamma(100.0, K, T, r, sigma)
        assert g > 0

    def test_theta_call_negative(self):
        th = bs_theta(100.0, K, T, r, sigma, "call")
        assert th < 0

    def test_vega_positive(self):
        v = bs_vega(100.0, K, T, r, sigma)
        assert v > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Crank-Nicolson European / American
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrankNicolson:

    def test_european_call_converges_to_bs(self):
        res = price_european_american(K, T, r, sigma, "call", "european", FINE)
        bs = bs_price(K, K, T, r, sigma, "call")
        assert abs(res.price - bs) / bs < 0.005  # < 0.5%

    def test_european_put_converges_to_bs(self):
        res = price_european_american(K, T, r, sigma, "put", "european", FINE)
        bs = bs_price(K, K, T, r, sigma, "put")
        assert abs(res.price - bs) / bs < 0.005

    def test_pde_put_call_parity(self):
        c = price_european_american(K, T, r, sigma, "call", "european", FINE)
        p = price_european_american(K, T, r, sigma, "put", "european", FINE)
        parity = c.price - p.price - (K - K * np.exp(-r * T))
        # PDE parity: C - P = S - K·e^{-rT}  at S=K
        assert abs(parity) < 0.5

    def test_american_put_geq_european(self):
        eu = price_european_american(K, T, r, sigma, "put", "european", COARSE)
        am = price_european_american(K, T, r, sigma, "put", "american", COARSE)
        assert am.price >= eu.price - 0.01  # allow tiny numerical tolerance

    def test_grid_refinement_improves_accuracy(self):
        coarse = price_european_american(K, T, r, sigma, "call", "european",
                                         GridConfig(N=100, M=100))
        fine = price_european_american(K, T, r, sigma, "call", "european",
                                       GridConfig(N=500, M=500))
        bs = bs_price(K, K, T, r, sigma, "call")
        assert abs(fine.price - bs) < abs(coarse.price - bs) + 0.01

    def test_result_fields(self):
        res = price_european_american(K, T, r, sigma, "call", "european", COARSE)
        assert res.S_grid.shape == (200,)
        assert res.V_grid.shape == (200,)
        assert res.option_type == "call"
        assert res.elapsed > 0

    def test_values_non_negative(self):
        res = price_european_american(K, T, r, sigma, "put", "european", COARSE)
        assert np.all(res.V_grid >= -1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# Barrier option with local vol
# ═══════════════════════════════════════════════════════════════════════════════

class TestBarrierOption:

    def test_barrier_leq_vanilla(self):
        barrier = price_barrier_local_vol(K, T, r, 0.20, 0.4, 130.0, COARSE)
        vanilla = price_european_american(K, T, r, 0.20, "call", "european", COARSE)
        assert barrier.price <= vanilla.price + 0.5

    def test_value_zero_above_barrier(self):
        res = price_barrier_local_vol(K, T, r, 0.20, 0.4, 130.0, COARSE)
        above = res.V_grid[res.S_grid >= res.barrier]
        assert np.allclose(above, 0.0, atol=1e-10)

    def test_local_vol_skew(self):
        res = price_barrier_local_vol(K, T, r, 0.20, 0.4, 130.0, COARSE)
        # alpha > 0 ⇒ vol decreasing in S (skew)
        mid = len(res.vol_grid) // 2
        assert res.vol_grid[mid - 10] > res.vol_grid[mid + 10]

    def test_barrier_price_positive(self):
        res = price_barrier_local_vol(K, T, r, 0.20, 0.4, 130.0, COARSE)
        assert res.price > 0

    def test_higher_barrier_higher_price(self):
        low_b = price_barrier_local_vol(K, T, r, 0.20, 0.4, 120.0, COARSE)
        high_b = price_barrier_local_vol(K, T, r, 0.20, 0.4, 150.0, COARSE)
        assert high_b.price >= low_b.price - 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# Free-boundary extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestFreeBoundary:

    def test_boundary_at_maturity_equals_K(self):
        fb = extract_free_boundary(K, T, r, sigma, "put", COARSE)
        assert abs(fb.boundary[-1] - K) < 1.0

    def test_put_boundary_below_K(self):
        fb = extract_free_boundary(K, T, r, sigma, "put", COARSE)
        # For a put, the exercise boundary should be ≤ K at all times
        assert np.all(fb.boundary <= K + 1.0)

    def test_time_grid_length(self):
        fb = extract_free_boundary(K, T, r, sigma, "put", COARSE)
        assert len(fb.time_grid) == COARSE.M + 1
        assert len(fb.boundary) == COARSE.M + 1

    def test_boundary_monotone_put(self):
        fb = extract_free_boundary(K, T, r, sigma, "put", COARSE)
        # Put boundary should be roughly non-decreasing towards maturity
        # (S* increases as t → T)
        assert fb.boundary[-1] >= fb.boundary[0] - 1.0

    def test_exercise_value_shape(self):
        fb = extract_free_boundary(K, T, r, sigma, "put", COARSE)
        assert fb.exercise_value.shape == fb.S_grid.shape


# ═══════════════════════════════════════════════════════════════════════════════
# PSOR with discrete dividends
# ═══════════════════════════════════════════════════════════════════════════════

class TestPSOR:

    def test_convergence(self):
        res = price_american_dividends_psor(
            K, T, r, 0.25, "call", 5.0, 0.48, grid=GridConfig(N=100, M=150),
        )
        assert res.price > 0
        assert res.iterations > 0

    def test_price_geq_intrinsic(self):
        res = price_american_dividends_psor(
            K, T, r, 0.25, "call", 5.0, 0.48, grid=GridConfig(N=100, M=150),
        )
        intrinsic = max(K - K, 0)  # at S=K for a call
        assert res.price >= intrinsic - 0.01

    def test_dividend_reduces_call(self):
        no_div = price_american_dividends_psor(
            K, T, r, 0.25, "call", 0.0, 0.48, grid=GridConfig(N=100, M=150),
        )
        with_div = price_american_dividends_psor(
            K, T, r, 0.25, "call", 5.0, 0.48, grid=GridConfig(N=100, M=150),
        )
        assert no_div.price > with_div.price - 0.1

    def test_full_grid_shape(self):
        g = GridConfig(N=100, M=150)
        res = price_american_dividends_psor(
            K, T, r, 0.25, "call", 5.0, 0.48, grid=g,
        )
        assert res.V_full.shape == (g.N + 1, g.M + 1)
        assert res.S_grid.shape == (g.N + 1,)

    def test_free_boundary_shape(self):
        g = GridConfig(N=100, M=150)
        res = price_american_dividends_psor(
            K, T, r, 0.25, "call", 5.0, 0.48, grid=g,
        )
        assert res.free_boundary.shape == (g.M + 1,)


# ═══════════════════════════════════════════════════════════════════════════════
# Price surface
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceSurface:

    def test_surface_shape(self):
        surf = compute_price_surface(K, r, sigma, "put", n_S=50, n_T=40)
        assert surf.V_surface.shape == (40, 50)
        assert surf.S_values.shape == (50,)
        assert surf.T_values.shape == (40,)

    def test_surface_non_negative(self):
        surf = compute_price_surface(K, r, sigma, "call")
        assert np.all(surf.V_surface >= -1e-10)
