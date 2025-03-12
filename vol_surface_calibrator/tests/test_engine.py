"""Unit tests for the vol surface calibration engine."""

import numpy as np
import pytest

from engine.iv_calculator import bs_call_price, bs_put_price, implied_volatility, compute_iv_chain
from engine.svi_model import SVISlice, SVICalibrator, SVISurface

import pandas as pd


# =========================================================================
# Black-Scholes tests
# =========================================================================

class TestBlackScholes:
    def test_call_put_parity(self):
        """C - P = S - K*e^{-rT}."""
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
        C = bs_call_price(S, K, T, r, sigma)
        P = bs_put_price(S, K, T, r, sigma)
        assert C - P == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)

    def test_call_price_positive(self):
        C = bs_call_price(100, 100, 1.0, 0.05, 0.2)
        assert C > 0

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*e^{-rT}."""
        C = bs_call_price(200, 50, 1.0, 0.05, 0.2)
        intrinsic = 200 - 50 * np.exp(-0.05)
        assert C == pytest.approx(intrinsic, rel=0.01)

    def test_zero_vol_call(self):
        """σ=0 → max(S - K*e^{-rT}, 0)."""
        C = bs_call_price(100, 95, 1.0, 0.05, 0.0)
        assert C == pytest.approx(max(100 - 95 * np.exp(-0.05), 0), abs=1e-8)

    def test_put_positive_otm(self):
        P = bs_put_price(100, 90, 0.5, 0.03, 0.30)
        assert P > 0


# =========================================================================
# Implied Volatility tests
# =========================================================================

class TestImpliedVolatility:
    @pytest.mark.parametrize("true_sigma", [0.10, 0.20, 0.35, 0.60, 1.00])
    def test_call_round_trip(self, true_sigma):
        """Price a call then recover σ."""
        S, K, T, r = 100, 100, 1.0, 0.04
        price = bs_call_price(S, K, T, r, true_sigma)
        iv = implied_volatility(price, S, K, T, r, "call")
        assert iv is not None
        assert iv == pytest.approx(true_sigma, abs=1e-6)

    @pytest.mark.parametrize("true_sigma", [0.15, 0.30, 0.50])
    def test_put_round_trip(self, true_sigma):
        S, K, T, r = 100, 110, 0.75, 0.03
        price = bs_put_price(S, K, T, r, true_sigma)
        iv = implied_volatility(price, S, K, T, r, "put")
        assert iv is not None
        assert iv == pytest.approx(true_sigma, abs=1e-6)

    def test_returns_none_for_negative_price(self):
        assert implied_volatility(-1.0, 100, 100, 1.0, 0.04, "call") is None

    def test_returns_none_below_intrinsic(self):
        # Price below intrinsic is impossible
        assert implied_volatility(0.001, 100, 50, 1.0, 0.04, "call") is None

    def test_otm_put(self):
        """OTM put should give valid IV."""
        S, K, T, r, sigma = 100, 85, 0.5, 0.04, 0.25
        P = bs_put_price(S, K, T, r, sigma)
        iv = implied_volatility(P, S, K, T, r, "put")
        assert iv is not None
        assert iv == pytest.approx(sigma, abs=1e-5)


# =========================================================================
# compute_iv_chain
# =========================================================================

class TestComputeIVChain:
    def test_adds_iv_column(self):
        """Generate synthetic option prices and verify IV recovery."""
        sigma_true = 0.25
        S, r = 100.0, 0.04
        T = 0.5
        strikes = np.arange(80, 121, 5.0)
        rows = []
        for K in strikes:
            mid = bs_call_price(S, K, T, r, sigma_true)
            rows.append({
                "strike": K, "T": T, "mid_price": mid,
                "option_type": "call", "forward": S * np.exp(r * T),
            })
        df = pd.DataFrame(rows)
        result = compute_iv_chain(df, S, r)
        assert "iv" in result.columns
        assert len(result) > 0
        np.testing.assert_allclose(result["iv"].values, sigma_true, atol=1e-4)


# =========================================================================
# SVI model tests
# =========================================================================

class TestSVISlice:
    def test_total_variance_at_center(self):
        """At k = m, w = a + b * sigma."""
        s = SVISlice(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.15, T=1.0, expiry="test")
        w = s.total_variance(np.array([0.0]))
        assert w[0] == pytest.approx(0.02 + 0.1 * 0.15, abs=1e-10)

    def test_wings_grow(self):
        """Variance should grow in the wings (large |k|)."""
        s = SVISlice(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1, T=1.0, expiry="test")
        k = np.array([-1.0, 0.0, 1.0])
        w = s.total_variance(k)
        assert w[0] > w[1]  # left wing
        assert w[2] > w[1]  # right wing (for moderate rho)

    def test_implied_vol_positive(self):
        s = SVISlice(a=0.04, b=0.1, rho=-0.2, m=0.0, sigma=0.1, T=1.0, expiry="test")
        iv = s.implied_vol(np.linspace(-0.5, 0.5, 50))
        assert np.all(iv > 0)

    def test_arbitrage_free(self):
        s = SVISlice(a=0.04, b=0.1, rho=-0.2, m=0.0, sigma=0.1, T=1.0, expiry="test")
        assert s.is_arbitrage_free()

    def test_not_arbitrage_free(self):
        # a very negative → violates condition
        s = SVISlice(a=-1.0, b=0.01, rho=0.0, m=0.0, sigma=0.01, T=1.0, expiry="test")
        assert not s.is_arbitrage_free()


# =========================================================================
# SVI Calibrator tests
# =========================================================================

class TestSVICalibrator:
    @pytest.fixture
    def synthetic_smile(self):
        """Generate a synthetic SVI smile and return (k, w_market, T, params)."""
        true = SVISlice(a=0.03, b=0.12, rho=-0.35, m=-0.02, sigma=0.12,
                        T=0.5, expiry="synth")
        k = np.linspace(-0.4, 0.4, 30)
        w = true.total_variance(k)
        # Add tiny noise
        rng = np.random.default_rng(42)
        w_noisy = w + rng.normal(0, 1e-5, size=len(w))
        return k, w_noisy, true

    def test_recovers_params(self, synthetic_smile):
        k, w_market, true = synthetic_smile
        cal = SVICalibrator()
        fitted = cal.calibrate_slice(k, w_market, true.T, true.expiry)

        # Check RMSE is small
        assert fitted.rmse < 1e-3

        # Check fitted smile matches market
        w_fitted = fitted.total_variance(k)
        np.testing.assert_allclose(w_fitted, w_market, atol=5e-4)

    def test_calibrated_slice_is_arb_free(self, synthetic_smile):
        k, w_market, true = synthetic_smile
        cal = SVICalibrator()
        fitted = cal.calibrate_slice(k, w_market, true.T, true.expiry)
        assert fitted.is_arbitrage_free()

    def test_calibrate_surface(self):
        """Multi-slice calibration."""
        rows = []
        for T, expiry in [(0.25, "2025-06"), (0.5, "2025-09"), (1.0, "2026-03")]:
            true = SVISlice(a=0.02 + 0.01 * T, b=0.10, rho=-0.30,
                            m=0.0, sigma=0.10, T=T, expiry=expiry)
            k = np.linspace(-0.3, 0.3, 20)
            w = true.total_variance(k)
            for ki, wi in zip(k, w):
                rows.append({"expiry": expiry, "T": T,
                             "log_moneyness": ki, "total_var": wi})

        df = pd.DataFrame(rows)
        cal = SVICalibrator()
        surface = cal.calibrate_surface(df, ticker="TEST")

        assert len(surface.slices) == 3
        assert all(s.rmse < 1e-3 for s in surface.slices)

    def test_surface_grid(self):
        """Verify implied_vol_grid returns sensible values."""
        slices = [
            SVISlice(a=0.03, b=0.10, rho=-0.3, m=0.0, sigma=0.10, T=0.25, expiry="Q1"),
            SVISlice(a=0.04, b=0.10, rho=-0.3, m=0.0, sigma=0.10, T=1.0, expiry="Q4"),
        ]
        surface = SVISurface(slices=slices)
        k_grid = np.linspace(-0.3, 0.3, 15)
        T_grid = np.array([0.25, 0.5, 0.75, 1.0])
        iv = surface.implied_vol_grid(k_grid, T_grid)

        assert iv.shape == (4, 15)
        assert np.all(np.isfinite(iv))
        assert np.all(iv > 0)
