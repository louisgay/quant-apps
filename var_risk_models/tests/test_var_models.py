"""VaR model tests: conventions, ordering, GJR asymmetry, stat tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.backtest import (
    christoffersen_test,
    combined_coverage_test,
    kupiec_test,
    var_exceedance_test,
)
from engine.data_loader import compute_log_returns
from engine.garch import fit_gjr_garch, news_impact_curve, simulate_fhs
from engine.var_models import (
    historical_simulation_var,
    parametric_normal_var,
    parametric_student_t_var,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_returns():
    """Generate synthetic GARCH-like returns with known properties."""
    rng = np.random.default_rng(42)
    n = 1000
    returns = rng.normal(loc=-0.0001, scale=0.015, size=n)
    # Add some volatility clustering
    for i in range(1, n):
        if abs(returns[i - 1]) > 0.02:
            returns[i] *= 1.5
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(returns, index=index, name="log_return")


@pytest.fixture
def asymmetric_returns():
    """Returns with clear negative skew for GJR testing."""
    rng = np.random.default_rng(123)
    n = 1500
    # GJR-GARCH DGP
    omega, alpha, gamma, beta = 0.00001, 0.05, 0.10, 0.85
    mu = 0.0002
    sigma2 = np.zeros(n)
    returns = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - gamma / 2 - beta)

    for t in range(1, n):
        z = rng.standard_t(df=5)
        eps = np.sqrt(sigma2[t - 1]) * z
        leverage = gamma if eps < 0 else 0.0
        sigma2[t] = omega + (alpha + leverage) * eps ** 2 + beta * sigma2[t - 1]
        returns[t] = mu + eps

    index = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(returns, index=index, name="log_return")


# ---------------------------------------------------------------------------
# Test: VaR sign conventions
# ---------------------------------------------------------------------------

class TestVaRConventions:
    """VaR should be negative (loss threshold) for typical equity returns."""

    def test_historical_var_is_negative(self, synthetic_returns):
        var = historical_simulation_var(synthetic_returns, window=252, confidence=0.99)
        valid = var.dropna()
        assert len(valid) > 0
        # Most VaR values should be negative for reasonable equity returns
        assert (valid < 0).mean() > 0.9

    def test_normal_var_is_negative(self, synthetic_returns):
        var = parametric_normal_var(synthetic_returns, window=252, confidence=0.99)
        valid = var.dropna()
        assert len(valid) > 0
        assert (valid < 0).mean() > 0.9

    def test_student_t_var_is_negative(self, synthetic_returns):
        var = parametric_student_t_var(synthetic_returns, window=252, confidence=0.99)
        valid = var.dropna()
        assert len(valid) > 0
        assert (valid < 0).mean() > 0.9


# ---------------------------------------------------------------------------
# Test: VaR ordering (99% more extreme than 95%)
# ---------------------------------------------------------------------------

class TestVaROrdering:
    """VaR at higher confidence should be more extreme (more negative)."""

    def test_historical_ordering(self, synthetic_returns):
        var_95 = historical_simulation_var(synthetic_returns, window=252, confidence=0.95)
        var_99 = historical_simulation_var(synthetic_returns, window=252, confidence=0.99)
        common = var_95.dropna().index.intersection(var_99.dropna().index)
        assert len(common) > 0
        # 99% VaR should be <= 95% VaR (more negative)
        assert (var_99.loc[common] <= var_95.loc[common]).mean() > 0.95

    def test_normal_ordering(self, synthetic_returns):
        var_95 = parametric_normal_var(synthetic_returns, window=252, confidence=0.95)
        var_99 = parametric_normal_var(synthetic_returns, window=252, confidence=0.99)
        common = var_95.dropna().index.intersection(var_99.dropna().index)
        assert len(common) > 0
        # Allow for small floating point issues
        assert (var_99.loc[common] <= var_95.loc[common] + 1e-10).all()


# ---------------------------------------------------------------------------
# Test: GJR-GARCH asymmetry
# ---------------------------------------------------------------------------

class TestGJRGARCH:
    """GJR-GARCH should detect leverage effect in asymmetric data."""

    def test_gamma_positive(self, asymmetric_returns):
        """Gamma should be positive for data with leverage effect."""
        result = fit_gjr_garch(asymmetric_returns, distribution="studentt")
        assert result.gamma > 0, f"Expected gamma > 0, got {result.gamma}"

    def test_news_impact_curve_asymmetric(self, asymmetric_returns):
        """NIC should show higher variance for negative shocks."""
        result = fit_gjr_garch(asymmetric_returns, distribution="studentt")
        shocks, sigma2_next = news_impact_curve(result)

        # Find variance at symmetric points
        n = len(shocks)
        neg_idx = n // 4  # ~-1.5 sigma
        pos_idx = 3 * n // 4  # ~+1.5 sigma

        # Negative shock should produce higher next-period variance
        assert sigma2_next[neg_idx] > sigma2_next[pos_idx], \
            "NIC should show higher variance for negative shocks (leverage effect)"

    def test_persistence_less_than_one(self, asymmetric_returns):
        """Persistence should be < 1 for stationarity."""
        result = fit_gjr_garch(asymmetric_returns, distribution="studentt")
        assert result.persistence < 1.0

    def test_fhs_produces_results(self, asymmetric_returns):
        """FHS should produce valid VaR estimates."""
        result = fit_gjr_garch(asymmetric_returns, distribution="studentt")
        fhs = simulate_fhs(result, n_simulations=1000, horizon=1, seed=42)
        assert fhs.var_99 < fhs.var_95 < 0
        # ES is always more extreme than VaR at same confidence level
        assert fhs.es_99 < fhs.var_99
        assert fhs.es_95 < fhs.var_95


# ---------------------------------------------------------------------------
# Test: Statistical tests
# ---------------------------------------------------------------------------

class TestStatisticalTests:
    """Kupiec and Christoffersen tests should behave correctly."""

    def test_kupiec_rejects_bad_model(self):
        """50% exceedance rate should be rejected at 99% confidence."""
        # 99% VaR should have ~1% exceedances; 50% is way off
        stat, p_value = kupiec_test(n_total=1000, n_exceedances=500, confidence=0.99)
        assert p_value < 0.01, f"Kupiec should reject 50% rate, got p={p_value}"

    def test_kupiec_accepts_correct_model(self):
        """~1% exceedance rate should not be rejected at 99% confidence."""
        stat, p_value = kupiec_test(n_total=1000, n_exceedances=10, confidence=0.99)
        assert p_value > 0.05, f"Kupiec should not reject 1% rate, got p={p_value}"

    def test_christoffersen_rejects_clustering(self):
        """Clustered exceedances should be rejected."""
        # Create highly clustered exceedances: blocks of 1s
        exc = np.zeros(1000, dtype=int)
        exc[100:120] = 1
        exc[300:320] = 1
        exc[500:520] = 1
        exc[700:720] = 1
        stat, p_value = christoffersen_test(exc)
        assert p_value < 0.05, f"Christoffersen should reject clustering, got p={p_value}"

    def test_christoffersen_accepts_independent(self):
        """Randomly scattered exceedances should not be rejected."""
        rng = np.random.default_rng(42)
        exc = (rng.random(1000) < 0.01).astype(int)  # ~1% random
        stat, p_value = christoffersen_test(exc)
        # With random exceedances, we should NOT reject independence
        # (though with small sample randomness, allow p > 0.01)
        assert p_value > 0.01

    def test_combined_coverage(self):
        """Combined test should accumulate LR statistics."""
        cc = combined_coverage_test(
            n_total=1000, n_exceedances=10,
            exceedances=np.zeros(1000), confidence=0.99,
        )
        assert cc.lr_cc == cc.lr_uc + cc.lr_ind
        assert 0 <= cc.p_cc <= 1

    def test_exceedance_test_basic(self, synthetic_returns):
        """Exceedance test should count correctly."""
        # Create a VaR series that's always 0 (everything breaches)
        zero_var = pd.Series(0.0, index=synthetic_returns.index)
        result = var_exceedance_test(synthetic_returns, zero_var, confidence=0.99)
        # Almost all negative returns should be exceedances
        assert result["n_exceedances"] > 0
        assert result["n_obs"] == len(synthetic_returns)


# ---------------------------------------------------------------------------
# Test: Data utilities
# ---------------------------------------------------------------------------

class TestDataUtils:
    """Test compute_log_returns utility."""

    def test_log_returns_length(self):
        """Log returns should have one fewer observation than prices."""
        prices = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        returns = compute_log_returns(prices)
        assert len(returns) == 4

    def test_log_returns_values(self):
        """Log returns should be ln(P_t / P_{t-1})."""
        prices = pd.DataFrame({"close": [100.0, 110.0, 105.0]})
        returns = compute_log_returns(prices)
        expected = [np.log(110 / 100), np.log(105 / 110)]
        np.testing.assert_allclose(returns.values, expected, rtol=1e-10)

    def test_log_returns_no_nan(self):
        """Output should not contain NaN values."""
        prices = pd.DataFrame({"close": np.arange(100, 200, dtype=float)})
        returns = compute_log_returns(prices)
        assert not returns.isna().any()
