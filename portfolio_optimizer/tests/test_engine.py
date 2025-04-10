"""Tests for portfolio optimizer engine using synthetic data (no yfinance)."""

import numpy as np
import pandas as pd
import pytest

from engine.data import MarketData
from engine.optimizer import (
    BlackLittermanOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
)
from engine.analytics import Backtester, PortfolioMetrics, drawdown_series, rolling_sharpe


def _make_synthetic_returns(n_assets: int, n_days: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic daily returns with realistic correlation."""
    rng = np.random.RandomState(seed)
    # Random correlation structure
    A = rng.randn(n_assets, n_assets)
    cov_daily = A @ A.T / n_assets * 0.0004  # ~2% annual vol per asset
    mu_daily = rng.uniform(0.0002, 0.0006, n_assets)
    returns = rng.multivariate_normal(mu_daily, cov_daily, n_days)
    return returns, mu_daily * 252, cov_daily * 252


def _make_market_data(
    n_assets: int = 3,
    n_days: int = 504,
    seed: int = 42,
    market_caps: bool = False,
) -> MarketData:
    """Build a MarketData object from synthetic data."""
    tickers = [f"ASSET{i}" for i in range(n_assets)]
    returns_arr, mu_annual, cov_annual = _make_synthetic_returns(n_assets, n_days, seed)

    # Build prices from returns
    prices_arr = 100 * np.cumprod(1 + returns_arr, axis=0)
    dates = pd.bdate_range(end="2024-01-01", periods=n_days)

    prices = pd.DataFrame(prices_arr, index=dates, columns=tickers)
    returns = pd.DataFrame(returns_arr, index=dates, columns=tickers)

    mc = np.array([1e9 * (i + 1) for i in range(n_assets)]) if market_caps else None

    return MarketData(
        tickers=tickers,
        prices=prices,
        returns=returns,
        cov_matrix=cov_annual,
        expected_returns=mu_annual,
        risk_free_rate=0.02,
        market_caps=mc,
    )


@pytest.fixture
def simple_market() -> MarketData:
    """3-asset market with 504 days of data."""
    return _make_market_data(n_assets=3, n_days=504)


@pytest.fixture
def two_asset_market() -> MarketData:
    """2 identical assets for symmetry tests."""
    n_days = 504
    rng = np.random.RandomState(99)
    vol = 0.02 / np.sqrt(252)
    ret = rng.normal(0.0003, vol, (n_days, 1))
    returns_arr = np.hstack([ret, ret])  # Identical assets

    mu = np.array([0.0003, 0.0003]) * 252
    cov_daily = np.array([[vol**2, vol**2], [vol**2, vol**2]])
    cov_annual = cov_daily * 252

    # Make cov slightly non-singular for numerical stability
    cov_annual += np.eye(2) * 1e-8

    tickers = ["A", "B"]
    dates = pd.bdate_range(end="2024-01-01", periods=n_days)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns_arr, axis=0), index=dates, columns=tickers
    )
    returns = pd.DataFrame(returns_arr, index=dates, columns=tickers)

    return MarketData(
        tickers=tickers,
        prices=prices,
        returns=returns,
        cov_matrix=cov_annual,
        expected_returns=mu,
        risk_free_rate=0.02,
    )


class TestMarketData:
    def test_shapes(self, simple_market: MarketData):
        assert simple_market.n_assets == 3
        assert simple_market.cov_matrix.shape == (3, 3)
        assert simple_market.expected_returns.shape == (3,)

    def test_psd(self, simple_market: MarketData):
        eigvals = np.linalg.eigvalsh(simple_market.cov_matrix)
        assert np.all(eigvals >= -1e-8)

    def test_symmetry(self, simple_market: MarketData):
        assert np.allclose(
            simple_market.cov_matrix, simple_market.cov_matrix.T, atol=1e-8
        )

    def test_correlation_diagonal(self, simple_market: MarketData):
        corr = simple_market.correlation_matrix
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_annualized_vols_positive(self, simple_market: MarketData):
        assert np.all(simple_market.annualized_vols >= 0)

    def test_invalid_cov_shape(self):
        with pytest.raises(ValueError, match="cov_matrix shape"):
            MarketData(
                tickers=["A", "B"],
                prices=pd.DataFrame(),
                returns=pd.DataFrame(),
                cov_matrix=np.eye(3),
                expected_returns=np.array([0.1, 0.1]),
            )

    def test_non_symmetric_cov(self):
        cov = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="not symmetric"):
            MarketData(
                tickers=["A", "B"],
                prices=pd.DataFrame(),
                returns=pd.DataFrame(),
                cov_matrix=cov,
                expected_returns=np.array([0.1, 0.1]),
            )

    def test_market_cap_weights(self):
        data = _make_market_data(n_assets=3, market_caps=True)
        w = data.market_cap_weights
        assert w is not None
        np.testing.assert_allclose(w.sum(), 1.0)
        assert np.all(w > 0)


class TestMeanVariance:
    def test_weights_sum_one(self, simple_market: MarketData):
        opt = MeanVarianceOptimizer(simple_market)
        result = opt.min_variance()
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_long_only(self, simple_market: MarketData):
        opt = MeanVarianceOptimizer(simple_market, long_only=True)
        result = opt.min_variance()
        assert np.all(result.weights >= -1e-8)

    def test_max_weight_constraint(self, simple_market: MarketData):
        opt = MeanVarianceOptimizer(simple_market, max_weight=0.5)
        result = opt.max_sharpe()
        assert np.all(result.weights <= 0.5 + 1e-6)

    def test_max_sharpe_higher_return(self, simple_market: MarketData):
        opt = MeanVarianceOptimizer(simple_market)
        mv = opt.min_variance()
        ms = opt.max_sharpe()
        assert ms.expected_return >= mv.expected_return - 1e-8

    def test_efficient_frontier_monotone(self, simple_market: MarketData):
        opt = MeanVarianceOptimizer(simple_market)
        frontier = opt.efficient_frontier(n_points=20)
        returns = [r.expected_return for r in frontier]
        # Returns should be non-decreasing
        for i in range(1, len(returns)):
            assert returns[i] >= returns[i - 1] - 1e-6

    def test_target_return(self, simple_market: MarketData):
        opt = MeanVarianceOptimizer(simple_market)
        target = float(np.mean(simple_market.expected_returns))
        result = opt.target_return(target)
        np.testing.assert_allclose(result.expected_return, target, atol=1e-4)

    def test_weights_sum_short_allowed(self, simple_market: MarketData):
        opt = MeanVarianceOptimizer(simple_market, long_only=False)
        result = opt.min_variance()
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-6)


class TestRiskParity:
    def test_weights_sum_one(self, simple_market: MarketData):
        opt = RiskParityOptimizer(simple_market)
        result = opt.optimize()
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_weights_positive(self, simple_market: MarketData):
        opt = RiskParityOptimizer(simple_market)
        result = opt.optimize()
        assert np.all(result.weights > 0)

    def test_risk_contributions_equal(self, simple_market: MarketData):
        opt = RiskParityOptimizer(simple_market)
        result = opt.optimize()
        rc = result.extra["risk_contributions"]
        # All risk contributions should be approximately 1/n
        np.testing.assert_allclose(rc, 1.0 / simple_market.n_assets, atol=0.02)

    def test_identical_assets_equal_weights(self, two_asset_market: MarketData):
        opt = RiskParityOptimizer(two_asset_market)
        result = opt.optimize()
        np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=0.01)


class TestBlackLitterman:
    def test_no_views_equals_equilibrium(self, simple_market: MarketData):
        data = _make_market_data(n_assets=3, market_caps=True)
        bl = BlackLittermanOptimizer(data)
        mu_bl, _ = bl.posterior(P=None, Q=None)
        pi = bl.equilibrium_returns()
        np.testing.assert_allclose(mu_bl, pi, atol=1e-8)

    def test_posterior_shifts_toward_view(self):
        data = _make_market_data(n_assets=3, market_caps=True)
        bl = BlackLittermanOptimizer(data)
        pi = bl.equilibrium_returns()

        # View: asset 0 will return 20%
        P = np.array([[1, 0, 0]])
        Q = np.array([0.20])
        mu_bl, _ = bl.posterior(P, Q)

        # Posterior for asset 0 should shift toward 20%
        if pi[0] < 0.20:
            assert mu_bl[0] > pi[0]
        else:
            assert mu_bl[0] < pi[0]

    def test_weights_sum_one(self):
        data = _make_market_data(n_assets=3, market_caps=True)
        bl = BlackLittermanOptimizer(data)
        result = bl.optimize()
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_optimize_with_views(self):
        data = _make_market_data(n_assets=3, market_caps=True)
        bl = BlackLittermanOptimizer(data)
        P = np.array([[1, -1, 0]])  # Asset 0 outperforms Asset 1
        Q = np.array([0.05])        # By 5%
        result = bl.optimize(P, Q)
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-6)
        assert result.method == "black_litterman"


class TestPortfolioMetrics:
    def test_sharpe_finite(self, simple_market: MarketData):
        returns = simple_market.returns.mean(axis=1).values
        pm = PortfolioMetrics(returns)
        assert np.isfinite(pm.sharpe_ratio)

    def test_max_drawdown_known(self):
        # Create a simple up-then-down series
        returns = np.array([0.1, 0.1, -0.3, 0.05])
        pm = PortfolioMetrics(returns)
        assert pm.max_drawdown < 0  # Drawdown is negative

    def test_var_negative(self, simple_market: MarketData):
        returns = simple_market.returns.mean(axis=1).values
        pm = PortfolioMetrics(returns)
        var = pm.var_historical(0.95)
        # VaR at 95% should typically be negative for a portfolio
        assert var < 0 or np.isfinite(var)

    def test_cvar_le_var(self, simple_market: MarketData):
        returns = simple_market.returns.mean(axis=1).values
        pm = PortfolioMetrics(returns)
        var = pm.var_historical(0.95)
        cvar = pm.cvar(0.95)
        assert cvar <= var + 1e-10  # CVaR is at least as bad as VaR

    def test_full_report_keys(self, simple_market: MarketData):
        returns = simple_market.returns.mean(axis=1).values
        pm = PortfolioMetrics(returns)
        report = pm.full_report()
        expected_keys = {
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "var_95",
            "cvar_95",
        }
        assert set(report.keys()) == expected_keys

    def test_sortino_ratio_finite(self, simple_market: MarketData):
        returns = simple_market.returns.mean(axis=1).values
        pm = PortfolioMetrics(returns)
        assert np.isfinite(pm.sortino_ratio)


class TestBacktester:
    def _factory(self, data: MarketData) -> dict:
        """Simple equal-weight factory for testing."""
        n = data.n_assets
        return {"weights": np.ones(n) / n}

    def test_equity_starts_at_one(self, simple_market: MarketData):
        bt = Backtester(self._factory, rebalance_freq=21, lookback_window=252)
        result = bt.run(simple_market)
        assert result.equity_curve[0] == 1.0

    def test_benchmark_starts_at_one(self, simple_market: MarketData):
        bt = Backtester(self._factory, rebalance_freq=21, lookback_window=252)
        result = bt.run(simple_market)
        assert result.benchmark_curve[0] == 1.0

    def test_turnover_non_negative(self, simple_market: MarketData):
        bt = Backtester(self._factory, rebalance_freq=21, lookback_window=252)
        result = bt.run(simple_market)
        assert all(t >= 0 for t in result.turnover)

    def test_result_keys(self, simple_market: MarketData):
        bt = Backtester(self._factory, rebalance_freq=21, lookback_window=252)
        result = bt.run(simple_market)
        assert hasattr(result, "equity_curve")
        assert hasattr(result, "benchmark_curve")
        assert hasattr(result, "portfolio_returns")
        assert hasattr(result, "turnover")
        assert hasattr(result, "metrics")
        assert hasattr(result, "benchmark_metrics")

    def test_equity_curve_length(self, simple_market: MarketData):
        bt = Backtester(self._factory, rebalance_freq=21, lookback_window=252)
        result = bt.run(simple_market)
        expected_len = simple_market.returns.shape[0] - 252 + 1
        assert len(result.equity_curve) == expected_len


class TestUtilities:
    def test_drawdown_series(self):
        equity = np.array([1.0, 1.1, 1.05, 1.15, 0.9])
        dd = drawdown_series(equity)
        assert dd[0] == 0.0  # First point is always 0
        assert dd[-1] < 0    # Below peak
        assert np.min(dd) == dd[-1]  # Worst drawdown at end

    def test_rolling_sharpe_length(self):
        returns = np.random.randn(200) * 0.01
        rs = rolling_sharpe(returns, window=63)
        assert len(rs) == len(returns)
        assert np.all(np.isnan(rs[:63]))
        assert np.all(np.isfinite(rs[63:]))
