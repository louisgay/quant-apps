"""Unit tests for HRP allocation engine."""

import numpy as np
import pandas as pd
import pytest

from engine.clustering import (
    correlation_distance,
    hierarchical_cluster,
    quasi_diagonalize,
    get_cluster_members,
)
from engine.hrp import hrp_allocation, recursive_bisection
from engine.optimizers import (
    equal_weight,
    inverse_volatility,
    risk_parity_optimize,
    mean_variance_optimize,
)
from engine.analytics import (
    compute_metrics,
    effective_number_of_bets,
    Backtester,
)
from engine.data import ledoit_wolf_shrinkage


@pytest.fixture
def identity_cov():
    n = 5
    assets = [f"A{i}" for i in range(n)]
    return pd.DataFrame(np.eye(n), index=assets, columns=assets)


@pytest.fixture
def block_diagonal_cov():
    """Two clusters: (A0,A1,A2) with ρ=0.8, (A3,A4) with ρ=0.8, zero inter-cluster."""
    n = 5
    assets = [f"A{i}" for i in range(n)]

    corr = np.eye(n)
    corr[0, 1] = corr[1, 0] = 0.8
    corr[0, 2] = corr[2, 0] = 0.8
    corr[1, 2] = corr[2, 1] = 0.8
    corr[3, 4] = corr[4, 3] = 0.8

    std = np.ones(n) * 0.2
    cov = pd.DataFrame(
        corr * np.outer(std, std), index=assets, columns=assets
    )
    return cov


@pytest.fixture
def realistic_cov():
    """Approximate multi-asset correlations (equities + bonds + gold)."""
    assets = ["SPY", "QQQ", "EFA", "TLT", "GLD", "VNQ", "HYG"]
    n = len(assets)

    corr = np.array([
        [1.00, 0.90, 0.75, -0.30, 0.05, 0.70, 0.65],
        [0.90, 1.00, 0.70, -0.35, 0.00, 0.60, 0.55],
        [0.75, 0.70, 1.00, -0.20, 0.10, 0.55, 0.50],
        [-0.30, -0.35, -0.20, 1.00, 0.25, -0.10, -0.20],
        [0.05, 0.00, 0.10, 0.25, 1.00, 0.10, 0.05],
        [0.70, 0.60, 0.55, -0.10, 0.10, 1.00, 0.60],
        [0.65, 0.55, 0.50, -0.20, 0.05, 0.60, 1.00],
    ])

    std = np.array([0.16, 0.20, 0.15, 0.12, 0.14, 0.18, 0.08])
    cov = pd.DataFrame(
        corr * np.outer(std, std), index=assets, columns=assets
    )
    return cov


@pytest.fixture
def synthetic_returns():
    np.random.seed(42)
    n_assets = 5
    n_days = 500
    assets = [f"A{i}" for i in range(n_assets)]

    corr = np.eye(n_assets)
    corr[0, 1] = corr[1, 0] = 0.7
    corr[2, 3] = corr[3, 2] = 0.6
    L = np.linalg.cholesky(corr)

    returns_raw = np.random.randn(n_days, n_assets) @ L.T * 0.01
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(returns_raw, index=dates, columns=assets)


class TestCorrelationDistance:
    def test_perfect_correlation_gives_zero_distance(self):
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        dist = correlation_distance(corr)
        assert dist[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_zero_correlation_gives_sqrt_half(self):
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        dist = correlation_distance(corr)
        assert dist[0, 1] == pytest.approx(np.sqrt(0.5), rel=1e-10)

    def test_negative_correlation_gives_max_distance(self):
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        dist = correlation_distance(corr)
        assert dist[0, 1] == pytest.approx(1.0, abs=1e-10)

    def test_diagonal_is_zero(self):
        corr = np.random.uniform(-1, 1, (5, 5))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        dist = correlation_distance(corr)
        assert np.all(np.diag(dist) == 0.0)

    def test_symmetry(self):
        corr = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]])
        dist = correlation_distance(corr)
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_non_negative(self):
        np.random.seed(0)
        corr = np.random.uniform(-1, 1, (10, 10))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        dist = correlation_distance(corr)
        assert np.all(dist >= -1e-10)


class TestHierarchicalClustering:
    def test_linkage_shape(self, identity_cov):
        n = len(identity_cov)
        dist = correlation_distance(identity_cov.values)
        Z = hierarchical_cluster(dist)
        assert Z.shape == (n - 1, 4)

    def test_leaf_count_matches_assets(self, block_diagonal_cov):
        n = len(block_diagonal_cov)
        corr = block_diagonal_cov.values / np.outer(
            np.sqrt(np.diag(block_diagonal_cov.values)),
            np.sqrt(np.diag(block_diagonal_cov.values)),
        )
        dist = correlation_distance(corr)
        Z = hierarchical_cluster(dist)
        assert int(Z[-1, 3]) == n


class TestQuasiDiagonalize:
    def test_preserves_eigenvalues(self, realistic_cov):
        dist = correlation_distance(realistic_cov.values)
        Z = hierarchical_cluster(dist)
        cov_reordered, _ = quasi_diagonalize(realistic_cov, Z)

        eigs_original = sorted(np.linalg.eigvalsh(realistic_cov.values))
        eigs_reordered = sorted(np.linalg.eigvalsh(cov_reordered))

        np.testing.assert_array_almost_equal(eigs_original, eigs_reordered)

    def test_sort_order_is_permutation(self, realistic_cov):
        dist = correlation_distance(realistic_cov.values)
        Z = hierarchical_cluster(dist)
        _, sort_order = quasi_diagonalize(realistic_cov, Z)

        assert sorted(sort_order) == list(range(len(realistic_cov)))


class TestHRPAllocation:
    def test_weights_sum_to_one(self, realistic_cov):
        weights = hrp_allocation(realistic_cov)
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_weights_non_negative(self, realistic_cov):
        weights = hrp_allocation(realistic_cov)
        assert (weights >= -1e-10).all()

    def test_identity_cov_gives_equal_weight(self, identity_cov):
        weights = hrp_allocation(identity_cov)
        expected = 1.0 / len(identity_cov)
        np.testing.assert_array_almost_equal(
            weights.values, np.full(len(identity_cov), expected), decimal=5
        )

    def test_block_diagonal_clusters_detected(self, block_diagonal_cov):
        weights = hrp_allocation(block_diagonal_cov)

        cluster1_weight = weights["A0"] + weights["A1"] + weights["A2"]
        cluster2_weight = weights["A3"] + weights["A4"]

        assert cluster1_weight > 0.2
        assert cluster2_weight > 0.2

        # Within-cluster weights should be roughly equal (Ward introduces slight asymmetry)
        cluster1_weights = weights[["A0", "A1", "A2"]].values
        assert np.std(cluster1_weights) < 0.06

    def test_correct_index(self, realistic_cov):
        weights = hrp_allocation(realistic_cov)
        assert list(weights.index) == list(realistic_cov.index)

    def test_low_vol_gets_more_weight(self):
        assets = ["low_vol", "med_vol", "high_vol"]
        variances = [0.01, 0.04, 0.09]
        cov = pd.DataFrame(np.diag(variances), index=assets, columns=assets)
        weights = hrp_allocation(cov)
        assert weights["low_vol"] > weights["med_vol"] > weights["high_vol"]


class TestOptimizers:
    def test_equal_weight_uniform(self, realistic_cov):
        w = equal_weight(realistic_cov)
        expected = 1.0 / len(realistic_cov)
        np.testing.assert_array_almost_equal(w.values, expected)

    def test_inverse_vol_sums_to_one(self, realistic_cov):
        w = inverse_volatility(realistic_cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)

    def test_inverse_vol_favors_low_vol(self, realistic_cov):
        w = inverse_volatility(realistic_cov)
        # HYG has lowest vol (0.08)
        assert w["HYG"] == w.max()

    def test_risk_parity_sums_to_one(self, realistic_cov):
        w = risk_parity_optimize(realistic_cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-8)

    def test_risk_parity_non_negative(self, realistic_cov):
        w = risk_parity_optimize(realistic_cov)
        assert (w >= 0).all()

    def test_mean_variance_sums_to_one(self, realistic_cov):
        w = mean_variance_optimize(realistic_cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-8)

    def test_mean_variance_non_negative(self, realistic_cov):
        w = mean_variance_optimize(realistic_cov)
        assert (w >= -1e-8).all()


class TestAnalytics:
    def test_effective_bets_equal_weight(self):
        n = 10
        w = np.ones(n) / n
        assert effective_number_of_bets(w) == pytest.approx(n, rel=1e-5)

    def test_effective_bets_concentrated(self):
        w = np.array([1.0, 0.0, 0.0, 0.0])
        assert effective_number_of_bets(w) == pytest.approx(1.0, rel=1e-5)

    def test_effective_bets_bounded(self):
        w = np.array([0.5, 0.3, 0.15, 0.05])
        n_eff = effective_number_of_bets(w)
        assert 1.0 <= n_eff <= len(w)

    def test_metrics_computation(self, synthetic_returns):
        port_returns = synthetic_returns.mean(axis=1)
        metrics = compute_metrics(port_returns)
        assert isinstance(metrics.sharpe_ratio, float)
        assert metrics.max_drawdown <= 0

    def test_backtester_no_lookahead(self, synthetic_returns):
        bt = Backtester(
            synthetic_returns,
            allocation_fn=equal_weight,
            window=100,
            rebalance_freq=21,
        )
        port_returns, weights_hist = bt.run()

        # Should start after estimation window
        assert len(port_returns) == len(synthetic_returns) - 100

        for _, row in weights_hist.iterrows():
            assert row.sum() == pytest.approx(1.0, abs=1e-8)


class TestLedoitWolf:
    def test_shrinkage_produces_psd(self, synthetic_returns):
        cov = ledoit_wolf_shrinkage(synthetic_returns.iloc[:100])
        eigvals = np.linalg.eigvalsh(cov.values)
        assert (eigvals >= -1e-10).all()

    def test_shrunk_matrix_is_symmetric(self, synthetic_returns):
        cov = ledoit_wolf_shrinkage(synthetic_returns.iloc[:100])
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)
