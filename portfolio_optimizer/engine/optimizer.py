"""Portfolio optimization algorithms: Mean-Variance, Risk Parity, Black-Litterman."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .data import MarketData


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization."""

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str
    extra: dict


class MeanVarianceOptimizer:
    """Mean-Variance (Markowitz) portfolio optimizer using SLSQP.

    Parameters
    ----------
    data : MarketData
        Market data with covariance and expected returns.
    long_only : bool
        If True, enforce w >= 0.
    max_weight : float
        Maximum weight per asset.
    """

    def __init__(
        self,
        data: MarketData,
        long_only: bool = True,
        max_weight: float = 1.0,
    ) -> None:
        self.data = data
        self.mu = data.expected_returns
        self.cov = data.cov_matrix
        self.rf = data.risk_free_rate
        self.n = data.n_assets
        self.long_only = long_only
        self.max_weight = max_weight

    def _bounds(self):
        lb = 0.0 if self.long_only else -1.0
        return [(lb, self.max_weight)] * self.n

    def _constraints(self):
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def _portfolio_vol(self, w):
        return float(np.sqrt(w @ self.cov @ w))

    def _portfolio_return(self, w):
        return float(w @ self.mu)

    def _make_result(self, w, method, **extra):
        ret = self._portfolio_return(w)
        vol = self._portfolio_vol(w)
        sharpe = (ret - self.rf) / vol if vol > 1e-10 else 0.0
        return OptimizationResult(
            weights=w,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method=method,
            extra=extra,
        )

    def min_variance(self) -> OptimizationResult:
        """Minimum variance portfolio."""
        w0 = np.ones(self.n) / self.n
        res = minimize(
            lambda w: w @ self.cov @ w,
            w0,
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._constraints(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        return self._make_result(res.x, "min_variance")

    def max_sharpe(self) -> OptimizationResult:
        """Maximum Sharpe ratio portfolio."""
        w0 = np.ones(self.n) / self.n

        def neg_sharpe(w: np.ndarray) -> float:
            ret = w @ self.mu
            vol = np.sqrt(w @ self.cov @ w)
            if vol < 1e-10:
                return 0.0
            return -(ret - self.rf) / vol

        res = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._constraints(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        return self._make_result(res.x, "max_sharpe")

    def target_return(self, target: float) -> OptimizationResult:
        """Minimum variance portfolio with a target return constraint."""
        w0 = np.ones(self.n) / self.n
        constraints = self._constraints() + [
            {"type": "eq", "fun": lambda w: w @ self.mu - target}
        ]
        res = minimize(
            lambda w: w @ self.cov @ w,
            w0,
            method="SLSQP",
            bounds=self._bounds(),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        return self._make_result(res.x, "target_return", target=target)

    def efficient_frontier(self, n_points: int = 50) -> list[OptimizationResult]:
        """Compute the efficient frontier by sweeping target returns."""
        min_var = self.min_variance()
        min_ret = min_var.expected_return

        # Find max achievable return
        max_ret = float(np.max(self.mu))
        if self.long_only:
            # Max return is the highest single-asset return
            pass
        else:
            max_ret = max_ret * 1.5  # Allow some headroom for short positions

        targets = np.linspace(min_ret, max_ret, n_points)
        frontier = []
        for t in targets:
            try:
                result = self.target_return(t)
                frontier.append(result)
            except Exception:
                continue
        return frontier
