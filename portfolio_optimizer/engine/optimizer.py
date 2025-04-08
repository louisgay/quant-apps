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


class RiskParityOptimizer:
    """Risk Parity (Equal Risk Contribution) optimizer.

    Parameters
    ----------
    data : MarketData
        Market data with covariance matrix.
    """

    def __init__(self, data: MarketData) -> None:
        self.data = data
        self.cov = data.cov_matrix
        self.n = data.n_assets

    def optimize(self) -> OptimizationResult:
        """Find risk parity weights."""
        w0 = np.ones(self.n) / self.n

        def objective(w: np.ndarray) -> float:
            w = np.maximum(w, 1e-10)
            sigma_p_sq = w @ self.cov @ w
            marginal_contrib = self.cov @ w
            risk_contrib = w * marginal_contrib
            target_contrib = sigma_p_sq / self.n
            return float(np.sum((risk_contrib - target_contrib) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-6, 1.0)] * self.n

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-15, "maxiter": 2000},
        )

        weights = res.x
        # Compute risk contributions
        sigma_p_sq = weights @ self.cov @ weights
        marginal_contrib = self.cov @ weights
        risk_contrib = weights * marginal_contrib
        risk_contrib_pct = risk_contrib / sigma_p_sq

        ret = float(weights @ self.data.expected_returns)
        vol = float(np.sqrt(sigma_p_sq))
        sharpe = (ret - self.data.risk_free_rate) / vol if vol > 1e-10 else 0.0

        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method="risk_parity",
            extra={"risk_contributions": risk_contrib_pct},
        )


class BlackLittermanOptimizer:
    """Black-Litterman model with posterior mean-variance optimization.

    Parameters
    ----------
    data : MarketData
        Market data with covariance and market caps.
    tau : float
        Uncertainty scaling parameter (typically 0.025 to 0.05).
    risk_aversion : float
        Market risk aversion coefficient (delta).
    long_only : bool
        If True, enforce w >= 0 in final optimization.
    max_weight : float
        Maximum weight per asset.
    """

    def __init__(
        self,
        data: MarketData,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        long_only: bool = True,
        max_weight: float = 1.0,
    ) -> None:
        self.data = data
        self.cov = data.cov_matrix
        self.n = data.n_assets
        self.tau = tau
        self.delta = risk_aversion
        self.long_only = long_only
        self.max_weight = max_weight

        # Market-cap weights (fallback to equal weight)
        if data.market_caps is not None:
            self.w_mkt = data.market_cap_weights
        else:
            self.w_mkt = np.ones(self.n) / self.n

    def equilibrium_returns(self) -> np.ndarray:
        """Compute implied equilibrium returns: pi = delta * Sigma * w_mkt."""
        return self.delta * self.cov @ self.w_mkt

    def posterior(
        self,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Black-Litterman posterior returns and covariance.

        Parameters
        ----------
        P : np.ndarray, optional
            Pick matrix (k × n), one row per view.
        Q : np.ndarray, optional
            View vector (k,).
        omega : np.ndarray, optional
            View uncertainty matrix (k × k). Defaults to He-Litterman.

        Returns
        -------
        mu_bl : np.ndarray
            Posterior expected returns (n,).
        cov_bl : np.ndarray
            Posterior covariance (n × n).
        """
        pi = self.equilibrium_returns()
        tau_cov = self.tau * self.cov

        if P is None or Q is None:
            # No views: posterior equals equilibrium
            return pi, self.cov + tau_cov

        P = np.atleast_2d(P)
        Q = np.atleast_1d(Q)

        if omega is None:
            # He-Litterman default: Omega = diag(P * tau * Sigma * P')
            omega = np.diag(np.diag(P @ tau_cov @ P.T))

        tau_cov_inv = np.linalg.inv(tau_cov)
        omega_inv = np.linalg.inv(omega)

        # Posterior precision and mean
        posterior_precision = tau_cov_inv + P.T @ omega_inv @ P
        posterior_cov_mu = np.linalg.inv(posterior_precision)
        mu_bl = posterior_cov_mu @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

        # Posterior covariance of returns
        cov_bl = self.cov + posterior_cov_mu

        return mu_bl, cov_bl

    def optimize(
        self,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Run Black-Litterman and then max-Sharpe on the posterior.

        Parameters
        ----------
        P, Q, omega : see posterior().

        Returns
        -------
        OptimizationResult
        """
        mu_bl, cov_bl = self.posterior(P, Q, omega)

        # Build a temporary MarketData with BL posteriors for MV optimization
        bl_data = MarketData(
            tickers=self.data.tickers,
            prices=self.data.prices,
            returns=self.data.returns,
            cov_matrix=cov_bl,
            expected_returns=mu_bl,
            risk_free_rate=self.data.risk_free_rate,
            market_caps=self.data.market_caps,
        )

        mv = MeanVarianceOptimizer(
            bl_data,
            long_only=self.long_only,
            max_weight=self.max_weight,
        )
        result = mv.max_sharpe()
        result.method = "black_litterman"
        result.extra["equilibrium_returns"] = self.equilibrium_returns()
        result.extra["posterior_returns"] = mu_bl
        result.extra["posterior_cov"] = cov_bl
        return result
