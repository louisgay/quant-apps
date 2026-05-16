"""Benchmark allocation methods: Markowitz, Risk Parity, EW, Inverse Vol."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight(cov: pd.DataFrame) -> pd.Series:
    """w_i = 1/N."""
    n = len(cov)
    return pd.Series(np.ones(n) / n, index=cov.index)


def inverse_volatility(cov: pd.DataFrame) -> pd.Series:
    """w_i ∝ 1/σ_i. Ignores correlations entirely."""
    variances = np.diag(cov.values)
    inv_vol = 1.0 / np.sqrt(np.maximum(variances, 1e-10))
    weights = inv_vol / inv_vol.sum()
    return pd.Series(weights, index=cov.index)


def risk_parity_optimize(cov: pd.DataFrame) -> pd.Series:
    """Equalize risk contributions: RC_i = w_i * (Σw)_i = σ²_p / N for all i.

    Solved via SLSQP minimizing Σ(RC_i - target)².
    """
    n = len(cov)
    sigma = cov.values

    def objective(w):
        w = np.array(w)
        port_var = w @ sigma @ w
        marginal_risk = sigma @ w
        risk_contrib = w * marginal_risk
        target = port_var / n
        return np.sum((risk_contrib - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01 / n, 1.0)] * n

    w0 = inverse_volatility(cov).values

    result = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        return inverse_volatility(cov)

    weights = result.x / result.x.sum()
    return pd.Series(weights, index=cov.index)


def mean_variance_optimize(
    cov: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    target_return: float | None = None,
    risk_aversion: float = 1.0,
) -> pd.Series:
    """Min-variance (or mean-variance if μ provided). Long-only via SLSQP.

    Note: implicitly inverts Σ through the optimizer — unstable when κ(Σ) >> 1.
    """
    n = len(cov)
    sigma = cov.values

    if expected_returns is None:
        mu = np.zeros(n)
        risk_aversion = 0.0
    else:
        mu = expected_returns.values

    def objective(w):
        return 0.5 * (w @ sigma @ w) - risk_aversion * (mu @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w: mu @ w - target_return}
        )

    bounds = [(0.0, 1.0)] * n

    best_result = None
    best_obj = np.inf
    rng = np.random.default_rng(42)

    for _ in range(5):
        w0 = rng.dirichlet(np.ones(n))
        result = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success and result.fun < best_obj:
            best_obj = result.fun
            best_result = result

    if best_result is None:
        return equal_weight(cov)

    weights = best_result.x / best_result.x.sum()
    return pd.Series(weights, index=cov.index)
