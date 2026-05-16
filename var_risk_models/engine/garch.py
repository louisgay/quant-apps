"""GJR-GARCH(1,1) fitting + FHS Monte-Carlo (Barone-Adesi et al. 1999)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GARCHResult:
    """Fitted GJR-GARCH(1,1) model."""

    omega: float
    alpha: float
    gamma: float  # leverage coeff
    beta: float
    mu: float
    nu: Optional[float]  # Student-t df
    distribution: str
    conditional_vol: np.ndarray
    standardized_resids: np.ndarray
    log_likelihood: float
    aic: float
    bic: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta

    @property
    def half_life(self) -> float:
        """ln(0.5) / ln(persistence) — days for vol shock to halve."""
        p = self.persistence
        if p >= 1.0 or p <= 0.0:
            return np.inf
        return np.log(0.5) / np.log(p)


@dataclass
class FHSResult:
    simulated_returns: np.ndarray
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    simulated_paths: Optional[np.ndarray] = field(default=None)


def fit_gjr_garch(
    returns: pd.Series | np.ndarray,
    distribution: str = "studentt",
) -> GARCHResult:
    """MLE fit via `arch` library. Input returns in decimal (not pct)."""
    from arch import arch_model

    r = np.asarray(returns, dtype=float)
    r_pct = r * 100.0  # arch convention

    dist_map = {"studentt": "t", "normal": "normal", "t": "t"}
    dist_name = dist_map.get(distribution, "t")

    model = arch_model(
        r_pct,
        mean="Constant",
        vol="GARCH",
        p=1, o=1, q=1,  # GJR: o=1 adds the leverage (gamma) term
        dist=dist_name,
    )
    res = model.fit(disp="off", show_warning=False)

    params = res.params
    omega = float(params.get("omega", 0))
    alpha = float(params.get("alpha[1]", 0))
    gamma = float(params.get("gamma[1]", 0))
    beta = float(params.get("beta[1]", 0))
    mu = float(params.get("mu", 0))

    nu = None
    if dist_name == "t":
        nu = float(params.get("nu", 30))

    cond_vol = np.asarray(res.conditional_volatility) / 100.0  # back to decimal

    if hasattr(res, "std_resid"):
        std_resids = np.asarray(res.std_resid)
    else:
        std_resids = np.asarray(res.resid) / np.asarray(res.conditional_volatility)

    return GARCHResult(
        omega=omega / 10000.0,  # convert from percent^2 to decimal^2
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        mu=mu / 100.0,  # convert mu back from percent
        nu=nu,
        distribution=distribution,
        conditional_vol=cond_vol,
        standardized_resids=std_resids,
        log_likelihood=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
    )


def news_impact_curve(result: GARCHResult, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """sigma^2_{t+1} as a function of epsilon_t — visualises leverage asymmetry."""
    # Baseline: unconditional variance
    persistence = result.persistence
    if persistence < 1.0:
        sigma2_bar = result.omega / (1.0 - persistence)
    else:
        sigma2_bar = np.mean(result.conditional_vol ** 2)

    # Shock range: [-3sigma, +3sigma]
    sigma_bar = np.sqrt(sigma2_bar)
    shocks = np.linspace(-3 * sigma_bar, 3 * sigma_bar, n_points)

    sigma2_next = np.zeros(n_points)
    for i, eps in enumerate(shocks):
        leverage = result.gamma if eps < 0 else 0.0
        sigma2_next[i] = (
            result.omega
            + (result.alpha + leverage) * eps ** 2
            + result.beta * sigma2_bar
        )

    return shocks, sigma2_next


def simulate_fhs(
    result: GARCHResult,
    n_simulations: int = 10000,
    horizon: int = 1,
    seed: int = 42,
) -> FHSResult:
    """Bootstrap z* from standardised residuals, propagate through GJR-GARCH filter.

    r*_{t+h} = mu + sigma_{t+h} * z*  where z* ~ empirical{z_t}
    """
    rng = np.random.default_rng(seed)

    z = result.standardized_resids
    z_clean = z[np.isfinite(z)]
    if len(z_clean) < 50:
        raise ValueError("Insufficient standardized residuals for FHS simulation")

    sigma2_last = result.conditional_vol[-1] ** 2
    eps_last = z_clean[-1] * result.conditional_vol[-1]

    paths = np.zeros((n_simulations, horizon))

    for sim in range(n_simulations):
        sigma2_t = sigma2_last
        eps_t = eps_last

        for h in range(horizon):
            z_star = z_clean[rng.integers(len(z_clean))]

            # GJR variance update
            leverage = result.gamma if eps_t < 0 else 0.0
            sigma2_t = (
                result.omega
                + (result.alpha + leverage) * eps_t ** 2
                + result.beta * sigma2_t
            )
            sigma_t = np.sqrt(max(sigma2_t, 1e-10))

            r_star = result.mu + sigma_t * z_star
            paths[sim, h] = r_star
            eps_t = sigma_t * z_star

    if horizon == 1:
        sim_returns = paths[:, 0]
    else:
        sim_returns = paths.sum(axis=1)

    var_95 = float(np.percentile(sim_returns, 5))
    var_99 = float(np.percentile(sim_returns, 1))
    es_95 = float(sim_returns[sim_returns <= var_95].mean())
    es_99 = float(sim_returns[sim_returns <= var_99].mean())

    return FHSResult(
        simulated_returns=sim_returns,
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99,
        simulated_paths=paths if horizon > 1 else None,
    )


def fhs_convergence(
    result: GARCHResult,
    max_sims: int = 50000,
    confidence: float = 0.99,
    n_points: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """VaR as f(N_simulations) — checks MC has converged."""
    rng = np.random.default_rng(seed)

    z = result.standardized_resids
    z_clean = z[np.isfinite(z)]

    sigma2_last = result.conditional_vol[-1] ** 2
    eps_last = z_clean[-1] * result.conditional_vol[-1]

    all_returns = np.zeros(max_sims)
    for sim in range(max_sims):
        z_star = z_clean[rng.integers(len(z_clean))]
        leverage = result.gamma if eps_last < 0 else 0.0
        sigma2_next = (
            result.omega
            + (result.alpha + leverage) * eps_last ** 2
            + result.beta * sigma2_last
        )
        sigma_next = np.sqrt(max(sigma2_next, 1e-10))
        all_returns[sim] = result.mu + sigma_next * z_star

    quantile = (1.0 - confidence) * 100
    n_array = np.linspace(500, max_sims, n_points, dtype=int)
    var_array = np.array([np.percentile(all_returns[:n], quantile) for n in n_array])

    return n_array, var_array
