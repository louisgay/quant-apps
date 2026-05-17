"""Rolling VaR: 6 methods from empirical quantile to MC-FHS.

Convention: VaR < 0 (loss threshold). Methods 4-6 use expanding window + step.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from engine.garch import fit_gjr_garch, simulate_fhs

logger = logging.getLogger(__name__)


def historical_simulation_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.99,
) -> pd.Series:
    """Rolling empirical quantile. VaR_t = Q_{1-alpha}(r_{t-w:t-1})."""
    quantile = 1.0 - confidence
    # shift(1): VaR at index i uses data through i-1 (no lookahead)
    var_series = returns.rolling(window=window).quantile(quantile).shift(1)
    var_series.name = "historical_simulation"
    return var_series


def parametric_normal_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.99,
) -> pd.Series:
    """mu + z_alpha * sigma, rolling window. Quick but ignores fat tails."""
    z_alpha = norm.ppf(1.0 - confidence)
    # shift(1): forecast for day i uses data through i-1 (no lookahead)
    rolling_mean = returns.rolling(window=window).mean().shift(1)
    rolling_std = returns.rolling(window=window).std().shift(1)
    var_series = rolling_mean + z_alpha * rolling_std
    var_series.name = "parametric_normal"
    return var_series


def parametric_student_t_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.99,
    step: int = 5,
) -> pd.Series:
    """MLE-fit nu per window, then use t-quantile. Fatter tails → more conservative.

    Refits every `step` days and reuses last parameters between fits.
    """
    quantile = 1.0 - confidence
    n = len(returns)
    var_values = pd.Series(np.nan, index=returns.index, name="parametric_student_t")

    last_nu = last_loc = last_scale = None

    for i in range(window, n):
        if (i - window) % step == 0 or last_nu is None:
            window_data = returns.iloc[i - window:i].values
            try:
                nu, loc, scale = student_t.fit(window_data)
                last_nu = max(nu, 2.01)  # ensure finite variance
                last_loc = loc
                last_scale = scale
            except Exception:
                # Fallback to normal approximation
                mu = window_data.mean()
                sigma = window_data.std()
                last_nu = 30.0  # approximate normal
                last_loc = mu
                last_scale = sigma

        if last_nu is not None:
            var_values.iloc[i] = student_t.ppf(quantile, df=last_nu, loc=last_loc, scale=last_scale)

    return var_values


def garch_normal_var(
    returns: pd.Series,
    confidence: float = 0.99,
    min_window: int = 504,
    step: int = 5,
) -> pd.Series:
    """GARCH(1,1) + Phi^{-1}. Refit every `step` days, propagate between."""
    from arch import arch_model

    z_alpha = norm.ppf(1.0 - confidence)
    n = len(returns)
    r_pct = returns.values * 100.0
    var_values = pd.Series(np.nan, index=returns.index, name="garch_normal")

    last_omega = last_alpha = last_beta = last_mu = None
    last_sigma2 = None

    for i in range(min_window, n):
        if (i - min_window) % step == 0 or last_omega is None:
            try:
                model = arch_model(r_pct[:i], mean="Constant", vol="GARCH",
                                   p=1, q=1, dist="normal")
                res = model.fit(disp="off", show_warning=False)
                last_omega = res.params.get("omega", 0)
                last_alpha = res.params.get("alpha[1]", 0)
                last_beta = res.params.get("beta[1]", 0)
                last_mu = res.params.get("mu", 0)
                last_sigma2 = np.asarray(res.conditional_volatility)[-1] ** 2
            except Exception:
                continue

        if last_omega is None:
            continue

        # Forecast sigma^2_{t+1}
        eps = r_pct[i - 1] - last_mu
        sigma2_next = last_omega + last_alpha * eps ** 2 + last_beta * last_sigma2
        sigma_next = np.sqrt(sigma2_next) / 100.0  # back to decimal
        mu_dec = last_mu / 100.0

        var_values.iloc[i] = mu_dec + z_alpha * sigma_next
        last_sigma2 = sigma2_next

    return var_values


def garch_student_t_var(
    returns: pd.Series,
    confidence: float = 0.99,
    min_window: int = 504,
    step: int = 5,
) -> pd.Series:
    """GARCH(1,1) + t-quantile. Captures vol clustering + fat tails, no leverage."""
    from arch import arch_model

    n = len(returns)
    r_pct = returns.values * 100.0
    var_values = pd.Series(np.nan, index=returns.index, name="garch_student_t")

    last_omega = last_alpha = last_beta = last_mu = last_nu = None
    last_sigma2 = None

    for i in range(min_window, n):
        if (i - min_window) % step == 0 or last_omega is None:
            try:
                model = arch_model(r_pct[:i], mean="Constant", vol="GARCH",
                                   p=1, q=1, dist="t")
                res = model.fit(disp="off", show_warning=False)
                last_omega = res.params.get("omega", 0)
                last_alpha = res.params.get("alpha[1]", 0)
                last_beta = res.params.get("beta[1]", 0)
                last_mu = res.params.get("mu", 0)
                last_nu = res.params.get("nu", 5.0)
                last_sigma2 = np.asarray(res.conditional_volatility)[-1] ** 2
            except Exception:
                continue

        if last_omega is None:
            continue

        # Forecast sigma^2_{t+1} (percent^2 space)
        eps = r_pct[i - 1] - last_mu
        sigma2_next = last_omega + last_alpha * eps ** 2 + last_beta * last_sigma2
        sigma_next = np.sqrt(sigma2_next) / 100.0  # back to decimal
        mu_dec = last_mu / 100.0

        nu = max(last_nu, 2.01)
        t_quantile = student_t.ppf(1.0 - confidence, df=nu)
        var_values.iloc[i] = mu_dec + t_quantile * sigma_next

        last_sigma2 = sigma2_next

    return var_values


def gjr_garch_student_t_var(
    returns: pd.Series,
    confidence: float = 0.99,
    min_window: int = 504,
    step: int = 5,
) -> pd.Series:
    """GJR-GARCH + t-quantile. Handles both leverage and fat tails."""
    n = len(returns)
    var_values = pd.Series(np.nan, index=returns.index, name="gjr_garch_student_t")

    last_result = None
    last_sigma2 = None
    last_eps = None

    for i in range(min_window, n):
        if (i - min_window) % step == 0 or last_result is None:
            try:
                result = fit_gjr_garch(returns.iloc[:i], distribution="studentt")
                last_result = result
                last_sigma2 = result.conditional_vol[-1] ** 2
                last_eps = returns.iloc[i - 1] - result.mu
            except Exception:
                continue

        if last_result is None:
            continue

        # GJR variance forecast (all in decimal^2 space)
        eps = returns.iloc[i - 1] - last_result.mu
        leverage = last_result.gamma if eps < 0 else 0.0
        sigma2_next = (
            last_result.omega
            + (last_result.alpha + leverage) * eps ** 2
            + last_result.beta * last_sigma2
        )
        sigma_next = np.sqrt(max(sigma2_next, 1e-10))

        # Student-t quantile
        nu = last_result.nu if last_result.nu and last_result.nu > 2 else 5.0
        t_quantile = student_t.ppf(1.0 - confidence, df=nu)
        var_values.iloc[i] = last_result.mu + t_quantile * sigma_next

        last_sigma2 = sigma2_next

    return var_values


def monte_carlo_fhs_var(
    returns: pd.Series,
    confidence: float = 0.99,
    min_window: int = 504,
    n_sims: int = 2000,
    step: int = 5,
) -> pd.Series:
    """FHS per Barone-Adesi (1999). Slow but non-parametric tails + vol dynamics."""
    n = len(returns)
    var_values = pd.Series(np.nan, index=returns.index, name="monte_carlo_fhs")

    last_result = None

    for i in range(min_window, n):
        if (i - min_window) % step == 0 or last_result is None:
            try:
                result = fit_gjr_garch(returns.iloc[:i], distribution="studentt")
                last_result = result
            except Exception:
                continue

        if last_result is None:
            continue

        try:
            fhs = simulate_fhs(last_result, n_simulations=n_sims, horizon=1,
                               seed=42 + i)
            quantile = (1.0 - confidence) * 100
            var_values.iloc[i] = float(np.percentile(fhs.simulated_returns, quantile))
        except Exception:
            continue

    return var_values
