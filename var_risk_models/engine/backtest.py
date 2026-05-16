"""Kupiec (1995), Christoffersen (1998), combined conditional coverage tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

logger = logging.getLogger(__name__)


@dataclass
class CoverageTestResult:
    """LR_cc = LR_uc + LR_ind ~ chi2(2)."""
    lr_uc: float
    lr_ind: float
    lr_cc: float
    p_uc: float
    p_ind: float
    p_cc: float


def var_exceedance_test(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.99,
) -> Dict:
    """Count days where realized return < VaR (i.e. VaR breach)."""
    aligned = pd.concat([returns, var_series], axis=1, join="inner").dropna()
    aligned.columns = ["return", "var"]

    exceedances = aligned["return"] < aligned["var"]
    n_obs = len(aligned)
    n_exc = int(exceedances.sum())
    expected_rate = 1.0 - confidence
    actual_rate = n_exc / n_obs if n_obs > 0 else 0.0

    ratio = actual_rate / expected_rate if expected_rate > 0 else np.nan

    return {
        "n_obs": n_obs,
        "n_exceedances": n_exc,
        "exceedance_rate": actual_rate,
        "expected_rate": expected_rate,
        "exceedance_ratio": ratio,
        "exceedances": exceedances,
    }


def kupiec_test(
    n_total: int,
    n_exceedances: int,
    confidence: float = 0.99,
) -> Tuple[float, float]:
    """LR_uc ~ chi2(1). H0: exceedance rate = (1 - confidence)."""
    p0 = 1.0 - confidence
    n = n_total
    x = n_exceedances

    if n <= 0:
        return 0.0, 1.0
    if x <= 0:
        lr = -2.0 * n * np.log(1.0 - p0)
        return float(lr), float(1.0 - chi2.cdf(lr, df=1))
    if x >= n:
        lr = -2.0 * n * np.log(p0)
        return float(lr), float(1.0 - chi2.cdf(lr, df=1))

    p_hat = x / n
    log_l0 = x * np.log(p0) + (n - x) * np.log(1.0 - p0)
    log_l1 = x * np.log(p_hat) + (n - x) * np.log(1.0 - p_hat)

    lr = -2.0 * (log_l0 - log_l1)
    p_value = 1.0 - chi2.cdf(lr, df=1)

    return float(lr), float(p_value)


def christoffersen_test(
    exceedances: np.ndarray | pd.Series,
) -> Tuple[float, float]:
    """LR_ind ~ chi2(1). H0: exceedances are i.i.d. (no clustering)."""
    exc = np.asarray(exceedances, dtype=int)
    T = len(exc)

    if T < 2:
        return 0.0, 1.0

    # Build transition counts
    n_00 = n_01 = n_10 = n_11 = 0
    for t in range(1, T):
        prev, curr = exc[t - 1], exc[t]
        if prev == 0 and curr == 0:
            n_00 += 1
        elif prev == 0 and curr == 1:
            n_01 += 1
        elif prev == 1 and curr == 0:
            n_10 += 1
        else:
            n_11 += 1

    n_0 = n_00 + n_01
    n_1 = n_10 + n_11

    if n_0 == 0 or n_1 == 0:
        return 0.0, 1.0

    pi_01 = n_01 / n_0
    pi_11 = n_11 / n_1
    pi_hat = (n_01 + n_11) / (T - 1)

    if pi_hat <= 0 or pi_hat >= 1:
        return 0.0, 1.0

    # Log-likelihood under H0 (independence)
    log_l0 = 0.0
    if n_00 + n_10 > 0:
        log_l0 += (n_00 + n_10) * np.log(1.0 - pi_hat)
    if n_01 + n_11 > 0:
        log_l0 += (n_01 + n_11) * np.log(pi_hat)

    # Log-likelihood under H1 (Markov)
    log_l1 = 0.0
    if n_00 > 0:
        log_l1 += n_00 * np.log(1.0 - pi_01)
    if n_01 > 0:
        log_l1 += n_01 * np.log(pi_01)
    if n_10 > 0:
        log_l1 += n_10 * np.log(1.0 - pi_11)
    if n_11 > 0:
        log_l1 += n_11 * np.log(pi_11)

    lr = max(-2.0 * (log_l0 - log_l1), 0.0)
    p_value = 1.0 - chi2.cdf(lr, df=1)

    return float(lr), float(p_value)


def combined_coverage_test(
    n_total: int,
    n_exceedances: int,
    exceedances: np.ndarray | pd.Series,
    confidence: float = 0.99,
) -> CoverageTestResult:
    """LR_cc = LR_uc + LR_ind ~ chi2(2). Rejects on miscalibration OR clustering."""
    lr_uc, p_uc = kupiec_test(n_total, n_exceedances, confidence)
    lr_ind, p_ind = christoffersen_test(exceedances)

    lr_cc = lr_uc + lr_ind
    p_cc = float(1.0 - chi2.cdf(lr_cc, df=2))

    return CoverageTestResult(
        lr_uc=lr_uc,
        lr_ind=lr_ind,
        lr_cc=lr_cc,
        p_uc=p_uc,
        p_ind=p_ind,
        p_cc=p_cc,
    )


def run_full_backtest(
    returns: pd.Series,
    var_dict: dict[str, pd.Series],
    confidence: float = 0.99,
) -> pd.DataFrame:
    """Exceedance + Kupiec + Christoffersen + combined for all methods at once."""
    rows = []
    for method_name, var_series in var_dict.items():
        exc_result = var_exceedance_test(returns, var_series, confidence)
        kup_stat, kup_p = kupiec_test(
            exc_result["n_obs"], exc_result["n_exceedances"], confidence
        )
        chr_stat, chr_p = christoffersen_test(exc_result["exceedances"])
        cc = combined_coverage_test(
            exc_result["n_obs"], exc_result["n_exceedances"],
            exc_result["exceedances"], confidence,
        )

        rows.append({
            "Method": method_name,
            "n_obs": exc_result["n_obs"],
            "n_exceedances": exc_result["n_exceedances"],
            "exceedance_rate": exc_result["exceedance_rate"],
            "expected_rate": exc_result["expected_rate"],
            "ratio": exc_result["exceedance_ratio"],
            "kupiec_stat": kup_stat,
            "kupiec_p": kup_p,
            "christoffersen_stat": chr_stat,
            "christoffersen_p": chr_p,
            "combined_stat": cc.lr_cc,
            "combined_p": cc.p_cc,
        })

    return pd.DataFrame(rows)
