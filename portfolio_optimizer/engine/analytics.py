"""Portfolio analytics: metrics, backtesting, and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .data import MarketData, estimate_expected_returns, estimate_factor_covariance


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Compute the drawdown series from an equity curve."""
    running_max = np.maximum.accumulate(equity)
    return (equity - running_max) / running_max


def rolling_sharpe(
    returns: np.ndarray, window: int = 63, rf_daily: float = 0.0
) -> np.ndarray:
    """Compute rolling annualized Sharpe ratio.

    Parameters
    ----------
    returns : np.ndarray
        Daily returns.
    window : int
        Rolling window size in days.
    rf_daily : float
        Daily risk-free rate.

    Returns
    -------
    np.ndarray
        Rolling Sharpe values (NaN for initial window).
    """
    result = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        r = returns[i - window : i] - rf_daily
        mu = np.mean(r) * 252
        sigma = np.std(r, ddof=1) * np.sqrt(252)
        result[i] = mu / sigma if sigma > 1e-10 else 0.0
    return result


@dataclass
class PortfolioMetrics:
    """Compute standard portfolio performance metrics from daily returns.

    Parameters
    ----------
    returns : np.ndarray
        Daily portfolio returns.
    risk_free_rate : float
        Annualized risk-free rate.
    """

    returns: np.ndarray
    risk_free_rate: float = 0.0

    @property
    def annualized_return(self) -> float:
        cumulative = np.prod(1 + self.returns) ** (252 / len(self.returns)) - 1
        return float(cumulative)

    @property
    def annualized_volatility(self) -> float:
        return float(np.std(self.returns, ddof=1) * np.sqrt(252))

    @property
    def sharpe_ratio(self) -> float:
        vol = self.annualized_volatility
        if vol < 1e-10:
            return 0.0
        return (self.annualized_return - self.risk_free_rate) / vol

    @property
    def sortino_ratio(self) -> float:
        downside = self.returns[self.returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_vol = np.std(downside, ddof=1) * np.sqrt(252)
        if downside_vol < 1e-10:
            return 0.0
        return (self.annualized_return - self.risk_free_rate) / downside_vol

    @property
    def max_drawdown(self) -> float:
        equity = np.cumprod(1 + self.returns)
        dd = drawdown_series(equity)
        return float(np.min(dd))

    @property
    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown)
        if mdd < 1e-10:
            return 0.0
        return self.annualized_return / mdd

    def var_historical(self, confidence: float = 0.95) -> float:
        """Historical Value at Risk (daily)."""
        return float(np.percentile(self.returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall) at given confidence."""
        var = self.var_historical(confidence)
        tail = self.returns[self.returns <= var]
        if len(tail) == 0:
            return var
        return float(np.mean(tail))

    def full_report(self) -> dict:
        """Return a dictionary of all metrics."""
        return {
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "var_95": self.var_historical(0.95),
            "cvar_95": self.cvar(0.95),
        }


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: np.ndarray
    benchmark_curve: np.ndarray
    asset_curves: np.ndarray
    portfolio_returns: np.ndarray
    rebalance_dates: list[int]
    weights_history: list[np.ndarray]
    turnover: list[float]
    metrics: PortfolioMetrics
    benchmark_metrics: PortfolioMetrics


class Backtester:
    """Walk-forward backtester with periodic rebalancing.

    Parameters
    ----------
    optimizer_factory : Callable[[MarketData], dict]
        Function that takes MarketData and returns a dict with at least "weights" key.
    rebalance_freq : int
        Rebalance every N trading days.
    lookback_window : int
        Number of trading days for estimation window.
    risk_free_rate : float
        Annualized risk-free rate.
    """

    def __init__(
        self,
        optimizer_factory: Callable[[MarketData], dict],
        rebalance_freq: int = 21,
        lookback_window: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.optimizer_factory = optimizer_factory
        self.rebalance_freq = rebalance_freq
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate

    def run(self, data: MarketData) -> BacktestResult:
        """Execute the walk-forward backtest.

        Parameters
        ----------
        data : MarketData
            Full-period market data.

        Returns
        -------
        BacktestResult
        """
        returns = data.returns.values  # (T, n)
        T, n = returns.shape
        tickers = data.tickers

        start = self.lookback_window
        if start >= T:
            raise ValueError(
                f"lookback_window ({self.lookback_window}) >= total days ({T})"
            )

        # Initialize
        equity = [1.0]
        benchmark = [1.0]
        asset_equities = [np.ones(n)]
        port_returns = []
        rebalance_dates = []
        weights_history = []
        turnover = []

        # Equal-weight benchmark
        bm_weights = np.ones(n) / n

        # Current weights (will be set on first rebalance)
        current_weights = np.ones(n) / n
        days_since_rebalance = self.rebalance_freq  # Force rebalance on first day

        for t in range(start, T):
            # Rebalance?
            if days_since_rebalance >= self.rebalance_freq:
                # Build estimation window
                window_returns = returns[t - self.lookback_window : t]
                window_prices = data.prices.iloc[t - self.lookback_window : t + 1]

                window_ret_df = pd.DataFrame(
                    window_returns,
                    columns=tickers,
                    index=data.returns.index[t - self.lookback_window : t],
                )

                # Slice FF factors to the same window (if available)
                window_ff = None
                if data.ff_factors is not None:
                    window_ff = data.ff_factors.loc[
                        data.ff_factors.index.intersection(window_ret_df.index)
                    ]

                # Covariance matrix
                if data.ff_use_factor_cov and window_ff is not None:
                    cov_annual = estimate_factor_covariance(window_ret_df, window_ff)
                else:
                    lw = LedoitWolf().fit(window_returns)
                    cov_annual = lw.covariance_ * 252

                mu_annual = estimate_expected_returns(
                    window_ret_df, data.returns_model, window_ff,
                    include_alpha=data.ff_include_alpha,
                )

                window_data = MarketData(
                    tickers=tickers,
                    prices=window_prices,
                    returns=window_ret_df,
                    cov_matrix=cov_annual,
                    expected_returns=mu_annual,
                    risk_free_rate=self.risk_free_rate,
                    market_caps=data.market_caps,
                )

                try:
                    result = self.optimizer_factory(window_data)
                    new_weights = result["weights"]
                except Exception:
                    new_weights = current_weights

                turn = float(np.sum(np.abs(new_weights - current_weights)))
                turnover.append(turn)
                current_weights = new_weights.copy()
                rebalance_dates.append(t)
                weights_history.append(new_weights.copy())
                days_since_rebalance = 0

            # Day's returns
            day_returns = returns[t]

            # Portfolio return (weighted)
            port_ret = float(current_weights @ day_returns)
            port_returns.append(port_ret)

            # Update equity
            equity.append(equity[-1] * (1 + port_ret))

            # Benchmark
            bm_ret = float(bm_weights @ day_returns)
            benchmark.append(benchmark[-1] * (1 + bm_ret))

            # Asset equities
            asset_equities.append(asset_equities[-1] * (1 + day_returns))

            # Drift weights (mark-to-market)
            drifted = current_weights * (1 + day_returns)
            current_weights = drifted / drifted.sum()

            days_since_rebalance += 1

        port_returns_arr = np.array(port_returns)
        bm_returns = np.diff(benchmark) / np.array(benchmark[:-1])

        return BacktestResult(
            equity_curve=np.array(equity),
            benchmark_curve=np.array(benchmark),
            asset_curves=np.array(asset_equities),
            portfolio_returns=port_returns_arr,
            rebalance_dates=rebalance_dates,
            weights_history=weights_history,
            turnover=turnover,
            metrics=PortfolioMetrics(port_returns_arr, self.risk_free_rate),
            benchmark_metrics=PortfolioMetrics(bm_returns, self.risk_free_rate),
        )
