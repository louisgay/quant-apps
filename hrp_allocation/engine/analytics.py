"""Backtesting and portfolio metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioMetrics:
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    avg_turnover: float
    effective_bets: float
    avg_weight_change: float

    def to_dict(self) -> dict:
        return {
            "Annual Return": f"{self.annualized_return:.2%}",
            "Annual Volatility": f"{self.annualized_volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Avg Turnover": f"{self.avg_turnover:.4f}",
            "Effective Bets": f"{self.effective_bets:.1f}",
            "Avg |Δw|": f"{self.avg_weight_change:.4f}",
        }


def compute_metrics(
    returns: pd.Series,
    weights_history: pd.DataFrame | None = None,
    risk_free: float = 0.0,
) -> PortfolioMetrics:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0.0

    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    max_dd = (cum_returns / rolling_max - 1).min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    if weights_history is not None and len(weights_history) > 1:
        weight_changes = weights_history.diff().iloc[1:]  # drop first NaN row
        avg_turnover = weight_changes.abs().sum(axis=1).mean() / 2
        avg_weight_change = weight_changes.abs().mean().mean()
        last_weights = weights_history.iloc[-1]
        eff_bets = effective_number_of_bets(last_weights.values)
    else:
        avg_turnover = 0.0
        avg_weight_change = 0.0
        eff_bets = 0.0

    return PortfolioMetrics(
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        avg_turnover=avg_turnover,
        effective_bets=eff_bets,
        avg_weight_change=avg_weight_change,
    )


def effective_number_of_bets(weights: np.ndarray) -> float:
    """exp(-Σ w_i ln w_i) — ranges from 1 (concentrated) to N (equal weight)."""
    w = weights[weights > 1e-10]
    if len(w) == 0:
        return 0.0
    return np.exp(-np.sum(w * np.log(w)))


class Backtester:
    """Rolling-window backtest with strict no-lookahead.

    Uses returns[t-window:t] to estimate Σ, computes weights, holds until next rebalance.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        allocation_fn: callable,
        window: int = 252,
        rebalance_freq: int = 21,
        cov_method: str = "sample",
    ):
        self.returns = returns
        self.allocation_fn = allocation_fn
        self.window = window
        self.rebalance_freq = rebalance_freq
        self.cov_method = cov_method

    def run(self) -> tuple[pd.Series, pd.DataFrame]:
        T, N = self.returns.shape
        assets = self.returns.columns

        portfolio_returns = []
        weights_history = []
        rebalance_dates = []

        current_weights = np.ones(N) / N

        for t in range(self.window, T):
            date = self.returns.index[t]

            periods_since_start = t - self.window
            if periods_since_start % self.rebalance_freq == 0:
                window_returns = self.returns.iloc[t - self.window : t]

                if self.cov_method == "ledoit_wolf":
                    from .data import ledoit_wolf_shrinkage
                    cov = ledoit_wolf_shrinkage(window_returns)
                else:
                    cov = window_returns.cov()

                new_weights = self.allocation_fn(cov)
                current_weights = new_weights.values

                weights_history.append(current_weights.copy())
                rebalance_dates.append(date)

            daily_asset_returns = self.returns.iloc[t].values
            port_return = current_weights @ daily_asset_returns
            portfolio_returns.append(port_return)

            # Drift weights by relative asset performance (buy-and-hold between rebalances)
            drifted = current_weights * (1.0 + daily_asset_returns)
            current_weights = drifted / drifted.sum()

        port_ret_series = pd.Series(
            portfolio_returns,
            index=self.returns.index[self.window:],
            name="portfolio",
        )

        weights_df = pd.DataFrame(
            weights_history, index=rebalance_dates, columns=assets
        )

        return port_ret_series, weights_df
