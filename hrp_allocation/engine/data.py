"""Market data (Alpaca primary, yfinance fallback) and covariance estimation."""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_market_data(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    source: str = "auto",
) -> pd.DataFrame:
    """Fetch adjusted prices → log returns (T x N DataFrame)."""
    if source == "auto":
        if _alpaca_credentials_available():
            try:
                return _fetch_alpaca(tickers, start, end)
            except Exception as e:
                logger.warning(f"Alpaca failed ({e}), falling back to yfinance")
                return _fetch_yfinance(tickers, start, end)
        else:
            logger.info("No Alpaca credentials, using yfinance")
            return _fetch_yfinance(tickers, start, end)
    elif source == "alpaca":
        return _fetch_alpaca(tickers, start, end)
    elif source == "yfinance":
        return _fetch_yfinance(tickers, start, end)
    else:
        raise ValueError(f"Unknown source: {source}")


def _alpaca_credentials_available() -> bool:
    return bool(
        os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")
    )


def _fetch_alpaca(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Alpaca Markets (free IEX feed, split+dividend adjusted)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    logger.info(f"Fetching {len(tickers)} tickers from Alpaca [{start} → {end or 'now'}]")

    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )

    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d") if end else None,
        adjustment="all",
    )

    bars = client.get_stock_bars(request_params)
    df = bars.df

    prices = df["close"].unstack(level=0)
    prices.index = prices.index.date
    prices.index = pd.DatetimeIndex(prices.index)
    prices = prices[tickers]
    prices = prices.dropna(how="all").ffill()

    if prices.isnull().any().any():
        missing = prices.isnull().sum()
        missing = missing[missing > 0]
        logger.warning(f"Dropping columns with missing data: {missing.to_dict()}")
        prices = prices.dropna(axis=1)

    returns = np.log(prices / prices.shift(1)).dropna()
    logger.info(f"Alpaca: {len(returns)} days × {len(returns.columns)} assets")
    return returns


def _fetch_yfinance(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """yfinance fallback (no API key needed)."""
    import yfinance as yf

    logger.info(f"Fetching {len(tickers)} tickers from yfinance [{start} → {end or 'now'}]")

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.dropna(how="all").ffill()

    if prices.isnull().any().any():
        missing = prices.isnull().sum()
        missing = missing[missing > 0]
        logger.warning(f"Dropping columns with missing data: {missing.to_dict()}")
        prices = prices.dropna(axis=1)

    returns = np.log(prices / prices.shift(1)).dropna()
    logger.info(f"yfinance: {len(returns)} days × {len(returns.columns)} assets")
    return returns


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "sample",
    shrinkage_target: str = "identity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (cov, corr) DataFrames."""
    if method == "ledoit_wolf":
        cov = ledoit_wolf_shrinkage(returns, target=shrinkage_target)
    else:
        cov = returns.cov()

    std = np.sqrt(np.diag(cov.values))
    corr_values = cov.values / np.outer(std, std)
    np.fill_diagonal(corr_values, 1.0)
    corr = pd.DataFrame(corr_values, index=cov.index, columns=cov.columns)

    return cov, corr


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    target: str = "identity",
) -> pd.DataFrame:
    """Ledoit-Wolf (2004) shrinkage: Σ_shrunk = δ·F + (1-δ)·S.

    Optimal δ minimizes E[||Σ_shrunk - Σ_true||²_F].
    """
    X = returns.values
    T, N = X.shape
    X = X - X.mean(axis=0)

    S = (X.T @ X) / T

    if target == "identity":
        mu = np.trace(S) / N
        F = mu * np.eye(N)
    elif target == "constant_correlation":
        var = np.diag(S)
        std = np.sqrt(var)
        corr_sample = S / np.outer(std, std)
        np.fill_diagonal(corr_sample, 1.0)
        rho_bar = (corr_sample.sum() - N) / (N * (N - 1))
        F = rho_bar * np.outer(std, std)
        np.fill_diagonal(F, var)
    else:
        raise ValueError(f"Unknown target: {target}")

    delta = _optimal_shrinkage_intensity(X, S, F, T, N)
    delta = np.clip(delta, 0.0, 1.0)

    logger.info(f"Ledoit-Wolf shrinkage δ = {delta:.4f}")

    cov_shrunk = delta * F + (1 - delta) * S
    return pd.DataFrame(cov_shrunk, index=returns.columns, columns=returns.columns)


def _optimal_shrinkage_intensity(
    X: np.ndarray, S: np.ndarray, F: np.ndarray, T: int, N: int
) -> float:
    """Analytical oracle shrinkage (Ledoit-Wolf 2004, Lemma 3.1)."""
    X2 = X**2
    sample = (X2.T @ X2) / T - S**2
    pi_hat = sample.sum()

    delta_mat = S - F
    gamma_hat = np.sum(delta_mat**2)

    if gamma_hat == 0:
        return 1.0

    return (pi_hat / T) / gamma_hat
