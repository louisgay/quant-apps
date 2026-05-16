"""Alpaca + yfinance price fetcher with auto-fallback."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

## Sidebar dropdown tickers

TICKER_UNIVERSE: dict[str, str] = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IWM": "Russell 2000 ETF",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet",
    "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs",
    "GLD": "Gold ETF",
    "TLT": "20+ Year Treasury ETF",
    "XLE": "Energy Sector ETF",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}


def _get_secret(key: str) -> Optional[str]:
    """Try st.secrets first, then env vars."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key)


def _is_real_key(value: Optional[str]) -> bool:
    # Filter out placeholder strings like "YOUR_API_KEY"
    return bool(value) and not value.startswith("YOUR_")


def _fetch_alpaca(ticker: str, start: str, end: str) -> pd.DataFrame:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = _get_secret("ALPACA_API_KEY")
    secret_key = _get_secret("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    df = df[["timestamp", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.set_index("timestamp").sort_index()
    df.index.name = None
    return df


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        raise ValueError(f"No data returned by yfinance for {ticker}")
    df = data[["Close"]].copy()
    df.columns = ["close"]
    return df


def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    source: str = "auto",
) -> pd.DataFrame:
    """Daily close prices → DataFrame with DatetimeIndex + 'close' column.

    Crypto tickers (containing "-") bypass Alpaca and go straight to yfinance.
    """
    # Crypto tickers bypass Alpaca
    if "-" in ticker:
        logger.info("Crypto ticker detected — using yfinance for %s", ticker)
        return _fetch_yfinance(ticker, start, end)

    if source == "auto":
        api_key = _get_secret("ALPACA_API_KEY")
        secret_key = _get_secret("ALPACA_SECRET_KEY")
        if _is_real_key(api_key) and _is_real_key(secret_key):
            source = "alpaca"
        else:
            source = "yfinance"

    if source == "alpaca":
        try:
            df = _fetch_alpaca(ticker, start, end)
            logger.info("Fetched %d bars from Alpaca for %s", len(df), ticker)
            return df
        except Exception as exc:
            logger.warning("Alpaca failed for %s: %s — falling back to yfinance", ticker, exc)
            return _fetch_yfinance(ticker, start, end)

    return _fetch_yfinance(ticker, start, end)


def compute_log_returns(prices: pd.DataFrame) -> pd.Series:
    """ln(P_t / P_{t-1}), drops NaN."""
    close = prices["close"].squeeze()
    log_ret = np.log(close / close.shift(1)).dropna()
    log_ret.name = "log_return"
    return log_ret
