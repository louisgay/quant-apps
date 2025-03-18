"""Fetch and clean option chain data from Yahoo Finance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class OptionChainData:
    """Clean option chain ready for IV computation."""

    ticker: str
    spot: float
    risk_free_rate: float
    fetched_at: datetime
    chains: pd.DataFrame
    # chains columns: strike, expiry, T, mid_price, option_type, bid, ask,
    #                 open_interest, volume, forward

    @property
    def expiries(self) -> List[str]:
        return sorted(self.chains["expiry"].unique().tolist())

    @property
    def n_options(self) -> int:
        return len(self.chains)

    def slice(self, expiry: str) -> pd.DataFrame:
        return self.chains[self.chains["expiry"] == expiry].copy()


class DataFetcher:
    """Retrieves and sanitises equity option chains via yfinance."""

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self.risk_free_rate = risk_free_rate

    def fetch(
        self,
        ticker: str,
        min_oi: int = 10,
        min_volume: int = 1,
        max_expiries: Optional[int] = 8,
        moneyness_range: tuple[float, float] = (0.70, 1.30),
    ) -> OptionChainData:
        """
        Fetch option chains for *ticker*, clean and return structured data.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker (e.g. "SPY", "TSLA", "^SMI").
        min_oi : int
            Minimum open interest to keep an option.
        min_volume : int
            Minimum traded volume.
        max_expiries : int | None
            Cap on the number of expiration dates (nearest first).
        moneyness_range : tuple
            Keep strikes where K/S is within this range.
        """
        logger.info("Fetching option chain for %s ...", ticker)
        asset = yf.Ticker(ticker)

        # Current spot
        hist = asset.history(period="5d")
        if hist.empty:
            raise ValueError(f"No price data for {ticker}")
        spot = float(hist["Close"].iloc[-1])
        logger.info("Spot price for %s: %.2f", ticker, spot)

        expiries = list(asset.options)
        if not expiries:
            raise ValueError(f"No options listed for {ticker}")
        if max_expiries is not None:
            expiries = expiries[:max_expiries]
        logger.info("Expiries selected (%d): %s", len(expiries), expiries)

        now = datetime.now(tz=timezone.utc)
        frames: list[pd.DataFrame] = []

        for exp_str in expiries:
            try:
                chain = asset.option_chain(exp_str)
            except Exception as exc:
                logger.warning("Skipping expiry %s: %s", exp_str, exc)
                continue

            exp_date = pd.Timestamp(exp_str, tz=timezone.utc)
            T = max((exp_date - pd.Timestamp(now)).total_seconds() / (365.25 * 86400), 1 / 365)
            forward = spot * np.exp(self.risk_free_rate * T)

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                sub = df[["strike", "bid", "ask", "openInterest", "volume"]].copy()
                sub["expiry"] = exp_str
                sub["T"] = T
                sub["option_type"] = opt_type
                sub["forward"] = forward
                sub["openInterest"] = sub["openInterest"].fillna(0).astype(int)
                sub["volume"] = sub["volume"].fillna(0).astype(int)
                frames.append(sub)

        if not frames:
            raise ValueError(f"No valid option chains retrieved for {ticker}")

        raw = pd.concat(frames, ignore_index=True)
        logger.info("Raw options fetched: %d rows", len(raw))

        # --- Cleaning ---
        df = raw.copy()
        df["mid_price"] = (df["bid"] + df["ask"]) / 2

        # Remove zero/negative bids or mid
        df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["mid_price"] > 0)]

        # Open interest & volume filters
        df = df[(df["openInterest"] >= min_oi) & (df["volume"] >= min_volume)]

        # Moneyness filter
        df["moneyness"] = df["strike"] / spot
        lo, hi = moneyness_range
        df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]

        # Remove absurd bid/ask spreads (> 80% of mid)
        df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid_price"]
        df = df[df["spread_pct"] < 0.80]

        df = df.drop(columns=["moneyness", "spread_pct"]).reset_index(drop=True)

        logger.info("Clean options: %d rows across %d expiries",
                     len(df), df["expiry"].nunique())

        return OptionChainData(
            ticker=ticker,
            spot=spot,
            risk_free_rate=self.risk_free_rate,
            fetched_at=now,
            chains=df,
        )
