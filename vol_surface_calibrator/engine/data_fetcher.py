"""Fetch and clean option chain data from Yahoo Finance or Alpaca Markets."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_secret(key: str) -> Optional[str]:
    """Read a secret from Streamlit Cloud secrets or environment variables."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key)


def _parse_occ_symbol(symbol: str) -> Optional[dict]:
    """Parse OCC option symbol into components.

    Format: SYMBOL YYMMDD C/P STRIKE (e.g. SPY250620C00550000)
    Strike is in units of 1/1000 of a dollar.
    """
    m = re.match(
        r"^([A-Z]+)(\d{6})([CP])(\d{8})$",
        symbol.replace(" ", ""),
    )
    if not m:
        return None
    underlying, date_str, cp, strike_raw = m.groups()
    expiry = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
    return {
        "underlying": underlying,
        "expiry": expiry,
        "option_type": "call" if cp == "C" else "put",
        "strike": int(strike_raw) / 1000.0,
    }


@dataclass
class OptionChainData:
    """Clean option chain ready for IV computation."""

    ticker: str
    spot: float
    risk_free_rate: float
    fetched_at: datetime
    chains: pd.DataFrame
    dividend_yield: float = 0.0
    source: str = "yfinance"
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
    """Retrieves and sanitises equity option chains.

    Supports two data sources:
    - "alpaca": Alpaca Markets option chain API (requires API keys)
    - "yfinance": Yahoo Finance (free, no keys needed)
    - "auto" (default): tries Alpaca if keys are set, falls back to yfinance
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        source: str = "auto",
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.source = source

    @staticmethod
    def _is_real_key(value: Optional[str]) -> bool:
        """Return True if value looks like a real API key, not a placeholder."""
        if not value:
            return False
        return not value.startswith("YOUR_")

    def _resolve_source(self) -> str:
        if self.source == "auto":
            api_key = _get_secret("ALPACA_API_KEY")
            secret_key = _get_secret("ALPACA_SECRET_KEY")
            if self._is_real_key(api_key) and self._is_real_key(secret_key):
                return "alpaca"
            return "yfinance"
        return self.source

    def _estimate_implied_forwards(
        self, df: pd.DataFrame, spot: float
    ) -> dict[str, float]:
        """Estimate market-implied forward per expiry via put-call parity.

        For each expiry, find matched call-put pairs at the same strike and
        compute F = K + exp(r*T) * (C_mid - P_mid).  Take the median per
        expiry for robustness.  Falls back to naive S*exp(r*T) when fewer
        than 3 matched pairs are available.
        """
        forwards: dict[str, float] = {}
        for expiry, grp in df.groupby("expiry"):
            T = grp["T"].iloc[0]
            naive_fwd = spot * np.exp(self.risk_free_rate * T)

            calls = grp[grp["option_type"] == "call"].set_index("strike")
            puts = grp[grp["option_type"] == "put"].set_index("strike")
            matched_strikes = calls.index.intersection(puts.index)

            if len(matched_strikes) < 3:
                forwards[expiry] = naive_fwd
                continue

            f_estimates = []
            for K in matched_strikes:
                C_mid = calls.loc[K, "mid_price"]
                P_mid = puts.loc[K, "mid_price"]
                # Handle duplicate strikes (take first if Series)
                if isinstance(C_mid, pd.Series):
                    C_mid = C_mid.iloc[0]
                if isinstance(P_mid, pd.Series):
                    P_mid = P_mid.iloc[0]
                F = K + np.exp(self.risk_free_rate * T) * (C_mid - P_mid)
                f_estimates.append(F)

            median_fwd = float(np.median(f_estimates))
            # Sanity check: forward should be within reasonable range of spot
            if 0.8 * spot < median_fwd < 1.2 * spot:
                forwards[expiry] = median_fwd
            else:
                forwards[expiry] = naive_fwd

        return forwards

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
            Ticker symbol (e.g. "SPY", "TSLA").
        min_oi : int
            Minimum open interest to keep an option.
        min_volume : int
            Minimum traded volume.
        max_expiries : int | None
            Cap on the number of expiration dates (nearest first).
        moneyness_range : tuple
            Keep strikes where K/S is within this range.
        """
        source = self._resolve_source()
        logger.info("Fetching option chain for %s (source=%s) ...", ticker, source)

        if source == "alpaca":
            raw, spot = self._fetch_alpaca(ticker, max_expiries)
        else:
            raw, spot = self._fetch_yfinance(ticker, max_expiries)

        # Cross-check spot vs option prices
        spot = self._validate_spot(raw, spot, ticker)
        raw["forward"] = spot * np.exp(self.risk_free_rate * raw["T"])

        # Shared cleaning
        df = self._clean(raw, spot, min_oi, min_volume, moneyness_range)

        logger.info("Clean options: %d rows across %d expiries",
                     len(df), df["expiry"].nunique())

        # Estimate market-implied forwards from put-call parity
        implied_fwds = self._estimate_implied_forwards(df, spot)
        df["forward"] = df["expiry"].map(implied_fwds)

        # Derive per-expiry dividend yield: q = r - ln(F/S) / T
        q_values = []
        for expiry, fwd in implied_fwds.items():
            T_exp = df.loc[df["expiry"] == expiry, "T"].iloc[0]
            q = self.risk_free_rate - np.log(fwd / spot) / T_exp
            q_values.append(q)
        dividend_yield = float(np.median(q_values)) if q_values else 0.0

        logger.info("Implied dividend yield: %.4f (%.2f%%)", dividend_yield, dividend_yield * 100)

        return OptionChainData(
            ticker=ticker,
            spot=spot,
            risk_free_rate=self.risk_free_rate,
            fetched_at=datetime.now(tz=timezone.utc),
            chains=df,
            dividend_yield=dividend_yield,
            source=source,
        )

    def _fetch_yfinance(
        self,
        ticker: str,
        max_expiries: Optional[int],
    ) -> tuple[pd.DataFrame, float]:
        """Fetch via yfinance."""
        import yfinance as yf

        asset = yf.Ticker(ticker)

        # Current spot
        hist = asset.history(period="5d")
        if hist.empty:
            raise ValueError(f"No price data for {ticker}")
        spot = float(hist["Close"].iloc[-1])
        logger.info("Spot price for %s: %.2f (yfinance)", ticker, spot)

        expiries = list(asset.options)
        if not expiries:
            raise ValueError(f"No options listed for {ticker}")

        now = datetime.now(tz=timezone.utc)

        # Drop expired and 0DTE expiries
        min_fetch_T = 1 / 365
        expiries = [
            e for e in expiries
            if (pd.Timestamp(e, tz=timezone.utc) - pd.Timestamp(now)
                ).total_seconds() / (365.25 * 86400) >= min_fetch_T
        ]
        if not expiries:
            raise ValueError(f"No future expiries for {ticker}")

        if max_expiries is not None:
            expiries = expiries[:max_expiries]
        logger.info("Expiries selected (%d): %s", len(expiries), expiries)

        frames: list[pd.DataFrame] = []
        for exp_str in expiries:
            try:
                chain = asset.option_chain(exp_str)
            except Exception as exc:
                logger.warning("Skipping expiry %s: %s", exp_str, exc)
                continue

            exp_date = pd.Timestamp(exp_str, tz=timezone.utc)
            T = max((exp_date - pd.Timestamp(now)).total_seconds() / (365.25 * 86400), 1 / 365)

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                sub = df[["strike", "bid", "ask", "openInterest", "volume"]].copy()
                sub["expiry"] = exp_str
                sub["T"] = T
                sub["option_type"] = opt_type
                sub["openInterest"] = sub["openInterest"].fillna(0).astype(int)
                sub["volume"] = sub["volume"].fillna(0).astype(int)
                frames.append(sub)

        if not frames:
            raise ValueError(f"No valid option chains retrieved for {ticker}")

        raw = pd.concat(frames, ignore_index=True)
        raw = raw.rename(columns={"openInterest": "open_interest"})
        logger.info("Raw options fetched: %d rows (yfinance)", len(raw))
        return raw, spot

    def _fetch_alpaca(
        self,
        ticker: str,
        max_expiries: Optional[int],
    ) -> tuple[pd.DataFrame, float]:
        """Fetch via Alpaca Markets option chain API.

        Enriches snapshot data with open_interest from the contracts API
        and daily volume from the bars API.
        """
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest, StockLatestQuoteRequest

        api_key = _get_secret("ALPACA_API_KEY")
        secret_key = _get_secret("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        # Get spot price via stock latest quote
        stock_client = StockHistoricalDataClient(api_key, secret_key)
        quote = stock_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=ticker),
        )
        spot_quote = quote[ticker]
        spot = float((spot_quote.bid_price + spot_quote.ask_price) / 2)
        logger.info("Spot price for %s: %.2f (alpaca)", ticker, spot)

        # Fetch option chain snapshots (bid/ask quotes)
        option_client = OptionHistoricalDataClient(api_key, secret_key)
        request = OptionChainRequest(underlying_symbol=ticker)
        chain = option_client.get_option_chain(request)

        # Enrich with OI from contracts API and volume from bars API
        symbols = [s.replace(" ", "") for s in chain.keys()]
        oi_map = self._fetch_alpaca_oi(api_key, secret_key, ticker)
        volume_map = self._fetch_alpaca_volume(option_client, symbols)

        now = datetime.now(tz=timezone.utc)
        min_fetch_T = 1 / 365
        rows: list[dict] = []

        for symbol, snap in chain.items():
            parsed = _parse_occ_symbol(symbol)
            if parsed is None:
                continue

            exp_date = pd.Timestamp(parsed["expiry"], tz=timezone.utc)
            T = (exp_date - pd.Timestamp(now)).total_seconds() / (365.25 * 86400)
            if T < min_fetch_T:
                continue
            T = max(T, min_fetch_T)

            q = snap.latest_quote
            bid = float(q.bid_price) if q and q.bid_price else 0.0
            ask = float(q.ask_price) if q and q.ask_price else 0.0

            sym_clean = symbol.replace(" ", "")
            rows.append({
                "strike": parsed["strike"],
                "expiry": parsed["expiry"],
                "option_type": parsed["option_type"],
                "bid": bid,
                "ask": ask,
                "open_interest": oi_map.get(sym_clean, 0),
                "volume": volume_map.get(sym_clean, 0),
                "T": T,
            })

        if not rows:
            raise ValueError(f"No options returned by Alpaca for {ticker}")

        raw = pd.DataFrame(rows)
        n_with_oi = (raw["open_interest"] > 0).sum()
        n_with_vol = (raw["volume"] > 0).sum()
        logger.info("Raw options fetched: %d rows, %d with OI, %d with volume (alpaca)",
                     len(raw), n_with_oi, n_with_vol)

        # Limit expiries
        if max_expiries is not None:
            sorted_expiries = sorted(raw["expiry"].unique())[:max_expiries]
            raw = raw[raw["expiry"].isin(sorted_expiries)].reset_index(drop=True)

        return raw, spot

    @staticmethod
    def _fetch_alpaca_oi(
        api_key: str,
        secret_key: str,
        ticker: str,
    ) -> dict[str, int]:
        """Fetch open interest from Alpaca contracts API.

        Returns a mapping of OCC symbol -> open interest.
        Falls back to an empty dict on failure.
        """
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import GetOptionContractsRequest

            oi_map: dict[str, int] = {}
            page_token: Optional[str] = None

            # Try paper=True first (default), fall back to paper=False
            client = None
            for paper in [True, False]:
                try:
                    client = TradingClient(api_key, secret_key, paper=paper)
                    test_req = GetOptionContractsRequest(
                        underlying_symbols=[ticker],
                        limit=1,
                    )
                    client.get_option_contracts(test_req)
                    break
                except Exception:
                    client = None
                    continue

            if client is None:
                logger.warning("Could not connect to Alpaca Trading API for OI data")
                return {}

            while True:
                req = GetOptionContractsRequest(
                    underlying_symbols=[ticker],
                    limit=10000,
                    page_token=page_token,
                )
                response = client.get_option_contracts(req)

                contracts = response.option_contracts or []
                for c in contracts:
                    sym = c.symbol.replace(" ", "")
                    oi = int(float(c.open_interest)) if c.open_interest else 0
                    oi_map[sym] = oi

                if response.next_page_token:
                    page_token = response.next_page_token
                else:
                    break

            logger.info("Fetched OI for %d contracts via contracts API", len(oi_map))
            return oi_map

        except Exception as exc:
            logger.warning("Failed to fetch OI from Alpaca contracts API: %s", exc)
            return {}

    @staticmethod
    def _fetch_alpaca_volume(
        option_client,
        symbols: list[str],
    ) -> dict[str, int]:
        """Fetch daily volume from Alpaca option bars API.

        Returns a mapping of OCC symbol -> daily volume.
        Falls back to an empty dict on failure.
        """
        if not symbols:
            return {}
        try:
            from datetime import timedelta

            from alpaca.data.requests import OptionBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            start = datetime.now(tz=timezone.utc) - timedelta(days=5)

            volume_map: dict[str, int] = {}
            batch_size = 100

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                req = OptionBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
                    start=start,
                )
                bars = option_client.get_option_bars(req)
                bar_data = bars.data if hasattr(bars, "data") else bars
                for sym, bar_list in bar_data.items():
                    if bar_list:
                        volume_map[sym] = int(bar_list[-1].volume)

            logger.info("Fetched daily volume for %d options via bars API",
                         len(volume_map))
            return volume_map

        except Exception as exc:
            logger.warning("Failed to fetch volume from Alpaca bars API: %s", exc)
            return {}

    def _validate_spot(
        self,
        raw: pd.DataFrame,
        spot: float,
        ticker: str,
    ) -> float:
        """Cross-check spot price against option mid-prices.

        If spot is grossly inconsistent with observed option prices (e.g.
        stale/adjusted data after stock splits), infer a corrected spot from
        near-expiry ATM option prices using put-call parity.
        """
        calls = raw[(raw["option_type"] == "call") & (raw["bid"] > 0) & (raw["ask"] > 0)]
        if calls.empty:
            return spot

        # Use nearest expiry
        nearest_T = calls["T"].min()
        near_calls = calls[calls["T"] == nearest_T].copy()
        near_calls["mid"] = (near_calls["bid"] + near_calls["ask"]) / 2

        # Find the strike closest to spot
        near_calls["dist"] = (near_calls["strike"] - spot).abs()
        atm = near_calls.loc[near_calls["dist"].idxmin()]

        K = float(atm["strike"])
        mid = float(atm["mid"])
        T = float(atm["T"])
        discount = np.exp(-self.risk_free_rate * T)

        # Implied spot from this call: S_impl ~ mid + K*exp(-rT)
        implied_spot = mid + K * discount

        # If spot and implied_spot differ by more than 20%, the quote is suspect
        ratio = implied_spot / spot if spot > 0 else float("inf")
        if ratio > 1.20 or ratio < 0.80:
            logger.info(
                "Spot price for %s looks stale (%.2f vs option-implied %.2f). "
                "Using option-implied spot.",
                ticker, spot, implied_spot,
            )
            return implied_spot

        return spot

    def _clean(
        self,
        raw: pd.DataFrame,
        spot: float,
        min_oi: int,
        min_volume: int,
        moneyness_range: tuple[float, float],
    ) -> pd.DataFrame:
        """Apply shared cleaning filters to raw option data."""
        df = raw.copy()
        df["mid_price"] = (df["bid"] + df["ask"]) / 2

        # Remove zero/negative bids or mid
        df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["mid_price"] > 0)]

        # Open interest & volume filters — only apply when the data source
        # actually provides these fields (Alpaca's OI/volume enrichment can
        # fail silently, leaving all values at 0).
        has_oi = (df["open_interest"] > 0).any()
        has_vol = (df["volume"] > 0).any()
        if has_oi:
            df = df[df["open_interest"] >= min_oi]
        if has_vol:
            df = df[df["volume"] >= min_volume]

        # Moneyness filter
        df["moneyness"] = df["strike"] / spot
        lo, hi = moneyness_range
        df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]

        # Remove absurd bid/ask spreads (> 80% of mid)
        df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid_price"]
        df = df[df["spread_pct"] < 0.80]

        df = df.drop(columns=["moneyness", "spread_pct"]).reset_index(drop=True)
        return df
