#!/usr/bin/env python3
"""Pre-download Yahoo Finance data for all quant-apps tickers.

Saves daily Close prices as Parquet files and market caps as JSON.
Run periodically (e.g. weekly) to keep data fresh:

    python scripts/refresh_data.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"

# Union of all tickers across portfolio_optimizer, structured_product_factory,
# and vol_surface_calibrator.
ALL_TICKERS = sorted({
    # portfolio_optimizer presets
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ",
    "META", "NVDA", "TSLA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLB", "XLC",
    "SPY", "EFA", "EEM", "TLT", "GLD", "VNQ", "DBC",
    "AVGO", "CRM", "AMD",
    "IEF", "SHY", "LQD", "HYG", "TIP", "BND",
    # structured_product_factory
    "GS", "BAC", "PFE", "UNH", "CVX", "HD", "MCD", "KO", "PG", "DIS",
    "NFLX", "BA", "V", "MA", "CSCO", "INTC", "WMT", "NKE",
    # vol_surface_calibrator extras
    "QQQ", "IWM", "DIA",
})

LOOKBACK_YEARS = 10


def main() -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=LOOKBACK_YEARS)

    # --- Download prices ---
    failed = []
    for ticker in ALL_TICKERS:
        print(f"Downloading {ticker} ...", end=" ", flush=True)
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False)
            if raw.empty:
                raise ValueError("empty DataFrame")
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = "Close"
            close.to_frame().to_parquet(PRICES_DIR / f"{ticker}.parquet")
            print(f"OK ({len(close)} rows)")
        except Exception as exc:
            print(f"FAILED ({exc})")
            failed.append(ticker)
        time.sleep(0.3)

    # --- Download market caps ---
    print("\nFetching market caps...")
    market_caps = {}
    for ticker in ALL_TICKERS:
        if ticker in failed:
            continue
        try:
            info = yf.Ticker(ticker).info
            cap = info.get("marketCap", 0) or 0
            if cap > 0:
                market_caps[ticker] = cap
                print(f"  {ticker}: ${cap:,.0f}")
        except Exception:
            pass
        time.sleep(0.2)

    (DATA_DIR / "market_caps.json").write_text(
        json.dumps(market_caps, indent=2) + "\n"
    )

    # --- Manifest ---
    manifest = {
        "last_updated": pd.Timestamp.now().isoformat(),
        "tickers": [t for t in ALL_TICKERS if t not in failed],
        "lookback_years": LOOKBACK_YEARS,
        "failed": failed,
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"\nDone. {len(ALL_TICKERS) - len(failed)}/{len(ALL_TICKERS)} tickers saved.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
