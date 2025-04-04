"""Market data fetching and covariance estimation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import io
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.covariance import LedoitWolf

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PRICES_DIR = _CACHE_DIR / "prices"


@dataclass
class MarketData:
    """Container for market data used by portfolio optimizers.

    Parameters
    ----------
    tickers : list[str]
        Asset ticker symbols.
    prices : pd.DataFrame
        Daily price series (dates × tickers).
    returns : pd.DataFrame
        Daily log-return series (dates × tickers).
    cov_matrix : np.ndarray
        Annualized covariance matrix (n × n).
    expected_returns : np.ndarray
        Annualized expected returns (n,).
    risk_free_rate : float
        Annualized risk-free rate.
    market_caps : Optional[np.ndarray]
        Market capitalizations for Black-Litterman equilibrium weights.
    """

    tickers: list[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    cov_matrix: np.ndarray
    expected_returns: np.ndarray
    risk_free_rate: float = 0.0
    market_caps: Optional[np.ndarray] = field(default=None)
    ff_factors: Optional[pd.DataFrame] = field(default=None)
    returns_model: str = "historical"
    ff_include_alpha: bool = True
    ff_use_factor_cov: bool = False

    def __post_init__(self) -> None:
        n = len(self.tickers)
        if self.cov_matrix.shape != (n, n):
            raise ValueError(
                f"cov_matrix shape {self.cov_matrix.shape} doesn't match {n} tickers"
            )
        if self.expected_returns.shape != (n,):
            raise ValueError(
                f"expected_returns shape {self.expected_returns.shape} doesn't match {n} tickers"
            )
        # Symmetry check
        if not np.allclose(self.cov_matrix, self.cov_matrix.T, atol=1e-8):
            raise ValueError("cov_matrix is not symmetric")
        # Positive semi-definite check
        eigvals = np.linalg.eigvalsh(self.cov_matrix)
        if np.any(eigvals < -1e-8):
            raise ValueError("cov_matrix is not positive semi-definite")
        if self.market_caps is not None and self.market_caps.shape != (n,):
            raise ValueError(
                f"market_caps shape {self.market_caps.shape} doesn't match {n} tickers"
            )

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def correlation_matrix(self) -> np.ndarray:
        vols = np.sqrt(np.diag(self.cov_matrix))
        outer = np.outer(vols, vols)
        # Avoid division by zero for zero-vol assets
        outer = np.where(outer == 0, 1.0, outer)
        return self.cov_matrix / outer

    @property
    def annualized_vols(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov_matrix))

    @property
    def market_cap_weights(self) -> Optional[np.ndarray]:
        if self.market_caps is None:
            return None
        total = self.market_caps.sum()
        if total == 0:
            return np.ones(self.n_assets) / self.n_assets
        return self.market_caps / total


_FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


def _download_ff_csv(name: str) -> pd.DataFrame:
    """Download a Fama-French CSV zip from Ken French's data library."""
    url = f"{_FF_BASE}/{name}_CSV.zip"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            raw = f.read().decode("utf-8")

    # FF CSVs have header text before the data. Find the first line that
    # starts with a date (all digits) to locate the data block.
    lines = raw.splitlines()
    data_start = None
    header_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped.split(",")[0].strip().isdigit():
            data_start = i
            # The header is the preceding non-empty line
            for j in range(i - 1, -1, -1):
                if lines[j].strip():
                    header_line = j
                    break
            break

    if data_start is None:
        raise ValueError(f"Could not parse FF CSV from {name}")

    # Read from header through end of first data block
    block_lines = [lines[header_line]]
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or not stripped.split(",")[0].strip().replace("-", "").isdigit():
            break
        block_lines.append(stripped)

    df = pd.read_csv(io.StringIO("\n".join(block_lines)))
    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str).str.strip()
    df.index = pd.to_datetime(df[first_col], format="%Y%m%d")
    df = df.drop(columns=[first_col])
    df.columns = [c.strip() for c in df.columns]
    df.index.name = None
    return df


def fetch_ff_factors(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch Fama-French 6 factors (5 factors + Momentum) at daily frequency.

    Returns a DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    Values are in decimal form (not percent).
    """
    ff5 = _download_ff_csv("F-F_Research_Data_5_Factors_2x3_daily")
    mom = _download_ff_csv("F-F_Momentum_Factor_daily")

    # Rename momentum column for consistency
    col_map = {}
    for c in mom.columns:
        if "mom" in c.strip().lower():
            col_map[c] = "Mom"
    mom = mom.rename(columns=col_map)

    # Merge and convert from % to decimal
    factors = ff5.join(mom, how="inner") / 100.0

    # Filter to requested date range
    factors = factors.loc[
        (factors.index >= pd.Timestamp(start_date))
        & (factors.index <= pd.Timestamp(end_date))
    ]

    return factors


_FF6_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _ff6_regression(
    log_returns: pd.DataFrame,
    ff_factors: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run FF6 OLS regression for each asset.

    Returns
    -------
    alphas : (n,)
    betas : (n, 6) — floored at zero
    residuals : (T, n) — computed from raw (unfloored) coefficients
    """
    common = log_returns.index.intersection(ff_factors.index)
    if len(common) < 60:
        raise ValueError(
            f"Only {len(common)} overlapping dates between returns and FF factors "
            "(need at least 60)"
        )
    ret = log_returns.loc[common]
    ff = ff_factors.loc[common]
    X = np.column_stack([np.ones(len(common)), ff[_FF6_FACTOR_COLS].values])  # (T, 7)
    n = ret.shape[1]
    alphas = np.empty(n)
    betas = np.empty((n, 6))
    residuals = np.empty((len(common), n))
    for i in range(n):
        y = ret.iloc[:, i].values - ff["RF"].values
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alphas[i] = coeffs[0]
        # negative beta means unexposed, not short the factor — floor at zero
        betas[i] = np.maximum(coeffs[1:], 0.0)
        residuals[:, i] = y - X @ coeffs  # use raw coeffs for residuals
    return alphas, betas, residuals


def estimate_expected_returns(
    log_returns: pd.DataFrame,
    method: str = "historical",
    ff_factors: Optional[pd.DataFrame] = None,
    include_alpha: bool = True,
) -> np.ndarray:
    """Estimate annualized expected returns.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns (dates × tickers).
    method : str
        ``"historical"`` for simple mean, ``"ff6"`` for Fama-French 6-factor model.
    ff_factors : pd.DataFrame, optional
        Fama-French factors (required when *method* is ``"ff6"``).
    include_alpha : bool
        If True, include the regression intercept (alpha) in the FF6 expected
        return.  When False, the expected return is purely factor-implied
        (``betas @ factor_means + RF``), which is less noisy out-of-sample.

    Returns
    -------
    np.ndarray
        Annualized expected return per asset, shape ``(n_assets,)``.
    """
    if method == "historical":
        return log_returns.mean().values * 252

    if method != "ff6":
        raise ValueError(f"Unknown returns model: {method!r}")

    if ff_factors is None:
        raise ValueError("ff_factors must be provided for the ff6 model")

    common = log_returns.index.intersection(ff_factors.index)
    ff = ff_factors.loc[common]

    alphas, betas, _ = _ff6_regression(log_returns, ff_factors)

    factor_means = ff[_FF6_FACTOR_COLS].mean().values  # (6,)
    rf_mean = ff["RF"].mean()

    alpha_vec = alphas if include_alpha else np.zeros(len(alphas))
    expected = (alpha_vec + betas @ factor_means) * 252 + rf_mean * 252

    return expected


def estimate_factor_covariance(
    log_returns: pd.DataFrame,
    ff_factors: pd.DataFrame,
) -> np.ndarray:
    """Estimate annualized covariance via factor structure: B Σ_F Bᵀ + D."""
    common = log_returns.index.intersection(ff_factors.index)
    ff = ff_factors.loc[common]

    _, betas, residuals = _ff6_regression(log_returns, ff_factors)

    # Factor covariance (daily → annualized)
    cov_F = np.cov(ff[_FF6_FACTOR_COLS].values, rowvar=False) * 252  # (6, 6)

    # Idiosyncratic variance (diagonal, annualized)
    D = np.diag(np.var(residuals, axis=0, ddof=1) * 252)  # (n, n)

    # Structured covariance
    cov = betas @ cov_F @ betas.T + D  # (n, n)

    # Force symmetry (numerical precision)
    cov = (cov + cov.T) / 2
    return cov


def fetch_market_data(
    tickers: list[str],
    lookback_years: float = 5.0,
    risk_free_rate: float = 0.02,
    returns_model: str = "historical",
    ff_include_alpha: bool = True,
    ff_use_factor_cov: bool = False,
) -> MarketData:
    """Fetch market data from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols.
    lookback_years : float
        Number of years of historical data.
    risk_free_rate : float
        Annualized risk-free rate.
    returns_model : str
        ``"historical"`` or ``"ff6"`` (Fama-French 6-factor).
    ff_include_alpha : bool
        Whether to include alpha in FF6 expected returns.
    ff_use_factor_cov : bool
        Use factor-structured covariance (B Σ_F Bᵀ + D) instead of Ledoit-Wolf
        when the FF6 model is active.

    Returns
    -------
    MarketData
        Populated market data object.
    """
    price_frames = []
    for ticker in tickers:
        cache_path = _PRICES_DIR / f"{ticker}.parquet"
        if not cache_path.exists():
            raise RuntimeError(
                f"No cached data for {ticker}. Run scripts/refresh_data.py first."
            )
        cached = pd.read_parquet(cache_path)
        close = cached["Close"]
        close.name = ticker
        price_frames.append(close)

    prices = pd.concat(price_frames, axis=1).dropna()

    # Trim to lookback window
    n_days = int(lookback_years * 252)
    if len(prices) > n_days:
        prices = prices.iloc[-n_days:]

    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Fama-French factors (fetch once, reused by backtester)
    ff_factors = None
    if returns_model == "ff6":
        ff_factors = fetch_ff_factors(prices.index[0], prices.index[-1])

    # Covariance matrix (annualized)
    if returns_model == "ff6" and ff_use_factor_cov:
        cov_annual = estimate_factor_covariance(log_returns, ff_factors)
    else:
        lw = LedoitWolf().fit(log_returns.values)
        cov_annual = lw.covariance_ * 252

    # Annualized expected returns
    expected_returns = estimate_expected_returns(
        log_returns, returns_model, ff_factors, include_alpha=ff_include_alpha
    )

    # Market caps from cache
    market_caps = np.zeros(len(tickers))
    caps_path = _CACHE_DIR / "market_caps.json"
    if caps_path.exists():
        cached_caps = json.loads(caps_path.read_text())
        for i, ticker in enumerate(tickers):
            market_caps[i] = cached_caps.get(ticker, 0)

    mc = market_caps if market_caps.sum() > 0 else None

    return MarketData(
        tickers=tickers,
        prices=prices,
        returns=log_returns,
        cov_matrix=cov_annual,
        expected_returns=expected_returns,
        risk_free_rate=risk_free_rate,
        market_caps=mc,
        ff_factors=ff_factors,
        returns_model=returns_model,
        ff_include_alpha=ff_include_alpha,
        ff_use_factor_cov=ff_use_factor_cov,
    )
