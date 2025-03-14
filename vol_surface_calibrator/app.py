"""Streamlit dashboard — Volatility Surface Calibrator."""

import logging
import sys
import time
from pathlib import Path

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
import streamlit as st
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

_PRICES_DIR = Path(__file__).resolve().parent.parent / "data" / "prices"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("app")

from engine import DataFetcher, compute_iv_chain, SVICalibrator


# ---------------------------------------------------------------------------
# Risk-neutral density (Breeden-Litzenberger)
# ---------------------------------------------------------------------------
def compute_rnd_heatmap(
    chains: pd.DataFrame,
    spot: float,
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """Compute risk-neutral density from call prices via Breeden-Litzenberger.

    Returns (df_dense, strikes_grid, dates) where df_dense has strikes as rows
    and dates as columns.
    """
    calls = chains[chains["option_type"] == "call"].copy()
    if calls.empty:
        return pd.DataFrame(), np.array([]), pd.DatetimeIndex([])

    strikes_grid = np.linspace(spot * 0.6, spot * 1.4, 600)
    dk = strikes_grid[1] - strikes_grid[0]

    known_pdfs: dict[pd.Timestamp, np.ndarray] = {}

    for expiry, grp in calls.groupby("expiry"):
        grp = grp.sort_values("strike")
        if len(grp) < 6:
            continue
        prices_interp = np.interp(strikes_grid, grp["strike"].values, grp["mid_price"].values)
        prices_smooth = gaussian_filter1d(prices_interp, sigma=4)

        first_deriv = np.gradient(prices_smooth, dk)
        second_deriv = np.gradient(first_deriv, dk)

        pdf = np.maximum(second_deriv, 0)
        pdf = gaussian_filter1d(pdf, sigma=2)
        if pdf.sum() > 0:
            known_pdfs[pd.Timestamp(expiry)] = pdf / pdf.sum()

    if not known_pdfs:
        return pd.DataFrame(), np.array([]), pd.DatetimeIndex([])

    # T=0: tight Gaussian at spot (approximating a delta)
    today = pd.Timestamp.now().normalize()
    sigma_today = (strikes_grid.max() - strikes_grid.min()) * 0.004
    pdf_today = np.exp(-0.5 * ((strikes_grid - spot) / sigma_today) ** 2)
    known_pdfs[today] = pdf_today / pdf_today.sum()

    # Build sparse → dense via time interpolation
    df_sparse = pd.DataFrame(known_pdfs, index=strikes_grid).sort_index(axis=1)
    last_expiry = df_sparse.columns[-1]
    all_dates = pd.date_range(start=today, end=last_expiry, freq="D")

    df_dense = df_sparse.reindex(columns=all_dates)
    df_dense = df_dense.interpolate(method="time", axis=1).fillna(0)

    return df_dense, strikes_grid, all_dates


# ---------------------------------------------------------------------------
# Universe of selectable tickers (liquid options markets)
# ---------------------------------------------------------------------------
UNIVERSE = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IWM": "Russell 2000 ETF",
    "DIA": "Dow Jones ETF",
    "EEM": "Emerging Markets ETF",
    "GLD": "Gold ETF",
    "TLT": "20+ Year Treasury ETF",
    "XLF": "Financials Sector ETF",
    "XLE": "Energy Sector ETF",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "GS": "Goldman Sachs",
    "XOM": "ExxonMobil",
    "BA": "Boeing",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "DIS": "Disney",
    "V": "Visa",
    "KO": "Coca-Cola",
}

TICKER_OPTIONS = [f"{k} — {v}" for k, v in UNIVERSE.items()]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Vol Surface Calibrator", layout="wide")
st.title("Volatility Surface Calibrator")
st.markdown(
    "Fetch live option chains, compute implied volatilities (Brent root-finding on BS), "
    "and calibrate an **SVI (Stochastic Volatility Inspired)** surface."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Parameters")
selected = st.sidebar.selectbox("Underlying", TICKER_OPTIONS, index=0)
ticker = selected.split(" — ")[0]
risk_free = st.sidebar.number_input("Risk-free rate", value=0.045, step=0.005, format="%.3f")
max_expiries = st.sidebar.slider("Max expiries", 2, 12, 6)
min_oi = st.sidebar.number_input("Min Open Interest", value=50, step=10)
min_volume = st.sidebar.number_input("Min Volume", value=5, step=5)
moneyness_lo = st.sidebar.slider("Moneyness range (low)", 0.60, 0.95, 0.80, 0.05)
moneyness_hi = st.sidebar.slider("Moneyness range (high)", 1.05, 1.40, 1.20, 0.05)

# ---------------------------------------------------------------------------
# Fetch & Compute
# ---------------------------------------------------------------------------
if st.button("Fetch & Calibrate", type="primary"):
    global_t0 = time.perf_counter()

    # 1. Fetch
    st.header("1. Market Data")
    fetcher = DataFetcher(risk_free_rate=risk_free)
    try:
        with st.spinner(f"Fetching option chains for {ticker} ..."):
            data = fetcher.fetch(
                ticker,
                min_oi=min_oi,
                min_volume=min_volume,
                max_expiries=max_expiries,
                moneyness_range=(moneyness_lo, moneyness_hi),
            )
    except Exception as exc:
        st.error(f"Failed to fetch data: {exc}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Spot", f"${data.spot:,.2f}")
    col2.metric("Options", f"{data.n_options:,}")
    col3.metric("Expiries", f"{len(data.expiries)}")

    logger.info("Fetched %d options for %s (spot=%.2f)", data.n_options, ticker, data.spot)

    # 2. Compute IV
    st.header("2. Implied Volatility")
    with st.spinner("Computing IV via Brent's algorithm ..."):
        iv_t0 = time.perf_counter()
        iv_df = compute_iv_chain(data.chains, data.spot, data.risk_free_rate)
        iv_elapsed = time.perf_counter() - iv_t0

    st.markdown(f"**{len(iv_df)}** valid IVs computed in **{iv_elapsed:.2f}s**")

    if iv_df.empty:
        st.error("No valid IVs could be computed. Try adjusting filters.")
        st.stop()

    # 2D smile plots
    expiries = sorted(iv_df["expiry"].unique())
    n_exp = len(expiries)
    cols_per_row = min(n_exp, 3)
    rows = (n_exp + cols_per_row - 1) // cols_per_row

    fig_smile = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=[f"T = {iv_df[iv_df['expiry']==e]['T'].iloc[0]:.3f}y — {e}" for e in expiries],
        vertical_spacing=0.08,
    )

    for idx, exp in enumerate(expiries):
        r_idx = idx // cols_per_row + 1
        c_idx = idx % cols_per_row + 1
        grp = iv_df[iv_df["expiry"] == exp]

        for otype, color, sym in [("call", "#2196F3", "circle"), ("put", "#FF9800", "diamond")]:
            sub = grp[grp["option_type"] == otype]
            if sub.empty:
                continue
            fig_smile.add_trace(
                go.Scatter(
                    x=sub["strike"], y=sub["iv"] * 100,
                    mode="markers",
                    marker=dict(color=color, size=5, symbol=sym),
                    name=otype.title() if idx == 0 else None,
                    showlegend=(idx == 0),
                    hovertemplate="K=%{x:.0f}  IV=%{y:.1f}%<extra></extra>",
                ),
                row=r_idx, col=c_idx,
            )

        fig_smile.update_xaxes(title_text="Strike", row=r_idx, col=c_idx)
        fig_smile.update_yaxes(title_text="IV (%)", row=r_idx, col=c_idx)

    fig_smile.update_layout(
        height=300 * rows,
        template="plotly_white",
        title_text="Market Implied Volatility Smiles",
    )
    st.plotly_chart(fig_smile, width="stretch")

    # 3. SVI Calibration
    st.header("3. SVI Calibration")
    with st.spinner("Calibrating SVI slices ..."):
        cal_t0 = time.perf_counter()
        calibrator = SVICalibrator()
        surface = calibrator.calibrate_surface(iv_df, ticker=ticker)
        cal_elapsed = time.perf_counter() - cal_t0

    st.markdown(f"**{len(surface.slices)} slices** calibrated in **{cal_elapsed:.2f}s**")

    # Parameters table
    st.dataframe(
        surface.summary_df().style.format({
            "a": "{:.4f}", "b": "{:.4f}", "rho": "{:.3f}",
            "m": "{:.4f}", "sigma": "{:.4f}", "T": "{:.4f}",
            "rmse": "{:.2e}",
        }),
        width="stretch",
    )

    # Overlay SVI fit on smile plots
    fig_fit = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=[f"{e}" for e in expiries],
        vertical_spacing=0.08,
    )

    for idx, exp in enumerate(expiries):
        r_idx = idx // cols_per_row + 1
        c_idx = idx % cols_per_row + 1
        grp = iv_df[iv_df["expiry"] == exp]
        slc = surface.get_slice(exp)

        # Market points
        fig_fit.add_trace(
            go.Scatter(
                x=grp["strike"], y=grp["iv"] * 100,
                mode="markers",
                marker=dict(color="#888", size=4),
                name="Market" if idx == 0 else None,
                showlegend=(idx == 0),
            ),
            row=r_idx, col=c_idx,
        )

        # SVI fit curve
        if slc is not None:
            k_fine = np.linspace(grp["log_moneyness"].min() - 0.05,
                                 grp["log_moneyness"].max() + 0.05, 200)
            fwd = grp["forward"].iloc[0]
            strikes_fine = fwd * np.exp(k_fine)
            iv_fit = slc.implied_vol(k_fine) * 100

            fig_fit.add_trace(
                go.Scatter(
                    x=strikes_fine, y=iv_fit,
                    mode="lines",
                    line=dict(color="#e74c3c", width=2),
                    name="SVI Fit" if idx == 0 else None,
                    showlegend=(idx == 0),
                ),
                row=r_idx, col=c_idx,
            )

        fig_fit.update_xaxes(title_text="Strike", row=r_idx, col=c_idx)
        fig_fit.update_yaxes(title_text="IV (%)", row=r_idx, col=c_idx)

    fig_fit.update_layout(
        height=300 * rows,
        template="plotly_white",
        title_text="SVI Fit vs Market IV",
    )
    st.plotly_chart(fig_fit, width="stretch")

    # 4. 3D Surface
    st.header("4. 3D Volatility Surface")

    k_grid = np.linspace(-0.35, 0.35, 80)
    T_min = min(s.T for s in surface.slices)
    T_max = max(s.T for s in surface.slices)
    T_grid = np.linspace(T_min, T_max, 60)

    iv_grid = surface.implied_vol_grid(k_grid, T_grid) * 100  # to %

    fig_3d = go.Figure(data=[
        go.Surface(
            x=k_grid,
            y=T_grid,
            z=iv_grid,
            colorscale="Viridis",
            colorbar=dict(title="IV (%)"),
            opacity=0.9,
            hovertemplate=(
                "log-K/F: %{x:.3f}<br>"
                "T: %{y:.3f}y<br>"
                "IV: %{z:.1f}%<extra></extra>"
            ),
        ),
    ])

    # Add market data points
    fig_3d.add_trace(go.Scatter3d(
        x=iv_df["log_moneyness"],
        y=iv_df["T"],
        z=iv_df["iv"] * 100,
        mode="markers",
        marker=dict(size=2.5, color="#e74c3c"),
        name="Market IV",
        hovertemplate="k=%{x:.3f} T=%{y:.3f} IV=%{z:.1f}%<extra></extra>",
    ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Log-Moneyness (k = ln K/F)",
            yaxis_title="Time to Expiry (years)",
            zaxis_title="Implied Volatility (%)",
            camera=dict(eye=dict(x=1.8, y=-1.5, z=0.8)),
        ),
        height=650,
        template="plotly_white",
        title_text=f"SVI Volatility Surface — {ticker}",
    )
    st.plotly_chart(fig_3d, width="stretch")

    # 5. Risk-Neutral Density Heatmap
    st.header("5. Risk-Neutral Density")
    with st.spinner("Computing Breeden-Litzenberger risk-neutral density ..."):
        df_dense, rnd_strikes, rnd_dates = compute_rnd_heatmap(data.chains, data.spot)

    if df_dense.empty:
        st.warning("Not enough call data to compute risk-neutral density.")
    else:
        # Load historical closes for overlay from cache
        try:
            cache_path = _PRICES_DIR / f"{ticker}.parquet"
            cached = pd.read_parquet(cache_path)
            hist_data = cached["Close"].reset_index()
            hist_data.columns = ["date", "close"]
            hist_data["date"] = pd.to_datetime(hist_data["date"]).dt.tz_localize(None)
            hist_start = rnd_dates[0] - datetime.timedelta(days=90)
            hist_sub = hist_data[hist_data["date"] >= hist_start]
        except Exception:
            hist_sub = pd.DataFrame()

        # Apply gamma correction for better contrast
        Z = df_dense.values
        z_max = Z.max() if Z.max() > 0 else 1.0
        Z_norm = np.power(Z / z_max, 0.35) * z_max

        fig_rnd = go.Figure()

        fig_rnd.add_trace(go.Heatmap(
            z=Z_norm,
            x=rnd_dates,
            y=rnd_strikes,
            colorscale="Turbo",
            colorbar=dict(title="Probability Density"),
            hovertemplate="Date: %{x|%b %d}<br>Strike: %{y:.0f}<br>Density: %{z:.4f}<extra></extra>",
        ))

        # Historical price overlay
        if not hist_sub.empty:
            fig_rnd.add_trace(go.Scatter(
                x=hist_sub["date"],
                y=hist_sub["close"],
                mode="lines",
                line=dict(color="cyan", width=2.5),
                name="Historical Price",
                hovertemplate="Date: %{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>",
            ))

        # Current spot horizontal line
        fig_rnd.add_hline(
            y=data.spot,
            line_dash="dash",
            line_color="rgba(0,255,255,0.5)",
            line_width=1,
        )

        fig_rnd.update_layout(
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="rgba(0,0,0,0)",
            title_text=f"Market-Implied Risk-Neutral Density — {ticker}",
            xaxis_title="Date",
            yaxis_title="Strike",
            yaxis=dict(
                range=[data.spot * 0.85, data.spot * 1.15],
                color="gray",
                gridcolor="rgba(255,255,255,0.1)",
            ),
            xaxis=dict(
                range=[
                    (rnd_dates[0] - datetime.timedelta(days=90)).isoformat(),
                    rnd_dates[-1].isoformat(),
                ],
                color="gray",
                gridcolor="rgba(255,255,255,0.1)",
            ),
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(color="white"),
            ),
        )
        st.plotly_chart(fig_rnd, width="stretch")

    # Summary
    total = time.perf_counter() - global_t0
    logger.info("Total pipeline: %.2fs", total)
    st.success(f"Full pipeline completed in **{total:.1f}s**")
