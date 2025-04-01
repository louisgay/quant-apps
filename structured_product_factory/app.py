"""Streamlit UI for the Structured Product Factory."""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

_PRICES_DIR = Path(__file__).resolve().parent.parent / "data" / "prices"

# ---------------------------------------------------------------------------
# Logging — visible in docker logs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("app")

from sklearn.covariance import LedoitWolf

from engine import MarketData, MonteCarloEngine, AutocallablePhoenix, AutocallableAthena, GreeksCalculator

# ---------------------------------------------------------------------------
# Universe of selectable underlyings
# ---------------------------------------------------------------------------
UNIVERSE = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "GS": "Goldman Sachs",
    "BAC": "Bank of America",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth",
    "XOM": "ExxonMobil",
    "CVX": "Chevron",
    "HD": "Home Depot",
    "MCD": "McDonald's",
    "KO": "Coca-Cola",
    "PG": "Procter & Gamble",
    "DIS": "Disney",
    "NFLX": "Netflix",
    "BA": "Boeing",
    "V": "Visa",
    "MA": "Mastercard",
    "CSCO": "Cisco",
    "INTC": "Intel",
    "AMD": "AMD",
    "CRM": "Salesforce",
    "WMT": "Walmart",
    "NKE": "Nike",
}

TICKER_OPTIONS = [f"{k} — {v}" for k, v in UNIVERSE.items()]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_market_data(tickers: tuple[str, ...], lookback_years: int = 2) -> dict:
    """Load spots, annualized vols, and correlation matrix from cached Parquet data."""
    frames = []
    for ticker in tickers:
        cache_path = _PRICES_DIR / f"{ticker}.parquet"
        if not cache_path.exists():
            raise ValueError(
                f"No cached data for {ticker}. Run scripts/refresh_data.py first."
            )
        cached = pd.read_parquet(cache_path)
        close = cached["Close"]
        close.name = ticker
        frames.append(close)

    if not frames:
        raise ValueError("No price data retrieved for any ticker.")

    prices = pd.concat(frames, axis=1).dropna()

    # Trim to lookback window
    n_days = int(lookback_years * 252)
    if len(prices) > n_days:
        prices = prices.iloc[-n_days:]

    if prices.empty:
        raise ValueError("No overlapping price data after dropping NaNs.")

    log_returns = np.log(prices / prices.shift(1)).dropna()

    spots = prices.iloc[-1].values.astype(np.float64)

    # Ledoit-Wolf shrinkage covariance
    lw = LedoitWolf().fit(log_returns.values)
    cov_daily = lw.covariance_
    vols = np.sqrt(np.diag(cov_daily) * 252)

    # Correlation from shrunk covariance
    std = np.sqrt(np.diag(cov_daily))
    corr_matrix = cov_daily / np.outer(std, std)

    return {
        "spots": spots,
        "volatilities": vols,
        "correlation_matrix": corr_matrix,
    }


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Structured Product Factory", layout="wide")
st.title("Structured Product Factory")
st.markdown("Design, price, and risk-manage an **Autocallable** structured note.")

# ---------------------------------------------------------------------------
# Sidebar – Product parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Product Parameters")

product_type = st.sidebar.selectbox("Product type", ["Phoenix", "Athena"])

selected = st.sidebar.multiselect(
    "Underlyings",
    TICKER_OPTIONS,
    default=["AAPL — Apple", "MSFT — Microsoft", "GOOGL — Alphabet"],
)
tickers = [s.split(" — ")[0] for s in selected]
n = len(tickers)

if n < 2:
    st.sidebar.warning("Select at least 2 underlyings.")
    st.stop()

logger.info("Product type: %s | Underlyings: %s (n=%d)", product_type, tickers, n)

notional = st.sidebar.number_input("Notional (USD)", value=1_000_000, step=100_000)
maturity = st.sidebar.slider("Maturity (years)", 1.0, 5.0, 3.0, 0.5)
freq = st.sidebar.selectbox("Observation frequency", ["quarterly", "semi-annual", "annual", "monthly"])
autocall = st.sidebar.slider("Autocall barrier", 0.80, 1.20, 1.00, 0.05)

if product_type == "Phoenix":
    coupon_barrier = st.sidebar.slider("Coupon barrier", 0.50, 0.90, 0.65, 0.05)
    memory = st.sidebar.checkbox("Memory feature", value=True)

coupon_rate = st.sidebar.slider("Coupon rate (p.a.)", 0.02, 0.15, 0.07, 0.01)
put_barrier = st.sidebar.slider("Put barrier (capital protection)", 0.40, 0.80, 0.55, 0.05)

# ---------------------------------------------------------------------------
# Sidebar – Market data (auto-fetched)
# ---------------------------------------------------------------------------
st.sidebar.header("Market Data")

risk_free = st.sidebar.number_input("Risk-free rate", value=0.04, step=0.005, format="%.3f")

n_paths = st.sidebar.select_slider(
    "Monte Carlo paths", [10_000, 50_000, 100_000, 200_000, 500_000], value=100_000
)

with st.spinner("Fetching live market data..."):
    try:
        live = fetch_live_market_data(tuple(tickers))
    except (ValueError, Exception) as e:
        st.error(f"Failed to fetch market data: {e}")
        st.stop()
    spots = live["spots"]
    vols = live["volatilities"]
    corr = live["correlation_matrix"]

# Display fetched data in sidebar
st.sidebar.markdown("**Live data** (from Yahoo Finance)")
for t, s, v in zip(tickers, spots, vols):
    st.sidebar.text(f"{t}  spot=${s:.2f}  vol={v:.0%}")

logger.info("Market params | spots=%s | vols=%s | r=%.3f | paths=%d",
            spots, vols, risk_free, n_paths)

# ---------------------------------------------------------------------------
# Build objects
# ---------------------------------------------------------------------------
md = MarketData(tickers=tickers, spots=spots, volatilities=vols,
                correlation_matrix=corr, risk_free_rate=risk_free)
engine = MonteCarloEngine(n_paths=n_paths, seed=42)

if product_type == "Phoenix":
    product = AutocallablePhoenix(
        underlyings=tickers, notional=notional, maturity_years=maturity,
        observation_frequency=freq, autocall_barrier=autocall,
        coupon_barrier=coupon_barrier, coupon_rate=coupon_rate,
        put_barrier=put_barrier, memory_feature=memory,
    )
    logger.info(
        "Phoenix built | maturity=%.1fy | freq=%s | autocall=%.0f%% | coupon_barrier=%.0f%% | put=%.0f%%",
        maturity, freq, autocall*100, coupon_barrier*100, put_barrier*100,
    )
else:
    product = AutocallableAthena(
        underlyings=tickers, notional=notional, maturity_years=maturity,
        observation_frequency=freq, autocall_barrier=autocall,
        coupon_rate=coupon_rate, put_barrier=put_barrier,
    )
    logger.info(
        "Athena built | maturity=%.1fy | freq=%s | autocall=%.0f%% | coupon=%.0f%% p.a. | put=%.0f%%",
        maturity, freq, autocall*100, coupon_rate*100, put_barrier*100,
    )

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
st.header("Pricing")

if st.button("Run Pricing Engine", type="primary"):
    global_t0 = time.perf_counter()

    # ==================================================================
    # 1. Monte Carlo simulation
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STARTING PRICING RUN (%s)", product_type)
    logger.info("=" * 60)

    with st.spinner("Running Monte Carlo simulation..."):
        sim_t0 = time.perf_counter()
        paths = engine.simulate(md, product.observation_dates)
        sim_elapsed = time.perf_counter() - sim_t0
        logger.info("MC simulation completed in %.3fs", sim_elapsed)

    with st.spinner("Computing product price..."):
        price_t0 = time.perf_counter()
        result = product.price(paths, md.spots, md.risk_free_rate)
        price_elapsed = time.perf_counter() - price_t0
        logger.info("Product priced in %.3fs", price_elapsed)

    # Display pricing results
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fair Value", f"{result['price']:,.0f} USD")
    col2.metric("Price (% notional)", f"{result['price_pct']:.2f}%")
    col3.metric("Autocall Prob.", f"{result['autocall_prob']:.1%}")
    col4.metric("Capital Loss Prob.", f"{result['capital_loss_prob']:.1%}")

    st.markdown(f"**MC std error**: {result['std_error']:,.0f} USD &nbsp;|&nbsp; "
                f"**Avg life**: {result['avg_life_years']:.2f} years &nbsp;|&nbsp; "
                f"**Simulation**: {sim_elapsed:.2f}s &nbsp;|&nbsp; "
                f"**Pricing**: {price_elapsed:.2f}s")

    # ==================================================================
    # 1b. Price Decomposition
    # ==================================================================
    st.header("Price Decomposition")
    logger.info("Building decomposition display...")

    comp = result["components"]
    zcb = comp["zcb"]
    coupons = comp["coupons"]
    put = comp["short_put"]
    ac_opt = comp["autocall_option"]

    coupon_label = "Athena Coupons" if product_type == "Athena" else "Digital Coupons"

    # Summary equation
    st.markdown(
        f"**ZCB** {zcb['pct']:+.2f}% &nbsp;+&nbsp; "
        f"**Coupons** {coupons['pct']:+.2f}% &nbsp;+&nbsp; "
        f"**Short Put** {put['pct']:+.2f}% &nbsp;=&nbsp; "
        f"**Full Price** {result['price_pct']:.2f}%"
    )

    # Decomposition table
    decomp_df = pd.DataFrame({
        "Component": [
            "Zero-Coupon Bond (ZCB)",
            "  of which: ZCB to Maturity",
            "  of which: Autocall Option",
            coupon_label,
            "Short Put (Down-and-In)",
            "TOTAL",
        ],
        "PV (USD)": [
            zcb["mean"],
            zcb["mean"] - ac_opt["mean"],
            ac_opt["mean"],
            coupons["mean"],
            put["mean"],
            result["price"],
        ],
        "% Notional": [
            zcb["pct"],
            zcb["pct"] - ac_opt["pct"],
            ac_opt["pct"],
            coupons["pct"],
            put["pct"],
            result["price_pct"],
        ],
        "Std Error (%)": [
            zcb["std_error_pct"],
            None,
            ac_opt["std_error_pct"],
            coupons["std_error_pct"],
            put["std_error_pct"],
            result["std_error"] / notional * 100,
        ],
    }).set_index("Component")

    st.dataframe(
        decomp_df.style.format({
            "PV (USD)": "{:+,.0f}",
            "% Notional": "{:+.2f}",
            "Std Error (%)": "{:.3f}",
        }, na_rep="-"),
        width="stretch",
    )

    logger.info(
        "Decomposition | ZCB=%.2f%% (AC_opt=%.2f%%) + Coupons=%.2f%% + Put=%.2f%% = %.2f%%",
        zcb["pct"], ac_opt["pct"], coupons["pct"], put["pct"], result["price_pct"],
    )

    # Waterfall chart
    fig_wf = go.Figure(go.Waterfall(
        x=["ZCB to Maturity", "Autocall Option", coupon_label, "Short Put", "Full Price"],
        y=[
            zcb["pct"] - ac_opt["pct"],
            ac_opt["pct"],
            coupons["pct"],
            put["pct"],
            0,
        ],
        measure=["relative", "relative", "relative", "relative", "total"],
        text=[
            f"{zcb['pct'] - ac_opt['pct']:+.2f}%",
            f"{ac_opt['pct']:+.2f}%",
            f"{coupons['pct']:+.2f}%",
            f"{put['pct']:+.2f}%",
            f"{result['price_pct']:.2f}%",
        ],
        textposition="outside",
        connector=dict(line=dict(color="rgba(150,150,150,0.4)")),
        increasing=dict(marker=dict(color="#2ecc71")),
        decreasing=dict(marker=dict(color="#e74c3c")),
        totals=dict(marker=dict(color="#3498db")),
    ))
    fig_wf.update_layout(
        title="Price Waterfall — Component Decomposition (% of Notional)",
        yaxis_title="% of Notional",
        template="plotly_white",
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_wf, width="stretch")

    # ==================================================================
    # 2. Monte Carlo path visualization
    # ==================================================================
    st.header("Monte Carlo Simulation — Sample Paths")
    logger.info("Building MC path visualization...")

    obs = product.observation_dates
    times = np.concatenate([[0.0], obs])
    n_sample = min(300, engine.n_paths)

    fig_mc = make_subplots(
        rows=1, cols=n,
        subplot_titles=tickers,
        shared_yaxes=False,
        horizontal_spacing=0.05,
    )

    for i, ticker in enumerate(tickers):
        sample = paths[:n_sample, i, :]
        full = np.column_stack([np.full(n_sample, md.spots[i]), sample])

        # Plot sample paths
        for p_idx in range(n_sample):
            fig_mc.add_trace(
                go.Scatter(
                    x=times, y=full[p_idx],
                    mode="lines",
                    line=dict(color="steelblue", width=0.3),
                    opacity=0.08,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=i + 1,
            )

        # Mean path
        mean_path = np.concatenate([[md.spots[i]], paths[:, i, :].mean(axis=0)])
        fig_mc.add_trace(
            go.Scatter(
                x=times, y=mean_path,
                mode="lines",
                line=dict(color="white", width=2.5),
                name="Mean" if i == 0 else None,
                showlegend=(i == 0),
            ),
            row=1, col=i + 1,
        )

        # Barrier lines
        S0 = md.spots[i]
        barriers = [
            (autocall, "#2ecc71", "Autocall"),
            (put_barrier, "#e74c3c", "Put"),
        ]
        if product_type == "Phoenix":
            barriers.insert(1, (coupon_barrier, "#f39c12", "Coupon"))

        for level, color, label in barriers:
            fig_mc.add_trace(
                go.Scatter(
                    x=[0, obs[-1]], y=[S0 * level, S0 * level],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{label} ({level:.0%})" if i == 0 else None,
                    showlegend=(i == 0),
                ),
                row=1, col=i + 1,
            )

        fig_mc.update_xaxes(title_text="Time (years)", row=1, col=i + 1)
        if i == 0:
            fig_mc.update_yaxes(title_text="Price (USD)", row=1, col=i + 1)

    fig_mc.update_layout(
        height=450,
        template="plotly_dark",
        title_text=f"Simulated GBM Paths ({n_sample} of {engine.n_paths:,} paths)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_mc, width="stretch")
    logger.info("MC visualization rendered")

    # ==================================================================
    # 3. Greeks
    # ==================================================================
    st.header("Greeks")
    logger.info("Starting Greeks computation...")

    with st.spinner("Computing Delta..."):
        greeks_t0 = time.perf_counter()
        calc = GreeksCalculator(engine, product, md)
        deltas = calc.delta()
        logger.info("Delta computed: %s", deltas)

    with st.spinner("Computing Gamma..."):
        gammas = calc.gamma()
        logger.info("Gamma computed: %s", gammas)

    with st.spinner("Computing Vega..."):
        vegas = calc.vega()
        logger.info("Vega computed: %s", vegas)

    greeks_elapsed = time.perf_counter() - greeks_t0
    logger.info("All Greeks done in %.3fs", greeks_elapsed)

    greeks_df = pd.DataFrame({
        "Delta (USD/1pt)": deltas,
        "Gamma (USD/1pt²)": gammas,
        "Vega (USD/1vol pt)": vegas,
    })

    st.dataframe(
        greeks_df.style.format("{:+,.2f}"),
        width="stretch",
    )

    # Greeks bar chart
    fig_greeks = make_subplots(rows=1, cols=3, subplot_titles=["Delta", "Gamma", "Vega"])
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:n]

    for col_idx, (greek_name, greek_vals) in enumerate([
        ("Delta", deltas), ("Gamma", gammas), ("Vega", vegas)
    ]):
        fig_greeks.add_trace(
            go.Bar(
                x=list(greek_vals.keys()),
                y=list(greek_vals.values()),
                marker_color=colors,
                showlegend=False,
            ),
            row=1, col=col_idx + 1,
        )

    fig_greeks.update_layout(height=350, template="plotly_white",
                             title_text=f"Greeks Profile (computed in {greeks_elapsed:.1f}s)")
    st.plotly_chart(fig_greeks, width="stretch")

    # ==================================================================
    # 4. Correlation sensitivity
    # ==================================================================
    st.header("Correlation Sensitivity")
    logger.info("Starting correlation sensitivity...")

    with st.spinner("Running correlation stress scenarios..."):
        corr_t0 = time.perf_counter()
        bumps = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
        cs = calc.correlation_sensitivity(bumps)
        corr_elapsed = time.perf_counter() - corr_t0
        logger.info("Correlation sensitivity done in %.3fs: %s", corr_elapsed, cs)

    base_key = "+0%"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(cs.keys()),
        y=list(cs.values()),
        marker_color=["#e74c3c" if v < cs[base_key] else "#2ecc71" for v in cs.values()],
        text=[f"{(v - cs[base_key]) * notional / 100:+,.0f}" for v in cs.values()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Product Fair Value vs. Correlation Shock",
        xaxis_title="Correlation Bump",
        yaxis_title="Price (% of Notional)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "**Insight for the bank's trading desk:** A correlation drop of 10% "
        f"moves the product value from {cs[base_key]:.2f}% to {cs['-10%']:.2f}% of notional. "
        f"That's a P&L impact of **{(cs['-10%'] - cs[base_key]) * notional / 100:+,.0f} USD** "
        "on the issuer's hedging book."
    )

    # ==================================================================
    # Summary
    # ==================================================================
    total_elapsed = time.perf_counter() - global_t0
    logger.info("=" * 60)
    logger.info("TOTAL RUN TIME: %.2fs", total_elapsed)
    logger.info("=" * 60)
    st.success(f"All computations completed in **{total_elapsed:.1f}s**")
