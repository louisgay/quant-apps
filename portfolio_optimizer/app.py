"""Streamlit UI for Portfolio Optimizer."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.data import MarketData, fetch_market_data
from engine.optimizer import (
    BlackLittermanOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
)
from engine.analytics import (
    Backtester,
    PortfolioMetrics,
    drawdown_series,
    rolling_sharpe,
)

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimizer")
st.caption("Black-Litterman | Risk Parity | Mean-Variance (Markowitz)")

# ── Sidebar ──────────────────────────────────────────────────────────────────

PRESET_PORTFOLIOS = {
    "US Diversified": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ"],
    "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "Sector ETFs": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLB", "XLC"],
    "Global Macro ETFs": ["SPY", "EFA", "EEM", "TLT", "GLD", "VNQ", "DBC"],
    "US Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "CRM", "AMD"],
    "Bonds & Rates": ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "BND"],
    "Custom": [],
}

ALL_TICKERS = sorted(
    {t for tickers_list in PRESET_PORTFOLIOS.values() for t in tickers_list}
)

with st.sidebar:
    st.header("Parameters")

    preset = st.selectbox("Asset universe", list(PRESET_PORTFOLIOS.keys()))

    if preset == "Custom":
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL, MSFT, GOOGL, AMZN, JPM, XOM, JNJ",
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    else:
        tickers = st.multiselect(
            "Assets",
            options=ALL_TICKERS,
            default=PRESET_PORTFOLIOS[preset],
        )

    lookback = st.slider("Lookback (years)", 1, 10, 5)
    rf_rate = st.number_input("Risk-free rate", 0.0, 0.10, 0.02, 0.005, format="%.3f")

    returns_model_label = st.selectbox(
        "Expected returns model",
        ["Historical Mean", "Fama-French 6-Factor"],
    )
    returns_model = "historical" if returns_model_label == "Historical Mean" else "ff6"

    ff_include_alpha = True
    ff_use_factor_cov = False
    if returns_model == "ff6":
        ff_include_alpha = st.checkbox(
            "Include alpha (intercept)",
            value=False,
            help="When unchecked, expected returns are purely factor-implied (less noisy out-of-sample).",
        )
        ff_use_factor_cov = st.checkbox(
            "Factor covariance matrix",
            value=False,
            help="Use B Σ_F Bᵀ + D instead of Ledoit-Wolf. Structured estimate from factor betas.",
        )

    method = st.selectbox(
        "Optimization method",
        ["Max Sharpe", "Min Variance", "Risk Parity", "Black-Litterman"],
    )

    long_only = st.checkbox("Long-only", value=True)
    max_weight = st.slider("Max weight per asset", 0.1, 1.0, 1.0, 0.05)

    # Black-Litterman views
    bl_views_p = None
    bl_views_q = None
    if method == "Black-Litterman":
        with st.expander("Black-Litterman Views", expanded=True):
            st.markdown("Define relative or absolute views on assets.")
            n_views = st.number_input("Number of views", 0, 5, 0, key="n_views")
            views_p_rows = []
            views_q_vals = []
            for v in range(int(n_views)):
                st.markdown(f"**View {v + 1}**")
                col1, col2 = st.columns(2)
                with col1:
                    long_asset = st.selectbox(
                        f"Long asset (view {v+1})", tickers, key=f"long_{v}"
                    )
                    short_asset = st.selectbox(
                        f"Short asset (view {v+1}, or None for absolute)",
                        ["None"] + tickers,
                        key=f"short_{v}",
                    )
                with col2:
                    view_val = st.number_input(
                        f"Expected return/spread (view {v+1})",
                        -0.5,
                        0.5,
                        0.05,
                        0.01,
                        key=f"view_val_{v}",
                    )
                row = np.zeros(len(tickers))
                row[tickers.index(long_asset)] = 1.0
                if short_asset != "None":
                    row[tickers.index(short_asset)] = -1.0
                views_p_rows.append(row)
                views_q_vals.append(view_val)

            if n_views > 0:
                bl_views_p = np.array(views_p_rows)
                bl_views_q = np.array(views_q_vals)

    rebalance_freq = st.selectbox(
        "Rebalance frequency",
        {"Monthly": 21, "Quarterly": 63, "Semi-Annual": 126, "Annual": 252},
        format_func=lambda x: x,
    )
    rebalance_days = {"Monthly": 21, "Quarterly": 63, "Semi-Annual": 126, "Annual": 252}[
        rebalance_freq
    ]

    run_btn = st.button("Run Optimization", type="primary", use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────────

if run_btn:
    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers.")
        st.stop()

    # Fetch data
    with st.spinner("Fetching market data..."):
        try:
            data = fetch_market_data(tickers, lookback_years=lookback, risk_free_rate=rf_rate, returns_model=returns_model, ff_include_alpha=ff_include_alpha, ff_use_factor_cov=ff_use_factor_cov)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    # ── Section 1: Data Summary ──────────────────────────────────────────

    st.header("1. Data Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Heatmap")
        corr = data.correlation_matrix
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr,
                x=tickers,
                y=tickers,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=np.round(corr, 2),
                texttemplate="%{text}",
            )
        )
        fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.subheader("Returns & Volatility")
        summary = pd.DataFrame(
            {
                "Ann. Return": data.expected_returns,
                "Ann. Vol": data.annualized_vols,
                "Sharpe": (data.expected_returns - rf_rate) / data.annualized_vols,
            },
            index=tickers,
        )
        st.dataframe(summary.style.format("{:.2%}"), use_container_width=True)

    # ── Optimization ─────────────────────────────────────────────────────

    with st.spinner("Running optimization..."):
        if method == "Max Sharpe":
            opt = MeanVarianceOptimizer(data, long_only=long_only, max_weight=max_weight)
            result = opt.max_sharpe()
        elif method == "Min Variance":
            opt = MeanVarianceOptimizer(data, long_only=long_only, max_weight=max_weight)
            result = opt.min_variance()
        elif method == "Risk Parity":
            opt = RiskParityOptimizer(data)
            result = opt.optimize()
        else:  # Black-Litterman
            opt = BlackLittermanOptimizer(
                data, long_only=long_only, max_weight=max_weight
            )
            result = opt.optimize(P=bl_views_p, Q=bl_views_q)

    # ── Section 2: Efficient Frontier ────────────────────────────────────

    st.header("2. Efficient Frontier")
    mv_opt = MeanVarianceOptimizer(data, long_only=long_only, max_weight=max_weight)

    with st.spinner("Computing efficient frontier..."):
        frontier = mv_opt.efficient_frontier(n_points=50)

    fig_ef = go.Figure()

    # Frontier curve
    fig_ef.add_trace(
        go.Scatter(
            x=[r.volatility for r in frontier],
            y=[r.expected_return for r in frontier],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="blue", width=2),
        )
    )

    # Optimal portfolio
    fig_ef.add_trace(
        go.Scatter(
            x=[result.volatility],
            y=[result.expected_return],
            mode="markers",
            name=f"Optimal ({method})",
            marker=dict(color="red", size=14, symbol="star"),
        )
    )

    # Equal-weight portfolio
    ew_w = np.ones(data.n_assets) / data.n_assets
    ew_ret = float(ew_w @ data.expected_returns)
    ew_vol = float(np.sqrt(ew_w @ data.cov_matrix @ ew_w))
    fig_ef.add_trace(
        go.Scatter(
            x=[ew_vol],
            y=[ew_ret],
            mode="markers",
            name="Equal Weight",
            marker=dict(color="green", size=12, symbol="diamond"),
        )
    )

    # Individual assets
    fig_ef.add_trace(
        go.Scatter(
            x=data.annualized_vols.tolist(),
            y=data.expected_returns.tolist(),
            mode="markers+text",
            name="Assets",
            text=tickers,
            textposition="top center",
            marker=dict(color="gray", size=8),
        )
    )

    fig_ef.update_layout(
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        height=500,
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── Section 3: Optimal Weights ───────────────────────────────────────

    st.header("3. Optimal Weights")
    weights_df = pd.DataFrame(
        {"Weight": result.weights}, index=tickers
    ).sort_values("Weight", ascending=True)

    fig_w = go.Figure(
        go.Bar(
            x=weights_df["Weight"],
            y=weights_df.index,
            orientation="h",
            marker_color="steelblue",
            text=[f"{w:.1%}" for w in weights_df["Weight"]],
            textposition="outside",
        )
    )
    fig_w.update_layout(
        xaxis_title="Weight",
        xaxis_tickformat=".0%",
        height=max(300, 40 * len(tickers)),
        margin=dict(l=20, r=60, t=30, b=20),
    )
    st.plotly_chart(fig_w, use_container_width=True)

    # ── Section 4: Backtest (Walk-Forward) ──────────────────────────────

    st.header("4. Backtest (Walk-Forward)")

    def make_factory(method_name, lo, mw, P=None, Q=None):
        """Create optimizer factory for backtester."""
        def factory(md: MarketData) -> dict:
            if method_name == "Max Sharpe":
                o = MeanVarianceOptimizer(md, long_only=lo, max_weight=mw)
                return {"weights": o.max_sharpe().weights}
            elif method_name == "Min Variance":
                o = MeanVarianceOptimizer(md, long_only=lo, max_weight=mw)
                return {"weights": o.min_variance().weights}
            elif method_name == "Risk Parity":
                o = RiskParityOptimizer(md)
                return {"weights": o.optimize().weights}
            else:
                o = BlackLittermanOptimizer(md, long_only=lo, max_weight=mw)
                return {"weights": o.optimize(P=P, Q=Q).weights}
        return factory

    factory = make_factory(method, long_only, max_weight, bl_views_p, bl_views_q)
    bt = Backtester(factory, rebalance_freq=rebalance_days, lookback_window=252, risk_free_rate=rf_rate)

    with st.spinner("Running backtest..."):
        bt_result = bt.run(data)

    # Equity curve
    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(
            y=bt_result.equity_curve,
            mode="lines",
            name=f"Optimized ({method})",
            line=dict(color="blue", width=2),
        )
    )
    fig_eq.add_trace(
        go.Scatter(
            y=bt_result.benchmark_curve,
            mode="lines",
            name="Equal Weight",
            line=dict(color="gray", width=1, dash="dash"),
        )
    )
    fig_eq.update_layout(
        title="Equity Curve",
        yaxis_title="Portfolio Value",
        height=400,
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Rolling Sharpe + Drawdown
    col1, col2 = st.columns(2)

    with col1:
        rs = rolling_sharpe(bt_result.portfolio_returns, window=63, rf_daily=rf_rate / 252)
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(y=rs, mode="lines", name="Rolling Sharpe (63d)"))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rs.update_layout(title="Rolling Sharpe Ratio", height=300)
        st.plotly_chart(fig_rs, use_container_width=True)

    with col2:
        dd = drawdown_series(bt_result.equity_curve)
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                y=dd,
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="red"),
            )
        )
        fig_dd.update_layout(
            title="Drawdown",
            yaxis_tickformat=".1%",
            height=300,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Section 5: Portfolio Metrics (out-of-sample) ─────────────────────

    st.header("5. Portfolio Metrics (Out-of-Sample)")

    report = bt_result.metrics.full_report()

    cols = st.columns(6)
    metric_items = [
        ("Return", report["annualized_return"], "{:.2%}"),
        ("Volatility", report["annualized_volatility"], "{:.2%}"),
        ("Sharpe", report["sharpe_ratio"], "{:.2f}"),
        ("Max DD", report["max_drawdown"], "{:.2%}"),
        ("VaR 95%", report["var_95"], "{:.2%}"),
        ("CVaR 95%", report["cvar_95"], "{:.2%}"),
    ]
    for col, (label, val, fmt) in zip(cols, metric_items):
        col.metric(label, fmt.format(val))

    # ── Section 6: Comparison Table (out-of-sample) ──────────────────────

    st.header("6. Comparison (Out-of-Sample)")

    # Best single asset over the backtest period
    bt_start = bt_result.rebalance_dates[0]
    bt_returns = data.returns.values[bt_start:]

    best_asset_idx = -1
    best_asset_sharpe = -np.inf
    for i in range(data.n_assets):
        asset_pm = PortfolioMetrics(bt_returns[:, i], rf_rate)
        if asset_pm.sharpe_ratio > best_asset_sharpe:
            best_asset_sharpe = asset_pm.sharpe_ratio
            best_asset_idx = i

    best_asset_pm = PortfolioMetrics(bt_returns[:, best_asset_idx], rf_rate)

    comparison = pd.DataFrame(
        {
            f"Optimized ({method})": bt_result.metrics.full_report(),
            "Equal Weight": bt_result.benchmark_metrics.full_report(),
            f"Best Asset ({tickers[best_asset_idx]})": best_asset_pm.full_report(),
        }
    ).T

    format_dict = {
        "annualized_return": "{:.2%}",
        "annualized_volatility": "{:.2%}",
        "sharpe_ratio": "{:.2f}",
        "sortino_ratio": "{:.2f}",
        "max_drawdown": "{:.2%}",
        "calmar_ratio": "{:.2f}",
        "var_95": "{:.2%}",
        "cvar_95": "{:.2%}",
    }
    st.dataframe(
        comparison.style.format(format_dict),
        use_container_width=True,
    )
