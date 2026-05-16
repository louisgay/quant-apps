"""Streamlit dashboard for the HRP allocation engine."""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
import streamlit as st

from engine import (
    fetch_market_data,
    estimate_covariance,
    correlation_distance,
    hierarchical_cluster,
    quasi_diagonalize,
    hrp_allocation,
    mean_variance_optimize,
    risk_parity_optimize,
    equal_weight,
    inverse_volatility,
    Backtester,
    compute_metrics,
    effective_number_of_bets,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

st.set_page_config(page_title="HRP Allocation", page_icon="🌳", layout="wide")

st.title("Hierarchical Risk Parity")
st.markdown(
    "**López de Prado (2016)** — ML-based portfolio allocation without matrix inversion"
)

# Sidebar
with st.sidebar:
    st.header("Parameters")

    st.subheader("Universe")
    preset = st.selectbox(
        "Asset Preset",
        ["Multi-Asset (7)", "US Sectors (11)", "Global Equities (10)", "Custom"],
    )

    presets = {
        "Multi-Asset (7)": ["SPY", "QQQ", "EFA", "TLT", "GLD", "VNQ", "HYG"],
        "US Sectors (11)": [
            "XLK", "XLF", "XLV", "XLE", "XLI",
            "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        ],
        "Global Equities (10)": [
            "SPY", "EFA", "EEM", "VGK", "EWJ",
            "FXI", "EWZ", "EWA", "EWC", "EWG",
        ],
    }

    if preset == "Custom":
        tickers_input = st.text_input(
            "Tickers (comma-separated)", "SPY, TLT, GLD, QQQ, EFA"
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
    else:
        tickers = presets[preset]

    st.subheader("Estimation")
    start_date = st.date_input("Start Date", value=pd.Timestamp("2018-01-01"))
    cov_method = st.selectbox("Covariance Method", ["sample", "ledoit_wolf"])
    linkage_method = st.selectbox(
        "Linkage Method", ["ward", "average", "complete", "single"]
    )

    st.subheader("Backtest")
    window = st.slider("Estimation Window (days)", 126, 504, 252)
    rebalance_freq = st.slider("Rebalance Frequency (days)", 5, 63, 21)

    run_button = st.button("Run Analysis", type="primary")

if run_button:
    with st.spinner("Fetching market data..."):
        returns = fetch_market_data(tickers, start=str(start_date))

    if returns.empty:
        st.error("No data returned. Check tickers and date range.")
        st.stop()

    st.success(
        f"Loaded {len(returns)} days of returns for {len(returns.columns)} assets"
    )

    cov, corr = estimate_covariance(returns, method=cov_method)

    # Step 1: Clustering
    st.header("Step 1: Hierarchical Clustering")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Matrix (Original)")
        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="equal",
            title="Pairwise Correlations",
        )
        st.plotly_chart(fig_corr, width="stretch")

    with col2:
        st.subheader("Distance Matrix")
        dist = correlation_distance(corr)
        dist_df = pd.DataFrame(dist, index=corr.index, columns=corr.columns)
        fig_dist = px.imshow(
            dist_df,
            color_continuous_scale="Viridis",
            zmin=0, zmax=1,
            aspect="equal",
            title="d(i,j) = √((1-ρ)/2)",
        )
        st.plotly_chart(fig_dist, width="stretch")

    # Dendrogram
    st.subheader("Dendrogram")
    Z = hierarchical_cluster(dist, method=linkage_method)
    dendro_data = scipy_dendrogram(Z, labels=corr.index.tolist(), no_plot=True)

    fig_dendro = go.Figure()
    for i, (xs, ys) in enumerate(zip(dendro_data["icoord"], dendro_data["dcoord"])):
        fig_dendro.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="#1f77b4", width=1.5),
                showlegend=False,
            )
        )

    fig_dendro.update_layout(
        title=f"Hierarchical Clustering ({linkage_method} linkage)",
        xaxis=dict(
            tickmode="array",
            tickvals=[5 + 10 * i for i in range(len(dendro_data["ivl"]))],
            ticktext=dendro_data["ivl"],
            tickangle=45,
        ),
        yaxis_title="Distance",
        height=400,
    )
    st.plotly_chart(fig_dendro, width="stretch")

    # Step 2: Quasi-Diagonalization
    st.header("Step 2: Quasi-Diagonalization")

    cov_reordered, sort_order = quasi_diagonalize(cov, Z)
    reordered_labels = [corr.index[i] for i in sort_order]

    cov_reordered_df = pd.DataFrame(
        cov_reordered, index=reordered_labels, columns=reordered_labels
    )

    fig_quasi = px.imshow(
        cov_reordered_df,
        color_continuous_scale="Viridis",
        aspect="equal",
        title="Seriated Covariance — Block-Diagonal Structure",
    )
    st.plotly_chart(fig_quasi, width="stretch")

    st.markdown(
        """
        Correlated assets are now adjacent — clusters appear as blocks along the
        diagonal. This ordering drives the recursive bisection below.
        """
    )

    # Step 3: Allocation
    st.header("Step 3: Allocation Results")

    w_hrp = hrp_allocation(cov, corr, linkage_method=linkage_method)
    w_ew = equal_weight(cov)
    w_iv = inverse_volatility(cov)
    w_rp = risk_parity_optimize(cov)
    w_mv = mean_variance_optimize(cov)

    weights_all = pd.DataFrame({
        "HRP": w_hrp,
        "Equal Weight": w_ew,
        "Inverse Vol": w_iv,
        "Risk Parity": w_rp,
        "Min Variance": w_mv,
    })

    fig_weights = px.bar(
        weights_all.reset_index().melt(id_vars="index"),
        x="index", y="value", color="variable",
        barmode="group",
        title="Portfolio Weights Comparison",
        labels={"index": "Asset", "value": "Weight", "variable": "Method"},
    )
    st.plotly_chart(fig_weights, width="stretch")

    # Effective number of bets
    st.subheader("Diversification: Effective Number of Bets")
    n_eff = {
        method: effective_number_of_bets(weights_all[method].values)
        for method in weights_all.columns
    }
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, (method, n) in zip(
        [col1, col2, col3, col4, col5], n_eff.items()
    ):
        col.metric(method, f"{n:.1f} / {len(tickers)}")

    # Backtest
    st.header("Rolling Backtest")

    methods = {
        "HRP": lambda cov: hrp_allocation(cov, linkage_method=linkage_method),
        "Equal Weight": equal_weight,
        "Inverse Vol": inverse_volatility,
        "Risk Parity": risk_parity_optimize,
        "Min Variance": mean_variance_optimize,
    }

    results = {}
    for name, fn in methods.items():
        bt = Backtester(
            returns, fn, window=window, rebalance_freq=rebalance_freq,
            cov_method=cov_method,
        )
        port_ret, weights_hist = bt.run()
        metrics = compute_metrics(port_ret, weights_hist)
        results[name] = {
            "returns": port_ret,
            "weights": weights_hist,
            "metrics": metrics,
        }

    # Equity curves
    fig_equity = go.Figure()
    for name, res in results.items():
        cum_ret = (1 + res["returns"]).cumprod()
        fig_equity.add_trace(
            go.Scatter(x=cum_ret.index, y=cum_ret.values, name=name)
        )
    fig_equity.update_layout(
        title="Cumulative Returns (Rolling Backtest)",
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        hovermode="x unified",
    )
    st.plotly_chart(fig_equity, width="stretch")

    # Metrics table
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame(
        {name: res["metrics"].to_dict() for name, res in results.items()}
    )
    st.dataframe(metrics_df, width="stretch")

    # Weight stability
    st.subheader("Weight Stability Over Time (HRP vs Min Variance)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**HRP Weights**")
        fig_w_hrp = px.area(results["HRP"]["weights"], title="HRP — Stable")
        fig_w_hrp.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_w_hrp, width="stretch")

    with col2:
        st.markdown("**Min Variance Weights**")
        fig_w_mv = px.area(
            results["Min Variance"]["weights"], title="Min Variance — Unstable"
        )
        fig_w_mv.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_w_mv, width="stretch")

    # Condition number
    st.subheader("Condition Number κ(Σ)")
    kappa = np.linalg.cond(cov.values)
    st.metric(
        "κ(Σ)",
        f"{kappa:.1f}",
        delta=f"{'Well-conditioned' if kappa < 100 else 'ILL-CONDITIONED'}",
    )
    st.markdown(
        f"""
        κ = {kappa:.1f} means Markowitz amplifies estimation errors by ~{kappa:.0f}x.
        {"Manageable here, but still a problem at scale." if kappa < 100
         else "Highly ill-conditioned — small errors in Σ produce wild weight swings."}
        HRP sidesteps this entirely.
        """
    )

else:
    st.markdown(
        """
        ## Overview

        HRP uses unsupervised ML (hierarchical clustering) to build portfolios
        without inverting the covariance matrix:

        1. **Clustering** — groups correlated assets
        2. **Seriation** — reorders Σ to reveal block structure
        3. **Recursive bisection** — allocates capital top-down through the tree

        Configure parameters in the sidebar and click **Run Analysis**.
        """
    )
