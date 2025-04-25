"""Streamlit UI for the PDE Option Pricer."""

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import (
    GridConfig,
    bs_price,
    bs_delta,
    bs_gamma,
    bs_theta,
    bs_vega,
    compute_price_surface,
    price_european_american,
    price_barrier_local_vol,
    extract_free_boundary,
    price_american_dividends_psor,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDE Option Pricer",
    page_icon="📐",
    layout="wide",
)

st.title("📐 PDE Option Pricer")
st.caption("Finite-difference methods for vanilla, American, barrier, and dividend options")

# ---------------------------------------------------------------------------
# Sidebar — shared parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Option Parameters")
    option_type = st.selectbox("Type", ["call", "put"])
    K = st.number_input("Strike (K)", value=100.0, step=5.0, min_value=1.0)
    T = st.number_input("Maturity (T, years)", value=1.0, step=0.25, min_value=0.05)
    r = st.number_input("Risk-free rate (r)", value=0.05, step=0.01, format="%.3f")
    sigma = st.number_input("Volatility (σ)", value=0.20, step=0.01, format="%.3f")

    st.header("Grid Parameters")
    N = st.number_input("Spatial points (N)", value=500, step=50, min_value=50)
    M = st.number_input("Time steps (M)", value=500, step=50, min_value=50)
    grid = GridConfig(N=int(N), M=int(M))

    st.header("Barrier Parameters")
    sigma_atm = st.number_input("σ_atm", value=0.20, step=0.01, format="%.3f")
    alpha_skew = st.number_input("Skew α", value=0.40, step=0.05, format="%.2f")
    B = st.number_input("Barrier (B)", value=130.0, step=5.0, min_value=1.0)
    if B <= K:
        st.warning(f"Barrier must be > Strike ({K})")

    st.header("Dividend Parameters")
    div_amount = st.number_input("Dividend amount", value=5.0, step=1.0, min_value=0.0)
    div_time = st.number_input("Dividend time", value=0.48, step=0.05, min_value=0.01, max_value=5.0)
    if div_time >= T:
        st.warning(f"Dividend time must be < Maturity ({T})")
    omega = st.number_input("Relaxation ω", value=1.5, step=0.1, min_value=1.0, max_value=1.99)

    run = st.button("▶  Run All", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "European / American",
    "Barrier + Local Vol",
    "Free Boundary",
    "PSOR + Dividends",
])

if not run:
    st.info("Configure parameters in the sidebar and click **Run All**.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — European / American
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        eu_res = price_european_american(K, T, r, sigma, option_type, "european", grid)
        bs_ref = bs_price(K, K, T, r, sigma, option_type)
        st.metric("European (PDE)", f"{eu_res.price:.4f}", f"BS: {float(bs_ref):.4f}")
        st.caption(f"Elapsed: {eu_res.elapsed:.3f}s")

    with col2:
        am_res = price_european_american(K, T, r, sigma, option_type, "american", grid)
        early_ex = am_res.price - eu_res.price
        st.metric("American (PDE)", f"{am_res.price:.4f}",
                  f"Early-exercise premium: {early_ex:.4f}")
        st.caption(f"Elapsed: {am_res.elapsed:.3f}s")

    # Greeks
    st.subheader("Greeks at S = K")
    g_cols = st.columns(4)
    g_cols[0].metric("Delta", f"{float(bs_delta(K, K, T, r, sigma, option_type)):.4f}")
    g_cols[1].metric("Gamma", f"{float(bs_gamma(K, K, T, r, sigma)):.4f}")
    g_cols[2].metric("Theta", f"{float(bs_theta(K, K, T, r, sigma, option_type)):.4f}")
    g_cols[3].metric("Vega", f"{float(bs_vega(K, K, T, r, sigma)):.4f}")

    # V(S) chart
    S_g = eu_res.S_grid
    mask = (S_g > K * 0.5) & (S_g < K * 1.5)

    if option_type == "call":
        payoff = np.maximum(S_g - K, 0)
    else:
        payoff = np.maximum(K - S_g, 0)
    bs_curve = bs_price(S_g, K, T, r, sigma, option_type)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=S_g[mask], y=eu_res.V_grid[mask],
                              name="European PDE", line=dict(width=2)))
    fig1.add_trace(go.Scatter(x=S_g[mask], y=am_res.V_grid[mask],
                              name="American PDE", line=dict(width=2, dash="dot")))
    fig1.add_trace(go.Scatter(x=S_g[mask], y=bs_curve[mask],
                              name="BS Analytical", line=dict(width=1, dash="dash")))
    fig1.add_trace(go.Scatter(x=S_g[mask], y=payoff[mask],
                              name="Payoff", line=dict(width=1, color="grey")))
    fig1.update_layout(title="Option Value V(S) at t=0",
                       xaxis_title="Spot (S)", yaxis_title="V",
                       template="plotly_white", height=450)
    st.plotly_chart(fig1, use_container_width=True)

    # 3D surface
    st.subheader("Price Surface")
    surf = compute_price_surface(K, r, sigma, option_type,
                                 S_min=K * 0.5, S_max=K * 1.5)
    fig_surf = go.Figure(data=[go.Surface(
        x=surf.T_values, y=surf.S_values, z=surf.V_surface.T,
        colorscale="Viridis",
    )])
    fig_surf.update_layout(
        scene=dict(xaxis_title="T", yaxis_title="S", zaxis_title="V"),
        title=f"BS {option_type.title()} Price Surface",
        template="plotly_white", height=500,
    )
    st.plotly_chart(fig_surf, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Barrier + Local Vol
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    bar_res = price_barrier_local_vol(K, T, r, sigma_atm, alpha_skew, B, grid)
    van_res = price_european_american(K, T, r, sigma_atm, "call", "european", grid)

    c1, c2 = st.columns(2)
    c1.metric("Barrier Price", f"{bar_res.price:.4f}")
    c2.metric("Vanilla Call", f"{van_res.price:.4f}",
              f"Discount: {van_res.price - bar_res.price:.4f}")
    st.caption(f"Elapsed: {bar_res.elapsed:.3f}s")

    # V(S) with barrier line
    S_b = bar_res.S_grid
    mask_b = S_b < B + 5

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=S_b[mask_b], y=bar_res.V_grid[mask_b],
                              name="Barrier Call", line=dict(width=2)))
    fig2.add_vrect(x0=B, x1=S_b[mask_b][-1], fillcolor="red",
                   opacity=0.08, line_width=0)
    fig2.add_vline(x=B, line_dash="dash", line_color="red",
                   annotation_text=f"B={B}")
    fig2.add_vline(x=K, line_dash="dash", line_color="grey",
                   annotation_text=f"K={K}")
    fig2.update_layout(title="Up-and-Out Call with Local Vol",
                       xaxis_title="S", yaxis_title="V",
                       template="plotly_white", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Local vol curve
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=S_b[mask_b], y=bar_res.vol_grid[mask_b],
                                 name="σ(S)", line=dict(width=2)))
    fig_vol.add_vline(x=K, line_dash="dash", line_color="grey",
                      annotation_text=f"K={K}")
    fig_vol.update_layout(title="Local Volatility σ(S) = σ_atm · (S/K)^{-α}",
                          xaxis_title="S", yaxis_title="σ",
                          template="plotly_white", height=350)
    st.plotly_chart(fig_vol, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Free Boundary
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    fb_grid = GridConfig(N=int(N), M=int(M))
    fb = extract_free_boundary(K, T, r, sigma, option_type, fb_grid)
    st.caption(f"Elapsed: {fb.elapsed:.3f}s")

    col_a, col_b = st.columns(2)

    with col_a:
        fig3a = go.Figure()
        fig3a.add_trace(go.Scatter(x=fb.time_grid, y=fb.boundary,
                                   name="S*(t)", line=dict(width=2)))
        fig3a.add_hline(y=K, line_dash="dash", line_color="grey",
                        annotation_text=f"K={K}")
        fig3a.update_layout(title="Optimal Exercise Boundary S*(t)",
                            xaxis_title="Time (t)", yaxis_title="S*",
                            template="plotly_white", height=400)
        st.plotly_chart(fig3a, use_container_width=True)

    with col_b:
        S_fb = fb.S_grid
        mask_fb = (S_fb > K * 0.5) & (S_fb < K * 1.5)
        fig3b = go.Figure()
        fig3b.add_trace(go.Scatter(x=S_fb[mask_fb], y=fb.V_grid[mask_fb],
                                   name="American V(S)", line=dict(width=2)))
        fig3b.add_trace(go.Scatter(x=S_fb[mask_fb], y=fb.exercise_value[mask_fb],
                                   name="Exercise Value",
                                   line=dict(width=1, dash="dash", color="grey")))
        fig3b.update_layout(title="Option Value vs Exercise at t=0",
                            xaxis_title="S", yaxis_title="V",
                            template="plotly_white", height=400)
        st.plotly_chart(fig3b, use_container_width=True)

    st.metric("Boundary at t=0", f"{fb.boundary[0]:.2f}")
    st.metric("Boundary at t=T", f"{fb.boundary[-1]:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — PSOR + Dividends
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    psor_grid = GridConfig(N=int(min(N, 150)), M=int(min(M, 250)))
    psor = price_american_dividends_psor(
        K, T, r, sigma, option_type, div_amount, div_time, omega,
        grid=psor_grid,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("PSOR Price", f"{psor.price:.4f}")
    c2.metric("Total Iterations", f"{psor.iterations:,}")
    c3.metric("Elapsed", f"{psor.elapsed:.2f}s")

    col_l, col_r = st.columns(2)

    with col_l:
        S_p = psor.S_grid
        mask_p = (S_p > K * 0.3) & (S_p < K * 2.0)
        if option_type == "call":
            payoff_p = np.maximum(S_p - K, 0)
        else:
            payoff_p = np.maximum(K - S_p, 0)

        fig4a = go.Figure()
        fig4a.add_trace(go.Scatter(x=S_p[mask_p], y=psor.V_grid[mask_p],
                                   name="PSOR V(S)", line=dict(width=2)))
        fig4a.add_trace(go.Scatter(x=S_p[mask_p], y=payoff_p[mask_p],
                                   name="Payoff",
                                   line=dict(width=1, dash="dash", color="grey")))
        fig4a.update_layout(title="American Value with Dividend",
                            xaxis_title="S", yaxis_title="V",
                            template="plotly_white", height=400)
        st.plotly_chart(fig4a, use_container_width=True)

    with col_r:
        fig4b = go.Figure()
        fig4b.add_trace(go.Scatter(x=psor.time_grid, y=psor.free_boundary,
                                   name="S*(t)", line=dict(width=2)))
        fig4b.add_vline(x=div_time, line_dash="dash", line_color="red",
                        annotation_text=f"Ex-div t={div_time}")
        fig4b.add_hline(y=K, line_dash="dash", line_color="grey",
                        annotation_text=f"K={K}")
        fig4b.update_layout(title="Free Boundary with Dividend",
                            xaxis_title="Time (t)", yaxis_title="S*",
                            template="plotly_white", height=400)
        st.plotly_chart(fig4b, use_container_width=True)
