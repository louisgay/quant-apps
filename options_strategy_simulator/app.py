"""Options Strategy Simulator — Streamlit UI.

Build multi-leg option strategies on a synthetic underlying with configurable
volatility skew (SVI parameterization behind 3 intuitive sliders).
Monte Carlo simulation with Student-t innovations for P&L probability analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from engine import (
    OptionLeg, Strategy, VolSmile,
    compute_portfolio_greeks,
    compute_payoff_diagram, compute_pnl_surface,
    compute_greeks_over_spot, compute_greeks_over_time,
    simulate_strategy,
)

st.set_page_config(page_title="Options Strategy Simulator", layout="wide")
st.title("Options Strategy Simulator")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "legs" not in st.session_state:
    st.session_state.legs = []

# ---------------------------------------------------------------------------
# Sidebar — Underlying + Vol Smile
# ---------------------------------------------------------------------------
st.sidebar.header("Underlying")
S = st.sidebar.number_input("Spot Price", value=100.0, min_value=1.0, step=1.0)
r = st.sidebar.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=0.50, step=0.005, format="%.3f")
q = st.sidebar.number_input("Dividend Yield", value=0.00, min_value=0.0, max_value=0.20, step=0.005, format="%.3f")

st.sidebar.header("Volatility Smile")
atm_vol = st.sidebar.slider("ATM Volatility", 0.05, 1.00, 0.20, 0.01)

enable_skew = st.sidebar.checkbox("Enable Volatility Skew")
skew = 0.0
curvature = 0.0
use_raw_svi = False
raw_params = {}

if enable_skew:
    skew = st.sidebar.slider("Skew Strength", 0.0, 0.99, 0.30, 0.01)
    curvature = st.sidebar.slider("Smile Curvature", 0.0, 2.0, 0.50, 0.05)
    use_raw_svi = st.sidebar.checkbox("Advanced: Raw SVI Parameters")
    if use_raw_svi:
        raw_params["a"] = st.sidebar.number_input("a", value=0.04, step=0.01, format="%.4f")
        raw_params["b"] = st.sidebar.number_input("b", value=0.25, min_value=0.001, step=0.01, format="%.4f")
        raw_params["rho"] = st.sidebar.number_input("rho", value=-0.30, min_value=-0.999, max_value=0.999, step=0.01, format="%.3f")
        raw_params["m"] = st.sidebar.number_input("m", value=0.0, step=0.01, format="%.4f")
        raw_params["sigma_svi"] = st.sidebar.number_input("sigma_svi", value=0.10, min_value=0.001, step=0.01, format="%.4f")

# Monte Carlo settings in sidebar
st.sidebar.header("Monte Carlo")
n_paths = st.sidebar.select_slider(
    "Simulation Paths",
    options=[10_000, 25_000, 50_000, 100_000, 250_000],
    value=50_000,
)
mc_df = st.sidebar.slider(
    "Tail Fatness (Student-t df)",
    min_value=3.0, max_value=30.0, value=5.0, step=0.5,
    help="Lower = fatter tails (more extreme moves). 30 ≈ normal distribution, 3-5 = heavy tails typical of equity returns.",
)

# Build the smile
if use_raw_svi:
    smile = VolSmile.from_raw_svi(**raw_params)
elif enable_skew:
    smile = VolSmile.from_simple(atm_vol, skew, curvature)
else:
    smile = VolSmile.flat(atm_vol)

# Smile preview chart
st.sidebar.subheader("Smile Preview")
K_preview = np.linspace(S * 0.7, S * 1.3, 100)
iv_preview = smile.get_iv_for_strike(K_preview, S, 0.25) * 100  # as percentage
moneyness = K_preview / S

fig_smile = go.Figure()
fig_smile.add_trace(go.Scatter(
    x=moneyness, y=iv_preview,
    mode="lines", line=dict(color="#636EFA", width=2),
))
fig_smile.update_layout(
    xaxis_title="K / S",
    yaxis_title="IV (%)",
    height=220,
    margin=dict(l=40, r=10, t=10, b=40),
)
st.sidebar.plotly_chart(fig_smile, use_container_width=True)

# ---------------------------------------------------------------------------
# Main — Strategy Builder
# ---------------------------------------------------------------------------
st.header("Strategy Builder")

# Editable leg table
if st.session_state.legs:
    legs_to_delete = []
    new_legs = []

    for i, leg in enumerate(st.session_state.legs):
        cols = st.columns([1.5, 1.5, 2, 2, 1.5, 0.8])
        with cols[0]:
            opt_type = st.selectbox(
                "Type", ["call", "put"], index=0 if leg.option_type == "call" else 1,
                key=f"type_{i}",
            )
        with cols[1]:
            direction = st.selectbox(
                "Direction", ["long", "short"], index=0 if leg.direction == "long" else 1,
                key=f"dir_{i}",
            )
        with cols[2]:
            strike = st.number_input(
                "Strike", value=leg.strike, min_value=0.01, step=1.0,
                key=f"strike_{i}",
            )
        with cols[3]:
            T_leg = st.number_input(
                "Expiry (yrs)", value=leg.T, min_value=0.01, max_value=5.0, step=0.01,
                key=f"T_{i}", format="%.3f",
            )
        with cols[4]:
            qty = st.number_input(
                "Qty", value=leg.quantity, min_value=1, step=1,
                key=f"qty_{i}",
            )
        with cols[5]:
            st.write("")  # spacing
            if st.button("X", key=f"del_{i}", use_container_width=True):
                legs_to_delete.append(i)

        new_legs.append(OptionLeg(opt_type, direction, strike, T_leg, qty))

    # Apply deletes
    if legs_to_delete:
        st.session_state.legs = [l for j, l in enumerate(new_legs) if j not in legs_to_delete]
        st.rerun()
    else:
        st.session_state.legs = new_legs

# Add leg button
if st.button("+ Add Leg"):
    st.session_state.legs.append(OptionLeg("call", "long", S, 0.25))
    st.rerun()

# ---------------------------------------------------------------------------
# Main — Results
# ---------------------------------------------------------------------------
if not st.session_state.legs:
    st.info("Add legs to build a strategy.")
    st.stop()

strategy = Strategy(legs=st.session_state.legs)

# Compute analytical results
payoff_result = compute_payoff_diagram(strategy, S, r, q, smile, n_points=300)
pnl_surface = compute_pnl_surface(strategy, S, r, q, smile, n_S=300, n_t=6)
greeks_spot = compute_greeks_over_spot(strategy, S, r, q, smile, n_points=200)
greeks_time = compute_greeks_over_time(strategy, S, r, q, smile, n_points=200)
portfolio_greeks = compute_portfolio_greeks(strategy, S, r, q, smile)

# Monte Carlo simulation
sim = simulate_strategy(strategy, S, r, q, smile, n_paths=n_paths, df=mc_df)

# Summary metrics
st.header("Results")
mcols = st.columns(6)
mcols[0].metric("Entry Cost", f"${payoff_result.entry_cost:.2f}")
mcols[1].metric("Max Profit", f"${payoff_result.max_profit:.2f}")
mcols[2].metric("Max Loss", f"${payoff_result.max_loss:.2f}")
be_str = ", ".join(f"{b:.1f}" for b in payoff_result.breakeven_points) if len(payoff_result.breakeven_points) > 0 else "None"
mcols[3].metric("Breakevens", be_str)
mcols[4].metric("Delta", f"{portfolio_greeks.delta:.4f}")
mcols[5].metric("Theta/Day", f"{portfolio_greeks.theta_daily:.4f}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Payoff at Expiry", "P&L Over Time",
    "Greeks vs Spot", "Greeks vs Time",
    "Monte Carlo",
])

colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#FF6692"]

# -- Tab 1: Payoff at Expiry --
with tab1:
    fig = go.Figure()
    pnl = payoff_result.pnl_at_expiry
    S_range = payoff_result.S_range

    # Green/red fill
    fig.add_trace(go.Scatter(
        x=S_range, y=np.where(pnl >= 0, pnl, 0),
        fill="tozeroy", fillcolor="rgba(0, 200, 80, 0.3)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=S_range, y=np.where(pnl <= 0, pnl, 0),
        fill="tozeroy", fillcolor="rgba(255, 60, 60, 0.3)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # P&L line
    fig.add_trace(go.Scatter(
        x=S_range, y=pnl,
        mode="lines", line=dict(color="white", width=2),
        name="P&L at Expiry",
    ))
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    # Spot marker
    fig.add_vline(x=S, line_dash="dot", line_color="#636EFA", opacity=0.6,
                  annotation_text=f"Spot={S:.0f}")
    # Breakeven markers
    for be in payoff_result.breakeven_points:
        fig.add_vline(x=be, line_dash="dash", line_color="orange", opacity=0.5)
        fig.add_annotation(x=be, y=0, text=f"BE={be:.1f}", showarrow=True,
                          arrowhead=2, arrowcolor="orange", font=dict(size=10))

    fig.update_layout(
        xaxis_title="Spot Price at Expiry",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# -- Tab 2: P&L Over Time --
with tab2:
    fig2 = go.Figure()
    max_T = max(leg.T for leg in strategy.legs)

    for i, t_elapsed in enumerate(pnl_surface.t_range):
        dte = max((max_T - t_elapsed) * 365, 0)
        label = f"{dte:.0f} DTE" if dte > 0.5 else "Expiry"
        fig2.add_trace(go.Scatter(
            x=pnl_surface.S_range, y=pnl_surface.pnl_grid[i, :],
            mode="lines", name=label,
            line=dict(color=colors[i % len(colors)]),
        ))

    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig2.add_vline(x=S, line_dash="dot", line_color="#636EFA", opacity=0.4)
    fig2.update_layout(
        xaxis_title="Spot Price",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        height=450,
    )
    st.plotly_chart(fig2, use_container_width=True)

# -- Tab 3: Greeks vs Spot --
with tab3:
    greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    greek_data = [greeks_spot.delta, greeks_spot.gamma, greeks_spot.theta,
                  greeks_spot.vega, greeks_spot.rho]

    fig3 = make_subplots(rows=3, cols=2, subplot_titles=greek_names,
                         vertical_spacing=0.08, horizontal_spacing=0.08)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]

    for idx, (name, data) in enumerate(zip(greek_names, greek_data)):
        row, col = positions[idx]
        fig3.add_trace(go.Scatter(
            x=greeks_spot.x_values, y=data,
            mode="lines", line=dict(color=colors[idx]),
            name=name, showlegend=False,
        ), row=row, col=col)
        fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3,
                       row=row, col=col)

    fig3.update_layout(template="plotly_dark", height=700)
    st.plotly_chart(fig3, use_container_width=True)

# -- Tab 4: Greeks vs Time --
with tab4:
    greek_data_t = [greeks_time.delta, greeks_time.gamma, greeks_time.theta,
                    greeks_time.vega, greeks_time.rho]
    # Convert time elapsed to DTE
    max_T_all = max(leg.T for leg in strategy.legs)
    dte_values = np.maximum((max_T_all - greeks_time.x_values) * 365, 0)

    fig4 = make_subplots(rows=3, cols=2, subplot_titles=greek_names,
                         vertical_spacing=0.08, horizontal_spacing=0.08)

    for idx, (name, data) in enumerate(zip(greek_names, greek_data_t)):
        row, col = positions[idx]
        fig4.add_trace(go.Scatter(
            x=dte_values, y=data,
            mode="lines", line=dict(color=colors[idx]),
            name=name, showlegend=False,
        ), row=row, col=col)
        fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3,
                       row=row, col=col)

    fig4.update_xaxes(autorange="reversed")  # DTE decreases left to right
    fig4.update_layout(template="plotly_dark", height=700)
    st.plotly_chart(fig4, use_container_width=True)

# -- Tab 5: Monte Carlo --
with tab5:
    # Summary metrics row
    mc_cols = st.columns(4)
    mc_cols[0].metric("Prob. of Profit", f"{sim.prob_profit:.1%}")
    mc_cols[1].metric("Expected P&L", f"${sim.expected_pnl:.2f}")
    mc_cols[2].metric("VaR 95%", f"${sim.var_95:.2f}")
    mc_cols[3].metric("CVaR 95%", f"${sim.cvar_95:.2f}")

    # Two-column layout: histogram + percentile table
    col_hist, col_stats = st.columns([2, 1])

    with col_hist:
        # P&L histogram with profit/loss coloring
        bins = np.linspace(
            np.percentile(sim.pnl_samples, 1),
            np.percentile(sim.pnl_samples, 99),
            80,
        )
        profit_samples = sim.pnl_samples[sim.pnl_samples >= 0]
        loss_samples = sim.pnl_samples[sim.pnl_samples < 0]

        fig_mc = go.Figure()
        if len(loss_samples) > 0:
            fig_mc.add_trace(go.Histogram(
                x=loss_samples, xbins=dict(start=bins[0], end=0, size=(bins[-1] - bins[0]) / 80),
                marker_color="rgba(255, 60, 60, 0.7)",
                name="Loss",
            ))
        if len(profit_samples) > 0:
            fig_mc.add_trace(go.Histogram(
                x=profit_samples, xbins=dict(start=0, end=bins[-1], size=(bins[-1] - bins[0]) / 80),
                marker_color="rgba(0, 200, 80, 0.7)",
                name="Profit",
            ))

        # Vertical lines for key percentiles
        fig_mc.add_vline(x=0, line_dash="solid", line_color="white", opacity=0.5)
        fig_mc.add_vline(x=sim.expected_pnl, line_dash="dash", line_color="#636EFA",
                         annotation_text=f"E[P&L]={sim.expected_pnl:.1f}", annotation_position="top")
        fig_mc.add_vline(x=sim.percentiles[5], line_dash="dot", line_color="#EF553B",
                         annotation_text=f"5th={sim.percentiles[5]:.1f}", annotation_position="bottom left")
        fig_mc.add_vline(x=sim.percentiles[95], line_dash="dot", line_color="#00CC96",
                         annotation_text=f"95th={sim.percentiles[95]:.1f}", annotation_position="bottom right")

        fig_mc.update_layout(
            xaxis_title="P&L ($)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400,
            barmode="stack",
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_stats:
        st.markdown("**P&L Distribution**")
        stats_data = {
            "Metric": [
                "Prob. of Profit",
                "Expected P&L",
                "Median P&L",
                "",
                "5th percentile",
                "10th percentile",
                "25th percentile",
                "75th percentile",
                "90th percentile",
                "95th percentile",
                "",
                "VaR 95%",
                "VaR 99%",
                "CVaR 95%",
                "CVaR 99%",
            ],
            "Value": [
                f"{sim.prob_profit:.1%}",
                f"${sim.expected_pnl:.2f}",
                f"${sim.median_pnl:.2f}",
                "",
                f"${sim.percentiles[5]:.2f}",
                f"${sim.percentiles[10]:.2f}",
                f"${sim.percentiles[25]:.2f}",
                f"${sim.percentiles[75]:.2f}",
                f"${sim.percentiles[90]:.2f}",
                f"${sim.percentiles[95]:.2f}",
                "",
                f"${sim.var_95:.2f}",
                f"${sim.var_99:.2f}",
                f"${sim.cvar_95:.2f}",
                f"${sim.cvar_99:.2f}",
            ],
        }
        st.dataframe(stats_data, hide_index=True, use_container_width=True)

        st.caption(
            f"Student-t df={mc_df:.1f} | {n_paths:,} paths | "
            f"ATM vol={atm_vol:.0%} | T={max(leg.T for leg in strategy.legs):.3f}y"
        )

    # Simulated spot distribution
    st.markdown("**Simulated Spot Distribution at Expiry**")
    fig_spot = go.Figure()
    fig_spot.add_trace(go.Histogram(
        x=sim.spot_samples,
        nbinsx=100,
        marker_color="rgba(99, 110, 250, 0.6)",
        name="Spot at Expiry",
    ))
    fig_spot.add_vline(x=S, line_dash="dash", line_color="white", opacity=0.7,
                       annotation_text=f"Current={S:.0f}")
    fig_spot.update_layout(
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=300,
    )
    st.plotly_chart(fig_spot, use_container_width=True)
