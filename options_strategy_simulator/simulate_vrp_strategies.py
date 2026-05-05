"""VRP Strategy Simulation — Short Straddle vs Iron Butterfly comparison.

Runs Monte Carlo simulations with Student-t innovations across multiple
scenarios: varying maturities, wing widths, skew, and volatility levels.

Usage:
    streamlit run simulate_vrp_strategies.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.strategy import Strategy, OptionLeg
from engine.vol_surface import VolSmile
from engine.monte_carlo import simulate_strategy
from engine.analytics import compute_payoff_diagram, compute_pnl_surface
from engine.greeks import compute_portfolio_greeks

# ---------------------------------------------------------------------------
# Strategy builders (short variants — the backtester sells vol)
# ---------------------------------------------------------------------------

def short_straddle(strike: float, T: float) -> Strategy:
    """Short ATM straddle: sell call + sell put."""
    return Strategy(
        legs=[
            OptionLeg("call", "short", strike, T),
            OptionLeg("put", "short", strike, T),
        ],
        name="Short Straddle",
    )


def short_iron_butterfly(
    strike: float, T: float, wing_width_pct: float = 0.05,
) -> Strategy:
    """Short iron butterfly: short ATM straddle + long OTM wings.

    Parameters
    ----------
    strike : float
        ATM strike (= spot for ATM).
    T : float
        Time to expiry in years.
    wing_width_pct : float
        Wing distance as fraction of strike (e.g. 0.05 = 5%).
    """
    put_wing = strike * (1 - wing_width_pct)
    call_wing = strike * (1 + wing_width_pct)
    return Strategy(
        legs=[
            OptionLeg("call", "short", strike, T),
            OptionLeg("put", "short", strike, T),
            OptionLeg("put", "long", put_wing, T),
            OptionLeg("call", "long", call_wing, T),
        ],
        name=f"Iron Butterfly ({wing_width_pct:.0%} wings)",
    )


def short_bwb(
    strike: float, T: float,
    call_wing_pct: float = 0.04, put_wing_pct: float = 0.07,
) -> Strategy:
    """Broken Wing Butterfly: asymmetric wings (wider put wing)."""
    put_wing = strike * (1 - put_wing_pct)
    call_wing = strike * (1 + call_wing_pct)
    return Strategy(
        legs=[
            OptionLeg("call", "short", strike, T),
            OptionLeg("put", "short", strike, T),
            OptionLeg("put", "long", put_wing, T),
            OptionLeg("call", "long", call_wing, T),
        ],
        name=f"BWB (call {call_wing_pct:.0%} / put {put_wing_pct:.0%})",
    )


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VRP Strategy Simulation — Straddle vs Iron Butterfly",
    layout="wide",
)
st.title("VRP Strategy Simulation")
st.markdown(
    "Compare **Short Straddle** vs **Iron Butterfly** vs **Broken Wing Butterfly** "
    "under GBM with Student-t tails. All strategies are *short vol* (selling the straddle body)."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Underlying")
S = st.sidebar.number_input("Spot Price (SPY)", value=550.0, step=10.0)
r = st.sidebar.slider("Risk-Free Rate", 0.00, 0.10, 0.045, 0.005)
q = st.sidebar.slider("Dividend Yield", 0.00, 0.05, 0.013, 0.001)

st.sidebar.header("Volatility Smile")
atm_vol = st.sidebar.slider("ATM Vol (VIX proxy)", 0.08, 0.50, 0.18, 0.01)
skew = st.sidebar.slider("Skew Strength", 0.00, 0.99, 0.40, 0.05,
                          help="0 = flat, higher = steeper downside skew (realistic SPY ~0.3-0.5)")
curvature = st.sidebar.slider("Smile Curvature", 0.00, 2.00, 0.50, 0.05,
                                help="Wing steepness. SPY typically 0.3-0.7")

st.sidebar.header("Monte Carlo")
n_paths = st.sidebar.select_slider(
    "Simulation Paths", [10_000, 25_000, 50_000, 100_000, 250_000], value=100_000,
)
df_student = st.sidebar.slider(
    "Student-t df", 3.0, 30.0, 5.0, 0.5,
    help="Lower = fatter tails. df=5 is typical for equity markets, df=30 ~ normal",
)

st.sidebar.header("Strategy Parameters")
T_days = st.sidebar.select_slider(
    "DTE (Days to Expiry)", [7, 14, 21, 30, 45, 60], value=30,
)
T = T_days / 365.0

wing_width = st.sidebar.slider(
    "Symmetric Wing Width (%)", 2, 15, 5, 1,
    help="Distance of protective wings from ATM, as % of spot",
) / 100.0

bwb_call_wing = st.sidebar.slider("BWB Call Wing (%)", 2, 10, 4, 1) / 100.0
bwb_put_wing = st.sidebar.slider("BWB Put Wing (%)", 3, 15, 7, 1) / 100.0

# ---------------------------------------------------------------------------
# Smile preview
# ---------------------------------------------------------------------------
smile = VolSmile.from_simple(atm_vol, skew, curvature)
preview_K = np.linspace(S * 0.85, S * 1.15, 100)
preview_iv = smile.get_iv_for_strike(preview_K, S, T) * 100

st.sidebar.markdown("---")
st.sidebar.subheader("Smile Preview")
fig_smile = go.Figure()
fig_smile.add_trace(go.Scatter(
    x=preview_K, y=preview_iv, mode="lines",
    line=dict(color="#2196F3", width=2),
))
fig_smile.add_vline(x=S, line_dash="dash", line_color="gray",
                     annotation_text="ATM")
fig_smile.update_layout(
    xaxis_title="Strike", yaxis_title="IV (%)",
    template="plotly_white", height=250, margin=dict(l=40, r=20, t=30, b=30),
)
st.sidebar.plotly_chart(fig_smile, use_container_width=True)

# ---------------------------------------------------------------------------
# Build strategies
# ---------------------------------------------------------------------------
strat_straddle = short_straddle(S, T)
strat_ifly = short_iron_butterfly(S, T, wing_width)
strat_bwb = short_bwb(S, T, bwb_call_wing, bwb_put_wing)

strategies = {
    "Short Straddle": strat_straddle,
    "Iron Butterfly": strat_ifly,
    "BWB": strat_bwb,
}
colors = {
    "Short Straddle": "#F44336",
    "Iron Butterfly": "#2196F3",
    "BWB": "#4CAF50",
}

# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------
if st.button("Run Simulation", type="primary", use_container_width=True):

    # Compute payoffs, greeks, and MC for each strategy
    payoffs = {}
    mc_results = {}
    greeks_data = {}
    pnl_surfaces = {}

    for name, strat in strategies.items():
        payoffs[name] = compute_payoff_diagram(strat, S, r, q, smile)
        mc_results[name] = simulate_strategy(strat, S, r, q, smile, n_paths, df_student)
        greeks_data[name] = compute_portfolio_greeks(strat, S, r, q, smile)
        pnl_surfaces[name] = compute_pnl_surface(strat, S, r, q, smile)

    # ==================================================================
    # 1. Summary Comparison
    # ==================================================================
    st.header("1. Summary Comparison")

    summary_rows = []
    for name in strategies:
        mc = mc_results[name]
        pf = payoffs[name]
        gr = greeks_data[name]
        summary_rows.append({
            "Strategy": name,
            "Entry Credit": f"${-pf.entry_cost:,.2f}",
            "Max Profit": f"${pf.max_profit:,.2f}",
            "Max Loss": f"${pf.max_loss:,.2f}",
            "Prob of Profit": f"{mc.prob_profit:.1%}",
            "Expected P&L": f"${mc.expected_pnl:,.2f}",
            "Median P&L": f"${mc.median_pnl:,.2f}",
            "VaR 95%": f"${mc.var_95:,.2f}",
            "CVaR 95%": f"${mc.cvar_95:,.2f}",
            "VaR 99%": f"${mc.var_99:,.2f}",
            "CVaR 99%": f"${mc.cvar_99:,.2f}",
            "Delta": f"{gr.delta:.3f}",
            "Gamma": f"{gr.gamma:.4f}",
            "Theta/day": f"${gr.theta / 365:.2f}",
            "Vega": f"{gr.vega:.2f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # Key ratios
    st.subheader("Risk/Reward Ratios")
    ratio_rows = []
    for name in strategies:
        mc = mc_results[name]
        pf = payoffs[name]
        credit = -pf.entry_cost
        max_loss = abs(pf.max_loss) if pf.max_loss < 0 else 0.01
        ratio_rows.append({
            "Strategy": name,
            "Credit / Max Loss": f"{credit / max_loss:.2%}" if max_loss > 0.01 else "Unlimited risk",
            "Expected / Credit": f"{mc.expected_pnl / credit:.2%}" if credit > 0 else "N/A",
            "CVaR 99 / Credit": f"{mc.cvar_99 / credit:.2%}" if credit > 0 else "N/A",
            "Margin Estimate": f"${max_loss * 100:,.0f}" if max_loss < S else f"${S * 100 * 0.20:,.0f} (20% notional)",
        })
    st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True)

    # ==================================================================
    # 2. Payoff Diagrams Overlay
    # ==================================================================
    st.header("2. Payoff at Expiry")
    fig_payoff = go.Figure()
    for name in strategies:
        pf = payoffs[name]
        fig_payoff.add_trace(go.Scatter(
            x=pf.S_range, y=pf.pnl_at_expiry,
            mode="lines", name=name,
            line=dict(color=colors[name], width=2),
        ))
    fig_payoff.add_hline(y=0, line_color="black", line_width=0.5)
    fig_payoff.add_vline(x=S, line_dash="dash", line_color="gray",
                          annotation_text="ATM")
    fig_payoff.update_layout(
        xaxis_title="Spot at Expiry", yaxis_title="P&L per unit ($)",
        template="plotly_white", height=450,
        title_text=f"Payoff Comparison — {T_days} DTE, ATM Vol {atm_vol:.0%}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_payoff, use_container_width=True)

    # ==================================================================
    # 3. P&L Over Time (time decay comparison)
    # ==================================================================
    st.header("3. P&L Over Time")
    fig_pnl_time = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(strategies.keys()),
        shared_yaxes=True,
    )
    time_colors = ["#1a237e", "#283593", "#3949ab", "#5c6bc0", "#9fa8da", "#e8eaf6"]
    for col_idx, (name, strat) in enumerate(strategies.items(), 1):
        surf = pnl_surfaces[name]
        for t_idx, t_elapsed in enumerate(surf.t_range):
            dte = max(T_days - t_elapsed * 365, 0)
            fig_pnl_time.add_trace(go.Scatter(
                x=surf.S_range, y=surf.pnl_grid[t_idx],
                mode="lines",
                name=f"{dte:.0f} DTE" if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                line=dict(color=time_colors[t_idx % len(time_colors)], width=1.5),
            ), row=1, col=col_idx)
        fig_pnl_time.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=col_idx)

    fig_pnl_time.update_layout(
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    fig_pnl_time.update_xaxes(title_text="Spot", row=1)
    fig_pnl_time.update_yaxes(title_text="P&L ($)", row=1, col=1)
    st.plotly_chart(fig_pnl_time, use_container_width=True)

    # ==================================================================
    # 4. Monte Carlo P&L Distributions
    # ==================================================================
    st.header("4. Monte Carlo P&L Distributions")
    st.markdown(f"**{n_paths:,} paths** | Student-t df={df_student:.1f} | ATM Vol={atm_vol:.0%}")

    fig_mc = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(strategies.keys()),
        shared_xaxes=False,
    )
    for col_idx, (name, strat) in enumerate(strategies.items(), 1):
        mc = mc_results[name]
        pnl = mc.pnl_samples

        # Split into profit/loss for coloring
        profit_mask = pnl > 0
        loss_mask = pnl <= 0

        # Use a common bin range
        bin_min = np.percentile(pnl, 1)
        bin_max = np.percentile(pnl, 99)
        bins = np.linspace(bin_min, bin_max, 60)

        fig_mc.add_trace(go.Histogram(
            x=pnl[profit_mask], xbins=dict(start=bin_min, end=bin_max, size=(bin_max-bin_min)/60),
            marker_color="rgba(76,175,80,0.7)", name="Profit" if col_idx == 1 else None,
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)
        fig_mc.add_trace(go.Histogram(
            x=pnl[loss_mask], xbins=dict(start=bin_min, end=bin_max, size=(bin_max-bin_min)/60),
            marker_color="rgba(244,67,54,0.7)", name="Loss" if col_idx == 1 else None,
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

        # Add VaR lines
        fig_mc.add_vline(
            x=-mc.var_95, line_dash="dash", line_color="orange",
            annotation_text=f"VaR95: ${mc.var_95:.0f}",
            row=1, col=col_idx,
        )
        fig_mc.add_vline(
            x=mc.expected_pnl, line_dash="dot", line_color="blue",
            annotation_text=f"E[P&L]: ${mc.expected_pnl:.0f}",
            row=1, col=col_idx,
        )

    fig_mc.update_layout(
        template="plotly_white", height=400, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    fig_mc.update_xaxes(title_text="P&L ($)")
    fig_mc.update_yaxes(title_text="Count", row=1, col=1)
    st.plotly_chart(fig_mc, use_container_width=True)

    # ==================================================================
    # 5. Tail Risk Analysis
    # ==================================================================
    st.header("5. Tail Risk Analysis")
    st.markdown("What happens in the worst 5% of scenarios?")

    tail_rows = []
    for name in strategies:
        mc = mc_results[name]
        pnl = mc.pnl_samples
        worst_5pct = np.sort(pnl)[:int(len(pnl) * 0.05)]
        worst_1pct = np.sort(pnl)[:int(len(pnl) * 0.01)]

        tail_rows.append({
            "Strategy": name,
            "Worst Case": f"${np.min(pnl):,.0f}",
            "Worst 1% Avg": f"${np.mean(worst_1pct):,.0f}",
            "Worst 5% Avg": f"${np.mean(worst_5pct):,.0f}",
            "Worst 10% Avg": f"${np.mean(np.sort(pnl)[:int(len(pnl)*0.10)]):,.0f}",
            "% Paths Loss > $20": f"{np.mean(pnl < -20):.1%}",
            "% Paths Loss > $50": f"{np.mean(pnl < -50):.1%}",
            "Tail Reduction vs Straddle": (
                f"{(1 - mc.cvar_99 / mc_results['Short Straddle'].cvar_99):.0%}"
                if name != "Short Straddle" and mc_results["Short Straddle"].cvar_99 > 0
                else "—"
            ),
        })
    st.dataframe(pd.DataFrame(tail_rows), use_container_width=True)

    # ==================================================================
    # 6. Wing Width Sensitivity
    # ==================================================================
    st.header("6. Wing Width Sensitivity")
    st.markdown("How do metrics change as we move wings closer or further from ATM?")

    wing_widths = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
    sensitivity_rows = []
    straddle_mc = mc_results["Short Straddle"]

    for ww in wing_widths:
        ifly = short_iron_butterfly(S, T, ww)
        ifly_pf = compute_payoff_diagram(ifly, S, r, q, smile)
        ifly_mc = simulate_strategy(ifly, S, r, q, smile, n_paths=50_000, df=df_student)
        credit = -ifly_pf.entry_cost
        straddle_credit = -payoffs["Short Straddle"].entry_cost
        retention = credit / straddle_credit if straddle_credit > 0 else 0

        sensitivity_rows.append({
            "Wing Width": f"{ww:.0%}",
            "Put Wing": f"${S * (1 - ww):,.0f}",
            "Call Wing": f"${S * (1 + ww):,.0f}",
            "Credit": f"${credit:,.2f}",
            "Credit Retention": f"{retention:.0%}",
            "Max Loss": f"${ifly_pf.max_loss:,.2f}",
            "PoP": f"{ifly_mc.prob_profit:.1%}",
            "E[P&L]": f"${ifly_mc.expected_pnl:,.2f}",
            "VaR 95%": f"${ifly_mc.var_95:,.2f}",
            "CVaR 99%": f"${ifly_mc.cvar_99:,.2f}",
            "Tail Reduction": f"{(1 - ifly_mc.cvar_99 / straddle_mc.cvar_99):.0%}" if straddle_mc.cvar_99 > 0 else "—",
        })

    st.dataframe(pd.DataFrame(sensitivity_rows), use_container_width=True)

    # Chart: credit retention vs tail reduction
    retentions = []
    tail_reductions = []
    for ww in wing_widths:
        ifly = short_iron_butterfly(S, T, ww)
        ifly_pf = compute_payoff_diagram(ifly, S, r, q, smile)
        ifly_mc = simulate_strategy(ifly, S, r, q, smile, n_paths=50_000, df=df_student)
        credit = -ifly_pf.entry_cost
        straddle_credit = -payoffs["Short Straddle"].entry_cost
        retentions.append(credit / straddle_credit if straddle_credit > 0 else 0)
        tail_reductions.append(
            1 - ifly_mc.cvar_99 / straddle_mc.cvar_99 if straddle_mc.cvar_99 > 0 else 0
        )

    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(go.Scatter(
        x=[r * 100 for r in retentions],
        y=[t * 100 for t in tail_reductions],
        mode="markers+text",
        text=[f"{ww:.0%}" for ww in wing_widths],
        textposition="top center",
        marker=dict(size=10, color="#2196F3"),
    ))
    fig_tradeoff.update_layout(
        xaxis_title="Credit Retention vs Naked Straddle (%)",
        yaxis_title="Tail Risk Reduction (CVaR 99%) (%)",
        template="plotly_white", height=400,
        title_text="Wing Width Tradeoff: Credit Retention vs Tail Protection",
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    # ==================================================================
    # 7. Maturity Comparison
    # ==================================================================
    st.header("7. Maturity Comparison")
    st.markdown("Same strategy structure across different expiries.")

    maturities = [7, 14, 21, 30, 45, 60]
    for strat_name, builder in [
        ("Short Straddle", lambda T_yr: short_straddle(S, T_yr)),
        ("Iron Butterfly", lambda T_yr: short_iron_butterfly(S, T_yr, wing_width)),
    ]:
        mat_rows = []
        for dte in maturities:
            T_yr = dte / 365.0
            strat = builder(T_yr)
            pf = compute_payoff_diagram(strat, S, r, q, smile)
            mc = simulate_strategy(strat, S, r, q, smile, n_paths=50_000, df=df_student)
            gr = compute_portfolio_greeks(strat, S, r, q, smile)
            mat_rows.append({
                "DTE": dte,
                "Credit": f"${-pf.entry_cost:,.2f}",
                "Max Loss": f"${pf.max_loss:,.2f}",
                "PoP": f"{mc.prob_profit:.1%}",
                "E[P&L]": f"${mc.expected_pnl:,.2f}",
                "VaR 95%": f"${mc.var_95:,.2f}",
                "CVaR 99%": f"${mc.cvar_99:,.2f}",
                "Theta/day": f"${gr.theta / 365:.3f}",
                "Gamma": f"{gr.gamma:.5f}",
            })
        st.subheader(strat_name)
        st.dataframe(pd.DataFrame(mat_rows), use_container_width=True)

    # ==================================================================
    # 8. Skew Sensitivity
    # ==================================================================
    st.header("8. Skew Sensitivity")
    st.markdown("How does the cost of protection change with skew steepness?")

    skew_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    skew_rows = []
    for sk in skew_levels:
        sk_smile = VolSmile.from_simple(atm_vol, sk, curvature)
        # Price both strategies under this smile
        strad = short_straddle(S, T)
        ifly = short_iron_butterfly(S, T, wing_width)

        strad_pf = compute_payoff_diagram(strad, S, r, q, sk_smile)
        ifly_pf = compute_payoff_diagram(ifly, S, r, q, sk_smile)

        strad_credit = -strad_pf.entry_cost
        ifly_credit = -ifly_pf.entry_cost

        # Wing costs
        put_wing_K = S * (1 - wing_width)
        call_wing_K = S * (1 + wing_width)
        put_iv = float(sk_smile.get_iv_for_strike(put_wing_K, S, T))
        call_iv = float(sk_smile.get_iv_for_strike(call_wing_K, S, T))

        skew_rows.append({
            "Skew": f"{sk:.1f}",
            "ATM IV": f"{atm_vol:.0%}",
            f"Put Wing IV ({wing_width:.0%} OTM)": f"{put_iv:.1%}",
            f"Call Wing IV ({wing_width:.0%} OTM)": f"{call_iv:.1%}",
            "Put-Call IV Gap": f"{(put_iv - call_iv):.1%}",
            "Straddle Credit": f"${strad_credit:,.2f}",
            "IFly Credit": f"${ifly_credit:,.2f}",
            "Wing Cost": f"${strad_credit - ifly_credit:,.2f}",
            "Credit Retention": f"{ifly_credit / strad_credit:.0%}" if strad_credit > 0 else "N/A",
        })
    st.dataframe(pd.DataFrame(skew_rows), use_container_width=True)

    # ==================================================================
    # 9. Volatility Regime Comparison
    # ==================================================================
    st.header("9. Volatility Regime Comparison")
    st.markdown("Same structures under low, medium, and high vol environments.")

    vol_regimes = [
        ("Low Vol (VIX ~12)", 0.12),
        ("Normal Vol (VIX ~18)", 0.18),
        ("Elevated Vol (VIX ~25)", 0.25),
        ("High Vol (VIX ~35)", 0.35),
        ("Crisis Vol (VIX ~50)", 0.50),
    ]

    vol_rows = []
    for regime_name, vol_level in vol_regimes:
        vol_smile = VolSmile.from_simple(vol_level, skew, curvature)
        for strat_name, strat in [
            ("Straddle", short_straddle(S, T)),
            ("IFly", short_iron_butterfly(S, T, wing_width)),
        ]:
            pf = compute_payoff_diagram(strat, S, r, q, vol_smile)
            mc = simulate_strategy(strat, S, r, q, vol_smile, n_paths=50_000, df=df_student)
            vol_rows.append({
                "Regime": regime_name,
                "Strategy": strat_name,
                "Credit": f"${-pf.entry_cost:,.2f}",
                "PoP": f"{mc.prob_profit:.1%}",
                "E[P&L]": f"${mc.expected_pnl:,.2f}",
                "VaR 95%": f"${mc.var_95:,.2f}",
                "CVaR 99%": f"${mc.cvar_99:,.2f}",
            })
    st.dataframe(pd.DataFrame(vol_rows), use_container_width=True)

    # ==================================================================
    # 10. Greeks Comparison
    # ==================================================================
    st.header("10. Greeks at Entry")

    greeks_rows = []
    for name in strategies:
        gr = greeks_data[name]
        greeks_rows.append({
            "Strategy": name,
            "Delta": f"{gr.delta:.4f}",
            "Gamma": f"{gr.gamma:.5f}",
            "Theta (annual)": f"{gr.theta:.2f}",
            "Theta/day": f"${gr.theta / 365:.3f}",
            "Vega": f"{gr.vega:.3f}",
            "Vanna": f"{gr.vanna:.5f}",
            "Volga": f"{gr.volga:.5f}",
        })
    st.dataframe(pd.DataFrame(greeks_rows), use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Key takeaway:** Compare the *Credit Retention* vs *Tail Risk Reduction* tradeoff "
        "in Section 6 to find the optimal wing width. The iron butterfly gives up some premium "
        "but caps the worst-case loss. The BWB uses wider put wings (cheaper due to skew awareness) "
        "to balance protection cost vs downside coverage."
    )
