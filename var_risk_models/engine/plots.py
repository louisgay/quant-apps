"""Plotly charts for dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, t as student_t

COLORS = {
    "Historical Simulation": "#1f77b4",
    "Parametric Normal": "#2ca02c",
    "Parametric Student-t": "#ff7f0e",
    "GARCH Normal": "#9467bd",
    "GJR-GARCH Student-t": "#d62728",
    "Monte-Carlo FHS": "#e377c2",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(size=12),
    margin=dict(l=60, r=30, t=50, b=40),
    legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
)


def _apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**{**LAYOUT_DEFAULTS, **kwargs})
    return fig


def returns_distribution(
    returns: pd.Series,
    fitted_nu: float | None = None,
) -> go.Figure:
    """Histogram + Normal/Student-t PDF overlays."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns.values,
        nbinsx=100,
        histnorm="probability density",
        marker_color="rgba(100, 149, 237, 0.5)",
        name="Empirical",
    ))

    x_range = np.linspace(returns.min(), returns.max(), 300)
    mu, sigma = returns.mean(), returns.std()
    fig.add_trace(go.Scatter(
        x=x_range,
        y=norm.pdf(x_range, loc=mu, scale=sigma),
        mode="lines",
        line=dict(color="#2ca02c", width=2),
        name=f"Normal (mu={mu:.4f}, sigma={sigma:.4f})",
    ))

    if fitted_nu is not None:
        nu_display = fitted_nu
    else:
        nu_display, _, _ = student_t.fit(returns.values)

    t_pdf = student_t.pdf(x_range, df=nu_display, loc=mu, scale=sigma)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=t_pdf,
        mode="lines",
        line=dict(color="#d62728", width=2, dash="dash"),
        name=f"Student-t (nu={nu_display:.1f})",
    ))

    return _apply_layout(fig, title="Return Distribution", xaxis_title="Log Return",
                         yaxis_title="Density", height=400)


def qq_plot(returns: pd.Series) -> go.Figure:
    """Q-Q vs Normal."""
    sorted_returns = np.sort(returns.values)
    n = len(sorted_returns)
    theoretical = norm.ppf(np.linspace(0.001, 0.999, n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical,
        y=sorted_returns,
        mode="markers",
        marker=dict(color="#1f77b4", size=2, opacity=0.6),
        name="Data",
    ))

    line_min = min(theoretical.min(), sorted_returns.min())
    line_max = max(theoretical.max(), sorted_returns.max())
    fig.add_trace(go.Scatter(
        x=[line_min, line_max],
        y=[line_min, line_max],
        mode="lines",
        line=dict(color="red", width=1, dash="dash"),
        name="Normal reference",
    ))

    return _apply_layout(fig, title="Q-Q Plot (vs Normal)",
                         xaxis_title="Theoretical Quantiles",
                         yaxis_title="Sample Quantiles", height=400)


def rolling_var_comparison(
    returns: pd.Series,
    var_dict: dict[str, pd.Series],
) -> go.Figure:
    """All methods overlaid on returns."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=returns.index,
        y=returns.values,
        mode="lines",
        line=dict(color="gray", width=0.4),
        opacity=0.4,
        name="Returns",
    ))

    color_list = list(COLORS.values())
    for idx, (method_name, var_series) in enumerate(var_dict.items()):
        valid = var_series.dropna()
        color = color_list[idx % len(color_list)]
        fig.add_trace(go.Scatter(
            x=valid.index,
            y=valid.values,
            mode="lines",
            line=dict(color=color, width=1.2),
            name=method_name,
        ))

    return _apply_layout(fig, title="Rolling VaR Comparison",
                         yaxis_title="Log Return / VaR",
                         height=500)


def exceedance_scatter(
    returns: pd.Series,
    var_series: pd.Series,
    method_name: str,
) -> go.Figure:
    aligned = pd.concat([returns, var_series], axis=1, join="inner").dropna()
    aligned.columns = ["return", "var"]
    breaches = aligned[aligned["return"] < aligned["var"]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=returns.index,
        y=returns.values,
        mode="lines",
        line=dict(color="gray", width=0.4),
        opacity=0.4,
        name="Returns",
    ))

    valid = var_series.dropna()
    fig.add_trace(go.Scatter(
        x=valid.index,
        y=valid.values,
        mode="lines",
        line=dict(color=COLORS.get(method_name, "#1f77b4"), width=1.2),
        name=f"{method_name} VaR",
    ))

    fig.add_trace(go.Scatter(
        x=breaches.index,
        y=breaches["return"].values,
        mode="markers",
        marker=dict(color="red", size=5, opacity=0.7),
        name=f"Breaches ({len(breaches)})",
    ))

    return _apply_layout(fig, title=f"{method_name} — Exceedances",
                         yaxis_title="Log Return", height=300)


def conditional_volatility(
    dates: pd.DatetimeIndex,
    sigma: np.ndarray,
    returns: pd.Series,
) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.6], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=dates,
        y=returns.values * 100,
        mode="lines",
        line=dict(color="gray", width=0.5),
        opacity=0.5,
        name="Daily returns (%)",
    ), row=1, col=1)

    sigma_ann = sigma * np.sqrt(252) * 100
    fig.add_trace(go.Scatter(
        x=dates,
        y=sigma_ann,
        mode="lines",
        line=dict(color="#d62728", width=1),
        fill="tozeroy",
        fillcolor="rgba(214, 39, 40, 0.12)",
        name="GJR-GARCH Vol (ann. %)",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=2, col=1)
    fig.update_layout(**LAYOUT_DEFAULTS, height=500,
                      title="GJR-GARCH Conditional Volatility")
    return fig


def news_impact_curve_plot(
    shocks: np.ndarray,
    sigma2_next: np.ndarray,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=shocks * 100,
        y=sigma2_next * 10000,  # display in bps^2 or %^2
        mode="lines",
        line=dict(color="#d62728", width=2),
        name="NIC",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    return _apply_layout(
        fig,
        title="News Impact Curve (GJR Asymmetry)",
        xaxis_title="Shock (return, %)",
        yaxis_title="Next-period variance (x10^4)",
        height=400,
    )


def fhs_return_distribution(
    sim_returns: np.ndarray,
    var_95: float,
    var_99: float,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=sim_returns,
        nbinsx=150,
        histnorm="probability density",
        marker_color="rgba(100, 149, 237, 0.5)",
        name="Simulated returns",
    ))

    fig.add_vline(x=var_95, line_dash="dash", line_color="#ff7f0e",
                  annotation_text=f"VaR 95% = {var_95:.4f}")
    fig.add_vline(x=var_99, line_dash="solid", line_color="#d62728",
                  annotation_text=f"VaR 99% = {var_99:.4f}")

    return _apply_layout(fig, title="FHS Simulated Return Distribution",
                         xaxis_title="Simulated Return",
                         yaxis_title="Density", height=400)


def fhs_fan_chart(
    paths: np.ndarray,
    horizon: int,
    var_95: float,
    var_99: float,
) -> go.Figure:
    fig = go.Figure()

    if paths is None or horizon <= 1:
        fig.add_annotation(text="Fan chart requires horizon > 1",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False)
        return _apply_layout(fig, title="FHS Fan Chart", height=400)

    days = np.arange(1, horizon + 1)
    cum_paths = np.cumsum(paths, axis=1)

    bands = [1, 5, 25, 50, 75, 95, 99]
    percentiles = np.percentile(cum_paths, bands, axis=0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([percentiles[0], percentiles[6][::-1]]),
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.08)",
        line=dict(width=0),
        name="1%-99%",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([percentiles[1], percentiles[5][::-1]]),
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.15)",
        line=dict(width=0),
        name="5%-95%",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([percentiles[2], percentiles[4][::-1]]),
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.25)",
        line=dict(width=0),
        name="25%-75%",
    ))

    fig.add_trace(go.Scatter(
        x=days,
        y=percentiles[3],
        mode="lines",
        line=dict(color="#d62728", width=2),
        name="Median",
    ))

    return _apply_layout(fig, title="FHS Confidence Cone",
                         xaxis_title="Horizon (days)",
                         yaxis_title="Cumulative Return", height=400)


def fhs_convergence_plot(
    n_sims: np.ndarray,
    var_values: np.ndarray,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=n_sims,
        y=var_values,
        mode="lines+markers",
        line=dict(color="#d62728", width=1.5),
        marker=dict(size=3),
        name="VaR estimate",
    ))

    final_var = var_values[-1]
    fig.add_hline(y=final_var, line_dash="dash", line_color="gray",
                  annotation_text=f"Converged: {final_var:.5f}")

    return _apply_layout(fig, title="FHS Convergence Diagnostic",
                         xaxis_title="Number of Simulations",
                         yaxis_title="VaR Estimate", height=350)


def var_comparison_bar(
    var_values: dict[str, float],
    confidence: float,
) -> go.Figure:
    methods = list(var_values.keys())
    values = list(var_values.values())

    color_list = [COLORS.get(m, "#636EFA") for m in methods]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods,
        y=values,
        marker_color=color_list,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))

    return _apply_layout(
        fig,
        title=f"VaR Comparison ({confidence:.0%} Confidence)",
        yaxis_title="VaR (log return)",
        height=400,
        showlegend=False,
    )
