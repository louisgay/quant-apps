"""VaR Risk Models dashboard — 6 methods, historical → FHS.

streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import t as student_t, kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.data_loader import TICKER_UNIVERSE, compute_log_returns, fetch_prices
from engine.garch import (
    fit_gjr_garch,
    fhs_convergence,
    news_impact_curve,
    simulate_fhs,
)
from engine.var_models import (
    garch_normal_var,
    garch_student_t_var,
    historical_simulation_var,
    monte_carlo_fhs_var,
    parametric_normal_var,
    parametric_student_t_var,
)
from engine.backtest import (
    christoffersen_test,
    combined_coverage_test,
    kupiec_test,
    run_full_backtest,
    var_exceedance_test,
)
from engine.plots import (
    conditional_volatility,
    exceedance_scatter,
    fhs_convergence_plot,
    fhs_fan_chart,
    fhs_return_distribution,
    news_impact_curve_plot,
    qq_plot,
    returns_distribution,
    rolling_var_comparison,
    var_comparison_bar,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("app")


# --- Helper for parallel / cached VaR computation ---

def _compute_single_var(method_name, returns_values, returns_index, window, confidence, garch_step, mc_sims):
    """Compute a single VaR method. Designed for use with ProcessPoolExecutor."""
    log_returns = pd.Series(returns_values, index=pd.DatetimeIndex(returns_index))

    if method_name == "Historical Simulation":
        return historical_simulation_var(log_returns, window=window, confidence=confidence)
    elif method_name == "Parametric Normal":
        return parametric_normal_var(log_returns, window=window, confidence=confidence)
    elif method_name == "Parametric Student-t":
        return parametric_student_t_var(log_returns, window=window, confidence=confidence, step=garch_step)
    elif method_name == "GARCH Normal":
        return garch_normal_var(log_returns, confidence=confidence, step=garch_step)
    elif method_name == "GARCH Student-t":
        return garch_student_t_var(log_returns, confidence=confidence, step=garch_step)
    elif method_name == "GJR-GARCH FHS":
        return monte_carlo_fhs_var(log_returns, confidence=confidence, n_sims=mc_sims, step=garch_step)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_compute_all_var(returns_values, returns_index_vals, window, confidence, garch_step, mc_sims):
    """Compute all 6 VaR methods with caching. Uses thread-based parallelization.

    ThreadPoolExecutor works well here because numpy/scipy release the GIL during
    numerical computation (MLE optimization, linear algebra). This avoids the
    multiprocessing spawn issue with Streamlit.
    """
    returns_index = pd.DatetimeIndex(returns_index_vals)
    log_returns = pd.Series(returns_values, index=returns_index)

    methods = [
        "Historical Simulation",
        "Parametric Normal",
        "Parametric Student-t",
        "GARCH Normal",
        "GARCH Student-t",
        "GJR-GARCH FHS",
    ]

    var_dict = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                _compute_single_var,
                method, returns_values, returns_index_vals,
                window, confidence, garch_step, mc_sims
            ): method
            for method in methods
        }
        for future in as_completed(futures):
            method_name = futures[future]
            try:
                var_dict[method_name] = future.result()
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
                var_dict[method_name] = pd.Series(
                    np.nan, index=returns_index, name=method_name
                )

    return var_dict


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fit_full_garch(returns_values, returns_index_vals):
    """Cache full-sample GJR-GARCH fit."""
    log_returns = pd.Series(returns_values, index=pd.DatetimeIndex(returns_index_vals))
    return fit_gjr_garch(log_returns, distribution="studentt")



st.set_page_config(
    page_title="VaR Risk Models",
    page_icon="📊",
    layout="wide",
)

st.title("Value-at-Risk: 6-Method Comparison")
st.caption(
    "From Historical Simulation to Monte-Carlo GJR-GARCH Filtered Historical Simulation"
)



@st.cache_data(ttl=3600)
def cached_fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    return fetch_prices(ticker, start, end)



with st.sidebar:
    st.header("Configuration")

    # Quick mode toggle
    quick_mode = st.toggle("Quick Mode", value=True,
                           help="Faster computation with slightly lower precision (step=10, sims=1000)")

    st.divider()

    # Ticker selection
    ticker_options = [f"{k} — {v}" for k, v in TICKER_UNIVERSE.items()]
    ticker_selection = st.selectbox("Ticker", ticker_options, index=0)
    ticker = ticker_selection.split(" — ")[0]

    st.divider()

    # Date range
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start", value=pd.Timestamp("2015-01-01"))
    with col_end:
        end_date = st.date_input("End", value=pd.Timestamp("2024-12-31"))

    st.divider()

    # Confidence level
    confidence_choice = st.radio(
        "Confidence Level", ["95%", "99%", "Both"], index=1
    )

    # Rolling window
    window = st.slider("Rolling Window (days)", min_value=100, max_value=504,
                       value=252, step=10)

    # GARCH parameters
    st.divider()
    st.subheader("GARCH Settings")

    if quick_mode:
        garch_step = 10
        mc_sims = 1000
        st.caption(f"Refit step: {garch_step} days | MC sims: {mc_sims}")
    else:
        garch_step = st.slider("GARCH Refit Step (days)", min_value=1, max_value=20,
                               value=3, step=1,
                               help="Fit GARCH every N-th day for performance")
        mc_sims = st.slider("MC Simulations (FHS)", min_value=500, max_value=10000,
                            value=2000, step=500,
                            help="More simulations = more accurate but slower")

    st.divider()
    show_convergence = st.checkbox("Show FHS convergence diagnostic", value=False,
                                   help="Runs 50,000 extra simulations — slow")

    st.divider()
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)



if run_button:
    confidence_levels = []
    if confidence_choice == "95%":
        confidence_levels = [0.95]
    elif confidence_choice == "99%":
        confidence_levels = [0.99]
    else:
        confidence_levels = [0.95, 0.99]

    # Fetch data
    with st.spinner("Downloading price data..."):
        try:
            prices = cached_fetch_prices(ticker, str(start_date), str(end_date))
        except Exception as e:
            st.error(f"Failed to fetch data for {ticker}: {e}")
            st.stop()

    log_returns = compute_log_returns(prices)

    if len(log_returns) < 504:
        st.error(f"Insufficient data: only {len(log_returns)} observations "
                 f"(need at least 504 for GARCH fitting).")
        st.stop()

    # Progressive display with status
    status = st.status("Computing VaR models...", expanded=True)
    progress = st.progress(0)

    with status:
        # Step 1: Compute all 6 VaR methods (cached + parallel)
        st.write("Computing 6 VaR methods (parallel)...")
        var_dict = cached_compute_all_var(
            log_returns.values,
            log_returns.index.values,
            window,
            confidence_levels[0],
            garch_step,
            mc_sims,
        )
        progress.progress(60)

        # Step 2: Full-sample GJR-GARCH fit (cached)
        st.write("Fitting full-sample GJR-GARCH...")
        try:
            full_garch = cached_fit_full_garch(
                log_returns.values,
                log_returns.index.values,
            )
        except Exception as e:
            st.warning(f"Full-sample GARCH fit failed: {e}")
            full_garch = None
        progress.progress(80)

        # Step 3: FHS simulation
        fhs_result = None
        if full_garch:
            st.write("Running FHS Monte-Carlo simulation...")
            try:
                n_fhs_sims = 5000 if quick_mode else 10000
                fhs_result = simulate_fhs(full_garch, n_simulations=n_fhs_sims, horizon=1)
            except Exception as e:
                st.warning(f"FHS simulation failed: {e}")
        progress.progress(100)

        status.update(label="Analysis complete!", state="complete")

    # Store in session state
    st.session_state["results"] = {
        "log_returns": log_returns,
        "var_dict": var_dict,
        "full_garch": full_garch,
        "fhs_result": fhs_result,
        "confidence": confidence_levels[0],
        "ticker": ticker,
        "show_convergence": show_convergence,
    }



if "results" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Analysis** to start.")
    st.stop()

res = st.session_state["results"]
log_returns = res["log_returns"]
var_dict = res["var_dict"]
full_garch = res["full_garch"]
fhs_result = res["fhs_result"]
confidence = res["confidence"]
ticker = res["ticker"]

st.caption(
    f"**{ticker}** | {log_returns.index[0].date()} to {log_returns.index[-1].date()} "
    f"| {len(log_returns):,} observations | {confidence:.0%} confidence"
)

# Tabs
tab_overview, tab_comparison, tab_garch, tab_fhs, tab_tests, tab_theory = st.tabs([
    "Overview",
    "Method Comparison",
    "GARCH Deep Dive",
    "GJR-GARCH FHS",
    "Statistical Tests",
    "Theory",
])


# ---- Tab 1: Overview --------------------------------------------------------
with tab_overview:
    st.subheader("Return Statistics")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean (ann.)", f"{log_returns.mean() * 252:.2%}")
    with col2:
        st.metric("Vol (ann.)", f"{log_returns.std() * np.sqrt(252):.2%}")
    with col3:
        st.metric("Skewness", f"{skew(log_returns.values):.3f}")
    with col4:
        kurt = kurtosis(log_returns.values, fisher=True)
        st.metric("Excess Kurtosis", f"{kurt:.2f}")

    # Fit Student-t for nu display
    try:
        fitted_nu, _, _ = student_t.fit(log_returns.values)
    except Exception:
        fitted_nu = None

    col_dist, col_qq = st.columns(2)
    with col_dist:
        fig_dist = returns_distribution(log_returns, fitted_nu=fitted_nu)
        st.plotly_chart(fig_dist, width="stretch")
    with col_qq:
        fig_qq = qq_plot(log_returns)
        st.plotly_chart(fig_qq, width="stretch")

    # Quick VaR comparison table
    st.subheader("Latest VaR Estimates")
    latest_var = {}
    for method, series in var_dict.items():
        valid = series.dropna()
        if len(valid) > 0:
            latest_var[method] = valid.iloc[-1]

    if latest_var:
        fig_bar = var_comparison_bar(latest_var, confidence)
        st.plotly_chart(fig_bar, width="stretch", key="overview_var_bar")


# ---- Tab 2: Method Comparison -----------------------------------------------
with tab_comparison:
    st.subheader("Rolling VaR — All Methods")

    fig_comp = rolling_var_comparison(log_returns, var_dict)
    st.plotly_chart(fig_comp, width="stretch")

    # Exceedance summary table
    st.subheader("Exceedance Summary")
    bt_df = run_full_backtest(log_returns, var_dict, confidence)
    display_cols = ["Method", "n_obs", "n_exceedances", "exceedance_rate",
                    "expected_rate", "ratio", "kupiec_p", "christoffersen_p"]
    st.dataframe(
        bt_df[display_cols].style.format({
            "exceedance_rate": "{:.4f}",
            "expected_rate": "{:.4f}",
            "ratio": "{:.2f}",
            "kupiec_p": "{:.4f}",
            "christoffersen_p": "{:.4f}",
        }),
        hide_index=True,
        width="stretch",
    )

    # Exceedance scatter grid (2x3)
    st.subheader("Individual Method Exceedances")
    methods_list = list(var_dict.keys())
    for row_start in range(0, len(methods_list), 3):
        cols = st.columns(3)
        for col_idx, method_idx in enumerate(range(row_start, min(row_start + 3, len(methods_list)))):
            method_name = methods_list[method_idx]
            with cols[col_idx]:
                fig_exc = exceedance_scatter(log_returns, var_dict[method_name], method_name)
                st.plotly_chart(fig_exc, width="stretch")


# ---- Tab 3: GARCH Deep Dive ------------------------------------------------
with tab_garch:
    st.subheader("GJR-GARCH(1,1) with Student-t Innovations")

    if full_garch is None:
        st.warning("GARCH model fitting failed. Cannot display diagnostics.")
    else:
        # Parameter table
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Alpha", f"{full_garch.alpha:.4f}")
        with col2:
            st.metric("Gamma (leverage)", f"{full_garch.gamma:.4f}")
        with col3:
            st.metric("Beta", f"{full_garch.beta:.4f}")
        with col4:
            st.metric("Persistence", f"{full_garch.persistence:.4f}")
        with col5:
            nu_str = f"{full_garch.nu:.1f}" if full_garch.nu else "N/A"
            st.metric("d.o.f. (nu)", nu_str)

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Half-life (days)", f"{full_garch.half_life:.1f}")
        with col_info2:
            st.metric("Log-Likelihood", f"{full_garch.log_likelihood:.1f}")
        with col_info3:
            st.metric("AIC", f"{full_garch.aic:.1f}")

        # Conditional volatility plot
        dates = log_returns.index[:len(full_garch.conditional_vol)]
        fig_vol = conditional_volatility(dates, full_garch.conditional_vol, log_returns)
        st.plotly_chart(fig_vol, width="stretch")

        # News Impact Curve
        st.subheader("News Impact Curve")
        shocks, sigma2_next = news_impact_curve(full_garch)
        fig_nic = news_impact_curve_plot(shocks, sigma2_next)
        st.plotly_chart(fig_nic, width="stretch")

        st.markdown(
            "The asymmetric shape demonstrates the **leverage effect**: negative shocks "
            "(bad news) increase future volatility more than positive shocks of equal magnitude. "
            f"Gamma = {full_garch.gamma:.4f} captures this asymmetry."
        )

        # Residual diagnostics
        st.subheader("Standardized Residuals")
        col_hist, col_qq2 = st.columns(2)
        with col_hist:
            resids = pd.Series(full_garch.standardized_resids)
            fig_resid = returns_distribution(resids, fitted_nu=full_garch.nu)
            fig_resid.update_layout(title="Standardized Residual Distribution")
            st.plotly_chart(fig_resid, width="stretch")
        with col_qq2:
            fig_resid_qq = qq_plot(resids)
            fig_resid_qq.update_layout(title="Residuals Q-Q Plot")
            st.plotly_chart(fig_resid_qq, width="stretch")


# ---- Tab 4: GJR-GARCH FHS --------------------------------------------------
with tab_fhs:
    st.subheader("GJR-GARCH Filtered Historical Simulation")

    if fhs_result is None:
        st.warning("FHS simulation was not computed. Check GARCH fit.")
    else:
        # VaR and ES metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR 95%", f"{fhs_result.var_95:.4f}")
        with col2:
            st.metric("VaR 99%", f"{fhs_result.var_99:.4f}")
        with col3:
            st.metric("ES 95%", f"{fhs_result.es_95:.4f}")
        with col4:
            st.metric("ES 99%", f"{fhs_result.es_99:.4f}")

        # Simulated return distribution
        fig_fhs_dist = fhs_return_distribution(
            fhs_result.simulated_returns, fhs_result.var_95, fhs_result.var_99)
        st.plotly_chart(fig_fhs_dist, width="stretch")

        # Convergence diagnostic (optional — expensive)
        _show_conv = res.get("show_convergence", False)
        if _show_conv and full_garch:
            st.subheader("Convergence Diagnostic")
            with st.spinner("Running 50,000 convergence simulations..."):
                n_array, var_array = fhs_convergence(full_garch, max_sims=50000,
                                                      confidence=confidence)
            fig_conv = fhs_convergence_plot(n_array, var_array)
            st.plotly_chart(fig_conv, width="stretch")
            st.markdown(
                "VaR estimate stabilizes as simulation count increases. "
                "Convergence around N=5000 indicates sufficient precision."
            )
        elif not _show_conv:
            st.info("Enable 'Show FHS convergence diagnostic' in the sidebar to see convergence plot (adds ~15s).")

        # Comparison to parametric methods
        st.subheader("FHS vs Parametric VaR")
        compare_data = {}
        for method, series in var_dict.items():
            valid = series.dropna()
            if len(valid) > 0:
                compare_data[method] = valid.iloc[-1]
        if compare_data:
            fig_compare = var_comparison_bar(compare_data, confidence)
            st.plotly_chart(fig_compare, width="stretch", key="fhs_var_bar")

        # Multi-horizon fan chart
        if full_garch:
            st.subheader("Multi-Horizon Risk (10-day)")
            try:
                fhs_multi = simulate_fhs(full_garch, n_simulations=5000, horizon=10)
                fig_fan = fhs_fan_chart(fhs_multi.simulated_paths, 10,
                                        fhs_multi.var_95, fhs_multi.var_99)
                st.plotly_chart(fig_fan, width="stretch")
            except Exception as e:
                st.warning(f"Multi-horizon simulation failed: {e}")


# ---- Tab 5: Statistical Tests -----------------------------------------------
with tab_tests:
    st.subheader("VaR Model Validation")
    st.markdown(
        "Statistical tests at the **5% significance level** (p < 0.05 = reject H0 = model fails)."
    )

    results_rows = []
    for method_name, var_series in var_dict.items():
        exc = var_exceedance_test(log_returns, var_series, confidence)
        cc = combined_coverage_test(
            exc["n_obs"], exc["n_exceedances"],
            exc["exceedances"], confidence,
        )
        results_rows.append({
            "Method": method_name,
            "Exceedances": exc["n_exceedances"],
            "Rate": exc["exceedance_rate"],
            "Expected": exc["expected_rate"],
            "Ratio": exc["exceedance_ratio"],
            "Kupiec LR": cc.lr_uc,
            "Kupiec p": cc.p_uc,
            "Kupiec": "Pass" if cc.p_uc > 0.05 else "FAIL",
            "Christ. LR": cc.lr_ind,
            "Christ. p": cc.p_ind,
            "Independence": "Pass" if cc.p_ind > 0.05 else "FAIL",
            "Combined LR": cc.lr_cc,
            "Combined p": cc.p_cc,
            "Combined": "Pass" if cc.p_cc > 0.05 else "FAIL",
        })

    test_df = pd.DataFrame(results_rows)

    # Color-coded summary
    st.subheader("Test Results Summary")
    for _, row in test_df.iterrows():
        method = row["Method"]
        kup_color = "green" if row["Kupiec"] == "Pass" else "red"
        chr_color = "green" if row["Independence"] == "Pass" else "red"
        cc_color = "green" if row["Combined"] == "Pass" else "red"

        st.markdown(
            f"**{method}** — "
            f"Kupiec: :{kup_color}[{row['Kupiec']}] (p={row['Kupiec p']:.4f}) | "
            f"Independence: :{chr_color}[{row['Independence']}] (p={row['Christ. p']:.4f}) | "
            f"Combined: :{cc_color}[{row['Combined']}] (p={row['Combined p']:.4f})"
        )

    st.divider()

    # Full table
    st.subheader("Detailed Results")
    display_test_df = test_df[["Method", "Exceedances", "Rate", "Expected", "Ratio",
                               "Kupiec LR", "Kupiec p", "Kupiec",
                               "Christ. LR", "Christ. p", "Independence",
                               "Combined LR", "Combined p", "Combined"]].copy()
    st.dataframe(
        display_test_df.style.format({
            "Rate": "{:.4f}", "Expected": "{:.4f}", "Ratio": "{:.2f}",
            "Kupiec LR": "{:.3f}", "Kupiec p": "{:.4f}",
            "Christ. LR": "{:.3f}", "Christ. p": "{:.4f}",
            "Combined LR": "{:.3f}", "Combined p": "{:.4f}",
        }),
        hide_index=True,
        width="stretch",
    )

    st.markdown("""
    **Interpretation:**
    - **Kupiec test** (H0: correct unconditional coverage): Too many breaches → FAIL
    - **Christoffersen test** (H0: breaches are independent): Clustered breaches → FAIL
    - **Combined test** (H0: correct coverage AND independent): Joint criterion, chi2(df=2)
    - Significance level: alpha = 5%. Reject H0 if p < 0.05.
    """)


# ---- Tab 6: Theory ----------------------------------------------------------
with tab_theory:
    st.subheader("Mathematical Framework")

    st.markdown("### 1. Historical Simulation")
    st.latex(r"\text{VaR}_t^{\alpha} = Q_{1-\alpha}\left(r_{t-w}, \ldots, r_{t-1}\right)")
    st.markdown("Empirical quantile of the rolling window. Non-parametric, no distributional assumptions.")

    st.markdown("### 2. Parametric Normal")
    st.latex(r"\text{VaR}_t^{\alpha} = \hat{\mu} + z_{\alpha} \cdot \hat{\sigma}")
    st.latex(r"z_{\alpha} = \Phi^{-1}(1 - \alpha)")
    st.markdown("Assumes i.i.d. Gaussian returns. Fast but ignores fat tails and volatility clustering.")

    st.markdown("### 3. Parametric Student-t")
    st.latex(r"\text{VaR}_t^{\alpha} = \hat{\mu} + t_{\nu}^{-1}(1-\alpha) \cdot \hat{\sigma}")
    st.markdown(
        "MLE-fitted degrees of freedom captures heavy tails. "
        "Lower nu = heavier tails = more conservative VaR."
    )

    st.markdown("### 4. GARCH(1,1) Normal")
    st.latex(r"\sigma_t^2 = \omega + \alpha \, \varepsilon_{t-1}^2 + \beta \, \sigma_{t-1}^2")
    st.latex(r"\text{VaR}_t^{\alpha} = \mu + \Phi^{-1}(1-\alpha) \cdot \sigma_t")
    st.markdown("Captures volatility clustering. Persistence = alpha + beta < 1 for stationarity.")

    st.markdown("### 5. GARCH(1,1) Student-t")
    st.latex(r"\sigma_t^2 = \omega + \alpha \, \varepsilon_{t-1}^2 + \beta \, \sigma_{t-1}^2")
    st.latex(r"\text{VaR}_t^{\alpha} = \mu + t_{\nu}^{-1}(1-\alpha) \cdot \sigma_t")
    st.markdown(
        "Same variance dynamics as GARCH Normal, but uses Student-t quantile for the VaR threshold. "
        "Captures vol clustering + fat tails, isolating the effect of the distributional assumption."
    )

    st.markdown("### 6. GJR-GARCH FHS — Glosten, Jagannathan & Runkle (1993) + Barone-Adesi et al. (1999)")
    st.markdown("**Variance dynamics (GJR-GARCH):**")
    st.latex(r"\sigma_t^2 = \omega + \alpha \, \varepsilon_{t-1}^2 + \gamma \, \varepsilon_{t-1}^2 \, \mathbb{1}(\varepsilon_{t-1} < 0) + \beta \, \sigma_{t-1}^2")
    st.markdown(
        "The **leverage effect** (gamma > 0): negative shocks increase volatility more than positive shocks."
    )
    st.markdown("**Filtered Historical Simulation (Monte-Carlo):**")
    st.markdown("""
    1. Fit GJR-GARCH(1,1) → extract standardized residuals: $z_t = (r_t - \\mu) / \\sigma_t$
    2. Bootstrap $z^*$ from the empirical distribution of $\\{z_t\\}$ (non-parametric tails)
    3. Propagate through the GJR filter:
    """)
    st.latex(r"\sigma_{t+1}^{*2} = \omega + (\alpha + \gamma \cdot \mathbb{1}(\varepsilon_t^* < 0)) \cdot \varepsilon_t^{*2} + \beta \cdot \sigma_t^2")
    st.latex(r"r_{t+1}^* = \mu + \sigma_{t+1}^* \cdot z^*")
    st.markdown("""
    4. Repeat N times (e.g., 10,000) to build a simulated return distribution
    5. VaR = empirical quantile of {r*}; ES = mean of exceedances
    """)
    st.markdown(
        "This is a single unified model: GJR-GARCH provides the volatility dynamics (clustering + leverage), "
        "while the bootstrap avoids assuming any parametric distribution for the tails. "
        "The MC simulation approximates the full predictive return distribution."
    )

    st.divider()
    st.markdown("### Statistical Tests")

    st.markdown("**Kupiec (1995) — Unconditional Coverage:**")
    st.latex(r"LR_{uc} = -2 \left[ \ln L(p_0) - \ln L(\hat{p}) \right] \sim \chi^2(1)")
    st.markdown("where $p_0 = 1 - \\alpha$ and $\\hat{p} = N_{\\text{exc}} / N_{\\text{total}}$")

    st.markdown("**Christoffersen (1998) — Independence:**")
    st.latex(r"LR_{ind} = -2 \left[ \ln L(\hat{\pi}) - \ln L(\{\pi_{ij}\}) \right] \sim \chi^2(1)")
    st.markdown("Tests if exceedances cluster (first-order Markov transition matrix).")

    st.markdown("**Combined Conditional Coverage:**")
    st.latex(r"LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2(2)")
    st.markdown("Joint test: correct level AND independence.")

    st.divider()
    st.markdown("### References")
    st.markdown("""
    - Glosten, L., Jagannathan, R., Runkle, D. (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance*, 48(5).
    - Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives*, 3(2).
    - Christoffersen, P. (1998). "Evaluating Interval Forecasts." *International Economic Review*, 39(4).
    - Barone-Adesi, G., Giannopoulos, K., Vosper, L. (1999). "VaR without Correlations for Portfolios of Derivative Securities." *Journal of Futures Markets*, 19(5).
    """)
