# Quant Apps

Streamlit applications and engines covering derivatives pricing, volatility modeling, portfolio allocation, and risk management. Each project is self-contained with its own engine, tests, and requirements.

---

## Applications

### 1. [Structured Product Factory](structured_product_factory/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://structured-pricing-factory.streamlit.app/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/structured_product_factory/notebook.ipynb)

Monte Carlo pricing engine for Autocallable Phoenix & Athena notes on multi-asset worst-of baskets. Greeks via central FD, correlation P&L analysis with eigenvalue-floored PSD projection.

### 2. [Volatility Surface Calibrator](vol_surface_calibrator/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vol-surface-calibrator.streamlit.app/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/vol_surface_calibrator/notebook.ipynb)

SVI volatility surface calibration on live Yahoo Finance option chains. Brent IV solver, SLSQP + differential evolution optimizer, no-arbitrage enforcement, 3D Plotly surface.

### 3. [Portfolio Optimizer](portfolio_optimizer/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-portfolio-optimizer.streamlit.app/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/portfolio_optimizer/notebook.ipynb)

Markowitz, Risk Parity, and Black-Litterman allocation with Ledoit-Wolf covariance shrinkage and Fama-French 6-factor expected returns. Walk-forward backtesting with periodic rebalancing.

### 4. [PDE Option Pricer](pde_option_pricer/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pde-option-pricer.streamlit.app/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/pde_option_pricer/notebook.ipynb)

Finite-difference PDE solvers for European, American, barrier, and dividend-paying options. Crank-Nicolson in log-space, local vol, free-boundary extraction, PSOR with discrete dividends.

### 5. [Options Strategy Simulator](options_strategy_simulator/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://options-strategy-simulator.streamlit.app/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/options_strategy_simulator/notebook.ipynb)

Multi-leg option strategy builder with SVI volatility smile, full Greeks (delta through volga), and Monte Carlo simulation with Student-t fat tails. Payoff diagrams, P&L surfaces, VaR/CVaR, and a VRP strategy comparison app.

### 6. [Put-Call Parity Arb Screener](pcp_arb_screener/)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/pcp_arb_screener/notebook.ipynb)

Real-time screener scanning live option chains across equities, commodity ETFs, and FX ETFs for put-call parity violations. Checks profitability after bid-ask spreads and broker fees (IBKR vs Alpaca side-by-side).

### 7. [HRP Allocation](hrp_allocation/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hrp-allocation.streamlit.app/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/hrp_allocation/notebook.ipynb)

Hierarchical Risk Parity (López de Prado 2016) — ML-based allocation using Ward clustering and recursive bisection. Avoids covariance matrix inversion entirely. Rolling backtests against MVO, Risk Parity, and 1/N.

### 8. [VaR Risk Models](var_risk_models/)

Comparison of 6 Value-at-Risk methodologies: historical simulation, parametric normal, Student-t, EWMA, GARCH(1,1), and Monte-Carlo GJR-GARCH Filtered Historical Simulation (FHS). Kupiec and Christoffersen backtesting.

### 9. [GARCH VaR Backtest](garch_var_backtest/)

GARCH(1,1) VaR backtester with Kupiec/Christoffersen LR tests and comparison against naive benchmarks (historical sim, parametric normal, EWMA).

### 10. [Heston FFT Pricer](heston_fft_pricer/)

European option pricing under the Heston stochastic volatility model via Carr-Madan FFT. Calibration to market smiles and comparison against Black-Scholes.

### 11. [Options Mispricing Scanner](options_mispricing_scanner/)

Detects pricing inefficiencies by calibrating an SVI surface and comparing model-implied IVs to market. Checks put-call parity, butterfly, and calendar spread arbitrage. Ranks by edge-to-spread ratio.

### 12. [Adaptive Kalman Filter](adaptive_kalman/)

Standard Kalman filter vs adaptive variant that detects market regime shifts and inflates process noise for faster re-adaptation. Interactive comparison dashboard.

### 13. [Kalman-OU System](kalman_ou_system/)

Mean-reversion trading using a Kalman filter on a discrete Ornstein-Uhlenbeck state-space model. Entry/exit signals from filtered spread z-scores.

### 14. [IV Regime Split](iv_regime_split/)

VIX mean-reversion regime classification (high-IV / low-IV days) and regime-conditional SPY return analysis.

---

## Quick Start

```bash
# Run any app locally
cd options_strategy_simulator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# Or with Docker (SPF, VSC, PO have Dockerfiles)
cd vol_surface_calibrator && docker compose up --build

# Run tests (per project)
cd hrp_allocation && pytest tests/ -v
```

See each app's README for details.

---

## License

MIT
