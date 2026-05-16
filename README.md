# Quant Apps

Streamlit applications covering structured product pricing, volatility surface calibration, portfolio allocation, PDE option pricing, options strategy simulation, and more. Each app is self-contained with its own engine, tests, and requirements.

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

### 6. [HRP Allocation](hrp_allocation/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hrp-allocation.streamlit.app/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/hrp_allocation/notebook.ipynb)

Hierarchical Risk Parity (López de Prado 2016) — ML-based allocation using Ward clustering and recursive bisection. Avoids covariance matrix inversion entirely. Rolling backtests against MVO, Risk Parity, and 1/N.

### 7. [VaR Risk Models](var_risk_models/)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/var_risk_models/research.ipynb)

6 VaR methods compared — from empirical quantile to GJR-GARCH Filtered Historical Simulation (Barone-Adesi 1999). Rolling backtest with Kupiec unconditional coverage, Christoffersen independence, and combined conditional coverage tests.

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

# Run all tests
pytest structured_product_factory/tests/ \
       vol_surface_calibrator/tests/ \
       portfolio_optimizer/tests/ \
       pde_option_pricer/tests/ \
       options_strategy_simulator/tests/ \
       hrp_allocation/tests/ \
       var_risk_models/tests/ -v
```

See each app's README for details.

---

## License

MIT
