# Quant Apps

Four Streamlit applications covering structured product pricing, volatility surface calibration, portfolio optimization, and PDE option pricing. Each app is self-contained with its own engine, tests, and requirements.

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

---

## Quick Start

```bash
# Run any app locally
cd structured_product_factory
pip install -r requirements.txt
streamlit run app.py

# Or with Docker (SPF, VSC, PO have Dockerfiles)
cd vol_surface_calibrator && docker compose up --build

# Run all tests
pytest structured_product_factory/tests/ vol_surface_calibrator/tests/ portfolio_optimizer/tests/ pde_option_pricer/tests/ -v
```

See each app's README for details.

---

## License

MIT
