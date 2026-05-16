# HRP Allocation

> Hierarchical Risk Parity (López de Prado 2016) — ML-based portfolio allocation that never inverts the covariance matrix. Rolling backtests against Markowitz, Risk Parity, Equal Weight, and Inverse Volatility.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hrp-allocation.streamlit.app/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisgay/quant-apps/blob/main/hrp_allocation/notebook.ipynb)

---

## Quick Start

```bash
# Docker
docker compose up           # localhost:8501
docker compose --profile test up  # tests

# Local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
pytest tests/ -v
```

## Data Source

**Alpaca Markets** (primary, free tier, split/dividend-adjusted) with **yfinance** fallback:

```bash
export ALPACA_API_KEY="your-key"
export ALPACA_SECRET_KEY="your-secret"
```

No credentials → uses yfinance transparently.

---

## Algorithm

Correlation distance metric:

$$d(i,j) = \sqrt{\frac{1 - \rho_{i,j}}{2}}$$

Maps $\rho \in [-1,1]$ to $d \in [0,1]$, satisfies the triangle inequality. Then Ward's agglomerative clustering builds a dendrogram.

After seriation (reorder $\Sigma$ by dendrogram leaves → block-diagonal structure), the recursive bisection allocates top-down:

$$\alpha_L = \frac{1/V_L}{1/V_L + 1/V_R}, \quad \alpha_R = 1 - \alpha_L$$

where $V_L, V_R$ are cluster variances under inverse-variance weights ($\tilde{w}_i = (1/\sigma_i^2) / \sum_j (1/\sigma_j^2)$).

### Why not just Markowitz?

MVO requires $\Sigma^{-1}$. With 7 correlated assets over 2 years of data, $\kappa(\Sigma) \approx 300\text{-}600$. That means a 1% error in estimating $\Sigma$ produces up to 6% swings in optimal weights. In practice it's an "error maximizer" (Michaud 1989).

HRP sidesteps this entirely — no inversion, weights are Lipschitz-continuous in $\Sigma$.

---

## Architecture

```
hrp_allocation/
├── engine/
│   ├── data.py             # Alpaca/yfinance fetch + Ledoit-Wolf shrinkage
│   ├── clustering.py       # Distance, Ward clustering, seriation
│   ├── hrp.py              # Recursive bisection (core algorithm)
│   ├── optimizers.py       # Benchmarks: MVO, Risk Parity, EW, InvVol
│   └── analytics.py        # Backtester + metrics (Sharpe, drawdown, turnover)
├── tests/
│   └── test_engine.py      # 30 tests on synthetic covariance matrices
├── app.py                  # Streamlit dashboard
├── notebook.ipynb          # Walkthrough + Monte Carlo stability test
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Notes

HRP is a **risk allocator**, not a return maximizer. In a straight bull market (2018–2025), Equal Weight beats HRP on raw return because HRP deliberately limits exposure to the correlated equity cluster (SPY/QQQ/EFA get 2-5% each vs 14% in EW). The trade-off is lower drawdowns: COVID crash and 2022 rate hiking both showed 2-4% less drawdown for HRP.

The comparison that matters is HRP vs MVO, not HRP vs 1/N. DeMiguel et al. (2009) showed that 1/N is notoriously hard to beat out-of-sample with any optimization — that's a broader statement about estimation error, not a weakness of HRP specifically.

Ward linkage over single/complete: Ward minimizes intra-cluster variance, producing balanced dendrograms. Single linkage creates chain effects (one asset at a time joins the cluster), which gives degenerate bisections. Complete linkage can artificially split natural clusters.

Ledoit-Wolf with scaled identity target makes a noticeable difference — without it, the recursive bisection overweights one cluster because the raw sample covariance has an eigenvalue near zero that distorts the inverse-variance calculation within clusters.

---

## Test Suite

```bash
pytest tests/ -v
```

30 tests on synthetic covariance matrices (no API calls): weight constraints, dendrogram consistency, seriation eigenvalue preservation, identity cov → equal weight, block-diagonal cluster detection, Ledoit-Wolf PSD guarantee, backtester no-lookahead.

---

## References

- López de Prado (2016). "Building Diversified Portfolios that Outperform Out of Sample." JPM 42(4).
- Ledoit & Wolf (2004). "A well-conditioned estimator for large-dimensional covariance matrices." JMVA 88(2).
- DeMiguel, Garlappi, Uppal (2009). "Optimal Versus Naive Diversification." RFS 22(5).
- Michaud (1989). "The Markowitz Optimization Enigma." FAJ 45(1).

---

## License

MIT
