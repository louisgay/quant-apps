"""HRP allocation engine (López de Prado 2016)."""

from .data import fetch_market_data, estimate_covariance, ledoit_wolf_shrinkage
from .clustering import (
    correlation_distance,
    hierarchical_cluster,
    quasi_diagonalize,
    get_cluster_members,
)
from .hrp import hrp_allocation, recursive_bisection
from .optimizers import (
    mean_variance_optimize,
    risk_parity_optimize,
    equal_weight,
    inverse_volatility,
)
from .analytics import (
    Backtester,
    PortfolioMetrics,
    compute_metrics,
    effective_number_of_bets,
)

__all__ = [
    "fetch_market_data",
    "estimate_covariance",
    "ledoit_wolf_shrinkage",
    "correlation_distance",
    "hierarchical_cluster",
    "quasi_diagonalize",
    "get_cluster_members",
    "hrp_allocation",
    "recursive_bisection",
    "mean_variance_optimize",
    "risk_parity_optimize",
    "equal_weight",
    "inverse_volatility",
    "Backtester",
    "PortfolioMetrics",
    "compute_metrics",
    "effective_number_of_bets",
]
