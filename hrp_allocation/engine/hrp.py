"""Core HRP: recursive bisection (Step 3 of López de Prado 2016)."""

import numpy as np
import pandas as pd

from .clustering import (
    correlation_distance,
    hierarchical_cluster,
    get_children,
)


def hrp_allocation(
    cov: pd.DataFrame,
    corr: pd.DataFrame | None = None,
    linkage_method: str = "ward",
) -> pd.Series:
    """Full HRP pipeline: distance → cluster → seriate → bisect.

    Returns pd.Series of weights (all >= 0, sum = 1).
    """
    assets = cov.index.tolist()
    n = len(assets)

    if corr is None:
        std = np.sqrt(np.diag(cov.values))
        corr_values = cov.values / np.outer(std, std)
        np.fill_diagonal(corr_values, 1.0)
        corr = pd.DataFrame(corr_values, index=assets, columns=assets)

    dist = correlation_distance(corr)
    linkage_matrix = hierarchical_cluster(dist, method=linkage_method)
    weights_array = recursive_bisection(cov.values, linkage_matrix, n)

    return pd.Series(weights_array, index=assets)


def recursive_bisection(
    cov: np.ndarray,
    linkage_matrix: np.ndarray,
    n_assets: int,
) -> np.ndarray:
    """Top-down capital allocation through the dendrogram.

    At each split: α_L = (1/V_L) / (1/V_L + 1/V_R), where V is the
    cluster variance under inverse-variance weights.
    """
    weights = np.ones(n_assets)
    root = 2 * n_assets - 2

    # BFS through the tree
    nodes_to_process = [(root, 1.0)]

    while nodes_to_process:
        node_id, alpha = nodes_to_process.pop(0)

        if node_id < n_assets:
            weights[node_id] = alpha
            continue

        left_members, right_members = get_children(
            linkage_matrix, n_assets, node_id
        )

        v_left = _cluster_variance(cov, left_members)
        v_right = _cluster_variance(cov, right_members)

        alloc_left = (1.0 / v_left) / (1.0 / v_left + 1.0 / v_right)
        alloc_right = 1.0 - alloc_left

        row = int(node_id - n_assets)
        left_child = int(linkage_matrix[row, 0])
        right_child = int(linkage_matrix[row, 1])

        nodes_to_process.append((left_child, alpha * alloc_left))
        nodes_to_process.append((right_child, alpha * alloc_right))

    return weights


def _cluster_variance(cov: np.ndarray, members: list[int]) -> float:
    """Cluster variance using inverse-variance weights (w_i ∝ 1/σ²_i)."""
    cov_cluster = cov[np.ix_(members, members)]
    variances = np.maximum(np.diag(cov_cluster), 1e-10)
    inv_var = 1.0 / variances
    w = inv_var / inv_var.sum()
    return max(w @ cov_cluster @ w, 1e-10)
