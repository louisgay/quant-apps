"""Hierarchical clustering and seriation (Steps 1-2 of HRP)."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def correlation_distance(corr: pd.DataFrame | np.ndarray) -> np.ndarray:
    """d(i, j) = √((1 - ρ_ij) / 2). Maps ρ ∈ [-1,1] → d ∈ [0,1]."""
    if isinstance(corr, pd.DataFrame):
        corr = corr.values

    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt((1.0 - corr) / 2.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def hierarchical_cluster(
    dist_matrix: np.ndarray,
    method: str = "ward",
) -> np.ndarray:
    """Agglomerative clustering on a distance matrix.

    Returns scipy linkage matrix: (N-1) x 4, each row = [idx1, idx2, dist, count].
    """
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=method)
    return Z


def quasi_diagonalize(
    cov: pd.DataFrame | np.ndarray,
    linkage_matrix: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Reorder covariance by dendrogram leaf ordering (permutation similarity P^T Σ P)."""
    if isinstance(cov, pd.DataFrame):
        cov_values = cov.values
    else:
        cov_values = cov

    sort_order = list(leaves_list(linkage_matrix).astype(int))
    cov_reordered = cov_values[np.ix_(sort_order, sort_order)]

    return cov_reordered, sort_order


def get_cluster_members(
    linkage_matrix: np.ndarray,
    n_assets: int,
    node_id: int,
) -> list[int]:
    """Recursively get all leaf indices under a dendrogram node."""
    if node_id < n_assets:
        return [node_id]

    row = int(node_id - n_assets)
    left_child = int(linkage_matrix[row, 0])
    right_child = int(linkage_matrix[row, 1])

    return (get_cluster_members(linkage_matrix, n_assets, left_child)
            + get_cluster_members(linkage_matrix, n_assets, right_child))


def get_children(
    linkage_matrix: np.ndarray,
    n_assets: int,
    node_id: int,
) -> tuple[list[int], list[int]]:
    """Get (left_members, right_members) for an internal node."""
    if node_id < n_assets:
        raise ValueError(f"Node {node_id} is a leaf, not an internal node")

    row = int(node_id - n_assets)
    left_child = int(linkage_matrix[row, 0])
    right_child = int(linkage_matrix[row, 1])

    left_members = get_cluster_members(linkage_matrix, n_assets, left_child)
    right_members = get_cluster_members(linkage_matrix, n_assets, right_child)

    return left_members, right_members
