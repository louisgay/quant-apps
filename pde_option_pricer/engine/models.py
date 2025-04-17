"""Dataclasses for PDE option pricing results."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GridConfig:
    """Numerical grid parameters for PDE solvers."""
    N: int = 500
    M: int = 500


@dataclass
class PricingResult:
    """Result from a European/American PDE solver."""
    price: float
    S_grid: np.ndarray
    V_grid: np.ndarray
    option_type: str
    option_style: str
    elapsed: float = 0.0


@dataclass
class SurfaceData:
    """3D price surface for Plotly visualisation."""
    S_values: np.ndarray
    T_values: np.ndarray
    V_surface: np.ndarray


@dataclass
class BarrierResult:
    """Result from barrier option with local vol."""
    price: float
    S_grid: np.ndarray
    V_grid: np.ndarray
    vol_grid: np.ndarray
    barrier: float
    elapsed: float = 0.0


@dataclass
class FreeBoundary:
    """Free boundary extraction for American options."""
    S_grid: np.ndarray
    V_grid: np.ndarray
    exercise_value: np.ndarray
    time_grid: np.ndarray
    boundary: np.ndarray
    elapsed: float = 0.0


@dataclass
class DividendResult:
    """PSOR result with discrete dividends."""
    price: float
    S_grid: np.ndarray
    V_grid: np.ndarray
    V_full: np.ndarray
    free_boundary: np.ndarray
    time_grid: np.ndarray
    iterations: int
    elapsed: float = 0.0
