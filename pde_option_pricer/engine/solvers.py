"""PDE solvers for option pricing.

Four solvers ported from the research notebook:
1. Crank-Nicolson (log-space) — European / American
2. Implicit FD with local vol — Barrier options
3. Free-boundary extraction — American exercise boundary
4. Direct tridiagonal solver — Discrete dividends
"""

import time

import numpy as np
from scipy.linalg import solve_banded

from .models import (
    GridConfig,
    PricingResult,
    BarrierResult,
    FreeBoundary,
    DividendResult,
)


# ---------------------------------------------------------------------------
# 1. Crank-Nicolson — European & American
# ---------------------------------------------------------------------------

def price_european_american(
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
    option_type: str = "call",
    option_style: str = "european",
    grid: GridConfig | None = None,
) -> PricingResult:
    """Price European or American options with Crank-Nicolson in log-space.

    Uses the substitution x = log(S) to obtain a uniform grid, then
    solves the resulting tridiagonal system A·V^{n} = B·V^{n+1} at each
    time step.  For American options the early-exercise projection
    V = max(V, payoff) is applied after each solve.
    """
    t0 = time.perf_counter()
    grid = grid or GridConfig()
    N, M = grid.N, grid.M

    # log-space concentrates grid points near strike where accuracy matters most
    x_max = np.log(K) + 5 * sigma * np.sqrt(T)
    x_min = np.log(K) - 5 * sigma * np.sqrt(T)
    x = np.linspace(x_min, x_max, N)
    dt = T / M
    dx = (x_max - x_min) / (N - 1)
    S = np.exp(x)

    # Crank-Nicolson coefficients
    nu = r - 0.5 * sigma**2
    alpha = 0.25 * dt * ((sigma**2 / dx**2) - (nu / dx))
    beta = -0.5 * dt * (sigma**2 / dx**2 + r)
    gamma = 0.25 * dt * ((sigma**2 / dx**2) + (nu / dx))

    # Implicit matrix A in banded storage for solve_banded((1, 1), ...)
    # ab[0] = upper diagonal, ab[1] = main diagonal, ab[2] = lower diagonal
    A_band = np.zeros((3, N))
    A_band[0, 2:] = -gamma
    A_band[1, 0] = 1.0
    A_band[1, 1:-1] = 1 - beta
    A_band[1, -1] = 1.0
    A_band[2, :-2] = -alpha

    # Terminal payoff
    if option_type == "call":
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # Backward time stepping
    for j in range(M - 1, -1, -1):
        # Explicit side: B @ V as vectorised tridiagonal multiply
        B_V = np.empty(N)
        B_V[0] = V[0]
        B_V[-1] = V[-1]
        B_V[1:-1] = alpha * V[:-2] + (1 + beta) * V[1:-1] + gamma * V[2:]

        t_current = j * dt
        if option_type == "call":
            B_V[0] = 0.0
            B_V[-1] = np.exp(x_max) - K * np.exp(-r * (T - t_current))
        else:
            B_V[0] = K * np.exp(-r * (T - t_current))
            B_V[-1] = 0.0

        V = solve_banded((1, 1), A_band, B_V)

        # Early-exercise projection for American options
        if option_style == "american":
            if option_type == "call":
                V = np.maximum(V, np.maximum(S - K, 0.0))
            else:
                V = np.maximum(V, np.maximum(K - S, 0.0))

    elapsed = time.perf_counter() - t0
    price = float(np.interp(K, S, V))
    return PricingResult(
        price=price,
        S_grid=S,
        V_grid=V,
        option_type=option_type,
        option_style=option_style,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# 2. Implicit FD with local volatility — Barrier option
# ---------------------------------------------------------------------------

def price_barrier_local_vol(
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma_atm: float = 0.20,
    alpha: float = 0.4,
    B: float = 130.0,
    grid: GridConfig | None = None,
) -> BarrierResult:
    """Price an up-and-out call with local vol sigma(S) = sigma_atm * (S/K)^{-alpha}.

    Uses an implicit finite-difference scheme on a uniform S-grid.
    The barrier condition V = 0 for S >= B is enforced at every time step.
    """
    t0 = time.perf_counter()
    grid = grid or GridConfig()
    N, M = grid.N, grid.M

    S_max = B * 1.1
    S_min = 1e-6
    S = np.linspace(S_min, S_max, N)
    dt = T / M
    dS = (S_max - S_min) / (N - 1)

    # Local vol
    vol_grid = sigma_atm * (S / K) ** (-alpha)

    # Terminal payoff with barrier
    V = np.maximum(S - K, 0.0)
    V[S >= B] = 0.0

    # FD coefficients
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)

    drift = r * S[1:-1]
    diffusion = 0.5 * (vol_grid[1:-1] ** 2) * (S[1:-1] ** 2)
    a[1:-1] = dt * (drift / (2 * dS) - diffusion / (dS**2))
    b[1:-1] = 1 + dt * (2 * diffusion / (dS**2) + r)
    c[1:-1] = dt * (-drift / (2 * dS) - diffusion / (dS**2))

    # Boundary diagonal
    b[0] = 1.0
    b[-1] = 1.0

    # A matrix in banded storage
    A_band = np.zeros((3, N))
    A_band[0, 1:] = c[:-1]
    A_band[1, :] = b
    A_band[2, :-1] = a[1:]

    # Forward in time (implicit backward Euler)
    for _ in range(M):
        rhs = V.copy()
        rhs[0], rhs[-1] = 0.0, 0.0
        V = solve_banded((1, 1), A_band, rhs)
        V[S >= B] = 0.0

    elapsed = time.perf_counter() - t0
    price = float(np.interp(100.0, S, V))
    return BarrierResult(
        price=price,
        S_grid=S,
        V_grid=V,
        vol_grid=vol_grid,
        barrier=B,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# 3. Free-boundary extraction — American put/call
# ---------------------------------------------------------------------------

def extract_free_boundary(
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
    option_type: str = "put",
    grid: GridConfig | None = None,
) -> FreeBoundary:
    """Extract the early-exercise boundary S*(t) for an American option.

    Uses Crank-Nicolson in log-space with early-exercise projection.
    At each time step the boundary is identified as the last grid point
    where V equals the exercise value (within tolerance).
    """
    t0 = time.perf_counter()
    grid = grid or GridConfig(N=200, M=200)
    N, M = grid.N, grid.M

    x_max = np.log(K) + 5 * sigma * np.sqrt(T)
    x_min = np.log(K) - 5 * sigma * np.sqrt(T)
    x = np.linspace(x_min, x_max, N)
    dt = T / M
    dx = (x_max - x_min) / (N - 1)
    S = np.exp(x)

    nu = r - 0.5 * sigma**2
    a_coeff = 0.25 * dt * ((sigma**2 / dx**2) - (nu / dx))
    b_coeff = -0.5 * dt * (sigma**2 / dx**2 + r)
    g_coeff = 0.25 * dt * ((sigma**2 / dx**2) + (nu / dx))

    # Implicit matrix A in banded storage
    A_band = np.zeros((3, N))
    A_band[0, 2:] = -g_coeff
    A_band[1, 0] = 1.0
    A_band[1, 1:-1] = 1 - b_coeff
    A_band[1, -1] = 1.0
    A_band[2, :-2] = -a_coeff

    if option_type == "call":
        V = np.maximum(S - K, 0.0)
        exercise_value = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)
        exercise_value = np.maximum(K - S, 0.0)

    free_boundary = np.zeros(M + 1)
    free_boundary[M] = K

    for j in range(M - 1, -1, -1):
        # Explicit side: B_mat @ V as vectorised tridiagonal multiply
        B_V = np.empty(N)
        B_V[0] = V[0]
        B_V[-1] = V[-1]
        B_V[1:-1] = (a_coeff * V[:-2]
                      + (1 + b_coeff) * V[1:-1]
                      + g_coeff * V[2:])

        t_current = j * dt
        if option_type == "put":
            B_V[0] = K * np.exp(-r * (T - t_current))
            B_V[-1] = 0.0
        else:
            B_V[0] = 0.0
            B_V[-1] = np.exp(x_max) - K * np.exp(-r * (T - t_current))

        V_cont = solve_banded((1, 1), A_band, B_V)
        V = np.maximum(V_cont, exercise_value)

        # Find boundary: where V transitions from intrinsic to continuation.
        diff = V - exercise_value
        has_intrinsic = exercise_value > 1e-10
        exercised = has_intrinsic & (diff < 1e-6 * K)
        if np.any(exercised):
            if option_type == "put":
                idx = np.where(exercised)[0][-1]
            else:
                idx = np.where(exercised)[0][0]
            free_boundary[j] = S[idx]
        else:
            free_boundary[j] = K

    time_grid = np.linspace(0, T, M + 1)
    elapsed = time.perf_counter() - t0

    return FreeBoundary(
        S_grid=S,
        V_grid=V,
        exercise_value=exercise_value,
        time_grid=time_grid,
        boundary=free_boundary,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# 4. Direct tridiagonal solver — American with discrete dividends
# ---------------------------------------------------------------------------

def price_american_dividends_psor(
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.25,
    option_type: str = "call",
    div_amount: float = 5.0,
    div_time: float = 0.48,
    omega: float = 1.5,
    tol: float = 1e-5,
    grid: GridConfig | None = None,
) -> DividendResult:
    """Price an American option with discrete dividends.

    At the ex-dividend date the option values are shifted by interpolating
    V(S - D) onto the original grid.  Uses a direct tridiagonal solve
    with early-exercise projection at each time step.
    """
    t0 = time.perf_counter()
    grid = grid or GridConfig(N=150, M=250)
    N, M = grid.N, grid.M

    S_max = 200.0
    S_min = 1e-5
    S = np.linspace(S_min, S_max, N + 1)
    dt = T / M
    dS = (S_max - S_min) / N

    # Full value grid
    P = N + 1
    V_full = np.zeros((P, M + 1))
    if option_type == "call":
        exercise_value = np.maximum(S - K, 0.0)
    else:
        exercise_value = np.maximum(K - S, 0.0)
    V_full[:, M] = exercise_value

    free_boundary = np.zeros(M + 1)
    div_idx = int(div_time / dt)

    # FD coefficients
    drift = r * S
    diffusion = 0.5 * sigma**2 * S**2
    a = 0.5 * dt * (diffusion / dS**2 - drift / dS)
    b = 1 - dt * (diffusion / dS**2 * 2 + r)
    c = 0.5 * dt * (diffusion / dS**2 + drift / dS)

    # The PSOR iteration solves:
    #   V[i] = a[i]*V[i-1] + b[i]*V_old[i] + c[i]*V[i+1]
    # Rearranged as tridiagonal system:
    #   V[i] - a[i]*V[i-1] - c[i]*V[i+1] = b[i]*V_old[i]
    A_band = np.zeros((3, P))
    A_band[1, :] = 1.0              # main diagonal
    A_band[0, 2:] = -c[1:N]         # upper diagonal (interior)
    A_band[2, :N-1] = -a[1:N]       # lower diagonal (interior)

    for j in range(M - 1, -1, -1):
        V_old = V_full[:, j + 1].copy()

        # Dividend adjustment at ex-date
        if j + 1 == div_idx:
            S_after_div = S - div_amount
            V_old = np.interp(S_after_div, S, V_old)

        # RHS: boundary values preserved, interior uses FD relation
        rhs = np.empty(P)
        rhs[0] = V_old[0]
        rhs[1:N] = b[1:N] * V_old[1:N]
        rhs[N] = V_old[N]

        # Direct tridiagonal solve + early-exercise projection
        V_new = solve_banded((1, 1), A_band, rhs)
        V_new = np.maximum(V_new, exercise_value)

        V_full[:, j] = V_new

        # Track free boundary
        try:
            idx = np.where(np.abs(V_new - exercise_value) < tol)[0][0]
            free_boundary[j] = S[idx]
        except IndexError:
            free_boundary[j] = S_max

    elapsed = time.perf_counter() - t0
    price = float(np.interp(K, S, V_full[:, 0]))
    time_grid = np.linspace(0, T, M + 1)

    return DividendResult(
        price=price,
        S_grid=S,
        V_grid=V_full[:, 0],
        V_full=V_full,
        free_boundary=free_boundary,
        time_grid=time_grid,
        iterations=M,
        elapsed=elapsed,
    )
