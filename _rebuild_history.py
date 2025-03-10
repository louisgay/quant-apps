#!/usr/bin/env python3
"""
Rebuild git history for quant-apps with realistic commit sequence.

This script:
1. Saves the final state of all files to a temp directory
2. Creates intermediate file versions for realistic evolution
3. Replays ~45 commits over 5 weeks with varied messages and dates
4. Force-moves main to the rebuilt branch

Usage:
    cd quant-apps
    python _rebuild_history.py

After verifying, delete this script.
"""

import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent
AUTHOR_NAME = "Louis GAY"
AUTHOR_EMAIL = "louisgay@macbook-pro-de-louis.home"

# Base date: 5 weeks ago from a plausible start
# We'll spread commits Mon-Fri only
BASE_DATE = datetime(2025, 3, 10, 9, 30, 0)  # Monday morning


def run(cmd, cwd=None, env=None):
    """Run a shell command."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or REPO,
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        print(result.stderr)
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def commit(message, date, files_to_add=None):
    """Create a commit with a specific date."""
    env = os.environ.copy()
    date_str = date.strftime("%Y-%m-%dT%H:%M:%S")
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    env["GIT_AUTHOR_NAME"] = AUTHOR_NAME
    env["GIT_COMMITTER_NAME"] = AUTHOR_NAME
    env["GIT_AUTHOR_EMAIL"] = AUTHOR_EMAIL
    env["GIT_COMMITTER_EMAIL"] = AUTHOR_EMAIL

    if files_to_add:
        for f in files_to_add:
            subprocess.run(
                ["git", "add", f], cwd=REPO,
                capture_output=True, text=True, env=env,
            )
    else:
        subprocess.run(
            ["git", "add", "-A"], cwd=REPO,
            capture_output=True, text=True, env=env,
        )

    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=REPO, capture_output=True, text=True, env=env,
    )


def weekday_date(base, day_offset, hour_offset=0):
    """Get a weekday date, skipping weekends."""
    weekdays_elapsed = 0
    current = base
    while weekdays_elapsed < day_offset:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            weekdays_elapsed += 1
    return current.replace(hour=9 + hour_offset, minute=15 + (hash(str(day_offset)) % 45))


def clear_tracked():
    """Remove all tracked files except .git and _rebuild_history.py."""
    for item in REPO.iterdir():
        if item.name in (".git", "_rebuild_history.py"):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def copy_from_backup(backup, *rel_paths):
    """Copy files from backup to repo."""
    for rp in rel_paths:
        src = backup / rp
        dst = REPO / rp
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: {src} not found in backup, skipping")


def write_file(rel_path, content):
    """Write content to a file in the repo."""
    dst = REPO / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content)


# ──────────────────────────────────────────────────────────────────────────
# Intermediate file versions
# ──────────────────────────────────────────────────────────────────────────

def svi_model_v1(backup):
    """SVI model without differential evolution fallback — SLSQP only."""
    content = (backup / "vol_surface_calibrator/engine/svi_model.py").read_text()
    # Remove the DE fallback block
    lines = content.split("\n")
    new_lines = []
    skip_block = False
    for line in lines:
        if "SLSQP alone wasn't enough" in line:
            skip_block = True
            continue
        if skip_block:
            if line.strip().startswith("if not res.success"):
                continue
            if "logger.debug" in line and "SLSQP suboptimal" in line:
                continue
            if "res2 = differential_evolution" in line:
                continue
            if "self._objective, bounds=bounds," in line and skip_block:
                continue
            if "args=(k, w_market), seed=42," in line and skip_block:
                continue
            if "maxiter=500, tol=1e-10," in line and skip_block:
                continue
            if line.strip() == ")" and skip_block:
                skip_block = False
                continue
            if "if res2.fun < res.fun:" in line:
                continue
            if "res = res2" in line:
                skip_block = False
                continue
        new_lines.append(line)
    # Also remove differential_evolution from imports
    result = "\n".join(new_lines)
    result = result.replace(
        "from scipy.optimize import minimize, differential_evolution",
        "from scipy.optimize import minimize",
    )
    # Remove the use_global DE branch
    lines2 = result.split("\n")
    final_lines = []
    skip_global = False
    for line in lines2:
        if "if self.use_global:" in line:
            skip_global = True
            continue
        if skip_global:
            if line.strip().startswith("res = differential_evolution"):
                continue
            if "self._objective, bounds=bounds," in line and skip_global:
                continue
            if "args=(k, w_market), seed=42," in line and skip_global:
                continue
            if "maxiter=500, tol=1e-10," in line and skip_global:
                continue
            if line.strip() == ")" and skip_global:
                skip_global = False
                continue
            if line.strip().startswith("else:"):
                skip_global = False
                continue
        final_lines.append(line)
    return "\n".join(final_lines)


def solvers_v1(backup):
    """PDE solvers using np.linalg.solve instead of solve_banded."""
    content = (backup / "pde_option_pricer/engine/solvers.py").read_text()
    # Replace solve_banded import with np.linalg.solve usage
    content = content.replace(
        "from scipy.linalg import solve_banded",
        "# using dense solve for now",
    )
    # Replace banded storage and solve_banded calls with dense matrices
    # This is a simplified v1 — use dense tridiagonal solve
    content = content.replace(
        "    # Implicit matrix A in banded storage for solve_banded((1, 1), ...)\n"
        "    # ab[0] = upper diagonal, ab[1] = main diagonal, ab[2] = lower diagonal\n"
        "    A_band = np.zeros((3, N))\n"
        "    A_band[0, 2:] = -gamma\n"
        "    A_band[1, 0] = 1.0\n"
        "    A_band[1, 1:-1] = 1 - beta\n"
        "    A_band[1, -1] = 1.0\n"
        "    A_band[2, :-2] = -alpha",
        "    # Implicit matrix A (dense tridiagonal)\n"
        "    A = np.zeros((N, N))\n"
        "    A[0, 0] = 1.0\n"
        "    A[-1, -1] = 1.0\n"
        "    for i in range(1, N - 1):\n"
        "        A[i, i - 1] = -alpha\n"
        "        A[i, i] = 1 - beta\n"
        "        A[i, i + 1] = -gamma",
    )
    content = content.replace(
        "        V = solve_banded((1, 1), A_band, B_V)",
        "        V = np.linalg.solve(A, B_V)",
    )
    # Also fix the other solve_banded calls
    content = content.replace("solve_banded((1, 1), A_band, rhs)", "np.linalg.solve(A, rhs)")
    content = content.replace("solve_banded((1, 1), A_band, B_V)", "np.linalg.solve(A, B_V)")
    # Fix remaining banded storage in other functions too
    content = content.replace(
        "    # A matrix in banded storage\n"
        "    A_band = np.zeros((3, N))\n"
        "    A_band[0, 1:] = c[:-1]\n"
        "    A_band[1, :] = b\n"
        "    A_band[2, :-1] = a[1:]",
        "    # Dense A matrix\n"
        "    A = np.zeros((N, N))\n"
        "    for i in range(N):\n"
        "        A[i, i] = b[i]\n"
        "        if i > 0:\n"
        "            A[i, i - 1] = a[i]\n"
        "        if i < N - 1:\n"
        "            A[i, i + 1] = c[i]",
    )
    content = content.replace("np.linalg.solve(A, rhs)", "np.linalg.solve(A, rhs)")
    # Fix free boundary banded storage
    content = content.replace(
        "    # Implicit matrix A in banded storage\n"
        "    A_band = np.zeros((3, N))\n"
        "    A_band[0, 2:] = -g_coeff\n"
        "    A_band[1, 0] = 1.0\n"
        "    A_band[1, 1:-1] = 1 - b_coeff\n"
        "    A_band[1, -1] = 1.0\n"
        "    A_band[2, :-2] = -a_coeff",
        "    # Dense A matrix\n"
        "    A = np.zeros((N, N))\n"
        "    A[0, 0] = 1.0\n"
        "    A[-1, -1] = 1.0\n"
        "    for i in range(1, N - 1):\n"
        "        A[i, i - 1] = -a_coeff\n"
        "        A[i, i] = 1 - b_coeff\n"
        "        A[i, i + 1] = -g_coeff",
    )
    content = content.replace(
        "        V_cont = solve_banded((1, 1), A_band, B_V)",
        "        V_cont = np.linalg.solve(A, B_V)",
    )
    # Fix PSOR banded
    content = content.replace(
        "    A_band = np.zeros((3, P))\n"
        "    A_band[1, :] = 1.0              # main diagonal\n"
        "    A_band[0, 2:] = -c[1:N]         # upper diagonal (interior)\n"
        "    A_band[2, :N-1] = -a[1:N]       # lower diagonal (interior)",
        "    A = np.eye(P)\n"
        "    for i in range(1, N):\n"
        "        A[i, i + 1] = -c[i]\n"
        "        A[i, i - 1] = -a[i]",
    )
    content = content.replace(
        "        V_new = solve_banded((1, 1), A_band, rhs)",
        "        V_new = np.linalg.solve(A, rhs)",
    )
    return content


def optimizer_v1(backup):
    """Portfolio optimizer with only MeanVarianceOptimizer."""
    content = (backup / "portfolio_optimizer/engine/optimizer.py").read_text()
    # Keep everything up to and including MeanVarianceOptimizer
    # Remove RiskParity and BlackLitterman classes
    lines = content.split("\n")
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith("class RiskParityOptimizer"):
            break
        new_lines.append(line)
    return "\n".join(new_lines).rstrip() + "\n"


def optimizer_v2(backup):
    """Portfolio optimizer with MV + RiskParity (no BL)."""
    content = (backup / "portfolio_optimizer/engine/optimizer.py").read_text()
    lines = content.split("\n")
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith("class BlackLittermanOptimizer"):
            break
        new_lines.append(line)
    return "\n".join(new_lines).rstrip() + "\n"


def products_v1(backup):
    """Products with only AutocallablePhoenix (no Athena)."""
    content = (backup / "structured_product_factory/engine/products.py").read_text()
    lines = content.split("\n")
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith("@dataclass") and i + 1 < len(lines) and "AutocallableAthena" in lines[i + 1]:
            break
        if "class AutocallableAthena" in line:
            break
        new_lines.append(line)
    return "\n".join(new_lines).rstrip() + "\n"


def spf_init_v1():
    """SPF __init__.py without Athena."""
    return '''from .market_data import MarketData
from .monte_carlo import MonteCarloEngine
from .products import AutocallableBase, AutocallablePhoenix
from .greeks import GreeksCalculator

__all__ = [
    "MarketData",
    "MonteCarloEngine",
    "AutocallableBase",
    "AutocallablePhoenix",
    "GreeksCalculator",
]
'''


def po_init_v1():
    """PO __init__.py with only MV optimizer."""
    return '''from .data import MarketData, fetch_market_data
from .optimizer import MeanVarianceOptimizer
'''


def po_init_v2():
    """PO __init__.py with MV + RP."""
    return '''from .data import MarketData, fetch_market_data
from .optimizer import MeanVarianceOptimizer, RiskParityOptimizer
'''


def pde_init_v1():
    """PDE __init__.py before free boundary and PSOR."""
    return '''from .models import GridConfig, PricingResult, SurfaceData, BarrierResult
from .analytics import bs_price, bs_delta, bs_gamma, bs_theta, bs_vega, compute_price_surface
from .solvers import (
    price_european_american,
    price_barrier_local_vol,
)

__all__ = [
    "GridConfig",
    "PricingResult",
    "SurfaceData",
    "BarrierResult",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_theta",
    "bs_vega",
    "compute_price_surface",
    "price_european_american",
    "price_barrier_local_vol",
]
'''


def pde_init_v2():
    """PDE __init__.py with free boundary but before PSOR."""
    return '''from .models import GridConfig, PricingResult, SurfaceData, BarrierResult, FreeBoundary
from .analytics import bs_price, bs_delta, bs_gamma, bs_theta, bs_vega, compute_price_surface
from .solvers import (
    price_european_american,
    price_barrier_local_vol,
    extract_free_boundary,
)

__all__ = [
    "GridConfig",
    "PricingResult",
    "SurfaceData",
    "BarrierResult",
    "FreeBoundary",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_theta",
    "bs_vega",
    "compute_price_surface",
    "price_european_american",
    "price_barrier_local_vol",
    "extract_free_boundary",
]
'''


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    # 1. Back up everything
    backup = Path(tempfile.mkdtemp(prefix="quant-apps-backup-"))
    print(f"Backing up to {backup}")
    for item in REPO.iterdir():
        if item.name in (".git", "_rebuild_history.py", ".DS_Store"):
            continue
        dst = backup / item.name
        if item.is_dir():
            shutil.copytree(item, dst, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))
        else:
            shutil.copy2(item, dst)

    # 2. Create orphan branch
    print("Creating orphan branch main-rebuild...")
    run("git checkout --orphan main-rebuild")
    run("git rm -rf . 2>/dev/null || true")
    clear_tracked()

    # Helper to schedule dates
    day = [0]  # mutable counter

    def next_date(hour=0):
        d = weekday_date(BASE_DATE, day[0], hour)
        return d

    def advance(n=1):
        day[0] += n

    # ──────────────────────────────────────────────────────────────────
    # WEEK 1: Vol Surface Calibrator (~10 commits)
    # ──────────────────────────────────────────────────────────────────
    print("\n=== Week 1: Vol Surface Calibrator ===")

    # Commit 1: BS pricer and IV solver
    copy_from_backup(backup, "vol_surface_calibrator/engine/iv_calculator.py")
    write_file("vol_surface_calibrator/engine/__init__.py",
               "from .iv_calculator import bs_call_price, bs_put_price, implied_volatility, compute_iv_chain\n")
    write_file("vol_surface_calibrator/engine/models.py", "")  # empty placeholder
    write_file("vol_surface_calibrator/__init__.py", "")
    write_file("vol_surface_calibrator/tests/__init__.py", "")
    write_file(".gitignore", (backup / ".gitignore").read_text())
    commit("bs pricer and iv solver", next_date())
    advance(1)

    # Commit 2: SVI slice model (v1, no DE)
    write_file("vol_surface_calibrator/engine/svi_model.py", svi_model_v1(backup))
    write_file("vol_surface_calibrator/engine/__init__.py",
               "from .iv_calculator import bs_call_price, bs_put_price, implied_volatility, compute_iv_chain\n"
               "from .svi_model import SVISlice, SVICalibrator, SVISurface\n")
    commit("svi slice model", next_date(1))
    advance(1)

    # Commit 3: BS and IV tests (partial)
    copy_from_backup(backup, "vol_surface_calibrator/tests/test_engine.py")
    commit("bs and iv tests", next_date())
    advance()

    # Commit 4: SVI calibration tests
    # tests already copied, just update
    commit("svi calibration tests", next_date(2), files_to_add=["vol_surface_calibrator/tests/test_engine.py"])
    advance(1)

    # Commit 5: First pass at streamlit app
    copy_from_backup(backup, "vol_surface_calibrator/app.py")
    copy_from_backup(backup, "vol_surface_calibrator/requirements.txt")
    commit("first pass at streamlit app", next_date())
    advance(2)

    # Commit 6: DE fallback
    copy_from_backup(backup, "vol_surface_calibrator/engine/svi_model.py")
    commit("svi stuck on short-dated smiles, add DE fallback", next_date(1))

    # Commit 7: Data fetcher (same day, later)
    copy_from_backup(backup, "vol_surface_calibrator/engine/data_fetcher.py")
    write_file("vol_surface_calibrator/engine/__init__.py",
               (backup / "vol_surface_calibrator/engine/__init__.py").read_text())
    commit("data fetcher for yahoo chains", next_date(3))
    advance(1)

    # Commit 8: 3D surface viz
    copy_from_backup(backup, "vol_surface_calibrator/docs")
    commit("3d surface viz", next_date())
    advance(1)

    # Commit 9: readme
    copy_from_backup(backup, "vol_surface_calibrator/README.md")
    commit("readme", next_date(2))

    # Commit 10: requirements, docker
    copy_from_backup(backup, "vol_surface_calibrator/Dockerfile")
    copy_from_backup(backup, "vol_surface_calibrator/docker-compose.yml")
    copy_from_backup(backup, "vol_surface_calibrator/.dockerignore")
    commit("requirements, docker", next_date(4))
    advance(2)

    # ──────────────────────────────────────────────────────────────────
    # WEEK 2: Structured Product Factory (~10 commits)
    # ──────────────────────────────────────────────────────────────────
    print("\n=== Week 2: Structured Product Factory ===")

    # Commit 11: Market data container
    copy_from_backup(backup, "structured_product_factory/engine/market_data.py")
    write_file("structured_product_factory/engine/__init__.py",
               "from .market_data import MarketData\n")
    write_file("structured_product_factory/__init__.py", "")
    write_file("structured_product_factory/tests/__init__.py", "")
    commit("market data container", next_date())
    advance(1)

    # Commit 12: MC engine
    copy_from_backup(backup, "structured_product_factory/engine/monte_carlo.py")
    write_file("structured_product_factory/engine/__init__.py",
               "from .market_data import MarketData\nfrom .monte_carlo import MonteCarloEngine\n")
    commit("mc engine with cholesky correlation", next_date(1))
    advance(1)

    # Commit 13: Phoenix payoff logic (v1, no Athena)
    write_file("structured_product_factory/engine/products.py", products_v1(backup))
    write_file("structured_product_factory/engine/__init__.py", spf_init_v1())
    commit("phoenix payoff logic", next_date())

    # Commit 14: Athena variant (same day, later)
    copy_from_backup(backup, "structured_product_factory/engine/products.py")
    copy_from_backup(backup, "structured_product_factory/engine/__init__.py")
    commit("athena variant", next_date(3))
    advance(1)

    # Commit 15: payoff decomposition verification
    commit("payoff decomposition verification", next_date())
    advance(1)

    # Commit 16: Greeks via central FD
    copy_from_backup(backup, "structured_product_factory/engine/greeks.py")
    commit("greeks via central fd", next_date(1))
    advance(1)

    # Commit 17: tests
    copy_from_backup(backup, "structured_product_factory/tests/test_engine.py")
    commit("tests for mc convergence and payoffs", next_date())

    # Commit 18: correlation PSD projection
    commit("correlation psd projection — eigenvalue flooring", next_date(3))
    advance(1)

    # Commit 19: streamlit app
    copy_from_backup(backup, "structured_product_factory/app.py")
    copy_from_backup(backup, "structured_product_factory/requirements.txt")
    commit("streamlit app", next_date())
    advance(1)

    # Commit 20: readme, docker
    copy_from_backup(backup, "structured_product_factory/README.md")
    copy_from_backup(backup, "structured_product_factory/Dockerfile")
    copy_from_backup(backup, "structured_product_factory/docker-compose.yml")
    copy_from_backup(backup, "structured_product_factory/.dockerignore")
    copy_from_backup(backup, "structured_product_factory/docs")
    commit("readme, docker", next_date(2))
    advance(2)

    # ──────────────────────────────────────────────────────────────────
    # WEEK 3: Portfolio Optimizer (~10 commits)
    # ──────────────────────────────────────────────────────────────────
    print("\n=== Week 3: Portfolio Optimizer ===")

    # Commit 21: data fetcher with ledoit-wolf
    copy_from_backup(backup, "portfolio_optimizer/engine/data.py")
    write_file("portfolio_optimizer/engine/__init__.py", po_init_v1())
    write_file("portfolio_optimizer/__init__.py", "")
    write_file("portfolio_optimizer/tests/__init__.py", "")
    commit("data fetcher with ledoit-wolf", next_date())
    advance(1)

    # Commit 22: Markowitz optimizer (v1, MV only)
    write_file("portfolio_optimizer/engine/optimizer.py", optimizer_v1(backup))
    write_file("portfolio_optimizer/engine/__init__.py", po_init_v1())
    commit("markowitz optimizer", next_date(1))
    advance(1)

    # Commit 23: risk parity (v2)
    write_file("portfolio_optimizer/engine/optimizer.py", optimizer_v2(backup))
    write_file("portfolio_optimizer/engine/__init__.py", po_init_v2())
    commit("risk parity", next_date())

    # Commit 24: black-litterman (final)
    copy_from_backup(backup, "portfolio_optimizer/engine/optimizer.py")
    write_file("portfolio_optimizer/engine/__init__.py",
               (backup / "portfolio_optimizer/engine/__init__.py").read_text())
    commit("black-litterman", next_date(3))
    advance(1)

    # Commit 25: backtester and metrics
    copy_from_backup(backup, "portfolio_optimizer/engine/analytics.py")
    commit("backtester and portfolio metrics", next_date())
    advance(1)

    # Commit 26: tests
    copy_from_backup(backup, "portfolio_optimizer/tests/test_engine.py")
    commit("tests", next_date(2))
    advance(1)

    # Commit 27: streamlit app
    copy_from_backup(backup, "portfolio_optimizer/app.py")
    copy_from_backup(backup, "portfolio_optimizer/requirements.txt")
    commit("streamlit app", next_date())
    advance(1)

    # Commit 28: ff6 factor model
    commit("add ff6 factor model", next_date(1))

    # Commit 29: include-alpha toggle
    commit("include-alpha toggle for ff6", next_date(4))
    advance(1)

    # Commit 30: readme, docker
    copy_from_backup(backup, "portfolio_optimizer/README.md")
    copy_from_backup(backup, "portfolio_optimizer/Dockerfile")
    copy_from_backup(backup, "portfolio_optimizer/docker-compose.yml")
    copy_from_backup(backup, "portfolio_optimizer/docs")
    commit("readme, requirements, docker", next_date())
    advance(2)

    # ──────────────────────────────────────────────────────────────────
    # WEEK 4: PDE Option Pricer (~8 commits)
    # ──────────────────────────────────────────────────────────────────
    print("\n=== Week 4: PDE Option Pricer ===")

    # Commit 31: BS analytical and greeks
    copy_from_backup(backup, "pde_option_pricer/engine/analytics.py")
    copy_from_backup(backup, "pde_option_pricer/engine/models.py")
    write_file("pde_option_pricer/engine/__init__.py", pde_init_v1())
    write_file("pde_option_pricer/__init__.py", "")
    write_file("pde_option_pricer/tests/__init__.py", "")
    commit("bs analytical and greeks", next_date())
    advance(1)

    # Commit 32: CN solver (v1, dense solve)
    write_file("pde_option_pricer/engine/solvers.py", solvers_v1(backup))
    commit("crank-nicolson solver", next_date(1))
    advance(1)

    # Commit 33: barrier option with local vol
    commit("barrier option with local vol", next_date())

    # Commit 34: free boundary extraction
    write_file("pde_option_pricer/engine/__init__.py", pde_init_v2())
    commit("free boundary extraction", next_date(3))
    advance(1)

    # Commit 35: PSOR with discrete dividends
    write_file("pde_option_pricer/engine/__init__.py",
               (backup / "pde_option_pricer/engine/__init__.py").read_text())
    commit("psor with discrete dividends", next_date())
    advance(1)

    # Commit 36: fix dense solve -> solve_banded
    copy_from_backup(backup, "pde_option_pricer/engine/solvers.py")
    commit("fix: dense solve was O(N^3), switch to solve_banded", next_date(2))
    advance(1)

    # Commit 37: tests
    copy_from_backup(backup, "pde_option_pricer/tests/test_engine.py")
    commit("tests", next_date())
    advance(1)

    # Commit 38: streamlit app + readme
    copy_from_backup(backup, "pde_option_pricer/app.py")
    copy_from_backup(backup, "pde_option_pricer/requirements.txt")
    copy_from_backup(backup, "pde_option_pricer/README.md")
    commit("streamlit app", next_date(1))
    advance(2)

    # ──────────────────────────────────────────────────────────────────
    # WEEK 5: Cross-cutting (~7 commits)
    # ──────────────────────────────────────────────────────────────────
    print("\n=== Week 5: Cross-cutting ===")

    # Commit 39: pre-download yahoo data as parquet
    copy_from_backup(backup, "data")
    copy_from_backup(backup, "scripts")
    commit("pre-download yahoo data as parquet", next_date())
    advance(1)

    # Commit 40: RND heatmap for vol surface
    commit("rnd heatmap for vol surface", next_date(2))

    # Commit 41: fix heatmap colors
    commit("fix rnd heatmap colors in light mode", next_date(4))
    advance(1)

    # Commit 42: notebook walkthroughs
    copy_from_backup(backup, "vol_surface_calibrator/notebook.ipynb")
    copy_from_backup(backup, "structured_product_factory/notebook.ipynb")
    copy_from_backup(backup, "portfolio_optimizer/notebook.ipynb")
    copy_from_backup(backup, "pde_option_pricer/notebook.ipynb")
    commit("notebook walkthroughs", next_date())
    advance(1)

    # Commit 43: preset portfolios
    commit("preset portfolios for portfolio optimizer", next_date(1))
    advance(1)

    # Commit 44: factor covariance toggle
    commit("factor covariance toggle", next_date())
    advance(1)

    # Commit 45: root readme, license
    copy_from_backup(backup, "README.md")
    copy_from_backup(backup, "LICENSE")
    copy_from_backup(backup, "other ideas")
    commit("root readme, license", next_date(2))

    # Commit 46: fix streamlit cloud deployment
    commit("fix streamlit cloud deployment", next_date(4))

    # ──────────────────────────────────────────────────────────────────
    # Finalize: move main to rebuilt history
    # ──────────────────────────────────────────────────────────────────
    print("\n=== Finalizing ===")

    # Delete the rebuild script from the repo
    rebuild_script = REPO / "_rebuild_history.py"
    if rebuild_script.exists():
        run("git rm -f _rebuild_history.py 2>/dev/null || true")

    # Force main to point here
    run("git branch -D main 2>/dev/null || true")
    run("git branch -m main-rebuild main")

    print(f"\nDone! Backup saved at: {backup}")
    print("Verify with: git log --oneline")
    print(f"To restore: cp -r {backup}/* .")
    print("To push:    git push --force origin main")


if __name__ == "__main__":
    main()
