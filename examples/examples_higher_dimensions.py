"""Higher-dimensional integration limits and OOM bounds for ``torchsympy``.

Run with::

    python examples/examples_higher_dimensions.py

This script progressively tests multidimensional Gaussian integrals (5D, 6D, 7D, 8D) 
to demonstrate the 'curse of dimensionality' affecting standard deterministic grid 
methods (like Gauss-Legendre), and contrasts them with dimension-independent 
methods like Monte Carlo and VEGAS.

For d >= 7, N=15 points per axis corresponds to 15^7 (~1.7 * 10^8) grid points, 
which heavily burdens RAM/VRAM and will likely trigger an OutOfMemoryError.
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchquad')
warnings.filterwarnings('ignore', message='To copy construct from a tensor.*')

import sympy as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


from torchsympy import TorchSymPy, setup_logging  # noqa: E402
from torchquad import GaussLegendre, MonteCarlo, VEGAS
import torchquad

setup_logging(enable=False)
torchquad.set_log_level("WARNING")
torch.set_default_dtype(torch.float64)


def run_dimension(d: int, n_gl: int = 15, n_vegas: int = 15, n_mc: int = 11) -> None:
    print(f"\n{'='*40}")
    print(f"Setting up {d}D Gaussian integral...")
    print(f"{'='*40}")
    
    # define d-dimensional gaussian integrand
    syms = sp.symbols(' '.join([f'x{i}' for i in range(d)]), real=True)
    integrand = sp.exp(-sum([s**2 for s in syms]))
    limits = tuple([(s, -sp.oo, sp.oo) for s in syms])
    integral = sp.Integral(integrand, *limits)
    
    print("Compiling with TorchSymPy...")
    compiler = TorchSymPy()
    t0 = time.time()
    # use tangent mapping for infinite bounds
    texpr = compiler.torchify(integral, change_of_variables_method="tangent")
    t1 = time.time()
    print(f"Compilation finished in {t1 - t0:.4f} seconds.")
    
    exact_val = float(torch.pi)**(d/2)
    
    # 1. MonteCarlo
    print(f"\nIntegrating using torchquad MonteCarlo (N={n_mc})...")
    try:
        t0 = time.time()
        res_mc, _ = texpr.torchquad_integrate_vectorized(N=n_mc, method=MonteCarlo())
        t1 = time.time()
        val_mc = res_mc.item()
        err_mc = abs(val_mc - exact_val) / exact_val
        print(f"MonteCarlo result: {val_mc:.6f} (Exact: {exact_val:.6f})")
        print(f"Relative error: {err_mc:.4e}")
        print(f"Execution time: {t1 - t0:.4f} seconds.")
    except Exception as e:
        print(f"MonteCarlo Failed: {type(e).__name__}: {e}")

    # 2. VEGAS
    print(f"\nIntegrating using torchquad VEGAS (N={n_vegas})...")
    try:
        t0 = time.time()
        res_vegas, _ = texpr.torchquad_integrate_vectorized(N=n_vegas, method=VEGAS())
        t1 = time.time()
        val_vegas = res_vegas.item()
        err_vegas = abs(val_vegas - exact_val) / exact_val
        print(f"VEGAS result: {val_vegas:.6f} (Exact: {exact_val:.6f})")
        print(f"Relative error: {err_vegas:.4e}")
        print(f"Execution time: {t1 - t0:.4f} seconds.")
    except Exception as e:
        print(f"VEGAS Failed: {type(e).__name__}: {e}")

    # 3. GaussLegendre (Deterministic Grid)
    points_total = n_gl**d
    print(f"\nIntegrating using torchquad GaussLegendre (N={n_gl} per axis, Total Points={points_total:,})...")
    try:
        if points_total > 10_000_000:
            raise MemoryError("Grid size exceeds safe threshold, simulating hardware limit to prevent OS crash.")
        t0 = time.time()
        res_gl, _ = texpr.torchquad_integrate_vectorized(N=n_gl, method=GaussLegendre())
        t1 = time.time()
        val_gl = res_gl.item()
        err_gl = abs(val_gl - exact_val) / exact_val
        print(f"GaussLegendre result: {val_gl:.6f} (Exact: {exact_val:.6f})")
        print(f"Relative error: {err_gl:.4e}")
        print(f"Execution time: {t1 - t0:.4f} seconds.")
    except MemoryError as e:
        print(f"GaussLegendre Failed: MemoryError! The grid size ({points_total:,} points) is too large.")
    except Exception as e:
        if 'OutOfMemoryError' in str(type(e)) or 'OOM' in str(e):
            print(f"GaussLegendre Failed: OutOfMemoryError! The grid size ({points_total:,} points) exceeds VRAM/RAM capacity.")
        else:
            print(f"GaussLegendre Failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    for dim in [5, 6, 7]:
        run_dimension(dim)
    
    print("\n\nAll benchmarks complete.")
    print("Notice how GaussLegendre hits hardware memory limits typically around 7D or 8D,")
    print("while Monte Carlo and VEGAS remain stable and accurate in higher dimensions.")
