<div align="center">
  <h1>TorchSymPy</h1>
  <p><strong>SymPy-to-Torch Transcompilation for Massively Batched, GPU-Accelerated Numerical Integration</strong></p>

  [![PyPI - Version](https://img.shields.io/pypi/v/torchsympy)](https://pypi.org/project/torchsympy/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchsympy)](https://pypi.org/project/torchsympy/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

**TorchSymPy** bridges the gap between SymPy's symbolic mathematics and PyTorch's highly optimized, GPU-accelerated tensor operations. By employing a unique "compile-once, evaluate-many" architecture, you can transcompile symbolic integrals into native PyTorch engines capable of extremely fast, zero-overhead numerical evaluation across massive parameter grids.

*Note: this module was first developed for [libphysics](https://github.com/ferhatpy/libphysics) — then split out into a standalone library to tackle generalized parallel computational bottlenecks in integration.*

---

## Why TorchSymPy?

When working with analytical integrals in computational physics, optics, or machine learning, researchers often hit performance bottlenecks: 
1. **SymPy** is great for exact mathematical manipulation but is painfully slow (or completely fails) for heavy numerical grid evaluations.
2. **SciPy** (e.g., `scipy.integrate.quad`) is highly accurate but inherently sequential, single-threaded, and cannot natively leverage GPUs. 
3. **PyTorch** thrives on massively parallel grid evaluations, but writing structural integrators by hand is tedious and error-prone.

**TorchSymPy** gives you the best of all worlds. You write the math symbolically in `SymPy`, and TorchSymPy applies automated changes-of-variables (to handle infinite domains) and structural optimizations before transpiling it into highly optimized `TorchExpr` kernels. These kernels can run up to **3,400x faster** than `SciPy` by leveraging tensor-product grids and batched GPU execution.

## Installation

To install the latest stable version from PyPI:
```bash
pip install torchsympy
```

To install from source (development):
```bash
git clone https://github.com/ibeuler/TorchSymPy.git
cd TorchSymPy
pip install -e .
```

> **Note on PyTorch:** For GPU acceleration, ensure you have a CUDA-compatible `torch` wheel installed (e.g., `torch==2.5.1+cu121`).

## Quickstart

The easiest path from a symbolic integral to a batched GPU evaluation:

```python
import torch
import torchsympy
import sympy as sp

# 1. Define your integrand symbolically
x = sp.Symbol("x", real=True)
p = sp.Symbol("p", real=True)
expr = sp.Integral(sp.exp(-p * x**2), (x, -sp.oo, sp.oo))

# 2. Compile to a TorchSymPy engine
lt = torchsympy.TorchSymPy()
texpr = lt.torchify(expr)

# 3. Evaluate massively batched parameter grids on accelerators
p_grid = torch.linspace(0.5, 100.0, 10000, dtype=torch.float64, device="cuda").unsqueeze(-1)
re, im = texpr.torch_integrate_batched(
    params_values=p_grid,
    method="gauss-legendre",
    N=501,                       # Quadrature nodes
    device="cuda",               # Target accelerator
    dtype=torch.float64,         
    chunk_size_params=4096       # Safely chunk massive batches to avoid OOM
)

print(f"Real part shape: {re.shape}") # Output: torch.Size([10000])
```

## Core Concepts: Solvers and Backends

Once you define a symbolic integration expression, `TorchSymPy` provides distinct evaluation paths tailored to your mathematical structure and parameter scale:

### 1. `eval_numeric` (The Smart Wrapper)
This is the recommended high-level entry point. It traverses the SymPy expression tree and detects mathematical structures that can be vastly optimized. For instance, in highly oscillatory multi-dimensional integrals (like Fresnel diffraction), it automatically factors the problem into a **separable** path, avoiding the catastrophic $\mathcal{O}(N^d)$ exponential blowup of tensor-product grids. It also features a **shift-invariant** convolution path and controls automatic mesh refinement.

### 2. `torch_integrate_batched()` (Native Batched Backend)
The workhorse for small-to-large deterministic parameter sweeps. This backend natively implements PyTorch tensor-product rules (Gauss-Legendre, Simpson). It features **zero setup overhead** ($\approx 1.3$ ms for evaluation) because it globally caches its quadrature grids and entirely skips dynamic object instantiation, achieving up to 100x speedups over traditional numerical wrappers on small batches. It cleanly chunks evaluations to prevent OOM limits.

### 3. `torchquad_integrate_vectorized()` (Vectorized Backend)
Delegates evaluation entirely to the external `torchquad` library. It uses dynamic PyTorch broadcasting to avoid creating dense parameter meshgrids in memory. However, because it dynamically instantiates `IntegrationGrid` objects and performs an $\mathcal{O}(N^3)$ eigenvalue solve on every call, it carries a severe **$\approx 150$ ms fixed overhead**. It should only be used for massive parameter grids ($\ge 10^5$ points) where this fixed penalty is amortized, or when utilizing stochastic rules like Monte Carlo in $d \ge 4$ dimensions.

## Benchmarks: Speed Gains & Accuracy vs. SciPy & SymPy

TorchSymPy evaluates parameterized integrals across vast grids immensely faster than traditional methods. In our benchmark suite evaluating a parameterized Damped Cosine $\int_{0}^{\infty} e^{-x} \cos(k x) dx$, we observe huge multi-order speedups on GPUs. 

The following table demonstrates the inherent trade-off between quadrature resolution ($N$) and accuracy/speed:

| Execution | Time per Point | Speedup vs SciPy | Accuracy (vs Analytical) |
| :--- | :---: | :---: | :---: | 
| **SciPy (nquad)** | 1.664 ms | 1.0x | $\sim 2.90 \times 10^{-9}$ |
| **TorchSymPy (Vectorized, N=121)** | 0.00048 ms | **3,467x** | $\sim 2.07 \times 10^{-1}$ (Low N) |
| **TorchSymPy (Batched, N=121)** | 0.00073 ms | **2,279x** | $\sim 2.07 \times 10^{-1}$ (Low N) |
| **TorchSymPy (Vectorized, N=2001)** | 0.00854 ms | **194x** | $\sim 5.72 \times 10^{-5}$ (Medium N) |
| **TorchSymPy (Batched, N=2001)** | 0.05164 ms | **32x** | $\sim 5.72 \times 10^{-5}$ (Medium N) |
| **TorchSymPy (Vectorized, N=5001)** | 0.02589 ms | **64x** | $\sim 2.90 \times 10^{-9}$ (High N) |
| **TorchSymPy (Batched, N=5001)** | 0.55701 ms | **3.0x** | $\sim 2.90 \times 10^{-9}$ (High N) |

*(Benchmarks run on an NVIDIA RTX GPU across a 10,000 parameter grid. `TorchSymPy` converges to parity with SciPy while remaining orders of magnitude faster at standard resolutions).*

### The "Hard Integrals" Problem (Experimental Analytical Check)

While `TorchSymPy` achieves numeric parity with `SciPy` for well-behaved integrals (like $\int x^{-x} dx$), evaluating conditionally convergent oscillatory integrals over infinite domains numerically pushes *all* quadrature engines to their breaking points. 

Consider the famously difficult oscillatory integral:
$$ \int_0^\infty \frac{\sin(x)}{\sqrt{x^2 + 1}} dx $$

The true, analytical exact value (calculated symbolically via SymPy hypergeometric functions) is `0.873084`. However, if we force pure numerical evaluation without symbolic reduction:

| Method | Output Value | Absolute Error | Notes |
| :--- | :---: | :---: | :--- |
| **SymPy (True Analytical)** | `0.873084` | **0.0** | Solved symbolically via Hypergeometric functions |
| **SymPy (Pure `evalf()`)** | `-4.000000` | `4.873` | Completely fails convergence natively |
| **SciPy (`nquad`)** | `1.550175` | `0.677` | Fails with `IntegrationWarning` (Divergent) |
| **TorchSymPy (`GaussLegendre`)** | `-1.343219` | `2.216` | Breaks due to mapped infinite oscillations |

**Takeaway:** `TorchSymPy` provides incredible performance scaling and accurate results matching `SciPy` on standard mapping domains. However, for pathological integrands (like conditionally convergent oscillations at infinity), you should rely on `SymPy`'s exact symbolic analytical integrations *before* attempting numerical grid sweeps.

## Running the Test Suite

```bash
pytest tests/ -v
```

## Examples & Tutorials

Check the [`examples/`](examples/) directory for specific physics applications and basic integration usage, including generating Wigner functions.

## License
Distributed under the MIT License. See `LICENSE` for more information.