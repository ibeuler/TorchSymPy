"""Tests for TorchSymPy.eval_numeric -- the high-level wrapper that handles
composite expressions with Integral nodes, separable fast paths, and the
standard tensor-product fallback.

Run with::

    pytest tests/test_eval_numeric.py -v --tb=short
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest


def _rel_err(computed, reference):
    return abs(computed - reference) / max(abs(reference), 1e-300)


# 1d scalar integral via eval_numeric
def test_eval_numeric_1d_scalar_gaussian(torchsympy_instance, sp, torch, device):
    x = sp.Symbol("x", real=True)
    integral = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo))
    result = torchsympy_instance.eval_numeric(
        integral, solver="batched", N=201, method="gauss-legendre",
        device=device, dtype=torch.float64,
    )
    val = float(result)
    assert _rel_err(val, math.sqrt(math.pi)) < 1e-6


# 1d batched fourier via eval_numeric
def test_eval_numeric_1d_batched_fourier(torchsympy_instance, sp, torch, device):
    x, k = sp.symbols("x k", real=True)
    integral = sp.Integral(sp.exp(-x**2) * sp.exp(sp.I * k * x), (x, -sp.oo, sp.oo))
    k_vals = torch.linspace(-2.0, 2.0, 5, device=device, dtype=torch.float64)
    re, im = torchsympy_instance.eval_numeric(
        integral, params_values=[k_vals],
        solver="batched", N=401, method="gauss-legendre",
        device=device, dtype=torch.float64,
    )
    ref = (math.sqrt(math.pi) * torch.exp(-k_vals**2 / 4.0)).to(device=re.device)
    assert torch.allclose(re, ref, atol=1e-6, rtol=1e-6)


# 2d separable via eval_numeric
def test_eval_numeric_2d_separable(torchsympy_instance, sp, torch, device):
    x, y = sp.symbols("x y", real=True)
    integral = sp.Integral(sp.cos(10 * x**2 + 10 * y**2), (x, -1, 1), (y, -1, 1))
    val = torchsympy_instance.eval_numeric(
        integral, solver="batched", N=501, method="gauss-legendre",
        device=device, dtype=torch.float64,
    )
    val_f = float(val)
    val_plain = torchsympy_instance.eval_numeric(
        integral, solver="batched", N=501, method="gauss-legendre", smart=False,
        device=device, dtype=torch.float64,
    )
    val_plain_f = float(val_plain)
    assert _rel_err(val_f, val_plain_f) < 1e-4


# composite expression with squared integrals
def test_eval_numeric_composite_intensity(torchsympy_instance, sp, torch, device):
    x0, X = sp.symbols("x0 X", real=True)
    c = 10.0
    cos_int = sp.Integral(sp.cos(c * (X - x0)**2), (x0, -1, 1))
    sin_int = sp.Integral(sp.sin(c * (X - x0)**2), (x0, -1, 1))
    intensity = cos_int**2 + sin_int**2
    x_vals = torch.tensor([0.0, 0.5], device=device, dtype=torch.float64)
    re, im = torchsympy_instance.eval_numeric(
        intensity, params_values=[x_vals],
        solver="batched", N=501, method="gauss-legendre",
        device=device, dtype=torch.float64,
    )
    re_cpu = re.detach().cpu()
    im_cpu = im.detach().cpu()
    assert torch.all(re_cpu >= -1e-10)
    assert torch.allclose(im_cpu, torch.zeros_like(im_cpu), atol=1e-8)


# triangular domain via eval_numeric
def test_eval_numeric_2d_triangular(torchsympy_instance, sp, torch, device):
    x, y = sp.symbols("x y", real=True)
    integral = sp.Integral(x + y, (y, 0, x), (x, 0, 1))
    val = torchsympy_instance.eval_numeric(
        integral, solver="batched", N=101, method="gauss-legendre",
        device=device, dtype=torch.float64,
    )
    assert _rel_err(float(val), 0.5) < 1e-10


# 3d simplex via eval_numeric
def test_eval_numeric_3d_simplex(torchsympy_instance, sp, torch, device):
    x, y, z = sp.symbols("x y z", real=True)
    integral = sp.Integral(1, (z, 0, y), (y, 0, x), (x, 0, 1))
    val = torchsympy_instance.eval_numeric(
        integral, solver="batched", N=61, method="gauss-legendre",
        device=device, dtype=torch.float64,
    )
    assert _rel_err(float(val), 1.0 / 6.0) < 1e-6


# benchmark table
def test_eval_numeric_benchmark_table(torchsympy_instance, sp, torch, device):
    """Produce a comparison table of eval_numeric vs batched vs vectorized."""
    dtype = torch.float64
    torchquad = pytest.importorskip("torchquad")
    from torchquad import GaussLegendre

    x = sp.Symbol("x", real=True)
    p = sp.Symbol("p", real=True)

    cases = [
        {
            "name": "Gaussian (inf)",
            "expr": sp.exp(-p * x**2),
            "limits": [(x, -sp.oo, sp.oo)],
            "ref": lambda pv: torch.sqrt(torch.tensor(math.pi, device=device, dtype=dtype) / pv),
        },
        {
            "name": "Fourier-Gauss (inf)",
            "expr": sp.exp(-x**2) * sp.exp(sp.I * p * x),
            "limits": [(x, -sp.oo, sp.oo)],
            "ref": lambda pv: math.sqrt(math.pi) * torch.exp(-pv**2 / 4.0),
        },
        {
            "name": "Damped cos (semi-inf)",
            "expr": sp.exp(-x) * sp.cos(p * x),
            "limits": [(x, 0, sp.oo)],
            "ref": lambda pv: 1.0 / (1.0 + pv**2),
        },
    ]

    p_grid = torch.linspace(0.5, 20.0, 200, device=device, dtype=dtype)
    N_eval = 1001
    results = []

    with torch.no_grad():
        for case in cases:
            integral = sp.Integral(case["expr"], *case["limits"])
            ref = case["ref"](p_grid)

            # eval_numeric (smart=True)
            t0 = time.perf_counter()
            re_en, _ = torchsympy_instance.eval_numeric(
                integral, params_values=[p_grid],
                solver="batched", N=N_eval, method="gauss-legendre", smart=True,
                device=device, dtype=dtype,
            )
            t_en = time.perf_counter() - t0
            re_en_cpu = re_en.detach().cpu() if torch.is_tensor(re_en) else torch.tensor(float(re_en))
            ref_cpu = ref.detach().cpu()
            err_en = float((re_en_cpu - ref_cpu).abs().max())

            # batched backend
            texpr = torchsympy_instance.torchify(integral)
            t0 = time.perf_counter()
            re_bat, _ = texpr.torch_integrate_batched(
                params_values=p_grid.unsqueeze(-1), N=N_eval, method="gauss-legendre",
                device=device, dtype=dtype,
            )
            t_bat = time.perf_counter() - t0
            re_bat_cpu = re_bat.detach().cpu()
            err_bat = float((re_bat_cpu - ref_cpu).abs().max())

            # vectorized backend
            t0 = time.perf_counter()
            re_vec, _ = texpr.torchquad_integrate_vectorized(
                params_values=[p_grid.unsqueeze(-1)], method=GaussLegendre(), N=N_eval,
                device=device, dtype=dtype
            )
            t_vec = time.perf_counter() - t0
            re_vec_cpu = re_vec.detach().cpu()
            err_vec = float((re_vec_cpu - ref_cpu).abs().max())

            results.append({
                "name": case["name"],
                "eval_numeric_ms": t_en * 1000,
                "batched_ms": t_bat * 1000,
                "vectorized_ms": t_vec * 1000,
                "eval_numeric_err": err_en,
                "batched_err": err_bat,
                "vectorized_err": err_vec,
            })

    # print table
    print("\n--- eval_numeric Benchmark Table ---")
    print(f"{'Case':<25} | {'eval_num ms':>12} | {'batched ms':>12} | {'vec ms':>12} | "
          f"{'EN err':>10} | {'Bat err':>10} | {'Vec err':>10}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:<25} | {r['eval_numeric_ms']:12.2f} | {r['batched_ms']:12.2f} | "
              f"{r['vectorized_ms']:12.2f} | {r['eval_numeric_err']:10.2e} | "
              f"{r['batched_err']:10.2e} | {r['vectorized_err']:10.2e}")

    # write latex table
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "tests" / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        "\\textbf{Integral} & \\textbf{eval\\_numeric (ms)} & \\textbf{Batched (ms)} & "
        "\\textbf{Vectorized (ms)} & \\textbf{EN Error} & \\textbf{Bat. Error} & \\textbf{Vec. Error} \\\\",
        "\\midrule",
    ]
    for r in results:
        safe = r["name"].replace("_", "\\_")
        tex.append(
            f"{safe} & {r['eval_numeric_ms']:.1f} & {r['batched_ms']:.1f} & "
            f"{r['vectorized_ms']:.1f} & {r['eval_numeric_err']:.2e} & "
            f"{r['batched_err']:.2e} & {r['vectorized_err']:.2e} \\\\"
        )
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Comparison of \\texttt{eval\\_numeric} (smart separable path) against "
        "the batched and vectorized backends at $N=1001$ over 200 parameter values.}",
        "\\label{tab:eval_numeric_comparison}",
        "\\end{table}",
    ])
    (results_dir / "eval_numeric_benchmark.tex").write_text("\n".join(tex), encoding="utf-8")

    for r in results:
        assert r["eval_numeric_err"] < 0.5, f"{r['name']}: eval_numeric error too large"
