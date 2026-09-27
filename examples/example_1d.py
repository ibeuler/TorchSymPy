"""One-dimensional integration examples for ``torchsympy``.

Run with::

    python examples/example_1d.py

Every example prints the value computed by the library next to a closed-form
reference value and the resulting relative error.
"""
from __future__ import annotations

import math
import os
import sys

import sympy as sp
import torch
from scipy.special import fresnel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from torchsympy import TorchSymPy  # noqa: E402

torch.set_default_dtype(torch.float64)

SQRT_PI = math.sqrt(math.pi)


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-300)


def example_1_gaussian_on_the_real_line() -> None:
    """int_{-oo}^{oo} exp(-x^2) dx = sqrt(pi).

    The infinite interval is removed by the change of variables selected with
    ``change_of_variables_method``; only the finite transformed box is ever
    handed to the quadrature rule.
    """
    print("\n[1D-1] int_{-oo}^{+oo} exp(-x^2) dx = sqrt(pi)")
    x = sp.Symbol("x", real=True)
    integral = sp.Integral(sp.exp(-x ** 2), (x, -sp.oo, sp.oo))

    compiler = TorchSymPy()
    for cov in ("tangent", "algebraic", "tanh-sinh"):
        texpr = compiler.torchify(integral, change_of_variables_method=cov)
        for rule in ("gauss-legendre", "simpson", "trapezoid"):
            re_part, _ = texpr.torch_integrate_batched(N=201, method=rule)
            value = float(re_part)
            print(
                f"    cov={cov:<10s} rule={rule:<15s} domain="
                f"[{texpr.domain[0][0]:+.4f}, {texpr.domain[0][1]:+.4f}]  "
                f"value={value:.15f}  rel.err={relative_error(value, SQRT_PI):.3e}"
            )


def example_2_parametric_fourier_transform() -> None:
    """int exp(-x^2) exp(i k x) dx = sqrt(pi) exp(-k^2/4), on a grid of k."""
    print("\n[1D-2] Fourier transform of a Gaussian, batched over k")
    x, k = sp.symbols("x k", real=True)
    integral = sp.Integral(sp.exp(-x ** 2) * sp.exp(sp.I * k * x), (x, -sp.oo, sp.oo))

    compiler = TorchSymPy()
    texpr = compiler.torchify(integral)          # SymPy work happens once
    k_grid = torch.linspace(-3.0, 3.0, 7).unsqueeze(-1)   # shape (7, 1) = (batch, n_params)

    re_part, im_part = texpr.torch_integrate_batched(     # torch-only, fast
        params_values=k_grid, N=401, method="gauss-legendre", chunk_size_params=64
    )
    reference = SQRT_PI * torch.exp(-k_grid.squeeze(-1) ** 2 / 4)
    re_cpu = re_part.detach().cpu()
    im_cpu = im_part.detach().cpu()
    print(f"    k          = {[f'{v:+.2f}' for v in k_grid.flatten().tolist()]}")
    print(f"    Re(result) = {[f'{v:.10f}' for v in re_cpu.tolist()]}")
    print(f"    reference  = {[f'{v:.10f}' for v in reference.tolist()]}")
    print(f"    max |Re - ref| = {float((re_cpu - reference).abs().max()):.3e}")
    print(f"    max |Im|       = {float(im_cpu.abs().max()):.3e}   (exactly 0 by symmetry)")


def example_3_oscillatory_fresnel() -> None:
    """int_0^1 cos(c x^2) dx = sqrt(pi/(2c)) C(sqrt(2c/pi)), a Fresnel integral."""
    print("\n[1D-3] Oscillatory integrand: int_0^1 cos(c x^2) dx")
    x = sp.Symbol("x", real=True)
    for c in (10.0, 200.0, 5000.0):
        integral = sp.Integral(sp.cos(c * x ** 2), (x, 0, 1))
        texpr = TorchSymPy().torchify(integral)
        argument = math.sqrt(2 * c / math.pi)
        _, cosine_fresnel = fresnel(argument)
        reference = math.sqrt(math.pi / (2 * c)) * cosine_fresnel
        for n_nodes in (101, 1001, 20001):
            value = float(texpr.torch_integrate_batched(N=n_nodes, method="gauss-legendre")[0])
            print(
                f"    c={c:<8.0f} N={n_nodes:<6d} value={value:+.12f} "
                f"reference={reference:+.12f}  rel.err={relative_error(value, reference):.3e}"
            )


def example_4_gradient_through_the_quadrature() -> None:
    """d/dk int exp(-x^2) cos(k x) dx = -(k/2) sqrt(pi) exp(-k^2/4) by autograd."""
    print("\n[1D-4] Differentiating an integral with respect to its parameter")
    x, k = sp.symbols("x k", real=True)
    texpr = TorchSymPy().torchify(
        sp.Integral(sp.exp(-x ** 2) * sp.cos(k * x), (x, -sp.oo, sp.oo))
    )

    k_values = torch.tensor([[1.0], [2.0]], requires_grad=True)
    re_part, _ = texpr.torch_integrate_batched(params_values=k_values, N=401,
                                               method="gauss-legendre")
    (gradient,) = torch.autograd.grad(re_part.sum(), k_values)

    exact_value = [SQRT_PI * math.exp(-v ** 2 / 4) for v in (1.0, 2.0)]
    exact_grad = [-v / 2 * SQRT_PI * math.exp(-v ** 2 / 4) for v in (1.0, 2.0)]
    print(f"    I(k)      = {[f'{v:.12f}' for v in re_part.detach().tolist()]}")
    print(f"    exact     = {[f'{v:.12f}' for v in exact_value]}")
    print(f"    dI/dk     = {[f'{v:.12f}' for v in gradient.flatten().tolist()]}")
    print(f"    exact     = {[f'{v:.12f}' for v in exact_grad]}")


if __name__ == "__main__":
    example_1_gaussian_on_the_real_line()
    example_2_parametric_fourier_transform()
    example_3_oscillatory_fresnel()
    example_4_gradient_through_the_quadrature()
