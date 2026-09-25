"""Two-dimensional integration examples for ``torchsympy``.

Run with::

    python examples/example_2d.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import sympy as sp
import torch
from scipy.special import erf, fresnel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from torchsympy import TorchSymPy  # noqa: E402

torch.set_default_dtype(torch.float64)


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-300)


def example_1_separable_gaussian_box() -> None:
    """int_{-2}^{2} int_{-2}^{2} exp(-a x^2 - b y^2) dx dy, with (a, b) as parameters."""
    print("\n[2D-1] Anisotropic Gaussian over the box [-2, 2]^2")
    x, y, a, b = sp.symbols("x y a b", real=True, positive=True)
    integral = sp.Integral(sp.exp(-a * x ** 2 - b * y ** 2), (x, -2, 2), (y, -2, 2))

    texpr = TorchSymPy().torchify(integral)
    # Parameters are ordered alphabetically by symbol name: (a, b).
    params = torch.tensor([[1.0, 2.0], [0.5, 0.5], [3.0, 0.25]])
    re_part, _ = texpr.torch_integrate_batched(params_values=params, N=101,
                                               method="gauss-legendre")

    def reference(alpha: float, beta: float) -> float:
        one_d = lambda c: math.sqrt(math.pi / c) * erf(2 * math.sqrt(c))
        return one_d(alpha) * one_d(beta)

    for (alpha, beta), value in zip(params.tolist(), re_part.tolist()):
        ref = reference(alpha, beta)
        print(f"    a={alpha:<5.2f} b={beta:<5.2f} value={value:.14f} "
              f"reference={ref:.14f}  rel.err={relative_error(value, ref):.3e}")


def example_2_triangular_domain() -> None:
    """A domain with a variable inner limit: int_0^1 int_0^x (x + y) dy dx = 1/2.

    Finite *symbolic* limits are mapped affinely onto [0, 1], so the triangle is
    integrated as a square with the Jacobian folded into the integrand.
    """
    print("\n[2D-2] Triangular domain 0 <= y <= x <= 1")
    x, y = sp.symbols("x y", real=True)
    integral = sp.Integral(x + y, (y, 0, x), (x, 0, 1))

    texpr = TorchSymPy().torchify(integral)
    print(f"    transformed integrand : {texpr.sympy_integrand}")
    print(f"    transformed variables : {texpr.variables}")
    print(f"    quadrature box        : {texpr.domain}")
    value = float(texpr.torch_integrate_batched(N=81, method="gauss-legendre")[0])
    print(f"    value={value:.15f}  reference={0.5:.15f}  rel.err={relative_error(value, 0.5):.3e}")


def example_3_fresnel_diffraction_screen() -> None:
    """Fresnel diffraction of a square aperture, evaluated on a screen grid.

    I(X, Y) = |int int exp(i k/(2z) [(X-x0)^2 + (Y-y0)^2]) dx0 dy0|^2 / (lambda z)^2

    The phase is additively separable, so ``eval_numeric`` factorises the 2D
    integral into two 1D integrals and refines each one until it converges.
    The reference value is the textbook product of Fresnel integrals.
    """
    print("\n[2D-3] Fresnel diffraction by a square aperture (separable fast path)")
    x0, y0, X, Y = sp.symbols("x0 y0 X Y", real=True)
    wavelength = 500e-6      # mm  (500 nm)
    distance = 10.0          # mm
    half_width = 0.5         # mm
    wavenumber = 2 * math.pi / wavelength
    alpha = wavenumber / (2 * distance)

    phase = alpha * ((X - x0) ** 2 + (Y - y0) ** 2)
    cosine_part = sp.Integral(sp.cos(phase), (x0, -half_width, half_width),
                              (y0, -half_width, half_width))
    sine_part = sp.Integral(sp.sin(phase), (x0, -half_width, half_width),
                            (y0, -half_width, half_width))
    intensity = (cosine_part ** 2 + sine_part ** 2) / (wavelength * distance) ** 2

    screen_x = torch.tensor([0.0, 0.3, 0.6, 1.0])
    screen_y = torch.zeros_like(screen_x)

    def analytic_axis(centre: float) -> complex:
        scale = math.sqrt(2 * alpha / math.pi)
        lower, upper = scale * (-half_width - centre), scale * (half_width - centre)
        sin_upper, cos_upper = fresnel(upper)
        sin_lower, cos_lower = fresnel(lower)
        return math.sqrt(math.pi / (2 * alpha)) * ((cos_upper - cos_lower)
                                                   + 1j * (sin_upper - sin_lower))

    reference = [abs(analytic_axis(float(sx)) * analytic_axis(0.0)) ** 2
                 / (wavelength * distance) ** 2 for sx in screen_x]

    # (a) default: the separable fast path is used automatically
    start = time.perf_counter()
    smart_value, _ = TorchSymPy().eval_numeric(
        intensity, params_values=[screen_x, screen_y],
        solver="batched", N=501, method="gauss-legendre",
    )
    smart_time = time.perf_counter() - start

    # (b) plain tensor-product quadrature with ~5000 nodes in total (~71 per axis),
    #     the budget a naive 2D rule would use
    plain_value, _ = TorchSymPy().eval_numeric(
        intensity, params_values=[screen_x, screen_y], smart=False,
        solver="batched", N=71, method="gauss-legendre",
    )

    print(f"    screen X        = {[f'{v:.2f}' for v in screen_x.tolist()]}")
    print(f"    separable path  = {[f'{v:.6e}' for v in smart_value.tolist()]}  ({smart_time:.2f} s)")
    print(f"    Fresnel formula = {[f'{v:.6e}' for v in reference]}")
    print(f"    max rel.err     = "
          f"{max(relative_error(v, r) for v, r in zip(smart_value.tolist(), reference)):.3e}")
    print(f"    plain 71^2 rule = {[f'{v:.6e}' for v in plain_value.tolist()]}   <-- aliased")


def example_4_rayleigh_sommerfeld_screen() -> None:
    """Rayleigh-Sommerfeld propagation: non-separable, but shift invariant.

    The kernel depends on (X - x0, Y - y0) only, so ``eval_numeric`` builds one
    cumulative integral of the kernel shared by every screen point.
    """
    print("\n[2D-4] Rayleigh-Sommerfeld propagation (shift-invariant fast path)")
    x0, y0, X, Y = sp.symbols("x0 y0 X Y", real=True)
    wavelength, distance, half_width = 500e-6, 10.0, 0.5
    wavenumber = 2 * math.pi / wavelength
    radius = sp.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + distance ** 2)

    cosine_part = sp.Integral(distance * sp.cos(wavenumber * radius) / radius ** 2,
                              (x0, -half_width, half_width), (y0, -half_width, half_width))
    sine_part = sp.Integral(distance * sp.sin(wavenumber * radius) / radius ** 2,
                            (x0, -half_width, half_width), (y0, -half_width, half_width))
    intensity = (cosine_part ** 2 + sine_part ** 2) / wavelength ** 2

    screen_x = torch.tensor([0.0, 0.3, 0.6, 1.0])
    screen_y = torch.zeros_like(screen_x)

    start = time.perf_counter()
    fast_value, _ = TorchSymPy().eval_numeric(
        intensity, params_values=[screen_x, screen_y],
        solver="batched", N=201, method="gauss-legendre",
    )
    fast_time = time.perf_counter() - start

    start = time.perf_counter()
    brute_value, _ = TorchSymPy().eval_numeric(
        intensity, params_values=[screen_x, screen_y], smart=False,
        solver="batched", N=2001, method="gauss-legendre",
    )
    brute_time = time.perf_counter() - start

    print(f"    shift-invariant  = {[f'{v:.6e}' for v in fast_value.tolist()]}  ({fast_time:.2f} s)")
    print(f"    brute 2001^2 rule= {[f'{v:.6e}' for v in brute_value.tolist()]}  ({brute_time:.2f} s)")
    print(f"    max rel. difference = "
          f"{max(relative_error(a, b) for a, b in zip(fast_value.tolist(), brute_value.tolist())):.3e}")


if __name__ == "__main__":
    example_1_separable_gaussian_box()
    example_2_triangular_domain()
    example_3_fresnel_diffraction_screen()
    example_4_rayleigh_sommerfeld_screen()
