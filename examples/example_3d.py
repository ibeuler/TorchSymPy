"""Three-dimensional integration examples for ``torchsympy``.

Run with::

    python examples/example_3d.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import sympy as sp
import torch
from scipy.special import fresnel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from torchsympy import TorchSymPy  # noqa: E402

torch.set_default_dtype(torch.float64)


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-300)


def example_1_gaussian_over_r3() -> None:
    """int_{R^3} exp(-(x^2+y^2+z^2)) dV = pi^{3/2}.

    Three infinite axes, three changes of variables, one tensor-product rule of
    N^3 nodes.
    """
    print("\n[3D-1] int_{R^3} exp(-r^2) dV = pi^{3/2}")
    x, y, z = sp.symbols("x y z", real=True)
    integral = sp.Integral(
        sp.exp(-(x ** 2 + y ** 2 + z ** 2)),
        (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo),
    )
    texpr = TorchSymPy().torchify(integral)
    reference = math.pi ** 1.5
    for n_nodes, rule in ((41, "gauss-legendre"), (81, "gauss-legendre"),
                          (41, "simpson"), (81, "simpson")):
        start = time.perf_counter()
        value = float(texpr.torch_integrate_batched(N=n_nodes, method=rule,
                                                    chunk_size_points=2 ** 20)[0])
        elapsed = time.perf_counter() - start
        print(f"    rule={rule:<15s} N={n_nodes:<4d} ({n_nodes ** 3:>9d} nodes) "
              f"value={value:.14f} rel.err={relative_error(value, reference):.3e} "
              f"({elapsed:.2f} s)")


def example_2_parametric_gaussian() -> None:
    """int_{R^3} exp(-w r^2) dV = (pi/w)^{3/2}, batched over the width w."""
    print("\n[3D-2] Width-parametrised Gaussian, batched over w")
    x, y, z, w = sp.symbols("x y z w", real=True, positive=True)
    integral = sp.Integral(
        sp.exp(-w * (x ** 2 + y ** 2 + z ** 2)),
        (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo),
    )
    texpr = TorchSymPy().torchify(integral)
    w_values = torch.tensor([[0.5], [1.0], [2.0], [5.0]])
    re_part, _ = texpr.torch_integrate_batched(params_values=w_values, N=61,
                                               method="gauss-legendre")
    for width, value in zip(w_values.flatten().tolist(), re_part.tolist()):
        ref = (math.pi / width) ** 1.5
        print(f"    w={width:<5.2f} value={value:.13f} reference={ref:.13f} "
              f"rel.err={relative_error(value, ref):.3e}")


def example_3_tetrahedron_with_nested_limits() -> None:
    """Volume of the simplex 0 <= z <= y <= x <= 1, i.e. 1/6.

    Each inner limit depends on the next outer variable; those finite symbolic
    limits are mapped affinely onto [0, 1] and the Jacobian x * (x) folded into
    the integrand, so the simplex is integrated as a unit cube.
    """
    print("\n[3D-3] Simplex 0 <= z <= y <= x <= 1 (nested, variable limits)")
    x, y, z = sp.symbols("x y z", real=True)
    integral = sp.Integral(1, (z, 0, y), (y, 0, x), (x, 0, 1))
    texpr = TorchSymPy().torchify(integral)
    print(f"    transformed integrand : {texpr.sympy_integrand}")
    print(f"    transformed variables : {texpr.variables}")
    print(f"    quadrature box        : {texpr.domain}")
    value = float(texpr.torch_integrate_batched(N=61, method="gauss-legendre")[0])
    print(f"    value={value:.15f}  reference={1/6:.15f}  "
          f"rel.err={relative_error(value, 1 / 6):.3e}")


def example_4_oscillatory_cube() -> None:
    """int_{[-1,1]^3} cos(c (x^2+y^2+z^2)) dV = Re[(C + iS)^3] with the Fresnel pair.

    A genuinely 3D oscillatory integrand: the tensor-product rule needs N^3
    nodes, whereas the separable fast path of ``eval_numeric`` needs 3 * N.
    """
    print("\n[3D-4] int_{[-1,1]^3} cos(c r^2) dV (separable fast path)")
    x, y, z = sp.symbols("x y z", real=True)
    for c in (20.0, 200.0):
        integral = sp.Integral(sp.cos(c * (x ** 2 + y ** 2 + z ** 2)),
                               (x, -1, 1), (y, -1, 1), (z, -1, 1))
        argument = math.sqrt(2 * c / math.pi)
        sin_part, cos_part = fresnel(argument)
        axis_value = 2 * math.sqrt(math.pi / (2 * c)) * (cos_part + 1j * sin_part)
        reference = (axis_value ** 3).real

        start = time.perf_counter()
        smart = float(TorchSymPy().eval_numeric(integral, solver="batched", N=301,
                                                method="gauss-legendre"))
        smart_time = time.perf_counter() - start

        start = time.perf_counter()
        plain = float(TorchSymPy().eval_numeric(integral, smart=False, solver="batched",
                                                N=301, method="gauss-legendre",
                                                chunk_size_points=2 ** 21))
        plain_time = time.perf_counter() - start

        print(f"    c={c:<6.0f} separable={smart:+.12f} ({smart_time:.2f} s)   "
              f"tensor-product 301^3={plain:+.12f} ({plain_time:.2f} s)   "
              f"reference={reference:+.12f}")
        print(f"            rel.err separable={relative_error(smart, reference):.3e}   "
              f"rel.err tensor-product={relative_error(plain, reference):.3e}")


def example_5_singular_box_integral() -> None:
    """A weakly singular integrand: int_{[0,1]^3} dV / r = (3/2) ln(2 + sqrt 3) - pi/4.

    Nothing in the library removes the 1/r singularity at the origin: the
    Gauss-Legendre nodes simply never touch it, and the rule still converges,
    but only algebraically rather than at the spectral rate of the smooth
    examples above.
    """
    print("\n[3D-5] Weakly singular box integral int_{[0,1]^3} dV / r")
    x, y, z = sp.symbols("x y z", real=True, positive=True)
    integral = sp.Integral(1 / sp.sqrt(x ** 2 + y ** 2 + z ** 2),
                           (x, 0, 1), (y, 0, 1), (z, 0, 1))
    texpr = TorchSymPy().torchify(integral)
    reference = 1.5 * math.log(2 + math.sqrt(3)) - math.pi / 4
    for n_nodes in (51, 101, 201):
        value = float(texpr.torch_integrate_batched(N=n_nodes, method="gauss-legendre",
                                                    chunk_size_points=2 ** 20)[0])
        print(f"    N={n_nodes:<4d} value={value:.10f} reference={reference:.10f} "
              f"rel.err={relative_error(value, reference):.3e}")


if __name__ == "__main__":
    example_1_gaussian_over_r3()
    example_2_parametric_gaussian()
    example_3_tetrahedron_with_nested_limits()
    example_4_oscillatory_cube()
    example_5_singular_box_integral()
