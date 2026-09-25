# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import torch
import sympy as smp
from scipy.integrate import nquad
from torchquad import GaussLegendre

# Path setup for libphysics and torchsympy
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
libphysics_path = os.path.join(base_dir, 'private', 'libphysics_private', 'libphysics', 'src')
sys.path.insert(0, libphysics_path)
torchsympy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, torchsympy_path)

from libsympy import *
from optics import *
import torchsympy

class sets:
    global dictflow, test_all
    
    def __init__(self):
        pass
    
    input_dir  = "input/optics"
    output_dir = "output/optics"
    
    plot_time_scale = {1:"xy", 2:"xz", 3:"yz"}[3]
    
    test_all = {0:False, 1:True}[0]
    usecupy = {0:False, 1:True}[0]
    usetorchsympy = {0:False, 1:True}[1]
    dictflow = dict(
        ch1 = {100:"get_formulary", 150:"get_subformulary",
               200:"ABCD_2_thin_lens", 202:"ABCD_microscope",
               250:"diffraction_rectangular", 300:"Fraunhofer_Diff_Int",
               350:"FBG_Reflection"})
    flow = [dictflow["ch1"][i] for i in [250]]

print("Test of the {0}.".format(sets.flow))
print("Using cupy: {0}".format(sets.usecupy))

if "diffraction_rectangular" in sets.flow:
    print("Diffraction from a Rectangular Aperture")
    class_type = {1:"Rayleigh_Sommerfeld", 2:"Fraunhofer", 3:"Fresnel",
                  4:"FresnelJit"}[3] # We'll use Fresnel (3) to test numerical integration
    oopti.__init__(class_type)
    oopti.verbose = False
    
    [Lx,Ly] = symbols('L_x L_y', real=True, positive=True)
    rectX = oopti.subformulary.rect(x0/Lx)
    rectY = oopti.subformulary.rect(y0/Ly)
    xreplaces = {Uapr:1, Eli:1, 
                 x0min:-S(1)/2*Lx, x0max:S(1)/2*Lx,
                 y0min:-S(1)/2*Ly, y0max:S(1)/2*Ly}
    
    # Numerical Calculations
    n = 100
    screen_factor = 2.5;
    config = {0:0, 1:"LakshminarayananFig11_3",
              61:"Abedin2005Fig6a", 62:"Abedin2005Fig6b",
              63:"Abedin2005Fig6c", 64:"Abedin2005Fig6d", 
              72:"Abedin2005Fig7b"}[61]
    
    [nLx, nLy] = {0:[1, 1],
                "LakshminarayananFig11_3":[0.11, 0.11],
                "Abedin2005Fig6a":[2,2], "Abedin2005Fig6b":[2,2],
                "Abedin2005Fig6c":[2,2], "Abedin2005Fig6d":[2,2], 
                "Abedin2005Fig7b":[2,2]}[config]
    
    nl = {0:1, 
          "LakshminarayananFig11_3":560e-6,
          "Abedin2005Fig6a":632e-6, "Abedin2005Fig6b":632e-6,
          "Abedin2005Fig6c":632e-6, "Abedin2005Fig6d":632e-6,
          "Abedin2005Fig7b":1264e-6}[config]
    
    nz = {0:0.5, 
          "LakshminarayananFig11_3":3, 
          "Abedin2005Fig6a":400, "Abedin2005Fig6b":800,
          "Abedin2005Fig6c":1700,"Abedin2005Fig6d":8000, 
          "Abedin2005Fig7b":400}[config]
    
    brightness = {"Rayleigh_Sommerfeld":1, "Fraunhofer":0.1, "Fresnel":1}[oopti.class_type];
    lsX = np.linspace(-nLx*screen_factor, nLx*screen_factor, n)
    lsY = np.linspace(-nLy*screen_factor, nLy*screen_factor, n)
    X,Y = np.meshgrid(lsX, lsY)
    subs = {z:nz, l:nl, Lx:nLx, Ly:nLy, k:2*pi/l}
    print(config, " ", oopti.class_type)
    
    # Symbolic Calculations
    commands = ["xreplace", "oopti."+class_type+".integral", xreplaces]
    oopti.process(commands)
    diffr_form = oopti.result.doit()
    display(diffr_form)
    
    commands = ["xreplace", "oopti."+class_type+".EField", xreplaces]
    oopti.process(commands)
    El = oopti.result.doit()
    display(El)
    
    commands = ["xreplace", "oopti."+class_type+".intensity", xreplaces]
    oopti.process(commands)
    Int = oopti.result
    if oopti.class_type == "Fraunhofer":
        Int = Int.doit()
        Int = simplify(Int.rewrite(sin))
        oopti.result = Int
    display(Int)
    
    commands = ["subs", "oopti.result", subs]
    Int = oopti.process(commands)
    
    lt = torchsympy.TorchSymPy()
    
    print("\n--- Benchmarking ---")
    print(f"Grid Size: {n}x{n} = {n*n} points")
    # TorchSymPy Execution
    X_t = torch.tensor(X, dtype=torch.float64)
    Y_t = torch.tensor(Y, dtype=torch.float64)
    
    start = time.perf_counter()
    if not sets.usetorchsympy:
        pass # We want to benchmark torchsympy so skip standard scipy lambdify loop initially
    else:
        Z_tsp = lt.eval_numeric(Int, params_values=[X_t, Y_t], method=GaussLegendre(),
                solver="vectorized", N=5501) 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tsp_time = time.perf_counter() - start
        print(f"TorchSymPy Time (10,000 points): {tsp_time:.4f} seconds")

    print("\nBenchmarking SciPy (lambdify)...")
    fInt = smp.lambdify([x,y], Int.rhs.xreplace({x:x, y:y}).doit().evalf(quad='osc'), "scipy")
    
    start = time.perf_counter()
    # Evaluate all points using np.vectorize
    Z_scipy = np.vectorize(fInt)(X, Y)
    scipy_time = time.perf_counter() - start
    
    print(f"SciPy Time (All 10,000 points): {scipy_time:.4f} seconds")
    
    speedup = scipy_time / tsp_time
    print(f"\nResulting Speedup: {speedup:.2f}x")
    
    print("\nPlotting results...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if torch.is_tensor(Z_tsp):
        Z_tsp_np = Z_tsp.detach().cpu().numpy()
    else:
        Z_tsp_np = Z_tsp
        
    # We take the real part in case there are residual imaginary components close to 0
    Z_tsp_np = np.real(Z_tsp_np)
    Z_scipy = np.real(Z_scipy)
    
    im1 = axes[0].imshow(Z_tsp_np, cmap=plt.cm.gray, interpolation='bilinear', origin='lower')
    axes[0].set_title(f'TorchSymPy ({tsp_time:.4f}s)')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(Z_scipy, cmap=plt.cm.gray, interpolation='bilinear', origin='lower')
    axes[1].set_title(f'SciPy ({scipy_time:.4f}s)')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

