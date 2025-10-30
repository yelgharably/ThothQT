"""
Debug YQT 1D System - Check self-energy and transmission calculation
"""

import numpy as np
import scipy.sparse as sp
from yqt_production import *

# Simple 1D parameters
t = 1.0
N = 5  # Very short chain for debugging
E_test = 0.5  # In-band energy

print("=" * 70)
print("YQT 1D DEBUG")
print("=" * 70)
print()

# Build device
print(f"Device: 1D chain, N={N}, t={t}, E={E_test}")
H_dev = sp.diags([-t, -t], offsets=[-1, 1], shape=(N, N)).tocsr()
print(f"Device Hamiltonian:\n{H_dev.toarray()}")
print()

# Build leads
H00 = np.array([[0.0]])
H01 = np.array([[-t]])
print(f"Lead H00: {H00}")
print(f"Lead H01: {H01}")
print()

# Create lead decimator
lead = PeriodicLead(H00=H00, H01=H01)
dec = SanchoRubioDecimator(lead, eta=0.05, gpu=False)

# Compute surface Green's function
print(f"Computing surface Green's function at E={E_test}...")
g_s = dec.surface_g(E_test)
print(f"g_s = {g_s}")
print(f"  Re[g_s] = {np.real(g_s[0,0]):.6f}")
print(f"  Im[g_s] = {np.imag(g_s[0,0]):.6f}")
print()

# Analytical solution for comparison
eps0 = H00[0, 0]
z = E_test + 1j * 0.05
delta = (z - eps0)**2 - 4 * t**2
sqrt_delta = np.sqrt(delta)
g_analytical_1 = ((z - eps0) + sqrt_delta) / (2 * t**2)
g_analytical_2 = ((z - eps0) - sqrt_delta) / (2 * t**2)

print(f"Analytical solutions:")
print(f"  g1 = {g_analytical_1:.6f} (Im = {np.imag(g_analytical_1):.6f})")
print(f"  g2 = {g_analytical_2:.6f} (Im = {np.imag(g_analytical_2):.6f})")
print(f"  Using g with Im < 0: ", end="")
if np.imag(g_analytical_1) < np.imag(g_analytical_2):
    g_correct = g_analytical_1
    print("g1")
else:
    g_correct = g_analytical_2
    print("g2")
print(f"  Correct g_s = {g_correct:.6f}")
print()

# Check if YQT matches analytical
match = np.allclose(g_s[0,0], g_correct, rtol=1e-4)
print(f"YQT matches analytical: {match}")
if not match:
    print(f"  Error: {abs(g_s[0,0] - g_correct):.6e}")
print()

# Compute self-energy
print("Computing self-energy...")
Sigma = dec.sigma(E_test)
print(f"Sigma = \n{Sigma}")
print(f"  Re[Sigma] = {np.real(Sigma[0,0]):.6f}")
print(f"  Im[Sigma] = {np.imag(Sigma[0,0]):.6f}")
print()

# Analytical self-energy: Sigma = tau^dagger * g_s * tau = H01^dagger * g_s * H01
Sigma_analytical = np.conj(H01[0,0]) * g_s[0,0] * H01[0,0]
print(f"Analytical Sigma = {Sigma_analytical:.6f}")
print(f"  Match: {np.allclose(Sigma[0,0], Sigma_analytical, rtol=1e-4)}")
print()

# Compute broadening
Gamma = 1j * (Sigma - Sigma.conj().T)
print(f"Gamma (broadening) = \n{Gamma}")
print(f"  Re[Gamma] = {np.real(Gamma[0,0]):.6f}")
print(f"  Im[Gamma] = {np.imag(Gamma[0,0]):.6f}")
print()

# Now test full transmission
print("=" * 70)
print("FULL TRANSMISSION CALCULATION")
print("=" * 70)
print()

device = Device(H=H_dev, S=None, left=lead, right=lead, Ef=0.0)
engine = NEGFEngine(device, Temp=1, eta=0.05, gpu=False)

T = engine.transmission(E_test)
print(f"Transmission T({E_test}) = {T:.6f}")
print()

# For a perfect 1D chain, T should be close to 1.0 in-band
print(f"Expected T ≈ 1.0 for E in (-2t, 2t)")
print(f"E_test = {E_test} is in band: {abs(E_test) < 2*t}")
print()

# Check what KWANT would give
try:
    import kwant
    lat = kwant.lattice.chain(a=1)
    syst = kwant.Builder()
    syst[(lat(i) for i in range(N))] = 0
    syst[lat.neighbors()] = -t
    
    lead_kwant = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead_kwant[lat(0)] = 0
    lead_kwant[lat.neighbors()] = -t
    
    syst.attach_lead(lead_kwant)
    syst.attach_lead(lead_kwant.reversed())
    
    fsyst = syst.finalized()
    
    smat = kwant.smatrix(fsyst, E_test)
    T_kwant = smat.transmission(1, 0)
    
    print(f"KWANT transmission: {T_kwant:.6f}")
    print(f"YQT vs KWANT error: {abs(T - T_kwant):.6f}")
except Exception as e:
    print(f"KWANT not available: {e}")

print()
print("=" * 70)
