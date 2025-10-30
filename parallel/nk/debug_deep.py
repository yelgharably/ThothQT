"""
Deep Debug: Check Green's function calculation step-by-step
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from yqt_production import *

# Simple test
t = 1.0
N = 3  # Even shorter for easier visualization
E = 0.5
eta = 0.05

print("=" * 80)
print(f"DEEP DEBUG: 1D Chain N={N}, E={E}, t={t}")
print("=" * 80)
print()

# Build system
H_dev = sp.diags([-t, -t], offsets=[-1, 1], shape=(N, N)).tocsr()
H00 = np.array([[0.0]])
H01 = np.array([[-t]])
lead = PeriodicLead(H00=H00, H01=H01)

print("Device Hamiltonian:")
print(H_dev.toarray())
print()

# Manual NEGF calculation
print("=" * 80)
print("MANUAL NEGF CALCULATION")
print("=" * 80)
print()

# Step 1: Compute self-energies
dec = SanchoRubioDecimator(lead, eta=eta, gpu=False)
g_s = dec.surface_g(E)
Sigma_L = dec.sigma(E)
Sigma_R = Sigma_L.copy()  # Symmetric system

print(f"1. Surface Green's function:")
print(f"   g_s = {g_s[0,0]:.6f}")
print()

print(f"2. Self-energies:")
print(f"   Sigma_L = {Sigma_L[0,0]:.6f}")
print(f"   Sigma_R = {Sigma_R[0,0]:.6f}")
print()

# Step 2: Assemble system matrix A = E*I - H - Sigma_L - Sigma_R
A = (E + 1j * eta) * sp.eye(N) - H_dev

# Add self-energies to corners
A = A.tolil()
A[0, 0] -= Sigma_L[0, 0]
A[N-1, N-1] -= Sigma_R[0, 0]
A = A.tocsr()

print("3. System matrix A = (E + i*eta)*I - H - Sigma_L - Sigma_R:")
print(A.toarray())
print()

# Step 3: Compute full Green's function G = A^-1
G_full = np.linalg.inv(A.toarray())

print("4. Full Green's function G = A^{-1}:")
print(G_full)
print()

# Step 4: Compute broadening matrices
Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T)
Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T)

print(f"5. Broadening matrices:")
print(f"   Gamma_L = {Gamma_L[0,0]:.6f}")
print(f"   Gamma_R = {Gamma_R[0,0]:.6f}")
print()

# Step 5: Compute transmission using Fisher-Lee
# T = Tr[Gamma_L * G * Gamma_R * G^dagger]

# Extract relevant Green's function elements
# For full formula, we need the full matrices
Gamma_L_full = np.zeros((N, N), dtype=complex)
Gamma_R_full = np.zeros((N, N), dtype=complex)
Gamma_L_full[0, 0] = Gamma_L[0, 0]
Gamma_R_full[N-1, N-1] = Gamma_R[0, 0]

print("6. Full Gamma matrices (padded to device size):")
print("   Gamma_L_full:")
print(Gamma_L_full)
print()
print("   Gamma_R_full:")
print(Gamma_R_full)
print()

# Fisher-Lee formula
product = Gamma_L_full @ G_full @ Gamma_R_full @ G_full.conj().T
T_manual = np.real(np.trace(product))

print(f"7. Fisher-Lee calculation:")
print(f"   Product matrix:")
print(product)
print()
print(f"   Trace = {np.trace(product):.6f}")
print(f"   T (real part) = {T_manual:.6f}")
print()

# Compare with YQT
print("=" * 80)
print("YQT CALCULATION")
print("=" * 80)
print()

device = Device(H=H_dev, S=None, left=lead, right=lead, Ef=0.0)
engine = NEGFEngine(device, Temp=1, eta=eta, gpu=False)
T_yqt = engine.transmission(E)

print(f"YQT transmission: {T_yqt:.6f}")
print(f"Manual transmission: {T_manual:.6f}")
print(f"Difference: {abs(T_yqt - T_manual):.6e}")
print()

# Compare with KWANT
print("=" * 80)
print("KWANT REFERENCE")
print("=" * 80)
print()

try:
    import kwant
    lat = kwant.lattice.chain(a=1, norbs=1)
    syst = kwant.Builder()
    syst[(lat(i) for i in range(N))] = 0
    syst[lat.neighbors()] = -t
    
    lead_kwant = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead_kwant[lat(0)] = 0
    lead_kwant[lat.neighbors()] = -t
    
    syst.attach_lead(lead_kwant)
    syst.attach_lead(lead_kwant.reversed())
    
    fsyst = syst.finalized()
    smat = kwant.smatrix(fsyst, E)
    T_kwant = smat.transmission(1, 0)
    
    print(f"KWANT transmission: {T_kwant:.6f}")
    print()
    
    print("Comparison:")
    print(f"  YQT:    {T_yqt:.6f}")
    print(f"  Manual: {T_manual:.6f}")
    print(f"  KWANT:  {T_kwant:.6f}")
    print()
    print(f"  YQT error:    {abs(T_yqt - T_kwant):.6f}")
    print(f"  Manual error: {abs(T_manual - T_kwant):.6f}")
    
except Exception as e:
    print(f"KWANT error: {e}")

print()
print("=" * 80)

# Additional check: verify unitarity
# For a good NEGF calculation: T + R ≈ number of channels
print("UNITARITY CHECK")
print("=" * 80)
print()

# Compute reflection
R_L = np.real(np.trace(Gamma_L_full @ G_full @ Gamma_L_full @ G_full.conj().T))
R_R = np.real(np.trace(Gamma_R_full @ G_full @ Gamma_R_full @ G_full.conj().T))

print(f"Reflection left:  R_L = {R_L:.6f}")
print(f"Reflection right: R_R = {R_R:.6f}")
print(f"Transmission:     T   = {T_manual:.6f}")
print(f"Sum:              T + R_L + R_R = {T_manual + R_L + R_R:.6f}")
print(f"Expected (1 channel): 1.000")
print()
