"""
Test Sancho-Rubio convergence for graphene lead.
"""
import numpy as np
from toothqt_builders import GrapheneBuilder

# Build ribbon
gb = GrapheneBuilder()
device, _ = gb.zigzag_ribbon(width=2, length=3)

H00 = device.left.H00
H01 = device.left.H01

print("=" * 80)
print("SANCHO-RUBIO CONVERGENCE TEST")
print("=" * 80)
print(f"\nLead size: {H00.shape[0]} atoms/cell")
print(f"\nH00:\n{H00}")
print(f"\nH01:\n{H01}")

# Check if H00 + H01 + H01† forms a valid Hamiltonian
H_full = H00 + H01 + H01.conj().T
print(f"\nH00 + H01 + H01†:\n{H_full}")
print(f"Is Hermitian? {np.allclose(H_full, H_full.conj().T)}")

# Eigenvalues (should be real and show the band structure)
eigs = np.linalg.eigvalsh(H_full)
print(f"\nEigenvalues of H00 + H01 + H01†: {eigs}")
print(f"Band range: [{eigs[0]:.3f}, {eigs[-1]:.3f}] eV")

# Test surface Green's function at different energies
E_test = [0.1, 0.5, 1.0, 2.0, 3.0]
eta = 1e-6

print("\n" + "=" * 80)
print("SURFACE GREEN'S FUNCTION TEST")
print("=" * 80)

for E in E_test:
    print(f"\nE = {E} eV:")
    
    # Direct calculation
    zI = (E + 1j * eta) * np.eye(H00.shape[0])
    
    # Try simple inversion (this should fail or give huge values)
    try:
        g_simple = np.linalg.inv(zI - H00)
        g_diag = np.diag(g_simple)
        print(f"  Simple g = (zI - H00)^-1:")
        print(f"    Diagonal: {g_diag}")
        print(f"    Max |g|: {np.max(np.abs(g_simple)):.3e}")
    except Exception as e:
        print(f"  Simple inversion failed: {e}")
    
    # Sancho-Rubio (using ThoothQT)
    from toothqt_production import NEGFEngine
    negf = NEGFEngine(device, Temp=300.0)
    
    # Access the Sancho-Rubio decimator for the left lead
    g_sr = negf.decL.surface_g(E)
    g_sr_diag = np.diag(g_sr)
    print(f"  Sancho-Rubio g_s:")
    print(f"    Diagonal: {g_sr_diag}")
    print(f"    Max |g_s|: {np.max(np.abs(g_sr)):.3e}")
    print(f"    Im[g_s] < 0? {np.all(np.imag(g_sr_diag) < 0)}")
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print("""
For a semi-infinite lead, the surface Green's function should:
1. Have Im[g_s] < 0 (causality)
2. Be O(1/t) in magnitude (for typical tight-binding)
3. Converge rapidly in Sancho-Rubio (< 50 iterations)

If g_s is O(10^10), this suggests:
- Sancho-Rubio is not converging properly
- The H01 structure might create numerical instability
- Need to check if the lead is truly periodic
""")

# Check periodicity
print("\nPERIODICITY CHECK:")
print("For a properly periodic lead, H01 should represent all bonds")
print("that connect one unit cell to the next.")
print(f"\nH01 has {np.count_nonzero(H01)} non-zero elements")
print(f"H00 has {np.count_nonzero(H00)} non-zero elements")

# Check if H01 is singular
det_H01 = np.linalg.det(H01)
print(f"\ndet(H01) = {det_H01:.3e}")
print("If det(H01) ≈ 0, the lead may not be properly coupled.")

print("=" * 80)
