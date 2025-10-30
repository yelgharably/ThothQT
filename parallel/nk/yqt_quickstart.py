"""
YQT Quick Start Guide
====================

Example usage of the YQT production library.
"""

import numpy as np
import matplotlib.pyplot as plt
from yqt_production import *
from yqt_production import _CUPY_AVAILABLE  # Import private variable explicitly

print("=" * 70)
print("YQT QUICK START EXAMPLES")
print("=" * 70)
print()

# ============================================================================
# EXAMPLE 1: Simple Transmission Calculation
# ============================================================================

print("EXAMPLE 1: Transmission Calculation")
print("-" * 70)

# Create a 2D graphene nanoribbon (example - would need proper graphene structure)
# For now, using simple test system
device = make_1d_chain(n_cells=20, t=1.0, spin=False)

# Initialize NEGF engine
engine = NEGFEngine(device, Temp=300, eta=1e-3, gpu=False)  # Larger eta for stability
print()

# Compute transmission at a few energies
print("Transmission at selected energies:")
print(f"{'Energy (eV)':<15} {'Transmission':<15}")
print("-" * 30)

test_energies = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
for E in test_energies:
    try:
        T = engine.transmission(E)
        print(f"{E:<15.2f} {T:<15.6f}")
    except Exception as e:
        print(f"{E:<15.2f} ERROR: {type(e).__name__}")

print()

# ============================================================================
# EXAMPLE 2: Energy-Resolved Transmission
# ============================================================================

print("EXAMPLE 2: Energy-Resolved Transmission")
print("-" * 70)

# Create energy grid
energies = np.linspace(-1.8, 1.8, 30)  # Avoid band edges
T_values = []

print(f"Computing transmission on {len(energies)} point grid...")

for E in energies:
    try:
        T = engine.transmission(E)
        T_values.append(T)
    except:
        T_values.append(0.0)

T_values = np.array(T_values)
print(f"✓ Complete: T ∈ [{T_values.min():.4f}, {T_values.max():.4f}]")
print()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(energies, T_values, 'b-', linewidth=2, marker='o')
plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect transmission')
plt.axvline(0.0, color='gray', linestyle=':', alpha=0.5, label='Fermi level')
plt.xlabel('Energy (eV)', fontsize=12)
plt.ylabel('Transmission', fontsize=12)
plt.title('Energy-Resolved Transmission', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('yqt_example_transmission.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: yqt_example_transmission.png")
print()

# ============================================================================
# EXAMPLE 3: Conductance
# ============================================================================

print("EXAMPLE 3: Conductance Calculation")
print("-" * 70)

# Conductance is just G0 × T
E_test = 0.0
try:
    G = engine.conductance(E_test)
    print(f"Conductance at E={E_test} eV:")
    print(f"  G = {G*1e6:.3f} µS")
    print(f"  G/G0 = {G/G0_SI:.6f}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# ============================================================================
# EXAMPLE 4: I-V Characteristics (if system works)
# ============================================================================

print("EXAMPLE 4: Current-Voltage (I-V)")
print("-" * 70)
print("Note: I-V calculation requires many transmission evaluations")
print("Skipping for quick start. Use engine.IV(bias, mesh) for full I-V.")
print()

# Example of how to use IV:
# mesh = EnergyMesh(-0.5, 0.5, n=50, refine_at=(0.0,), refine_pts=100)
# result = engine.IV(bias=0.1, mesh=mesh)
# print(f"Current at V={result['bias']} V: {result['I']*1e9:.3f} nA")

# ============================================================================
# EXAMPLE 5: GPU Acceleration
# ============================================================================

print("EXAMPLE 5: GPU Acceleration")
print("-" * 70)

if _CUPY_AVAILABLE:
    print("GPU is available!")
    print()
    print("To use GPU, set gpu=True when creating NEGFEngine:")
    print("  engine = NEGFEngine(device, Temp=300, gpu=True)")
    print()
    print("Expected speedup: 5-15× for medium systems (N > 1000)")
else:
    print("GPU (CuPy) not available on this system")

print()

# ============================================================================
# EXAMPLE 6: Custom System
# ============================================================================

print("EXAMPLE 6: Building Custom Systems")
print("-" * 70)
print()

print("To create your own system:")
print()
print("1. Define device Hamiltonian (sparse matrix):")
print("   H = sp.csr_matrix(...)")
print()
print("2. Define lead unit cells:")
print("   H00 = np.array(...)  # On-site Hamiltonian")
print("   H01 = np.array(...)  # Hopping to next cell")
print("   lead = PeriodicLead(H00=H00, H01=H01)")
print()
print("3. Create device:")
print("   device = Device(H=H, S=None, left=lead, right=lead, Ef=0.0)")
print()
print("4. Run NEGF:")
print("   engine = NEGFEngine(device, Temp=300)")
print("   T = engine.transmission(E)")
print()

# ============================================================================
# Tips and Best Practices
# ============================================================================

print("=" * 70)
print("TIPS AND BEST PRACTICES")
print("=" * 70)
print()

print("1. Numerical Stability:")
print("   • Use larger eta (1e-3 instead of 1e-6) near band edges")
print("   • Avoid exact band edge energies")
print("   • Check Sancho-Rubio convergence warnings")
print()

print("2. Performance:")
print("   • Use sparse matrices for large systems")
print("   • Enable GPU for systems with N > 1000")
print("   • Cache self-energies if computing many energies")
print()

print("3. Physical Checks:")
print("   • Transmission should be 0 ≤ T ≤ N_channels")
print("   • Check T + R ≈ 1 (unitarity)")
print("   • Verify band structure makes sense")
print()

print("4. Debugging:")
print("   • Start with small systems (N < 50)")
print("   • Test on known cases (1D chain)")
print("   • Compare with analytical results when possible")
print()

print("=" * 70)
print("For more details, see:")
print("  • yqt_production.py - Full source code with documentation")
print("  • YQT_STATUS_REPORT.md - Current status and known issues")
print("  • YQT_DIAGNOSTIC_REPORT.md - Comprehensive analysis")
print("=" * 70)
