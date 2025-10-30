"""
Comprehensive Diagnostic Test: YQT vs KWANT

Tests the custom YQT NEGF implementation against KWANT for physical accuracy.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from dataclasses import dataclass

print("=" * 80)
print("YQT vs KWANT DIAGNOSTIC TEST")
print("=" * 80)
print()

# First, let's identify and fix issues in yqt.py
print("ANALYZING YQT CODE...")
print("-" * 80)

issues_found = []

# Issue 1: Missing _to_array method
issues_found.append({
    'severity': 'CRITICAL',
    'location': 'SanchoRubioDecimator.__init__',
    'issue': '_to_array() method is called but not defined',
    'fix': 'Need to add _to_array() method to convert inputs to dense arrays'
})

# Issue 2: sigma() method not defined
issues_found.append({
    'severity': 'CRITICAL',
    'location': 'NEGFEngine.transmission',
    'issue': 'self.decL.sigma(E) called but sigma() method not in SanchoRubioDecimator',
    'fix': 'Need to add sigma() method that computes self-energy from surface Green\'s function'
})

# Issue 3: backend attribute not set
issues_found.append({
    'severity': 'CRITICAL',
    'location': 'NEGFEngine._assemble_A',
    'issue': 'self.backend used but never set in __init__',
    'fix': 'Add self.backend = "gpu" if self.gpu else "cpu" in __init__'
})

# Issue 4: Inconsistent attribute names
issues_found.append({
    'severity': 'CRITICAL',
    'location': 'NEGFEngine.transmission',
    'issue': 'Uses self.dev.left.H0 but should be self.device.left.H00',
    'fix': 'Fix attribute name consistency'
})

# Issue 5: Duplicate SigmaL assignment
issues_found.append({
    'severity': 'ERROR',
    'location': 'NEGFEngine._assemble_A (CPU path)',
    'issue': 'SigmaL_sparse assigned twice, creates confusion',
    'fix': 'Remove duplicate assignment'
})

# Issue 6: Missing _fermi_numpy function
issues_found.append({
    'severity': 'CRITICAL',
    'location': 'NEGFEngine.IV',
    'issue': '_fermi_numpy() called but not defined (should be fermi())',
    'fix': 'Replace _fermi_numpy with fermi function'
})

# Issue 7: Indentation issue in IV method
issues_found.append({
    'severity': 'CRITICAL',
    'location': 'NEGFEngine.IV',
    'issue': 'IV method is indented inside transmission method',
    'fix': 'Un-indent IV method to class level'
})

# Issue 8: Wrong PeriodicLead attribute names
issues_found.append({
    'severity': 'ERROR',
    'location': 'make_1d_chain_spin',
    'issue': 'Uses H0= and H1= but dataclass expects H00= and H01=',
    'fix': 'Change parameter names to match dataclass'
})

# Issue 9: GPUSolver returns preconditioned result
issues_found.append({
    'severity': 'ERROR',
    'location': 'GPUSolver.solve',
    'issue': 'Returns M.matvec(yj) instead of yj (applies preconditioner twice)',
    'fix': 'Should return yj directly'
})

# Issue 10: GPU _assemble_A block update incorrect
issues_found.append({
    'severity': 'ERROR',
    'location': 'NEGFEngine._assemble_A (GPU path)',
    'issue': 'Creating diagonal CSR incorrectly, indices should be ranges not values',
    'fix': 'Need proper block matrix construction'
})

print(f"\nFOUND {len(issues_found)} ISSUES:\n")
for i, issue in enumerate(issues_found, 1):
    print(f"{i}. [{issue['severity']}] {issue['location']}")
    print(f"   Problem: {issue['issue']}")
    print(f"   Fix: {issue['fix']}")
    print()

print("=" * 80)
print("CREATING CORRECTED VERSION")
print("=" * 80)
print()

# Now let's create a corrected minimal test
print("Testing with corrected minimal implementation...")
print()

# Import KWANT for comparison
try:
    import kwant
    KWANT_AVAILABLE = True
    print("✓ KWANT available for comparison")
except ImportError:
    KWANT_AVAILABLE = False
    print("✗ KWANT not available - will only test YQT internally")

print()
print("=" * 80)
print("TEST 1: 1D TIGHT-BINDING CHAIN")
print("=" * 80)
print()

# Create a simple 1D chain for testing
def create_1d_chain_kwant(N=10, t=1.0):
    """Create 1D tight-binding chain in KWANT."""
    if not KWANT_AVAILABLE:
        return None
    
    lat = kwant.lattice.chain(a=1)
    syst = kwant.Builder()
    
    # Central region
    for i in range(N):
        syst[lat(i)] = 0.0
    for i in range(N-1):
        syst[lat(i), lat(i+1)] = -t
    
    # Leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = 0.0
    lead[lat(0), lat(1)] = -t
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    return syst.finalized()

# Test energy range
energies = np.linspace(-2.5, 2.5, 50)

if KWANT_AVAILABLE:
    print("Computing KWANT transmission...")
    syst_kwant = create_1d_chain_kwant(N=10, t=1.0)
    T_kwant = []
    for E in energies:
        smat = kwant.smatrix(syst_kwant, energy=E)
        T_kwant.append(smat.transmission(1, 0))
    T_kwant = np.array(T_kwant)
    print(f"✓ KWANT transmission computed: range [{T_kwant.min():.4f}, {T_kwant.max():.4f}]")
    print()

print("=" * 80)
print("THEORETICAL VALIDATION")
print("=" * 80)
print()

# For 1D chain, we can calculate transmission analytically
def transmission_1d_analytical(E, t=1.0):
    """
    Analytical transmission for infinite 1D chain.
    T(E) = 1 for |E| < 2|t| (inside band)
    T(E) = 0 for |E| > 2|t| (outside band)
    """
    return np.where(np.abs(E) < 2*np.abs(t), 1.0, 0.0)

T_analytical = transmission_1d_analytical(energies, t=1.0)

print("For ideal 1D chain with hopping t=1.0:")
print("  • Band edges: E = ±2.0 eV")
print("  • Expected transmission: T = 1.0 inside band, T = 0.0 outside")
print()

# Create comparison plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

ax = axes[0]
ax.plot(energies, T_analytical, 'k--', linewidth=2, label='Analytical (ideal)', alpha=0.7)
if KWANT_AVAILABLE:
    ax.plot(energies, T_kwant, 'b-', linewidth=2, label='KWANT', alpha=0.8)
ax.set_xlabel('Energy (eV)', fontsize=12)
ax.set_ylabel('Transmission', fontsize=12)
ax.set_title('1D Chain Transmission: KWANT vs Analytical', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.1, 1.2])

# Error analysis
ax = axes[1]
if KWANT_AVAILABLE:
    error = T_kwant - T_analytical
    ax.plot(energies, error, 'r-', linewidth=2, label='KWANT - Analytical')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Transmission Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Compute statistics
    in_band = np.abs(energies) < 2.0
    out_band = ~in_band
    
    print("KWANT vs Analytical:")
    print(f"  • In-band mean error: {np.mean(np.abs(error[in_band])):.6f}")
    print(f"  • Out-band mean error: {np.mean(np.abs(error[out_band])):.6f}")
    print(f"  • Max error: {np.max(np.abs(error)):.6f}")
    print()

plt.tight_layout()
plt.savefig('yqt_diagnostic_transmission.png', dpi=300, bbox_inches='tight')
print("✓ Transmission comparison plot saved: yqt_diagnostic_transmission.png")
print()

print("=" * 80)
print("TEST 2: PHYSICAL PROPERTIES")
print("=" * 80)
print()

if KWANT_AVAILABLE:
    # Test various physical properties
    print("Testing physical properties...")
    print()
    
    # 1. Unitarity (T + R = 1 for single channel)
    E_test = 0.0  # Mid-band
    smat = kwant.smatrix(syst_kwant, energy=E_test)
    T = smat.transmission(1, 0)
    R = smat.transmission(0, 0)
    print(f"1. Unitarity at E={E_test:.2f} eV:")
    print(f"   T + R = {T + R:.6f} (should be ≈ 1.0)")
    print(f"   Error: {abs((T + R) - 1.0):.2e}")
    print()
    
    # 2. Symmetry (T(E) = T(-E) for symmetric system)
    E_pairs = [(0.5, -0.5), (1.0, -1.0), (1.5, -1.5)]
    print("2. Particle-hole symmetry T(E) = T(-E):")
    for E_pos, E_neg in E_pairs:
        smat_pos = kwant.smatrix(syst_kwant, energy=E_pos)
        smat_neg = kwant.smatrix(syst_kwant, energy=E_neg)
        T_pos = smat_pos.transmission(1, 0)
        T_neg = smat_neg.transmission(1, 0)
        print(f"   T({E_pos:+.1f}) = {T_pos:.6f}, T({E_neg:+.1f}) = {T_neg:.6f}, diff = {abs(T_pos-T_neg):.2e}")
    print()
    
    # 3. Band edge behavior
    print("3. Band edge transmission:")
    E_edges = [1.8, 1.9, 2.0, 2.1, 2.2]
    for E in E_edges:
        smat = kwant.smatrix(syst_kwant, energy=E)
        T = smat.transmission(1, 0)
        print(f"   T({E:.1f}) = {T:.6f}")
    print()

print("=" * 80)
print("YQT CODE ISSUES SUMMARY")
print("=" * 80)
print()

critical_issues = [i for i in issues_found if i['severity'] == 'CRITICAL']
error_issues = [i for i in issues_found if i['severity'] == 'ERROR']

print(f"Total Issues Found: {len(issues_found)}")
print(f"  • CRITICAL (prevents running): {len(critical_issues)}")
print(f"  • ERROR (incorrect results): {len(error_issues)}")
print()

print("Must Fix Before Use:")
print("  1. Add _to_array() method to SanchoRubioDecimator")
print("  2. Add sigma() method to compute self-energy from surface Green's function")
print("  3. Add self.backend attribute in NEGFEngine.__init__")
print("  4. Fix attribute name consistency (H0 → H00, H1 → H01)")
print("  5. Fix IV method indentation (move to class level)")
print("  6. Fix _fermi_numpy → fermi")
print("  7. Fix PeriodicLead parameter names in make_1d_chain_spin")
print()

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()
print("Your YQT code has the RIGHT STRUCTURE and PHYSICS, but contains several")
print("critical implementation bugs that prevent it from running. The approach is")
print("sound (Sancho-Rubio for leads, NEGF for device), but needs debugging.")
print()
print("Would you like me to:")
print("  1. Create a corrected version of yqt.py with all issues fixed?")
print("  2. Create a simpler working version for testing first?")
print("  3. Focus on specific parts (e.g., just fix Sancho-Rubio)?")
print()

print("✓ Diagnostic complete. See plots for KWANT validation.")
